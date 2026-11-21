"""Logowanie do Weights & Biases i szacowanie MFU dla treningu modelu tekst->latenty (tylko karta 0).

MFU (model FLOPs utilization) = FLOPs wykonane w kroku / (czas kroku × szczytowa wydajność karty).
FLOPs kroku liczone są wzorem 6·N·pozycje dla każdej części modelu (pole wektorowe na B·Ke sekwencjach
po T klatkach, enkoder tekstu na B tekstach po L znakach, enkoder wzorca na B wycinkach), z rozmiarami
po dopełnieniu, bo tyle karta faktycznie liczy. Wzór pomija część operacji (uwagę, sploty wgłębne,
warunkowanie), więc raz, przy pierwszym mierzonym kroku, jest kalibrowany licznikiem FLOPs PyTorcha
(torch.utils.flop_counter) na prawdziwej partii; dalej wzór × współczynnik kalibracji.

Szczytowa wydajność domyślnie 153 TFLOPS na kartę (--peak_tflops w CLI). Dla porównania: NVIDIA podaje dla
RTX 3090 Ti 80 TFLOPS w bf16 z akumulacją w fp32 (tak liczy PyTorch) i 160 z akumulacją w fp16.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict

import torch


def params(m) -> int:
    return sum(p.numel() for p in m.parameters())


class Metryki:
    def __init__(self, rank, log_every, peak_tflops, n_vf, n_te, n_re, wandb_project=None, wandb_name=None,
                 checkpoint_dir=".", config=None, world_size=1, device=None):
        self.main = rank == 0
        self.log_every, self.peak = max(1, log_every), peak_tflops * 1e12
        self.n_vf, self.n_te, self.n_re = n_vf, n_te, n_re
        self.world, self.device = world_size, device
        self.kalibracja = None
        self.sumy = defaultdict(float)
        self.n = 0
        self.t_okna = time.perf_counter()
        self.run = None
        if self.main and wandb_project:
            try:
                import wandb
            except ImportError as e:
                raise SystemExit("--wandb wymaga pakietu wandb (uv sync) i `wandb login`.") from e
            plik_id = os.path.join(checkpoint_dir, "wandb_id.txt")
            run_id = open(plik_id).read().strip() if os.path.exists(plik_id) else uuid.uuid4().hex[:8]
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(plik_id, "w") as f:
                f.write(run_id)
            # ten sam katalog wyjściowy = ta sama sesja wandb, więc wznowienie dopisuje do tego samego wykresu
            self.run = wandb.init(project=wandb_project, name=wandb_name, id=run_id, resume="allow",
                                  dir=checkpoint_dir, config=config or {})
            self.wandb = wandb
            print(f"[wandb] projekt {wandb_project}, sesja {run_id}: {self.run.url}")

    def flops_wzor(self, B, Ke, T, L, T_ref, spfm=False):
        f = 6 * self.n_vf * B * Ke * T + 6 * self.n_te * B * L + 6 * self.n_re * B * T_ref
        if spfm:   # dwa przejścia pola wektorowego w przód bez gradientów (z tekstem i bez), na B sekwencjach
            f += 2 * 2 * self.n_vf * B * T
        return f

    def kalibruj(self, flops_licznik, flops_wzor):
        if self.kalibracja is None and flops_wzor > 0:
            self.kalibracja = flops_licznik / flops_wzor
            if self.main:
                print(f"[mfu] kalibracja: licznik PyTorcha {flops_licznik / 1e12:.2f} TFLOP, wzór {flops_wzor / 1e12:.2f} "
                      f"TFLOP na krok -> współczynnik {self.kalibracja:.3f}")

    def krok(self, global_step, loss, lr, B, Ke, T, L, T_ref, klatki_prawdziwe, sekundy_audio, spfm=False, extra=None):
        """Wołane po każdym kroku optymalizatora; co log_every kroków liczy średnie i wysyła je do wandb."""
        if not self.main:
            return
        s = self.sumy
        s["loss"] += loss; s["klipy"] += B; s["sekundy_audio"] += sekundy_audio
        s["flops"] += self.flops_wzor(B, Ke, T, L, T_ref, spfm) * (self.kalibracja or 1.0)
        s["dopelnienie"] += 1.0 - klatki_prawdziwe / max(B * T, 1)
        for k, v in (extra or {}).items():
            s[k] += v
        self.n += 1
        if self.n < self.log_every:
            return
        dt = time.perf_counter() - self.t_okna
        n = self.n
        wynik = {
            "train/loss": s["loss"] / n,
            "train/lr": lr,
            "speed/kroki_na_s": n / dt,
            "speed/klipy_na_s_wszystkie_karty": s["klipy"] * self.world / dt,
            "speed/sekundy_audio_na_s_wszystkie_karty": s["sekundy_audio"] * self.world / dt,
            "speed/klipy_na_partie": s["klipy"] / n,
            "speed/dopelnienie": s["dopelnienie"] / n,
            "speed/mfu": s["flops"] / dt / self.peak,
            "speed/tflops_na_karte": s["flops"] / dt / 1e12,
        }
        for k in s:
            if k not in ("loss", "klipy", "sekundy_audio", "flops", "dopelnienie"):
                wynik[k] = s[k] / n
        if self.device is not None and self.device.type == "cuda":
            wynik["gpu/pamiec_maks_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.run is not None:
            self.run.log(wynik, step=global_step)
        self.ostatnie = wynik
        self.sumy, self.n, self.t_okna = defaultdict(float), 0, time.perf_counter()

    def loguj(self, dane: dict, global_step: int):
        if self.run is not None:
            self.run.log(dane, step=global_step)

    def audio(self, log_dir: str, global_step: int, sample_rate: int):
        """Próbki odsłuchowe z danego kroku (step_<krok>_*.wav w log_dir) jako wandb.Audio."""
        if self.run is None:
            return
        import glob
        pliki = sorted(glob.glob(os.path.join(log_dir, f"step_{global_step}_*.wav")))
        if pliki:
            self.run.log({f"probki/{os.path.basename(p)[len(f'step_{global_step}_'):-4]}":
                          self.wandb.Audio(p, sample_rate=sample_rate) for p in pliki}, step=global_step)

    def koniec(self):
        if self.run is not None:
            self.run.finish()
