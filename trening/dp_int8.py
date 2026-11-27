#!/usr/bin/env python3
"""Ile predyktor długości traci w INT8 i które warstwy za to odpowiadają.

Symulowana kwantyzacja (fake-quant, na GPU) każdej warstwy Linear / Conv1d i MultiheadAttention predyktora:
  W8      wagi INT8 symetrycznie, skala na kanał wyjściowy (jak torch.ao per-channel); aktywacje fp32
  W8A8    jak W8 plus wejście warstwy INT8 symetrycznie, skala dynamiczna na cały tensor (jak kwantyzacja dynamiczna)
Najpierw cały model w obu wariantach, potem po jednej warstwie W8A8 (reszta fp32), posortowane od najgorszej.
Ocena jak w usrednij_dp.py: średni błąd względny długości, ten sam wycinek wzorca dla każdego wariantu.

Uruchamiać z katalogu trening/:
  python dp_int8.py --punkt ../modele/dp_v3/duration_predictor_wsm.pt --val ../dane/bluetts_val.csv \\
    --config config/polski_100m.json --wyjscie ../wyniki/dp_int8.txt
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.data.text2latent_dataset import Text2LatentDataset
from training.dp.data_module import collate_dp
from training.dp.trainer import _build_dp_models, _load_dp_config
from usrednij_dp import ocen, policz_latenty


def kwant(x: torch.Tensor, os_kanalu: int | None) -> torch.Tensor:
    """INT8 symetrycznie: skala na kanał (os_kanalu) albo na cały tensor (None)."""
    if os_kanalu is None:
        s = x.abs().amax().clamp(min=1e-8) / 127
    else:
        wymiary = [d for d in range(x.dim()) if d != os_kanalu]
        s = x.abs().amax(dim=wymiary, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def warstwy(model: nn.Module) -> list[str]:
    """Warstwy Linear / Conv1d i całe bloki MultiheadAttention (ich out_proj nie liczymy osobno)."""
    mha = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]
    zwykle = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv1d))
              and not any(n.startswith(x + ".") for x in mha)]
    return zwykle + mha


def wagi_warstwy(m: nn.Module) -> list[nn.Parameter]:
    if isinstance(m, nn.MultiheadAttention):
        return [p for p in (m.in_proj_weight, m.q_proj_weight, m.k_proj_weight, m.v_proj_weight, m.out_proj.weight)
                if p is not None]
    return [m.weight]


def podmien(model: nn.Module, nazwy: list[str], aktywacje: bool) -> list:
    """Wagi INT8 na kanał wyjściowy (w miejscu) i opcjonalnie wejścia INT8 na tensor (hook); zwraca stan do przywrócenia."""
    stare = []
    for nazwa in nazwy:
        m = model.get_submodule(nazwa)
        for p in wagi_warstwy(m):
            stare.append(("waga", p, p.detach().clone()))
            with torch.no_grad():
                p.copy_(kwant(p, 0))
        if aktywacje:
            def hak(_m, args, kwargs):
                args = tuple(kwant(x, None) if torch.is_tensor(x) and x.is_floating_point() else x for x in args)
                kwargs = {k: kwant(v, None) if k in ("query", "key", "value") and torch.is_tensor(v) else v
                          for k, v in kwargs.items()}
                return args, kwargs
            stare.append(("hak", m.register_forward_pre_hook(hak, with_kwargs=True), None))
    return stare


def przywroc(stare: list) -> None:
    for rodzaj, obiekt, kopia in stare:
        if rodzaj == "waga":
            with torch.no_grad():
                obiekt.copy_(kopia)
        else:
            obiekt.remove()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--punkt", required=True)
    p.add_argument("--val", required=True, help="CSV trenera (bluetts_val.csv)")
    p.add_argument("--config", default="config/polski_100m.json")
    p.add_argument("--ae_checkpoint", default="../modele/kodek_48k/ae_wsm_210_220.pt")
    p.add_argument("--stats_path", default="../modele/stats.pt")
    p.add_argument("--wyjscie", help="plik z wynikami (tekst)")
    p.add_argument("--powtorzenia", type=int, default=3, help="różne losowania wycinka wzorca (ziarna 0..N-1)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False   # różnice INT8 są małe: bez TF32, żeby ich nie zaszumiał
    torch.backends.cudnn.allow_tf32 = False
    dp_cfg = _load_dp_config(a.config)
    full = dp_cfg["_full"]
    stl = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    mel, enc, model = _build_dp_models(full, int(stl.get("n_style", 8)), int(stl.get("style_value_dim", 16)),
                                       dp_cfg.get("sentence_encoder", {}), dp_cfg.get("style_encoder", {}),
                                       dp_cfg.get("predictor", {}), a.ae_checkpoint, device,
                                       edge_fill=dp_cfg.get("edge_fill", False))
    enc.eval()
    model.load_state_dict(torch.load(a.punkt, map_location="cpu"))
    model.eval()
    ccf = int(dp_cfg.get("chunk_compress_factor", 6))
    scale = float(dp_cfg.get("normalizer", {}).get("scale", 1.0))
    spec = full["ae"]["encoder"].get("spec_processor", {})
    sr, hop = int(full["ae"].get("sample_rate", spec.get("sample_rate", 44100))), int(spec.get("hop_length", 512))
    stats = torch.load(a.stats_path, map_location=device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    if mean.dim() != 3:
        mean, std = mean.view(1, -1, 1), std.view(1, -1, 1)
    limity = full.get("data", {})
    ds = Text2LatentDataset(a.val, sample_rate=sr, max_wav_len=int(sr * float(limity.get("max_wav_sec", 20))),
                            max_text_len=int(limity.get("max_text_len", 800)), load_ref=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=partial(collate_dp, spk2idx=None, unknown_spk=0))
    with torch.no_grad():
        lat = policz_latenty(loader, mel, enc, mean, std, scale, ccf, hop, device)

    def blad() -> float:
        return sum(ocen(model, lat, ccf, hop, sr, device, ziarno=z)[1] for z in range(a.powtorzenia)) / a.powtorzenia

    nazwy = warstwy(model)
    parametry = sum(p.numel() for p in model.parameters())
    linie = [f"Predyktor {a.punkt}: {parametry / 1e6:.2f}M parametrów, {len(nazwy)} warstw Linear/Conv1d; "
             f"walidacja {len(ds)} klipów ({a.val}), {a.powtorzenia} losowania wzorca"]
    fp32 = blad()
    linie.append(f"fp32: błąd długości {fp32:.3f}%")
    for opis, akt in (("W8 (same wagi)", False), ("W8A8 (wagi + wejścia)", True)):
        stare = podmien(model, nazwy, akt)
        b = blad()
        przywroc(stare)
        linie.append(f"{opis}, cały model: {b:.3f}% ({b - fp32:+.3f} pkt proc.)")
    linie.append("Po jednej warstwie W8A8 (reszta fp32), od najgorszej:")
    wyniki = []
    for nazwa in nazwy:
        stare = podmien(model, [nazwa], True)
        wyniki.append((blad() - fp32, nazwa, sum(p.numel() for p in wagi_warstwy(model.get_submodule(nazwa)))))
        przywroc(stare)
    for d, nazwa, n in sorted(wyniki, reverse=True):
        linie.append(f"  {d:+.3f} pkt proc.  {nazwa}  ({n} wag)")
    print("\n".join(linie))
    if a.wyjscie:
        os.makedirs(os.path.dirname(os.path.abspath(a.wyjscie)), exist_ok=True)
        open(a.wyjscie, "w", encoding="utf-8").write("\n".join(linie) + "\n")


if __name__ == "__main__":
    main()
