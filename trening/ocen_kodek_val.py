#!/usr/bin/env python3
"""Ocena kodeka (enkoder + dekoder) na całych klipach walidacyjnych, osobno dla każdego pliku CSV.

Każdy klip: dopełnienie zerami do pełnych porcji 512 × 6 próbek (jak przy liczeniu latentów i w inferencji),
enkoder, dekoder, obcięcie do długości oryginału. Miary (im mniej, tym bliżej oryginału):
  recon        miara z treningu kodeka (L1 liniowych widm melowych do 12 kHz, 3 rozdzielczości)
               — ta sama co Loss/Heldout_recon w TensorBoard
  log-mel      L1 logarytmicznych widm melowych, całe pasmo do 24 kHz (Heldout_logmel_full)
  LSD          odległość widm w dB, osobno dla 0–8, 8–16 i 16–24 kHz
Średnie są po klipach. Uwaga: w treningu kodeka miary liczone były na 64 klipach obciętych do 15 s,
tutaj na wszystkich klipach w całości, więc liczby mogą się trochę różnić od TensorBoard.

Uruchamiać z katalogu trening/:
  python ocen_kodek_val.py --device cuda:1
  python ocen_kodek_val.py --val ../dane/bluetts_val_liczby.csv --zapisz ../modele/kodek_48k/ocena_val --ile_zapisz 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

from bluecodec.audio_utils import ensure_sr
from bluecodec.utils import LinearMelSpectrogram
from train_autoencoder import get_mel_transforms, recon_l1
from usrednij_kodek import wczytaj_wagi, zbuduj

PASMA = ((0, 8000), (8000, 16000), (16000, 24000))


def sciezki_z_csv(plik: str) -> list[str]:
    with open(plik, encoding="utf-8", newline="") as f:
        return [w["filename"].strip() for w in csv.DictReader(f)]


@torch.no_grad()
def lsd_pasma(x: torch.Tensor, y: torch.Tensor, sr: int) -> list[float]:
    """LSD (dB) w pasmach PASMA; x, y: [T] na tym samym urządzeniu."""
    okno = torch.hann_window(2048, device=x.device)
    X = torch.stft(x, 2048, 512, window=okno, return_complex=True).abs()
    Y = torch.stft(y, 2048, 512, window=okno, return_complex=True).abs()
    f = torch.fft.rfftfreq(2048, 1 / sr).to(x.device)
    wynik = []
    for od, do in PASMA:
        m = (f >= od) & (f < do)
        r = 20 * torch.log10(X[m] + 1e-5) - 20 * torch.log10(Y[m] + 1e-5)
        wynik.append(float(r.pow(2).mean(0).sqrt().mean()))
    return wynik


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="../modele/kodek_48k/ae_wsm_210_220.pt")
    p.add_argument("--val", nargs="+", default=["../dane/bluetts_val.csv", "../dane/bluetts_val_liczby.csv"])
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--max_sec", type=float, default=30.0, help="dłuższe klipy są pomijane (jak w treningu TTS)")
    p.add_argument("--zapisz", help="katalog na pary oryginał/rekonstrukcja do odsłuchu")
    p.add_argument("--ile_zapisz", type=int, default=5, help="ile par z każdego pliku CSV zapisać")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = json.load(open(args.config, encoding="utf-8"))
    ae_cfg, data_cfg = cfg["ae"], cfg["ae"]["data"]
    sr = int(data_cfg["sample_rate"])
    spec = ae_cfg["encoder"].get("spec_processor", {})
    porcja = int(spec.get("hop_length", 512)) * int(cfg["ttl"]["chunk_compress_factor"])
    mel = LinearMelSpectrogram(sample_rate=spec.get("sample_rate", sr), n_fft=spec.get("n_fft", 2048),
                               hop_length=spec.get("hop_length", 512), win_length=spec.get("win_length", 2048),
                               n_mels=spec.get("n_mels", 228)).to(device)
    miary = {"recon": get_mel_transforms(data_cfg, device), "logmel": get_mel_transforms(data_cfg, device, "logmel_fullband")}
    enc, dec = zbuduj(ae_cfg, device)
    e, d, krok = wczytaj_wagi(args.ckpt)
    enc.load_state_dict(e); dec.load_state_dict(d)
    enc.eval(); dec.eval()
    print(f"Kodek: {args.ckpt} (krok {krok}), {sr} Hz")
    if args.zapisz:
        os.makedirs(args.zapisz, exist_ok=True)

    naglowek = f"{'zbiór':<22} {'klipy':>6} {'h':>5} {'recon':>8} {'log-mel':>8} " + " ".join(
        f"{f'LSD {od // 1000}-{do // 1000}k':>10}" for od, do in PASMA)
    wiersze = []
    for plik in args.val:
        t0 = time.time()
        nazwa = os.path.splitext(os.path.basename(plik))[0].replace("bluetts_", "")
        sumy, n, sekundy, pominiete = np.zeros(2 + len(PASMA)), 0, 0.0, 0
        for i, sciezka in enumerate(sciezki_z_csv(plik)):
            x, sr_x = sf.read(sciezka, dtype="float32", always_2d=True)
            x = torch.from_numpy(x).mean(dim=1)
            if sr_x != sr:
                x = ensure_sr(x, sr_x, sr, device="cpu").reshape(-1)
            if x.shape[0] > args.max_sec * sr:
                pominiete += 1
                continue
            x = x.to(device)
            dl = x.shape[0]
            wej = torch.nn.functional.pad(x, (0, (-dl) % porcja)).unsqueeze(0)
            with torch.no_grad():
                y = dec(enc(mel(wej))).reshape(-1)[:dl]
                x1, y1 = x.view(1, 1, -1), y.view(1, 1, -1)
                wartosci = [recon_l1(x1, y1, miary["recon"]).item(), recon_l1(x1, y1, miary["logmel"]).item()]
            wartosci += lsd_pasma(x, y, sr)
            sumy += wartosci; n += 1; sekundy += dl / sr
            if args.zapisz and i < args.ile_zapisz:
                baza = os.path.join(args.zapisz, f"{nazwa}_{i:02d}")
                sf.write(baza + "_oryginal.wav", x.cpu().numpy(), sr)
                sf.write(baza + "_kodek.wav", y.clamp(-1, 1).cpu().numpy(), sr)
        sr_w = sumy / max(n, 1)
        wiersze.append(f"{nazwa:<22} {n:>6} {sekundy / 3600:>5.2f} {sr_w[0]:>8.4f} {sr_w[1]:>8.4f} "
                       + " ".join(f"{v:>10.2f}" for v in sr_w[2:]))
        print(f"  {nazwa}: {n} klipów ({pominiete} pominiętych > {args.max_sec:g} s) w {time.time() - t0:.0f} s", flush=True)
    print("\n" + naglowek)
    for w in wiersze:
        print(w)
    if args.zapisz:
        print(f"\nPróbki do odsłuchu: {args.zapisz}/*_oryginal.wav i *_kodek.wav")


if __name__ == "__main__":
    main()
