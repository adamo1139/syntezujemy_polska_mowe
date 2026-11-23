#!/usr/bin/env python3
"""[syntezujemy_polska_mowe] Ocena postfiltru COSE na parach walidacyjnych: sam kodek (noisy) vs kodek + COSE.

Dla każdej pary <base_dir>/valid/{clean,noisy}/<id>.wav liczy względem oryginału VoxCPM2 (clean):
PESQ (wb, 16 kHz), ESTOI, SI-SDR (48 kHz), LSD 16–24 kHz (dB) i UTMOS (także dla samego oryginału).
COSE jednym krokiem (t=1 -> r=0.003, jak samplefortest_COSE), stałe ziarno szumu, wagi EMA.
Długie klipy idą odcinkami z przenikaniem, jedna normalizacja na cały klip (jak w treningu: max |noisy|).

Uruchamiać z katalogu cose/ (venv COSE):
  CUDA_VISIBLE_DEVICES=4 python ocen_cose.py --ckpt ../modele/cose_nf128/checkpoints/last.ckpt \
      --wyjscie ../syntezy/porownanie_cose/epoka1
Bez --ckpt liczy tylko punkt odniesienia (oryginał i sam kodek).
"""

import argparse
import csv
import json
import os
import sys
from glob import glob
from os.path import basename, join

import librosa
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, join(os.path.dirname(os.path.abspath(__file__)), "..", "trening"))
from training.utmos import utmos, wczytaj_utmos  # noqa: E402

SR = 48000


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    cel = a * ref
    return float(10 * np.log10(np.sum(cel ** 2) / (np.sum((est - cel) ** 2) + 1e-12) + 1e-12))


def lsd_gora(est: np.ndarray, ref: np.ndarray, od_hz: float = 16000) -> float:
    def widmo(x):
        return np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) ** 2
    pe, pr = widmo(est), widmo(ref)
    k = int(od_hz / (SR / 2) * (pe.shape[0] - 1))
    d = 10 * np.log10(pe[k:] + 1e-10) - 10 * np.log10(pr[k:] + 1e-10)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def miary(est: np.ndarray, ref: np.ndarray, mos_model, dev) -> dict:
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    e16, r16 = (librosa.resample(v, orig_sr=SR, target_sr=16000) for v in (est, ref))
    try:
        p = pesq(16000, r16, e16, "wb")
    except Exception:
        p = float("nan")
    return {"pesq": p, "estoi": float(stoi(r16, e16, 16000, extended=True)), "si_sdr": si_sdr(est, ref),
            "lsd_16_24": lsd_gora(est, ref), "utmos": utmos(mos_model, torch.from_numpy(est).to(dev), SR)}


@torch.no_grad()
def popraw(model, y: torch.Tensor, gen: torch.Generator, odcinek_s: float, zakladka_s: float) -> torch.Tensor:
    """y: [N] na urządzeniu modelu -> [N]."""
    from sgmse.util.other import pad_spec
    norm = y.abs().max().clamp_min(1e-8)
    y = y / norm
    N = y.shape[0]
    L, Z = int(odcinek_s * SR), int(zakladka_s * SR)
    starty = [0] if N <= L else list(range(0, N - Z, L - Z))
    wynik = torch.zeros(N, device=y.device)
    waga = torch.zeros(N, device=y.device)
    for s in starty:
        e = min(s + L, N)
        kaw = y[s:e].unsqueeze(0)
        cond = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(kaw)), 0))
        z = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=cond.dtype)
        t = torch.ones(1, device=z.device)
        r = torch.full((1,), 0.003, device=z.device)
        z = z - model.dnn(z, t, r, cond)
        x = model.to_audio(z.squeeze(), e - s).reshape(-1)
        w = torch.ones(e - s, device=y.device)
        if len(starty) > 1:
            if s > 0:
                w[:Z] = torch.linspace(0, 1, Z, device=y.device)
            if e < N:
                w[-Z:] = torch.minimum(w[-Z:], torch.linspace(1, 0, Z, device=y.device))
        wynik[s:e] += x * w
        waga[s:e] += w
    return wynik / waga.clamp_min(1e-8) * norm


def srednie(wiersze: list[dict], klucz_prefiks: str) -> dict:
    out = {}
    for k in ("pesq", "estoi", "si_sdr", "lsd_16_24", "utmos"):
        v = np.array([w[f"{klucz_prefiks}_{k}"] for w in wiersze if f"{klucz_prefiks}_{k}" in w], dtype=float)
        if len(v):
            out[k] = float(np.nanmean(v))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", default="../dane/cose_pary")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--wyjscie", required=True, help="tu: wyniki.csv, podsumowanie.json i próbki do odsłuchu")
    p.add_argument("--klipy", type=int, default=100)
    p.add_argument("--odsluch", type=int, default=12, help="ile klipów zapisać jako wav (clean/noisy/cose)")
    p.add_argument("--odcinek", type=float, default=10.0, help="s; dłuższe klipy liczone odcinkami")
    p.add_argument("--zakladka", type=float, default=0.5)
    p.add_argument("--ziarno", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.wyjscie, exist_ok=True)
    pliki = sorted(glob(join(args.base_dir, "valid", "clean", "*.wav")))[: args.klipy]
    mos_model = wczytaj_utmos(dev)

    model = None
    if args.ckpt:
        from sgmse.model import ScoreModel
        model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=dev, base_dir=args.base_dir,
                                             weights_only=False)   # własny checkpoint
        model.to(dev)
        model.eval()   # przełącza na wagi EMA
        print(f"COSE: {args.ckpt}")
    gen = torch.Generator(device=dev).manual_seed(args.ziarno)

    wiersze = []
    for i, pc in enumerate(pliki):
        nazwa = basename(pc)
        x, _ = sf.read(pc, dtype="float32")
        y, _ = sf.read(join(args.base_dir, "valid", "noisy", nazwa), dtype="float32")
        w = {"plik": nazwa, "sek": len(x) / SR,
             "oryginal_utmos": utmos(mos_model, torch.from_numpy(x).to(dev), SR)}
        w.update({f"kodek_{k}": v for k, v in miary(y, x, mos_model, dev).items()})
        if model is not None:
            xh = popraw(model, torch.from_numpy(y).to(dev), gen, args.odcinek, args.zakladka).clamp(-1, 1)
            xh = xh.cpu().numpy()
            w.update({f"cose_{k}": v for k, v in miary(xh, x, mos_model, dev).items()})
        if i < args.odsluch:
            stem = os.path.splitext(nazwa)[0]
            sf.write(join(args.wyjscie, f"{i:02d}_{stem}_1_oryginal.wav"), x, SR, subtype="PCM_16")
            sf.write(join(args.wyjscie, f"{i:02d}_{stem}_2_kodek.wav"), y, SR, subtype="PCM_16")
            if model is not None:
                sf.write(join(args.wyjscie, f"{i:02d}_{stem}_3_cose.wav"), xh, SR, subtype="PCM_16")
        wiersze.append(w)
        print(f"[{i + 1}/{len(pliki)}] {nazwa}  UTMOS oryg {w['oryginal_utmos']:.3f}  kodek {w['kodek_utmos']:.3f}"
              + (f"  cose {w['cose_utmos']:.3f}" if model is not None else ""), flush=True)

    with open(join(args.wyjscie, "wyniki.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(wiersze[0].keys()))
        wr.writeheader(); wr.writerows(wiersze)
    pods = {"ckpt": args.ckpt, "klipy": len(wiersze),
            "oryginal_utmos": float(np.mean([w["oryginal_utmos"] for w in wiersze])),
            "kodek": srednie(wiersze, "kodek")}
    if model is not None:
        pods["cose"] = srednie(wiersze, "cose")
    with open(join(args.wyjscie, "podsumowanie.json"), "w", encoding="utf-8") as f:
        json.dump(pods, f, indent=2, ensure_ascii=False)

    print(f"\nOryginał UTMOS {pods['oryginal_utmos']:.3f}")
    print(f"{'':8s} {'UTMOS':>7s} {'PESQ':>6s} {'ESTOI':>6s} {'SI-SDR':>7s} {'LSD16-24':>9s}")
    for k in ("kodek", "cose"):
        if k in pods:
            m = pods[k]
            print(f"{k:8s} {m['utmos']:7.3f} {m['pesq']:6.3f} {m['estoi']:6.3f} {m['si_sdr']:7.2f} {m['lsd_16_24']:9.2f}")


if __name__ == "__main__":
    main()
