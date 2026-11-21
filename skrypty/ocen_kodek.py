#!/usr/bin/env python3
"""Liczbowa ocena próbek kodeka: odległość rekonstrukcji od oryginału dla każdego kroku treningu.

Czyta pliki `step_<krok>_*_recon.wav` z katalogu eval trenera kodeka i porównuje je z nagraniem źródłowym.
Im mniej, tym bliżej oryginału:
  log-mel L1   średnia różnica logarytmicznych widm melowych (80 pasm, całe pasmo)
  LSD          odległość widm w dB, osobno dla 0–8, 8–16 i 16–24 kHz

Przed porównaniem próbka jest obcinana do długości oryginału i wyrównywana głośnością (mediana energii ramek).
Próbki zapisane przez starszą wersję trenera miały za końcem nagrania duży skok, przez który normalizacja
ściszała całą mowę o kilkadziesiąt dB; kolumna „ściszenie” pokazuje, ile trzeba było oddać.
`--zapisz KATALOG` zapisuje poprawione próbki do odsłuchu.

To miara na jednym nagraniu, więc pokazuje kierunek i moment nasycenia, a nie jakość ogólną.

Przykład:
  python skrypty/ocen_kodek.py --eval modele/kodek_48k/eval --oryginal glosy/poziomka_dluga.wav
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import soundfile as sf

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
N_FFT, HOP = 2048, 512


def widmo(x: np.ndarray) -> np.ndarray:
    okno = np.hanning(N_FFT)
    ramki = np.stack([x[i : i + N_FFT] * okno for i in range(0, len(x) - N_FFT, HOP)])
    return np.abs(np.fft.rfft(ramki, axis=1))


def filtry_melowe(sr: int, n_mels: int = 80) -> np.ndarray:
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    na_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    z_mel = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    punkty = z_mel(np.linspace(na_mel(0), na_mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        a, b, c = punkty[i], punkty[i + 1], punkty[i + 2]
        fb[i] = np.clip(np.minimum((f - a) / (b - a + 1e-9), (c - f) / (c - b + 1e-9)), 0, None)
    return fb


def mono(sciezka: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(sciezka)
    return (x.mean(1) if x.ndim > 1 else x), sr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default=str(KATALOG_PROJEKTU / "modele" / "kodek_48k" / "eval"))
    p.add_argument("--oryginal", default=str(KATALOG_PROJEKTU / "glosy" / "poziomka_dluga.wav"))
    p.add_argument("--zapisz", help="katalog na próbki obcięte do długości oryginału i wyrównane głośnością")
    args = p.parse_args()

    oryginal, sr = mono(args.oryginal)
    fb = filtry_melowe(sr)
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    krok_z = lambda sciezka: int(re.search(r"step_(\d+)", sciezka).group(1))
    pliki = sorted(glob.glob(str(Path(args.eval) / "step_*_recon.wav")), key=krok_z)
    if not pliki:
        raise SystemExit(f"Brak plików step_*_recon.wav w {args.eval}")

    print(f"oryginał: {args.oryginal} ({len(oryginal) / sr:.2f} s, {sr} Hz)")
    print(f"{'krok':>7} | {'log-mel L1':>10} | {'LSD 0-8k':>9} | {'LSD 8-16k':>9} | {'LSD 16-24k':>10} | {'ściszenie':>9}")
    if args.zapisz:
        Path(args.zapisz).mkdir(parents=True, exist_ok=True)
    R = widmo(oryginal)
    for sciezka in pliki:
        y, sr_y = mono(sciezka)
        if sr_y != sr:
            raise SystemExit(f"{sciezka}: {sr_y} Hz, oryginał ma {sr} Hz")
        y = y[: len(oryginal)]
        Y = widmo(np.pad(y, (0, len(oryginal) - len(y))))
        wzmocnienie = float(np.sqrt(np.median((R**2).sum(1)) / (np.median((Y**2).sum(1)) + 1e-20)))
        Y = Y * wzmocnienie
        log_mel = np.abs(np.log(R @ fb.T + 1e-5) - np.log(Y @ fb.T + 1e-5)).mean()

        def lsd(od: float, do: float) -> float:
            pasmo = (f >= od) & (f < do)
            roznica = 20 * np.log10(R[:, pasmo] + 1e-5) - 20 * np.log10(Y[:, pasmo] + 1e-5)
            return float(np.sqrt((roznica**2).mean(1)).mean())

        print(
            f"{krok_z(sciezka):>7} | {log_mel:>10.3f} | {lsd(0, 8000):>9.2f} | {lsd(8000, 16000):>9.2f} | "
            f"{lsd(16000, 24000):>10.2f} | {20 * np.log10(wzmocnienie):>6.1f} dB"
        )
        if args.zapisz:
            sf.write(str(Path(args.zapisz) / f"krok_{krok_z(sciezka):06d}.wav"), np.clip(y * wzmocnienie, -1.0, 1.0), sr)


if __name__ == "__main__":
    main()
