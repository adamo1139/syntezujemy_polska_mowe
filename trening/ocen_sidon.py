#!/usr/bin/env python3
"""UTMOS łańcucha z Sidonem v0.1 (MIT, sarulab-speech/sidon-v0.1) i stałym korektorem barwy (EQ).

Dwa tryby (można oba naraz):
  --pary KATALOG    para clean/noisy (np. ../dane/cose_pary/valid): oryginał VoxCPM2, kodek, kodek→Sidon,
                    kodek→Sidon→EQ, a dla porównania też oryginał→Sidon(→EQ)
  --katalog KATALOG dowolne wav (np. wyjście TTS): surowe, →Sidon, →Sidon→EQ
EQ: korektor_sidona_db.npy (różnica średnich widm oryginałów VoxCPM2 i tych samych oryginałów po Sidonie, pasma 1/3
oktawy; nie zależy od dekodera). Sidon: ta sama ścieżka co demo autorów (górnoprzepustowy 50 Hz, w2v-BERT 2.0 na
16 kHz, wokoder 48 kHz), wejście normalizowane do szczytu 0,9.

Wymaga transformers i huggingface_hub (wagi Sidona i ekstraktor cech pobierane z HF przy pierwszym uruchomieniu).
Uruchamiać z katalogu trening/ (karta 4):
  CUDA_VISIBLE_DEVICES=4 python ocen_sidon.py --pary ../dane/cose_pary/valid --wyjscie ../syntezy/sidon_utmos
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import istft, stft

from training.utmos import utmos, wczytaj_utmos

SR = 48000
NP = dict(nperseg=2048, noverlap=1536)


class Sidon:
    def __init__(self, dev):
        import transformers
        from huggingface_hub import hf_hub_download
        rodzaj = "cuda" if dev.type == "cuda" else "cpu"
        self.dev = dev
        self.fe = torch.jit.load(hf_hub_download("sarulab-speech/sidon-v0.1", f"feature_extractor_{rodzaj}.pt"),
                                 map_location=dev).to(dev)
        self.dec = torch.jit.load(hf_hub_download("sarulab-speech/sidon-v0.1", f"decoder_{rodzaj}.pt"),
                                  map_location=dev).to(dev)
        self.pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    @torch.inference_mode()
    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        docelowo = int(SR / sr * x.shape[0])
        x = torch.from_numpy(0.9 * x / max(np.abs(x).max(), 1e-8)).float().view(1, -1)
        wav = torchaudio.functional.highpass_biquad(x, sr, 50)
        wav16 = torch.nn.functional.pad(torchaudio.functional.resample(wav, sr, 16_000), (0, 24000))
        wyniki, cache = [], None
        for kawalek in wav16.view(-1).split(16000 * 96):
            wej = self.pre(torch.nn.functional.pad(kawalek, (160, 160)).numpy(), sampling_rate=16000,
                           return_tensors="pt")
            cecha = self.fe(wej["input_features"].to(self.dev))["last_hidden_state"]
            if cache is not None:
                cecha = torch.cat([cache, cecha], dim=1)
            wyniki.append(self.dec(cecha.transpose(1, 2)).view(-1)[:-960])
            cache = cecha[:, -1:]
        return torch.cat(wyniki).float().cpu().numpy()[:docelowo]


def korektor(y: np.ndarray, krzywa_db: np.ndarray) -> np.ndarray:
    _, _, Y = stft(y, SR, **NP)
    _, yc = istft(Y * 10 ** (krzywa_db[:, None] / 20), SR, **NP)
    yc = yc[: len(y)]
    return (yc / max(1.0, np.abs(yc).max() / 0.99)).astype(np.float32)


def wczytaj(p: str) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pary", help="katalog z clean/ i noisy/")
    p.add_argument("--katalog", help="katalog dowolnych wav (bez par)")
    p.add_argument("--wyjscie", required=True)
    p.add_argument("--klipy", type=int, default=100)
    p.add_argument("--odsluch", type=int, default=8, help="ile klipów zapisać do odsłuchu")
    p.add_argument("--eq", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "korektor_sidona_db.npy"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not (args.pary or args.katalog):
        p.error("podaj --pary albo --katalog")

    dev = torch.device(args.device)
    os.makedirs(args.wyjscie, exist_ok=True)
    krzywa = np.load(args.eq)[1]
    mos, sidon = wczytaj_utmos(dev), Sidon(dev)

    def ocen(w: np.ndarray) -> float:
        return utmos(mos, torch.from_numpy(w).to(dev), SR)

    zestawy = []
    if args.pary:
        zestawy.append(("pary", sorted(glob(os.path.join(args.pary, "clean", "*.wav")))[: args.klipy]))
    if args.katalog:
        zestawy.append(("katalog", sorted(glob(os.path.join(args.katalog, "**", "*.wav"), recursive=True))[: args.klipy]))

    podsumowanie = {}
    for tryb, pliki in zestawy:
        wyniki: dict[str, list[float]] = {}
        for i, plik in enumerate(pliki):
            wersje = {}
            if tryb == "pary":
                wersje["oryginal"] = wczytaj(plik)
                wersje["kodek"] = wczytaj(plik.replace(f"{os.sep}clean{os.sep}", f"{os.sep}noisy{os.sep}"))
                zrodla = ("kodek", "oryginal")
            else:
                wersje["surowe"] = wczytaj(plik)
                zrodla = ("surowe",)
            for z in zrodla:
                wersje[f"{z}_sidon"] = sidon(wersje[z], SR)
                wersje[f"{z}_sidon_eq"] = korektor(wersje[f"{z}_sidon"], krzywa)
            for nazwa, w in wersje.items():
                wyniki.setdefault(nazwa, []).append(ocen(w))
                if i < args.odsluch:
                    stem = os.path.splitext(os.path.basename(plik))[0]
                    sf.write(os.path.join(args.wyjscie, f"{tryb}_{i:02d}_{stem}_{nazwa}.wav"), w, SR, subtype="PCM_16")
            print(f"[{tryb} {i + 1}/{len(pliki)}] " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in wyniki.items()), flush=True)
        podsumowanie[tryb] = {k: float(np.mean(v)) for k, v in wyniki.items()}

    with open(os.path.join(args.wyjscie, "utmos.json"), "w", encoding="utf-8") as f:
        json.dump(podsumowanie, f, indent=2, ensure_ascii=False)
    print()
    for tryb, m in podsumowanie.items():
        print(f"{tryb}:")
        for k, v in m.items():
            print(f"  {k:22s} UTMOS {v:.3f}")


if __name__ == "__main__":
    main()
