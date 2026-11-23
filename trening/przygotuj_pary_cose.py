#!/usr/bin/env python3
"""Pary do treningu postfiltru COSE: oryginał VoxCPM2 (clean) i ten sam klip po kodeku (noisy), 48 kHz.

Postfiltr ma się nauczyć zamieniać wyjście naszego dekodera w nagranie jak z VoxCPM2. Układ katalogów taki, jakiego
oczekuje kod COSE (sgmse/data_module.py): <wyjscie>/{train,valid}/{clean,noisy}/<id>.wav, te same nazwy w clean
i noisy. Kodek: enkoder + dekoder z --ae_checkpoint (domyślnie dekoder z modułem poprawki); klip dopełniany do pełnych
porcji 512 × 6 próbek, jak przy liczeniu latentów, a wynik obcinany do długości oryginału.

Uruchamiać z katalogu trening/ (karta 4):
  python przygotuj_pary_cose.py --wyjscie ../dane/cose_pary --klipy 12000 --device cuda:4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import soundfile as sf
import torch

from bluecodec.utils import LinearMelSpectrogram
from training.data.audio_utils import ensure_sr
from usrednij_kodek import wczytaj_wagi, zbuduj


def sciezki(plik: str) -> list[str]:
    with open(plik, encoding="utf-8", newline="") as f:
        return [w["filename"].strip() for w in csv.DictReader(f)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wyjscie", required=True)
    p.add_argument("--ae_checkpoint", default="../modele/kodek_48k_postnet/ae_wsm_6_9.pt")
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--train_csv", default="../dane/bluetts_train.csv")
    p.add_argument("--val_csv", default="../dane/bluetts_val.csv")
    p.add_argument("--klipy", type=int, default=12000, help="ile klipów treningowych (losowo, stałe ziarno)")
    p.add_argument("--klipy_val", type=int, default=100)
    p.add_argument("--min_sec", type=float, default=2.2, help="krótsze pomijane (wycinek treningowy COSE ma ok. 2 s)")
    p.add_argument("--max_sec", type=float, default=30.0)
    p.add_argument("--ziarno", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = json.load(open(args.config, encoding="utf-8"))
    ae = cfg["ae"]
    sr = int(ae["data"]["sample_rate"])
    spec = ae["encoder"].get("spec_processor", {})
    porcja = int(spec.get("hop_length", 512)) * int(cfg["ttl"]["chunk_compress_factor"])
    mel = LinearMelSpectrogram(sample_rate=spec.get("sample_rate", sr), n_fft=spec.get("n_fft", 2048),
                               hop_length=spec.get("hop_length", 512), win_length=spec.get("win_length", 2048),
                               n_mels=spec.get("n_mels", 228)).to(dev)
    ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    enc, dec = zbuduj(ae, dev, ckpt)
    e, d, _ = wczytaj_wagi(args.ae_checkpoint)
    enc.load_state_dict(e); dec.load_state_dict(d); enc.eval(); dec.eval()
    print(f"Kodek: {args.ae_checkpoint}")

    los = random.Random(args.ziarno)
    for podzial, plik, ile in (("valid", args.val_csv, args.klipy_val), ("train", args.train_csv, args.klipy)):
        lista = sciezki(plik)
        los.shuffle(lista)
        for k in ("clean", "noisy"):
            os.makedirs(os.path.join(args.wyjscie, podzial, k), exist_ok=True)
        zrobione, sekundy, t0 = 0, 0.0, time.time()
        for sciezka in lista:
            if zrobione >= ile:
                break
            nazwa = os.path.splitext(os.path.basename(sciezka))[0] + ".wav"
            cel_c, cel_n = (os.path.join(args.wyjscie, podzial, k, nazwa) for k in ("clean", "noisy"))
            info = sf.info(sciezka)
            if not (args.min_sec <= info.duration <= args.max_sec):
                continue
            if os.path.exists(cel_c) and os.path.exists(cel_n):   # wznawianie
                zrobione += 1; sekundy += info.duration
                continue
            x, sr_x = sf.read(sciezka, dtype="float32", always_2d=True)
            x = torch.from_numpy(x.mean(1)).to(dev)
            if sr_x != sr:
                x = ensure_sr(x, sr_x, sr, device=dev).reshape(-1)
            n = x.shape[0]
            with torch.no_grad():
                y = dec(enc(mel(torch.nn.functional.pad(x, (0, (-n) % porcja)).unsqueeze(0)))).reshape(-1)[:n]
            sf.write(cel_c, x.cpu().numpy(), sr, subtype="PCM_16")
            sf.write(cel_n, y.clamp(-1, 1).cpu().numpy(), sr, subtype="PCM_16")
            zrobione += 1; sekundy += n / sr
            if zrobione % 500 == 0:
                print(f"  {podzial}: {zrobione}/{ile} klipów, {sekundy / 3600:.1f} h  [{time.time() - t0:.0f} s]", flush=True)
        print(f"{podzial}: {zrobione} par, {sekundy / 3600:.2f} h -> {os.path.join(args.wyjscie, podzial)}")


if __name__ == "__main__":
    main()
