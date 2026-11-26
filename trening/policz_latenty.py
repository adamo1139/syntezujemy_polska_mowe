#!/usr/bin/env python3
"""Liczy raz latenty kodeka dla wszystkich klipów i zapisuje je na dysk (float16), żeby trener ich nie liczył.

Dla każdego klipu: dopełnienie zerami do pełnych porcji 512 × 6 próbek (jak w treningu i inferencji),
widmo, enkoder kodeka, kompresja ×6 i obcięcie do ceil(próbki / 3072) klatek. Zapisywane są latenty
NIEznormalizowane (normalizację statystykami robi trener), więc ten sam plik działa z każdymi statystykami
policzonymi tym samym kodekiem.

Wynik w katalogu --wyjscie:
  latenty.npy    float16 [łącznie_klatek, 144], wszystkie klipy jeden za drugim
  indeks.json    odcisk enkodera i dla każdej ścieżki: [przesunięcie, liczba_klatek, sekundy]

Uruchamiać z katalogu trening/:
  python policz_latenty.py --metadata ../dane/bluetts_train.csv ../dane/bluetts_val.csv \\
    --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --wyjscie ../dane/latenty --device cuda:1
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bluecodec import LatentEncoder
from training.data.audio_utils import ensure_sr
from training.utils import LinearMelSpectrogram, compress_latents, encoder_fingerprint


class Klipy(Dataset):
    def __init__(self, sciezki, sr):
        self.sciezki, self.sr = sciezki, sr

    def __len__(self):
        return len(self.sciezki)

    def __getitem__(self, i):
        x, sr = sf.read(self.sciezki[i], dtype="float32", always_2d=True)
        x = torch.from_numpy(x).mean(dim=1)
        if sr != self.sr:
            x = ensure_sr(x, sr, self.sr, device="cpu").reshape(-1)
        return i, x


def zloz(partia, porcja):
    """Każdy klip dopełniony zerami do wielokrotności `porcja` (w partii wszystkie mają tę samą liczbę klatek)."""
    idx = [i for i, _ in partia]
    n = max(-(-x.shape[0] // porcja) * porcja for _, x in partia)
    return idx, torch.stack([F.pad(x, (0, n - x.shape[0])) for _, x in partia]), [x.shape[0] for _, x in partia]



def partie_gpu(moje, sciezki, sr, porcja, device, rozmiar):
    """Jak DataLoader z `zloz`, ale klipy .ogg dekodowane przez opus_gpu na tej samej karcie, partiami po `rozmiar`
    (następna grupa czyta się z dysku w tle); inne formaty przez soundfile. Wynik od razu na `device`."""
    import cupy as cp
    from opus_gpu import decode_many
    cp.cuda.Device(device.index).use()
    plaskie = [sciezki[i] for p in moje for i in p if sciezki[i].lower().endswith((".ogg", ".opus"))]

    def odporny():
        """decode_many po kolei; plik, którego czytnik Ogg opus_gpu nie przyjmie, daje None (dekoduje go wtedy
        soundfile), a dekodowanie rusza od nowa od tej samej grupy bez niego."""
        poz, zle = 0, set()
        while poz < len(plaskie):
            reszta = [s for s in plaskie[poz:] if s not in zle]
            try:
                for sciezka, pcm in decode_many(reszta, backend="cuda", output="cupy", batch_size=rozmiar):
                    while plaskie[poz] in zle:   # pominięte wcześniej: na swoim miejscu None
                        poz += 1
                        yield None
                    if str(sciezka) != plaskie[poz]:
                        raise RuntimeError(f"opus_gpu: kolejność się rozjechała ({sciezka} zamiast {plaskie[poz]})")
                    poz += 1
                    yield pcm
                while poz < len(plaskie) and plaskie[poz] in zle:
                    poz += 1
                    yield None
                return
            except ValueError as blad:
                zly = str(blad).split(": ", 1)[0]
                if zly not in plaskie or zly in zle:
                    raise
                print(f"  opus_gpu nie czyta {zly} ({blad.__cause__ or blad}); ten plik przez soundfile", flush=True)
                zle.add(zly)

    strumien = odporny()
    for p in moje:
        xs = []
        for i in p:
            pcm = next(strumien) if sciezki[i].lower().endswith((".ogg", ".opus")) else None
            if pcm is not None:
                x = torch.from_dlpack(pcm).to(torch.float32).mean(dim=1) / 32768.0
                # pcm to widok na bufor dekodera, który kolejna grupa nadpisze: kopia musi się skończyć, zanim
                # generator ruszy dalej (inaczej latenty z cudzego audio)
                torch.cuda.current_stream(device).synchronize()
                if sr != 48000:
                    x = ensure_sr(x, 48000, sr, device=device).reshape(-1)
            else:
                try:
                    _, x = Klipy(sciezki, sr)[i]
                except Exception as blad:
                    raise SystemExit(f"Nie da się odczytać {sciezki[i]} ani przez opus_gpu, ani przez soundfile ({blad}). "
                                     "Usuń ten klip z danych (przygotuj_audio.py --usun) i policz latenty od nowa.")
                x = x.to(device)
            xs.append(x)
        n = max(-(-x.shape[0] // porcja) * porcja for x in xs)
        yield p, torch.stack([F.pad(x, (0, n - x.shape[0])) for x in xs]), [x.shape[0] for x in xs]


def licz(rank: int, z: dict) -> None:
    """Jeden proces na kartę: partie rank, rank + n, ... zapisywane do wspólnego pliku (memmap)."""
    device = torch.device(z["urzadzenia"][rank])
    ae = z["cfg"]["ae"]
    spec = ae["encoder"].get("spec_processor", {})
    sr, porcja, ccf = z["sr"], z["porcja"], z["ccf"]
    mel = LinearMelSpectrogram(sample_rate=spec.get("sample_rate", sr), n_fft=spec.get("n_fft", 2048),
                               win_length=spec.get("win_length", spec.get("n_fft", 2048)),
                               hop_length=spec.get("hop_length", 512), n_mels=spec.get("n_mels", 228)).to(device)
    enc = LatentEncoder(cfg=ae["encoder"]).to(device)
    enc.load_state_dict(torch.load(z["ae_checkpoint"], map_location="cpu")["encoder"])
    enc.eval().requires_grad_(False)
    n = len(z["urzadzenia"])
    moje = z["partie"][rank::n]

    class Partie(torch.utils.data.Sampler):
        def __iter__(self): return iter(moje)
        def __len__(self): return len(moje)

    mm = np.load(z["plik"], mmap_mode="r+")
    przes, klatki = z["przes"], z["klatki"]
    if z["dekoder"] == "gpu":
        loader = partie_gpu(moje, z["sciezki"], sr, porcja, device, z["gpu_partia"])
    else:
        loader = DataLoader(Klipy(z["sciezki"], sr), batch_sampler=Partie(), num_workers=z["num_workers"],
                            collate_fn=functools.partial(zloz, porcja=porcja))
    with torch.no_grad():
        for nr, (idx, x, dl) in enumerate(loader, 1):
            x = x.to(device)
            lat = compress_latents(enc(mel(x)), factor=ccf).float().cpu().numpy()   # [B, 144, T]
            for j, i in enumerate(idx):
                mm[przes[i] : przes[i] + klatki[i]] = lat[j, :, : klatki[i]].T.astype(np.float16)
            if rank == 0 and (nr % 100 == 0 or nr == len(moje)):
                print(f"  karta 0: {nr}/{len(moje)} partii (razem {len(z['partie'])})  [{time.time() - z['t0']:.0f} s]",
                      flush=True)
    mm.flush()
    del mm

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", nargs="+", required=True, help="pliki CSV trenera (bluetts_train.csv, bluetts_val.csv)")
    p.add_argument("--config", default="config/polski_100m.json")
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--wyjscie", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                   help="karta albo kilka po przecinku, np. cuda:1,cuda:2 (partie dzielone między nie)")
    p.add_argument("--sekund_na_partie", type=float, default=600.0, help="budżet dopełnionej długości partii")
    p.add_argument("--num_workers", type=int, default=8, help="procesy wczytujące na każdą kartę (dekoder cpu)")
    p.add_argument("--dekoder", choices=["auto", "cpu", "gpu"], default="auto",
                   help="gpu: pliki .ogg dekodowane przez opus_gpu na karcie liczącej latenty; auto = gpu, gdy "
                        "opus_gpu jest zainstalowane i liczymy na CUDA")
    p.add_argument("--gpu_partia", type=int, default=1024, help="dekoder gpu: klipów w jednym wywołaniu dekodera")
    args = p.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    ae = cfg["ae"]
    spec = ae["encoder"].get("spec_processor", {})
    sr = int(ae.get("sample_rate", spec.get("sample_rate", 44100)))
    hop = int(spec.get("hop_length", 512))
    ccf = int(cfg["ttl"]["chunk_compress_factor"])
    porcja = hop * ccf

    urzadzenia = [d.strip() for d in args.device.split(",")]
    if args.dekoder == "auto":
        import importlib.util
        args.dekoder = "gpu" if (all(d.startswith("cuda") for d in urzadzenia)
                                 and importlib.util.find_spec("opus_gpu") and importlib.util.find_spec("cupy")) else "cpu"
    print(f"Dekoder audio: {args.dekoder}")
    enc = LatentEncoder(cfg=ae["encoder"])
    enc.load_state_dict(torch.load(args.ae_checkpoint, map_location="cpu")["encoder"])
    odcisk = encoder_fingerprint(enc.state_dict())[:12]
    del enc

    sciezki = []
    for m in args.metadata:
        df = pd.read_csv(m, sep=None, engine="python")
        sciezki += [str(x).strip() for x in df["filename"]]
    sciezki = list(dict.fromkeys(sciezki))
    print(f"Klipów: {len(sciezki)} z {len(args.metadata)} plików CSV; odcisk enkodera {odcisk}")

    t0 = time.time()
    probki = [sf.info(s).frames if sf.info(s).samplerate == sr else int(sf.info(s).duration * sr) for s in sciezki]
    klatki = [max(1, math.ceil(n / porcja)) for n in probki]
    przes = np.concatenate([[0], np.cumsum(klatki)[:-1]]).astype(np.int64)
    C = int(cfg["ttl"]["latent_dim"]) * ccf
    print(f"Długości odczytane ({time.time() - t0:.0f} s); łącznie {sum(klatki)} klatek × {C} = "
          f"{sum(klatki) * C * 2 / 1e9:.1f} GB w float16")

    os.makedirs(args.wyjscie, exist_ok=True)
    plik = os.path.join(args.wyjscie, "latenty.npy")
    mm = np.lib.format.open_memmap(plik + ".tmp", mode="w+", dtype=np.float16, shape=(int(sum(klatki)), C))

    # Partie tylko z klipów o tej samej liczbie klatek: każdy jest wtedy dopełniony dokładnie do swojej
    # wielokrotności 3072 próbek, czyli tak samo jak przy kodowaniu pojedynczego klipu w inferencji.
    # (Dopełnienie do najdłuższego w partii zmieniałoby ostatnie klatki krótszych klipów.)
    grupy: dict[int, list[int]] = {}
    for i in range(len(sciezki)):
        grupy.setdefault(klatki[i], []).append(i)
    partie = []
    for k, ids in sorted(grupy.items()):
        na_partie = max(1, int(args.sekund_na_partie // (k * porcja / sr)))
        partie += [ids[j:j + na_partie] for j in range(0, len(ids), na_partie)]

    mm.flush(); del mm
    t0 = time.time()
    zadanie = dict(cfg=cfg, ae_checkpoint=args.ae_checkpoint, sciezki=sciezki, sr=sr, porcja=porcja, ccf=ccf,
                   partie=partie, przes=przes, klatki=klatki, plik=plik + ".tmp", num_workers=args.num_workers,
                   dekoder=args.dekoder, gpu_partia=args.gpu_partia,
                   urzadzenia=urzadzenia, t0=t0)
    if len(urzadzenia) == 1:
        licz(0, zadanie)
    else:
        print(f"Karty: {', '.join(urzadzenia)} (każda co {len(urzadzenia)}. partię)")
        torch.multiprocessing.spawn(licz, args=(zadanie,), nprocs=len(urzadzenia), join=True)
    os.replace(plik + ".tmp", plik)

    indeks = {"odcisk_enkodera": odcisk, "sample_rate": sr, "hop": hop, "ccf": ccf, "kanaly": C,
              "klipy": {s: [int(przes[i]), int(klatki[i]), probki[i] / sr] for i, s in enumerate(sciezki)}}
    with open(os.path.join(args.wyjscie, "indeks.json"), "w", encoding="utf-8") as f:
        json.dump(indeks, f)
    print(f"Zapisano {plik} i indeks.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
