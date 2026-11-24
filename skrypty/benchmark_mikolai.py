#!/usr/bin/env python3
"""Benchmark MikołAI (biblioteka `mikolai`) na długich tekstach: synteza każdego pliku .txt z katalogu, pomiar czasu,
zapis nagrań i wyników do JSON. Pasek postępu po kawałkach tekstu.

Na jednym rdzeniu CPU (taskset przypina proces do rdzenia 0, --watki 1 ustawia jeden wątek PyTorch):
  taskset -c 0 python skrypty/benchmark_mikolai.py --device cpu --watki 1 --partia 64 \\
    --wyjscie probki/benchmark/wyniki/mikolai_cpu_1rdzen

Na GPU:
  python skrypty/benchmark_mikolai.py --device cuda --partia 64 --wyjscie probki/benchmark/wyniki/mikolai_gpu

Wyniki: <wyjscie>/<nazwa>.wav (48 kHz) i <wyjscie>/wyniki.json (czas, długość nagrania, RTF, sprzęt, ustawienia).
Wymaga biblioteki mikolai (github.com/adamo1139/mikolai) i soundfile.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time


def nazwa_cpu() -> str:
    try:
        for linia in open("/proc/cpuinfo", encoding="utf-8"):
            if linia.startswith("model name"):
                return linia.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teksty", default="probki/benchmark", help="katalog z plikami .txt")
    p.add_argument("--wyjscie", required=True)
    p.add_argument("--repo", default=None, help="repo HF albo katalog z wagami (domyślnie cpral/MikolAI-109M-1-Polski-TTS)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--watki", type=int, default=None, help="wątki PyTorch (1 = jeden rdzeń; razem z taskset -c 0)")
    p.add_argument("--partia", type=int, default=64)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--kroki", type=int, default=16)
    p.add_argument("--ziarno", type=int, default=0)
    p.add_argument("--pliki", nargs="*", default=None, help="tylko te nazwy plików (bez .txt)")
    a = p.parse_args()

    if a.watki:   # przed importem torcha, żeby biblioteki BLAS/OpenMP też wzięły jeden wątek
        for zm in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[zm] = str(a.watki)
    import numpy as np
    import soundfile as sf
    import torch
    if a.watki:
        torch.set_num_threads(a.watki)
        torch.set_num_interop_threads(1)
    from mikolai import DOMYSLNE_REPO, MikolAI, podziel, sklej
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    pliki = sorted(glob.glob(os.path.join(a.teksty, "*.txt")))
    if a.pliki:
        pliki = [f for f in pliki if os.path.splitext(os.path.basename(f))[0] in a.pliki]
    if not pliki:
        raise SystemExit(f"Brak plików .txt w {a.teksty}")
    os.makedirs(a.wyjscie, exist_ok=True)

    t0 = time.time()
    tts = MikolAI.from_pretrained(a.repo or DOMYSLNE_REPO, device=a.device)
    tts.synteza("Rozgrzewka.", kroki=2)
    print(f"Model wczytany w {time.time() - t0:.1f} s; {a.device}, wątki {torch.get_num_threads()}, partia {a.partia}, "
          f"CFG {a.cfg:g}, kroki {a.kroki}", flush=True)
    sr = tts.sample_rate

    wyniki = {"sprzet": {"device": a.device, "cpu": nazwa_cpu(), "watki_torch": torch.get_num_threads(),
                         "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
                         "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
              "ustawienia": {"partia": a.partia, "cfg": a.cfg, "kroki": a.kroki, "ziarno": a.ziarno,
                             "repo": a.repo or DOMYSLNE_REPO, "torch": torch.__version__,
                             "max_znakow": __import__("mikolai").MAKS_ZNAKOW, "przerwy": __import__("mikolai").PRZERWY},
              "pliki": {}}
    for plik in pliki:
        nazwa = os.path.splitext(os.path.basename(plik))[0]
        tekst = open(plik, encoding="utf-8").read()
        podzial = podziel(tekst)
        kawalki = [k for k, _ in podzial]
        # to samo co MikolAI.synteza(), ale partia po partii, żeby pokazać postęp
        gen = torch.Generator(device=tts.device)
        gen.manual_seed(a.ziarno)
        start = time.time()
        el = tts._przygotuj(kawalki, "poziomka", 1.0, gen)
        kolejnosc = sorted(range(len(el)), key=lambda i: el[i]["T"])
        nagrania = [None] * len(el)
        pasek = tqdm(total=len(el), desc=nazwa, unit="kaw", dynamic_ncols=True) if tqdm else None
        for i in range(0, len(kolejnosc), a.partia):
            grupa = kolejnosc[i: i + a.partia]
            for k, w in zip(grupa, tts._dekoduj(tts._plyn([el[k] for k in grupa], "poziomka", a.cfg, a.kroki))):
                nagrania[k] = w
            if pasek:
                pasek.update(len(grupa))
            else:
                print(f"  {nazwa}: {min(i + a.partia, len(el))}/{len(el)} kawałków [{time.time() - start:.0f} s]", flush=True)
        if pasek:
            pasek.close()
        audio = sklej(nagrania, [k for _, k in podzial], sr).clamp(-1, 1).float().cpu().numpy()
        czas = time.time() - start
        dl = len(audio) / sr
        sf.write(os.path.join(a.wyjscie, f"{nazwa}.wav"), audio, sr)
        wyniki["pliki"][nazwa] = {"znakow": len(tekst), "kawalkow": len(kawalki), "sekund_audio": round(dl, 2),
                                  "sekund_liczenia": round(czas, 2), "rtf": round(czas / dl, 4),
                                  "x_czasu_rzeczywistego": round(dl / czas, 2)}
        print(f"{nazwa}: {dl / 60:.1f} min mowy w {czas / 60:.1f} min -> RTF {czas / dl:.3f} "
              f"({dl / czas:.1f}× czas rzeczywisty)", flush=True)
        json.dump(wyniki, open(os.path.join(a.wyjscie, "wyniki.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    s_a = sum(v["sekund_audio"] for v in wyniki["pliki"].values())
    s_l = sum(v["sekund_liczenia"] for v in wyniki["pliki"].values())
    wyniki["razem"] = {"sekund_audio": round(s_a, 1), "sekund_liczenia": round(s_l, 1), "rtf": round(s_l / s_a, 4)}
    json.dump(wyniki, open(os.path.join(a.wyjscie, "wyniki.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Razem: {s_a / 60:.1f} min mowy w {s_l / 60:.1f} min, RTF {s_l / s_a:.3f}. Wyniki: {a.wyjscie}/wyniki.json")


if __name__ == "__main__":
    main()
