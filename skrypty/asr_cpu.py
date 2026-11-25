#!/usr/bin/env python3
"""Porównanie modeli ASR na CPU: szybkość (RTF) i dokładność (CER) na klipach z syntezy, partia 1.

Modele:
  ultra    moondream/parakeet-ultra   (Photon)
  redux    moondream/parakeet-redux   (Photon)
  canary   nvidia/canary-1b-v2        (NeMo; osobne środowisko z nemo_toolkit[asr])

Każdy model liczy te same klipy (losowane deterministycznie z manifestu), po jednym, na --watki wątkach CPU.
CER jak w czysc_asr.py ocen: tekst podany VoxCPM2 i transkrypcja przechodzą przez tę samą normalizację
(liczby słownie, skróty). Wyniki dopisywane do --wyjscie (JSON), więc modele z różnych środowisk trafiają do
jednego pliku; tabela na końcu.

  taskset -c 4 python skrypty/asr_cpu.py wyniki/partia2_proba --modele ultra redux --wyjscie asr_cpu.json
  taskset -c 4 <venv z NeMo>/bin/python skrypty/asr_cpu.py wyniki/partia2_proba --modele canary --wyjscie asr_cpu.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("katalog", help="katalog syntezy (manifest.jsonl + audio/)")
    p.add_argument("--modele", nargs="+", default=["ultra", "redux"], choices=["ultra", "redux", "canary"])
    p.add_argument("--klipy", type=int, default=20)
    p.add_argument("--watki", type=int, default=1)
    p.add_argument("--wyjscie", default="asr_cpu.json")
    a = p.parse_args()
    for zm in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[zm] = str(a.watki)

    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(a.watki)
    sys.path.insert(0, str(KATALOG_PROJEKTU / "skrypty"))
    sys.path.insert(0, str(KATALOG_PROJEKTU / "liczby"))
    from czysc_asr import do_porownania, porownaj, wczytaj_manifest
    from normalizuj import normalizuj

    katalog = Path(a.katalog)
    klipy = sorted(wczytaj_manifest(katalog), key=lambda w: w["id"])
    klipy = random.Random(0).sample(klipy, min(a.klipy, len(klipy)))

    def pcm16k(w):
        x, sr = sf.read(str(katalog / w["audio"]), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return np.ascontiguousarray(np.clip(x, -1.0, 1.0), dtype=np.float32)

    audio = [pcm16k(w) for w in klipy]
    sekundy = sum(len(x) for x in audio) / 16000
    print(f"{len(klipy)} klipów, {sekundy:.1f} s mowy; {a.watki} wątek(i) CPU", flush=True)

    wyniki = json.load(open(a.wyjscie, encoding="utf-8")) if os.path.exists(a.wyjscie) else {}
    for model in a.modele:
        t0 = time.time()
        if model in ("ultra", "redux"):
            import moondream as md
            silnik = md.photon(f"moondream/parakeet-{model}", device="cpu", cpu_threads=a.watki)
            transkrybuj = lambda x: silnik.transcribe(audio=x, sample_rate=16000)["text"]   # noqa: E731
        else:
            from nemo.collections.asr.models import ASRModel
            silnik = ASRModel.from_pretrained("nvidia/canary-1b-v2", map_location="cpu").eval()

            def transkrybuj(x):
                out = silnik.transcribe([x], batch_size=1, source_lang="pl", target_lang="pl", verbose=False)[0]
                return out.text if hasattr(out, "text") else str(out)
        wczytanie = time.time() - t0
        print(f"{model}: wczytany w {wczytanie:.0f} s, rozgrzewka...", flush=True)
        transkrybuj(audio[0])   # rozgrzewka
        t0, hip = time.time(), []
        for i, x in enumerate(audio, 1):
            t1 = time.time()
            hip.append(transkrybuj(x))
            print(f"  {model} {i}/{len(audio)}: {len(x) / 16000:.1f} s mowy w {time.time() - t1:.1f} s", flush=True)
        czas = time.time() - t0
        cer = [porownaj(do_porownania(w["tekst"], normalizuj), do_porownania(h, normalizuj))["cer"] for w, h in zip(klipy, hip)]
        wyniki[model] = {"rtf": round(czas / sekundy, 4), "sekund_liczenia": round(czas, 2), "sekund_mowy": round(sekundy, 2),
                         "cer_sredni": round(float(np.mean(cer)), 4), "cer_mediana": round(float(np.median(cer)), 4),
                         "wczytanie_s": round(wczytanie, 1), "watki": a.watki,
                         "przyklady": [{"tekst": w["tekst"], "asr": h} for w, h in list(zip(klipy, hip))[:5]]}
        print(f"{model:7s}: RTF {czas / sekundy:.3f} ({czas:.1f} s na {sekundy:.0f} s mowy), "
              f"CER średni {np.mean(cer):.3f}, mediana {np.median(cer):.3f}", flush=True)
        json.dump(wyniki, open(a.wyjscie, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        del silnik

    print("\nmodel     RTF     CER śr.  CER med.")
    for m, v in wyniki.items():
        print(f"{m:7s}  {v['rtf']:6.3f}  {v['cer_sredni']:6.3f}  {v['cer_mediana']:6.3f}")


if __name__ == "__main__":
    main()
