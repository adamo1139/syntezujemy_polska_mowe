#!/usr/bin/env python3
"""INT8 na CPU dla MikołAI: kwantyzacja dynamiczna wag (PyTorch) i porównanie z fp32 — szybkość i jakość.

Konwolucje 1×1 (≈95% wag w blokach ConvNeXt) są zamieniane na równoważne warstwy liniowe, bo kwantyzacja dynamiczna
PyTorch obsługuje tylko nn.Linear; potem torch.ao.quantization.quantize_dynamic(qint8): wagi INT8 (per kanał),
aktywacje kwantyzowane w locie. Wynik: nagrania (.ogg), wyniki.json z RTF, UTMOS i CER (Whisper, opcjonalnie).

  taskset -c 4 python skrypty/int8_cpu.py --watki 1 --warianty fp32 vf te+vf wszystko \\
    --wyjscie probki/benchmark/wyniki/int8_1rdzen

Warianty (części w INT8): fp32 (nic), te (enkoder tekstu), vf (pole wektorowe), dek (dekoder), dp (predyktor
długości), łączone plusem (np. te+vf), wszystko = te+vf+dek+dp.
Wymaga biblioteki mikolai; do UTMOS katalogu trening/ (training.utmos), do CER transformers + jiwer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time


def zamien_1x1_na_linear(modul):
    """Każde nn.Conv1d z jądrem 1 (bez grup, bez dylatacji) -> Linear działający na osi kanałów [B, C, T]."""
    import torch.nn as nn

    class Conv1x1(nn.Module):
        def __init__(self, conv: nn.Conv1d):
            super().__init__()
            self.lin = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
            self.lin.weight.data.copy_(conv.weight.data[:, :, 0])
            if conv.bias is not None:
                self.lin.bias.data.copy_(conv.bias.data)

        def forward(self, x):
            return self.lin(x.transpose(1, 2)).transpose(1, 2)

    n = 0
    for nazwa, dziecko in list(modul.named_children()):
        if isinstance(dziecko, nn.Conv1d) and dziecko.kernel_size == (1,) and dziecko.groups == 1 \
                and dziecko.padding in ((0,), "valid") and dziecko.dilation == (1,) and dziecko.stride == (1,):
            setattr(modul, nazwa, Conv1x1(dziecko))
            n += 1
        else:
            n += zamien_1x1_na_linear(dziecko)
    return n


def kwantyzuj(modul):
    import torch
    import torch.nn as nn
    zamien_1x1_na_linear(modul)
    q = torch.ao.quantization.quantize_dynamic(modul, {nn.Linear}, dtype=torch.qint8)
    return q


def rozmiar_mb(modul) -> float:
    import io
    import torch
    b = io.BytesIO()
    torch.save(modul.state_dict(), b)
    return b.tell() / 1e6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="modele/hf_mikolai")
    p.add_argument("--tekst", default="probki/benchmark/przygoda_pokemon.txt")
    p.add_argument("--linie", type=int, default=60, help="ile pierwszych linii tekstu")
    p.add_argument("--kawalki", type=int, default=12, help="ile pierwszych kawałków")
    p.add_argument("--watki", type=int, default=1)
    p.add_argument("--warianty", nargs="+", default=["fp32", "vf", "te+vf", "wszystko"])
    p.add_argument("--wyjscie", required=True)
    p.add_argument("--bez_utmos", action="store_true")
    p.add_argument("--bez_cer", action="store_true")
    a = p.parse_args()

    for zm in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[zm] = str(a.watki)
    import numpy as np
    import torch
    torch.set_num_threads(a.watki)
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from mikolai import MikolAI, podziel, sklej, zapisz_audio

    os.makedirs(a.wyjscie, exist_ok=True)
    tekst = "\n".join(open(a.tekst, encoding="utf-8").read().split("\n")[: a.linie])
    podzial = podziel(tekst)[: a.kawalki]
    kawalki = [k for k, _ in podzial]
    wyniki = {"ustawienia": vars(a), "warianty": {}}

    for wariant in a.warianty:
        tts = MikolAI.from_pretrained(a.repo, device="cpu")
        czesci = {"te", "vf", "dek", "dp"} if wariant == "wszystko" else (set() if wariant == "fp32" else set(wariant.split("+")))
        for c in czesci:
            atr = {"te": "te", "vf": "vf", "dek": "dekoder", "dp": "dp"}[c]
            setattr(tts, atr, kwantyzuj(getattr(tts, atr)))
        mb = {c: round(rozmiar_mb(getattr(tts, c)), 1) for c in ("te", "vf", "dekoder", "dp")}
        g = torch.Generator()
        g.manual_seed(0)
        with torch.no_grad():
            el = tts._przygotuj(kawalki, "poziomka", 1.0, g)
            tts._dekoduj(tts._plyn(el[:1], "poziomka", 4.0, 2))   # rozgrzewka
            t0 = time.time()
            nagrania = []
            for e in el:
                nagrania += tts._dekoduj(tts._plyn([e], "poziomka", 4.0, 16))
            czas = time.time() - t0
        audio = sklej(nagrania, [k for _, k in podzial], tts.sample_rate).clamp(-1, 1).float().numpy()
        dl = len(audio) / tts.sample_rate
        zapisz_audio(audio, os.path.join(a.wyjscie, f"{wariant}.ogg"), tts.sample_rate)
        np.save(os.path.join(a.wyjscie, f"{wariant}_kawalki.npy"), np.array([w.numpy() for w in nagrania], dtype=object), allow_pickle=True)
        wyniki["warianty"][wariant] = {"czesci_int8": sorted(czesci), "rozmiar_mb": mb, "sekund_audio": round(dl, 2),
                                       "sekund_liczenia": round(czas, 2), "rtf": round(czas / dl, 4)}
        print(f"{wariant:10s}: RTF {czas / dl:.3f} ({dl:.0f} s mowy w {czas:.0f} s); MB {mb}", flush=True)
        json.dump(wyniki, open(os.path.join(a.wyjscie, "wyniki.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        del tts

    # jakość: UTMOS i CER na kawałkach (te same kawałki i ziarno w każdym wariancie)
    if not a.bez_utmos:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trening"))
        from training.utmos import utmos, wczytaj_utmos
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = wczytaj_utmos(dev)
        for wariant in a.warianty:
            kaw = np.load(os.path.join(a.wyjscie, f"{wariant}_kawalki.npy"), allow_pickle=True)
            wyniki["warianty"][wariant]["utmos"] = round(float(np.mean([utmos(m, torch.from_numpy(w).to(dev), 48000) for w in kaw])), 3)
        del m
    if not a.bez_cer:
        import jiwer
        import torchaudio
        from transformers import pipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                       torch_dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)

        def norm(t):
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
        for wariant in a.warianty:
            kaw = np.load(os.path.join(a.wyjscie, f"{wariant}_kawalki.npy"), allow_pickle=True)
            hip = [asr({"raw": torchaudio.functional.resample(torch.from_numpy(w), 48000, 16000).numpy(), "sampling_rate": 16000},
                       generate_kwargs={"language": "polish", "task": "transcribe"})["text"] for w in kaw]
            wyniki["warianty"][wariant]["cer"] = round(float(jiwer.cer([norm(t) for t in kawalki], [norm(h) for h in hip])), 4)
    json.dump(wyniki, open(os.path.join(a.wyjscie, "wyniki.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwariant     RTF     UTMOS   CER")
    for w, v in wyniki["warianty"].items():
        print(f"{w:10s} {v['rtf']:6.3f}  {v.get('utmos', float('nan')):6.3f}  {v.get('cer', float('nan')):6.4f}")


if __name__ == "__main__":
    main()
