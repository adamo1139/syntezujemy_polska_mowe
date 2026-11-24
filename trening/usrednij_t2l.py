#!/usr/bin/env python3
"""WSM dla modelu tekst->latenty: średnia wag z kilku punktów kontrolnych tego samego przebiegu (stałe tempo).

Uśrednia vf_estimator, text_encoder, reference_encoder i tokeny CFG (u_text, u_ref, u_key). Działa na zwykłych
punktach (ckpt_step_N.pt: bieżące wagi) i na plikach z ema/ (wagi EMA) — nie mieszać obu rodzajów w jednym
uśrednieniu. Wynik to zwykły punkt kontrolny bez optymalizatora, który czytają syntezuj.py i eksportuj_mikolai.py
(wyniki z EMA zachowują znacznik ema_decay, więc syntezuj.py podpisuje je jako „ema”). Pliki są wczytywane po
jednym (pełny punkt to ok. 1,2 GB), suma liczona w float64.

Uruchamiać z katalogu trening/:
  python usrednij_t2l.py --punkty ../modele/t2l_czyste/ckpt_step_{225..250..5}000.pt --wyjscie ../modele/t2l_czyste/wsm_225_250.pt
  python usrednij_t2l.py --punkty ../modele/t2l_czyste/ema/ckpt_step_{225..250..5}000.pt --wyjscie ../modele/t2l_czyste/wsm_ema_225_250.pt
"""

from __future__ import annotations

import argparse
import os

import torch

MODULY = ("vf_estimator", "text_encoder", "reference_encoder")
TOKENY = ("u_text", "u_ref", "u_key")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--punkty", nargs="+", required=True)
    p.add_argument("--wyjscie", required=True)
    args = p.parse_args()
    for f in args.punkty:
        if not os.path.isfile(f):
            raise SystemExit(f"Nie ma pliku: {f}")

    suma: dict[str, dict] = {}
    rodzaj, kroki, ostatni = None, [], None
    for i, f in enumerate(args.punkty):
        c = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        r = "ema" if "ema_decay" in c else ("pelny" if "optimizer" in c else "wagi")
        if rodzaj is None:
            rodzaj = r
        elif (r == "ema") != (rodzaj == "ema"):
            raise SystemExit(f"{f}: mieszanie punktów EMA i bieżących wag ({r} vs {rodzaj})")
        kroki.append(int(c.get("global_step", -1)))
        czesci = {m: c[m] for m in MODULY}
        czesci["_tokeny"] = {t: c[t] for t in TOKENY if c.get(t) is not None}
        for m, sd in czesci.items():
            dst = suma.setdefault(m, {})
            for k, v in sd.items():
                if v.is_floating_point():
                    dst[k] = dst.get(k, 0) + v.double()
                else:
                    dst[k] = v.clone()   # liczniki / bufory całkowite: z ostatniego punktu
        ostatni = {k: c[k] for k in ("ema_decay", "ema_updates") if k in c}
        print(f"  {f}: krok {kroki[-1]} ({r})")
        del c

    n = len(args.punkty)
    wynik = {m: {k: (v / n).float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in suma[m].items()} for m in MODULY}
    wynik.update({t: (v / n).float() for t, v in suma["_tokeny"].items()})
    wynik.update(global_step=max(kroki), wsm_kroki=kroki, wsm_rodzaj=rodzaj, **ostatni)
    os.makedirs(os.path.dirname(os.path.abspath(args.wyjscie)), exist_ok=True)
    torch.save(wynik, args.wyjscie)
    print(f"Średnia z {n} punktów ({rodzaj}, kroki {kroki[0]}–{kroki[-1]}) -> {args.wyjscie}")


if __name__ == "__main__":
    main()
