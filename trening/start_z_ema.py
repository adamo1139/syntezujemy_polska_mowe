#!/usr/bin/env python3
"""Punkt startowy do dalszego treningu tekst->latenty z wag EMA zamiast bieżących.

Z pełnego punktu kontrolnego (ckpt_step_N.pt z optymalizatorem i stanem EMA) robi nowy pełny punkt, w którym
vf_estimator, text_encoder, reference_encoder i tokeny CFG (u_text, u_ref, u_key) to wagi EMA. Optymalizator
(momenty AdamW), harmonogram, krok i stan EMA zostają, więc trener wznawia od kroku N jak zwykle; EMA dalej
startuje z tych samych wag. Wynik zapisywany jako <katalog>/ckpt_step_N.pt: ten katalog podać jako --out.

Uruchamiać z katalogu trening/:
  python start_z_ema.py --punkt ../modele/t2l_czyste/kamienie/ckpt_step_250000.pt --katalog ../modele/t2l_opus
"""

from __future__ import annotations

import argparse
import os

import torch

from training.t2l.ema import ema_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--punkt", required=True, help="pełny punkt kontrolny z kluczem 'ema'")
    p.add_argument("--katalog", required=True, help="nowy katalog przebiegu (--out treningu)")
    a = p.parse_args()

    c = torch.load(a.punkt, map_location="cpu", weights_only=False)
    if "ema" not in c or "optimizer" not in c:
        raise SystemExit(f"{a.punkt}: to nie jest pełny punkt z EMA i optymalizatorem (klucze: {sorted(c)[:12]})")
    wagi = ema_checkpoint(c)
    for k in ("vf_estimator", "text_encoder", "reference_encoder", "u_text", "u_ref", "u_key"):
        if k in wagi and wagi[k] is not None:
            c[k] = wagi[k]
    c["start_z_ema"] = {"punkt": os.path.abspath(a.punkt), "ema_decay": wagi["ema_decay"], "ema_updates": wagi["ema_updates"]}
    krok = int(c["global_step"])
    os.makedirs(a.katalog, exist_ok=True)
    wyjscie = os.path.join(a.katalog, f"ckpt_step_{krok}.pt")
    if os.path.exists(wyjscie):
        raise SystemExit(f"{wyjscie} już istnieje — nie nadpisuję")
    torch.save(c, wyjscie)
    print(f"Krok {krok}: wagi EMA (decay {wagi['ema_decay']}, {wagi['ema_updates']} aktualizacji) + optymalizator -> {wyjscie}")


if __name__ == "__main__":
    main()
