"""UTMOS (UTMOS22 strong learner): sieć przewidująca ocenę naturalności mowy przez słuchaczy (MOS, skala ok. 1–5).

Kod modelu skopiowany z tarepan/SpeechMOS v1.2.0 (licencja MIT, LICENSE-SpeechMOS), który przenosi oficjalny
checkpoint sarulab-speech/UTMOS22 (MIT) do czystego PyTorch. Wagi (ok. 411 MB) pobierane raz z wydania SpeechMOS
v1.0.0 jako sam słownik tensorów (weights_only) do modele/utmos/ w katalogu projektu (albo $UTMOS_DIR).

Używane tylko do oceny (nie wpływa na trenowane modele). Uczone na mowie angielskiej, ale artefakty wokodera
(metaliczność, szum, przester) wychwytuje niezależnie od języka; porównywać względne wartości, nie bezwzględne.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import UTMOS22Strong

URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"


def katalog_wag() -> str:
    return os.environ.get("UTMOS_DIR") or str(Path(__file__).resolve().parents[3] / "modele" / "utmos")


def wczytaj_utmos(device) -> UTMOS22Strong:
    model = UTMOS22Strong()
    sd = torch.hub.load_state_dict_from_url(URL, model_dir=katalog_wag(), map_location="cpu", progress=True,
                                            weights_only=True)
    model.load_state_dict(sd)
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def utmos(model: UTMOS22Strong, wav: torch.Tensor, sr: int) -> float:
    """Ocena jednego nagrania [N] albo [1, N] (dowolne sr; model przelicza na 16 kHz)."""
    return float(model(wav.reshape(1, -1).float(), sr)[0])
