"""
Słownik znakowy dla polskiego: model czyta surowy tekst (litery, cyfry, interpunkcję), bez fonemów.

Zastępuje słownik IPA z BlueTTS, zachowując te same nazwy funkcji i stałych, z których korzysta reszta kodu.
Wielkie litery są osobnymi znakami (odróżniają skróty: „VAT”, „PIT”). Tekst jest w postaci NFC, więc „ą”
to jeden znak, a nie „a” + ogonek.

VOCAB_SIZE zostaje 256 (rozmiar tablicy osadzeń); nieużywane pozycje są wolne na przyszłe znaki.
Znak spoza słownika zamienia się na PAD_ID — `sprawdz_pokrycie` pozwala wykryć to przed treningiem.
"""

import re
from unicodedata import normalize as uni_normalize

VOCAB_SIZE = 256
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

_ZNAKI = (
    " "
    ".,!?;:-()\"'/%+&°§€"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ąćęłńóśźż"
    "ĄĆĘŁŃÓŚŹŻ"
    # litery obce spotykane w nazwach własnych
    "áàâäãåçčďéèêëěíìîïľĺňñôöõøőřšßťúùûüůűýÿž"
    "ÁÀÂÄÅÇČÉÈÊËÍÎÑÖØŘŠÚÜÝŽ"
    # znaczniki języka (nieużywane przy lang_tags=false, ale wrap_lang_tags ich wymaga)
    "<>"
)
assert len(set(_ZNAKI)) == len(_ZNAKI), "powtórzony znak w słowniku"

# 0..2 zarezerwowane (PAD, BOS, EOS); znaki od 3.
CHAR_TO_ID: dict[str, int] = {znak: i + 3 for i, znak in enumerate(_ZNAKI)}
assert max(CHAR_TO_ID.values()) < VOCAB_SIZE
ID_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_ID.items()}
VOCAB_LIST: list[str] = list(CHAR_TO_ID.keys())

# Różne zapisy tego samego znaku sprowadzone do jednego.
_ZAMIANY = {
    "„": '"', "”": '"', "“": '"', "«": '"', "»": '"',
    "’": "'", "‘": "'", "‚": "'", "´": "'", "`": "'",
    "–": "-", "—": "-", "‑": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", "\t": " ",
}


def text_to_indices(text: str, lang: str = "pl") -> list[int]:
    """Tekst (już po normalize_text) -> identyfikatory znaków. `lang` jest ignorowany."""
    return [CHAR_TO_ID.get(ch, PAD_ID) for ch in text]


def text_to_indices_multilang(text: str, base_lang: str = "pl") -> list[int]:
    return text_to_indices(text, lang=base_lang)


def wrap_lang_tags(ids: list[int], lang: str) -> list[int]:
    """'<lang>' + ids + '</lang>' (używane tylko przy lang_tags=true w konfiguracji)."""
    code = str(lang).strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}", code):
        raise ValueError(f"language tag needs a 2-3 letter language code, got {lang!r}")
    letters = [CHAR_TO_ID[c] for c in code]
    lt, gt = CHAR_TO_ID["<"], CHAR_TO_ID[">"]
    return [lt, *letters, gt] + list(ids) + [lt, CHAR_TO_ID["/"], *letters, gt]


def indices_to_text(indices: list[int]) -> str:
    return "".join(ID_TO_CHAR.get(i, "?") for i in indices)


def normalize_text(text: str, apply_hebrew_fixes: bool = False, affricate_ligatures: bool = False) -> str:
    """Ujednolica zapis (NFC, cudzysłowy, myślniki, odstępy). Nie zmienia liter, wielkości liter ani cyfr.

    Dwa ostatnie argumenty zostały dla zgodności z resztą kodu i są ignorowane.
    """
    text = uni_normalize("NFC", str(text).strip())
    for stary, nowy in _ZAMIANY.items():
        text = text.replace(stary, nowy)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" ([,.!?;:])", r"\1", text)
    if text and not re.search(r"[.!?;:,'\")]$", text):
        text += "."
    return text


def sprawdz_pokrycie(teksty) -> dict[str, int]:
    """Znaki, których po normalizacji nie ma w słowniku, z liczbą wystąpień."""
    brak: dict[str, int] = {}
    for tekst in teksty:
        for znak in normalize_text(tekst):
            if znak not in CHAR_TO_ID:
                brak[znak] = brak.get(znak, 0) + 1
    return brak


print(f"[Vocab] znaki (polski, surowy tekst) | VOCAB_SIZE={VOCAB_SIZE} | używane: {len(CHAR_TO_ID)} | Max Used ID={max(CHAR_TO_ID.values())}")
