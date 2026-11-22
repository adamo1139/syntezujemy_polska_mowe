#!/usr/bin/env python3
"""Ile liczb, dat i matematyki jest naprawdę w zbiorach tekstów (zamiast zgadywać po nazwie filtra).

Dla każdego pliku: .jsonl z wybierz_teksty.py / przygotuj_teksty.py (pole „tekst”) albo .csv trenera
(kolumna „whisper_phonemes”, np. dane/bluetts_train.csv). Liczone na całym tekście, w przeliczeniu na godzinę
szacowanej mowy (14 znaków/s, jak w wybierz_teksty.py), żeby zbiory różnej wielkości dało się porównać.

  python skrypty/policz_liczby.py dane/bluetts_train.csv teksty/liczby_daty.jsonl teksty/liczby_matematyka.jsonl \\
    teksty/liczby_inne.jsonl
"""

from __future__ import annotations

import argparse
import csv
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wybierz_teksty import GOTOWE_WZORCE, ZNAKI_NA_SEKUNDE  # noqa: E402

LICZBA = re.compile(r"\d+(?:[.,]\d+)*")
ROK = re.compile(r"\b(1\d|20)\d\d\b")
DZIALANIE = re.compile(r"\d\s*[+/:]\s*\d|\d\s+-\s+\d|równa się|razy|do kwadratu|do potęgi|pierwiastek z|podzielić przez")
PROCENT = re.compile(r"\d\s*%")
MIARY = {
    "fragmenty z cyfrą": None,
    "liczby / h": LICZBA,
    "cyfry / h": re.compile(r"\d"),
    "lata (1000–2099) / h": ROK,
    "dopasowania @daty / h": re.compile(GOTOWE_WZORCE["@daty"]),
    "działania / h": DZIALANIE,
    "procenty / h": PROCENT,
}
TEMATY_SCISLE = ("matematyce", "zagadkach", "fizyce", "inżynierii", "nauce i technice", "programowaniu")


def teksty(sciezka: Path):
    """(tekst, temat) z pliku .jsonl albo .csv."""
    if sciezka.suffix == ".csv":
        with sciezka.open(encoding="utf-8", newline="") as plik:
            for wiersz in csv.DictReader(plik):
                yield wiersz["whisper_phonemes"], None
    else:
        with sciezka.open(encoding="utf-8") as plik:
            for wiersz in plik:
                if wiersz.strip():
                    f = json.loads(wiersz)
                    yield f["tekst"], f.get("temat")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pliki", nargs="+")
    args = p.parse_args()

    wyniki = []
    for nazwa in args.pliki:
        sciezka = Path(nazwa)
        n = znaki = z_cyfra = 0
        liczniki = collections.Counter()
        scisle = 0
        tematy_znane = False
        for tekst, temat in teksty(sciezka):
            n += 1
            znaki += len(tekst)
            z_cyfra += bool(re.search(r"\d", tekst))
            for miara, wzorzec in MIARY.items():
                if wzorzec is not None:
                    liczniki[miara] += len(wzorzec.findall(tekst))
            if temat is not None:
                tematy_znane = True
                scisle += len(tekst) * any(t in str(temat) for t in TEMATY_SCISLE)
        godziny = znaki / ZNAKI_NA_SEKUNDE / 3600
        wiersz = {"plik": sciezka.name, "fragmenty": n, "h (szac.)": godziny,
                  "fragmenty z cyfrą": 100 * z_cyfra / max(n, 1)}
        for miara in MIARY:
            if MIARY[miara] is not None:
                wiersz[miara] = liczniki[miara] / max(godziny, 1e-9)
        wiersz["tematy ścisłe"] = 100 * scisle / max(znaki, 1) if tematy_znane else None
        wyniki.append(wiersz)

    # miary w wierszach, pliki w kolumnach (czytelne przy kilku plikach)
    nazwy = [w["plik"].removesuffix(".jsonl").removesuffix(".csv") for w in wyniki]
    szer = max(14, *(len(n) + 2 for n in nazwy))
    print(f"{'':<24}" + "".join(f"{n:>{szer}}" for n in nazwy))
    for k in list(wyniki[0])[1:]:
        pola = []
        for w in wyniki:
            v = w[k]
            if v is None:
                pola.append(f"{'—':>{szer}}")
            elif k == "fragmenty":
                pola.append(f"{v:>{szer}d}")
            elif k in ("fragmenty z cyfrą", "tematy ścisłe"):
                pola.append(f"{v:>{szer - 1}.1f}%")
            else:
                pola.append(f"{v:>{szer}.1f}")
        print(f"{k:<24}" + "".join(pola))
    print("\n/h = na godzinę szacowanej mowy; „tematy ścisłe” = udział czasu z tematów: " + ", ".join(TEMATY_SCISLE))


if __name__ == "__main__":
    main()
