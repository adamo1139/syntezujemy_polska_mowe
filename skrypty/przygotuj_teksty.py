#!/usr/bin/env python3
"""Z datasetu cpral/polskie-sprawy-v8-jakosc50 robi fragmenty tekstu gotowe do syntezy.

Kroki:
  1. pobranie `sft.jsonl` z HF (do HF_HOME) albo użycie lokalnej kopii (--zrodlo);
  2. odrzucenie rozmów, w których brał udział model `private` (licencja CC-BY-NC);
  3. zostawienie górnych 10% rozmów według `jakosc.ocena` (próg liczony po kroku 2);
  4. czyszczenie markdownu i cięcie na fragmenty 1–3 zdań o długości ok. 2–15 s mowy;
  5. zapis `teksty/fragmenty.jsonl` (jeden fragment na wiersz) i statystyk.

Tekst zostaje surowy (liczby, skróty) — tak go dostaje VoxCPM2, patrz PLAN.md.
Wybór podzbioru na konkretną liczbę godzin: skrypty/wybierz_teksty.py.

Przykłady:
  python skrypty/przygotuj_teksty.py
  python skrypty/przygotuj_teksty.py --zrodlo /dane/sft.jsonl --percentyl 90
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
REPO = "cpral/polskie-sprawy-v8-jakosc50"
PLIK = "sft.jsonl"
ZNAKI_NA_SEKUNDE = 14.0  # tempo VoxCPM2 przyjęte w PLAN.md
ZAKAZANE_MODELE = {"private"}

# ---------------------------------------------------------------- źródło


def znajdz_zrodlo(sciezka: str | None) -> Path:
    if sciezka:
        plik = Path(sciezka).expanduser()
        if not plik.is_file():
            sys.exit(f"Nie ma pliku: {plik}")
        return plik
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Brak huggingface_hub (uv pip install huggingface_hub) — albo podaj --zrodlo PLIK.")
    print(f"Pobieranie {REPO}/{PLIK} (2,2 GB) do pamięci HF...")
    return Path(hf_hub_download(repo_id=REPO, filename=PLIK, repo_type="dataset"))


def modele_rozmowy(meta: dict) -> set[str]:
    modele = {str(meta.get("model"))}
    modele.update(str(m) for m in (meta.get("turn_models") or []))
    return modele


# ---------------------------------------------------------------- czyszczenie

EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002b00-\U00002bff\U0000fe00-\U0000fe0f\U0000200d\U000020e3\U00002190-\U000021ff]"
)
ZNACZNIK_LISTY = re.compile(r"^\s*(?:[-*+•·▪◦–—]\s+|\(?\d{1,2}[.)]\s+|\(?[a-zA-Z]\)\s+)")
NAGLOWEK = re.compile(r"^\s*#{1,6}\s")
CALA_POGRUBIONA = re.compile(r"^\s*(\*\*|__)[^*_]+(\*\*|__)\s*:?\s*$")
LINIA_POZIOMA = re.compile(r"^\s*([-*_=]\s*){3,}$")
# „**Wsparcie emocjonalne**: Pamiętaj, że…” -> sama treść punktu, bez etykiety.
ETYKIETA_PUNKTU = re.compile(r"^\s*(\*\*|__)[^*_]{1,80}?:?(\*\*|__)\s*[:–—-]?\s+(?=\S)")
# Ślady po generowaniu rozmów, np. „Użytkownik pyta: …”.
ARTEFAKT = re.compile(r"\b(Użytkownik|Asystent)\s*(pyta|odpowiada|mówi|pisze)?\s*:", re.IGNORECASE)
ODNOSNIK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
ADRES = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)

# Litery łacińskie (z polskimi i innymi znakami diakrytycznymi), cyfry i interpunkcja, którą da się przeczytać.
DOZWOLONE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ſ0-9 .,;:!?…\-–—()„”\"'’%/+°§&€]+$")
LITERY = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]")

# Skróty, po których kropka nie kończy zdania.
SKROTY = {
    "np", "m.in", "tj", "tzn", "tzw", "itd", "itp", "ok", "dr", "prof", "mgr", "inż", "hab", "ul", "al", "pl",
    "os", "św", "im", "nr", "art", "ust", "pkt", "par", "poz", "zob", "por", "wg", "ww", "ds", "godz", "min",
    "sek", "tys", "mln", "mld", "r", "w", "s", "str", "t", "cz", "red", "wyd", "ang", "łac", "niem", "fr",
    "gr", "wł", "ros", "gen", "płk", "mjr", "kpt", "por", "ppor", "sierż", "ks", "bp", "abp", "o", "p",
    "p.n.e", "n.e", "n.p.m", "m.st", "woj", "pow", "gm", "zm", "ur", "cd", "jw", "br", "ub", "bm", "sp",
    "z o.o", "s.a", "ew", "wew", "zewn", "tel", "kol", "pt", "ps", "vs", "tzw", "dot", "nt", "pn", "wsch",
    "zach", "płn", "płd", "rys", "tab", "ryc", "przyp", "tłum", "oprac", "dn", "mies", "kw", "proc",
}
GRANICA = re.compile(r"([.!?…]+[\"”’)]*)\s+(?=[\"„(–—-]?\s?[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


def oczysc_linie(linia: str) -> str | None:
    """Zwraca linię bez markdownu albo None, jeśli linii nie da się przeczytać (nagłówek, tabela itd.)."""
    linia = linia.replace(" ", " ").replace(" ", " ").replace(" ", " ").replace("\t", " ")
    if not linia.strip():
        return None
    if NAGLOWEK.match(linia) or CALA_POGRUBIONA.match(linia) or LINIA_POZIOMA.match(linia):
        return None
    if linia.lstrip().startswith("|") or linia.count("|") >= 2:
        return None  # tabela
    linia = re.sub(r"^\s*>+\s?", "", linia)  # cytat
    linia = ZNACZNIK_LISTY.sub("", linia)
    linia = ETYKIETA_PUNKTU.sub("", linia)
    linia = ODNOSNIK.sub(r"\1", linia)
    linia = re.sub(r"(\*\*|__)(.+?)\1", r"\2", linia)  # pogrubienie
    linia = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"\1", linia)  # kursywa *...*
    linia = re.sub(r"(?<!\w)_(?!\s)([^_]+?)(?<!\s)_(?!\w)", r"\1", linia)  # kursywa _..._
    linia = re.sub(r"`([^`]*)`", r"\1", linia)  # kod w linii
    linia = EMOJI.sub("", linia)
    linia = linia.replace("“", "„").replace("«", "„").replace("»", "”").replace("‚", ",").replace("‘", "’")
    linia = linia.replace("−", "-").replace("‑", "-").replace("...", "…")
    linia = re.sub(r"\s+", " ", linia).strip()
    linia = re.sub(r"\s+([.,;:!?…])", r"\1", linia)
    if not linia:
        return None
    if linia[-1] in ",;":
        linia = linia[:-1] + "."
    elif linia[-1] not in ".!?…:”\")":
        linia += "."
    return linia


def podziel_na_zdania(linia: str) -> list[str]:
    zdania, poczatek = [], 0
    for dopasowanie in GRANICA.finditer(linia):
        koniec = dopasowanie.end(1)
        znaki = dopasowanie.group(1)
        if znaki.startswith(".") and not znaki.startswith(".."):
            przed = linia[poczatek : dopasowanie.start(1)]
            ostatnie = przed.rsplit(" ", 1)[-1].lstrip("(„\"").lower()
            if ostatnie in SKROTY or (len(ostatnie) == 1 and ostatnie.isalpha()):
                continue  # skrót albo inicjał
            if ostatnie.isdigit() and len(ostatnie) <= 2:
                continue  # liczebnik porządkowy: „3. miejsce”, „12. Dywizja”
        zdania.append(linia[poczatek:koniec].strip())
        poczatek = dopasowanie.end()
    reszta = linia[poczatek:].strip()
    if reszta:
        zdania.append(reszta)
    return zdania


def zdanie_czytelne(zdanie: str) -> bool:
    if not DOZWOLONE.match(zdanie) or ADRES.search(zdanie) or ARTEFAKT.search(zdanie):
        return False
    litery = len(LITERY.findall(zdanie))
    if litery < 2 or litery < 0.5 * len(zdanie):
        return False  # same liczby i znaki
    if zdanie.count("(") != zdanie.count(")") or zdanie.count("„") != zdanie.count("”"):
        return False  # ucięty nawias albo cudzysłów
    return zdanie.count('"') % 2 == 0


def ciagi_zdan(tresc: str) -> list[list[str]]:
    """Tekst wiadomości -> ciągi sąsiadujących czytelnych zdań (przerwa tam, gdzie coś wypadło)."""
    ciagi: list[list[str]] = [[]]
    w_kodzie = False
    for linia in tresc.split("\n"):
        if linia.lstrip().startswith("```") or linia.lstrip().startswith("~~~"):
            w_kodzie = not w_kodzie
            ciagi.append([])
            continue
        if w_kodzie:
            continue
        czysta = oczysc_linie(linia)
        if czysta is None:
            if linia.strip():
                ciagi.append([])  # wypadł nagłówek albo tabela: nie sklejać przez tę dziurę
            continue
        for zdanie in podziel_na_zdania(czysta):
            if zdanie_czytelne(zdanie):
                ciagi[-1].append(zdanie)
            else:
                ciagi.append([])
    return [ciag for ciag in ciagi if ciag]


def potnij(ciag: list[str], los: random.Random, min_znakow: int, max_znakow: int) -> list[tuple[str, int]]:
    """Ciąg zdań -> fragmenty (tekst, liczba zdań): 1–3 zdań, długość w [min_znakow, max_znakow]."""
    fragmenty, i = [], 0
    while i < len(ciag):
        if len(ciag[i]) > max_znakow:
            i += 1  # zdanie za długie na jeden klip; nie tniemy w środku zdania
            continue
        cel = los.choice((1, 1, 2, 2, 3))
        wziete = [ciag[i]]
        dlugosc = len(ciag[i])
        j = i + 1
        while j < len(ciag) and len(wziete) < 3:
            za_krotki = dlugosc < min_znakow
            if len(wziete) >= cel and not za_krotki:
                break
            if dlugosc + 1 + len(ciag[j]) > max_znakow:
                break
            wziete.append(ciag[j])
            dlugosc += 1 + len(ciag[j])
            j += 1
        i = j
        while wziete and wziete[-1].endswith(":"):
            wziete.pop()  # zapowiedź listy na końcu klipu brzmi jak urwane zdanie
        tekst = " ".join(wziete)
        if wziete and min_znakow <= len(tekst) <= max_znakow:
            fragmenty.append((tekst, len(wziete)))
    return fragmenty


def klucz_powtorki(tekst: str) -> str:
    return re.sub(r"[^a-ząćęłńóśźż0-9]+", " ", tekst.lower()).strip()


# ---------------------------------------------------------------- przebieg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zrodlo", help=f"lokalny {PLIK}; bez tej flagi plik jest pobierany z HF ({REPO})")
    p.add_argument("--wyjscie", default=str(KATALOG_PROJEKTU / "teksty" / "fragmenty.jsonl"))
    p.add_argument("--percentyl", type=float, default=90.0, help="próg jakości: zostają rozmowy od tego percentyla w górę")
    p.add_argument("--min-sekund", type=float, default=1.0)
    p.add_argument("--max-sekund", type=float, default=15.0)
    p.add_argument("--ziarno", type=int, default=0)
    args = p.parse_args()

    zrodlo = znajdz_zrodlo(args.zrodlo)
    min_znakow = max(1, round(args.min_sekund * ZNAKI_NA_SEKUNDE))
    max_znakow = round(args.max_sekund * ZNAKI_NA_SEKUNDE)
    t0 = time.time()

    # Przebieg 1: licencja i oceny; zapamiętujemy tylko położenie wiersza w pliku.
    wszystkie = 0
    odrzucone_modele: collections.Counter = collections.Counter()
    kandydaci: list[tuple[float, int]] = []  # (ocena, położenie)
    with zrodlo.open("rb") as plik:
        while True:
            polozenie = plik.tell()
            wiersz = plik.readline()
            if not wiersz:
                break
            wszystkie += 1
            rozmowa = json.loads(wiersz)
            zakazane = modele_rozmowy(rozmowa["meta"]) & ZAKAZANE_MODELE
            if zakazane:
                odrzucone_modele.update(zakazane)
                continue
            kandydaci.append((float(rozmowa["jakosc"]["ocena"]), polozenie))
    oceny = sorted(ocena for ocena, _ in kandydaci)
    prog = oceny[min(len(oceny) - 1, int(len(oceny) * args.percentyl / 100.0))]
    wybrane = sorted(polozenie for ocena, polozenie in kandydaci if ocena >= prog)
    print(f"Rozmów w pliku: {wszystkie}")
    print(f"Odrzucone przez model: {sum(odrzucone_modele.values())} {dict(odrzucone_modele)}")
    print(f"Zostało: {len(kandydaci)}; próg jakości (percentyl {args.percentyl:g}): {prog}")
    print(f"Rozmów od progu w górę: {len(wybrane)} ({100 * len(wybrane) / len(kandydaci):.1f}%)  [{time.time() - t0:.0f} s]")

    # Przebieg 2: czyszczenie i cięcie tylko wybranych rozmów.
    wyjscie = Path(args.wyjscie)
    wyjscie.parent.mkdir(parents=True, exist_ok=True)
    widziane: set[str] = set()
    widziane_id: set[str] = set()
    licznik: collections.Counter = collections.Counter()
    znaki = collections.Counter()
    dlugosci: list[int] = []
    tematy: collections.Counter = collections.Counter()
    with zrodlo.open("rb") as plik, wyjscie.open("w", encoding="utf-8") as wynik:
        for polozenie in wybrane:
            plik.seek(polozenie)
            rozmowa = json.loads(plik.readline())
            assert not (modele_rozmowy(rozmowa["meta"]) & ZAKAZANE_MODELE)
            zrodlo_id = str(rozmowa["id"])
            meta = rozmowa["meta"]
            ziarno = int(hashlib.sha1(f"{args.ziarno}:{zrodlo_id}".encode()).hexdigest()[:12], 16)
            los = random.Random(ziarno)
            tura = 0
            dal_fragment = False
            for wiadomosc in rozmowa["messages"]:
                rola = wiadomosc["role"]
                if rola == "user":
                    tura += 1
                if rola not in ("user", "assistant"):
                    continue
                ciagi = ciagi_zdan(wiadomosc["content"] or "")
                if rola == "user":
                    # Pytanie użytkownika tylko w całości: jeden ciąg, nic nie wypadło, mieści się w klipie.
                    oryginalne = [l for l in (wiadomosc["content"] or "").split("\n") if l.strip()]
                    caly = " ".join(ciagi[0]) if len(ciagi) == 1 and len(oryginalne) == 1 else ""
                    kawalki = [(caly, len(ciagi[0]))] if caly and min_znakow <= len(caly) <= max_znakow else []
                    if caly and not kawalki:
                        licznik["pytania poza zakresem długości"] += 1
                else:
                    kawalki = [k for ciag in ciagi for k in potnij(ciag, los, min_znakow, max_znakow)]
                for nr, (tekst, ile_zdan) in enumerate(kawalki):
                    klucz = klucz_powtorki(tekst)
                    if klucz in widziane:
                        licznik["powtórki"] += 1
                        continue
                    widziane.add(klucz)
                    identyfikator = f"{zrodlo_id[:16]}_t{tura}{rola[0]}{nr:02d}"
                    if identyfikator in widziane_id:
                        sys.exit(f"Kolizja id fragmentu: {identyfikator}")
                    widziane_id.add(identyfikator)
                    wynik.write(
                        json.dumps(
                            {
                                "id": identyfikator,
                                "tekst": tekst,
                                "zrodlo_id": zrodlo_id,
                                "rola": rola,
                                "tura": tura,
                                "zdania": ile_zdan,
                                "znaki": len(tekst),
                                "pytanie": tekst.rstrip("”\")").endswith("?"),
                                "temat": meta.get("topic"),
                                "jakosc": rozmowa["jakosc"]["ocena"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    licznik[f"fragmenty {rola}"] += 1
                    znaki[rola] += len(tekst)
                    dlugosci.append(len(tekst))
                    dal_fragment = True
            if dal_fragment:
                licznik["rozmowy z fragmentami"] += 1
                tematy[meta.get("topic")] += 1

    razem_znakow = sum(znaki.values())
    print(f"\nZapisano {len(dlugosci)} fragmentów do {wyjscie}  [{time.time() - t0:.0f} s]")
    for nazwa, ile in sorted(licznik.items()):
        print(f"  {nazwa}: {ile}")
    print(f"  tematów: {len(tematy)}")
    print(
        f"  znaków: {razem_znakow} = ok. {razem_znakow / ZNAKI_NA_SEKUNDE / 3600:.0f} h mowy przy {ZNAKI_NA_SEKUNDE:g} znakach/s "
        f"(user {znaki['user'] / ZNAKI_NA_SEKUNDE / 3600:.0f} h, assistant {znaki['assistant'] / ZNAKI_NA_SEKUNDE / 3600:.0f} h)"
    )
    if dlugosci:
        dlugosci.sort()
        sek = lambda ulamek: dlugosci[min(len(dlugosci) - 1, int(len(dlugosci) * ulamek))] / ZNAKI_NA_SEKUNDE
        print(f"  długość klipu (szacowana): p10 {sek(0.1):.1f} s, mediana {sek(0.5):.1f} s, p90 {sek(0.9):.1f} s")
        przedzialy = [(0, 2), (2, 5), (5, 10), (10, 15.01)]
        opis = ", ".join(
            f"{a:g}–{b:.0f} s: {100 * sum(a <= d / ZNAKI_NA_SEKUNDE < b for d in dlugosci) / len(dlugosci):.0f}%" for a, b in przedzialy
        )
        print(f"  rozkład: {opis}")


if __name__ == "__main__":
    main()
