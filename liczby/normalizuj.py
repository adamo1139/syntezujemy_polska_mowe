#!/usr/bin/env python3
"""Zamiana liczb w polskim tekście na słowa z odmianą, dla nauczyciela (VoxCPM2) przy ponownej syntezie.

Reguły, nie LLM: przypadek bierzemy z przyimka przed liczbą i z formy rzeczownika po niej (Morfeusz 2),
formy liczebników też z Morfeusza. Obsługiwane: liczby całkowite (też „20 000”), ułamki dziesiętne, ułamki
zwykłe, procenty i jednostki (zł, km, kg, °C, GB…), tys./mln/mld, lata („w 2010 roku”, „od 2021”), daty
(„15 września”, „15.09.2024”), godziny („o 9:30”, „9:00-17:00”), zakresy („5-10 minut”), złożenia („15-minutowa”),
porządkowe („3. miejsce”, „lat 80.”, „21 wiek”), działania (+, =, x, ^2).

Każdy tekst dostaje listę flag; flaga oznacza, że reguła zgadywała (np. rzeczownik nie pasuje do żadnego przypadku)
— takie teksty warto obejrzeć albo dać LLM-owi.

  python liczby/normalizuj.py --test "W 2010 roku kupiłem 3 kg jabłek za 20 tys. zł."
  python liczby/normalizuj.py --wejscie liczby/teksty_hf.parquet --wyjscie liczby/teksty_slowne.jsonl
  python liczby/normalizuj.py --wejscie teksty/val_liczby.jsonl --wyjscie teksty/val_liczby_slownie.jsonl
(plik z polem „tekst”, np. z wybierz_teksty.py, daje od razu wejście dla generuj.py: tekst słowny + tekst_cyfry)
Wymaga: pip install morfeusz2
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys

import morfeusz2

M = morfeusz2.Morfeusz()

# ---------------------------------------------------------------- Morfeusz

def _tag(tag: str) -> dict:
    p = tag.split(":")
    d = {"pos": p[0], "num": None, "case": set(), "gen": set(), "rest": p[4:]}
    if len(p) > 1 and p[1] in ("sg", "pl"):
        d["num"] = p[1]
    if len(p) > 2:
        d["case"] = set(p[2].split("."))
    if len(p) > 3:
        g = set()
        for x in p[3].split("."):
            g |= {"n"} if x.startswith("n") else {"pl"} if x.startswith("p") else {x}
        d["gen"] = g
    return d


@functools.lru_cache(None)
def gen_forms(lemma: str):
    return [(f, _tag(t)) for f, _, t, *_ in M.generate(lemma)]


@functools.lru_cache(None)
def analiza(word: str):
    out = []
    for _, _, (f, lem, t, *_) in M.analyse(word):
        out.append((lem.split(":")[0], _tag(t)))
    return out


GORSZE = {"dwu", "obu", "trzema", "czterema", "ośm", "ośmią", "ośmią"}  # archaiczne/rzadsze warianty


def num_form(lemma: str, case: str, gender: str) -> str:
    if case == "voc":
        case = "nom"
    cands = []
    for f, t in gen_forms(lemma):
        if t["pos"] != "num" or "col" in t["rest"]:
            continue
        if case in t["case"] and (gender in t["gen"] or not t["gen"]):
            cands.append((f, t))
    if not cands:
        return lemma
    if case in ("nom", "acc") and gender == "m1":  # „dwóch/pięciu” zamiast „dwaj”
        rec = [c for c in cands if "rec" in c[1]["rest"]]
        cands = rec or cands
    cands.sort(key=lambda c: (c[0] in GORSZE, case in ("nom", "acc") and c[0] != lemma, len(c[0])))
    return cands[0][0]


def noun_form(lemma: str, num: str, case: str) -> str | None:
    if case == "voc":
        case = "nom"
    for f, t in gen_forms(lemma):
        if t["pos"] == "subst" and t["num"] == num and case in t["case"]:
            return f
    return None


# ---------------------------------------------------------------- liczebniki główne

JEDNOSTKI = ["zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć", "dziesięć",
             "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście", "szesnaście", "siedemnaście",
             "osiemnaście", "dziewiętnaście"]
DZIESIATKI = [None, None, "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt", "sześćdziesiąt",
              "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt"]
SETKI = [None, "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset", "siedemset", "osiemset",
         "dziewięćset"]
JEDEN = {  # przypadek -> (m1/m2, m3, f, n)
    "nom": ("jeden", "jeden", "jedna", "jedno"), "gen": ("jednego", "jednego", "jednej", "jednego"),
    "dat": ("jednemu", "jednemu", "jednej", "jednemu"), "acc": ("jednego", "jeden", "jedną", "jedno"),
    "inst": ("jednym", "jednym", "jedną", "jednym"), "loc": ("jednym", "jednym", "jednej", "jednym"),
}
SKALE = [(10**9, "miliard"), (10**6, "milion"), (1000, "tysiąc")]


def jeden(case: str, gender: str) -> str:
    row = JEDEN["nom" if case == "voc" else case]
    return row[{"m1": 0, "m2": 0, "m3": 1, "f": 2, "n": 3}.get(gender, 1)]


def rzeczownik_po(v: int, case: str) -> tuple[str, str]:
    """Liczba i przypadek rzeczownika po liczebniku v w przypadku case (np. 5 w mianowniku -> pl, gen)."""
    if v == 1:
        return "sg", case
    if case in ("nom", "acc", "voc"):
        if v >= 1000 and v % 1000 == 0:
            return "pl", "gen"
        if v % 10 in (2, 3, 4) and v % 100 not in (12, 13, 14):
            return "pl", case
        return "pl", "gen"
    return "pl", case


def _ponizej_tysiaca(v: int, case: str, gender: str, w_zlozeniu: bool) -> list[str]:
    if v == 1 and not w_zlozeniu:
        return [jeden(case, gender)]
    out = []
    h, r = divmod(v, 100)
    if h:
        out.append(num_form(SETKI[h], case, gender))
    if r:
        if r < 20:
            out.append("jeden" if r == 1 else num_form(JEDNOSTKI[r], case, gender))
        else:
            t, u = divmod(r, 10)
            out.append(num_form(DZIESIATKI[t], case, gender))
            if u:
                out.append("jeden" if u == 1 else num_form(JEDNOSTKI[u], case, gender))
    return out


def glowny(v: int, case: str = "nom", gender: str = "m3") -> str:
    if v == 0:
        return noun_form("zero", "sg", case) or "zero" if case not in ("nom", "acc", "voc") else "zero"
    if v >= 10**12:
        return cyfry(str(v))
    out = []
    for skala, lemat in SKALE:
        k, v = divmod(v, skala)
        if not k:
            continue
        if k == 1:
            out.append(noun_form(lemat, "sg", case) or lemat)
        else:
            out += _ponizej_tysiaca(k, case, "m3", w_zlozeniu=True)
            out.append(noun_form(lemat, *rzeczownik_po(k, case)) or lemat)
    if v:
        out += _ponizej_tysiaca(v, case, gender, w_zlozeniu=bool(out))
    return " ".join(out)


def cyfry(s: str) -> str:
    """Ciąg z zerem na początku („05”, „007”) albo bardzo długi: cyfra po cyfrze / trójkami."""
    s = re.sub(r"\D", "", s)
    if len(s) > 1 and s[0] == "0":
        lead = len(s) - len(s.lstrip("0"))
        reszta = s[lead:]
        return " ".join(["zero"] * lead + ([cyfry(reszta)] if reszta else []))
    if len(s) >= 7:
        grupy = [s[max(0, i - 3):i] for i in range(len(s), 0, -3)][::-1]
        return " ".join(cyfry(g) if g[0] == "0" else glowny(int(g)) for g in grupy)
    return glowny(int(s))


# ---------------------------------------------------------------- liczebniki porządkowe

PORZ = ["zerowy", "pierwszy", "drugi", "trzeci", "czwarty", "piąty", "szósty", "siódmy", "ósmy", "dziewiąty",
        "dziesiąty", "jedenasty", "dwunasty", "trzynasty", "czternasty", "piętnasty", "szesnasty", "siedemnasty",
        "osiemnasty", "dziewiętnasty"]
PORZ_DZ = [None, None, "dwudziesty", "trzydziesty", "czterdziesty", "pięćdziesiąty", "sześćdziesiąty",
           "siedemdziesiąty", "osiemdziesiąty", "dziewięćdziesiąty"]
PORZ_SETKI = [None, "setny", "dwusetny", "trzechsetny", "czterechsetny", "pięćsetny", "sześćsetny", "siedemsetny",
              "osiemsetny", "dziewięćsetny"]
PORZ_TYS = {1: "tysięczny", 2: "dwutysięczny", 3: "trzytysięczny"}

# końcówki: M(nom m) A E EGO EMU YM Ą EJ YCH YMI
KONC = {"y": ["y", "a", "e", "ego", "emu", "ym", "ą", "ej", "ych", "ymi"],
        "gi": ["i", "a", "ie", "iego", "iemu", "im", "ą", "iej", "ich", "imi"],
        "ci": ["", "a", "e", "ego", "emu", "m", "ą", "ej", "ch", "mi"]}
SLOT = {"M": 0, "A": 1, "E": 2, "EGO": 3, "EMU": 4, "YM": 5, "Ą": 6, "EJ": 7, "YCH": 8, "YMI": 9}


def przym(lemat: str, case: str, num: str, gender: str) -> str:
    if lemat.endswith("gi"):
        stem, k = lemat[:-1], KONC["gi"]
    elif lemat.endswith("ci"):
        stem, k = lemat, KONC["ci"]
    else:
        stem, k = lemat[:-1], KONC["y"]
    case = "nom" if case == "voc" else case
    f, n, m1 = gender == "f", gender == "n", gender == "m1"
    if num == "sg":
        s = {"nom": "A" if f else "E" if n else "M",
             "gen": "EJ" if f else "EGO", "dat": "EJ" if f else "EMU",
             "acc": "Ą" if f else "E" if n else "EGO" if gender in ("m1", "m2") else "M",
             "inst": "Ą" if f else "YM", "loc": "EJ" if f else "YM"}[case]
    else:
        s = {"nom": "E", "gen": "YCH", "dat": "YM", "acc": "YCH" if m1 else "E", "inst": "YMI",
             "loc": "YCH"}[case]
    return stem + k[SLOT[s]]


def porzadkowy(v: int, case: str = "nom", num: str = "sg", gender: str = "m3") -> str:
    tys, r = divmod(v, 1000)
    if r == 0 and tys in PORZ_TYS:
        return przym(PORZ_TYS[tys], case, num, gender)
    out = [glowny(tys * 1000)] if tys else []
    h, r2 = divmod(r, 100)
    if r2 == 0 and h:
        return " ".join(out + [przym(PORZ_SETKI[h], case, num, gender)])
    if h:
        out.append(SETKI[h])
    if r2 < 20:
        out.append(przym(PORZ[r2], case, num, gender))
    else:
        t, u = divmod(r2, 10)
        out.append(przym(PORZ_DZ[t], case, num, gender))
        if u:
            out.append(przym(PORZ[u], case, num, gender))
    return " ".join(out)


# ---------------------------------------------------------------- formy złożone („pięcio-”, „dwudziesto-”)

def zlozenie(v: int) -> str:
    """„15” w „15-minutowy” -> „piętnasto”."""
    specj = {1: "jedno", 2: "dwu", 3: "trzy", 4: "cztero", 100: "stu", 1000: "tysiąc"}
    if v in specj:
        return specj[v]
    out = []
    for f, t in gen_forms(glowny(v).split()[-1] if v < 20 or v % 10 == 0 else "x"):
        if t["pos"] == "numcomp":
            out.append(f)
    if out and v < 100:
        return out[0]
    # wielowyrazowe: każda część w formie złożonej („dwudziestopięcio”)
    czesci = []
    h, r = divmod(v % 1000, 100)
    if v >= 1000:
        return glowny(v).replace(" ", "")  # rzadkie, byle bez cyfr
    if h:
        czesci.append(zlozenie(h * 100) if h * 100 in specj else SETKI[h])
    if r:
        if r < 20:
            czesci.append(zlozenie(r))
        else:
            t, u = divmod(r, 10)
            czesci.append(zlozenie(t * 10))
            if u:
                czesci.append(zlozenie(u))
    return "".join(czesci)


# ---------------------------------------------------------------- kontekst

PRZYIMKI = {
    **{p: ["gen"] for p in "od do bez dla u około ok. koło oprócz podczas według wśród spośród zamiast sprzed "
                          "powyżej poniżej wokół obok spod znad zza naprzeciw spoza wobec".split()},
    "z": ["gen", "inst"], "ze": ["gen", "inst"], "ciągu": ["gen"], "trakcie": ["gen"],
    **{p: ["dat"] for p in "dzięki ku przeciw wbrew".split()},
    "przy": ["loc"], "o": ["acc", "loc"], "po": ["loc", "acc"], "w": ["loc", "acc"], "we": ["loc", "acc"],
    "na": ["acc", "loc"], "przez": ["acc"], "ponad": ["acc"], "za": ["acc", "inst"], "przed": ["inst"],
    "nad": ["inst", "acc"], "pod": ["inst", "acc"], "między": ["inst"], "pomiędzy": ["inst"],
    "niż": ["nom"], "jak": ["nom"],
}
PRZEZROCZYSTE = set("blisko nawet aż tylko prawie niemal zaledwie maksymalnie minimum maksimum mniej więcej "
                    "łącznie razem już jeszcze nawet dokładnie średnio najwyżej najmniej przynajmniej".split())
WSZYSTKIE = ["nom", "acc", "gen", "inst", "dat", "loc"]
SLOWO = r"[^\W\d_]+\.?"
MIESIACE = {m: i + 1 for i, m in enumerate("stycznia lutego marca kwietnia maja czerwca lipca sierpnia września "
                                           "października listopada grudnia".split())}
MIES_WSZ = set(MIESIACE) | set("styczeń styczniu luty lutym marzec marcu kwiecień kwietniu maj maju czerwiec "
                               "czerwcu lipiec lipcu sierpień sierpniu wrzesień wrześniu październik październiku "
                               "listopad listopadzie grudzień grudniu".split())


def slowo_przed(text: str, start: int) -> tuple[str | None, str | None]:
    """(przyimek wyznaczający przypadek, ostatnie słowo) przed pozycją; pomija „nawet”, „aż” itp."""
    przed = text[:start]
    slowa = []
    pos = len(przed)
    while len(slowa) < 3:
        m = re.search(r"(?:^|(?<=[\s(„\"]))(" + SLOWO + r")\s+$", przed[:pos])
        if not m:
            break
        slowa.append(m.group(1).lower())
        pos = m.start(1)
    if not slowa:
        return None, None
    ostatnie = slowa[0]
    for s in slowa:
        if s in PRZYIMKI:
            return s, ostatnie
        if s not in PRZEZROCZYSTE:
            break
    return None, ostatnie


def slowo_po(text: str, end: int) -> str | None:
    m = re.match(r"\s*(" + r"[^\W\d_]+" + r")", text[end:])
    return m.group(1) if m else None


NIE_RZECZOWNIK = {"równa", "plus", "minus", "razy_", "około", "ok"}
ORD_RZECZ = set("rok wiek miejsce piętro klasa dzień tydzień miesiąc sezon odcinek część edycja rocznica stulecie "
                "tysiąclecie raz etap krok poziom kwartał semestr liga runda kolejka tura wydanie stopień urodziny "
                "lekcja dekada kadencja zmiana pokolenie generacja wersja".split())


def analizy_rzeczownika(slowo: str | None):
    if not slowo or slowo.lower() in NIE_RZECZOWNIK:
        return []
    an = analiza(slowo)
    if any(t["pos"] in ("prep", "conj", "comp") for _, t in an):
        return []
    out = []
    for lem, t in an:
        if t["pos"] in ("subst", "depr", "ger", "adj", "ppas", "pact") and t["num"]:
            out.append((lem, t))
    return out


def analizy_po(text: str, end: int):
    """Analizy pierwszego rzeczownika po pozycji, przeskakując do dwóch przymiotników („6 ostatnich lat”)."""
    slowa = re.findall(r"[^\W\d_]+", text[end:end + 80].split(".")[0].split(",")[0])[:3]
    if not re.match(r"\s*[^\W\d_]", text[end:]):
        return []
    pierwsze = None
    for w in slowa:
        an = analizy_rzeczownika(w)
        if not an:
            break
        pierwsze = pierwsze or an
        subst = [x for x in an if x[1]["pos"] in ("subst", "depr", "ger")]
        if subst and not any(t["pos"] == "adj" for _, t in an):
            return subst
        if not any(t["pos"] in ("adj", "ppas", "pact") for _, t in an):
            break
    return pierwsze or []


KOLEJNOSC_RODZ = {"m3": 0, "f": 1, "n": 2, "m2": 3, "m1": 4}


def rodzaj(g) -> str:
    for x in ("m3", "f", "n", "m2", "m1"):
        if x in g:
            return x
    return "m3"


def ustal_przypadek(prep, v, analizy, decimal=False):
    """Zwraca (przypadek, rodzaj, czy_pewne, lemat_rzeczownika, liczba_rzeczownika)."""
    kandydaci = PRZYIMKI.get(prep, ["nom", "acc"]) if prep else ["nom", "acc"]
    if not analizy:
        return kandydaci[0], "m3", True, None, None
    for zbior in (kandydaci, [] if prep else WSZYSTKIE):
        for c in zbior:
            num, nc = ("sg", "gen") if decimal else rzeczownik_po(v, c)
            for g, lem, t in sorted(((g, l, t) for l, t in analizy for g in (t["gen"] or {"m3"})),
                                    key=lambda x: KOLEJNOSC_RODZ.get(x[0], 9)):
                if True:
                    ok = t["num"] == num and nc in t["case"]
                    if not ok and g == "m1" and c == "nom" and t["num"] == "pl" and "nom" in t["case"]:
                        ok = True  # „dwaj panowie”
                    if ok:
                        return c, g, True, lem, t["num"]
    g = rodzaj(analizy[0][1]["gen"])
    return kandydaci[0], g, False, analizy[0][0], analizy[0][1]["num"]


# ---------------------------------------------------------------- jednostki

JEDN = {  # skrót -> (lemat rzeczownika, dopisek)
    "zł": ("złoty", ""), "gr": ("grosz", ""), "€": ("euro", ""), "$": ("dolar", ""), "%": ("procent", ""),
    "km/h": ("kilometr", " na godzinę"), "m/s": ("metr", " na sekundę"), "km": ("kilometr", ""),
    "cm": ("centymetr", ""), "mm": ("milimetr", ""), "m": ("metr", ""), "kg": ("kilogram", ""),
    "mg": ("miligram", ""), "dag": ("dekagram", ""), "g": ("gram", ""), "t": ("tona", ""), "ml": ("mililitr", ""),
    "l": ("litr", ""), "h": ("godzina", ""), "godz.": ("godzina", ""), "min": ("minuta", ""), "min.": ("minuta", ""),
    "s": ("sekunda", ""), "sek.": ("sekunda", ""), "ms": ("milisekunda", ""), "kB": ("kilobajt", ""),
    "KB": ("kilobajt", ""), "MB": ("megabajt", ""), "GB": ("gigabajt", ""), "TB": ("terabajt", ""),
    "Hz": ("herc", ""), "kHz": ("kiloherc", ""), "MHz": ("megaherc", ""), "GHz": ("gigaherc", ""),
    "W": ("wat", ""), "kW": ("kilowat", ""), "kWh": ("kilowatogodzina", ""), "MW": ("megawat", ""),
    "V": ("wolt", ""), "kV": ("kilowolt", ""), "mA": ("miliamper", ""), "mAh": ("miliamperogodzina", ""),
    " A": ("amper", ""), "°C": ("stopień", " Celsjusza"), "°F": ("stopień", " Fahrenheita"), "°": ("stopień", ""),
    "tys.": ("tysiąc", ""), "tys": ("tysiąc", ""), "mln": ("milion", ""), "mld": ("miliard", ""),
    "r.": ("rok", ""), "zł.": ("złoty", ""), "szt.": ("sztuka", ""), "os.": ("osoba", ""),
    "m²": ("metr", " kwadratowy"), "m2": ("metr", " kwadratowy"), "m³": ("metr", " sześcienny"),
    "m3": ("metr", " sześcienny"), "km²": ("kilometr", " kwadratowy"), "km2": ("kilometr", " kwadratowy"),
    "cm²": ("centymetr", " kwadratowy"), "cm2": ("centymetr", " kwadratowy"), "cm³": ("centymetr", " sześcienny"),
    "cm3": ("centymetr", " sześcienny"), "kcal": ("kilokaloria", ""), "mph": ("mila", " na godzinę"),
}
MNOZNIKI = {"tysiąc", "milion", "miliard"}
_jedn_alt = "|".join(re.escape(k) for k in sorted(JEDN, key=len, reverse=True))
RE_JEDN = re.compile(r"\s?(" + _jedn_alt + r")(?![^\W_])")
JEDN["A"] = JEDN.pop(" A")  # „5 A” tylko ze spacją („4A” to zwykle model)


def forma_jednostki(lemat: str, dopisek: str, num: str, case: str) -> str:
    if lemat == "procent":
        return "procent"
    if lemat == "euro":
        return "euro"
    f = noun_form(lemat, num, case) or lemat
    if dopisek.strip() in ("kwadratowy", "sześcienny"):
        f += " " + przym(dopisek.strip(), case, num, "m3")
        return f
    return f + dopisek


# ---------------------------------------------------------------- główna zamiana

LICZBA = r"\d{1,3}(?:[  ]\d{3})+(?!\d)|\d+(?:[.,]\d+)?"
RE = re.compile(
    r"(?P<data>\b(?P<dd>\d{1,2})\.(?P<mm>\d{1,2})\.(?P<rrrr>\d{4})\b)"
    r"|(?P<godzz>\b(?P<h1>\d{1,2}):(?P<m1>\d{2})\s*[-–]\s*(?P<h2>\d{1,2}):(?P<m2>\d{2})\b)"
    r"|(?P<godz>\b(?P<h>\d{1,2}):(?P<m>\d{2})\b)"
    r"|(?P<stos>\b(?P<sa>\d+):(?P<sb>\d+)\b)"
    r"|(?P<zakres>(?<![\d.,])(?P<za>" + LICZBA + r")(?P<zsep>\s*[-–]\s*)(?P<zb>" + LICZBA + r")(?![\d]|-[^\W\d_]))"
    r"|(?P<zloz>(?<![\d.,])(?P<zl>\d+)-(?P<zs>[^\W\d_]+))"
    r"|(?P<ulam>(?<![\d.,/])(?P<ua>\d+)/(?P<ub>\d+)(?![\d/]))"
    r"|(?P<porz>(?<![\d.,])(?P<pl>\d+)\.(?=\s+[a-ząćęłńóśźż]))"
    r"|(?P<rzym>\b(?:[IVX]{2,}|I(?=\s+wojn))\b(?:\s?w\.)?)"
    r"|(?P<liczba>(?<![\d])(?P<minus>-(?=\d))?(?P<l>" + LICZBA + r"))"
)
SUF_PORZ = re.compile(r"^(?:y|i|a|e|ego|emu|ym|ą|ej|ych|ymi|ty|ta|te|ci|cia|cie|gi|ga|gie|go|mu|ma|me|ąt)$")


class Normalizator:
    def __init__(self, text: str):
        self.text = text
        self.flagi: list[str] = []
        self.math = bool(re.search(r"=|równa się|\d\s*[+*×^]\s*\d", text))

    # wartość liczby z zapisu („20 000”, „1,5”, „2.000 zł”)
    @staticmethod
    def wartosc(s: str):
        s = s.replace(" ", "").replace(" ", "")
        if re.fullmatch(r"\d+", s):
            return int(s), None
        a, b = re.split(r"[.,]", s)
        return int(a), b

    def czytaj(self, s: str, case="nom", gender="m3") -> str:
        """Liczba w zapisie s w przypadku/rodzaju; ułamki dziesiętne zawsze w mianowniku."""
        s2 = s.replace(" ", "").replace(" ", "")
        if re.fullmatch(r"\d+", s2):
            if (len(s2) > 1 and s2[0] == "0") or len(s2) >= 10:
                return cyfry(s2)
            return glowny(int(s2), case, gender)
        a, b = re.split(r"[.,]", s2)
        if b == "5" and a != "0":
            if a == "1":
                return "półtorej" if gender == "f" else "półtora"
            return f"{glowny(int(a), 'nom', gender)} i pół"
        return f"{glowny(int(a), 'nom', 'm3')} przecinek {cyfry(b)}"

    def jednostka(self, end: int):
        m = RE_JEDN.match(self.text, end)
        if not m:
            return None, end
        return m.group(1).strip(), m.end()

    def zamien(self) -> str:
        t = self.text
        # działania i znaki przy cyfrach
        t = re.sub(r"([\w)])\s*\^\s*2\b", r"\1 do kwadratu", t)
        t = re.sub(r"([\w)])\s*\^\s*3\b", r"\1 do sześcianu", t)
        t = re.sub(r"([\w)])\s*\^\s*(\d+)", lambda m: f"{m.group(1)} do potęgi "
                   f"{porzadkowy(int(m.group(2)), 'gen', 'sg', 'f')}", t)
        t = re.sub(r"(\d)[x×](\d)", r"\1 na \2", t)
        t = re.sub(r"(\d)\s+[x×*]\s+(\d)", r"\1 razy \2", t)
        if not self.math:
            t = re.sub(r"(\d)\s*[x×](?=\s|$)", r"\1 razy", t)
        t = re.sub(r"\s*\+\s*", " plus ", t) if re.search(r"\d\s*\+|\+\s*\d", t) else t
        t = re.sub(r"\s*=\s*", " równa się ", t)
        t = re.sub(r"√\s*", "pierwiastek z ", t)
        t = t.replace("24/7", "dwadzieścia cztery na siedem")
        def _godz(m):
            p = (m.group(1) or "").strip().lower()
            c = {"o": "loc", "po": "loc", "od": "gen", "do": "gen", "około": "gen", "przed": "inst",
                 "między": "inst", "za": "acc", "na": "acc"}.get(p, "nom")
            return (m.group(1) or "") + (noun_form("godzina", "sg", c) or "godzina")
        t = re.sub(r"(\b[^\W\d_]+\s+)?\bgodz\.(?=\s*\d)", _godz, t)
        t = re.sub(r"\b(o|od|do|około|po|przed|między|godz\.|godzin\w*)(\s+)([01]?\d|2[0-4])\.([0-5]\d)\b(?![.,]\d)",
                   r"\1\2\3:\4", t, flags=re.I)  # „około 19.30”
        t = re.sub(r"\bnr\.?(?=\s*\d)", "numer", t)
        t = re.sub(r"\bok\.(?=\s*\d)", "około", t)
        t = re.sub(r"\bpkt\.?(?=\s*\d)", "punkt", t)
        t = re.sub(r"\bstr\.(?=\s*\d)", "strona", t)
        self.text = t
        out, pos = [], 0
        while True:
            m = RE.search(self.text, pos)
            if not m:
                break
            try:
                zamiana, koniec = self.obsluz(m)
            except Exception as e:  # nie przerywaj całego zbioru: czytaj cyfry w mianowniku i oflaguj
                zamiana, koniec = self._liczba_tekst(m.group(0)), m.end()
                self.flagi.append(f"wyjatek:{type(e).__name__}:{m.group(0)}")
            out.append(self.text[pos:m.start()])
            out.append(zamiana)
            pos = koniec
        out.append(self.text[pos:])
        wynik = "".join(out)
        wynik = re.sub(r"[ \t]{2,}", " ", wynik).strip()
        if re.search(r"\d", wynik):
            self.flagi.append("zostały_cyfry")
        return wynik

    # --------------------------------------------------------- przypadki szczególne
    def obsluz(self, m):
        k = m.lastgroup if m.lastgroup in ("data", "godzz", "godz", "stos", "zakres", "zloz", "ulam", "porz",
                                           "liczba") else None
        for name in ("data", "godzz", "godz", "stos", "zakres", "zloz", "ulam", "porz", "rzym", "liczba"):
            if m.group(name) is not None:
                k = name
                break
        return getattr(self, "_" + k)(m)

    def _data(self, m):
        d, mies, r = int(m["dd"]), int(m["mm"]), int(m["rrrr"])
        if not (1 <= d <= 31 and 1 <= mies <= 12):
            return self._liczba_tekst(m.group(0)), m.end()
        nazwa = list(MIESIACE)[mies - 1]
        rok = porzadkowy(r, "gen")
        reszta = self.text[m.end():]
        mr = re.match(r"\s*r\.", reszta)
        dop = "" if re.match(r"\s*roku", reszta) else " roku"
        return f"{porzadkowy(d, 'gen')} {nazwa} {rok}{dop}", m.end() + (mr.end() if mr else 0)

    def _godzina(self, h: int, mi: int, case: str) -> str:
        hs = porzadkowy(h, case, "sg", "f") if 0 < h <= 24 else "zero"
        if mi == 0:
            return hs
        ms = glowny(mi) if mi >= 10 else "zero " + glowny(mi)
        return f"{hs} {ms}"

    def _przypadek_godziny(self, start):
        prep, ost = slowo_przed(self.text, start)
        if ost and ost.startswith("godz"):
            # „o godz. 15:00”, „godzina 13”: przypadek z przyimka przed „godz.”
            m = re.search(r"(" + SLOWO + r")\s+" + re.escape(ost) + r"\s*$", self.text[:start], re.I)
            p2 = m.group(1).lower() if m else None
            mapa = {"o": "loc", "po": "loc", "od": "gen", "do": "gen", "około": "gen", "przed": "inst",
                    "między": "inst", "za": "acc", "na": "acc"}
            c = mapa.get(p2, {"godzina": "nom", "godzinie": "loc", "godziny": "gen", "godzinę": "acc",
                               "godziną": "inst"}.get(ost, "nom"))
            return c
        if prep is None and ost == "a" and re.search(r"(między|pomiędzy)\s+\S+\s+a\s+$", self.text[:start]):
            return "inst"
        mapa = {"o": "loc", "po": "loc", "od": "gen", "do": "gen", "około": "gen", "koło": "gen", "przed": "inst",
                "między": "inst", "pomiędzy": "inst", "za": "acc", "na": "acc", "w": "loc", "z": "gen"}
        return mapa.get(prep, "nom")

    def _godzz(self, m):
        h1, m1, h2, m2 = (int(m[x]) for x in ("h1", "m1", "h2", "m2"))
        if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
            return self._liczba_tekst(m.group(0)), m.end()
        prep, _ = slowo_przed(self.text, m.start())
        if prep in ("od", "z"):
            return f"{self._godzina(h1, m1, 'gen')} do {self._godzina(h2, m2, 'gen')}", m.end()
        if prep in ("w", "między", "pomiędzy"):
            return f"{self._godzina(h1, m1, 'inst')} a {self._godzina(h2, m2, 'inst')}" if prep != "w" else \
                f"godzinach od {self._godzina(h1, m1, 'gen')} do {self._godzina(h2, m2, 'gen')}", m.end()
        return f"od {self._godzina(h1, m1, 'gen')} do {self._godzina(h2, m2, 'gen')}", m.end()

    def _godz(self, m):
        h, mi = int(m["h"]), int(m["m"])
        if h > 24 or mi > 59:
            return self._stos(m, m["h"], m["m"])
        return self._godzina(h, mi, self._przypadek_godziny(m.start())), m.end()

    def _stos(self, m, a=None, b=None):
        a, b = a or m["sa"], b or m["sb"]
        return f"{self.czytaj(a)} do {self.czytaj(b, 'gen')}", m.end()

    def _zakres(self, m):
        a, b, sep = m["za"], m["zb"], m["zsep"]
        if self.math and sep.strip() == "-" and sep != sep.strip():
            return f"{self.czytaj(a)} minus {self.czytaj(b)}", m.end()
        va, vb = self.wartosc(a)[0], self.wartosc(b)[0]
        # kody pocztowe, numery: „00-950”, „123-456”
        if a[0] == "0" and len(a) > 1 or (len(a) == 3 and len(b) == 3 and not RE_JEDN.match(self.text, m.end())):
            return f"{cyfry(a)} {cyfry(b)}", m.end()
        prep, ost = slowo_przed(self.text, m.start())
        nast = slowo_po(self.text, m.end())
        if not self.wartosc(a)[1] and not self.wartosc(b)[1] and nast and nast.lower() in MIESIACE \
                and 1 <= va <= 31 and 1 <= vb <= 31:
            od = "" if prep in ("od", "z") else "od "
            return f"{od}{porzadkowy(va, 'gen')} do {porzadkowy(vb, 'gen')}", m.end()
        if 1000 <= va <= 2100 and vb < 100 and len(b) == 2 and va % 100 < vb:  # „1983-87”
            vb = va // 100 * 100 + vb
            b = str(vb)
        lata = all(1000 <= x <= 2100 for x in (va, vb)) and not self.wartosc(a)[1]
        if lata:
            if ost in ("latach", "lata", "lat", "latami", "sezonie", "sezon", "roku"):
                return f"{self.czytaj(a)} – {self.czytaj(str(vb))}", m.end()
            koniec = m.end()
            dop = ""
            mr = re.match(r"\s*(r\.|rok\b|roku\b|lata\b|latach\b)", self.text[koniec:])
            if mr and mr.group(1) == "r.":
                koniec += mr.end()
                dop = " roku"
            od = "" if prep in ("od", "z") else "od "
            return f"{od}{porzadkowy(va, 'gen')} do {porzadkowy(vb, 'gen')}{dop}", koniec
        # zakres zwykły: pierwsza w przypadku frazy, druga w dopełniaczu po „do”
        jedn, koniec = self.jednostka(m.end())
        rz = None if jedn else analizy_po(self.text, m.end())
        case, g, pewne, _, _ = ustal_przypadek(prep, vb, rz or [], decimal=bool(self.wartosc(b)[1]))
        if jedn:
            g = self.rodzaj_jednostki(jedn)
        cb = case if case not in ("nom", "acc") else "gen"
        tekst = f"{self.czytaj(a, case, g)} do {self.czytaj(b, cb, g)}"
        if prep == "od":
            tekst = f"{self.czytaj(a, 'gen', g)} do {self.czytaj(b, 'gen', g)}"
        if jedn:
            tekst += " " + self.jednostka_po(jedn, b, "gen")
        if not pewne:
            self.flagi.append(f"zakres_przypadek:{m.group(0)}")
        return tekst, koniec

    def _zloz(self, m):
        v, suf = int(m["zl"]), m["zs"]
        if SUF_PORZ.match(suf.lower()):
            # „5-ty”, „XX-lecie” nie; „3-cie miejsce” -> porządkowy w formie pasującej do końcówki
            for case in WSZYSTKIE:
                for num in ("sg", "pl"):
                    for g in ("m3", "f", "n", "m1"):
                        f = porzadkowy(v, case, num, g)
                        if f.endswith(suf.lower()):
                            return f, m.end()
            return porzadkowy(v), m.end()
        if suf.lower() in ("u", "ch", "ech", "ma", "oma", "ema", "em"):
            for case in ("gen", "inst", "dat", "loc"):
                for g in ("m3", "f", "m1"):
                    f = glowny(v, case, g)
                    if f.endswith(suf.lower()):
                        return f, m.end()
            return glowny(v, "gen"), m.end()
        if suf.lower() in ("latek", "latka", "latki", "latkiem", "latków", "latkom", "latce"):
            return f"{zlozenie(v)}{suf}", m.end()
        return f"{zlozenie(v)}{suf}", m.end()

    def _ulam(self, m):
        a, b = int(m["ua"]), int(m["ub"])
        if a >= 1000 and b >= 1000:  # „2023/2024”
            return f"{glowny(a)} na {glowny(b)}", m.end()
        if b == 0 or b > 1000:
            return f"{self.czytaj(m['ua'])} na {self.czytaj(m['ub'])}", m.end()
        licz = glowny(a, "nom", "f")
        if a == 1:
            mian = porzadkowy(b, "nom", "sg", "f")
        elif a % 10 in (2, 3, 4) and a % 100 not in (12, 13, 14):
            mian = porzadkowy(b, "nom", "pl", "f")
        else:
            mian = porzadkowy(b, "gen", "pl", "f")
        return f"{licz} {mian}", m.end()

    def _porz(self, m):
        v = int(m["pl"])
        prep, ost = slowo_przed(self.text, m.start())
        if ost in ("lat", "lata", "latach", "latami", "latom") and v % 10 == 0 and v >= 10:
            c = {"lat": "gen", "lata": "nom", "latach": "loc", "latami": "inst", "latom": "dat"}[ost]
            return porzadkowy(v % 100 if v >= 1000 else v, c, "pl", "m3"), m.end()
        rz = analizy_po(self.text, m.end())
        if rz:
            lem, t = rz[0]
            c = sorted(set(PRZYIMKI.get(prep, ["nom"])) & t["case"]) or sorted(t["case"])
            c = [x for x in WSZYSTKIE if x in c][0]
            return porzadkowy(v, c, t["num"], rodzaj(t["gen"])), m.end()
        return porzadkowy(v), m.end()

    def _rzym(self, m):
        txt = m.group(0)
        rz_txt = re.sub(r"\s?w\.$", "", txt)
        v = rzymska(rz_txt)
        if v is None or v > 40:
            return txt, m.end()
        prep, ost = slowo_przed(self.text, m.start())
        kand = PRZYIMKI.get(prep, None)
        if txt != rz_txt:  # „XX w.” -> „wiek”
            c = kand[0] if kand else "gen"
            return f"{porzadkowy(v, c)} {noun_form('wiek', 'sg', c) or 'wieku'}", m.end()
        rz = analizy_po(self.text, m.end())
        rz = [x for x in rz if x[1]["pos"] in ("subst", "depr", "ger")] or rz
        if rz:
            for c in (kand or []) + WSZYSTKIE:
                for lem, t in rz:
                    if c in t["case"]:
                        return porzadkowy(v, c, t["num"], rodzaj(t["gen"])), m.end()
        # „Jana Pawła II”: zgoda z poprzednim słowem
        if ost:
            for lem, t in analiza(ost):
                if t["pos"] == "subst" and t["case"]:
                    c = [x for x in WSZYSTKIE if x in t["case"]][0]
                    return porzadkowy(v, c, t["num"] or "sg", rodzaj(t["gen"])), m.end()
        return porzadkowy(v), m.end()

    def _liczba_tekst(self, s):
        return " ".join(self.czytaj(x) for x in re.findall(r"\d+", s))

    def rodzaj_jednostki(self, jedn):
        g = set()
        for _, t in gen_forms(JEDN[jedn][0]):
            if t["pos"] == "subst":
                g |= t["gen"]
        return rodzaj(g)

    def jednostka_po(self, jedn, liczba_zapis, case):
        lem, dop = JEDN[jedn]
        v, dec = self.wartosc(liczba_zapis)
        num, nc = ("sg", "gen") if dec else rzeczownik_po(v, case)
        return forma_jednostki(lem, dop, num, nc)

    def _liczba(self, m):
        s = m["l"]
        minus = "minus " if m["minus"] and (m.start() == 0 or self.text[m.start() - 1] in " (") else ""
        v, dec = self.wartosc(s)
        koniec = m.end()
        prep, ost = slowo_przed(self.text, m.start())
        przed = self.text[:m.start()]
        # liczba w środku słowa („MP3”, „H2O”, „COVID-19”): mianownik
        if re.search(r"[^\W\d_]-?$", przed) or re.match(r"[^\W\d_]", self.text[koniec:]) and not \
                RE_JEDN.match(self.text, koniec):
            lewo = " " if re.search(r"[^\W\d_]-?$", przed) else ""
            prawo = " " if re.match(r"[^\W\d_]", self.text[koniec:]) else ""
            return lewo + minus + self.czytaj(s) + prawo, koniec
        # „godz. 15”, „godzina 13”
        if ost and ost.startswith("godz") and not dec and v <= 24:
            return self._godzina(v, 0, self._przypadek_godziny(m.start())), koniec
        # „lat 80”, „latach 90”
        if ost in ("lat", "lata", "latach", "latami") and not dec and v % 10 == 0 and 20 <= v <= 90:
            c = {"lat": "gen", "lata": "nom", "latach": "loc", "latami": "inst"}[ost]
            if re.match(r"\.(?!\s+[A-ZĄĆĘŁŃÓŚŹŻ]|\s*$)", self.text[koniec:]):
                koniec += 1  # „lat 80.,” / „lat 80. z”: kropka porządkowa, nie koniec zdania
            return porzadkowy(v, c, "pl", "m3"), koniec
        # „w roku 1980”, „za rok 2005”: porządkowy zgodny z „rok” przed liczbą
        jr = RE_JEDN.match(self.text, koniec)
        if ost in ("rok", "roku", "rokiem") and not dec and 1000 <= v <= 2100 and (not jr or jr.group(1) == "r."):
            if jr:
                koniec = jr.end()  # „w roku 2023 r.” -> bez drugiego „roku”
            mr = re.search(r"(" + SLOWO + r")\s+" + ost + r"\s+$", przed, re.I)
            p2 = mr.group(1).lower() if mr else None
            c = {"rok": "acc" if p2 in PRZYIMKI else "nom", "rokiem": "inst"}.get(ost)
            if ost == "roku":
                c = "loc" if p2 in ("w", "we", "o", "po", "przy") else "gen"
            return porzadkowy(v, c), koniec
        # mnożnik + jednostka: „20 tys. zł”
        jedn, k2 = self.jednostka(koniec)
        if jedn:
            lem, _ = JEDN[jedn]
            g = self.rodzaj_jednostki(jedn)
            case = PRZYIMKI.get(prep, ["nom"])[0] if prep else "nom"
            if lem == "rok" and 1000 <= v <= 2100 and not dec:  # „2020 r.”
                case, _, _, _, _ = ustal_przypadek(prep, v, [])
                return f"{porzadkowy(v, case)} {noun_form('rok', 'sg', case) or 'rok'}", k2
            tekst = f"{minus}{self.czytaj(s, case, g)} {self.jednostka_po(jedn, s, case)}"
            if lem in MNOZNIKI:  # po „tysiąc” kolejna jednostka zawsze w dopełniaczu lm
                j2, k3 = self.jednostka(k2)
                if j2 and JEDN[j2][0] not in MNOZNIKI:
                    l2, d2 = JEDN[j2]
                    tekst += " " + forma_jednostki(l2, d2, "pl", "gen")
                    k2 = k3
            return tekst, k2
        # data słowna: „15 września”, rok po miesiącu
        nast = slowo_po(self.text, koniec)
        mies = re.match(r"\s*(?:(?:na|do|i|lub|,|-|–)\s*\d{1,2}\s+)?(" + "|".join(MIESIACE) + r")\b",
                        self.text[koniec:], re.I)
        if not dec and mies and 1 <= v <= 31:
            return porzadkowy(v, "acc" if prep == "na" else "gen"), koniec
        if not dec and ost in MIES_WSZ and 1000 <= v <= 2100:
            dop = "" if re.match(r"\s*(r\.|roku|rok)", self.text[koniec:]) else " roku"
            return porzadkowy(v, "gen") + dop, koniec
        rz = analizy_po(self.text, koniec)
        # rzeczownik może być po „lub/i/albo/czy/do/-” i kolejnej liczbie („2 lub 3 dni”)
        if not rz:
            mm = re.match(r"\s*(?:lub|i|albo|czy|do|,|-|–)\s*\d+(?:[.,]\d+)?\s*([^\W\d_]+)", self.text[koniec:])
            if mm:
                rz = analizy_rzeczownika(mm.group(1))
        # rok: „w 2010 roku”, „od 2021”, „2025 rok”
        rok = 1000 <= v <= 2100 and not dec and not re.fullmatch(r"\d{1,3}(?:[  ]\d{3})+", s)
        if rok and rz and rz[0][0] == "rok" and any(t["num"] == "sg" for _, t in rz):
            kand = PRZYIMKI.get(prep, ["nom", "acc"]) if prep else ["nom", "acc"]
            c = next((c for c in kand + WSZYSTKIE for l, t in rz if l == "rok" and t["num"] == "sg" and c in t["case"]),
                     kand[0])
            return porzadkowy(v, c), koniec
        porown = re.search(r"(większ|mniejsz|równ|niż|wynosi|przedział|zakres)\w*\s+\S+\s*$", przed, re.I)
        if rok and (self.math or porown or v == 1000):
            rok = False
        if rok and prep and not rz:
            return porzadkowy(v, PRZYIMKI[prep][0]), koniec
        if rok and prep and prep in ("w", "od", "do", "z", "sprzed", "po", "przed", "około") and \
                nast and nast[0].isupper():
            return porzadkowy(v, PRZYIMKI[prep][0]), koniec
        # porządkowy po liczbie z rzeczownikiem tylko w lp („21 wiek”, „3 miejsce”)
        if not dec and v >= 2 and nast and nast[0].islower():
            rz_ord = [(l, t) for l, t in rz if t["num"] == "sg" and t["pos"] == "subst"
                      and (l in ORD_RZECZ or (v <= 100 and not any(t2["num"] == "pl" for _, t2 in rz)))]
            lem_ord = {l for l, _ in rz_ord}
            mn = [(l, t) for l, t in rz if t["num"] == "pl" and l in lem_ord]
            ok_kard = ustal_przypadek(prep, v, mn)[2] if mn else False
            if rz_ord and not ok_kard:
                case, g, pewne, lem, num = ustal_przypadek(prep, 1, rz_ord)
                return porzadkowy(v, case, "sg", g), koniec
        case, g, pewne, lem, num = ustal_przypadek(prep, v, rz, decimal=bool(dec))
        if not pewne:
            self.flagi.append(f"przypadek:{s} {nast}")
        return minus + self.czytaj(s, case, g), koniec


def rzymska(s: str):
    w = {"I": 1, "V": 5, "X": 10}
    if not re.fullmatch(r"X{0,3}(IX|IV|V?I{0,3})", s):
        return None
    v = 0
    for i, ch in enumerate(s):
        v += -w[ch] if i + 1 < len(s) and w[s[i + 1]] > w[ch] else w[ch]
    return v or None


def normalizuj(text: str) -> tuple[str, list[str]]:
    n = Normalizator(text)
    return n.zamien(), n.flagi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--wejscie")
    ap.add_argument("--wyjscie")
    a = ap.parse_args()
    if a.test:
        for t in a.test:
            w, f = normalizuj(t)
            print(w, f if f else "")
        return
    if a.wejscie.endswith(".parquet"):
        import pyarrow.parquet as pq
        rows = pq.read_table(a.wejscie, columns=["id", "text"]).to_pylist()
    else:
        rows = [json.loads(l) for l in open(a.wejscie)]
    with open(a.wyjscie, "w") as out:
        for r in rows:
            if "tekst" in r:  # plik z wybierz_teksty.py -> gotowe wejście generuj.py (tekst słowny + tekst_cyfry)
                w, f = normalizuj(r["tekst"])
                wpis = {**r, "tekst": w, "tekst_cyfry": r["tekst"], "flagi": f}
            else:
                w, f = normalizuj(r["text"])
                wpis = {"id": r["id"], "text": r["text"], "text_slowny": w, "flagi": f}
            out.write(json.dumps(wpis, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
