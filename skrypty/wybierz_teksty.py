#!/usr/bin/env python3
"""Z `teksty/fragmenty.jsonl` wybiera podzbiór na zadaną liczbę godzin mowy i dzieli go na części.

Zasady wyboru (deterministyczne dla danego --ziarno):
  - tematy na zmianę (po jednej rozmowie z każdego tematu w kółko), żeby dwa największe tematy
    nie zdominowały zbioru;
  - przy jednych odwiedzinach rozmowy: --pytan pytań użytkownika i --odpowiedzi fragmentów odpowiedzi;
    gdy temat nie ma już nieodwiedzonych rozmów, wraca do tych samych po kolejne fragmenty,
    aż się wyczerpie (małe tematy wchodzą wtedy w całości);
  - koniec, gdy szacowany czas (znaki / --znakow-na-sekunde, domyślnie 14) osiągnie --godziny; głos „poziomka”
    mówi ok. 18,7 znaku/s, więc dla faktycznych godzin nagrań podać --znakow-na-sekunde 18.7;
  - każda rozmowa trafia w całości do jednej z --czesci części (pole „czesc”), a każda część ma
    ten sam rozkład tematów. Syntezę jednej części robi `generuj.py --czesc N`.

Filtry (do dobierania danych o konkretnym charakterze, np. z liczbami):
  --wzorzec REGEX   tylko fragmenty, w których jest dopasowanie (można powtórzyć: wystarczy jedno z nich);
                    gotowe nazwy: @daty, @matematyka, @liczby;
  --temat TEKST     tylko rozmowy, których temat zawiera ten tekst (można powtórzyć).

Przykłady:
  python skrypty/wybierz_teksty.py                      # 500 h w 10 częściach po ok. 50 h
  python skrypty/wybierz_teksty.py --godziny 50 --czesci 1 --wyjscie teksty/maly.jsonl
  python skrypty/wybierz_teksty.py --fragmenty teksty/fragmenty_wszystkie.jsonl --pomin teksty/wybrane.jsonl \
    --wzorzec @daty --godziny 25 --czesci 1 --wyjscie teksty/liczby_daty.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
ZNAKI_NA_SEKUNDE = 14.0

_MIESIACE = "stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia"
GOTOWE_WZORCE = {
    # rok (1000–2099), data liczbowa, dzień z miesiącem, godzina, wiek rzymski, lata 90., „r.”
    "@daty": (r"\b(1\d|20)\d\d\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|\b\d{1,2}\s+(" + _MIESIACE + r")\b"
              r"|\b\d{1,2}:\d{2}\b|\b[XVI]{1,5}\s+(w\.|wiek)|\blat(ach|a)?\s+\d0\.?|\b\d{3,4}\s?r\."),
    # działania i zapis liczbowy: 2 + 3, 3/4, 12,5, 40%, działania słownie przy liczbach
    "@matematyka": (r"\d\s*[+/:]\s*\d|\d\s+-\s+\d|\b\d+,\d+\b|\d\s*%"
                    r"|\d\s+(do\s+potęgi|razy|podzielić|plus|minus|procent)"
                    # słowa, na które przygotuj_teksty.py zamienia symbole „=”, „×”, „²”, „√”, „≤” …
                    r"|równa się|do kwadratu|do sześcianu|do potęgi|pierwiastek z|jest (mniejsze|większe)( lub równe)? od"),
    "@liczby": r"\d",
}


def main() -> None:
    global ZNAKI_NA_SEKUNDE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fragmenty", default=str(KATALOG_PROJEKTU / "teksty" / "fragmenty.jsonl"))
    p.add_argument("--wyjscie", default=str(KATALOG_PROJEKTU / "teksty" / "wybrane.jsonl"))
    p.add_argument("--godziny", type=float, default=500.0, help="docelowy szacowany czas mowy")
    p.add_argument("--czesci", type=int, default=10, help="na ile równych części podzielić wybór (pole „czesc”, liczone od 0)")
    p.add_argument("--rund", type=int, default=0, help="ile razy wolno wrócić do tej samej rozmowy po kolejne fragmenty; 0 = bez limitu")
    p.add_argument("--pytan", type=int, default=1, help="tyle pytań użytkownika z rozmowy przy jednych odwiedzinach")
    p.add_argument("--odpowiedzi", type=int, default=2, help="tyle fragmentów odpowiedzi z rozmowy przy jednych odwiedzinach")
    p.add_argument("--pomin", action="append", default=[], help="plik .jsonl z wcześniejszym wyborem; jego rozmowy są pomijane")
    p.add_argument("--ziarno", type=int, default=0)
    p.add_argument("--znakow-na-sekunde", type=float, default=ZNAKI_NA_SEKUNDE,
                   help="tempo głosu do szacowania czasu (domyślnie 14; „poziomka” w VoxCPM2 mówi ok. 18,7)")
    p.add_argument("--wzorzec", action="append", default=[], help="regex albo @daty/@matematyka/@liczby (patrz wyżej)")
    p.add_argument("--temat", action="append", default=[], help="tylko tematy zawierające ten tekst")
    args = p.parse_args()
    ZNAKI_NA_SEKUNDE = args.znakow_na_sekunde
    import re
    wzorce = [re.compile(GOTOWE_WZORCE.get(w, w)) for w in args.wzorzec]

    fragmenty = Path(args.fragmenty)
    if not fragmenty.is_file():
        sys.exit(f"Nie ma {fragmenty} — najpierw skrypty/przygotuj_teksty.py")

    zajete_rozmowy: set[str] = set()
    for sciezka in args.pomin:
        with open(sciezka, encoding="utf-8") as plik:
            zajete_rozmowy.update(json.loads(wiersz)["zrodlo_id"] for wiersz in plik if wiersz.strip())

    rozmowy: dict[str, list[dict]] = collections.defaultdict(list)
    with fragmenty.open(encoding="utf-8") as plik:
        for wiersz in plik:
            fragment = json.loads(wiersz)
            if fragment["zrodlo_id"] in zajete_rozmowy:
                continue
            if args.temat and not any(t in str(fragment.get("temat")) for t in args.temat):
                continue
            if wzorce and not any(w.search(fragment["tekst"]) for w in wzorce):
                continue
            rozmowy[fragment["zrodlo_id"]].append(fragment)
    if wzorce or args.temat:
        print(f"Po filtrach: {sum(map(len, rozmowy.values()))} fragmentów z {len(rozmowy)} rozmów, "
              f"ok. {sum(f['znaki'] for r in rozmowy.values() for f in r) / ZNAKI_NA_SEKUNDE / 3600:.1f} h")

    los = random.Random(args.ziarno)
    # Dla każdej rozmowy: potasowane pytania i odpowiedzi, z których kolejne odwiedziny biorą po kilka.
    zapas: dict[str, dict[str, list[dict]]] = {}
    tematy_rozmow: dict[str, list[str]] = collections.defaultdict(list)
    for zrodlo_id in sorted(rozmowy):
        pytania = [f for f in rozmowy[zrodlo_id] if f["rola"] == "user"]
        odpowiedzi = [f for f in rozmowy[zrodlo_id] if f["rola"] == "assistant"]
        los.shuffle(pytania)
        los.shuffle(odpowiedzi)
        zapas[zrodlo_id] = {"user": pytania, "assistant": odpowiedzi}
        tematy_rozmow[str(rozmowy[zrodlo_id][0]["temat"])].append(zrodlo_id)
    tematy = sorted(tematy_rozmow)
    los.shuffle(tematy)
    kolejki: dict[str, list[str]] = {temat: [] for temat in tematy}
    rundy: collections.Counter = collections.Counter()
    czesc_rozmowy: dict[str, int] = {}
    nowe_w_temacie: collections.Counter = collections.Counter()

    def nastepna_rozmowa(temat: str) -> str | None:
        """Kolejna rozmowa z tematu; gdy wszystkie były już odwiedzone, zaczyna następną rundę po resztkach."""
        if not kolejki[temat]:
            if args.rund and rundy[temat] >= args.rund:
                return None
            zostalo = [z for z in tematy_rozmow[temat] if zapas[z]["user"] or zapas[z]["assistant"]]
            if not zostalo:
                return None
            los.shuffle(zostalo)
            kolejki[temat] = zostalo
            rundy[temat] += 1
        return kolejki[temat].pop()

    cel_znakow = args.godziny * 3600 * ZNAKI_NA_SEKUNDE
    wybrane: list[dict] = []
    znaki = 0
    zywe = list(tematy)
    while znaki < cel_znakow and zywe:
        for temat in list(zywe):
            if znaki >= cel_znakow:
                break
            zrodlo_id = nastepna_rozmowa(temat)
            if zrodlo_id is None:
                zywe.remove(temat)
                continue
            if zrodlo_id not in czesc_rozmowy:
                # Części przydzielane po kolei w obrębie tematu: każda część ma ten sam rozkład tematów.
                czesc_rozmowy[zrodlo_id] = nowe_w_temacie[temat] % args.czesci
                nowe_w_temacie[temat] += 1
            for rola, ile in (("user", args.pytan), ("assistant", args.odpowiedzi)):
                for _ in range(min(ile, len(zapas[zrodlo_id][rola]))):
                    fragment = zapas[zrodlo_id][rola].pop()
                    fragment["czesc"] = czesc_rozmowy[zrodlo_id]
                    wybrane.append(fragment)
                    znaki += fragment["znaki"]

    wyjscie = Path(args.wyjscie)
    wyjscie.parent.mkdir(parents=True, exist_ok=True)
    with wyjscie.open("w", encoding="utf-8") as plik:
        for fragment in wybrane:
            plik.write(json.dumps(fragment, ensure_ascii=False) + "\n")

    godziny = znaki / ZNAKI_NA_SEKUNDE / 3600
    ile = len(wybrane)
    print(f"Wybrano {ile} fragmentów z {len({f['zrodlo_id'] for f in wybrane})} rozmów -> {wyjscie}")
    print(f"  szacowany czas: {godziny:.2f} h (cel {args.godziny:g} h), {znaki} znaków")
    na_czesc: collections.Counter = collections.Counter()
    for f in wybrane:
        na_czesc[f["czesc"]] += f["znaki"]
    print("  części (h): " + ", ".join(f"{c}: {na_czesc[c] / ZNAKI_NA_SEKUNDE / 3600:.1f}" for c in sorted(na_czesc)))
    wyczerpane = [t for t in tematy if t not in zywe]
    print(f"  tematy wzięte w całości: {len(wyczerpane)} z {len(tematy)}")
    if godziny < args.godziny * 0.99:
        print("  UWAGA: zabrakło fragmentów — obniż --percentyl w przygotuj_teksty.py")
    if not ile:
        return
    udzial = lambda warunek: 100 * sum(1 for f in wybrane if warunek(f)) / ile
    print(f"  pytania użytkownika: {udzial(lambda f: f['rola'] == 'user'):.0f}%, kończące się „?”: {udzial(lambda f: f['pytanie']):.0f}%")
    przedzialy = [("do 5 s", 0, 5), ("5–15 s", 5, 15), ("15–30 s", 15, 30), ("ponad 30 s", 30, 1e9)]
    w = [[f["znaki"] for f in wybrane if a <= f["znaki"] / ZNAKI_NA_SEKUNDE < b] for _, a, b in przedzialy]
    print("  rozkład klipów: " + ", ".join(f"{n}: {100 * len(x) / ile:.0f}%" for (n, _, _), x in zip(przedzialy, w)))
    print("  rozkład czasu:  " + ", ".join(f"{n}: {100 * sum(x) / znaki:.0f}%" for (n, _, _), x in zip(przedzialy, w)))
    na_temat: collections.Counter = collections.Counter()
    for f in wybrane:
        na_temat[str(f["temat"])] += f["znaki"]
    print("  tematy (udział czasu):")
    for temat, ile_znakow in na_temat.most_common():
        print(f"    {100 * ile_znakow / znaki:5.1f}%  {temat}")


if __name__ == "__main__":
    main()
