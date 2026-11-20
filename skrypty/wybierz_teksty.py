#!/usr/bin/env python3
"""Z `teksty/fragmenty.jsonl` wybiera podzbiór na zadaną liczbę godzin mowy i dzieli go na części.

Zasady wyboru (deterministyczne dla danego --ziarno):
  - tematy na zmianę (po jednej rozmowie z każdego tematu w kółko), żeby dwa największe tematy
    nie zdominowały zbioru;
  - przy jednych odwiedzinach rozmowy: --pytan pytań użytkownika i --odpowiedzi fragmentów odpowiedzi;
    gdy temat nie ma już nieodwiedzonych rozmów, wraca do tych samych po kolejne fragmenty,
    aż się wyczerpie (małe tematy wchodzą wtedy w całości);
  - koniec, gdy szacowany czas (znaki / 14 na sekundę) osiągnie --godziny;
  - każda rozmowa trafia w całości do jednej z --czesci części (pole „czesc”), a każda część ma
    ten sam rozkład tematów. Syntezę jednej części robi `generuj.py --czesc N`.

Przykłady:
  python skrypty/wybierz_teksty.py                      # 500 h w 10 częściach po ok. 50 h
  python skrypty/wybierz_teksty.py --godziny 50 --czesci 1 --wyjscie teksty/maly.jsonl
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


def main() -> None:
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
    args = p.parse_args()

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
            if fragment["zrodlo_id"] not in zajete_rozmowy:
                rozmowy[fragment["zrodlo_id"]].append(fragment)

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
    sek = [f["znaki"] / ZNAKI_NA_SEKUNDE for f in wybrane]
    przedzialy = [(0, 2), (2, 5), (5, 10), (10, 15.01)]
    print("  długość: " + ", ".join(f"{a:g}–{b:.0f} s: {100 * sum(a <= s < b for s in sek) / ile:.0f}%" for a, b in przedzialy))
    na_temat: collections.Counter = collections.Counter()
    for f in wybrane:
        na_temat[str(f["temat"])] += f["znaki"]
    print("  tematy (udział czasu):")
    for temat, ile_znakow in na_temat.most_common():
        print(f"    {100 * ile_znakow / znaki:5.1f}%  {temat}")


if __name__ == "__main__":
    main()
