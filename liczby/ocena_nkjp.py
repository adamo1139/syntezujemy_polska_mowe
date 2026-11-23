#!/usr/bin/env python3
"""Ocena normalizuj.py na zbiorze Poświata i Perełkiewicz (BSNLP 2019, zdania z NKJP z wersją słowną).

  python liczby/ocena_nkjp.py ../zewn/text-normalization/data_set.txt [--pokaz 30]
Miary: dokładne trafienie zdania (bez wielkości liter i interpunkcji) i odsetek poprawnych wyrażeń liczbowych
(token z cyfrą w oryginale -> porównanie słów w miejscu tego tokenu, wyrównanie difflib).
"""
import argparse, difflib, json, random, re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from normalizuj import normalizuj

def slowa(t): return re.findall(r"[^\W_]+", t.lower())

ap = argparse.ArgumentParser(); ap.add_argument("plik"); ap.add_argument("--pokaz", type=int, default=20)
a = ap.parse_args()
dane = [json.loads(l) for l in open(a.plik)]
dobre = zle = 0; bledy = []
for d in dane:
    wyn, _ = normalizuj(d["sentence"])
    if slowa(wyn) == slowa(d["normalized_sentence"]): dobre += 1
    else:
        zle += 1
        x, y = slowa(wyn), slowa(d["normalized_sentence"])
        sm = difflib.SequenceMatcher(a=x, b=y, autojunk=False)
        roz = [(" ".join(x[i1:i2]), " ".join(y[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
        bledy.append((d["sentence"], roz))
print(f"zdań {len(dane)}, całe zdanie dobrze: {dobre/len(dane):.1%}")
# błędy na poziomie różnic
n_roz = sum(len(r) for _, r in bledy)
print(f"zdań z błędem {zle}, różnic {n_roz}")
random.seed(0)
for s, r in random.sample(bledy, min(a.pokaz, len(bledy))):
    print("-", s[:160]); [print("     ma:", x, "| wzorzec:", y) for x, y in r[:4]]
