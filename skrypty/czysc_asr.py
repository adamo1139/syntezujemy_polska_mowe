#!/usr/bin/env python3
"""Czyszczenie syntezy przez ASR: Parakeet Ultra -> porównanie z tekstem podanym nauczycielowi -> lista do usunięcia.

Trzy kroki (każdy wznawialny, wyniki w katalogu --praca, domyślnie <katalog syntezy>/asr):

  asr    transkrypcja klipów z manifest.jsonl przez Parakeet Ultra (moondream/parakeet-ultra, runtime Photon).
         --karty 1,2,3 puszcza po jednym procesie na kartę (każdy bierze swoją część zbioru, log w <praca>/).
  ocen   porównanie transkrypcji z polem „tekst” (to, co dostał VoxCPM2): obie strony przechodzą przez
         liczby/normalizuj.py (Parakeet pisze liczby cyframi) i tę samą normalizację słów. Miary:
           cer, wer            błąd znakowy/wyrazowy po normalizacji
           cos                 podobieństwo embeddingów (domyślnie intfloat/multilingual-e5-large)
           brak_konca          ostatnie słowa tekstu nie zostały powiedziane (ucięcie)
           dopisek             słowa dopowiedziane na końcu/początku (bełkot, halucynacja)
           powtorzenie         zapętlenie: n-gram powtórzony w ASR częściej niż w tekście
           tempo               znaki tekstu na sekundę daleko od mediany (cisza, przyspieszenie)
         Wynik: oceny.jsonl (wszystko), usun.txt (id do usunięcia), odsluch.txt (granica), raport.txt
         i --probki N najgorszych klipów skopiowanych do <praca>/probki/ do odsłuchu.
  Usunięcie z danych: python skrypty/przygotuj_audio.py --zastap KATALOG --usun KATALOG/asr/usun.txt

Przykład (karty 1,2,3,5,6,7, środowisko z `pip install moondream sentence-transformers morfeusz2 rapidfuzz`):

  python skrypty/czysc_asr.py asr wyniki/poziomka_liczby_slownie --karty 1,2,3,5,6,7
  python skrypty/czysc_asr.py ocen wyniki/poziomka_liczby_slownie --probki 40
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent


def wczytaj_manifest(katalog: Path) -> list[dict]:
    wiersze = []
    with (katalog / "manifest.jsonl").open(encoding="utf-8") as plik:
        for w in plik:
            if w.strip():
                wpis = json.loads(w)
                if "audio" in wpis:  # wpisy błędów z generuj.py nie mają audio
                    wiersze.append(wpis)
    return wiersze


def czesc_klipu(id_: str, n: int) -> int:
    return int(hashlib.sha1(id_.encode()).hexdigest(), 16) % n


def gotowe_id(sciezka: Path) -> set[str]:
    if not sciezka.is_file():
        return set()
    return {json.loads(w)["id"] for w in sciezka.open(encoding="utf-8") if w.strip()}


# ------------------------------------------------------------------------------------------------ ASR

def krok_asr_na_kartach(args) -> None:
    """Po jednym procesie potomnym na kartę, każdy z inną częścią zbioru; czeka na wszystkie."""
    import os
    import subprocess

    karty = [k.strip() for k in args.karty.split(",") if k.strip()]
    praca = Path(args.praca or Path(args.katalog) / "asr")
    praca.mkdir(parents=True, exist_ok=True)
    procesy = []
    for i, karta in enumerate(karty):
        polecenie = [sys.executable, __file__, "asr", args.katalog, "--praca", str(praca), "--czesc", f"{i}/{len(karty)}",
                     "--model", args.model, "--watki", str(args.watki)]
        if args.limit:
            polecenie += ["--limit", str(args.limit)]
        log = (praca / f"asr_karta{karta}.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": karta}
        procesy.append((karta, subprocess.Popen(polecenie, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"karta {karta}: część {i}/{len(karty)}, log {praca / f'asr_karta{karta}.log'}", flush=True)
    try:
        ostatnio = time.time()
        while any(p.poll() is None for _, p, _ in procesy):
            time.sleep(5)
            if time.time() - ostatnio < 60:
                continue
            ostatnio = time.time()
            zrobione = sum(sum(1 for _ in f.open(encoding="utf-8")) for f in praca.glob("asr_*_z_*.jsonl"))
            print(f"  transkrypcji: {zrobione}", flush=True)
    except KeyboardInterrupt:
        for _, p, _ in procesy:
            p.terminate()
        raise
    nieudane = [k for k, p, log in procesy if (log.close() or p.returncode)]
    if nieudane:
        sys.exit(f"Procesy na kartach {', '.join(nieudane)} skończyły się błędem — zobacz logi w {praca}. "
                 "To samo polecenie wznowi pracę.")
    print(f"Gotowe. Dalej: python {Path(__file__).relative_to(KATALOG_PROJEKTU)} ocen {args.katalog}")


def krok_asr(args) -> None:
    if args.karty:
        return krok_asr_na_kartach(args)
    import moondream as md

    katalog = Path(args.katalog)
    praca = Path(args.praca or katalog / "asr")
    praca.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in args.czesc.split("/"))
    wyjscie = praca / f"asr_{k}_z_{n}.jsonl"
    gotowe = gotowe_id(wyjscie)
    klipy = [w for w in wczytaj_manifest(katalog) if czesc_klipu(w["id"], n) == k and w["id"] not in gotowe]
    if args.limit:
        klipy = klipy[: args.limit]
    print(f"część {k}/{n}: {len(klipy)} klipów do zrobienia ({len(gotowe)} gotowych) -> {wyjscie}", flush=True)
    if not klipy:
        return

    t0, zrobione, bledy = time.time(), 0, 0
    with md.photon(args.model) as mowa, wyjscie.open("a", encoding="utf-8") as plik:
        def jeden(w):
            try:
                wynik = mowa.transcribe(audio=str(katalog / w["audio"]))
                return w, wynik["text"], None
            except Exception as e:  # jeden zły plik nie zatrzymuje części
                return w, None, repr(e)

        with ThreadPoolExecutor(args.watki) as pula:
            for w, tekst, blad in pula.map(jeden, klipy):
                if blad:
                    bledy += 1
                    print(f"  błąd {w['id']}: {blad}", flush=True)
                    continue
                plik.write(json.dumps({"id": w["id"], "asr": tekst}, ensure_ascii=False) + "\n")
                zrobione += 1
                if zrobione % 500 == 0:
                    plik.flush()
                    tempo = zrobione / (time.time() - t0)
                    print(f"  {zrobione}/{len(klipy)}  {tempo:.1f} klipów/s  błędów {bledy}", flush=True)
    print(f"część {k}/{n}: gotowe {zrobione}, błędów {bledy}, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ porównanie

def slowa(tekst: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+", tekst.lower())


def odleglosc(a: list, b: list) -> int:
    """Levenshtein na listach (słowa albo znaki); rapidfuzz, jeśli jest (ok. 100x szybciej)."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    poprzedni = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        biezacy = [i]
        for j, y in enumerate(b, 1):
            biezacy.append(min(poprzedni[j] + 1, biezacy[j - 1] + 1, poprzedni[j - 1] + (x != y)))
        poprzedni = biezacy
    return poprzedni[-1]


def powtorzenia(s: list[str], n: int = 3) -> int:
    """Ile razy najczęstszy n-gram występuje ponad pierwsze wystąpienie."""
    if len(s) < n * 2:
        return 0
    licz = collections.Counter(tuple(s[i:i + n]) for i in range(len(s) - n + 1))
    return max(licz.values()) - 1


def porownaj(ref: list[str], hyp: list[str]) -> dict:
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ops = [o for o in sm.get_opcodes()]
    brak_konca = dopisek_konca = dopisek_poczatku = 0
    if ops:
        op, i1, i2, j1, j2 = ops[-1]
        if op in ("delete", "replace") and i2 == len(ref):
            brak_konca = (i2 - i1) - (j2 - j1 if op == "replace" else 0)
        if op in ("insert", "replace") and j2 == len(hyp):
            dopisek_konca = (j2 - j1) - (i2 - i1 if op == "replace" else 0)
        op, i1, i2, j1, j2 = ops[0]
        if op in ("insert", "replace") and j1 == 0:
            dopisek_poczatku = (j2 - j1) - (i2 - i1 if op == "replace" else 0)
    zr, zh = " ".join(ref), " ".join(hyp)
    return {
        "wer": odleglosc(ref, hyp) / max(1, len(ref)),
        "cer": odleglosc(list(zr), list(zh)) / max(1, len(zr)),
        "brak_konca": max(0, brak_konca),
        "dopisek": max(0, dopisek_konca) + max(0, dopisek_poczatku),
        "powtorzenie": max(0, powtorzenia(hyp) - powtorzenia(ref)),
    }


def krok_ocen(args) -> None:
    sys.path.insert(0, str(KATALOG_PROJEKTU / "liczby"))
    from normalizuj import normalizuj

    katalog = Path(args.katalog)
    praca = Path(args.praca or katalog / "asr")
    asr = {}
    for p in sorted(praca.glob("asr_*_z_*.jsonl")):
        for w in p.open(encoding="utf-8"):
            if w.strip():
                x = json.loads(w)
                asr[x["id"]] = x["asr"]
    klipy = [w for w in wczytaj_manifest(katalog) if w["id"] in asr]
    brak = len(wczytaj_manifest(katalog)) - len(klipy)
    print(f"{len(klipy)} klipów z transkrypcją, {brak} bez (te nie są oceniane)")
    if not klipy:
        sys.exit("Brak transkrypcji: najpierw krok asr.")

    oceny = []
    for w in klipy:
        ref = slowa(normalizuj(w["tekst"])[0])
        hyp = slowa(normalizuj(asr[w["id"]] or "")[0])
        o = {"id": w["id"], "tekst": w["tekst"], "asr": asr[w["id"]], "sekundy": w["sekundy"], "audio": w["audio"],
             **porownaj(ref, hyp), "znaki_na_s": len(" ".join(ref)) / max(0.1, w["sekundy"])}
        oceny.append(o)

    # tempo: odchylenie od mediany w skali MAD (odporne na wartości odstające)
    tempa = [o["znaki_na_s"] for o in oceny]
    med = statistics.median(tempa)
    mad = statistics.median(abs(t - med) for t in tempa) or 1e-6
    for o in oceny:
        o["tempo_z"] = (o["znaki_na_s"] - med) / (1.4826 * mad)

    if not args.bez_embeddingow:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.embeddingi, device=args.urzadzenie)
        pref = "query: " if "e5" in args.embeddingi else ""
        ea = model.encode([pref + o["tekst"] for o in oceny], batch_size=256, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True)
        eb = model.encode([pref + (o["asr"] or "") for o in oceny], batch_size=256, normalize_embeddings=True,
                          show_progress_bar=True, convert_to_numpy=True)
        for o, a, b in zip(oceny, ea, eb):
            o["cos"] = float((a * b).sum())

    # decyzja
    usun, odsluch = [], []
    for o in oceny:
        powody = []
        if o["cer"] > args.max_cer:
            powody.append(f"cer {o['cer']:.2f}")
        if o["brak_konca"] >= args.min_brak:
            powody.append(f"ucięty koniec ({o['brak_konca']} sł.)")
        if o["dopisek"] >= args.min_dopisek:
            powody.append(f"dopisek ({o['dopisek']} sł.)")
        if o["powtorzenie"] >= 2:
            powody.append(f"zapętlenie (+{o['powtorzenie']})")
        if abs(o["tempo_z"]) > args.max_tempo_z:
            powody.append(f"tempo z={o['tempo_z']:+.1f}")
        if "cos" in o and o["cos"] < args.min_cos:
            powody.append(f"cos {o['cos']:.2f}")
        o["powody"] = powody
        if powody:
            usun.append(o)
        elif o["cer"] > args.odsluch_cer or ("cos" in o and o["cos"] < args.odsluch_cos):
            odsluch.append(o)

    with (praca / "oceny.jsonl").open("w", encoding="utf-8") as plik:
        for o in oceny:
            plik.write(json.dumps(o, ensure_ascii=False) + "\n")
    (praca / "usun.txt").write_text("".join(o["id"] + "\n" for o in usun), encoding="utf-8")
    (praca / "odsluch.txt").write_text("".join(o["id"] + "\n" for o in odsluch), encoding="utf-8")

    h = lambda xs: sum(x["sekundy"] for x in xs) / 3600  # noqa: E731
    linie = [
        f"klipów ocenionych: {len(oceny)} ({h(oceny):.1f} h)",
        f"mediana CER {statistics.median(o['cer'] for o in oceny):.3f}, WER {statistics.median(o['wer'] for o in oceny):.3f}, "
        f"tempo {med:.1f} znaków/s",
        f"do usunięcia: {len(usun)} ({len(usun) / len(oceny):.1%}, {h(usun):.1f} h)",
        f"do odsłuchu (granica): {len(odsluch)} ({h(odsluch):.1f} h)",
        "powody (klip może mieć kilka):",
    ]
    licznik = collections.Counter(p.split(" (")[0].split(" ")[0] for o in usun for p in o["powody"])
    linie += [f"  {p}: {c}" for p, c in licznik.most_common()]
    for prog in (0.02, 0.05, 0.1, 0.2, 0.5):
        linie.append(f"  CER > {prog}: {sum(o['cer'] > prog for o in oceny)}")
    linie.append("\nprzykłady do usunięcia:")
    for o in sorted(usun, key=lambda o: -o["cer"])[:15]:
        linie += [f"- {o['id']}  [{', '.join(o['powody'])}]", f"    tekst: {o['tekst'][:200]}",
                  f"    asr:   {(o['asr'] or '')[:200]}"]
    raport = "\n".join(linie)
    (praca / "raport.txt").write_text(raport + "\n", encoding="utf-8")
    print(raport)

    if args.probki:
        katalog_probek = praca / "probki"
        shutil.rmtree(katalog_probek, ignore_errors=True)
        katalog_probek.mkdir()
        wybrane = sorted(usun, key=lambda o: -o["cer"])[: args.probki // 2]
        wybrane += sorted(odsluch, key=lambda o: -o["cer"])[: args.probki - len(wybrane)]
        with (katalog_probek / "opis.txt").open("w", encoding="utf-8") as opis:
            for nr, o in enumerate(wybrane):
                nazwa = f"{nr:03d}_{'usun' if o['powody'] else 'odsluch'}_{o['id']}.flac"
                shutil.copy(katalog / o["audio"], katalog_probek / nazwa)
                opis.write(f"{nazwa}\n  powody: {', '.join(o['powody']) or '-'}  cer {o['cer']:.2f}\n"
                           f"  tekst: {o['tekst']}\n  asr:   {o['asr']}\n\n")
        print(f"próbki do odsłuchu: {katalog_probek} ({len(wybrane)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="krok", required=True)

    a = sub.add_parser("asr", help="transkrypcja Parakeet Ultra")
    a.add_argument("katalog", help="katalog syntezy (manifest.jsonl + audio/)")
    a.add_argument("--praca", help="katalog na wyniki (domyślnie <katalog>/asr)")
    a.add_argument("--karty", help="numery kart po przecinku: po jednym procesie na kartę")
    a.add_argument("--czesc", default="0/1", help="K/N: tylko ta część zbioru (używane wewnętrznie przez --karty)")
    a.add_argument("--model", default="moondream/parakeet-ultra")
    a.add_argument("--watki", type=int, default=32, help="równoległe żądania do Photon")
    a.add_argument("--limit", type=int, help="tylko N klipów (próba)")

    o = sub.add_parser("ocen", help="porównanie z tekstem, decyzja, raport")
    o.add_argument("katalog")
    o.add_argument("--praca")
    o.add_argument("--embeddingi", default="intfloat/multilingual-e5-large")
    o.add_argument("--urzadzenie", default="cuda")
    o.add_argument("--bez-embeddingow", action="store_true")
    o.add_argument("--max-cer", type=float, default=0.15, help="powyżej: usuń")
    o.add_argument("--min-cos", type=float, default=0.85, help="poniżej: usuń")
    o.add_argument("--min-brak", type=int, default=2, help="tyle niewypowiedzianych słów na końcu: usuń")
    o.add_argument("--min-dopisek", type=int, default=3, help="tyle dopowiedzianych słów na brzegu: usuń")
    o.add_argument("--max-tempo-z", type=float, default=5.0, help="odchylenie tempa (w MAD): usuń")
    o.add_argument("--odsluch-cer", type=float, default=0.06, help="powyżej (i nie usunięte): do odsłuchu")
    o.add_argument("--odsluch-cos", type=float, default=0.92)
    o.add_argument("--probki", type=int, default=0, help="skopiuj N najgorszych klipów do odsłuchu")

    args = p.parse_args()
    {"asr": krok_asr, "ocen": krok_ocen}[args.krok](args)


if __name__ == "__main__":
    main()
