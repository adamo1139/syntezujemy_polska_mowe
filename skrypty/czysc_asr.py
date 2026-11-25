#!/usr/bin/env python3
"""Czyszczenie syntezy przez ASR: Parakeet Ultra -> porównanie z tekstem podanym nauczycielowi -> lista do usunięcia.

Trzy kroki (każdy wznawialny, wyniki w katalogu --praca, domyślnie <katalog syntezy>/asr):

  asr    transkrypcja klipów z manifest.jsonl przez Parakeet Ultra (moondream/parakeet-ultra, runtime Photon).
         --karty 1,2,3 puszcza po jednym procesie na kartę (każdy bierze swoją część zbioru, log w <praca>/).
  ocen   porównanie transkrypcji z polem „tekst” (to, co dostał VoxCPM2): obie strony przechodzą przez
         liczby/normalizuj.py (Parakeet pisze liczby cyframi) i tę samą normalizację słów. Miary:
           cer, wer            błąd znakowy (bez spacji) / wyrazowy po normalizacji
           dziura              najdłuższy ciągły fragment niezgodności w znakach (bełkot, pominięcie)
           brak_konca          ostatnie słowa tekstu nie zostały powiedziane (ucięcie)
           dopisek             słowa dopowiedziane na końcu/początku (bełkot, halucynacja)
           powtorzenie         zapętlenie: n-gram powtórzony w ASR częściej niż w tekście
           tempo               znaki tekstu na sekundę daleko od mediany (cisza, przyspieszenie)
         Wynik: oceny.jsonl (wszystko), usun.txt (id do usunięcia), odsluch.txt (granica), raport.txt
         i --probki N najgorszych klipów skopiowanych do <praca>/probki/ do odsłuchu.
  Usunięcie z danych: python skrypty/przygotuj_audio.py --zastap KATALOG --usun KATALOG/asr/usun.txt

Przykład (karty 1,2,3,5,6,7, środowisko z `uv pip install moondream morfeusz2 rapidfuzz`):

  python skrypty/czysc_asr.py asr wyniki/poziomka_liczby_slownie --karty 1,2,3,5,6,7
  python skrypty/czysc_asr.py ocen wyniki/poziomka_liczby_slownie --usun-procent 10 --usun-bez-asr --probki 40
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
    """Klipy z manifest.jsonl (katalog syntezy) albo z metadane.jsonl (katalog dane/ po przygotuj_audio.py)."""
    if not (katalog / "manifest.jsonl").is_file() and (katalog / "metadane.jsonl").is_file():
        wiersze = [json.loads(w) for w in (katalog / "metadane.jsonl").open(encoding="utf-8") if w.strip()]
        for w in wiersze:
            w["audio"] = w["sciezka"]  # ścieżka bezwzględna; katalog / bezwzględna = bezwzględna
            w["tekst"] = w.get("tekst_slowny") or w["tekst"]  # to, co dostał nauczyciel
        return wiersze
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
    zrobione = sum(sum(1 for _ in f.open(encoding="utf-8")) for f in praca.glob("asr_*_z_*.jsonl"))
    print(f"Transkrypcji razem: {zrobione} z {len(wczytaj_manifest(Path(args.katalog)))} klipów.")
    print(f"Gotowe. Dalej: python {Path(__file__).relative_to(KATALOG_PROJEKTU)} ocen {args.katalog}")


def krok_asr(args) -> None:
    if args.karty:
        return krok_asr_na_kartach(args)

    katalog = Path(args.katalog)
    praca = Path(args.praca or katalog / "asr")
    praca.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in args.czesc.split("/"))
    wyjscie = praca / f"asr_{k}_z_{n}.jsonl"
    gotowe = gotowe_id(wyjscie)
    klipy = [w for w in wczytaj_manifest(katalog) if czesc_klipu(w["id"], n) == k and w["id"] not in gotowe]
    if klipy and not (katalog / klipy[0]["audio"]).is_file():
        sys.exit(f"Nie ma pliku audio {katalog / klipy[0]['audio']} — manifest jest, ale nagrań w {katalog} brak. "
                 "Jeśli klipy leżą w dane/audio, podaj katalog dane (z metadane.jsonl) i --praca.")
    if args.limit:
        klipy = klipy[: args.limit]
    import moondream as md

    print(f"część {k}/{n}: {len(klipy)} klipów do zrobienia ({len(gotowe)} gotowych) -> {wyjscie}", flush=True)
    if not klipy:
        return

    t0, zrobione, bledy = time.time(), 0, 0
    with md.photon(args.model) as mowa, wyjscie.open("a", encoding="utf-8") as plik:
        def jeden(w):
            try:
                sciezka = katalog / w["audio"]
                if sciezka.suffix.lower() in (".ogg", ".opus"):   # Opus: dekodowanie do tymczasowego WAV (pewne wejście)
                    import tempfile
                    import soundfile as sf
                    x, sr = sf.read(str(sciezka), dtype="float32")
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                        sf.write(tmp.name, x, sr, subtype="PCM_16")
                        wynik = mowa.transcribe(audio=tmp.name)
                else:
                    wynik = mowa.transcribe(audio=str(sciezka))
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
    if bledy > max(10, zrobione):  # większość klipów się nie udała: to nie pojedyncze złe pliki
        sys.exit(f"część {k}/{n}: {bledy} błędów na {zrobione} transkrypcji — coś jest nie tak (patrz błędy wyżej)")


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


SKROTY = [(r"\bnp\.", "na przykład"), (r"\bm\.\s?in\.", "między innymi"), (r"\bitd\.", "i tak dalej"),
          (r"\bitp\.", "i tym podobne"), (r"\btj\.", "to jest"), (r"\btzn\.", "to znaczy"), (r"\+", " plus "),
          (r"π", " pi "), (r"&", " i ")]


def do_porownania(tekst: str, normalizuj) -> list[str]:
    """Słowa do porównania: skróty i symbole rozwinięte tak, jak je zapisuje ASR, liczby słownie."""
    for wzor, zamiana in SKROTY:
        tekst = re.sub(wzor, zamiana, tekst, flags=re.I)
    return slowa(normalizuj(tekst)[0])


def porownaj(ref: list[str], hyp: list[str]) -> dict:
    """Miary liczone na znakach bez spacji („Ergo Hestia” = „ergohestia”). „dziura” to najdłuższy ciągły
    fragment niezgodności (odległość znakowa w jednym miejscu): bełkot daje dużą, różnica w pisowni małą."""
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]

    def znaki(op):
        _, i1, i2, j1, j2 = op
        return odleglosc(list("".join(ref[i1:i2])), list("".join(hyp[j1:j2])))

    dziura = max((znaki(o) for o in ops), default=0)
    brak_konca = dopisek_konca = dopisek_poczatku = 0
    if ops:
        op, i1, i2, j1, j2 = ops[-1]
        if znaki(ops[-1]) >= 6:  # mniejsza różnica na końcu to pisownia, nie ucięcie
            if op in ("delete", "replace") and i2 == len(ref):
                brak_konca = (i2 - i1) - (j2 - j1 if op == "replace" else 0)
            if op in ("insert", "replace") and j2 == len(hyp):
                dopisek_konca = (j2 - j1) - (i2 - i1 if op == "replace" else 0)
        op, i1, i2, j1, j2 = ops[0]
        if op in ("insert", "replace") and j1 == 0 and znaki(ops[0]) >= 6:
            dopisek_poczatku = (j2 - j1) - (i2 - i1 if op == "replace" else 0)
    zr, zh = "".join(ref), "".join(hyp)
    return {
        "wer": odleglosc(ref, hyp) / max(1, len(ref)),
        "cer": odleglosc(list(zr), list(zh)) / max(1, len(zr)),
        "dziura": dziura,
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
    bez_asr = [w for w in wczytaj_manifest(katalog) if w["id"] not in asr]
    print(f"{len(klipy)} klipów z transkrypcją, {len(bez_asr)} bez"
          + (" (idą do usun.txt: --usun-bez-asr)" if args.usun_bez_asr else " (nie są oceniane ani usuwane)"))
    if not klipy:
        sys.exit("Brak transkrypcji: najpierw krok asr.")
    if args.usun_bez_asr and len(bez_asr) > 0.05 * len(klipy):
        sys.exit(f"{len(bez_asr)} klipów bez transkrypcji to ponad 5% zbioru — wygląda na niedokończony krok asr, "
                 "a nie na pojedyncze złe pliki. Dokończ asr albo pomiń --usun-bez-asr.")

    oceny = []
    for w in klipy:
        ref = do_porownania(w["tekst"], normalizuj)
        hyp = do_porownania(asr[w["id"]] or "", normalizuj)
        o = {"id": w["id"], "tekst": w["tekst"], "asr": asr[w["id"]], "sekundy": w["sekundy"], "audio": w["audio"],
             **porownaj(ref, hyp), "znaki_na_s": len(" ".join(ref)) / max(0.1, w["sekundy"])}
        oceny.append(o)

    # tempo: odchylenie od mediany w skali MAD (odporne na wartości odstające)
    tempa = [o["znaki_na_s"] for o in oceny]
    med = statistics.median(tempa)
    mad = statistics.median(abs(t - med) for t in tempa) or 1e-6
    for o in oceny:
        o["tempo_z"] = (o["znaki_na_s"] - med) / (1.4826 * mad)

    # decyzja
    usun, odsluch = [], []
    for o in oceny:
        powody = []
        if o["cer"] > args.max_cer:
            powody.append(f"cer {o['cer']:.2f}")
        if o["dziura"] >= args.max_dziura:
            powody.append(f"dziura {o['dziura']} zn.")
        if o["brak_konca"] >= args.min_brak:
            powody.append(f"ucięty koniec ({o['brak_konca']} sł.)")
        if o["dopisek"] >= args.min_dopisek:
            powody.append(f"dopisek ({o['dopisek']} sł.)")
        if o["powtorzenie"] >= 2:
            powody.append(f"zapętlenie (+{o['powtorzenie']})")
        if abs(o["tempo_z"]) > args.max_tempo_z:
            powody.append(f"tempo z={o['tempo_z']:+.1f}")
        o["powody"] = powody
        if powody:
            usun.append(o)
        elif o["cer"] > args.odsluch_cer or o["dziura"] >= args.odsluch_dziura:
            odsluch.append(o)

    if args.usun_procent:  # prosto: najgorsze N% według CER, progi powyżej nie mają znaczenia
        kolejnosc = sorted(oceny, key=lambda o: -o["cer"])
        ile = round(len(oceny) * args.usun_procent / 100)
        usun, odsluch = kolejnosc[:ile], kolejnosc[ile:ile + 200]  # do odsłuchu: 200 klipów tuż za granicą
        for o in oceny:
            o["powody"] = []
        for o in usun:
            o["powody"] = [f"najgorsze {args.usun_procent:g}% (cer {o['cer']:.2f})"]
        print(f"Usuwam najgorsze {args.usun_procent:g}% według CER: {ile} klipów, granica CER {usun[-1]['cer']:.3f}")

    with (praca / "oceny.jsonl").open("w", encoding="utf-8") as plik:
        for o in oceny:
            plik.write(json.dumps(o, ensure_ascii=False) + "\n")
    usun_bez_asr = bez_asr if args.usun_bez_asr else []
    (praca / "usun.txt").write_text("".join(o["id"] + "\n" for o in usun + usun_bez_asr), encoding="utf-8")
    (praca / "odsluch.txt").write_text("".join(o["id"] + "\n" for o in odsluch), encoding="utf-8")

    h = lambda xs: sum(x["sekundy"] for x in xs) / 3600  # noqa: E731
    linie = [
        f"klipów ocenionych: {len(oceny)} ({h(oceny):.1f} h)",
        f"mediana CER {statistics.median(o['cer'] for o in oceny):.3f}, WER {statistics.median(o['wer'] for o in oceny):.3f}, "
        f"tempo {med:.1f} znaków/s",
        f"do usunięcia: {len(usun)} ({len(usun) / len(oceny):.1%}, {h(usun):.1f} h)"
        + (f" + {len(usun_bez_asr)} bez transkrypcji ({h(usun_bez_asr):.1f} h)" if usun_bez_asr else ""),
        f"do odsłuchu (granica): {len(odsluch)} ({h(odsluch):.1f} h)",
        *([f"granica CER: {usun[-1]['cer']:.3f} (najgorsze {args.usun_procent:g}%)"] if args.usun_procent and usun else []),
        "powody (klip może mieć kilka):",
    ]
    licznik = collections.Counter(p.split(" (")[0].split(" ")[0] for o in usun for p in o["powody"])
    linie += [f"  {p}: {c}" for p, c in licznik.most_common()]
    for prog in (6, 8, 10, 12, 16, 24, 40):
        linie.append(f"  dziura >= {prog} zn.: {sum(o['dziura'] >= prog for o in oceny)}")
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
        if args.usun_procent:  # klipy po obu stronach granicy: ostatnie usunięte i pierwsze zostawione
            wybrane = usun[-(args.probki // 2):] + odsluch[: args.probki - args.probki // 2]
        else:
            wybrane = sorted(usun, key=lambda o: -o["cer"])[: args.probki // 2]
            wybrane += sorted(odsluch, key=lambda o: -o["cer"])[: args.probki - len(wybrane)]
        with (katalog_probek / "opis.txt").open("w", encoding="utf-8") as opis:
            for nr, o in enumerate(wybrane):
                nazwa = f"{nr:03d}_{'usun' if o['powody'] else 'odsluch'}_{o['id']}{Path(o['audio']).suffix}"
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
    o.add_argument("--usun-bez-asr", action="store_true",
                   help="klipy bez transkrypcji (ASR się nie udał, np. przester) też do usun.txt")
    o.add_argument("--usun-procent", type=float,
                   help="usuń po prostu najgorsze N%% klipów według CER (zamiast progów poniżej)")
    o.add_argument("--max-cer", type=float, default=0.25, help="powyżej: usuń")
    o.add_argument("--max-dziura", type=int, default=12,
                   help="tyle znaków niezgodności w jednym miejscu (bełkot, pominięty fragment): usuń")
    o.add_argument("--odsluch-dziura", type=int, default=8)
    o.add_argument("--min-brak", type=int, default=2, help="tyle niewypowiedzianych słów na końcu: usuń")
    o.add_argument("--min-dopisek", type=int, default=3, help="tyle dopowiedzianych słów na brzegu: usuń")
    o.add_argument("--max-tempo-z", type=float, default=5.0, help="odchylenie tempa (w MAD): usuń")
    o.add_argument("--odsluch-cer", type=float, default=0.10, help="powyżej (i nie usunięte): do odsłuchu")
    o.add_argument("--probki", type=int, default=0, help="skopiuj N najgorszych klipów do odsłuchu")

    args = p.parse_args()
    {"asr": krok_asr, "ocen": krok_ocen}[args.krok](args)


if __name__ == "__main__":
    main()
