#!/usr/bin/env python3
"""Przygotowuje dane do treningu: pliki FLAC na dysku, podział train/val, listy i konfigurację.

Źródło audio (pierwsze, które istnieje):
  1. `dane/metadane.jsonl` z wcześniejszego uruchomienia — pliki już są, powstają tylko listy na nowo;
  2. `wyniki/poziomka/audio/` + `manifest.jsonl` — jeśli pliki po syntezie jeszcze leżą, są użyte w miejscu;
  3. pliki parquet (`wyniki/poziomka_parquet/data/`, albo pobrane z HF przez --repo) — klipy są wypakowywane
     do `dane/audio/` (ok. 61 GB; wznawialne: istniejące pliki są pomijane).

Wynik w `dane/`:
  metadane.jsonl      jeden wiersz na klip: id, sciezka (bezwzględna), tekst, sekundy, zrodlo_id, podzial, ...
  kodek_train.txt     ścieżki klipów treningowych (dla kodeka)
  kodek_val.txt       ścieżki klipów walidacyjnych
  bluetts_train.csv   klipy treningowe z tekstem (dla modelu tekst->latenty i predyktora długości)
  bluetts_val.csv     to samo dla walidacji
  polski_100m.json    konfiguracja z `trening/config/polski_100m.json` z wpisanymi ścieżkami

Walidacja to całe rozmowy (`zrodlo_id`), nie pojedyncze klipy, więc żaden tekst z walidacji nie jest w treningu.

Dodatkowe syntezy (--dodaj KATALOG, można powtórzyć; katalog z manifest.jsonl i audio/ po generuj.py)
trafiają w całości do treningu, a podział istniejących klipów z `dane/metadane.jsonl` zostaje bez zmian,
więc walidacja jest dalej ta sama.

Nowe nagrania tych samych klipów (--zastap KATALOG, np. po ponownej syntezie z tekstem słownym): wpisy o tym
samym id dostają nowe audio i tekst, a podział zostaje; id, których jeszcze nie ma, trafiają do treningu jak
przy --dodaj.

Dodatkowe zbiory walidacyjne (--dodaj-val NAZWA=KATALOG, np. liczby=wyniki/poziomka_val_liczby): klipy dostają
podział „val_NAZWA” i trafiają tylko do `bluetts_val_NAZWA.csv` (nie do treningu ani do list kodeka). Ich rozmowy
nie mogą występować w żadnym innym podziale.

Przykłady:
  python skrypty/przygotuj_audio.py
  python skrypty/przygotuj_audio.py --repo RockTheBoat/VoxCPM2-pol-one-voice-367h
  python skrypty/przygotuj_audio.py --dodaj wyniki/poziomka_liczby
  python skrypty/przygotuj_audio.py --dodaj-val liczby=wyniki/poziomka_val_liczby
  python skrypty/przygotuj_audio.py --zastap wyniki/poziomka_liczby_slownie
  python skrypty/przygotuj_audio.py --usun wyniki/poziomka_liczby_slownie/asr/usun.txt   (id z czysc_asr.py)
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
POLA = ("id", "tekst", "tekst_slowny", "sekundy", "glos", "zrodlo_id", "rola", "tura", "temat", "czesc", "jakosc", "pytanie")


def z_plikow(wyniki: Path) -> list[dict]:
    """Klipy leżące na dysku po syntezie: ścieżki z manifestu."""
    wiersze = []
    with (wyniki / "manifest.jsonl").open(encoding="utf-8") as plik:
        for wiersz in plik:
            if wiersz.strip():
                wpis = json.loads(wiersz)
                wpis["sciezka"] = str((wyniki / wpis["audio"]).resolve())
                if "tekst_cyfry" in wpis:  # synteza z tekstu słownego (liczby/normalizuj.py): trenujemy na cyfrach
                    wpis["tekst_slowny"] = wpis["tekst"]
                    wpis["tekst"] = wpis.pop("tekst_cyfry")
                wiersze.append(wpis)
    brak = [w["sciezka"] for w in wiersze if not Path(w["sciezka"]).is_file()]
    if brak:
        sys.exit(f"W {wyniki} brakuje {len(brak)} plików z manifestu (np. {brak[0]}). Usuń katalog audio albo użyj parquet (--parquet).")
    return wiersze


def z_metadanych(metadane: Path) -> list[dict] | None:
    """Klipy z wcześniejszego uruchomienia, jeśli plik metadanych istnieje i wszystkie pliki audio są na miejscu."""
    if not metadane.is_file():
        return None
    wiersze = [json.loads(w) for w in metadane.open(encoding="utf-8") if w.strip()]
    brak = sum(1 for w in wiersze if not Path(w["sciezka"]).is_file())
    if not wiersze or brak:
        print(f"{metadane}: brakuje {brak} z {len(wiersze)} plików audio — przygotowuję dane od nowa.")
        return None
    return wiersze


def z_parquet(katalog: Path, cel: Path) -> list[dict]:
    """Wypakowuje klipy z plików parquet do `cel`; istniejące pliki o właściwym rozmiarze pomija."""
    import pyarrow.parquet as pq

    pliki = sorted(katalog.glob("*.parquet"))
    if not pliki:
        sys.exit(f"W {katalog} nie ma plików parquet.")
    cel.mkdir(parents=True, exist_ok=True)
    wiersze, t0, zapisane = [], time.time(), 0
    for nr, plik in enumerate(pliki, 1):
        tabela = pq.ParquetFile(plik)
        for paczka in tabela.iter_batches(batch_size=100):
            for wpis in paczka.to_pylist():
                audio = wpis.pop("audio")
                wpis["tekst"] = wpis.pop("text")
                if wpis.get("text_cyfry"):  # zbiór „liczby słownie”: trening na zapisie z cyframi
                    wpis["tekst_slowny"] = wpis["tekst"]
                    wpis["tekst"] = wpis.pop("text_cyfry")
                rozsz = ".ogg" if audio["bytes"][:4] == b"OggS" else ".flac"   # partie w Opus albo FLAC
                sciezka = cel / f"{wpis['id']}{rozsz}"
                if not sciezka.exists() or sciezka.stat().st_size != len(audio["bytes"]):
                    tymczasowa = sciezka.with_suffix(rozsz + ".tmp")
                    tymczasowa.write_bytes(audio["bytes"])
                    tymczasowa.rename(sciezka)
                    zapisane += 1
                wpis["sciezka"] = str(sciezka.resolve())
                wiersze.append(wpis)
        if nr % 10 == 0 or nr == len(pliki):
            print(f"  parquet {nr}/{len(pliki)}: {len(wiersze)} klipów, zapisano {zapisane} nowych  [{time.time() - t0:.0f} s]", flush=True)
    return wiersze


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wyniki", default=str(KATALOG_PROJEKTU / "wyniki" / "poziomka"), help="katalog po syntezie (manifest.jsonl + audio/)")
    p.add_argument("--parquet", help="katalog z plikami parquet (domyślnie <wyniki>_parquet/data)")
    p.add_argument("--repo", help="dataset na HF do pobrania, gdy parquet nie ma lokalnie")
    p.add_argument("--dane", default=str(KATALOG_PROJEKTU / "dane"), help="katalog wynikowy")
    p.add_argument("--rozmow-val", type=int, default=300, help="ile całych rozmów odłożyć do walidacji")
    p.add_argument("--szablon", default=str(KATALOG_PROJEKTU / "trening" / "config" / "polski_100m.json"))
    p.add_argument("--dodaj", action="append", default=[], help="katalog dodatkowej syntezy (manifest.jsonl + audio/), tylko do treningu")
    p.add_argument("--zastap", action="append", default=[],
                   help="katalog ponownej syntezy: klipy o tych samych id dostają nowe audio (podział bez zmian)")
    p.add_argument("--usun", action="append", default=[],
                   help="plik z id klipów do usunięcia z danych (po jednym w wierszu, np. usun.txt z czysc_asr.py)")
    p.add_argument("--podzial-z", metavar="METADANE",
                   help="przypisz klipom podział (train/val) ich rozmów z innego metadane.jsonl, np. sprzed przebudowy")
    p.add_argument("--dodaj-val", action="append", default=[], metavar="NAZWA=KATALOG",
                   help="dodatkowy zbiór walidacyjny: klipy z KATALOG -> dane/bluetts_val_NAZWA.csv")
    args = p.parse_args()

    wyniki, dane = Path(args.wyniki), Path(args.dane)
    dane.mkdir(parents=True, exist_ok=True)
    parquet = Path(args.parquet) if args.parquet else wyniki.with_name(wyniki.name + "_parquet") / "data"

    wiersze = z_metadanych(dane / "metadane.jsonl")
    if wiersze is not None:
        print(f"Używam klipów z {dane / 'metadane.jsonl'} ({len(wiersze)}); audio bez zmian.")
    elif (wyniki / "manifest.jsonl").is_file() and (wyniki / "audio").is_dir() and any((wyniki / "audio").iterdir()):
        print(f"Audio po syntezie leży w {wyniki / 'audio'} — używam w miejscu, bez kopiowania.")
        wiersze = z_plikow(wyniki)
    else:
        if not parquet.is_dir() or not any(parquet.glob("*.parquet")):
            if not args.repo:
                sys.exit(f"Nie ma audio w {wyniki} ani parquet w {parquet}. Podaj --parquet KATALOG albo --repo UZYTKOWNIK/NAZWA.")
            from huggingface_hub import snapshot_download

            print(f"Pobieranie {args.repo} do pamięci HF...")
            parquet = Path(snapshot_download(args.repo, repo_type="dataset", allow_patterns=["data/*.parquet"])) / "data"
        print(f"Wypakowuję klipy z {parquet} do {dane / 'audio'}...")
        wiersze = z_parquet(parquet, dane / "audio")

    identyfikatory = [w["id"] for w in wiersze]
    if len(set(identyfikatory)) != len(identyfikatory):
        sys.exit("Powtarzające się id klipów.")

    if args.podzial_z:   # ten sam podział rozmów co w starych danych (np. przebudowa z Opus po czyszczeniu)
        stary = {}
        for linia in open(args.podzial_z, encoding="utf-8"):
            if linia.strip():
                w = json.loads(linia)
                stary.setdefault(str(w["zrodlo_id"]), w["podzial"])
        bez = [w for w in wiersze if str(w["zrodlo_id"]) not in stary]
        if len(bez) > 0.01 * len(wiersze):
            sys.exit(f"--podzial-z: {len(bez)} klipów z rozmów, których nie ma w {args.podzial_z} — to inne dane niż poprzednio")
        if bez:   # rozmowy, których wszystkie klipy wypadły wcześniej (inny filtr): nie wiadomo, czy były w walidacji
            print(f"--podzial-z: pomijam {len(bez)} klipów z {len({w['zrodlo_id'] for w in bez})} rozmów, których nie ma "
                  f"w {args.podzial_z} (nie wiadomo, czy to trening, czy walidacja)")
            wiersze = [w for w in wiersze if str(w["zrodlo_id"]) in stary]
        for w in wiersze:
            w["podzial"] = stary[str(w["zrodlo_id"])]
        print(f"Podział rozmów z {args.podzial_z}: walidacja {len({w['zrodlo_id'] for w in wiersze if w['podzial'] == 'val'})} "
              f"rozmów, {sum(w['podzial'] == 'val' for w in wiersze)} klipów")

    # Walidacja: rozmowy o najmniejszym skrócie zrodlo_id — stały, niezależny od kolejności wybór.
    # Klipy z wcześniejszych metadanych zachowują swój podział (dodanie danych nie przesuwa walidacji).
    if not all(w.get("podzial") in ("train", "val") or str(w.get("podzial", "")).startswith("val_") for w in wiersze):
        rozmowy = sorted({str(w["zrodlo_id"]) for w in wiersze}, key=lambda z: hashlib.sha1(z.encode()).hexdigest())
        walidacyjne = set(rozmowy[: args.rozmow_val])
        for w in wiersze:
            w["podzial"] = "val" if str(w["zrodlo_id"]) in walidacyjne else "train"
    rozmowy_val = {str(w["zrodlo_id"]) for w in wiersze if w["podzial"] != "train"}

    znane = set(identyfikatory)
    dodatkowe = {}
    for katalog in args.zastap:
        po_id = {w["id"]: i for i, w in enumerate(wiersze)}
        nowe = z_plikow(Path(katalog))
        stare_s = nowe_s = 0.0
        for w in nowe:
            if w["id"] in po_id:
                stary = wiersze[po_id[w["id"]]]
                stare_s += stary["sekundy"]
                nowe_s += w["sekundy"]
                wiersze[po_id[w["id"]]] = {**stary, **{k: v for k, v in w.items() if v is not None}, "podzial": stary["podzial"]}
        reszta = [w for w in nowe if w["id"] not in po_id]
        print(f"Zastąpione z {katalog}: {len(nowe) - len(reszta)} klipów ({stare_s / 3600:.1f} h -> {nowe_s / 3600:.1f} h); "
              f"nowych id: {len(reszta)}")
        if reszta:
            dodatkowe[katalog] = reszta
    for katalog in args.dodaj + list(dodatkowe):
        nowe = [w for w in (dodatkowe.get(katalog) or z_plikow(Path(katalog))) if w["id"] not in znane]
        kolizje = sum(1 for w in nowe if str(w["zrodlo_id"]) in rozmowy_val)
        if kolizje:
            sys.exit(f"{katalog}: {kolizje} klipów z rozmów walidacyjnych — wybierz teksty z --pomin teksty/wybrane.jsonl")
        for w in nowe:
            w["podzial"] = "train"
            znane.add(w["id"])
        wiersze += nowe
        print(f"Dodane z {katalog}: {len(nowe)} klipów, {sum(w['sekundy'] for w in nowe) / 3600:.1f} h (trening)")
    for wpis in args.dodaj_val:
        nazwa, _, katalog = wpis.partition("=")
        if not nazwa or not katalog or not nazwa.replace("_", "").isalnum():
            sys.exit(f"--dodaj-val {wpis!r}: oczekiwane NAZWA=KATALOG (nazwa z liter, cyfr i _)")
        nowe = [w for w in z_plikow(Path(katalog)) if w["id"] not in znane]
        zajete = {str(w["zrodlo_id"]) for w in wiersze}
        kolizje = sum(1 for w in nowe if str(w["zrodlo_id"]) in zajete)
        if kolizje:
            sys.exit(f"{katalog}: {kolizje} klipów z rozmów, które już są w danych — wybierz teksty z --pomin wszystkich wcześniejszych wyborów")
        for w in nowe:
            w["podzial"] = f"val_{nazwa}"
            znane.add(w["id"])
        wiersze += nowe
        print(f"Dodane z {katalog}: {len(nowe)} klipów, {sum(w['sekundy'] for w in nowe) / 3600:.1f} h (walidacja val_{nazwa})")
    for plik_usun in args.usun:
        do_usuniecia = {w.strip() for w in Path(plik_usun).read_text(encoding="utf-8").splitlines() if w.strip()}
        przed = wiersze
        wiersze = [w for w in wiersze if w["id"] not in do_usuniecia]
        usuniete = [w for w in przed if w["id"] in do_usuniecia]
        print(f"Usunięte wg {plik_usun}: {len(usuniete)} klipów, {sum(w['sekundy'] for w in usuniete) / 3600:.1f} h "
              f"(z listy {len(do_usuniecia)}; " + ", ".join(f"{p}: {c}" for p, c in
              collections.Counter(w["podzial"] for w in usuniete).items()) + ")")
    wiersze.sort(key=lambda w: w["id"])
    podzialy = ["train", "val"] + sorted({w["podzial"] for w in wiersze} - {"train", "val"})

    with (dane / "metadane.jsonl").open("w", encoding="utf-8") as plik:
        for w in wiersze:
            wpis = {"sciezka": w["sciezka"], "podzial": w["podzial"], **{pole: w.get(pole) for pole in POLA}}
            plik.write(json.dumps(wpis, ensure_ascii=False) + "\n")
    for podzial in ("train", "val"):
        (dane / f"kodek_{podzial}.txt").write_text(
            "".join(w["sciezka"] + "\n" for w in wiersze if w["podzial"] == podzial), encoding="utf-8"
        )

    # Format, którego oczekuje trener BlueTTS. Wszystkie pola w cudzysłowach, żeby przecinki i cudzysłowy
    # w tekście nie zmyliły wykrywania separatora. "phonemized" = True wyłącza espeak: tekst idzie surowy.
    for podzial in podzialy:
        with (dane / f"bluetts_{podzial}.csv").open("w", encoding="utf-8", newline="") as plik:
            zapis = csv.writer(plik, quoting=csv.QUOTE_ALL)
            zapis.writerow(["filename", "whisper_phonemes", "speaker_id", "wer_score", "lang", "phonemized"])
            for w in wiersze:
                if w["podzial"] == podzial:
                    zapis.writerow([w["sciezka"], w["tekst"], 1, 0.0, "pl", True])

    konfiguracja = json.loads(Path(args.szablon).read_text(encoding="utf-8"))
    konfiguracja["ae"]["data"]["train_metadata"] = [str((dane / "kodek_train.txt").resolve())]
    konfiguracja["ae"]["data"]["val_metadata"] = str((dane / "kodek_val.txt").resolve())
    (dane / "polski_100m.json").write_text(json.dumps(konfiguracja, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for podzial in podzialy:
        czesc = [w for w in wiersze if w["podzial"] == podzial]
        print(f"{podzial}: {len(czesc)} klipów, {sum(w['sekundy'] for w in czesc) / 3600:.1f} h, {len({w['zrodlo_id'] for w in czesc})} rozmów")
    print(f"Zapisano w {dane}: metadane.jsonl, kodek_train.txt, kodek_val.txt, "
          + ", ".join(f"bluetts_{p}.csv" for p in podzialy) + ", polski_100m.json")


if __name__ == "__main__":
    main()
