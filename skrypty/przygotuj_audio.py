#!/usr/bin/env python3
"""Przygotowuje dane do treningu: pliki FLAC na dysku, podział train/val i listy dla kodeka.

Źródło audio (pierwsze, które istnieje):
  1. `wyniki/poziomka/audio/` + `manifest.jsonl` — jeśli pliki po syntezie jeszcze leżą, są użyte w miejscu;
  2. pliki parquet (`wyniki/poziomka_parquet/data/`, albo pobrane z HF przez --repo) — klipy są wypakowywane
     do `dane/audio/` (ok. 61 GB; wznawialne: istniejące pliki są pomijane).

Wynik w `dane/`:
  metadane.jsonl     jeden wiersz na klip: id, sciezka (bezwzględna), tekst, sekundy, zrodlo_id, podzial, ...
  kodek_train.txt    ścieżki klipów treningowych (dla blue-codec)
  kodek_val.txt      ścieżki klipów walidacyjnych
  kodek_48k.json     konfiguracja kodeka z `trening/kodek_48k.json` z wpisanymi ścieżkami

Walidacja to całe rozmowy (`zrodlo_id`), nie pojedyncze klipy, więc żaden tekst z walidacji nie jest w treningu.

Przykłady:
  python skrypty/przygotuj_audio.py
  python skrypty/przygotuj_audio.py --repo HoldForQuake/VoxCPM2-pol-one-voice-367h
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
POLA = ("id", "tekst", "sekundy", "glos", "zrodlo_id", "rola", "tura", "temat", "czesc", "jakosc", "pytanie")


def z_plikow(wyniki: Path) -> list[dict]:
    """Klipy leżące na dysku po syntezie: ścieżki z manifestu."""
    wiersze = []
    with (wyniki / "manifest.jsonl").open(encoding="utf-8") as plik:
        for wiersz in plik:
            if wiersz.strip():
                wpis = json.loads(wiersz)
                wpis["sciezka"] = str((wyniki / wpis["audio"]).resolve())
                wiersze.append(wpis)
    brak = [w["sciezka"] for w in wiersze if not Path(w["sciezka"]).is_file()]
    if brak:
        sys.exit(f"W {wyniki} brakuje {len(brak)} plików z manifestu (np. {brak[0]}). Usuń katalog audio albo użyj parquet (--parquet).")
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
                sciezka = cel / f"{wpis['id']}.flac"
                if not sciezka.exists() or sciezka.stat().st_size != len(audio["bytes"]):
                    tymczasowa = sciezka.with_suffix(".flac.tmp")
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
    p.add_argument("--szablon", default=str(KATALOG_PROJEKTU / "trening" / "kodek_48k.json"))
    args = p.parse_args()

    wyniki, dane = Path(args.wyniki), Path(args.dane)
    dane.mkdir(parents=True, exist_ok=True)
    parquet = Path(args.parquet) if args.parquet else wyniki.with_name(wyniki.name + "_parquet") / "data"

    if (wyniki / "manifest.jsonl").is_file() and (wyniki / "audio").is_dir() and any((wyniki / "audio").iterdir()):
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

    # Walidacja: rozmowy o najmniejszym skrócie zrodlo_id — stały, niezależny od kolejności wybór.
    rozmowy = sorted({str(w["zrodlo_id"]) for w in wiersze}, key=lambda z: hashlib.sha1(z.encode()).hexdigest())
    walidacyjne = set(rozmowy[: args.rozmow_val])
    wiersze.sort(key=lambda w: w["id"])
    for w in wiersze:
        w["podzial"] = "val" if str(w["zrodlo_id"]) in walidacyjne else "train"

    with (dane / "metadane.jsonl").open("w", encoding="utf-8") as plik:
        for w in wiersze:
            wpis = {"sciezka": w["sciezka"], "podzial": w["podzial"], **{pole: w.get(pole) for pole in POLA}}
            plik.write(json.dumps(wpis, ensure_ascii=False) + "\n")
    for podzial in ("train", "val"):
        (dane / f"kodek_{podzial}.txt").write_text(
            "".join(w["sciezka"] + "\n" for w in wiersze if w["podzial"] == podzial), encoding="utf-8"
        )

    konfiguracja = json.loads(Path(args.szablon).read_text(encoding="utf-8"))
    konfiguracja["ae"]["data"]["train_metadata"] = [str((dane / "kodek_train.txt").resolve())]
    konfiguracja["ae"]["data"]["val_metadata"] = str((dane / "kodek_val.txt").resolve())
    (dane / "kodek_48k.json").write_text(json.dumps(konfiguracja, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for podzial in ("train", "val"):
        czesc = [w for w in wiersze if w["podzial"] == podzial]
        print(f"{podzial}: {len(czesc)} klipów, {sum(w['sekundy'] for w in czesc) / 3600:.1f} h, {len({w['zrodlo_id'] for w in czesc})} rozmów")
    print(f"Zapisano: {dane / 'metadane.jsonl'}, {dane / 'kodek_train.txt'}, {dane / 'kodek_val.txt'}, {dane / 'kodek_48k.json'}")


if __name__ == "__main__":
    main()
