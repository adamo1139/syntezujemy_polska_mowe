#!/usr/bin/env python3
"""Pakuje wynik syntezy (manifest.jsonl + audio/*.flac albo *.ogg) do plików parquet i wysyła jako dataset na HF.

Format: kolumny `audio` (plik FLAC albo Ogg Opus w środku, typ Audio z `datasets`) i `text`, plus metadane z manifestu.
Pliki `data/train-NNNNN-of-MMMMM.parquet` po ok. 500 MB i karta `README.md`.

Jeśli pełny zestaw plików parquet już leży w katalogu paczek, pakowanie jest pomijane w całości
(karta README.md też zostaje nietknięta). Wysyłka idzie partiami po 10 plików, każda partia to osobny
commit; po przerwaniu to samo polecenie pomija pliki, które są już w repo. Repo jest tworzone jako prywatne, chyba że podasz --publiczne.

Wymaga: pyarrow, huggingface_hub (uv pip install pyarrow) i zalogowania (hf auth login).

Przykłady:
  python skrypty/wyslij_na_hf.py --repo UZYTKOWNIK/NAZWA
  python skrypty/wyslij_na_hf.py --repo UZYTKOWNIK/NAZWA --tylko-pakuj
  python skrypty/wyslij_na_hf.py --repo UZYTKOWNIK/NAZWA --tylko-wyslij --publiczne
  python skrypty/wyslij_na_hf.py --repo UZYTKOWNIK/NAZWA-czyste --wyniki KATALOG --usun KATALOG/asr/usun.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent

# Kolumny wynikowe: (nazwa, pole manifestu, typ pyarrow jako tekst, typ `datasets`).
KOLUMNY = [
    ("id", "id", "string", "string"),
    ("text", "tekst", "string", "string"),
    ("sekundy", "sekundy", "float32", "float32"),
    ("glos", "glos", "string", "string"),
    ("zrodlo_id", "zrodlo_id", "string", "string"),
    ("rola", "rola", "string", "string"),
    ("tura", "tura", "int32", "int32"),
    ("temat", "temat", "string", "string"),
    ("czesc", "czesc", "int32", "int32"),
    ("jakosc", "jakosc", "float32", "float32"),
    ("pytanie", "pytanie", "bool", "bool"),
]
# Synteza z tekstu słownego (liczby/normalizuj.py): `text` to tekst słowny podany modelowi, a `text_cyfry`
# oryginał z cyframi (ten idzie do treningu).
KOLUMNA_CYFRY = ("text_cyfry", "tekst_cyfry", "string", "string")


def kolumny(wiersze: list[dict]) -> list[tuple]:
    return KOLUMNY + [KOLUMNA_CYFRY] if any("tekst_cyfry" in w for w in wiersze) else KOLUMNY


def wczytaj_manifest(katalog: Path) -> list[dict]:
    manifest = katalog / "manifest.jsonl"
    if not manifest.is_file():
        sys.exit(f"Nie ma {manifest}")
    wiersze, widziane = [], set()
    with manifest.open(encoding="utf-8") as plik:
        for nr, wiersz in enumerate(plik, 1):
            if not wiersz.strip():
                continue
            wpis = json.loads(wiersz)
            if wpis["id"] in widziane:
                sys.exit(f"{manifest}:{nr}: powtórzone id {wpis['id']}")
            widziane.add(wpis["id"])
            wiersze.append(wpis)
    # Części obok siebie, a w części fragmenty tej samej rozmowy po kolei.
    wiersze.sort(key=lambda w: (w.get("czesc") if w.get("czesc") is not None else -1, w["id"]))
    return wiersze


def zaplanuj_pliki(katalog: Path, wiersze: list[dict], mb_na_plik: int) -> list[list[dict]]:
    """Dzieli wiersze na pliki po ok. mb_na_plik; zawsze tak samo dla tego samego manifestu."""
    limit = mb_na_plik * 1_000_000
    pliki: list[list[dict]] = [[]]
    rozmiar = 0
    brakujace = []
    for wpis in wiersze:
        sciezka = katalog / wpis["audio"]
        try:
            bajty = sciezka.stat().st_size
        except FileNotFoundError:
            brakujace.append(wpis["audio"])
            continue
        if bajty == 0:
            brakujace.append(wpis["audio"] + " (pusty)")
            continue
        if pliki[-1] and rozmiar + bajty > limit:
            pliki.append([])
            rozmiar = 0
        wpis["_sciezka"] = sciezka
        wpis["_bajty"] = bajty
        pliki[-1].append(wpis)
        rozmiar += bajty
    if brakujace:
        sys.exit(f"Brakuje {len(brakujace)} plików audio z manifestu, np. {brakujace[:3]} — nie pakuję niepełnego zbioru.")
    return pliki


def wybrane_klipy(wyniki: Path, usun: list[str]) -> tuple[list[dict], int]:
    """Wpisy manifestu, które trafią do paczek (po --usun), i liczba wszystkich przed filtrem."""
    wiersze = [json.loads(w) for w in (wyniki / "manifest.jsonl").open(encoding="utf-8") if w.strip()]
    wszystkich = len(wiersze)
    if usun:
        do_usuniecia = {w.strip() for plik in usun for w in Path(plik).read_text(encoding="utf-8").splitlines() if w.strip()}
        wiersze = [w for w in wiersze if w["id"] not in do_usuniecia]
    return wiersze, wszystkich


def odcisk(wiersze: list[dict]) -> str:
    import hashlib
    return hashlib.sha1("".join(sorted(f"{w['id']}\t{w.get('audio')}\n" for w in wiersze)).encode()).hexdigest()


def sprawdz_odcisk(paczki: Path, wiersze: list[dict]) -> None:
    """Paczki z innej wersji danych (np. FLAC sprzed na_opus, inny filtr) nie mogą być wysłane ani dopakowane."""
    plik = paczki / "zrodlo.txt"
    teraz = odcisk(wiersze)
    if not list((paczki / "data").glob("*.parquet*")):
        paczki.mkdir(parents=True, exist_ok=True)
        plik.write_text(teraz + "\n", encoding="utf-8")
        return
    if plik.is_file():
        zgodne = plik.read_text(encoding="utf-8").strip() == teraz
    else:   # paczki sprzed odcisków: porównanie formatu i liczby klipów z karty
        karta_txt = (paczki / "README.md").read_text(encoding="utf-8") if (paczki / "README.md").is_file() else ""
        format_teraz = "Ogg Opus" if str(wiersze[0].get("audio", "")).endswith(".ogg") else "FLAC"
        zgodne = f" {len(wiersze)} klipów," in karta_txt and f"mono ({format_teraz})" in karta_txt
        if zgodne:
            plik.write_text(teraz + "\n", encoding="utf-8")
    if not zgodne:
        sys.exit(f"W {paczki} leżą paczki z innej wersji danych (inne klipy albo format, np. FLAC sprzed na_opus.py). "
                 f"Usuń ten katalog (rm -r {paczki}) albo podaj inny --paczki i uruchom ponownie.")


def paczki_kompletne(paczki: Path) -> int:
    """Liczba plików parquet, jeśli w katalogu leży pełny, dokończony zestaw i karta; inaczej 0."""
    dane = paczki / "data"
    if not dane.is_dir() or list(dane.glob("*.tmp")) or not (paczki / "README.md").is_file():
        return 0
    pliki = sorted(dane.glob("train-*-of-*.parquet"))
    if not pliki:
        return 0
    ile = int(pliki[0].stem.rsplit("-of-", 1)[1])
    oczekiwane = [f"train-{i:05d}-of-{ile:05d}.parquet" for i in range(ile)]
    return ile if [plik.name for plik in pliki] == oczekiwane else 0


def schemat(czestotliwosc: int, KOLUMNY: list[tuple]):
    import pyarrow as pa

    typy = {"string": pa.string(), "float32": pa.float32(), "int32": pa.int32(), "bool": pa.bool_()}
    pola = [pa.field("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))]
    pola += [pa.field(nazwa, typy[typ]) for nazwa, _, typ, _ in KOLUMNY]
    cechy = {"audio": {"sampling_rate": czestotliwosc, "_type": "Audio"}}
    cechy.update({nazwa: {"dtype": typ_hf, "_type": "Value"} for nazwa, _, _, typ_hf in KOLUMNY})
    return pa.schema(pola, metadata={"huggingface": json.dumps({"info": {"features": cechy}})})


def zapisz_plik(sciezka: Path, wpisy: list[dict], schemat_pa, KOLUMNY: list[tuple]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    kolumny = {"audio": [{"bytes": w["_sciezka"].read_bytes(), "path": w["_sciezka"].name} for w in wpisy]}
    for nazwa, pole, _, _ in KOLUMNY:
        kolumny[nazwa] = [w.get(pole) for w in wpisy]
    tabela = pa.Table.from_pydict(kolumny, schema=schemat_pa)
    tymczasowy = sciezka.with_suffix(".parquet.tmp")
    # Audio jest już skompresowane (FLAC / Opus), więc bez dodatkowej kompresji; małe grupy wierszy dla podglądu na HF.
    pq.write_table(tabela, tymczasowy, compression="none", row_group_size=100)
    tymczasowy.rename(sciezka)


def karta(repo: str, wiersze: list[dict], czestotliwosc: int, zrodlo_tekstow: str, usuniete: int = 0,
          opis_filtra: str | None = None, wybor_tekstow: str = "górne 10% według\n  `jakosc.ocena`") -> str:
    format_audio = "Ogg Opus" if str(wiersze[0]["_sciezka"]).endswith(".ogg") else "FLAC"
    if opis_filtra:   # własny opis z pliku (--opis-filtra); {usuniete}, {procent}, {godziny_usuniete} wstawiane tu
        opis_filtra = opis_filtra.strip().format(
            usuniete=usuniete, wszystkich=usuniete + len(wiersze), procent=f"{100 * usuniete / (usuniete + len(wiersze)):.0f}")
    elif usuniete:
        opis_filtra = (
            f"**Zbiór po filtrze ASR.** Usunięto {usuniete} klipów ({100 * usuniete / (usuniete + len(wiersze)):.0f}%) "
            "o najwyższym błędzie znakowym między tekstem a transkrypcją "
            "[Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) (`skrypty/czysc_asr.py`). "
            "Kolumna `text` to dalej tekst podany modelowi, a nie transkrypcja — drobne przekręcenia mogły zostać.")
    else:
        opis_filtra = (
            "**Klipy nie przeszły jeszcze filtra jakości.** Kolumna `text` to tekst podany modelowi, a nie zapis tego,\n"
            "co model faktycznie powiedział — część klipów może mieć pominięte lub przekręcone słowa.")
    godziny = sum(w["sekundy"] for w in wiersze) / 3600
    glosy = sorted({str(w.get("glos")) for w in wiersze})
    tryby = sorted({str(w.get("tryb_glosu")) for w in wiersze})
    cfg = sorted({str(w.get("cfg")) for w in wiersze})
    kroki = sorted({str(w.get("kroki")) for w in wiersze})
    rozmowy = len({w.get("zrodlo_id") for w in wiersze})
    pytania = 100 * sum(1 for w in wiersze if w.get("rola") == "user") / len(wiersze)
    przedzialy = [("do 5 s", 0, 5), ("5–15 s", 5, 15), ("15–30 s", 15, 30), ("ponad 30 s", 30, 1e9)]
    wiersze_dlugosci = "\n".join(
        f"| {nazwa} | {100 * sum(1 for w in wiersze if a <= w['sekundy'] < b) / len(wiersze):.0f}% "
        f"| {100 * sum(w['sekundy'] for w in wiersze if a <= w['sekundy'] < b) / (godziny * 3600):.0f}% |"
        for nazwa, a, b in przedzialy
    )
    czas_tematow: collections.Counter = collections.Counter()
    for w in wiersze:
        czas_tematow[str(w.get("temat"))] += w["sekundy"]
    wiersze_tematow = "\n".join(f"| {temat} | {sek / 3600:.1f} |" for temat, sek in czas_tematow.most_common())
    czesci = sorted({w.get("czesc") for w in wiersze if w.get("czesc") is not None})
    slownie = any("tekst_cyfry" in w for w in wiersze)
    opis_liczb = (
        "Liczby, daty, godziny, jednostki i skróty modelowi podano **słownie, z odmianą** (np. „w 2026 roku” → „w dwa "
        "tysiące dwudziestym szóstym roku”, „20 tys. zł” → „dwadzieścia tysięcy złotych”), bo VoxCPM2 źle czyta "
        "cyfry. Zamianę zrobił regułowy normalizator `liczby/normalizuj.py` (Morfeusz 2, bez LLM). Kolumna "
        "`text_cyfry` ma oryginalny zapis z cyframi."
        if slownie else "Liczby i skróty są w formie surowej (bez normalizacji).")
    wiersz_cyfry = "\n| `text_cyfry` | ten sam tekst w oryginalnym zapisie, z cyframi i skrótami |" if slownie else ""
    return f"""---
license: apache-2.0
language:
- pl
task_categories:
- text-to-speech
tags:
- synthetic
- voxcpm2
pretty_name: {repo.split("/")[-1]}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# {repo.split("/")[-1]}

Syntetyczna polska mowa: {godziny:.1f} h, {len(wiersze)} klipów, jeden głos, {czestotliwosc // 1000} kHz mono ({format_audio}).
Zbiór powstał do trenowania małych polskich modeli TTS (destylacja z VoxCPM2).

{opis_filtra}

## Jak powstał

- **Model:** [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) przez silnik `nano-vllm-voxcpm`;
  CFG {", ".join(cfg)}, kroki dyfuzji: {", ".join(kroki)}.
- **Głos:** {", ".join(glosy)} (tryb: {", ".join(tryby)}), jedno nagranie wzorcowe dla wszystkich klipów.
  Głos jest syntetyczny (wylosowany przez VoxCPM2), nie należy do żadnej osoby.
- **Teksty:** {zrodlo_tekstow}: rozmowy bez udziału modelu `private`, {wybor_tekstow}, oczyszczone z markdownu i pocięte na fragmenty z całych zdań. {opis_liczb}
  {rozmowy} rozmów źródłowych; {pytania:.0f}% klipów to pytania użytkownika,
  reszta to fragmenty odpowiedzi asystenta.
- **Kod:** https://github.com/adamo1139/syntezujemy_polska_mowe

## Kolumny

| kolumna | opis |
|---|---|
| `audio` | klip {format_audio}, {czestotliwosc} Hz, mono |
| `text` | tekst podany modelowi |{wiersz_cyfry}
| `id` | identyfikator klipu |
| `sekundy` | długość klipu |
| `glos` | nazwa głosu wzorcowego |
| `zrodlo_id` | id rozmowy źródłowej; dzielić zbiór na trening i test po tym polu, nie po klipach |
| `rola` | `user` (pytanie w całości) albo `assistant` (fragment odpowiedzi) |
| `tura` | numer tury w rozmowie |
| `temat` | temat rozmowy |
| `czesc` | część zbioru ({len(czesci)} rozłącznych części o tym samym rozkładzie tematów; rozmowa zawsze w jednej) |
| `jakosc` | ocena jakości rozmowy źródłowej |
| `pytanie` | czy tekst kończy się znakiem zapytania |

## Długość klipów

| długość | klipy | czas |
|---|---|---|
{wiersze_dlugosci}

## Tematy

| temat | godziny |
|---|---|
{wiersze_tematow}

## Użycie

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train", streaming=True)
klip = next(iter(ds))
print(klip["text"], klip["audio"]["sampling_rate"])
```
"""


def wyslij_partiami(api, repo: str, paczki: Path, na_commit: int, prob: int = 5) -> None:
    """Wysyła kartę, a potem pliki parquet partiami: każda partia to osobny commit.

    Po przerwaniu to samo polecenie pomija pliki, które są już w repo, więc traci się najwyżej jedną partię.
    """
    w_repo = set(api.list_repo_files(repo, repo_type="dataset"))
    wszystkie = sorted(plik.name for plik in (paczki / "data").glob("*.parquet"))
    brakujace = [nazwa for nazwa in wszystkie if f"data/{nazwa}" not in w_repo]
    print(f"Plików parquet: {len(wszystkie)}, już w repo: {len(wszystkie) - len(brakujace)}, do wysłania: {len(brakujace)}")

    def wyslij(wzorce: list[str], opis: str) -> None:
        for proba in range(1, prob + 1):
            try:
                api.upload_folder(
                    repo_id=repo, repo_type="dataset", folder_path=str(paczki), allow_patterns=wzorce, commit_message=opis
                )
                return
            except KeyboardInterrupt:
                raise
            except Exception as blad:
                if proba == prob:
                    raise
                czekaj = 30 * proba
                print(f"  błąd ({blad!r}); próba {proba}/{prob}, ponawiam za {czekaj} s", flush=True)
                time.sleep(czekaj)

    wyslij(["README.md"], "Karta datasetu")
    t0 = time.time()
    for i in range(0, len(brakujace), na_commit):
        partia = brakujace[i : i + na_commit]
        wyslij([f"data/{nazwa}" for nazwa in partia], f"Dane: {partia[0]} – {partia[-1]}")
        zrobione = i + len(partia)
        print(f"  w repo {len(wszystkie) - len(brakujace) + zrobione}/{len(wszystkie)} plików  [{time.time() - t0:.0f} s]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="repo datasetu na HF, np. uzytkownik/nazwa")
    p.add_argument("--wyniki", default=str(KATALOG_PROJEKTU / "wyniki" / "poziomka"), help="katalog z manifest.jsonl i audio/")
    p.add_argument("--paczki", help="katalog na pliki parquet (domyślnie <wyniki>_parquet)")
    p.add_argument("--usun", action="append", default=[],
                   help="plik z id klipów do pominięcia (usun.txt z czysc_asr.py): wysyłka zbioru po filtrze")
    p.add_argument("--mb-na-plik", type=int, default=500)
    p.add_argument("--zrodlo-tekstow", default="[cpral/polskie-sprawy-v8-jakosc50](https://huggingface.co/datasets/cpral/polskie-sprawy-v8-jakosc50)")
    p.add_argument("--opis-filtra", help="plik markdown z opisem filtra do karty (zamiast domyślnego zdania o ASR); "
                   "{usuniete} i {procent} są wstawiane")
    p.add_argument("--wybor-tekstow", default="górne 10% według\n  `jakosc.ocena`",
                   help="do karty: które rozmowy wybrano ze źródła tekstów")
    p.add_argument("--plikow-na-commit", type=int, default=10, help="ile plików parquet w jednej partii wysyłki (jeden commit)")
    p.add_argument("--publiczne", action="store_true", help="utwórz repo jako publiczne (domyślnie prywatne)")
    p.add_argument("--tylko-pakuj", action="store_true", help="zapisz parquet i README, nic nie wysyłaj")
    p.add_argument("--tylko-wyslij", action="store_true", help="wyślij to, co już leży w katalogu paczek")
    args = p.parse_args()
    if args.tylko_pakuj and args.tylko_wyslij:
        p.error("--tylko-pakuj i --tylko-wyslij wykluczają się")

    wyniki = Path(args.wyniki)
    # zbiór po filtrze ma własny katalog paczek, żeby nie mieszać się z surowym
    paczki = Path(args.paczki) if args.paczki else wyniki.with_name(wyniki.name + ("_czyste_parquet" if args.usun else "_parquet"))

    if not args.tylko_wyslij:
        sprawdz_odcisk(paczki, wybrane_klipy(wyniki, args.usun)[0])
    gotowe = paczki_kompletne(paczki)
    if gotowe and not args.tylko_wyslij:
        print(f"Dane już spakowane: {gotowe} plików parquet i karta w {paczki}, z tych samych klipów — nie pakuję od nowa.")
        print("  (żeby spakować ponownie, usuń ten katalog)")
    elif not args.tylko_wyslij:
        import soundfile as sf

        t0 = time.time()
        wiersze = wczytaj_manifest(wyniki)
        wszystkich = len(wiersze)
        odrzucone_wczesniej = wyniki / "manifest.odrzucone.jsonl"   # klipy usunięte już przez na_opus.py --pomin
        if odrzucone_wczesniej.is_file():
            ile = sum(1 for w in odrzucone_wczesniej.open(encoding="utf-8") if w.strip())
            wszystkich += ile
            print(f"{ile} klipów usuniętych wcześniej (manifest.odrzucone.jsonl) liczę do odfiltrowanych w karcie")
        if args.usun:
            do_usuniecia = {w.strip() for plik in args.usun for w in Path(plik).read_text(encoding="utf-8").splitlines() if w.strip()}
            wiersze = [w for w in wiersze if w["id"] not in do_usuniecia]
            print(f"Filtr: pomijam {wszystkich - len(wiersze)} z {wszystkich} klipów (lista: {len(do_usuniecia)} id)")
        pliki = zaplanuj_pliki(wyniki, wiersze, args.mb_na_plik)
        razem = sum(w["_bajty"] for w in wiersze)
        czestotliwosc = sf.info(str(wiersze[0]["_sciezka"])).samplerate
        katalog_danych = paczki / "data"
        katalog_danych.mkdir(parents=True, exist_ok=True)
        nazwy = [katalog_danych / f"train-{i:05d}-of-{len(pliki):05d}.parquet" for i in range(len(pliki))]
        obce = sorted(set(katalog_danych.glob("*.parquet*")) - set(nazwy))
        if obce:
            sys.exit(f"W {katalog_danych} leżą pliki z innego podziału (np. {obce[0].name}) — usuń katalog i zacznij od nowa.")
        do_zapisu = [(nazwa, wpisy) for nazwa, wpisy in zip(nazwy, pliki) if not nazwa.exists()]
        potrzeba = sum(w["_bajty"] for _, wpisy in do_zapisu for w in wpisy)
        wolne = shutil.disk_usage(paczki).free
        print(
            f"Klipów: {len(wiersze)}, {sum(w['sekundy'] for w in wiersze) / 3600:.1f} h, {razem / 1e9:.1f} GB audio "
            f"-> {len(pliki)} plików parquet w {katalog_danych} (gotowych: {len(pliki) - len(do_zapisu)})"
        )
        if potrzeba * 1.02 > wolne:
            sys.exit(f"Za mało miejsca: potrzeba ok. {potrzeba / 1e9:.1f} GB, wolne {wolne / 1e9:.1f} GB. Podaj --paczki na innym dysku.")
        kol = kolumny(wiersze)
        schemat_pa = schemat(czestotliwosc, kol)
        for nr, (nazwa, wpisy) in enumerate(do_zapisu, 1):
            zapisz_plik(nazwa, wpisy, schemat_pa, kol)
            if nr % 10 == 0 or nr == len(do_zapisu):
                print(f"  zapisano {nr}/{len(do_zapisu)} plików  [{time.time() - t0:.0f} s]", flush=True)
        (paczki / "README.md").write_text(karta(args.repo, wiersze, czestotliwosc, args.zrodlo_tekstow, wszystkich - len(wiersze),
            Path(args.opis_filtra).read_text(encoding="utf-8") if args.opis_filtra else None, args.wybor_tekstow), encoding="utf-8")
        print(f"Karta datasetu: {paczki / 'README.md'}")

    if args.tylko_pakuj:
        print("Nic nie wysłano (--tylko-pakuj).")
        return

    if not (paczki / "data").is_dir() or not any((paczki / "data").glob("*.parquet")):
        sys.exit(f"W {paczki} nie ma plików parquet — najpierw pakowanie.")
    if list((paczki / "data").glob("*.tmp")):
        sys.exit(f"W {paczki / 'data'} leży niedokończony plik .tmp — uruchom pakowanie jeszcze raz.")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        kto = api.whoami()["name"]
    except Exception:
        sys.exit("Brak zalogowania do HF: uruchom `hf auth login` (token z prawem zapisu).")
    print(f"Wysyłam jako {kto} do datasetu {args.repo} ({'publiczny' if args.publiczne else 'prywatny'})...")
    api.create_repo(args.repo, repo_type="dataset", private=not args.publiczne, exist_ok=True)
    if args.publiczne:
        # create_repo nie zmienia widoczności repo, które już istnieje (np. utworzonego wcześniej jako prywatne).
        api.update_repo_settings(args.repo, repo_type="dataset", private=False)
    wyslij_partiami(api, args.repo, paczki, args.plikow_na_commit)
    print(f"Gotowe: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
