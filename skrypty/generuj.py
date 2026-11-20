#!/usr/bin/env python3
"""Synteza wsadowa VoxCPM2 przez nano-vllm-voxcpm na wielu kartach.

Jeden proces sterujący; silnik sam uruchamia po jednym procesie na kartę
i kieruje każde żądanie do karty z najmniejszą liczbą zadań w toku.

Wybór kart (numery jak w `nvidia-smi`):
  --pomin 0,2        wszystkie oprócz 0 i 2
  --karty 1,3,4      tylko te
  --karty wszystkie  wszystkie, bez pytania
Bez żadnej z tych flag skrypt pokazuje listę kart i pyta, które pominąć.

Przykłady:
  python skrypty/generuj.py --lista-kart
  python skrypty/generuj.py --tekst "Dzień dobry, to jest próba." --karty 1
  python skrypty/generuj.py --teksty teksty.jsonl --wyjscie wyniki/proba --pomin 0,2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- karty


def wykryj_karty() -> list[dict]:
    """Lista kart z nvidia-smi: numer, nazwa, pamięć zajęta i całkowita (MiB)."""
    try:
        wynik = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as blad:
        sys.exit(f"Nie udało się uruchomić nvidia-smi: {blad}")
    karty = []
    for wiersz in wynik.strip().splitlines():
        numer, nazwa, zajete, calosc = [pole.strip() for pole in wiersz.split(",")]
        karty.append({"numer": int(numer), "nazwa": nazwa, "zajete": int(zajete), "calosc": int(calosc)})
    if not karty:
        sys.exit("nvidia-smi nie pokazuje żadnej karty.")
    return karty


def pokaz_karty(karty: list[dict]) -> None:
    print("Karty widoczne w systemie:")
    for karta in karty:
        uwaga = "  <- zajęta pamięć" if karta["zajete"] > 1024 else ""
        print(f"  [{karta['numer']}] {karta['nazwa']}  {karta['zajete']}/{karta['calosc']} MiB{uwaga}")


def parsuj_numery(tekst: str, dostepne: set[int], flaga: str) -> list[int]:
    numery = []
    for kawalek in tekst.replace(" ", "").split(","):
        if not kawalek:
            continue
        if not kawalek.isdigit():
            sys.exit(f"{flaga}: „{kawalek}” to nie numer karty.")
        numer = int(kawalek)
        if numer not in dostepne:
            sys.exit(f"{flaga}: nie ma karty {numer}. Dostępne: {sorted(dostepne)}")
        if numer not in numery:
            numery.append(numer)
    return numery


def wybierz_karty(karty: list[dict], arg_karty: str | None, arg_pomin: str | None) -> list[int]:
    dostepne = {karta["numer"] for karta in karty}
    if arg_karty and arg_pomin:
        sys.exit("Podaj --karty albo --pomin, nie oba naraz.")
    if arg_karty:
        if arg_karty.strip().lower() in ("wszystkie", "all"):
            wybrane = sorted(dostepne)
        else:
            wybrane = sorted(parsuj_numery(arg_karty, dostepne, "--karty"))
    elif arg_pomin is not None:
        pominiete = set(parsuj_numery(arg_pomin, dostepne, "--pomin"))
        wybrane = sorted(dostepne - pominiete)
    else:
        pokaz_karty(karty)
        if not sys.stdin.isatty():
            sys.exit("Brak terminala do zadania pytania: podaj --karty albo --pomin.")
        odpowiedz = input("Które karty pominąć? (numery po przecinku, Enter = żadnej): ")
        pominiete = set(parsuj_numery(odpowiedz, dostepne, "odpowiedź"))
        wybrane = sorted(dostepne - pominiete)
    if not wybrane:
        sys.exit("Po pominięciu nie została żadna karta.")
    return wybrane


# ---------------------------------------------------------------- teksty


def wczytaj_teksty(args: argparse.Namespace) -> list[dict]:
    """Zwraca listę {id, tekst, ...}. JSONL: pola `id` i `tekst`; .txt: jeden tekst na wiersz."""
    if args.tekst:
        return [{"id": f"proba_{i:03d}", "tekst": tekst} for i, tekst in enumerate(args.tekst)]
    sciezka = Path(args.teksty)
    pozycje = []
    with sciezka.open(encoding="utf-8") as plik:
        for nr, wiersz in enumerate(plik):
            wiersz = wiersz.strip()
            if not wiersz:
                continue
            if sciezka.suffix == ".jsonl":
                pozycja = json.loads(wiersz)
                if "tekst" not in pozycja:
                    sys.exit(f"{sciezka}:{nr + 1}: brak pola „tekst”.")
            else:
                pozycja = {"tekst": wiersz}
            if "id" not in pozycja:
                skrot = hashlib.sha1(pozycja["tekst"].encode("utf-8")).hexdigest()[:12]
                pozycja["id"] = f"{nr:07d}_{skrot}"
            pozycja["id"] = str(pozycja["id"])
            pozycje.append(pozycja)
    if args.czesc is not None:
        if any("czesc" not in pozycja for pozycja in pozycje):
            sys.exit(f"--czesc: {sciezka} nie ma pola „czesc” (plik z wybierz_teksty.py).")
        pozycje = [pozycja for pozycja in pozycje if pozycja["czesc"] == args.czesc]
        if not pozycje:
            sys.exit(f"--czesc {args.czesc}: nie ma takiej części w {sciezka}.")
    if args.limit:
        pozycje = pozycje[: args.limit]
    return pozycje


def juz_zrobione(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    # Po twardym przerwaniu (kill -9, zanik zasilania) ostatni wiersz może być urwany: obcinamy go,
    # żeby następny wpis nie skleił się z nim w jeden zepsuty wiersz.
    dane = manifest.read_bytes()
    if dane and not dane.endswith(b"\n"):
        ostatni_koniec = dane.rfind(b"\n") + 1
        print(f"Manifest kończył się urwanym wierszem ({len(dane) - ostatni_koniec} bajtów) — obcięty.")
        with manifest.open("r+b") as plik:
            plik.truncate(ostatni_koniec)
    gotowe = set()
    with manifest.open(encoding="utf-8") as plik:
        for nr, wiersz in enumerate(plik, 1):
            try:
                gotowe.add(json.loads(wiersz)["id"])
            except (json.JSONDecodeError, KeyError):
                sys.exit(f"{manifest}:{nr}: zepsuty wiersz w środku pliku — popraw albo usuń go ręcznie.")
    return gotowe


# ---------------------------------------------------------------- synteza


async def syntezuj(args: argparse.Namespace, pozycje: list[dict], liczba_kart: int) -> None:
    import numpy as np
    import soundfile as sf
    from nanovllm_voxcpm import VoxCPM

    wyjscie = Path(args.wyjscie)
    katalog_audio = wyjscie / "audio"
    katalog_audio.mkdir(parents=True, exist_ok=True)
    manifest = wyjscie / "manifest.jsonl"
    plik_bledow = wyjscie / "bledy.jsonl"

    if not Path(args.model).expanduser().is_dir():
        hf_home = os.environ.get("HF_HOME")
        print(f"Model z pamięci HF: {hf_home if hf_home else '~/.cache/huggingface (HF_HOME nieustawione)'}")
    print(f"Ładowanie modelu {args.model} na {liczba_kart} kartach...")
    t_start = time.time()
    serwer = VoxCPM.from_pretrained(
        model=args.model,
        inference_timesteps=args.kroki,
        max_num_batched_tokens=args.max_tokenow,
        max_num_seqs=args.rownolegle,
        max_model_len=4096,
        gpu_memory_utilization=args.pamiec,
        devices=list(range(liczba_kart)),  # numery po zawężeniu CUDA_VISIBLE_DEVICES
    )
    try:
        await serwer.wait_for_ready()
        info = await serwer.get_model_info()
        czestotliwosc = int(info["sample_rate"])
        print(f"Gotowe po {time.time() - t_start:.0f} s. Częstotliwość wyjściowa: {czestotliwosc} Hz.")

        # Głos: latenty liczone raz i wysyłane z każdym żądaniem.
        parametry_glosu: dict = {}
        if args.glos:
            sciezka_glosu = Path(args.glos)
            latenty = await serwer.encode_latents(sciezka_glosu.read_bytes(), sciezka_glosu.suffix.lstrip("."))
            if args.tryb in ("referencja", "oba"):
                parametry_glosu["ref_audio_latents"] = latenty
            if args.tryb in ("prompt", "oba"):
                plik_tekstu = Path(args.glos_tekst) if args.glos_tekst else sciezka_glosu.with_suffix(".txt")
                if not plik_tekstu.exists():
                    sys.exit(f"Tryb „{args.tryb}” wymaga transkrypcji głosu: brak {plik_tekstu}")
                parametry_glosu["prompt_latents"] = latenty
                parametry_glosu["prompt_text"] = plik_tekstu.read_text(encoding="utf-8").strip()
            nazwa_glosu = sciezka_glosu.stem
        else:
            nazwa_glosu = None
            print("Bez głosu wzorcowego: każdy klip dostanie losowego mówcę.")

        semafor = asyncio.Semaphore(args.rownolegle * liczba_kart)
        blokada_zapisu = asyncio.Lock()
        stan = {"sekundy": 0.0, "gotowe": 0, "bledy": 0, "bledy_z_rzedu": 0}
        przerwij = asyncio.Event()
        t_synteza = time.time()

        async def jeden(pozycja: dict) -> None:
            async with semafor:
                if przerwij.is_set():
                    return  # nie zapisujemy jako błąd: tekst zostanie zrobiony po wznowieniu
                t0 = time.time()
                try:
                    kawalki = []
                    async for kawalek in serwer.generate(
                        target_text=pozycja["tekst"],
                        max_generate_length=args.max_krokow,
                        temperature=args.temperatura,
                        cfg_value=args.cfg,
                        seed=args.ziarno,
                        **parametry_glosu,
                    ):
                        kawalki.append(kawalek)
                    fala = np.concatenate(kawalki, axis=0)
                    sciezka = katalog_audio / f"{pozycja['id']}.flac"
                    await asyncio.to_thread(sf.write, sciezka, fala, czestotliwosc, format="FLAC", subtype="PCM_16")
                except Exception as blad:  # jeden zły tekst nie zatrzymuje całości
                    stan["bledy"] += 1
                    stan["bledy_z_rzedu"] += 1
                    if stan["bledy_z_rzedu"] >= args.max_bledow_z_rzedu and not przerwij.is_set():
                        przerwij.set()
                        print(
                            f"\nPRZERYWAM: {stan['bledy_z_rzedu']} błędów z rzędu — najpewniej padł proces którejś karty "
                            f"(ostatni błąd: {blad!r}). Sprawdź nvidia-smi, popraw --rownolegle/--pamiec albo pomiń kartę "
                            "i uruchom to samo polecenie jeszcze raz.",
                            flush=True,
                        )
                    async with blokada_zapisu:
                        with plik_bledow.open("a", encoding="utf-8") as plik:
                            plik.write(json.dumps({"id": pozycja["id"], "blad": repr(blad)}, ensure_ascii=False) + "\n")
                    return
                sekundy = len(fala) / czestotliwosc
                wpis = {
                    **pozycja,
                    "audio": str(sciezka.relative_to(wyjscie)),
                    "sekundy": round(sekundy, 3),
                    "glos": nazwa_glosu,
                    "tryb_glosu": args.tryb if nazwa_glosu else None,
                    "cfg": args.cfg,
                    "kroki": args.kroki,
                    "czas_generacji": round(time.time() - t0, 3),
                }
                async with blokada_zapisu:
                    with manifest.open("a", encoding="utf-8") as plik:
                        plik.write(json.dumps(wpis, ensure_ascii=False) + "\n")
                stan["sekundy"] += sekundy
                stan["gotowe"] += 1
                stan["bledy_z_rzedu"] = 0
                if stan["gotowe"] % args.co_ile == 0:
                    minelo = time.time() - t_synteza
                    print(
                        f"  {stan['gotowe']}/{len(pozycje)} klipów, {stan['sekundy'] / 3600:.2f} h audio, "
                        f"{stan['sekundy'] / minelo:.0f}x czas rzeczywisty, błędy: {stan['bledy']}",
                        flush=True,
                    )

        await asyncio.gather(*(jeden(pozycja) for pozycja in pozycje))

        minelo = time.time() - t_synteza
        print(
            f"\nKoniec: {stan['gotowe']} klipów, {stan['sekundy'] / 3600:.3f} h audio w {minelo:.0f} s "
            f"= {stan['sekundy'] / max(minelo, 1e-9):.0f}x czas rzeczywisty na {liczba_kart} kartach "
            f"({stan['sekundy'] / max(minelo, 1e-9) / liczba_kart:.0f}x na kartę). Błędy: {stan['bledy']}."
        )
        print(f"Manifest: {manifest}")
        if przerwij.is_set():
            print(f"Przebieg przerwany; zostało {len(pozycje) - stan['gotowe'] - stan['bledy']} tekstów. Wznowienie: to samo polecenie.")
        if stan["bledy"]:
            print(f"Błędy:    {plik_bledow}")
    finally:
        await serwer.stop()


# ---------------------------------------------------------------- start


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("karty")
    g.add_argument("--karty", help="numery kart po przecinku albo „wszystkie”")
    g.add_argument("--pomin", help="numery kart do pominięcia, po przecinku")
    g.add_argument("--lista-kart", action="store_true", help="pokaż karty i zakończ")

    g = p.add_argument_group("wejście i wyjście")
    g.add_argument("--teksty", help="plik .jsonl (pola id, tekst) albo .txt (jeden tekst na wiersz)")
    g.add_argument("--tekst", action="append", help="pojedynczy tekst (można powtórzyć) zamiast pliku")
    g.add_argument("--czesc", type=int, help="weź tylko teksty z tej części (pole „czesc” z wybierz_teksty.py)")
    g.add_argument("--limit", type=int, help="weź tylko N pierwszych tekstów (po --czesc)")
    g.add_argument("--wyjscie", default="wyniki/proba", help="katalog na audio/ i manifest.jsonl")

    g = p.add_argument_group("model i głos")
    g.add_argument("--model", default="openbmb/VoxCPM2", help="repo HF albo katalog lokalny")
    g.add_argument("--glos", default=str(KATALOG_PROJEKTU / "glosy" / "poziomka.wav"), help="nagranie wzorcowe; „brak” = losowy mówca")
    g.add_argument("--glos-tekst", help="transkrypcja nagrania (domyślnie plik .txt obok nagrania)")
    g.add_argument(
        "--tryb",
        choices=("referencja", "prompt", "oba"),
        default="referencja",
        help="referencja = wzorzec barwy; prompt = kontynuacja nagrania z transkrypcją; oba = jedno i drugie",
    )

    g = p.add_argument_group("generowanie")
    g.add_argument("--cfg", type=float, default=2.0)
    g.add_argument("--kroki", type=int, default=10, help="kroki dyfuzji")
    g.add_argument("--temperatura", type=float, default=1.0)
    g.add_argument("--ziarno", type=int, help="stałe ziarno (to samo dla każdego tekstu)")
    g.add_argument("--max-krokow", type=int, default=700, help="limit kroków generacji na klip; 1 krok = 0,16 s audio, 700 = 112 s")

    g = p.add_argument_group("wydajność")
    g.add_argument("--rownolegle", type=int, default=16, help="równoległe żądania NA KARTĘ")
    g.add_argument("--pamiec", type=float, default=0.8, help="ułamek VRAM dla silnika (gpu_memory_utilization)")
    g.add_argument("--max-tokenow", type=int, default=16384, help="max_num_batched_tokens")
    g.add_argument("--co-ile", type=int, default=100, help="co ile klipów wypisać postęp")
    g.add_argument("--max-bledow-z-rzedu", type=int, default=50, help="po tylu błędach pod rząd przerwij przebieg")
    args = p.parse_args()

    karty = wykryj_karty()
    if args.lista_kart:
        pokaz_karty(karty)
        return
    if not args.teksty and not args.tekst:
        p.error("podaj --teksty PLIK albo --tekst \"...\"")
    if args.glos and args.glos.lower() == "brak":
        args.glos = None
    if args.glos and not Path(args.glos).exists():
        sys.exit(f"Nie ma pliku głosu: {args.glos}")

    wybrane = wybierz_karty(karty, args.karty, args.pomin)
    pominiete = sorted({karta["numer"] for karta in karty} - set(wybrane))
    print(f"Używam kart: {wybrane}" + (f"  (pominięte: {pominiete})" if pominiete else ""))
    for karta in karty:
        if karta["numer"] in wybrane and karta["zajete"] > 1024:
            print(f"  Uwaga: karta {karta['numer']} ma zajęte {karta['zajete']} MiB — możliwy brak pamięci.")

    # Musi być ustawione przed importem torch. PCI_BUS_ID = ta sama numeracja co w nvidia-smi.
    # Pominiętych kart procesy w ogóle nie zobaczą.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(numer) for numer in wybrane)

    pozycje = wczytaj_teksty(args)
    identyfikatory = [pozycja["id"] for pozycja in pozycje]
    if len(set(identyfikatory)) != len(identyfikatory):
        sys.exit("Powtarzające się id w tekstach — nadpisałyby sobie pliki audio.")
    gotowe = juz_zrobione(Path(args.wyjscie) / "manifest.jsonl")
    do_zrobienia = [pozycja for pozycja in pozycje if pozycja["id"] not in gotowe]
    print(f"Teksty: {len(pozycje)}, już w manifeście: {len(pozycje) - len(do_zrobienia)}, do zrobienia: {len(do_zrobienia)}")
    if not do_zrobienia:
        return

    asyncio.run(syntezuj(args, do_zrobienia, len(wybrane)))


if __name__ == "__main__":
    main()
