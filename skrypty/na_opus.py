#!/usr/bin/env python3
"""Przekodowanie klipów FLAC z katalogu syntezy na Ogg Opus (domyślnie 128 kbit/s), na wszystkich rdzeniach.

Kolejność (bezpieczna przy przerwaniu w dowolnym momencie):
  1. każdy klip z manifestu, którego audio to .flac, jest kodowany do .ogg obok (ta sama nazwa); gotowy .ogg
     o zgodnej długości jest pomijany, więc to samo polecenie wznawia pracę;
  2. każdy .ogg jest sprawdzany: da się odczytać, ta sama częstotliwość i długość co FLAC (±10 ms);
  3. dopiero gdy WSZYSTKIE się zgadzają: kopia manifestu jako manifest.przed_opus.jsonl, nowy manifest
     (audio -> .ogg, format „opus”, kbps) zapisany przez plik tymczasowy;
  4. usunięcie plików FLAC (chyba że --zostaw-flac).

--pomin LISTA (np. utmos/usun_razem.txt po czysc_asr.py i czysc_utmos.py): klipy z listy nie są przekodowywane;
ich wpisy trafiają z manifestu do manifest.odrzucone.jsonl, a ich pliki (FLAC i ewentualny .ogg) są usuwane
(chyba że --zostaw-flac). Dzięki temu najpierw filtr, potem konwersja tylko tego, co zostaje.

Nie uruchamiać, gdy generuj.py jeszcze pisze do tego samego katalogu.

  python skrypty/na_opus.py wyniki/partia2                # wszystkie rdzenie, 128 kbit/s
  python skrypty/na_opus.py wyniki/partia2 --procesy 20 --zostaw-flac
  python skrypty/na_opus.py wyniki/partia2 --pomin wyniki/partia2/utmos/usun_razem.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def koduj(zadanie: tuple[str, str, int]) -> tuple[str, bool, str]:
    """(flac, ogg, kbps) -> (flac, ok, opis). Koduje, jeśli trzeba, i sprawdza długość."""
    import soundfile as sf
    flac, ogg, kbps = zadanie
    try:
        i_f = sf.info(flac)
        dl_f = i_f.frames / i_f.samplerate
        if os.path.exists(ogg):
            try:
                i_o = sf.info(ogg)
                if i_o.samplerate == i_f.samplerate and abs(i_o.frames / i_o.samplerate - dl_f) < 0.01:
                    return flac, True, "było"
            except Exception:
                pass
        x, sr = sf.read(flac, dtype="float32")
        tmp = ogg + ".tmp"
        sf.write(tmp, x, sr, format="OGG", subtype="OPUS", compression_level=min(1.0, max(0.0, (256 - kbps) / 250)))
        i_o = sf.info(tmp)
        if i_o.samplerate != sr or abs(i_o.frames / i_o.samplerate - dl_f) >= 0.01:
            os.remove(tmp)
            return flac, False, f"długość {i_o.frames / i_o.samplerate:.3f} s zamiast {dl_f:.3f} s"
        os.replace(tmp, ogg)
        return flac, True, "nowy"
    except Exception as e:
        return flac, False, f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("katalog", help="katalog syntezy (manifest.jsonl + audio/)")
    p.add_argument("--kbps", type=int, default=128)
    p.add_argument("--procesy", type=int, default=os.cpu_count())
    p.add_argument("--zostaw-flac", action="store_true", help="nie usuwaj plików FLAC (ani plików odrzuconych klipów)")
    p.add_argument("--pomin", action="append", default=[], help="lista id (jedno na wiersz) do odrzucenia zamiast konwersji")
    a = p.parse_args()

    k = Path(a.katalog)
    plik_manifestu = k / "manifest.jsonl"
    wiersze = [json.loads(l) for l in plik_manifestu.open(encoding="utf-8") if l.strip()]
    pomin = {l.strip() for f in a.pomin for l in open(f, encoding="utf-8") if l.strip()}
    odrzucone = [w for w in wiersze if w.get("id") in pomin]
    wiersze = [w for w in wiersze if w.get("id") not in pomin]
    if pomin:
        print(f"--pomin: {len(pomin)} id na liście, z czego w manifeście {len(odrzucone)} — bez konwersji, do usunięcia", flush=True)
    do_zrobienia = [w for w in wiersze if "audio" in w and w["audio"].endswith(".flac")]
    juz_opus = sum(1 for w in wiersze if "audio" in w and w["audio"].endswith(".ogg"))
    print(f"{len(wiersze)} wpisów: FLAC do przekodowania {len(do_zrobienia)}, już w Opus {juz_opus}; "
          f"{a.procesy} procesów, {a.kbps} kbit/s", flush=True)
    if not do_zrobienia and not odrzucone:
        print("Nic do zrobienia.")
        return

    zadania = [(str(k / w["audio"]), str(k / w["audio"])[:-5] + ".ogg", a.kbps) for w in do_zrobienia]
    sek = sum(float(w.get("sekundy") or 0) for w in do_zrobienia)
    t0, gotowe, nowe, bledy = time.time(), 0, 0, []
    with ProcessPoolExecutor(max(1, a.procesy)) as pula:
        for flac, ok, opis in pula.map(koduj, zadania, chunksize=64):
            gotowe += 1
            if not ok:
                bledy.append((flac, opis))
            elif opis == "nowy":
                nowe += 1
            if gotowe % 5000 == 0 or gotowe == len(zadania):
                dt = time.time() - t0
                zostalo = (len(zadania) - gotowe) * dt / gotowe
                print(f"  {gotowe}/{len(zadania)} ({nowe} nowych, błędów {len(bledy)}) [{dt / 60:.1f} min, "
                      f"zostało ok. {zostalo / 60:.0f} min]", flush=True)
    if bledy:
        for flac, opis in bledy[:20]:
            print(f"  BŁĄD {flac}: {opis}")
        sys.exit(f"{len(bledy)} klipów nie przekodowało się poprawnie — manifest i pliki FLAC bez zmian. "
                 "Sprawdź pliki i uruchom ponownie (gotowe .ogg zostaną pominięte).")

    # wszystkie .ogg zgodne: podmiana manifestu (przez plik tymczasowy), kopia starego
    kopia = k / "manifest.przed_opus.jsonl"
    if not kopia.exists():
        shutil.copy2(plik_manifestu, kopia)
    for w in do_zrobienia:
        w["audio"] = w["audio"][:-5] + ".ogg"
        w["format"], w["kbps"] = "opus", a.kbps
    if odrzucone:   # najpierw zapis odrzuconych (dopisywany), potem nowy manifest bez nich
        with (k / "manifest.odrzucone.jsonl").open("a", encoding="utf-8") as f:
            for w in odrzucone:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    tmp = plik_manifestu.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for w in wiersze:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, plik_manifestu)
    print(f"Manifest zaktualizowany ({len(do_zrobienia)} klipów -> .ogg, {len(odrzucone)} odrzuconych "
          f"-> manifest.odrzucone.jsonl); kopia: {kopia}")

    if not a.zostaw_flac:
        zwolnione = 0
        for flac, _, _ in zadania:
            try:
                zwolnione += os.path.getsize(flac)
                os.remove(flac)
            except FileNotFoundError:
                pass
        for w in odrzucone:
            sciezka = str(k / w["audio"])
            for plik in {sciezka, sciezka.rsplit(".", 1)[0] + ".ogg", sciezka.rsplit(".", 1)[0] + ".flac"}:
                try:
                    zwolnione += os.path.getsize(plik)
                    os.remove(plik)
                except FileNotFoundError:
                    pass
        print(f"Usunięto {len(zadania)} plików FLAC i pliki {len(odrzucone)} odrzuconych klipów, "
              f"zwolniono {zwolnione / 1e9:.1f} GB.")
    print(f"Gotowe: {sek / 3600:.1f} h audio w {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
