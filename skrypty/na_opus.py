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
  python skrypty/na_opus.py wyniki/partia2 --gpu 0,1,2,3 --pomin wyniki/partia2/utmos/usun_razem.txt

--gpu KARTY: kodowanie na kartach NVIDIA przez opus_gpu (https://github.com/adamo1139/opus_gpu; fixed-point
libopus na CUDA, CBR, partie po --partia klipów; po jednym procesie na kartę, FLAC czytane w wątkach). Wymaga
48 kHz; pliki sprawdzane tak samo (soundfile, długość ±10 ms). Bez --gpu: libsndfile na CPU (VBR).
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


def _gpu_proces(zadania: list[tuple[str, str, int]], karta: str, partia: int, zlozonosc: int, watki: int,
                kolejka) -> list[tuple[str, bool, str]]:
    """Jeden proces na kartę: czyta FLAC w wątkach, koduje partiami na GPU, zapisuje .ogg przez .tmp i sprawdza."""
    os.environ["CUDA_VISIBLE_DEVICES"] = karta   # przed importem cupy
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import soundfile as sf
    from opus_gpu import encode_many

    wyniki = []

    def wczytaj(z):
        flac, ogg, _ = z
        x, sr = sf.read(flac, dtype="int16")
        if sr != 48000:
            raise ValueError(f"{sr} Hz (opus_gpu koduje tylko 48 kHz)")
        return np.ascontiguousarray(x if x.ndim == 1 else x[:, :2])

    def zapisz(z, dane, ramki):
        flac, ogg, _ = z
        tmp = ogg + ".tmp"
        with open(tmp, "wb") as f:
            f.write(dane)
        i_o = sf.info(tmp)
        if i_o.samplerate != 48000 or abs(i_o.frames - ramki) > 480:
            os.remove(tmp)
            return flac, False, f"długość {i_o.frames} próbek zamiast {ramki}"
        os.replace(tmp, ogg)
        return flac, True, "nowy"

    with ThreadPoolExecutor(watki) as pula:
        partie = [zadania[i:i + partia] for i in range(0, len(zadania), partia)]
        nastepna = pula.map(lambda z: _bezpiecznie(wczytaj, z), partie[0]) if partie else None
        for nr, biezaca_z in enumerate(partie):
            biezaca = list(nastepna)
            if nr + 1 < len(partie):   # następna partia czyta się w tle, gdy ta się koduje
                nastepna = pula.map(lambda z: _bezpiecznie(wczytaj, z), partie[nr + 1])
            dobre = [(z, pcm) for z, pcm in zip(biezaca_z, biezaca) if not isinstance(pcm, str)]
            wyniki += [(z[0], False, pcm) for z, pcm in zip(biezaca_z, biezaca) if isinstance(pcm, str)]
            ogg = encode_many((pcm for _, pcm in dobre), bitrate=zadania[0][2] * 1000, complexity=zlozonosc,
                              batch_size=len(dobre) or 1)
            zapisy = [pula.submit(zapisz, z, dane, len(pcm)) for (z, pcm), dane in zip(dobre, ogg)]
            wyniki += [_wynik(f) for f in zapisy]
            kolejka.put(len(biezaca_z))
    return wyniki


def _bezpiecznie(funkcja, z):
    try:
        return funkcja(z)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _wynik(przyszly):
    try:
        return przyszly.result()
    except Exception as e:
        return "?", False, f"{type(e).__name__}: {e}"


def gotowy(flac: str, ogg: str) -> bool:
    """Czy .ogg już jest i ma tę samą długość co FLAC (wznawianie)."""
    import soundfile as sf
    try:
        i_f, i_o = sf.info(flac), sf.info(ogg)
        return i_o.samplerate == i_f.samplerate and abs(i_o.frames / i_o.samplerate - i_f.frames / i_f.samplerate) < 0.01
    except Exception:
        return False


def koduj_gpu(zadania: list[tuple[str, str, int]], karty: list[str], partia: int, zlozonosc: int, watki: int,
              procesy: int):
    """Generator (flac, ok, opis) jak przy CPU; gotowe .ogg pomijane, reszta dzielona po równo między karty."""
    import multiprocessing as mp
    with ProcessPoolExecutor(max(1, procesy)) as pula:
        czy = list(pula.map(gotowy, [z[0] for z in zadania], [z[1] for z in zadania], chunksize=256))
    for z, g in zip(zadania, czy):
        if g:
            yield z[0], True, "było"
    reszta = [z for z, g in zip(zadania, czy) if not g]
    print(f"  GPU: {len(zadania) - len(reszta)} już gotowych, do kodowania {len(reszta)} na kartach {','.join(karty)}",
          flush=True)
    if not reszta:
        return
    kontekst = mp.get_context("spawn")
    kolejka = kontekst.Manager().Queue()
    with ProcessPoolExecutor(len(karty), mp_context=kontekst) as pula:
        przyszle = [pula.submit(_gpu_proces, reszta[i::len(karty)], k, partia, zlozonosc, watki, kolejka)
                    for i, k in enumerate(karty)]
        zrobione, ostatnio, t0 = 0, 0.0, time.time()
        while not all(f.done() for f in przyszle) or not kolejka.empty():
            try:
                zrobione += kolejka.get(timeout=1)
            except Exception:
                continue
            if time.time() - ostatnio >= 30:
                ostatnio, dt = time.time(), time.time() - t0
                print(f"  GPU: {zrobione}/{len(reszta)} [{dt / 60:.1f} min, zostało ok. "
                      f"{(len(reszta) - zrobione) * dt / zrobione / 60:.0f} min]", flush=True)
        for f in przyszle:
            yield from f.result()


def test_zgodnosci(zadania: list[tuple[str, str, int]], karta: str, zlozonosc: int, n: int) -> None:
    """--test N: N klipów kodowanych na GPU i na CPU do katalogu tymczasowego; długość, SNR, przepływność. Nic nie zmienia."""
    import io
    import random
    import tempfile
    import numpy as np
    import soundfile as sf
    os.environ["CUDA_VISIBLE_DEVICES"] = karta
    from opus_gpu import encode_many
    proba = random.Random(0).sample(zadania, min(n, len(zadania)))
    pcm = [sf.read(z[0], dtype="int16")[0] for z in proba]
    sek = sum(len(x) for x in pcm) / 48000
    list(encode_many(pcm[:8], bitrate=proba[0][2] * 1000, complexity=zlozonosc, batch_size=8))   # rozgrzewka
    t0 = time.time()
    gpu = list(encode_many(sorted(pcm, key=len), bitrate=proba[0][2] * 1000, complexity=zlozonosc, batch_size=len(pcm)))
    t_gpu = time.time() - t0
    kolej = sorted(range(len(pcm)), key=lambda i: len(pcm[i]))
    gpu = [b for _, b in sorted(zip(kolej, gpu))]

    def snr(x, o):
        m = min(len(x), len(o))
        x, o = x[:m].astype(np.float64), o[:m].astype(np.float64)
        return 10 * np.log10(np.sum(x ** 2) / max(1e-9, np.sum((x - o) ** 2)))

    wynik = {"GPU (opus_gpu, CBR)": [], "CPU (libsndfile, VBR)": []}
    zla_dlugosc = {k: 0 for k in wynik}
    bajty = {k: 0 for k in wynik}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (z, x) in enumerate(zip(proba, pcm)):
            sciezka = os.path.join(tmp, f"{i}.ogg")
            koduj((z[0], sciezka, z[2]))
            for k, dane in (("GPU (opus_gpu, CBR)", gpu[i]), ("CPU (libsndfile, VBR)", open(sciezka, "rb").read())):
                o = sf.read(io.BytesIO(dane), dtype="int16")[0]
                zla_dlugosc[k] += len(o) != len(x)
                wynik[k].append(snr(x, o))
                bajty[k] += len(dane)
    print(f"Test {len(proba)} klipów ({sek / 60:.1f} min audio), complexity {zlozonosc}:")
    for k in wynik:
        print(f"  {k}: SNR mediana {np.median(wynik[k]):.1f} dB (min {min(wynik[k]):.1f}), "
              f"inna długość niż FLAC {zla_dlugosc[k]}/{len(proba)}, średnio {bajty[k] * 8 / sek / 1000:.0f} kbit/s")
    print(f"  GPU: {sek / t_gpu:.0f}x czas rzeczywisty (jedna partia {len(proba)} klipów; przy pełnym zbiorze "
          "partie po --partia klipów o podobnej długości)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("katalog", help="katalog syntezy (manifest.jsonl + audio/)")
    p.add_argument("--kbps", type=int, default=128)
    p.add_argument("--procesy", type=int, default=os.cpu_count())
    p.add_argument("--zostaw-flac", action="store_true", help="nie usuwaj plików FLAC (ani plików odrzuconych klipów)")
    p.add_argument("--pomin", action="append", default=[], help="lista id (jedno na wiersz) do odrzucenia zamiast konwersji")
    p.add_argument("--gpu", help="numery kart po przecinku: kodowanie na GPU przez opus_gpu (CBR)")
    p.add_argument("--partia", type=int, default=1024, help="--gpu: klipów w jednym wywołaniu kodera")
    p.add_argument("--zlozonosc", type=int, default=10, help="--gpu: complexity libopus 0..10 (10 jak domyślnie na CPU)")
    p.add_argument("--test", type=int, metavar="N",
                   help="z --gpu: tylko porównanie GPU z CPU na N losowych klipach FLAC (nic nie zmienia)")
    p.add_argument("--watki", type=int, default=8, help="--gpu: wątki czytające FLAC i zapisujące .ogg na kartę")
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
          + (f"GPU {a.gpu}, " if a.gpu else f"{a.procesy} procesów, ") + f"{a.kbps} kbit/s", flush=True)
    if not do_zrobienia and not odrzucone:
        print("Nic do zrobienia.")
        return

    zadania = [(str(k / w["audio"]), str(k / w["audio"])[:-5] + ".ogg", a.kbps) for w in do_zrobienia]
    if a.test:
        if not a.gpu:
            sys.exit("--test wymaga --gpu KARTA")
        return test_zgodnosci(zadania, a.gpu.split(",")[0].strip(), a.zlozonosc, a.test)
    if a.gpu:   # partia na GPU trwa tyle, co jej najdłuższy klip: podobne długości razem (1,6x -> ok. 3,8x szybciej)
        dlugosc = {str(k / w["audio"]): float(w.get("sekundy") or 0) for w in do_zrobienia}
        zadania.sort(key=lambda z: dlugosc[z[0]])
    sek = sum(float(w.get("sekundy") or 0) for w in do_zrobienia)
    t0, gotowe, nowe, bledy = time.time(), 0, 0, []
    with ProcessPoolExecutor(max(1, a.procesy)) as pula:
        strumien = (koduj_gpu(zadania, [x.strip() for x in a.gpu.split(",") if x.strip()], a.partia, a.zlozonosc,
                              a.watki, a.procesy)
                    if a.gpu else pula.map(koduj, zadania, chunksize=64))
        for flac, ok, opis in strumien:
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
