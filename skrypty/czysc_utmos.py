#!/usr/bin/env python3
"""Czyszczenie syntezy przez UTMOS (przewidywana ocena słuchaczy): ocena każdego klipu -> lista do usunięcia.

Uzupełnia filtr ASR (skrypty/czysc_asr.py): ASR łapie złe słowa (połknięcia, bełkot, zapętlenia), UTMOS — złe
brzmienie (szum, przester, metaliczność, „pływający” głos), którego ASR nie widzi. Dwa kroki, wyniki w --praca
(domyślnie <katalog syntezy>/utmos):

  ocen     UTMOS22 strong (trening/training/utmos, wagi w modele/utmos/) dla klipów z manifest.jsonl (albo
           metadane.jsonl). --karty 1,2,3 puszcza po jednym procesie na kartę (każdy swoją część zbioru, log
           w <praca>/). Wznawialne: klipy już ocenione są pomijane. Wynik: utmos_K_z_N.jsonl (id, utmos, sekundy).
  wybierz  lista do usunięcia: najgorsze --usun-procent procent według UTMOS i/albo poniżej --prog; z --asr
           PLIK łączy ją z listą z ASR (usun.txt z czysc_asr.py ocen) w jedną. Wynik: usun.txt (sam UTMOS),
           usun_razem.txt (ASR + UTMOS, gdy --asr), raport.txt (rozkład, progi, ile wspólnych z ASR)
           i --probki N najgorszych klipów skopiowanych do <praca>/probki/ do odsłuchu.
  --pomin LISTA (np. asr/usun.txt): te klipy są pomijane w obu krokach — UTMOS liczony tylko na tym, co zostało po
  ASR, a procent w „wybierz” liczony od klipów, które zostały.
  Usunięcie z danych: python skrypty/przygotuj_audio.py --zastap KATALOG --usun KATALOG/utmos/usun_razem.txt

Środowisko: torch + torchaudio (np. środowisko treningowe trening/.venv). Przykład (karty 1,2,3,5,6,7):

  python skrypty/czysc_utmos.py ocen wyniki/partia2 --karty 1,2,3,5,6,7
  python skrypty/czysc_utmos.py wybierz wyniki/partia2 --usun-procent 10 --asr wyniki/partia2/asr/usun.txt --probki 40
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from czysc_asr import czesc_klipu, gotowe_id, wczytaj_manifest  # noqa: E402


def wyniki_oceny(praca: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(praca.glob("utmos_*_z_*.jsonl")):
        for w in f.open(encoding="utf-8"):
            try:   # ostatnia linia może być w trakcie dopisywania przez inny proces albo ucięta przy przerwaniu
                d = json.loads(w)
            except json.JSONDecodeError:
                continue
            out[d["id"]] = d
    return out


def wczytaj_pomin(args) -> set[str]:
    return {l.strip() for f in args.pomin for l in open(f, encoding="utf-8") if l.strip()}


def krok_ocen_na_kartach(args) -> None:
    karty = [k.strip() for k in args.karty.split(",") if k.strip()]
    praca = Path(args.praca or Path(args.katalog) / "utmos")
    praca.mkdir(parents=True, exist_ok=True)
    procesy = []
    sloty = [(karta, j) for karta in karty for j in range(args.na_karte)]
    for i, (karta, j) in enumerate(sloty):
        polecenie = [sys.executable, __file__, "ocen", args.katalog, "--praca", str(praca), "--czesc", f"{i}/{len(sloty)}"]
        if args.limit:
            polecenie += ["--limit", str(args.limit)]
        for f in args.pomin:
            polecenie += ["--pomin", f]
        log = (praca / f"utmos_karta{karta}_{j}.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": karta, "OMP_NUM_THREADS": "1"}
        procesy.append((karta, subprocess.Popen(polecenie, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"{len(sloty)} procesów ({args.na_karte} na kartę {','.join(karty)}), logi w {praca}", flush=True)
    pomin = wczytaj_pomin(args)
    wszystkich = sum(1 for w in wczytaj_manifest(Path(args.katalog)) if w["id"] not in pomin)
    try:
        ostatnio = time.time()
        while any(p.poll() is None for _, p, _ in procesy):
            time.sleep(5)
            if time.time() - ostatnio >= 60:
                ostatnio = time.time()
                print(f"  ocenionych: {len(wyniki_oceny(praca))} z {wszystkich}", flush=True)
    except KeyboardInterrupt:
        for _, p, _ in procesy:
            p.terminate()
        raise
    nieudane = [k for k, p, log in procesy if (log.close() or p.returncode)]
    if nieudane:
        sys.exit(f"Procesy na kartach {', '.join(nieudane)} skończyły się błędem — zobacz logi w {praca}. "
                 "To samo polecenie wznowi pracę.")
    print(f"Ocenionych razem: {len(wyniki_oceny(praca))} z {wszystkich} klipów.")
    print(f"Gotowe. Dalej: python {Path(__file__).relative_to(KATALOG_PROJEKTU)} wybierz {args.katalog} --usun-procent 10")


def krok_ocen(args) -> None:
    if args.karty:
        return krok_ocen_na_kartach(args)
    import soundfile as sf
    import torch
    sys.path.insert(0, str(KATALOG_PROJEKTU / "trening"))
    from training.utmos import utmos, wczytaj_utmos

    katalog = Path(args.katalog)
    praca = Path(args.praca or katalog / "utmos")
    praca.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in args.czesc.split("/"))
    plik = praca / f"utmos_{k}_z_{n}.jsonl"
    gotowe = set(wyniki_oceny(praca))   # ze wszystkich plików, także z wcześniejszego podziału na inną liczbę części
    pomin = wczytaj_pomin(args)
    klipy = [w for w in wczytaj_manifest(katalog)
             if czesc_klipu(w["id"], n) == k and w["id"] not in gotowe and w["id"] not in pomin]
    if args.limit:
        klipy = klipy[: args.limit]
    if klipy and not (katalog / klipy[0]["audio"]).is_file():
        sys.exit(f"Nie ma pliku audio {katalog / klipy[0]['audio']} — manifest jest, ale nagrań w {katalog} brak.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = wczytaj_utmos(dev)
    print(f"część {k}/{n}: {len(klipy)} klipów do oceny ({len(gotowe)} już ocenionych), {dev}", flush=True)
    t0 = time.time()
    with plik.open("a", encoding="utf-8") as out:
        for i, w in enumerate(klipy, 1):
            try:
                x, sr = sf.read(str(katalog / w["audio"]), dtype="float32", always_2d=True)
                wynik = utmos(model, torch.from_numpy(x.mean(1)).to(dev), sr)
            except Exception as e:   # uszkodzony plik itp.: zapisz brak oceny, nie przerywaj
                print(f"  {w['id']}: błąd {type(e).__name__}: {e}", flush=True)
                wynik = None
            out.write(json.dumps({"id": w["id"], "utmos": wynik, "sekundy": w.get("sekundy")}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                out.flush()
                print(f"  {i}/{len(klipy)} [{time.time() - t0:.0f} s]", flush=True)
    print(f"Gotowe: {len(klipy)} klipów w {time.time() - t0:.0f} s -> {plik}")


def krok_wybierz(args) -> None:
    katalog = Path(args.katalog)
    praca = Path(args.praca or katalog / "utmos")
    oceny = wyniki_oceny(praca)
    pomin = wczytaj_pomin(args)
    manifest = {w["id"]: w for w in wczytaj_manifest(katalog) if w["id"] not in pomin}
    if args.czesciowo:   # próba: tylko klipy, które już mają ocenę
        manifest = {i: w for i, w in manifest.items() if i in oceny and oceny[i]["utmos"] is not None}
    if pomin:
        print(f"--pomin: {len(pomin)} klipów pominiętych (np. odrzuconych przez ASR); liczę z {len(manifest)}")
    bez_oceny = [i for i in manifest if i not in oceny or oceny[i]["utmos"] is None]
    if len(bez_oceny) > 0.05 * len(manifest):
        sys.exit(f"{len(bez_oceny)} z {len(manifest)} klipów nie ma oceny UTMOS — najpierw dokończ krok „ocen”.")
    ok = sorted(((o["utmos"], i) for i, o in oceny.items() if i in manifest and o["utmos"] is not None))
    wartosci = [u for u, _ in ok]
    usun = set(bez_oceny)
    prog_procent = None
    if args.usun_procent:
        n = int(len(ok) * args.usun_procent / 100)
        usun |= {i for _, i in ok[:n]}
        prog_procent = ok[n - 1][0] if n else None
    if args.prog is not None:
        usun |= {i for u, i in ok if u < args.prog}
    sek = lambda ids: sum(float(manifest[i].get("sekundy") or 0) for i in ids) / 3600   # noqa: E731
    (praca / "usun.txt").write_text("".join(f"{i}\n" for i in sorted(usun)), encoding="utf-8")
    q = statistics.quantiles(wartosci, n=20)
    raport = [f"Klipów: {len(manifest)} ({sek(manifest):.1f} h), ocenionych: {len(ok)}, bez oceny: {len(bez_oceny)}",
              f"UTMOS: średnia {statistics.mean(wartosci):.3f}, mediana {statistics.median(wartosci):.3f}, "
              f"p5 {q[0]:.3f}, p10 {q[1]:.3f}, p25 {q[4]:.3f}, p75 {q[14]:.3f}, p95 {q[18]:.3f}",
              f"Do usunięcia (UTMOS): {len(usun)} klipów ({sek(usun):.1f} h)"
              + (f"; najgorsze {args.usun_procent:g}% to UTMOS < {prog_procent:.3f}" if prog_procent is not None else "")
              + (f"; próg {args.prog:g}" if args.prog is not None else "")]
    if args.asr:
        asr = {w.strip() for w in open(args.asr, encoding="utf-8") if w.strip()}
        razem = usun | asr
        (praca / "usun_razem.txt").write_text("".join(f"{i}\n" for i in sorted(razem)), encoding="utf-8")
        raport.append(f"ASR ({args.asr}): {len(asr)} klipów; wspólnych z UTMOS: {len(usun & asr)}; "
                      f"razem do usunięcia: {len(razem)} ({sek(razem & manifest.keys()):.1f} h, "
                      f"{100 * len(razem) / len(manifest):.1f}%); zostaje {sek(manifest.keys() - razem):.1f} h")
    if args.probki or args.najlepsze:
        kat = praca / "probki"
        shutil.rmtree(kat, ignore_errors=True)
        kat.mkdir(parents=True)
        wybrane = [("najgorszy", nr, u, i) for nr, (u, i) in enumerate(ok[: args.probki])]
        wybrane += [("najlepszy", nr, u, i) for nr, (u, i) in enumerate(ok[::-1][: args.najlepsze])]
        opis = []
        for rodzaj, nr, u, i in wybrane:
            zr = katalog / manifest[i]["audio"]
            nazwa = f"{rodzaj}_{nr:03d}_utmos{u:.2f}_{i}{zr.suffix}"
            shutil.copy(zr, kat / nazwa)
            opis.append(f"{nazwa}\n    {manifest[i].get('tekst', '')}\n")
        (kat / "opis.txt").write_text("".join(opis), encoding="utf-8")
        raport.append(f"Próbki do odsłuchu: {kat} ({args.probki} najgorszych, {args.najlepsze} najlepszych; teksty w opis.txt)")
    (praca / "raport.txt").write_text("\n".join(raport) + "\n", encoding="utf-8")
    print("\n".join(raport))
    print(f"Lista: {praca / ('usun_razem.txt' if args.asr else 'usun.txt')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("krok", choices=["ocen", "wybierz"])
    p.add_argument("katalog", help="katalog syntezy (manifest.jsonl + audio/) albo dane/ (metadane.jsonl)")
    p.add_argument("--praca", help="katalog wyników (domyślnie <katalog>/utmos)")
    p.add_argument("--karty", help="ocen: numery kart po przecinku, po jednym procesie na kartę")
    p.add_argument("--czesc", default="0/1", help="K/N: tylko ta część zbioru (używane wewnętrznie przez --karty)")
    p.add_argument("--na-karte", type=int, default=4,
                   help="ocen: procesy na kartę (każdy ocenia po jednym klipie; kilka naraz lepiej zapełnia GPU)")
    p.add_argument("--limit", type=int, help="ocen: tylko N klipów (próba)")
    p.add_argument("--pomin", action="append", default=[], help="lista id do pominięcia w obu krokach (np. asr/usun.txt)")
    p.add_argument("--usun-procent", type=float, help="wybierz: najgorsze N%% według UTMOS")
    p.add_argument("--prog", type=float, help="wybierz: usuń klipy z UTMOS poniżej tej wartości")
    p.add_argument("--asr", help="wybierz: usun.txt z czysc_asr.py — połącz z listą UTMOS w usun_razem.txt")
    p.add_argument("--probki", type=int, default=0, help="wybierz: tyle najgorszych klipów do odsłuchu")
    p.add_argument("--najlepsze", type=int, default=0, help="wybierz: tyle najlepszych klipów do odsłuchu")
    p.add_argument("--czesciowo", action="store_true",
                   help="wybierz: licz tylko z już ocenionych klipów (próba po ocen --limit)")
    a = p.parse_args()
    if a.krok == "wybierz" and not a.usun_procent and a.prog is None and not a.czesciowo:
        p.error("wybierz: podaj --usun-procent albo --prog")
    {"ocen": krok_ocen, "wybierz": krok_wybierz}[a.krok](a)


if __name__ == "__main__":
    main()
