#!/usr/bin/env python3
"""Wysyła pliki modeli (punkty kontrolne, kodek, karta modelu) do repozytorium modelu na HF.

Każdy plik to osobny commit; pliki, które już są w repo w tym samym miejscu i mają ten sam rozmiar, są
pomijane, więc przerwaną wysyłkę wznawia to samo polecenie. Na końcu skrypt porównuje rozmiary w repo
z lokalnymi i wypisuje, które pliki można bezpiecznie usunąć z dysku.

Repo powstaje jako prywatne, chyba że podasz --publiczne (wtedy istniejące też jest przełączane).
Wymaga zalogowania: hf auth login (token z prawem zapisu).

Przykłady (z katalogu projektu):
  python skrypty/wyslij_modele_na_hf.py --repo UZYTKOWNIK/NAZWA --do . \\
    --pliki modele/kodek_48k/ae_wsm_210_220.pt modele/kodek_48k/ae_wsm_210_220_probka.wav \\
            trening/karty/kodek_48k/README.md trening/config/polski_100m.json
  python skrypty/wyslij_modele_na_hf.py --repo UZYTKOWNIK/NAZWA --do checkpoints \\
    --pliki modele/kodek_48k/ae_*.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def w_repo(api, repo: str) -> dict[str, int]:
    """Ścieżka -> rozmiar dla wszystkich plików w repo."""
    from huggingface_hub.hf_api import RepoFile

    return {f.path: f.size for f in api.list_repo_tree(repo, repo_type="model", recursive=True) if isinstance(f, RepoFile)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="repo modelu na HF, np. uzytkownik/nazwa")
    p.add_argument("--pliki", nargs="+", required=True, help="pliki do wysłania")
    p.add_argument("--do", default=".", help="katalog w repo (domyślnie główny)")
    p.add_argument("--publiczne", action="store_true", help="repo publiczne (domyślnie prywatne)")
    p.add_argument("--prob", type=int, default=5, help="ile razy ponawiać nieudaną wysyłkę pliku")
    args = p.parse_args()

    pliki = [Path(x) for x in dict.fromkeys(args.pliki)]
    brak = [str(x) for x in pliki if not x.is_file()]
    if brak:
        sys.exit(f"Nie ma plików: {', '.join(brak)}")
    nazwy = [x.name for x in pliki]
    if len(set(nazwy)) != len(nazwy):
        sys.exit("Dwa pliki o tej samej nazwie trafiłyby w to samo miejsce w repo.")
    katalog = args.do.strip("/")
    cel = lambda plik: plik.name if katalog in ("", ".") else f"{katalog}/{plik.name}"

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        kto = api.whoami()["name"]
    except Exception:
        sys.exit("Brak zalogowania do HF: uruchom `hf auth login` (token z prawem zapisu).")
    api.create_repo(args.repo, repo_type="model", private=not args.publiczne, exist_ok=True)
    if args.publiczne:
        api.update_repo_settings(args.repo, repo_type="model", private=False)

    obecne = w_repo(api, args.repo)
    do_wyslania = [x for x in pliki if obecne.get(cel(x)) != x.stat().st_size]
    lacznie = sum(x.stat().st_size for x in do_wyslania)
    print(f"Repo {args.repo} ({'publiczne' if args.publiczne else 'prywatne'}), jako {kto}: "
          f"{len(pliki)} plików, już w repo {len(pliki) - len(do_wyslania)}, do wysłania {len(do_wyslania)} ({lacznie / 1e9:.1f} GB)")

    t0 = time.time()
    for nr, plik in enumerate(do_wyslania, 1):
        for proba in range(1, args.prob + 1):
            try:
                api.upload_file(path_or_fileobj=str(plik), path_in_repo=cel(plik), repo_id=args.repo, repo_type="model",
                                commit_message=f"Dodaj {cel(plik)}")
                break
            except KeyboardInterrupt:
                raise
            except Exception as blad:
                if proba == args.prob:
                    raise
                print(f"  błąd przy {plik.name} ({blad!r}); próba {proba}/{args.prob}, ponawiam za {30 * proba} s", flush=True)
                time.sleep(30 * proba)
        print(f"  {nr}/{len(do_wyslania)}: {cel(plik)} ({plik.stat().st_size / 1e9:.2f} GB)  [{time.time() - t0:.0f} s]", flush=True)

    obecne = w_repo(api, args.repo)
    zgodne = [x for x in pliki if obecne.get(cel(x)) == x.stat().st_size]
    print(f"\nW repo z właściwym rozmiarem: {len(zgodne)}/{len(pliki)} plików — https://huggingface.co/{args.repo}")
    if len(zgodne) != len(pliki):
        print("Niezgodne albo brakujące:", ", ".join(x.name for x in pliki if x not in zgodne))


if __name__ == "__main__":
    main()
