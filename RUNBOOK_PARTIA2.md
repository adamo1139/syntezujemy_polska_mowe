# Runbook: druga partia ok. 2500 h z VoxCPM2 (głos „poziomka”), filtr ASR + UTMOS

Kolejna partia danych do treningu MikołAI: ok. **2500 h** mowy wygenerowanej przez VoxCPM2 z tekstów
`cpral/polskie-sprawy-v8-jakosc50`, potem odfiltrowana przez **ASR** (Parakeet Ultra, `czysc_asr.py`) i **UTMOS**
(`czysc_utmos.py`). Instalacja silnika, wybór kart i strojenie: `RUNBOOK.md` (sekcje 1–5, 7). Ten plik opisuje tylko
różnice i kolejne kroki dla tej partii.

Zakładam katalog projektu na rigu (`/media/nvme_2tb/maked/audio/syntezujemy_polska_mowe`) i karty 1,2,3,5,6,7.

## 0. Przed startem

- **Format: Ogg Opus 128 kbit/s** (`generuj.py --format opus --kbps 128`). Mowa mono jest w Opusie przezroczysta
  od ok. 64–96 kbit/s; 128 to zapas. FLAC z pierwszej partii miał ok. 370 kbit/s (61 GB na 367,7 h), więc Opus
  128 to ok. 3× mniej: **2500 h ≈ 145 GB** na `wyniki/partia2/audio/`, drugie tyle po spakowaniu do parquet na HF.
  Odczyt wycinka z przeskokiem (jak w treningu kodeka) ok. 7 ms. `df -h` na dysku z `wyniki/` przed startem.
- **Czas syntezy:** pierwsza partia szła ok. 220× szybciej niż czas rzeczywisty na 6 kartach (`--rownolegle 32`),
  więc 2500 h to ok. **11–12 h** czystej syntezy (więcej, jeśli dysk dławi zapis).
- **Środowiska:** synteza w `.venv` z `wymagania.txt` (nano-vllm-voxcpm); liczby na słowa i ASR w środowisku
  z `morfeusz2` i `moondream` (jak przy `czysc_asr.py`); UTMOS w środowisku treningowym (`trening/.venv`: torch,
  torchaudio, soundfile).

## 1. Teksty: cały zbiór jakości 50+

```bash
python skrypty/przygotuj_teksty.py --percentyl 0 --wyjscie teksty/fragmenty_jakosc50.jsonl
```

- `--percentyl 0` = wszystkie rozmowy z datasetu (bez progu jakości ponad to, co już jest w nazwie zbioru);
  dalej odrzucane są rozmowy z modelem `private` (CC-BY-NC), czyszczony markdown i cięcie na fragmenty
  1–60 s jak w pierwszej partii.
- Cięcie jest deterministyczne dla każdej rozmowy (ziarno z `zrodlo_id`), więc rozmowa obecna w pierwszej partii
  daje te same fragmenty — dzięki temu `--pomin` w kroku 2 wyklucza je dokładnie.

## 2. Wybór 2500 h bez powtórek

Rozmowy z pierwszej partii (`teksty/wybrane.jsonl`, w tym 300 rozmów walidacji) i ze zbioru z liczbami
(`liczby/do_syntezy.jsonl`) są pomijane w całości. Po ich pominięciu w całym zbiorze zostaje ok. 17 550 h tekstu
(sprawdzone na rigu), więc tekstu wystarczy z dużym zapasem. Wybór w 50 częściach po ok. 50 h:

```bash
python skrypty/wybierz_teksty.py --fragmenty teksty/fragmenty_jakosc50.jsonl \
  --pomin teksty/wybrane.jsonl --pomin liczby/do_syntezy.jsonl \
  --znakow-na-sekunde 18.7 --godziny 2500 --czesci 50 --wyjscie teksty/partia2.jsonl
```

- `--znakow-na-sekunde 18.7`: tyle znaków na sekundę mówi „poziomka” w VoxCPM2. Domyślne 14 zawyża szacunek —
  w pierwszej partii „500 h” dało 367,7 h nagrań. Z 18,7 wybór 2500 h powinien dać ok. 2500 h audio.
- Po filtrze ASR i UTMOS (krok 5–6) odpadnie zwykle 10–20%, więc zostanie ok. 2000–2250 h.
- Tematy idą na zmianę (jak w pierwszej partii), więc dwa największe tematy nie zdominują zbioru.
- Gdyby tekstu zabrakło, skrypt wypisze na końcu „UWAGA: zabrakło fragmentów”.

## 3. Liczby, daty i godziny słownie

VoxCPM2 często przekręca cyfry („20 tys. zł”, „1,5 godziny”), więc nauczyciel dostaje tekst z liczbami zamienionymi
na słowa z odmianą (reguły + Morfeusz 2), tak jak przy zbiorze `liczby-slownie`:

```bash
python liczby/normalizuj.py --wejscie teksty/partia2.jsonl --wyjscie teksty/partia2_slownie.jsonl
```

- Wynik: `tekst` (słownie, dla VoxCPM2), `tekst_cyfry` (oryginał, z cyframi — ten zapis dostaje potem model
  uczony na tych danych), `flagi` (miejsca, gdzie reguła zgadywała).
- Kontrola: ile tekstów ma flagi i kilka przykładów do przejrzenia:

```bash
python - <<'EOF'
import json, random
w = [json.loads(l) for l in open("teksty/partia2_slownie.jsonl", encoding="utf-8")]
f = [x for x in w if x["flagi"]]
print(f"{len(w)} tekstów, z flagami {len(f)} ({100 * len(f) / len(w):.1f}%)")
for x in random.Random(0).sample(f, min(10, len(f))):
    print("-", x["flagi"], "|", x["tekst_cyfry"][:120], "->", x["tekst"][:120])
EOF
```

## 4. Synteza

Ustawienia jak w pierwszej partii (głos `glosy/poziomka.wav`, domyślne `--cfg`, `--kroki`, `--tryb` z `generuj.py`),
żeby zbiór był spójny z poprzednim. Próba na 200 tekstach, odsłuch, potem całość:

```bash
tmux new -s partia2
source .venv/bin/activate

python skrypty/generuj.py --karty 1,2,3,5,6,7 --teksty teksty/partia2_slownie.jsonl --limit 200 \
  --format opus --kbps 128 --rownolegle 32 --wyjscie wyniki/partia2_proba

python skrypty/generuj.py --karty 1,2,3,5,6,7 --teksty teksty/partia2_slownie.jsonl \
  --format opus --kbps 128 --rownolegle 32 --wyjscie wyniki/partia2 2>&1 | tee -a wyniki/partia2.log
```

- Nagrania: `wyniki/partia2/audio/<id>.ogg`; manifest zapisuje `format` i `kbps` przy każdym klipie.
- Wznawianie: to samo polecenie pomija `id` obecne w `wyniki/partia2/manifest.jsonl` (`RUNBOOK.md`, sekcja 8).
- Częściami (np. gdy nie chcemy zajmować kart na 11–12 h ciągiem): `--czesc 0`, `--czesc 1`, … do tego samego
  `--wyjscie` — manifest jest wspólny.
- Kontrola po syntezie: `RUNBOOK.md`, sekcja 9 (liczba klipów, godziny, podejrzanie długie > 90 s).

## 5. Filtr ASR (Parakeet Ultra)

```bash
python skrypty/czysc_asr.py asr  wyniki/partia2 --karty 1,2,3,5,6,7
python skrypty/czysc_asr.py ocen wyniki/partia2 --usun-procent 10 --usun-bez-asr --probki 40
```

- Klipy `.ogg` są przed transkrypcją dekodowane do tymczasowego WAV, więc Parakeet dostaje to samo co z FLAC.
- Porównanie transkrypcji z polem `tekst` (słownie — to, co dostał VoxCPM2); obie strony przechodzą przez
  `liczby/normalizuj.py`, bo Parakeet pisze liczby cyframi.
- Wynik: `wyniki/partia2/asr/usun.txt` (najgorsze 10% według CER + klipy bez transkrypcji), `raport.txt`,
  40 najgorszych do odsłuchu w `asr/probki/`.

## 6. Filtr UTMOS

Nowy skrypt `skrypty/czysc_utmos.py` (środowisko treningowe; wagi UTMOS pobierają się raz do `modele/utmos/`):

```bash
source trening/.venv/bin/activate
python skrypty/czysc_utmos.py ocen wyniki/partia2 --karty 1,2,3,5,6,7 --limit 500   # próba: czas na kartę
python skrypty/czysc_utmos.py ocen wyniki/partia2 --karty 1,2,3,5,6,7
python skrypty/czysc_utmos.py wybierz wyniki/partia2 --usun-procent 10 \
  --asr wyniki/partia2/asr/usun.txt --probki 40
```

- `ocen`: UTMOS22 dla każdego klipu, po jednym procesie na kartę, wznawialne (`utmos/utmos_K_z_N.jsonl`).
- `wybierz`: najgorsze 10% według UTMOS (albo `--prog 2.5` zamiast/oprócz procentu) → `utmos/usun.txt`;
  z `--asr` łączy z listą ASR w **`utmos/usun_razem.txt`**. `raport.txt` podaje rozkład UTMOS, ile klipów
  odrzucają oba filtry naraz i ile godzin zostaje.
- Przed usunięciem odsłuchać `utmos/probki/` (40 najgorszych). Jeśli najgorsze są dalej dobre — mniejszy procent;
  jeśli granica brzmi źle — większy albo `--prog`. UTMOS uczony był na angielskim: porównujemy względnie
  (najgorsze w tym zbiorze), nie wartość bezwzględną.

## 7. Do treningu i na HF

Dołączenie do danych treningowych (cała partia do treningu; walidacja bez zmian), bez odrzuconych klipów:

```bash
python skrypty/przygotuj_audio.py --dodaj wyniki/partia2 --usun wyniki/partia2/utmos/usun_razem.txt
```

Wysyłka na HF (konto `RockTheBoat`; repo powstaje prywatne): wersja po filtrze z własnym katalogiem paczek i kartą
z opisem filtra (`wyslij_na_hf.py --usun`, jak przy zbiorach `-czyste`):

```bash
python skrypty/wyslij_na_hf.py --wyniki wyniki/partia2 --usun wyniki/partia2/utmos/usun_razem.txt \
  --repo RockTheBoat/VoxCPM2-pol-one-voice-partia2-czyste
```

Po dołożeniu danych latenty trzeba policzyć na nowo dla nowych klipów (`trening/policz_latenty.py`,
`TRENING.md`, sekcja 6c), zanim ruszy dalszy trening modelu tekst→latenty.

## Podsumowanie poleceń

| krok | polecenie | wynik |
|---|---|---|
| 1 | `przygotuj_teksty.py --percentyl 0` | `teksty/fragmenty_jakosc50.jsonl` |
| 2 | `wybierz_teksty.py … --pomin … --znakow-na-sekunde 18.7 --godziny 2500 --czesci 50` | `teksty/partia2.jsonl` |
| 3 | `liczby/normalizuj.py` | `teksty/partia2_slownie.jsonl` |
| 4 | `generuj.py --karty 1,2,3,5,6,7 --format opus --kbps 128 --rownolegle 32` | `wyniki/partia2/` (ok. 145 GB) |
| 5 | `czysc_asr.py asr` + `ocen --usun-procent 10` | `wyniki/partia2/asr/usun.txt` |
| 6 | `czysc_utmos.py ocen` + `wybierz --usun-procent 10 --asr …` | `wyniki/partia2/utmos/usun_razem.txt` |
| 7 | `przygotuj_audio.py --dodaj … --usun …`, `wyslij_na_hf.py` | `dane/`, dataset na HF |
