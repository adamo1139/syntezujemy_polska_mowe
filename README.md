# Syntezujemy polską mowę

Kod, dane i notatki z budowy **MikołAI** — rodziny małych polskich modeli zamiany tekstu na mowę (TTS),
wytrenowanych od zera na mowie syntetycznej. Duży model nauczyciel
([VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)) czyta polskie teksty jednym głosem, a mały model uczeń
uczy się go naśladować i działa szybciej niż w czasie rzeczywistym na zwykłym procesorze.

| | |
|---|---|
| **Model** | [cpral/MikolAI-113M-1-Polski-TTS](https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS) — ok. 113M parametrów, 48 kHz, jeden głos |
| **Biblioteka** | [adamo1139/mikolai](https://github.com/adamo1139/mikolai) — `pip install "git+https://github.com/adamo1139/mikolai"` |
| **Dane** | [RockTheBoat/VoxCPM2-pol-one-voice-367h](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h) (367 h) oraz zbiory z liczbami, datami i matematyką |
| **Teksty źródłowe** | [cpral/polskie-sprawy-v8-jakosc50](https://huggingface.co/datasets/cpral/polskie-sprawy-v8-jakosc50) |

```python
from mikolai import MikolAI

tts = MikolAI.from_pretrained()
tts.zapisz("Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?",
           "wizyta.wav")
```

## Jak to działa

1. **Teksty** — rozmowy z `polskie-sprawy-v8-jakosc50`, oczyszczone z markdownu i pocięte na fragmenty z całych
   zdań (`skrypty/przygotuj_teksty.py`, `skrypty/wybierz_teksty.py`).
2. **Mowa nauczyciela** — VoxCPM2 przez silnik `nano-vllm-voxcpm` na 6 kartach, jeden stały głos „poziomka”
   (`skrypty/generuj.py`, instrukcja w [`RUNBOOK.md`](RUNBOOK.md)). Teksty z liczbami najpierw zamieniane na
   słowa z odmianą (`liczby/normalizuj.py`, reguły + Morfeusz 2), bo nauczyciel przekręcał cyfry.
3. **Czyszczenie** — transkrypcja ASR każdego klipu i odrzucenie najgorszych 10% według CER (`skrypty/czysc_asr.py`).
4. **Kodek mowy 48 kHz** — autoenkoder (schemat blue-codec): dźwięk ↔ latenty 24 × 93,75 klatki/s
   (`trening/train_autoencoder.py`). Dekoder potem osobno ulepszany: moduł poprawki widma, dyskryminator SLM
   (WavLM), straty prążków i modulacji szumu.
5. **Predyktor długości** (0,5M) i **model tekst → latenty** (75M, flow matching z CFG, schemat BlueTTS /
   Supertonic) na latentach policzonych raz z kodeka (`trening/training/`).
6. **Eksport** do biblioteki i na HF (`trening/eksportuj_mikolai.py`).

Pełny przebieg z poleceniami, parametrami, wynikami pomiarów i porzuconymi pomysłami jest w
[`TRENING.md`](TRENING.md); założenia i decyzje w [`PLAN.md`](PLAN.md).

## Układ repozytorium

| katalog / plik | zawartość |
|---|---|
| `PLAN.md` | cel, plan danych i drabina modeli, decyzje |
| `RUNBOOK.md` | instalacja i uruchomienie syntezy VoxCPM2 na wielu kartach |
| `TRENING.md` | dziennik treningu: polecenia, wyniki, odsłuchy, co nie zadziałało |
| `skrypty/` | dane: przygotowanie tekstów, synteza, czyszczenie ASR, pakowanie i wysyłka na HF |
| `liczby/` | zamiana liczb, dat, godzin, kwot i jednostek na słowa z odmianą |
| `glosy/` | nagrania wzorcowe głosu „poziomka” |
| `trening/` | kod treningowy (kopia BlueTTS i blue-codec z naszymi zmianami, opis w `trening/ZRODLO.md`), oceny, eksport, karty modeli |
| `trening/karty/` | karty modeli publikowanych na HF |
| `cose/` | porzucony eksperyment z postfiltrem COSE (kod autorów bez licencji, patrz `cose/README.md`) |
| `wymagania.txt` | środowisko do syntezy nauczycielem |

Środowisko treningowe opisuje `trening/pyproject.toml` (uv, PyTorch z CUDA 12.8); polecenia uruchamia się
z katalogu `trening/`.

## Sprzęt

Wszystko powstało na jednej maszynie z 8× RTX 3090 Ti (trening na 6 kartach, z okresowym uśrednianiem wag
zamiast synchronizacji gradientów) i komputerze z RTX 5080 do odsłuchów i testów.

## Licencje

- **Kod w `trening/`** — pochodzi z [BlueTTS](https://github.com/maxmelichov/BlueTTS) i
  [blue-codec](https://github.com/maxmelichov/blue-codec) (MIT, `trening/LICENSE-BlueTTS`,
  `trening/LICENSE-blue-codec`).
- **`cose/`** — kod autorów COSE bez licencji (UNLICENSED), tylko do eksperymentu.
- **Wagi MikołAI** — OpenMDW 1.1 (w repo modelu na HF).
- **Dane** — synteza VoxCPM2 (Apache 2.0) z tekstów `polskie-sprawy-v8-jakosc50`; licencje zbiorów podane
  na ich stronach HF.

Głos „poziomka” jest syntetyczny i nie należy do żadnej osoby.
