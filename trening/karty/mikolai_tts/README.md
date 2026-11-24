---
license: other
license_name: openmdw-1.1
license_link: LICENSE
language:
- pl
tags:
- text-to-speech
- tts
- speech
- audio
- flow-matching
- polish
pipeline_tag: text-to-speech
datasets:
- RockTheBoat/VoxCPM2-pol-one-voice-367h
---

# MikołAI 113M v1 — mały polski TTS, 48 kHz

**MikołAI** to mały model zamiany tekstu na mowę dla języka polskiego: ok. 113M parametrów, dźwięk 48 kHz,
jeden głos. Czyta surowy polski tekst (bez fonemów i bez zewnętrznego G2P) i działa szybciej niż w czasie
rzeczywistym także na zwykłym procesorze.

- **Głos:** „poziomka” — syntetyczny, wylosowany przez model VoxCPM2; nie należy do żadnej osoby.
- **Wytrenowany od zera** (bez dostrajania cudzych wag) na ok. DO UZUPEŁNIENIA h polskiej mowy wygenerowanej przez
  VoxCPM2 (destylacja: duży model nauczyciel → mały uczeń).
- **Biblioteka:** [adamo1139/mikolai](https://github.com/adamo1139/mikolai) (PyTorch, `pip install`).
- **Kod treningowy:** [adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe).

## Szybki start

```bash
pip install "git+https://github.com/adamo1139/mikolai"
mikolai "Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?" -o wizyta.wav
```

```python
from mikolai import MikolAI

tts = MikolAI.from_pretrained("cpral/MikolAI-113M-1-Polski-TTS")
tts.zapisz("Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?", "wizyta.wav")  # 48 kHz
audio = tts.synteza("Szybciej, ale trochę gorzej.", cfg=1.0, kroki=8)  # numpy float32
```

## Pliki

| plik | zawartość |
|---|---|
| `model.safetensors` | wszystkie sieci potrzebne do syntezy (fp32, ok. 440 MB) i statystyki latentów |
| `config.json` | budowa sieci (sekcje `ttl`, `dp`, `dekoder`), częstotliwość, skala latentów, lista głosów |
| `glosy/poziomka.safetensors` | styl głosu policzony raz z nagrania wzorcowego (dla modelu i predyktora długości) |
| `LICENSE` | OpenMDW 1.1 (wagi i materiały modelu) |
| `LICENSE-BlueTTS`, `LICENSE-blue-codec` | noty MIT kodu, z którego pochodzi architektura |

## Architektura

Schemat BlueTTS / Supertonic: model tekst→latenty z dopasowaniem przepływu (flow matching) w przestrzeni
latentów własnego kodeka mowy 48 kHz, długość wypowiedzi z osobnego predyktora, dekoder kodeka zamienia
latenty na dźwięk. Bez autoregresji: całą wypowiedź liczy się naraz w kilkunastu krokach.

| część | parametry | co robi |
|---|---|---|
| enkoder tekstu | 6,9M | znaki (litery, cyfry, interpunkcja, wielkie litery osobno) → 6 bloków ConvNeXt + 4 warstwy attention; łączony ze stylem głosu |
| pole wektorowe | 64,0M | 4 bloki: ConvNeXt z dylatacjami 1/4/16/48, warunkowanie czasem przepływu i stylem, cross-attention do tekstu (RoPE); latenty 144-wymiarowe przy 15,6 klatki/s |
| predyktor długości | 0,35M (+0,15M enkodera stylu) | długość całej wypowiedzi z tekstu i stylu głosu |
| dekoder kodeka | 37,8M | latenty 24 × 93,75 klatki/s → 10 przyczynowych bloków ConvNeXt (25,3M) → głowica po 512 próbek → moduł poprawki widma STFT (12 bloków ConvNeXt, 12,5M) → 48 kHz |
| enkoder stylu | 3,7M | z nagrania wzorcowego → styl głosu; potrzebny tylko raz, wynik jest w `glosy/` |

Razem ok. 113M; przy syntezie wbudowanym głosem pracuje 109M (enkodery stylu i enkoder kodeka nie są
potrzebne i nie ma ich w repo).

Synteza: szum → 16 kroków Eulera z classifier-free guidance (CFG 4, czyli 2 przebiegi pola na krok) →
dekoder. Ok. 70 GFLOP na sekundę dźwięku przy ustawieniach domyślnych, ok. 22 GFLOP przy CFG 1 i 8 krokach.

## Trening

Wszystko od zera, na 6× RTX 3090 Ti, z okresowym uśrednianiem wag między kartami zamiast synchronizacji
gradientów (local SGD), bo PCIe nie nadążało za zwykłym DDP.

**Dane** — mowa jednego głosu wygenerowana przez [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) z polskich
tekstów (rozmowy z [cpral/polskie-sprawy-v8-jakosc50](https://huggingface.co/datasets/cpral/polskie-sprawy-v8-jakosc50)):
- [RockTheBoat/VoxCPM2-pol-one-voice-367h](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h),
- zbiór z liczbami, datami i matematyką (ok. DO UZUPEŁNIENIA h), teksty z liczbami zamienionymi na słowa z odmianą
  przed syntezą (VoxCPM2 często przekręcał cyfry): DO UZUPEŁNIENIA (repo HF),
- od kroku 200 tys. oba zbiory po filtrze ASR: odrzucone najgorsze 10% klipów według CER transkrypcji
  względem tekstu (połknięte słowa, bełkot, zapętlenia).

300 całych rozmów odłożonych do walidacji (osobno walidacja z liczbami).

**Kodek 48 kHz** (enkoder 25,6M + dekoder): autoenkoder w schemacie blue-codec, 220 tys. kroków, wynik jako
średnia wag (WSM). Dekoder potem osobno ulepszany przy zamrożonym enkoderze (latenty bez zmian):
moduł poprawki widma STFT (12 bloków, start od zera), dyskryminator SLM na zamrożonym WavLM-Base+ (jak
w StyleTTS2), segmenty 2,8 s, wycinek dla dyskryminatorów 1 s, odmrożony cały dekoder, strata prążków
sklejania (k · 93,75 Hz) i strata modulacji obwiedni szumu (usuwa „schodki” szumu w rytmie klatek).
Wynik: średnia wag z kroków 26–29 tys. (DO UZUPEŁNIENIA, jeśli zmieni się do publikacji).

**Predyktor długości**: 0,5M, ok. 16 tys. kroków, średnia wag z kroków 14–16 tys. (wersja v3, dotrenowana
na danych z liczbami).

**Model tekst→latenty**: flow matching z CFG (losowe odrzucanie tekstu i stylu w treningu), stałe tempo
uczenia 5e-4 (rozgrzewka 5000 kroków), EMA 0,9999, 250 tys. kroków (0–200 tys. na danych przed filtrem,
200–250 tys. na danych po filtrze i z nowym zbiorem liczb). Opublikowane wagi: DO UZUPEŁNIENIA (EMA z kroku 250 tys.
albo średnia wag z kroków 225–250 tys.).

## Wyniki

UTMOS (przewidywana ocena słuchaczy, UTMOS22 strong; im wyżej, tym lepiej). Oryginały VoxCPM2 to sufit
osiągalny przy destylacji.

**Kodek** (pełna walidacja; rekonstrukcja = oryginał → enkoder → dekoder):

| | UTMOS val | UTMOS val z liczbami |
|---|---|---|
| oryginał VoxCPM2 | 2,999 | 2,858 |
| pierwszy dekoder (krok 220 tys.) | 2,739 | 2,624 |
| **dekoder opublikowany** | **2,880** | **2,763** |

**Synteza** (24 zdania próbne, w tym liczby, daty i nazwy; CFG 4, 16 kroków):

| model tekst→latenty | dekoder | UTMOS |
|---|---|---|
| EMA, krok 225 tys. | pierwszy | 2,983 |
| EMA, krok 225 tys. | opublikowany | 3,149 |
| **opublikowane wagi** | opublikowany | **DO UZUPEŁNIENIA** |

DO UZUPEŁNIENIA: strata walidacyjna, zrozumiałość (CER/WER transkrypcji Whisper), wyniki na walidacji z liczbami.

**Szybkość** (PyTorch fp32, bez kompilacji; RTF = czas liczenia / długość nagrania):

| sprzęt | CFG / kroki | RTF |
|---|---|---|
| RTX 5080 | 4 / 16 | ok. 0,17 (krótkie zdanie, z narzutem startu) |
| i7-11700K, 8 wątków | 4 / 16 | ok. 0,35 |
| i7-11700K, 1 wątek | 4 / 16 | ok. 0,8 |
| i7-11700K, 1 wątek | 1 / 8 | ok. 0,25 |

## Ograniczenia

- **Jeden głos.** Model nie klonuje głosów; inny wzorzec głosu zmieni niewiele albo zepsuje wynik.
- **Dane syntetyczne.** Brzmienie i błędy nauczyciela (VoxCPM2) przechodzą na ucznia: lekki szum tła,
  czasem przekręcone obce nazwy i skróty.
- **Tylko polski.** Angielskie wtrącenia są czytane „po polsku”.
- **Wypowiedzi do ok. 30 s.** Dłuższy tekst biblioteka dzieli na zdania; prozodia między kawałkami nie jest
  ciągła.
- **Bez sterowania prozodią** poza tempem całej wypowiedzi.

## Odpowiedzialne użycie

Głos jest syntetyczny i nie należy do żadnej osoby. Nie używaj modelu do podszywania się pod ludzi ani do
wprowadzania w błąd co do tego, że nagranie jest syntetyczne.

## Licencja

Wagi i materiały modelu: [OpenMDW 1.1](https://openmdw.ai/license/1-1/) (plik `LICENSE`). Rozpowszechniając
model, dołącz `LICENSE` oraz noty o pochodzeniu: architektura i kod treningowy pochodzą z
[BlueTTS](https://github.com/maxmelichov/BlueTTS) i [blue-codec](https://github.com/maxmelichov/blue-codec)
(MIT, pliki `LICENSE-BlueTTS` i `LICENSE-blue-codec`). Dane treningowe wygenerował
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2); dyskryminator WavLM-Base+ (MIT) był używany tylko
w treningu.
