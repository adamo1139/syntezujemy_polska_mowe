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
- RockTheBoat/VoxCPM2-pol-one-voice-367h-czyste
- RockTheBoat/VoxCPM2-pol-one-voice-liczby-slownie-czyste
- RockTheBoat/VoxCPM2-pol-one-voice-367h
- RockTheBoat/VoxCPM2-pol-one-voice-liczby
---

# MikołAI 109M v1

**MikołAI** to mały model zamiany tekstu na mowę dla języka polskiego: ok. 109M parametrów, dźwięk 48 kHz,
jeden głos. Czyta surowy polski tekst (bez fonemów i bez zewnętrznego G2P) i działa szybciej niż w czasie
rzeczywistym także na zwykłym procesorze.

- **Głos:** - syntetyczny, męski, wylosowany przez model VoxCPM2; nie należy do żadnej osoby, przypadł mi do gustu.
- **Wytrenowany od zera** (bez dostrajania cudzych wag) na ok. 440 h polskiej mowy wygenerowanej przez
  VoxCPM2 (destylacja: duży model nauczyciel → mały uczeń).
- **Biblioteka:** [adamo1139/mikolai](https://github.com/adamo1139/mikolai) (PyTorch, `pip install`).
- **Repo z kodem z eksperymentów, nieuporządkowane:** [adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe).

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

Wynik obu linii: <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/wizyta.wav"></audio> <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/szybciej_cfg1_kroki8.wav"></audio>

## Próbki

Ustawienia domyślne (CFG 4, 16 kroków), 48 kHz, tekst podany modelowi bez żadnych zmian:

| tekst | nagranie |
|---|---|
| Dzień dobry, to jest próbka nowego modelu. Jak brzmi? | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/01_powitanie.wav"></audio> |
| Czy stawka VAT na książki wynosi 5%, czy 8%? | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/02_vat.wav"></audio> |
| Chrząszcz brzmi w trzcinie w Szczebrzeszynie, a źdźbło trawy drży na wietrze. | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/03_chrzaszcz.wav"></audio> |
| Zdjęcie - za sterami samolotu Cessna 172, za szybą przednią widać Alpy zimą | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/04_cessna.wav"></audio> |
| Wypad na plażę. Na plaży są same parawany, bez ludzi. Morze jest bardzo niespokojne i fale są ogromne. | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/05_plaza.wav"></audio> |
| Wizyta w ZOO. Dwoje dzieci patrzy się na hipopotama który znajduje się za szybą, w dużym zbiorniku wodnym. Hipopotam energicznie płynie w ich stronę, co przeraża dzieci. | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/06_zoo.wav"></audio> |

Przykłady z „Szybkiego startu”: [`wizyta.wav`](https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/wizyta.wav) (domyślne ustawienia) i [`szybciej_cfg1_kroki8.wav`](https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS/resolve/main/probki/szybciej_cfg1_kroki8.wav) (CFG 1, 8 kroków).

## Pliki

| plik | zawartość |
|---|---|
| `model.safetensors` | wszystkie sieci potrzebne do syntezy (fp32, ok. 440 MB) i statystyki latentów |
| `config.json` | budowa sieci (sekcje `ttl`, `dp`, `dekoder`), częstotliwość, skala latentów, lista głosów |
| `probki/*.wav` | próbki z sekcji „Próbki” |
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


Synteza: szum → 16 kroków Eulera z classifier-free guidance (CFG 4, czyli 2 przebiegi pola na krok) →
dekoder. Ok. 70 GFLOP na sekundę dźwięku przy ustawieniach domyślnych, ok. 22 GFLOP przy CFG 1 i 8 krokach.

## Trening

Wszystko od zera, na 6× RTX 3090 Ti, z okresowym uśrednianiem wag między kartami zamiast synchronizacji
gradientów (local SGD), bo PCIe nie nadążało za zwykłym DDP.

**Dane** — mowa jednego głosu wygenerowana przez [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) z polskich
tekstów (rozmowy z [cpral/polskie-sprawy-v8-jakosc50](https://huggingface.co/datasets/cpral/polskie-sprawy-v8-jakosc50)):
- [RockTheBoat/VoxCPM2-pol-one-voice-367h-czyste](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h-czyste)
  — 331,3 h rozmów na różne tematy,
- [RockTheBoat/VoxCPM2-pol-one-voice-liczby-slownie-czyste](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-liczby-slownie-czyste)
  — 108,1 h tekstów z liczbami, datami i matematyką, z liczbami zamienionymi przed syntezą na słowa z odmianą,
- oba po filtrze ASR: odrzucone najgorsze 10% klipów według CER transkrypcji względem tekstu (połknięte
  słowa, bełkot, zapętlenia). Do kroku 200 tys. model uczył się na wersjach sprzed filtra
  ([367h](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h) i
  [liczby z cyframi](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-liczby)).

300 całych rozmów odłożonych do walidacji (osobno walidacja z liczbami).

**Kodek 48 kHz** (enkoder 25,6M + dekoder): autoenkoder w schemacie blue-codec, 220 tys. kroków, wynik jako
średnia wag (WSM). Dekoder potem osobno ulepszany przy zamrożonym enkoderze (latenty bez zmian):
moduł poprawki widma STFT (12 bloków, start od zera), dyskryminator SLM na zamrożonym WavLM-Base+ (jak
w StyleTTS2), segmenty 2,8 s, wycinek dla dyskryminatorów 1 s, odmrożony cały dekoder, strata prążków
sklejania (k · 93,75 Hz) i strata modulacji obwiedni szumu (usuwa „schodki” szumu w rytmie klatek).
Wynik: średnia wag z kroków 26–29 tys.

**Predyktor długości**: 0,5M, ok. 16 tys. kroków, średnia wag z kroków 14–16 tys. (wersja v3, dotrenowana
na danych z liczbami).

**Model tekst→latenty**: flow matching z CFG (losowe odrzucanie tekstu i stylu w treningu), stałe tempo
uczenia 5e-4 (rozgrzewka 5000 kroków), EMA 0,9999, 250 tys. kroków (0–200 tys. na danych przed filtrem,
200–250 tys. na danych po filtrze i z nowym zbiorem liczb). Opublikowane wagi: średnia (WSM) wag EMA z kroków
225, 230, 235, 240, 245 i 250 tys.


## Ograniczenia

- **Jeden głos.** Model nie klonuje głosów; inny wzorzec głosu zmieni niewiele albo zepsuje wynik.
- **Dane syntetyczne.** Brzmienie i błędy nauczyciela (VoxCPM2) przechodzą na ucznia: lekki szum tła,
  czasem przekręcone obce nazwy i skróty.
- **Tylko polski.** Angielskie wtrącenia są czytane „po polsku”.
- **Wypowiedzi do ok. 30 s.** Dłuższy tekst biblioteka dzieli na zdania; prozodia między kawałkami nie jest
  ciągła.
- **Bez sterowania prozodią** poza tempem całej wypowiedzi.

## Licencja

Wagi i materiały modelu: [OpenMDW 1.1](https://openmdw.ai/license/1-1/) (plik `LICENSE`). Rozpowszechniając
model, dołącz `LICENSE` oraz noty o pochodzeniu: architektura i kod treningowy pochodzą z
[BlueTTS](https://github.com/maxmelichov/BlueTTS) i [blue-codec](https://github.com/maxmelichov/blue-codec)
(MIT, pliki `LICENSE-BlueTTS` i `LICENSE-blue-codec`). Dane treningowe wygenerował
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2); dyskryminator WavLM-Base+ (MIT) był używany tylko
w treningu.
