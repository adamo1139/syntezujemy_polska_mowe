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

> Ten README i szczegóły implememtacyjne są WIP

# MikołAI 109M 0.1

**MikołAI** to mały model zamiany tekstu na mowę trenowany na języku polskim: ok. 109M parametrów, dźwięk 48 kHz,
jeden głos. Czyta surowy polski tekst i działa szybciej niż w czasie
rzeczywistym nawet na jednym rdzeniu desktopa.

Jeden męski syntetyczny głos który mi się podobał. Model jest wytrenowany od zera na około 400 godzinach mowy wygenerowanej przez model VoxCPM2.

Do Inferencji zalecam używać biblioteki [adamo1139/mikolai](https://github.com/adamo1139/mikolai) .

Repo z kodem treningu i eksperymentów, nieuporządkowane, jest [tu](https://github.com/adamo1139/syntezujemy_polska_mowe)

## Szybki start

```bash
pip install "git+https://github.com/adamo1139/mikolai"
mikolai "Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?" -o wizyta.ogg
```

```python
from mikolai import MikolAI

tts = MikolAI.from_pretrained("cpral/MikolAI-109M-v0.1-Polish-TTS-Preview")
tts.zapisz("Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?", "wizyta.ogg")  # 48 kHz, Opus 64 kbit/s (albo .wav / .flac)
audio = tts.synteza("Szybciej, ale trochę gorzej.", cfg=1.0, kroki=8)  # numpy float32
```

Wynik obu linii: <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/wizyta.wav"></audio> <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/szybciej_cfg1_kroki8.wav"></audio>

## Próbki

Ustawienia domyślne (CFG 4, 16 kroków)

| zdanie | z opowiadania (tytuł, autor) | nagranie |
|---|---|---|
| Barton szukał swojego nazwiska w oddziałach zmechanizowanych i zwiadowczych ale bez skutku. | „The Ghost”, Kapibar ([Wattpad](https://www.wattpad.com/story/90155243)) | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/zdanie_1.ogg"></audio> |
| Prezydent przemawiał do narodu że całe zamieszanie zostanie wyjaśnione jeszcze do końca tego roku. | „The Ghost”, Kapibar ([Wattpad](https://www.wattpad.com/story/90155243)) | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/zdanie_2.ogg"></audio> |
| Byłem ze znajomymi na boisku, podzieliliśmy się wcześniej na kilka drużyn i graliśmy mecze w piłkę nożną. | „Moje sny”, foxay77 ([Wattpad](https://www.wattpad.com/story/184317072)) | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/zdanie_3.ogg"></audio> |
| Jednak po chwili wszystko wróciło do normy, sklep się odbudował, my za to odstawiliśmy wózki i chodzimy ze szklanym pucharkiem do lodów. | „Moje sny”, foxay77 ([Wattpad](https://www.wattpad.com/story/184317072)) | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/zdanie_4.ogg"></audio> |
| Spacerowałam sobie po lesie z Eevee gdzie zauważyłam jakiegoś przechodzącego się pokemona sprawdziłam w pokedexie co to za pokemon. | „Moja Wspaniała Przygoda Pokemon”, Stasia110 ([Wattpad](https://www.wattpad.com/story/146673104)) | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/probki/zdanie_5.ogg"></audio> |

## Test na czytaniu rozdziałów tekstów

Znalazłem 3 teksty na Wattpadzie. TTS zrobił z tego mowę. Biblioteka sama dzieli tekst na kawałki i łączy tekst, wagi wspierają generację klipów tylko do około ~25 sekund długości, więc klejenie kawałków jest konieczne.

| tekst | autor, licencja | znaków (tekst) | nagranie | urządzenie | czas generacji | odsłuch |
|---|---|---|---|---|---|---|
| [Moje sny](https://www.wattpad.com/story/184317072) - zapiski snów | foxay77, CC BY | [33 522](https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/blob/main/teksty/moje_sny.txt) | 29,0 min | RTX 5080 (partia 64) | 9,2 s | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/moje_sny.ogg"></audio> |
| [Moja Wspaniała Przygoda Pokemon](https://www.wattpad.com/story/146673104) - bez emoji | Stasia110, domena publiczna | [21 309](https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/blob/main/teksty/przygoda_pokemon.txt) | 19,8 min | i7-11700K, 1 rdzeń (partia 1) | 14,8 min | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/przygoda_pokemon.ogg"></audio> |
| [The Ghost](https://www.wattpad.com/story/90155243) - fragment części I | Kapibar, domena publiczna | [55 743](https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/blob/main/teksty/the_ghost.txt) | 50,5 min | RTX 5080 (partia 64) | 16,2 s | <audio controls preload="none" src="https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview/resolve/main/the_ghost.ogg"></audio> |

## Szybkość

Synteza powyższych tekstów (`skrypty/benchmark_mikolai.py` w repo z kodem), PyTorch, bez kompilacji;
RTF = czas liczenia / długość nagrania.

| sprzęt | ustawienia | tekst | czas liczenia | RTF | szybciej niż czas rzeczywisty |
|---|---|---|---|---|---|
| RTX 5080 | CFG 4, 16 kroków, partia 64 | wszystkie trzy, 99 min mowy | **34 s** | **0,006** | **ok. 176×** |
| i7-11700K, 1 rdzeń | CFG 4, 16 kroków, partia 1 | Przygoda Pokemon, 19,8 min mowy | **14,8 min** | **0,75** | **1,3×** |

## Pliki

Do syntezy wystarczą `model.safetensors` (wagi, 440 MB), `config.json` i `glosy/`. Reszta to próbki
(`probki/`, `*.ogg`, `teksty/`) i licencje.

## Architektura

Architektura BlueTTS / Supertonic

| część | parametry | co robi |
|---|---|---|
| enkoder tekstu | 6,9M | surowe litery > 6 bloków ConvNeXt + 4 warstwy attention |
| pole wektorowe | 64,0M | 4 bloki: ConvNeXt z dylatacjami 1/4/16/48, warunkowanie czasem przepływu i stylem, cross-attention do tekstu (RoPE); latenty 144-wymiarowe przy 15,6 klatki/s |
| predyktor długości | 0,5M | przewiduje długość całej wypowiedzi z tekstu i stylu głosu |
| dekoder kodeka | 37,8M | latenty 24 × 93,75 klatki/s > 10 przyczynowych bloków ConvNeXt (25,3M) > głowica po 512 próbek > moduł poprawki widma STFT (12 bloków ConvNeXt, 12,5M) → 48 kHz |
| enkoder stylu | 3,7M | z nagrania wzorcowego > styl głosu; potrzebny tylko raz, wynik jest w `glosy/` |


Synteza: szum > 16 kroków Eulera z CFG 4 > dekoder.
Około 70 GFLOP na sekundę dźwięku przy ustawieniach domyślnych, około 22 GFLOP przy CFG 1 i 8 krokach.

## Trening

Wszystko od zera, na 6× RTX 3090 Ti, lokalnie.

Zbiory danych: 
[RockTheBoat/VoxCPM2-pol-one-voice-330h-czyste](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-330h-czyste)
[RockTheBoat/VoxCPM2-pol-one-voice-liczby-slownie-czyste](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-liczby-slownie-czyste)

**Kodek 48 kHz** (enkoder 25,6M + dekoder 37.8M): autoenkoder w schemacie blue-codec, 249 tys. kroków.

**Predyktor długości**: 0,5M, ok. 16 tys. kroków, średnia wag z kroków 14–16 tys.

**Model tekst > latenty**: flow matching z CFG, stałe tempo
uczenia 5e-4 po rozgrzewce, EMA 0,9999, 250 tys. kroków. Opublikowane wagi to średnia (WSM) wag EMA z kroków
225k/230k/235k/240k/245k/250k.


## Ograniczenia

Klonowanie nie jest wspierane, model nie umie wymawiać większości słów z innych języków, na przykład słów z języka angielskiego. Długi tekst jest zlepiany z krótkich klipów i nie jest jednorodny. Model często robi błędy w wymowie, często niewłaściwie wymawia zdania kończące się pytajnikiem, ogólnie ton wypowiedzi jest monotonny i słaby.

## Licencja

Użycie komercyjne jest dozwolone.
