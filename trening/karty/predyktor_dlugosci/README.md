---
license: apache-2.0
language:
- pl
tags:
- text-to-speech
- duration-predictor
datasets:
- RockTheBoat/VoxCPM2-pol-one-voice-367h
---

# MikołAI: predyktor długości wypowiedzi dla polskiego TTS, jeden głos

Mały model (ok. 0,5M parametrów) z architektury [BlueTTS](https://github.com/maxmelichov/BlueTTS) (schemat
Supertonic). Z tekstu (surowe znaki, z wielkimi literami) i krótkiego wycinka latentów głosu przewiduje
łączną długość wypowiedzi w sekundach. Model tekst→latenty dostaje od niego liczbę klatek do wygenerowania.

Latenty pochodzą z kodeka [RockTheBoat/MikolAI-kodek-checkpoints](https://huggingface.co/RockTheBoat/MikolAI-kodek-checkpoints)
(`ae_wsm_210_220.pt`, 48 kHz); z innym kodekiem ani innymi statystykami latentów predyktor nie zadziała poprawnie.

## Pliki

| plik | co to jest |
|---|---|
| `duration_predictor_wsm.pt` | **do użycia**: średnia wag z kroków 4500, 5000, 5500 i 6000 |
| `stats.pt` | statystyki latentów (średnia i odchylenie 144 kanałów) policzone tym kodekiem |
| `polski_100m.json` | konfiguracja (sekcja `dp` to ten model) |
| `checkpoints/duration_predictor_*.pt` | punkty kontrolne co 500 kroków i wersja końcowa |

## Trening

- Dane: [RockTheBoat/VoxCPM2-pol-one-voice-367h](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h),
  ok. 148 tys. klipów do 30 s; 300 całych rozmów odłożonych do walidacji. Bez filtra jakości.
- Cel: logarytm łącznej długości klipu, strata L1. Wycinek wzorcowy: losowe 5–95% latentów tego samego klipu.
- AdamW, tempo 1e-4 stałe, 6000 kroków, partia 16 na kartę × 6 kart (DDP).

## Wyniki

Walidacja: 1457 klipów spoza treningu.

| model | strata (L1 log) | średni błąd długości |
|---|---|---|
| najlepszy pojedynczy punkt (krok 6000) | 0,0767 | 7,46% |
| **`duration_predictor_wsm.pt`** | **0,0755** | **7,45%** |

Pozostały błąd to w dużej mierze naturalna zmienność tempa nauczyciela (VoxCPM2): ten sam tekst bywa czytany
szybciej albo wolniej.

## Kod

[adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe), katalog `trening/`
(kopia BlueTTS z poprawkami, MIT): trening `python -m training.dp.cli`, uśrednianie `usrednij_dp.py`.

## Licencje

Wagi: Apache-2.0, jak dane treningowe. Kod: MIT (BlueTTS).
