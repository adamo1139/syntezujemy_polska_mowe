---
license: apache-2.0
language:
- pl
tags:
- audio
- speech
- codec
- autoencoder
- text-to-speech
datasets:
- RockTheBoat/VoxCPM2-pol-one-voice-367h
---

# MikołAI: kodek mowy 48 kHz (autoenkoder) dla polskiego TTS, jeden głos

Autoenkoder mowy w architekturze [blue-codec](https://github.com/maxmelichov/blue-codec) (schemat Supertonic),
wytrenowany od zera na 48 kHz. Zamienia dźwięk na latent 24 liczb na 512 próbek (93,75 klatki/s) i z powrotem.
Służy jako warstwa audio dla modelu TTS typu BlueTTS: model główny generuje latenty, a dekoder z tego repo
zamienia je na dźwięk.

**To nie jest kodek ogólnego przeznaczenia.** Uczył się wyłącznie na jednym syntetycznym głosie („poziomka”,
wygenerowanym przez VoxCPM2); na innych głosach i nagraniach nie był sprawdzany.

## Pliki

| plik | co to jest |
|---|---|
| `ae_wsm_210_220.pt` | **kodek do użycia**: średnia wag z 6 punktów kontrolnych (kroki 210–220 tys.), klucze `encoder` i `decoder` |
| `ae_wsm_210_220_probka.wav` | rekonstrukcja nagrania wzorcowego tym kodekiem, do odsłuchu |
| `polski_100m.json` | konfiguracja (sekcja `ae` to kodek); ścieżki danych są do uzupełnienia |
| `checkpoints/ae_*.pt` | punkty kontrolne z treningu, z optymalizatorem i dyskryminatorami (do wznawiania treningu) |

## Architektura

| część | parametry | kiedy potrzebna |
|---|---|---|
| enkoder (widmo liniowe 1025 + 228 pasm melowych → 10 bloków ConvNeXt, szerokość 512 → latent 24) | 25,6M | trening modelu głównego (liczenie latentów) |
| dekoder (latent 24 → 10 bloków ConvNeXt, szerokość 512 → 512 próbek na klatkę) | 25,3M | każda synteza |

Próbkowanie 48 kHz, mono, krok 512 próbek.

## Trening

- **Dane:** [RockTheBoat/VoxCPM2-pol-one-voice-367h](https://huggingface.co/datasets/RockTheBoat/VoxCPM2-pol-one-voice-367h),
  ok. 364 h w treningu; 300 całych rozmów (1488 klipów) odłożonych do walidacji. Bez filtra jakości.
- **Straty:** przepis domyślny blue-codec (rekonstrukcja melowa do 12 kHz, dyskryminatory wielookresowe
  i wielorozdzielcze, dopasowanie cech) plus składnik górnego pasma (`--hiband_w 2.0`).
- **Sprzęt:** 6× RTX 3090 Ti, bez synchronizacji gradientów: każda karta trenowała własną kopię, a kopie były
  uśredniane co 20 kroków (local SGD), bo zwykły DDP dławił się na PCIe.
- **Przebieg:**
  - kroki 0–60 tys.: partia 64 na kartę, tempo uczenia 8,5e-5 z wygaszaniem cosinusowym (planowane na 100 tys.);
  - kroki 60–200 tys.: partia 16 na kartę, tempo praktycznie stałe 8,5e-5;
  - kroki 200–220 tys.: partia 64 na kartę, tempo stałe 8,5e-5;
  - wynik: średnia wag z kroków 210, 212, 214, 216, 218 i 220 tys. (WSM, bez końcowego wygaszania tempa).

## Wyniki

Odległość rekonstrukcji od oryginału na 64 odłożonych klipach (do 15 s każdy, ok. 10 min); mniej znaczy lepiej.

| model | rekonstrukcja melowa do 12 kHz | log-mel, całe pasmo |
|---|---|---|
| najlepszy pojedynczy punkt (krok 220 tys.) | 1,1379 | 0,6252 |
| **`ae_wsm_210_220.pt`** | **1,1334** | **0,5771** |

Uśrednianie najbardziej poprawiło ciche fragmenty i pasmo powyżej 16 kHz (mniej szumu).

## Użycie

Kod (kopia blue-codec z poprawkami, MIT) jest w repo
[adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe), katalog `trening/`:

```python
import json, torch, soundfile as sf
from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.utils import LinearMelSpectrogram

ae = json.load(open("polski_100m.json"))["ae"]
ckpt = torch.load("ae_wsm_210_220.pt", map_location="cpu")
enc = LatentEncoder(cfg=ae["encoder"]).eval(); enc.load_state_dict(ckpt["encoder"])
dec = LatentDecoder1D(cfg=ae["decoder"]).eval(); dec.load_state_dict(ckpt["decoder"])
sp = ae["encoder"]["spec_processor"]                            # widmo liniowe 1025 + 228 pasm melowych = 1253
mel = LinearMelSpectrogram(sample_rate=sp["sample_rate"], n_fft=sp["n_fft"], hop_length=sp["hop_length"],
                           win_length=sp["win_length"], n_mels=sp["n_mels"])

wav, sr = sf.read("nagranie_48k.wav", dtype="float32")          # mono, 48 kHz
x = torch.from_numpy(wav)[None]
with torch.no_grad():
    z = enc(mel(x))                                             # [1, 24, klatki]
    y = dec(z)[..., : x.shape[-1]]                              # [1, 1, próbki]
sf.write("rekonstrukcja.wav", y.squeeze().numpy(), 48000)
```

## Licencje

Wagi: Apache-2.0, jak dane treningowe. Kod architektury i treningu: MIT (blue-codec, BlueTTS).
