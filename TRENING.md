# Trening: BlueTTS od zera, kodek 48 kHz, model ok. 100M

Cel pierwszego modelu (decyzje z 2026-11-21): architektura BlueTTS (schemat Supertonic), wszystko od zera,
łącznie z kodekiem audio na 48 kHz. Docelowy rozmiar przy inferencji ok. 100M:

| część | parametry | etap |
|---|---|---|
| dekoder kodeka (latenty → dźwięk) | 25,3M | 1 |
| enkoder kodeka (tylko trening) | 25,6M | 1 |
| tekst → latenty (dopasowanie przepływu, konfiguracja jak BlueV3) | ok. 75M | 2 |
| predyktor długości | 0,5M | 3 |

Kod treningowy jest cudzy (MIT), przypięty do konkretnych commitów; nasze są dane, konfiguracje i ten opis.

**Stan (2026-11-21):** gotowy jest etap 0 (dane) i etap 1 (kodek). Trener kodeka sprawdzony tylko na CPU
(5 kroków przy 48 kHz, oba warianty strat, zapis punktów kontrolnych); na rigu jeszcze nie był uruchomiony.
Etapy 2–3 nie są jeszcze przygotowane.

## 0. Dane

W środowisku syntezy (ma `pyarrow`):

```bash
source .venv/bin/activate
python skrypty/przygotuj_audio.py
deactivate
```

- Jeśli pliki po syntezie nadal leżą w `wyniki/poziomka/audio/`, skrypt używa ich w miejscu. Jeśli zostały
  usunięte, wypakowuje klipy z `wyniki/poziomka_parquet/data/` do `dane/audio/` (ok. 61 GB, wznawialne).
  Bez lokalnego parquet: `--repo HoldForQuake/VoxCPM2-pol-one-voice-367h`.
- Walidacja to 300 całych rozmów (`--rozmow-val`), wybranych stale po skrócie `zrodlo_id`.
- Wynik w `dane/`: `metadane.jsonl`, `kodek_train.txt`, `kodek_val.txt`, `kodek_48k.json`.
- **Miejsce:** po wypakowaniu audio lokalna kopia parquet (64 GB) nie jest już potrzebna — jest na HF.
  Punkty kontrolne kodeka zajmą dalsze ok. 22 GB (patrz niżej).

## 1. Kodek 48 kHz

### 1.1 Kod i środowisko

Osobne środowisko: kodek wymaga Pythona 3.12 i `torch>=2.11`, a synteza ma torch 2.8.

```bash
mkdir -p zewn
git clone https://github.com/maxmelichov/blue-codec.git zewn/blue-codec
git -C zewn/blue-codec checkout aaf9f54dc06356e5d6292a50385ed4b18bb7be97
cd zewn/blue-codec
export UV_CACHE_DIR=/media/nvme_2tb/maked/audio/uv-cache     # nasz uv.toml tu nie sięga
uv sync --extra cu128
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

Do odsłuchów w trakcie treningu (`--eval_input`) potrzebny jest `ffmpeg` w systemie; bez niego trening
działa, tylko zamiast próbek pojawia się „Evaluation failed”.

### 1.2 Co jest zmienione względem oryginału

Tylko konfiguracja (`trening/kodek_48k.json`), bez zmian w kodzie:

- `sample_rate` 44100 → 48000 (dane, enkoder), `segment_size` 61740 → 67200 (1,4 s);
- latent zostaje 24-wymiarowy z krokiem 512 próbek, czyli 93,75 klatki/s (oryginał: 86);
- `batch_size` 64 na kartę (oryginał 128 łącznie), `save_interval` 10000.

Straty: `--recon_logmel_fullband --fm_composite`. Domyślny przepis liczy rekonstrukcję tylko do 12 kHz,
a przy 48 kHz chcemy, żeby obejmowała całe pasmo.

### 1.3 Próba na jednej karcie

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_autoencoder.py \
  --arch_config ../../dane/kodek_48k.json --checkpoint_dir ../../modele/kodek_48k_proba \
  --total_steps 300 --recon_logmel_fullband --fm_composite
```

Zanotuj tempo (kroki/s z paska postępu) i zajętość VRAM. Brak pamięci: `--batch_size 32`.
Potem `rm -rf ../../modele/kodek_48k_proba` (każdy punkt kontrolny ma ok. 0,74 GB).

### 1.4 Pełny trening na 6 kartach

```bash
tmux new -s kodek
cd zewn/blue-codec && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 train_autoencoder.py \
  --arch_config ../../dane/kodek_48k.json --checkpoint_dir ../../modele/kodek_48k \
  --total_steps 300000 --recon_logmel_fullband --fm_composite \
  --eval_input ../../glosy/poziomka_dluga.wav
```

- **Wznowienie:** to samo polecenie z `--resume ../../modele/kodek_48k/ae_latest.pt`.
- **Liczba kroków jest szacunkiem.** Oryginał to 1,5 mln kroków przy partii 128 na wielu mówcach; tu jest
  jeden głos i partia 384 (6 × 64), więc 300 tys. powinno wystarczyć, ale tego nikt nie zmierzył. Harmonogram
  uczenia (cosinus) jest rozpisany na `--total_steps`, więc liczbę trzeba ustalić przed startem.
- **Ocena:** co 10 tys. kroków w `modele/kodek_48k/eval/` ląduje rekonstrukcja `poziomka_dluga.wav` —
  odsłuch obok oryginału jest tu ważniejszy niż wartości strat. Wykresy: TensorBoard.
- **Uwaga na TensorBoard:** trener sam go uruchamia na porcie 8000 z dostępem z sieci (`--bind_all`).
- **Miejsce:** trener trzyma do 1000 punktów kontrolnych; przy 300 tys. kroków i zapisie co 10 tys. to
  30 plików po ok. 0,74 GB, czyli ok. 22 GB. Starsze można usuwać ręcznie (zostawić `ae_latest.pt`).
- **Dysk i wentylator:** trening czyta 150 tys. plików FLAC bez przerwy; nawiew na dysk jak przy syntezie.

## 2–3. Tekst → latenty i predyktor długości

Do przygotowania po wytrenowaniu kodeka (kod: `maxmelichov/BlueTTS`, commit `91b1789`). Do rozstrzygnięcia
przed etapem 2:

- **Wejście tekstowe.** BlueTTS czyta fonemy IPA z espeak-ng. espeak sam rozwija liczby, ale skróty czyta
  źle („VAT” → „fałat”), a nauczyciel czytał surowy tekst poprawnie. Alternatywa: słownik znakowy (surowy
  tekst), jak w oryginalnym Supertonic — wymaga zmiany w kodzie słownika.
- **Statystyki latentów** trzeba policzyć nowym kodekiem; klatka po kompresji ×6 to 15,6 Hz zamiast 14,4.
