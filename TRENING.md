# Trening: BlueTTS od zera, kodek 48 kHz, model ok. 100M

Cel pierwszego modelu (decyzje z 2026-11-21): architektura BlueTTS (schemat Supertonic), wszystko od zera,
łącznie z kodekiem audio na 48 kHz. Docelowy rozmiar przy inferencji ok. 100M:

| część | parametry | etap |
|---|---|---|
| dekoder kodeka (latenty → dźwięk) | 25,3M | 1 |
| enkoder kodeka (tylko trening) | 25,6M | 1 |
| tekst → latenty (dopasowanie przepływu, konfiguracja jak BlueV3) | ok. 75M | 2 |
| predyktor długości | 0,5M | 3 |

Kod treningowy leży w `trening/`: to kopia BlueTTS i blue-codec (MIT) z naszymi zmianami; pochodzenie
i lista zmian są w `trening/ZRODLO.md`. Jedna konfiguracja dla wszystkich etapów: `trening/config/polski_100m.json`.

**Stan (2026-11-21):** cały łańcuch (kodek → statystyki → predyktor długości → tekst→latenty) przeszedł
próbę na CPU na kilkudziesięciu sztucznych klipach: kilka kroków każdego etapu, zapis punktów kontrolnych
i próbek 48 kHz. Na GPU i na prawdziwych danych nic jeszcze nie było uruchomione z tej kopii kodu.

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
- Wynik w `dane/`: `metadane.jsonl`, `kodek_train.txt`, `kodek_val.txt`, `bluetts_train.csv`,
  `bluetts_val.csv`, `polski_100m.json` (konfiguracja z wpisanymi ścieżkami).
- Ponowne uruchomienie korzysta z `dane/metadane.jsonl` i niczego nie wypakowuje od nowa, więc działa też
  po usunięciu parquet.
- **Miejsce:** po wypakowaniu audio lokalna kopia parquet (64 GB) nie jest już potrzebna — jest na HF.
  Punkty kontrolne kodeka zajmą dalsze ok. 22 GB (patrz niżej).

## 1. Środowisko treningowe

Osobne od środowiska syntezy (trening wymaga `torch>=2.11`, synteza ma 2.8). Jedno dla wszystkich etapów:

```bash
cd trening
export UV_CACHE_DIR=/media/nvme_2tb/maked/audio/uv-cache     # nasz uv.toml tu nie sięga
uv sync
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

Wszystkie polecenia poniżej uruchamia się z katalogu `trening/`, z aktywnym tym środowiskiem.

## 2. Kodek 48 kHz

Konfiguracja: sekcja `ae` w `dane/polski_100m.json`. Względem oryginału zmienione jest tylko próbkowanie
(44100 → 48000) i długość wycinka (1,4 s = 67200 próbek); latent zostaje 24-wymiarowy z krokiem 512 próbek,
czyli 93,75 klatki/s. Partia 64 na kartę, zapis co 10 tys. kroków.

Straty: przepis domyślny, czyli ten, którym wytrenowano opublikowany kodek. Strata rekonstrukcji liczy się
w nim do 12 kHz; wyższym pasmem zajmują się dyskryminatory, które widzą cały sygnał. W nagraniach „poziomki”
powyżej 12 kHz jest ok. 0,05% energii (12–16 kHz: −32 do −36 dB, powyżej 20 kHz nic). Wariant z publikacji
(`--recon_logmel_fullband --fm_composite`) można dodać, jeśli odsłuch pokaże przytłumione głoski syczące.

Próba na jednej karcie (tempo, VRAM; brak pamięci: `--batch_size 32`):

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_autoencoder.py \
  --arch_config ../dane/polski_100m.json --checkpoint_dir ../modele/kodek_48k_proba --total_steps 300
rm -rf ../modele/kodek_48k_proba
```

Pełny trening na 6 kartach:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 train_autoencoder.py \
  --arch_config ../dane/polski_100m.json --checkpoint_dir ../modele/kodek_48k \
  --total_steps 100000 --eval_input ../glosy/poziomka_dluga.wav --eval_every 2000
```

- **Wznowienie:** to samo polecenie z `--resume ../modele/kodek_48k/ae_latest.pt`.
- **Trening rozpoczęty wcześniej z klonu `zewn/blue-codec`** jest z tym zgodny: ta sama architektura
  i format punktów kontrolnych, więc można go dokończyć tam albo wznowić stąd.
- **Liczba kroków jest szacunkiem** (decyzja 2026-11-21: 100 tys.). To ok. 256 przejść przez wszystkie klipy
  i ok. 9 h przy 3 krokach/s (zmierzone na jednej karcie: 3 kroki/s, 12,9 GB VRAM przy partii 64).
  Harmonogram uczenia jest rozpisany na `--total_steps`; przedłużenie: `--resume` z `--continue_cosine`.
- **Ocena:** co 2000 kroków (`--eval_every`) w `modele/kodek_48k/eval/` ląduje rekonstrukcja
  `poziomka_dluga.wav`; odsłuch obok oryginału jest ważniejszy niż wartości strat. Próbka to ułamek sekundy
  pracy jednej karty i nie zależy od zapisu punktów kontrolnych (te nadal co 10 tys. kroków).
- **TensorBoard:** trener sam go uruchamia na porcie 8000 z dostępem z sieci (`--bind_all`).
- **Miejsce:** punkt kontrolny ma ok. 0,74 GB i trener ich nie sprząta; 100 tys. kroków to ok. 8 GB.

## 3. Statystyki latentów

Po wytrenowaniu kodeka, jedna karta, jeden przebieg po danych:

```bash
python -m training.compute_latent_stats --tts-json ../dane/polski_100m.json \
  --ae-ckpt ../modele/kodek_48k/ae_latest.pt --metadata ../dane/bluetts_train.csv \
  --output ../modele/stats.pt --device cuda:1
```

Statystyki są związane z konkretnym kodekiem: po każdym nowym treningu kodeka trzeba je policzyć od nowa.

## 4. Predyktor długości

Mały model (0,5M), jedna karta, kilka tysięcy kroków:

```bash
python -m training.dp.cli --config ../dane/polski_100m.json --data ../dane/bluetts_train.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_latest.pt --stats_path ../modele/stats.pt \
  --out ../modele/dp --max_steps 6000 --batch_size 64 --device cuda:1
```

## 5. Model tekst → latenty (ok. 75M)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 -m training.t2l.cli \
  --config ../dane/polski_100m.json --data ../dane/bluetts_train.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_latest.pt --stats_path ../modele/stats.pt \
  --dp_checkpoint ../modele/dp/duration_predictor_final.pt --out ../modele/t2l \
  --batch_size 14 --max_steps 1000000 --inference_ref_wav ../glosy/poziomka_dluga.wav
```

- **Wznowienie:** to samo polecenie; trener sam bierze najnowszy punkt kontrolny z `--out`.
- **Wejście:** surowe znaki z zachowaniem wielkich liter (`training/data/text_vocab.py`); słownik pokrywa
  wszystkie znaki z `teksty/wybrane.jsonl`.
- **Limit długości:** `data.max_wav_sec` = 30 i `data.max_text_len` = 600 w konfiguracji. Zostaje 98% klipów
  i ok. 92% czasu; najdłuższe klipy (30–45 s) odpadają, bo partie są losowe i dopełniane do najdłuższego
  klipu, więc pojedyncze długie nagrania kilkukrotnie spowalniałyby każdy krok. Zmiana limitu to jedna liczba.
- **Próbki:** co 5000 kroków (`ttl.train.save_every`) w `modele/t2l/logs/` lądują cztery zdania z
  `eval_phrases`, w tym ze skrótem, liczbami i procentami.
- **Miejsce:** punkt kontrolny ma ok. 1,2 GB plus 0,3 GB wersji uśrednionej (EMA); trzymanych jest
  6 ostatnich (`ttl.train.keep_checkpoints`).
- **Niewiadome:** partia 14 na kartę i 1 mln kroków to wartości oryginału dla wielu języków i mówców.
  Tempo, zajętość pamięci i to, po ilu krokach mowa staje się zrozumiała, wyjdzie dopiero na rigu.
  Harmonogram uczenia (połowienie po 400, 600 i 800 tys. kroków) jest w `ttl.train.lr_milestones`.
