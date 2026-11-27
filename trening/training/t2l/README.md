# Light-BlueTTS Training

This folder contains the training loop and modules for Light-BlueTTS.

## Data Format
The dataset should be a CSV file with the following expected columns (or similar, depending on your dataset loader setup):
- `audio_path`: Path to the .wav file
- `text`: Transcription text
- `speaker_id`: Speaker identifier
- `lang`: Language code
- `sample_weight` (optional): per-row draw weight; rows are drawn with replacement by it, split across DDP ranks.

Expected sample rate for training is typically 44.1kHz.

## Example Data
Create a small file in `data/sample/train.csv`:
```csv
audio_path,text,speaker_id,lang
data/sample/wavs/1.wav,"שלום עולם",spk1,he
data/sample/wavs/2.wav,"מה שלומך",spk1,he
data/sample/wavs/3.wav,"מזג אוויר יפה היום",spk1,he
```

## Running Training
From the **main repository root**, with a venv that includes the training stack (Python deps for `training/`, e.g. PyTorch and pandas, per your `uv` setup):

```bash
uv run python -m training.t2l.cli --config config/tts.json --data generated_audio/combined_dataset_cleaned.csv --out runs/exp1
```

Adjust config and paths to your layout. The `blue-onnx` PyPI package is inference-only and does not install this CLI; use a full clone for training.

## Config knobs
The `configs/tts.json` has several settings:
- `latent_dim`, `chunk_compress_factor`: Autoencoder latent dimensions and compression factor.
- `text_encoder`, `vector_field`: Architectural dimensions and layers.
- `flow_matching.sig_min`: The minimum noise scale for Flow Matching.
- `batch_expander.n_batch_expand`: The $K_e$ factor. 
- `train.weight_decay` (optional, default `0.01`): AdamW weight decay (RobustSpeechFlow, arXiv 2605.22083, uses `0.0`).
- `train.lr_milestones` (optional, default `[300000, 600000]`): steps at which the learning rate is halved; a resume uses these, not the checkpoint's.
- `train.ema_decay` (optional, default `0` = off): EMA of the text encoder, vector field, reference encoder and the CFG tokens (u_text/u_ref/u_key), warm-up `min(ema_decay, (1 + n) / (10 + n))`. Checkpoints store it under `ema`; `<out>/ema/ckpt_step_N.pt` holds the EMA weights as an ordinary checkpoint.
- `rsf`: RobustSpeechFlow contrastive loss (arXiv 2605.22083), off by default; from `start_step` it subtracts L1 distances to an in-batch negative (`lambda_rand`) and a repeat/skip corruption of the target (`lambda_aug`). See `training/t2l/rsf.py`.

`config/tts_st3.json` is the 64M (Supertonic-3) layout; pass it with `--config` to train or export
that model. `config/tts.json` stays the 33M model.
`layer_scale_init` sets the initial ConvNeXt gamma of the TE, reference encoder and VF (default 1e-6;
`tts_st3.json` uses 0.1, and on the DP sentence encoder only via `dp.sentence_encoder.layer_scale_init`).
`edge_fill` (in `ttl` and `dp`) fills each row's padding with its last valid column inside every ConvNeXt,
so a padded training row computes what it computes alone at inference.
`speech_prompted_text_encoder.style_key_mode: "projected"` computes the TE style keys as
`tanh(W_key(style_key))`, where `style_key` is shared with the VF `tile`; `uncond_masker.style_key_uncond`
adds `u_key`, the style keys of every style-dropped row.

## Healthy Loss Curve
A healthy Flow Matching training loss curve will start high and exponentially decay, flattening out over time. If using SPFM, you will see it drop further when fine-tuning kicks in.

### Dotrenowanie T2L z kontekstem sąsiednich fragmentów

W kopii konfiguracji ustaw `ttl.text_encoder.context_chars` na maksymalną liczbę znaków z **każdej** strony, np. 64. Wartość 0 (domyślna) zachowuje dotychczasową architekturę. Przy pierwszym dotrenowaniu podaj plik EMA przez `--finetune --resume_from` i wybierz nowy katalog `--out`. Przy wznowieniu trener daje pierwszeństwo punktowi z `--out`. Krok końcowy `--max_steps` musi przekraczać numer kroku checkpointu. Do porównania samego kontekstu ustaw `ttl.rsf.enabled` i `ttl.train.spfm_enabled` na `false`.

Do `--data` i `--val_data` podaj CSV z granicami fragmentów oryginalnego nagrania. Przykład dla pliku 48 kHz:

```csv
wav_path,speaker_id,lang,phonemized,text,start_frame,end_frame
rozdzial.wav,1,pl,True,Było już późno.,0,96000
rozdzial.wav,1,pl,True,Ktoś zapukał.,96000,192000
rozdzial.wav,1,pl,True,Otworzyłam drzwi.,192000,312000
```

`start_frame` jest włącznie, `end_frame` wyłącznie, oba w próbkach przy oryginalnej częstotliwości pliku. Granicę wyznacz między słowami, a pauzę uwzględnij dokładnie raz. Trener dobiera tekst bezpośrednio przyległych przedziałów z tego samego pliku i od tego samego mówcy. Przy włączonym kontekście wzorzec głosu w treningu i walidacji pochodzi z wycinka tego samego celu. Przy dodatnim `ttl.text_encoder.history_frames` trener dodatkowo odczytuje końcówkę bezpośrednio poprzedzającego przedziału jako warunek predykcji celu. Pierwszy fragment dostaje zera. Kolejność wierszy jest dowolna. W polskim zbiorze znakowym `phonemized=True` pomija eSpeak. Przy podziale na trening i walidację trzymaj całe nagranie w jednym zbiorze.

Jeżeli przedziały są już zapisane jako osobne klipy, użyj `wav_path,text,speaker_id,lang,phonemized,left_context,right_context`. Kolumna `previous_audio` wskazuje poprzedni fragment tego samego nagrania. Dla starszych manifestów z `przygotuj_kontekst.py` trener odtwarza to powiązanie z nazw plików. W tym wariancie można użyć `--latents` z latentami tych klipów. Puste pola oznaczają brak danego sąsiada. Dla niezależnych klipów VoxCPM2 zostaw oba pola puste. Przykłady przejść przygotuj z ciągłych nagrań wielozdaniowych.

Eksportuj model z tą samą konfiguracją co trening. Po wczytaniu wyeksportowanego modelu `MikolAI.synteza(tekst)` przy włączonej historii generuje fragmenty po kolei, przekazując 32 końcowe ramki latentów (2,048 s dla tego kodeka) do generacji kolejnego fragmentu. Domyślnie zachowuje wygenerowane pauzy. Do odsłuchu bez eksportu użyj `syntezuj.py --akapit --tekst "Było już późno." --tekst "Ktoś zapukał."` razem ze zwykłymi opcjami `--config` i `--ckpt`. Wynik `akapit.wav` znajdziesz obok próbek dla danego ustawienia.

#### Przebieg od EMA 370k do 500k na siedmiu GPU

Polecenie z katalogu `trening/`. Plik `bluetts_train_kontekst.csv` zawiera krótkie klipy oraz wyrównane fragmenty z kolumnami `left_context,right_context`. Przy `--latents` wszystkie te klipy i klipy walidacyjne muszą być w indeksie latentów. Jeśli nowe fragmenty mają być czytane wprost z długich nagrań przez `start_frame,end_frame`, pomiń `--latents`. Dwa dotychczasowe zbiory zachowują nazwy metryk. Dodatkowy pomiar na wyrównanych fragmentach trafia do `val_kontekst/loss`.

```bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/inductor \
TRITON_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/triton \
NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 -m training.t2l.cli \
  --config config/polski_100m_kontynuacja.json --data ../dane/bluetts_train_kontekst.csv \
  --ae_checkpoint ../modele/kodek_48k_postnet/ae_wsm_6_9.pt --stats_path ../modele/stats.pt \
  --dp_checkpoint ../modele/dp_v3/duration_predictor_wsm.pt \
  --finetune --resume_from ../modele/t2l_opus/ema/ckpt_step_370000.pt \
  --out ../modele/t2l_opus_kontynuacja_370k --max_steps 500000 --avg_every 50 \
  --latents ../dane/latenty_kontekst_370k --inference_ref_wav ../glosy/poziomka_dluga.wav \
  --val_data ../dane/bluetts_val.csv ../dane/bluetts_val_liczby.csv ../dane/bluetts_val_kontekst.csv \
  --wandb mikolai --wandb_name t2l-100m-opus-kontynuacja-370k --compile
```

`--finetune` inicjalizuje nowy optymalizator z LR 0,00025. Po przerwaniu rozpoczętego przebiegu użyj tego samego `--out` i pomiń `--finetune`, aby przywrócić stan optymalizatora.

#### Próbki przy zapisie checkpointu

Co `ttl.train.save_every` kroków trener zapisuje jeden WAV na prompt z głosem z `--inference_ref_wav`. Przy braku tego pliku używa pierwszej dostępnej referencji z partii walidacyjnej. Próbki trafiają do `logs/` i, przy włączonym W&B, do `probki/`.
