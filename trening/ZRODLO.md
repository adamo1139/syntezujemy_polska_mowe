# Skąd jest ten kod

Kod w tym katalogu został skopiowany z dwóch repozytoriów na licencji MIT (pliki `LICENSE-BlueTTS`
i `LICENSE-blue-codec`) i od tej chwili rozwijamy go u siebie.

| co | skąd | commit |
|---|---|---|
| `training/` | `github.com/maxmelichov/BlueTTS`, katalog `training/` | `91b1789a4a84764cdf8679fe4bb4c4d0e0e28821` |
| `config/upstream_tts.json`, `config/upstream_tts_st3.json` | BlueTTS, `config/` (bez zmian, do porównań) | jak wyżej |
| `train_autoencoder.py`, `dataset.py`, `bluecodec/` | `github.com/maxmelichov/blue-codec` (`src/bluecodec` → `bluecodec`) | `aaf9f54dc06356e5d6292a50385ed4b18bb7be97` |
| `docs/` | dokumentacja obu repozytoriów (opisuje oryginał 44,1 kHz z fonemami) | jak wyżej |

Nie skopiowano: inferencji ONNX i PyTorch (`src/`), eksportu (`exports/`), gotowych głosów, testów,
skryptu `combine_datasets.py` (fonemizacja espeak) ani treningu RL poza tym, co leży w `training/rl/`.

## Nasze zmiany względem oryginału

- **Słownik znaków** (`training/data/text_vocab.py`): surowy polski tekst zamiast fonemów IPA, z zachowaniem
  wielkich liter; normalizacja NFC i ujednolicenie cudzysłowów, myślników i odstępów.
- **Częstotliwość próbkowania z konfiguracji** zamiast 44100 wpisanego na sztywno: `training/t2l/sampling.py`
  (`set_audio_format`), `training/t2l/train_loop.py`, `training/dp/trainer.py`, `training/dp/data_module.py`.
- **Limity długości klipów z konfiguracji** (sekcja `data`: `max_wav_sec`, `max_text_len`) zamiast 20 s
  i 300 znaków na sztywno.
- **Punkty kontrolne modelu tekst→latenty**: odstęp i liczba trzymanych plików z konfiguracji
  (`ttl.train.save_every`, `ttl.train.keep_checkpoints`); oryginał zapisywał co 1000 kroków i nic nie usuwał.
- **Zdania próbne z konfiguracji** (`eval_phrases`), bez espeak.
- **Predyktor długości w próbkach treningowych** budowany z sekcji `dp` konfiguracji (`--dp_checkpoint`),
  a nie z wartości domyślnych i stałej ścieżki.
- **Pliki FLAC**: ścieżki z rozszerzeniem `.flac` nie dostają już dopisanego `.wav`.
- **Importy**: wszystko uruchamia się z katalogu `trening/` jako `python -m training...`.
- **Kodek**: próbka odsłuchowa wczytywana przez `soundfile` (bez `torchcodec` i `ffmpeg`).
- **Kodek**: `--eval_every` (próbki niezależnie od punktów kontrolnych), wczytywanie samego wycinka z pliku
  zamiast całego klipu, `--avg_every` (okresowe uśrednianie wag między kartami zamiast DDP).
- **Kodek**: nagranie prawdziwe i wygenerowane przechodzą przez każdy dyskryminator jednym wywołaniem; w kroku
  generatora wagi dyskryminatorów są zamrożone (poza DDP); `--profile`, `--num_workers`; próbka odsłuchowa
  obcinana do długości nagrania.
- **Kodek**: `--heldout_every` (ocena na stałej puli odłożonych klipów, `Loss/Heldout_*` w TensorBoard); pasek
  postępu pokazuje średnią z ostatnich 2 epok zamiast średniej od startu przebiegu.
- **Kodek**: `--constant_lr` (stałe tempo uczenia od bieżącego kroku, pod uśrednianie punktów kontrolnych) i
  `--save_every` (częstość punktów kontrolnych bez zmiany konfiguracji).
- **`usrednij_kodek.py`** (nasz): WSM dla kodeka, uśrednianie wag z kilku punktów kontrolnych z oceną na odłożonych klipach.
- **Model tekst→latenty**: `--avg_every` (okresowe uśrednianie wag zamiast DDP); proces rozproszony na CPU
  używa gloo (do prób bez GPU).
- **Predyktor długości**: wiele kart przez torchrun i DDP, `--num_workers`; bez wczytywania drugiego klipu jako
  wzorca, którego ten trener nie używa (`load_ref=False` w zbiorze danych).
- **Model tekst→latenty**: stałe tempo uczenia z rozgrzewką (`ttl.train.lr_schedule: constant`, `warmup_steps`),
  weight decay tylko na macierzach wag.
- **`usrednij_dp.py`** (nasz): WSM predyktora długości z oceną na walidacji.
- **Model tekst→latenty**: opcja XSA w self-attention enkodera tekstu (`attn_encoder.xsa`); dylatacje pola
  wektorowego `[1, 4, 16, 48]` w konfiguracji.
- **Model tekst→latenty**: `ckpt_latest.pt` co `ttl.train.latest_every` kroków, wybierany przy wznowieniu, gdy jest
  nowszy od ostatniego `ckpt_step_*.pt`.
