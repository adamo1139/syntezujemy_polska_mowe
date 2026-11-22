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

**Trening na wielu kartach na rigu wymaga zmiennych NCCL:** `NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1
NCCL_CUMEM_HOST_ENABLE=0`. Bez nich start kończy się błędem `NCCL error ... Error while attaching to shared
memory segment /dev/shm/nccl-...`. Polecenia z `torchrun --nproc_per_node=6` poniżej już je zawierają.

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
NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 train_autoencoder.py \
  --arch_config ../dane/polski_100m.json --checkpoint_dir ../modele/kodek_48k \
  --total_steps 100000 --avg_every 10 --eval_input ../glosy/poziomka_dluga.wav --eval_every 1000
```

- **Wznowienie:** to samo polecenie z `--resume ../modele/kodek_48k/ae_latest.pt`.
- **`--avg_every 10`: uśrednianie wag zamiast synchronizacji gradientów.** Na rigu zwykły trening
  wielokartowy (DDP) przesyła gradienty między kartami po każdym kroku i dławi się na PCIe: 6 kart dawało
  1,5 kroku/s wobec 3 kroków/s na jednej. Z tą flagą każda karta trenuje własną kopię modelu na swojej części
  danych, a co 10 kroków kopie są uśredniane jednym przesłaniem (61M wartości). Punkty kontrolne i próbki
  dotyczą zawsze wag uśrednionych. Bez flagi trener działa jak w oryginale.
- **Partia:** 64 na kartę z konfiguracji (12,9 z 24 GB). Większą ustawia `--batch_size`, ale gdy karty liczą
  na 100%, tylko wydłuża to krok.
- **Trening rozpoczęty wcześniej z klonu `zewn/blue-codec`** jest z tym zgodny: ta sama architektura
  i format punktów kontrolnych, więc można go dokończyć tam albo wznowić stąd.
- **Liczba kroków jest szacunkiem** (decyzja 2026-11-21: 100 tys.). To ok. 256 przejść przez wszystkie klipy
  i ok. 9 h przy 3 krokach/s (zmierzone na jednej karcie: 3 kroki/s, 12,9 GB VRAM przy partii 64).
  Harmonogram uczenia jest rozpisany na `--total_steps`; przedłużenie: `--resume` z `--continue_cosine`.
- **Ocena:** co 1000 kroków (`--eval_every`) w `modele/kodek_48k/eval/` ląduje rekonstrukcja
  `poziomka_dluga.wav`; odsłuch obok oryginału jest ważniejszy niż wartości strat. Próbka to ułamek sekundy
  pracy jednej karty i nie zależy od zapisu punktów kontrolnych (te nadal co 10 tys. kroków).
- **TensorBoard:** trener sam go uruchamia na porcie 8000 z dostępem z sieci (`--bind_all`).
- **Miejsce:** punkt kontrolny ma ok. 0,74 GB i trener ich nie sprząta; 100 tys. kroków to ok. 8 GB.

## 3. Statystyki latentów

Po wytrenowaniu kodeka, jedna karta, jeden przebieg po danych:

```bash
python -m training.compute_latent_stats --tts-json ../dane/polski_100m.json \
  --ae-ckpt ../modele/kodek_48k/ae_wsm_210_220.pt --metadata ../dane/bluetts_train.csv \
  --output ../modele/stats.pt --device cuda:1
```

Statystyki są związane z konkretnym kodekiem: po każdym nowym treningu kodeka trzeba je policzyć od nowa.

## 4. Predyktor długości

Mały model (0,5M), kilka tysięcy kroków. Wiele kart przez zwykłe DDP: przy tak małym modelu synchronizacja
gradientów to ok. 2 MB na krok, więc PCIe riga jej nie dławi. `--batch_size` jest na kartę (16 × 6 = 96 klipów
na krok; przepis oryginału to 64–128).

```bash
NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 -m training.dp.cli \
  --config ../dane/polski_100m.json --data ../dane/bluetts_train.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --stats_path ../modele/stats.pt \
  --out ../modele/dp --max_steps 6000 --batch_size 16 --num_workers 3
```

Na jednej karcie: `python -m training.dp.cli ... --batch_size 64 --device cuda:1` (bez `torchrun`).
Co 20 kroków trener wypisuje przewidziane i prawdziwe długości kilku klipów w sekundach. Wynik:
`modele/dp/duration_predictor_final.pt`.

WSM z ostatnich punktów (z oceną na `bluetts_val.csv` każdego punktu i średniej):

```bash
python usrednij_dp.py --punkty ../modele/dp/duration_predictor_{4500,5000,5500,6000}.pt \
  --wyjscie ../modele/dp/duration_predictor_wsm.pt --device cuda:1
```

## 5. Model tekst → latenty (ok. 75M)

Najpierw latenty kodeka dla wszystkich klipów, raz (ok. 6 GB w float16 w `dane/latenty/`):

```bash
python policz_latenty.py --metadata ../dane/bluetts_train.csv ../dane/bluetts_val.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --wyjscie ../dane/latenty --device cuda:1
```

Każdy klip jest kodowany dopełniony tylko do swojej wielokrotności 3072 próbek, czyli tak jak w inferencji
(liczenie w trenerze dopełniało do najdłuższego klipu w partii, co zmieniało ostatnie klatki krótszych klipów).
Trener z `--latents ../dane/latenty` nie przepuszcza już audio przez enkoder kodeka w każdym kroku (było ok. 19%
czasu kroku) i odmówi startu, jeśli latenty pochodzą z innego enkodera niż `--ae_checkpoint`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/inductor \
TRITON_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/triton \
NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 torchrun --nproc_per_node=6 -m training.t2l.cli \
  --config config/polski_100m.json --data ../dane/bluetts_train.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --stats_path ../modele/stats.pt \
  --dp_checkpoint ../modele/dp/duration_predictor_wsm.pt --out ../modele/t2l \
  --max_steps 500000 --avg_every 50 --latents ../dane/latenty --inference_ref_wav ../glosy/poziomka_dluga.wav \
  --val_data ../dane/bluetts_val.csv \
  --wandb mikolai --wandb_name t2l-100m --compile
```

Strata walidacyjna (`--val_data`): co `ttl.train.val_every` (5000) kroków, na wszystkich klipach z
`bluetts_val.csv` (300 rozmów spoza treningu), ze stałym szumem, stałymi wycinkami wzorca i punktami czasu
t = 0,1…0,9, bez CFG, SPFM i RSF, więc liczby są porównywalne między krokami. Trafia do wandb (`val/loss`,
`val/loss_t…`) i do `../modele/t2l/walidacja.csv`. Rośnie, gdy treningowa dalej spada → przeuczenie.

Kamienie milowe: co `ttl.train.milestone_every` (50 000) kroków punkt kontrolny (z EMA) zostaje na stałe w
`../modele/t2l/kamienie/` (twarde dowiązanie, `keep_checkpoints` go nie kasuje; ok. 1,5 GB każdy). Wycofanie:
skopiuj wybrany do pustego katalogu i podaj go jako `--out` (albo przenieś nowsze punkty z `--out`).

- **Wznowienie:** to samo polecenie; trener sam bierze najnowszy punkt z `--out`: `ckpt_step_*.pt` (co 5000
  kroków, z próbkami) albo `ckpt_latest.pt` (nadpisywany co 1000 kroków, `ttl.train.latest_every`), jeśli jest
  nowszy. Przerwanie kosztuje więc najwyżej 1000 kroków.
- **`--avg_every 20`:** bez DDP; każda karta trenuje własną kopię, a co 20 kroków kopie są uśredniane (jak przy
  kodeku, bo DDP dławi się na PCIe riga). EMA jest wtedy aktualizowana tylko po uśrednieniu, z odpowiednio
  mniejszym współczynnikiem, więc obejmuje to samo okno czasowe.
- **Wejście:** surowe znaki z zachowaniem wielkich liter (`training/data/text_vocab.py`); słownik pokrywa
  wszystkie znaki z `teksty/wybrane.jsonl`.
- **Limit długości:** `data.max_wav_sec` = 30 i `data.max_text_len` = 600 w konfiguracji. Zostaje 98% klipów
  i ok. 92% czasu; najdłuższe klipy (30–45 s) odpadają, bo partie są losowe i dopełniane do najdłuższego
  klipu, więc pojedyncze długie nagrania kilkukrotnie spowalniałyby każdy krok. Zmiana limitu to jedna liczba.
- **Próbki:** co 5000 kroków (`ttl.train.save_every`) w `modele/t2l/logs/` lądują cztery zdania z
  `eval_phrases`, w tym ze skrótem, liczbami i procentami.
- **Miejsce:** punkt kontrolny ma ok. 1,2 GB plus 0,3 GB wersji uśrednionej (EMA); trzymanych jest
  6 ostatnich (`ttl.train.keep_checkpoints`).
- **Zasięg splotów (decyzja 2026-11-21):** w polu wektorowym `convnext_0.dilation_lst` = `[1, 4, 16, 48]` zamiast
  `[1, 2, 4, 8]` z BlueV3. Każda klatka widzi ±36,9 s (1153 klatki) zamiast ±9,2 s, więc cały klip do 30 s;
  zmierzone przez propagację wsteczną z jednej klatki, zasięg ciągły, bez dziur. Liczba parametrów i koszt
  obliczeń bez zmian (dylatacja nie dodaje wag).
- **XSA w enkoderze tekstu (decyzja 2026-11-21):** `ttl.text_encoder.attn_encoder.xsa` = true. Exclusive Self
  Attention (arXiv:2603.09078): z wyniku uwagi usuwana jest składowa wzdłuż własnej wartości tokenu; bez nowych
  parametrów. Dotyczy 4 warstw self-attention nad tekstem; cross-attention w polu wektorowym zostaje bez zmian
  (XSA jest zdefiniowane tylko dla self-attention). Praca testowała je wyłącznie na modelach językowych z uwagą
  przyczynową; tu uwaga jest dwukierunkowa i nad znakami, więc efekt jest nieznany.
- **Partie po długości i bf16 (decyzja 2026-11-21):** `ttl.train.batch_seconds` = 160: każda partia to klipy
  podobnej długości, a liczba_klipów × najdłuższy ≤ 160 s na kartę (przed powtórzeniem `Ke` = 6), najwyżej
  `max_batch` = 64 klipy. Przy losowych partiach 53% obliczeń szło na dopełnienie zerami, tak ok. 1%; średnio
  ok. 18 klipów na partię. `ttl.train.amp` = `bf16`: autocast dla enkoderów i pola wektorowego, strata w float32.
  Budżet zmienia `--batch_seconds`; przy OOM zmniejszyć, przy dużym zapasie pamięci zwiększyć.
- **Długość treningu i progi:** plan 100 tys. kroków (2026-11-21), potem 500 tys.; SPFM wyłączony
  (`ttl.train.spfm_enabled`, 2026-11-22: tylko ok. 0,06% brudnych klipów); RSF od 150 000 (`ttl.rsf.start_step`,
  2026-11-22, wcześniej 50 000; oryginał 200 000).
- **Weights & Biases:** `--wandb PROJEKT` (raz na rigu `wandb login`). Co `--log_every` kroków (10) średnie: strata,
  tempo uczenia, kroki/s, klipy i sekundy audio na sekundę (wszystkie karty), klipy na partię, dopełnienie,
  MFU i TFLOPS na kartę, szczyt pamięci; co epokę SPFM; co 5000 kroków próbki odsłuchowe. Sesja jest zapisana
  w `--out/wandb_id.txt`, więc wznowienie dopisuje do tego samego wykresu.
- **MFU:** FLOPs kroku ze wzoru 6·parametry·pozycje (po dopełnieniu), skalibrowanego raz licznikiem PyTorcha
  w 3. kroku przebiegu (w teście wzór zaniżał o ok. 18%); szczyt `--peak_tflops` domyślnie 153 TFLOPS na kartę. Liczone na karcie 1.
- **Diagnoza tempa:** `--profile 300` wypisuje na karcie 1 średni czas części kroku (czekanie na dane, latenty,
  enkodery, pole wektorowe, przejście wstecz, optymalizator); `--num_workers` to procesy wczytujące na kartę.
- **Optymalizator (decyzja 2026-11-21):** AdamW, tempo 5e-4 stałe po liniowej rozgrzewce przez 5000 kroków, weight
  decay 0,1 tylko na macierzach wag (bez biasów, norm i tokenów CFG). Ustawienia: `ttl.train` w konfiguracji
  (`lr`, `lr_schedule`, `warmup_steps`, `weight_decay`). Oryginał: decay 0 i połowienie tempa po 400/600/800 tys.
  kroków (`"lr_schedule": "multistep"`). Ze stałym tempem model końcowy robi się przez WSM, jak przy kodeku.
- **Niewiadome:** partia 14 na kartę i 1 mln kroków to wartości oryginału dla wielu języków i mówców.
  Tempo, zajętość pamięci i to, po ilu krokach mowa staje się zrozumiała, wyjdzie dopiero na rigu.
  Harmonogram uczenia (połowienie po 400, 600 i 800 tys. kroków) jest w `ttl.train.lr_milestones`.

## 6. Dodatkowe dane z liczbami, datami i matematyką (ok. 60 h)

Liczby w zbiorze są rzadkie, więc model czyta je źle. Dosyntezowujemy ok. 60 h tekstów z liczbami (tym samym
głosem i ustawieniami, cyfry w tekście jak dotąd) na wolnej karcie 4, a trening dalej chodzi na pozostałych.
Teksty pochodzą z rozmów spoza dotychczasowego wyboru (`--pomin teksty/wybrane.jsonl`), więc walidacja
zostaje nietknięta. Symbole spoza słownika modelu („=”, „×”, „·”, „²”, „√”, „≤”, „π” …) `przygotuj_teksty.py`
zamienia na słowa („równa się”, „razy”, „do kwadratu”, …) zamiast odrzucać zdanie, więc równania zostają w
danych; przy inferencji (przyszły `syntezuj.py`) tę samą zamianę trzeba zrobić przed modelem
(`SYMBOLE_MATEMATYCZNE` w `przygotuj_teksty.py`). Kodek tego nie wymaga: widzi tylko dźwięk, a przeczytana liczba to zwykłe słowa.

Z katalogu repo, w środowisku syntezy (`source .venv/bin/activate`):

```bash
# 1. fragmenty ze wszystkich rozmów datasetu (nie tylko górnych 10% jakości)
python skrypty/przygotuj_teksty.py --percentyl 0 --wyjscie teksty/fragmenty_wszystkie.jsonl

# 2. wybór: daty 25 h, matematyka 20 h, inne liczby 15 h (każdy pomija rozmowy poprzednich)
W="python skrypty/wybierz_teksty.py --fragmenty teksty/fragmenty_wszystkie.jsonl --czesci 1 --pomin teksty/wybrane.jsonl"
$W --wzorzec @daty --godziny 25 --wyjscie teksty/liczby_daty.jsonl
$W --pomin teksty/liczby_daty.jsonl --wzorzec @matematyka --godziny 20 --wyjscie teksty/liczby_matematyka.jsonl
$W --pomin teksty/liczby_daty.jsonl --pomin teksty/liczby_matematyka.jsonl --wzorzec @liczby --godziny 15 \
  --wyjscie teksty/liczby_inne.jsonl
cat teksty/liczby_daty.jsonl teksty/liczby_matematyka.jsonl teksty/liczby_inne.jsonl > teksty/liczby.jsonl

# 3. synteza na karcie 4 (wznawialna tym samym poleceniem)
python skrypty/generuj.py --karty 4 --teksty teksty/liczby.jsonl --wyjscie wyniki/poziomka_liczby --rownolegle 32

# 4. dołączenie do danych treningowych (stary podział train/val bez zmian)
python skrypty/przygotuj_audio.py --dodaj wyniki/poziomka_liczby
```

Szacunek godzin w `wybierz_teksty.py` liczy znaki, a cyfry czyta się dłużej niż wyglądają („2026” to ok.
30 znaków mowy), więc nagrań wyjdzie trochę więcej niż 60 h (ok. 11 GB FLAC).

Potem, w środowisku treningowym (`trening/`): latenty od nowa dla całości (stary plik zostaje do końca
liczenia; potrzeba chwilowo ok. 6,5 GB więcej) i restart treningu z punktu 5 (wznowi z `ckpt_latest.pt`):

```bash
python policz_latenty.py --metadata ../dane/bluetts_train.csv ../dane/bluetts_val.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --wyjscie ../dane/latenty --device cuda:4
```

Statystyki latentów (`stats.pt`) zostają: liczone tym samym kodekiem na 367 h, 60 h mowy tego samego głosu ich
nie zmieni.

Predyktor długości: dotrenowanie obecnego (WSM) o kolejne 6000 kroków na całości (stary zaniżał długość
tekstów z cyframi: „2026” to 4 znaki, a ok. 2 s mowy). Na karcie 4, do osobnego katalogu; wagi startowe z WSM,
optymalizator od nowa, kroki numerowane dalej (6500 … 12000). Najpierw ocena starego dla porównania:

```bash
python usrednij_dp.py --punkty ../modele/dp/duration_predictor_wsm.pt --device cuda:4
python -m training.dp.cli --config ../dane/polski_100m.json --data ../dane/bluetts_train.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --stats_path ../modele/stats.pt \
  --init ../modele/dp/duration_predictor_wsm.pt --start_step 6000 --max_steps 12000 \
  --out ../modele/dp_v2 --batch_size 96 --num_workers 6 --device cuda:4
python usrednij_dp.py --punkty ../modele/dp_v2/duration_predictor_{10500,11000,11500,12000}.pt \
  --wyjscie ../modele/dp_v2/duration_predictor_wsm.pt --device cuda:4
```

Ocena pokazuje osobno klipy z cyframi i bez; „odchylenie” ujemne = predyktor zaniża długość (mowa za szybka).
Nowy predyktor do treningu modelu tekst→latenty (próbki co 5000 kroków): `--dp_checkpoint
../modele/dp_v2/duration_predictor_wsm.pt` przy restarcie; na sam model tekst→latenty nie wpływa.

### 6a. Walidacja z liczbami

Zwykła walidacja (`bluetts_val.csv`) ma tylko ok. 80 klipów z cyframi, więc obok niej jest osobny zbiór
`bluetts_val_liczby.csv`: ok. 3 h szacunku tekstów z liczbami z rozmów, których nie ma ani w treningu, ani w
dodatkowych 150 h (predyktor długości dotrenowany na nich by je znał). Trener liczy obie straty osobno
(`val/loss` jak dotąd i `val_liczby/loss`, plik `../modele/t2l/walidacja_liczby.csv`).

```bash
# z katalogu repo, środowisko syntezy
W="python skrypty/wybierz_teksty.py --fragmenty teksty/fragmenty_wszystkie.jsonl --czesci 1 --pomin teksty/wybrane.jsonl \
  --pomin teksty/liczby_daty.jsonl --pomin teksty/liczby_matematyka.jsonl --pomin teksty/liczby_inne.jsonl"
$W --wzorzec @daty --godziny 1 --wyjscie teksty/val_liczby_daty.jsonl
$W --pomin teksty/val_liczby_daty.jsonl --wzorzec @matematyka --godziny 1 \
  --temat matematyce --temat zagadkach --temat fizyce --temat inżynierii --temat "nauce i technice" --temat programowaniu \
  --wyjscie teksty/val_liczby_matematyka.jsonl
$W --pomin teksty/val_liczby_daty.jsonl --pomin teksty/val_liczby_matematyka.jsonl --wzorzec @liczby --godziny 1 \
  --wyjscie teksty/val_liczby_inne.jsonl
cat teksty/val_liczby_{daty,matematyka,inne}.jsonl > teksty/val_liczby.jsonl
python skrypty/generuj.py --pomin 0 --teksty teksty/val_liczby.jsonl --wyjscie wyniki/poziomka_val_liczby --rownolegle 32
python skrypty/przygotuj_audio.py --dodaj-val liczby=wyniki/poziomka_val_liczby
```

`przygotuj_audio.py` bierze resztę z `dane/metadane.jsonl` (z dodanymi wcześniej 150 h), więc `--dodaj` nie
trzeba powtarzać. Potem z `trening/`: latenty dla wszystkich trzech list, ocena predyktorów na nowej walidacji
i trening z dwiema walidacjami:

```bash
python policz_latenty.py --metadata ../dane/bluetts_train.csv ../dane/bluetts_val.csv ../dane/bluetts_val_liczby.csv \
  --ae_checkpoint ../modele/kodek_48k/ae_wsm_210_220.pt --wyjscie ../dane/latenty --device cuda:1
python usrednij_dp.py --punkty ../modele/dp/duration_predictor_wsm.pt ../modele/dp_v2/duration_predictor_wsm.pt \
  --val ../dane/bluetts_val_liczby.csv --device cuda:1
# w poleceniu treningu: --val_data ../dane/bluetts_val.csv ../dane/bluetts_val_liczby.csv
```
