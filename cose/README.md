# COSE — postfiltr dekodera kodeka (kod bez licencji)

> **UNLICENSED.** Ten katalog to kopia https://github.com/ICDM-UESTC/COSE (commit
> `ce30e1f5b63824f92fac5597f905e5457a26964e`, 2026-08-01) z naszymi poprawkami. Repozytorium autorów **nie ma
> licencji** (brak pliku LICENSE), więc ten kod **nie jest objęty licencją projektu** i nie wolno go traktować jako
> udostępnionego na otwartej licencji. Części przejęte przez COSE z SGMSE i StoRM są na licencji MIT
> (https://github.com/sp-uhh/sgmse, https://github.com/sp-uhh/storm); FlowMSE (https://github.com/seongq/flowmse)
> również nie ma licencji. Oryginalne README autorów: `README_COSE.md`.
>
> Publikacja: Yang i in., „Compose Yourself: Average-Velocity Flow Matching for One-Step Speech Enhancement”,
> arXiv:2509.15952 (2025).

## Do czego

Jednokrokowy generatywny postfiltr: wyjście dekodera kodeka MikołAI → nagranie jak z VoxCPM2 (48 kHz). Uczony na
parach `noisy` = rekonstrukcja kodekiem, `clean` = oryginał VoxCPM2, z `trening/przygotuj_pary_cose.py`.

## Nasze zmiany (oznaczone w kodzie `[syntezujemy_polska_mowe]`)

- wczytywanie audio przez soundfile (`sgmse/util/wczytaj.py`) zamiast `torchaudio.load` (torchaudio 2.9+ wymaga torchcodec);
- własny moduł CUDA `upfirdn2d` kompilowany tylko przy `COSE_CUDA_OP=1`, domyślnie wersja w czystym PyTorch;
- opcja `--nf` (szerokość sieci `ncsnpp`: 128 = oryginał ok. 66M, 64 ≈ 16,5M, 32 ≈ 4,1M);
- walidacja przy częstotliwości innej niż 16 kHz (wynik próbkowania do numpy), metryki raz na walidację zamiast przy
  każdej porcji, rozmiar zbioru walidacyjnego `min(20, plików)` zamiast stałych 20;
- próbkowanie na urządzeniu modelu zamiast `.cuda()`; punkty kontrolne w `<log_dir>/checkpoints` zamiast bieżącego katalogu;
- `wandb` opcjonalne.
- krok COSE: przewidywania `u2` i `u1`, z których powstaje cel (i tak odcinany od gradientu), liczone bez gradientu —
  ten sam trening przy ok. 1/3 pamięci (oryginał trzymał graf trzech przejść sieci);
- opcja `--precision` (np. `bf16-mixed`).

## Środowisko (osobne, nowy PyTorch)

```bash
cd cose
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r wymagania.txt
```

## Trening (48 kHz, sieć `ncsnpp` bez atencji)

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --backbone ncsnpp --nf 128 --attn_resolutions 0 \
  --base_dir ../dane/cose_pary --log_dir ../modele/cose_nf128 \
  --sr 48000 --n_fft 1534 --hop_length 384 --num_frames 256 --batch_size 4 --num_workers 6 \
  --loss_type mean_flow --num_eval_files 20 --max_epochs 100 --save_ckpt_interval 5000
```

`--attn_resolutions 0` wyłącza atencję (w tym kodzie działa tylko przy wejściu o 256 ramkach, a przy inferencji
długość jest dowolna). Walidacja co epokę: PESQ (16 kHz), SI-SDR i ESTOI na 20 parach z `valid/`.
