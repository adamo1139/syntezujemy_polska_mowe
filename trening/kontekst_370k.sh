#!/usr/bin/env bash
# Uruchom w środowisku treningowym: bash kontekst_370k.sh przygotuj, potem bash kontekst_370k.sh trenuj
set -euo pipefail
cd "$(dirname "$0")"
dane=../dane
out=../modele/t2l_opus_kontekst_370k
start=../modele/t2l_opus/ema/ckpt_step_370000.pt
kodek=../modele/kodek_48k_postnet/ae_wsm_6_9.pt
train="$dane/bluetts_train_kontekst.csv"
val="$dane/bluetts_val_kontekst.csv"
latenty="$dane/latenty_kontekst_370k"
walidacja=("$dane/bluetts_val.csv" "$dane/bluetts_val_liczby.csv" "$val")
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

case "${1:-}" in
  przygotuj)
    for split in train val; do
      opcje=()
      if [[ "$split" == val ]]; then opcje+=(--tylko-kontekst); fi
      uv run --no-sync --with moondream==2.6.1 python przygotuj_kontekst.py \
        "$dane/bluetts_${split}.csv" "$dane/bluetts_${split}_kontekst.csv" "${opcje[@]}"
    done
    python policz_latenty.py --metadata "$train" "${walidacja[@]}" \
      --ae_checkpoint "$kodek" --wyjscie "$latenty" --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6
    ;;
  trenuj)
    for plik in "$train" "${walidacja[@]}" "$latenty/indeks.json" "$latenty/latenty.npy"; do
      [[ -s "$plik" ]] || { echo "Brak $plik. Uruchom najpierw: bash kontekst_370k.sh przygotuj" >&2; exit 1; }
    done
    mkdir -p "$out"
    opcje=()
    if ! compgen -G "$out/ckpt_step_*.pt" >/dev/null && ! [[ -s "$out/ckpt_latest.pt" && -s "$out/ckpt_latest.step" ]]; then
      [[ -s "$start" ]] || { echo "Brak punktu startowego $start" >&2; exit 1; }
      opcje+=(--finetune --resume_from "$start")
    fi
    config="$out/config_kontekst.json"
    if [[ ! -e "$config" ]]; then
      jq '.ttl.text_encoder.context_chars = 64 | .ttl.train.lr = 0.00025' config/polski_100m.json > "$config.tmp"
      mv "$config.tmp" "$config"
    fi
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCHINDUCTOR_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/inductor
    export TRITON_CACHE_DIR=/media/nvme_2tb/maked/audio/cache/triton
    export NCCL_P2P_LEVEL=SYS NCCL_SHM_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0
    torchrun --nproc_per_node=7 -m training.t2l.cli "${opcje[@]}" \
      --config "$config" --data "$train" --ae_checkpoint "$kodek" --stats_path ../modele/stats.pt \
      --dp_checkpoint ../modele/dp_v3/duration_predictor_wsm.pt --out "$out" --max_steps 400000 \
      --avg_every 50 --latents "$latenty" --inference_ref_wav ../glosy/poziomka_dluga.wav \
      --val_data "${walidacja[@]}" --wandb mikolai --wandb_name t2l-100m-opus-kontekst-370k --compile
    ;;
  *) echo "Użycie: bash kontekst_370k.sh przygotuj|trenuj" >&2; exit 2 ;;
esac
