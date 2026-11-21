#!/usr/bin/env python3
"""Pomiar pola wektorowego na GPU: czas przejścia w przód i wstecz w kilku wariantach i najdroższe kernele.

Odtwarza rozmiar typowego kroku treningu (domyślnie: 18 klipów × Ke 6 = 108 sekwencji po 139 klatek,
tekst 170 znaków, czyli budżet ok. 160 s) na losowych danych, bez wczytywania danych i bez optymalizatora.

Uruchamiać z katalogu trening/, na wolnej karcie:
  CUDA_VISIBLE_DEVICES=1 python profil_vf.py
"""

from __future__ import annotations

import argparse
import json

import torch

from training.t2l.builders import build_models


def zmierz(vf, wejscia, amp, n=10):
    """Średni czas przejścia w przód i wstecz (ms) z n powtórzeń po rozgrzewce."""
    t_fwd = t_bwd = 0.0
    for i in range(n + 3):
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.enable_grad()):
            out = vf(**wejscia)
        e1.record()
        out.float().square().mean().backward()
        e2.record()
        torch.cuda.synchronize()
        if i >= 3:
            t_fwd += e0.elapsed_time(e1); t_bwd += e1.elapsed_time(e2)
        vf.zero_grad(set_to_none=True)
    return t_fwd / n, t_bwd / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/polski_100m.json")
    p.add_argument("--sekwencje", type=int, default=108, help="B × Ke")
    p.add_argument("--klatki", type=int, default=139)
    p.add_argument("--znaki", type=int, default=170)
    args = p.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    _, re_enc, vf, _, _, _, _, _, _ = build_models(cfg["ttl"], cfg["ae"], 48000, dev, dp_ckpt_path="/brak", dp_cfg=cfg["dp"])
    vf.train()
    B, T, L = args.sekwencje, args.klatki, args.znaki
    with torch.no_grad():
        styl = re_enc(torch.randn(B, 144, 60, device=dev), mask=torch.ones(B, 1, 60, device=dev))
    wejscia = dict(
        noisy_latent=torch.randn(B, 144, T, device=dev), text_emb=torch.randn(B, 256, L, device=dev), style_ttl=styl,
        latent_mask=torch.ones(B, 1, T, device=dev), text_mask=torch.ones(B, 1, L, device=dev),
        current_step=torch.rand(B, device=dev),
        style_keys=vf.tile.expand(B, -1, -1) if getattr(vf, "tile", None) is not None else None,
    )
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | {B} sekwencji × {T} klatek, tekst {L}")
    print(f"{'wariant':<52} {'w przód':>9} {'wstecz':>9} {'wstecz/przód':>13}")
    warianty = [
        ("bf16, cuDNN niedeterministyczny (jak w treningu)", dict(amp=True, det=False, cudnn=True)),
        ("bf16, cuDNN deterministyczny (oryginał)", dict(amp=True, det=True, cudnn=True)),
        ("bf16, bez cuDNN (natywne sploty PyTorcha)", dict(amp=True, det=False, cudnn=False)),
        ("fp32 (TF32), cuDNN niedeterministyczny", dict(amp=False, det=False, cudnn=True)),
    ]
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    for nazwa, w in warianty:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = w["det"], w["cudnn"]
        try:
            f, b = zmierz(vf, wejscia, w["amp"])
            print(f"{nazwa:<52} {f:>7.1f} ms {b:>7.1f} ms {b / f:>12.2f}x")
        except Exception as e:   # np. brak pamięci w fp32
            print(f"{nazwa:<52} błąd: {type(e).__name__}: {str(e)[:60]}")
        torch.cuda.empty_cache()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = False, True
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vf(**wejscia)
        out.float().square().mean().backward(); vf.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vf(**wejscia)
            out.float().square().mean().backward(); vf.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    for klucz in ("self_cuda_time_total", "self_device_time_total"):
        try:
            print(prof.key_averages().table(sort_by=klucz, row_limit=20, max_name_column_width=70))
            break
        except Exception:
            continue


if __name__ == "__main__":
    main()
