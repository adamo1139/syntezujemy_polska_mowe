#!/usr/bin/env python3
"""Eksport MikołAI do formatu biblioteki `mikolai` (github.com/adamo1139/mikolai) i repo HF.

Wynik w --wyjscie:
  config.json             sekcje ttl, dp i dekoder z konfiguracji treningu + częstotliwość, krok, lista głosów
  model.safetensors       text_encoder.*, vf.*, uncond.*, dp.* (bez enkodera wzorca), dekoder.*, stats.mean/std
  glosy/<nazwa>.safetensors  styl głosu policzony raz z nagrania wzorcowego: styl [1, 50, 256] (model
                          tekst->latenty) i styl_dp [1, 8, 16] (predyktor długości)
Enkoder kodeka i enkoder wzorca są potrzebne tylko tutaj (do policzenia stylu), więc do repo nie trafiają.

Uruchamiać z katalogu trening/:
  python eksportuj_mikolai.py --ckpt ../modele/t2l_czyste/ema/ckpt_step_250000.pt \\
    --ae_checkpoint ../modele/kodek_48k_postnet12_slm_dlugie/ae_wsm_26_29.pt --wyjscie ../modele/hf_mikolai
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os

import soundfile as sf
import torch
from safetensors.torch import save_file

from syntezuj import wczytaj_kodek, wczytaj_t2l
from training.data.audio_utils import ensure_sr
from training.t2l.builders import build_models
from training.t2l.sampling import build_reference_only
from training.utils import compress_latents


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="punkt kontrolny modelu tekst->latenty (EMA albo WSM)")
    p.add_argument("--ae_checkpoint", required=True)
    p.add_argument("--dp", default="../modele/dp_v3/duration_predictor_wsm.pt")
    p.add_argument("--stats", default="../modele/stats.pt")
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--glos", action="append", default=None,
                   help="NAZWA=WAV (można powtórzyć); domyślnie poziomka=../glosy/poziomka_dluga.wav")
    p.add_argument("--wyjscie", required=True)
    args = p.parse_args()
    glosy = dict(g.split("=", 1) for g in (args.glos or ["poziomka=../glosy/poziomka_dluga.wav"]))

    dev = torch.device("cpu")
    cfg = json.load(open(args.config, encoding="utf-8"))
    ttl, ae = cfg["ttl"], cfg["ae"]
    sr = int(ae.get("sample_rate", 44100))
    ccf = int(ttl["chunk_compress_factor"])
    scale = float(ttl["normalizer"]["scale"])          # tak jak syntezuj.py i trener
    lang_tags = bool(cfg.get("lang_tags", False))
    with contextlib.redirect_stdout(io.StringIO()):
        te, re_, vf, uncond, dp, enc, dec, mel, hop = build_models(ttl, ae, sr, dev, dp_ckpt_path=args.dp,
                                                                  dp_cfg=cfg.get("dp", {}))
    if dp is None:
        raise SystemExit(f"Brak predyktora długości: {args.dp}")
    kodek = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    dec = wczytaj_kodek(args.ae_checkpoint, enc, dec, dev)
    dekoder_cfg = dict(kodek.get("ae_config", ae)["decoder"])
    krok, czy_ema = wczytaj_t2l(args.ckpt, te, re_, vf, uncond)
    stats = torch.load(args.stats, map_location="cpu")
    mean, std = stats["mean"].view(1, -1, 1).float(), stats["std"].view(1, -1, 1).float()

    wagi = {}
    for mod, prefiks in ((te, "text_encoder."), (vf, "vf."), (uncond, "uncond."), (dec, "dekoder.")):
        for k, v in mod.state_dict().items():
            wagi[prefiks + k] = v.detach().clone().contiguous()   # clone: wspólne tensory (tile) osobno
    for k, v in dp.state_dict().items():
        if not k.startswith("ref_encoder."):
            wagi["dp." + k] = v.detach().clone().contiguous()
    wagi["stats.mean"], wagi["stats.std"] = mean.reshape(-1).contiguous(), std.reshape(-1).contiguous()

    os.makedirs(os.path.join(args.wyjscie, "glosy"), exist_ok=True)
    for nazwa, wav in glosy.items():
        x, sr_x = sf.read(wav, dtype="float32", always_2d=True)
        x = torch.from_numpy(x.mean(1))
        x = ensure_sr(x, sr_x, sr, device=dev).reshape(1, -1) if sr_x != sr else x.view(1, -1)
        with torch.no_grad():
            z = ((compress_latents(enc(mel(x)), factor=ccf) - mean) / std) * scale
            ref_z, ref_mask = build_reference_only(z, torch.tensor([z.shape[2]]), dev, max_frames=256)
            styl = re_(ref_z, mask=ref_mask)
            styl_dp = dp.ref_encoder(ref_z, mask=ref_mask).reshape(1, 8, 16)
        save_file({"styl": styl.contiguous(), "styl_dp": styl_dp.contiguous()},
                  os.path.join(args.wyjscie, "glosy", f"{nazwa}.safetensors"))
        print(f"głos {nazwa}: {wav} ({x.shape[-1] / sr:.1f} s)")

    save_file(wagi, os.path.join(args.wyjscie, "model.safetensors"),
              metadata={"t2l": os.path.basename(args.ckpt), "krok": str(krok), "ema": str(czy_ema),
                        "kodek": os.path.basename(args.ae_checkpoint), "dp": os.path.basename(args.dp)})
    out_cfg = {"wersja": 1, "sample_rate": sr, "hop": hop, "normalizer_scale": scale, "lang_tags": lang_tags,
               "glosy": sorted(glosy),
               "ttl": ttl, "dp": cfg.get("dp", {}), "dekoder": dekoder_cfg,
               "zrodlo": {"t2l": os.path.basename(args.ckpt), "krok": krok, "ema": czy_ema,
                          "kodek": os.path.basename(args.ae_checkpoint), "dp": os.path.basename(args.dp)}}
    json.dump(out_cfg, open(os.path.join(args.wyjscie, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = {p: sum(v.numel() for k, v in wagi.items() if k.startswith(p)) for p in ("text_encoder.", "vf.", "uncond.", "dp.", "dekoder.")}
    print("parametry: " + ", ".join(f"{k.rstrip('.')} {v / 1e6:.2f}M" for k, v in n.items())
          + f"; razem {sum(n.values()) / 1e6:.1f}M -> {args.wyjscie}")


if __name__ == "__main__":
    main()
