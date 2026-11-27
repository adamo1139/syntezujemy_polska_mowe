#!/usr/bin/env python3
"""Synteza z punktu kontrolnego modelu tekst->latenty, z siatką ustawień do porównań.

Ta sama ścieżka co próbki co 5000 kroków w treningu (sample_audio): wzorzec głosu -> styl, predyktor długości ->
liczba klatek, całkowanie Eulera pola wektorowego z CFG, dekoder kodeka. Dla każdej kombinacji punktu kontrolnego,
CFG i liczby kroków powstaje ten sam zestaw zdań; szum startowy zależy tylko od ziarna i numeru zdania, więc
różnice między plikami pochodzą wyłącznie od ustawień.

Punkt kontrolny: zwykły (ckpt_step_N.pt, bieżące wagi) albo jego wersja EMA (ema/ckpt_step_N.pt) — format ten sam.
Wyniki w --wyjscie: wg_ustawien/<wagi>_cfg<C>_k<KK>/<NN>.wav, te same pliki (twarde dowiązania) w
wg_zdan/<NN>/<wagi>_cfg<C>_k<KK>.wav do porównania jednego zdania między ustawieniami, rekonstrukcje w kodek/.
--rekonstruuj WAV ...: dodatkowo przepuszcza podane nagrania przez sam kodek (enkoder + dekoder), żeby odróżnić
wady kodeka od wad modelu tekst->latenty.

Uruchamiać z katalogu trening/:
  python syntezuj.py --ckpt ../modele/t2l/ckpt_step_115000.pt ../modele/t2l/ema/ckpt_step_115000.pt \\
    --cfg 1.5 2 3 4 --kroki 16 32 --wyjscie ../syntezy/siatka_115k
  python syntezuj.py --ckpt ../modele/t2l/ema/ckpt_step_115000.pt --cfg 2 --kroki 32 --tekst "Dzień dobry."
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

import soundfile as sf
import torch

from training.data.audio_utils import ensure_sr
from training.data.text_vocab import normalize_text, text_to_indices_multilang, wrap_lang_tags
from training.t2l.builders import build_models
from training.t2l.sampling import build_reference_only, sample_audio, set_audio_format
from training.utils import compress_latents


def wczytaj_kodek(sciezka, enc, dec, device):
    """Wagi kodeka; jeśli punkt kontrolny zapisał inną budowę dekodera (np. głowicę istft), dekoder jest budowany od nowa."""
    from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
    ckpt = torch.load(sciezka, map_location="cpu", weights_only=False)
    if "ae_config" in ckpt:
        dec = LatentDecoder1D(cfg=dict(ckpt["ae_config"]["decoder"])).to(device)
    enc.load_state_dict(ckpt["encoder"], strict=False)
    dec.load_state_dict(ckpt["decoder"], strict=True)
    enc.eval().requires_grad_(False); dec.eval().requires_grad_(False)
    return dec


def wczytaj_t2l(sciezka, te, re_, vf, uncond):
    ckpt = torch.load(sciezka, map_location="cpu")
    for mod, klucz in ((vf, "vf_estimator"), (te, "text_encoder"), (re_, "reference_encoder")):
        brak, nadmiar = mod.load_state_dict(ckpt[klucz], strict=False)
        if "context_roles.weight" in nadmiar:
            raise ValueError(f"{sciezka}: ustaw ttl.text_encoder.context_chars zgodnie z treningiem")
        if brak:
            raise SystemExit(f"{sciezka}: brak wag {klucz}: {brak[:5]}")
    for nazwa in ("u_text", "u_ref", "u_key"):
        if ckpt.get(nazwa) is not None and getattr(uncond, nazwa, None) is not None:
            getattr(uncond, nazwa).data.copy_(ckpt[nazwa])
    for m in (te, re_, vf, uncond):
        m.eval()
    return int(ckpt.get("global_step", -1)), "ema_decay" in ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="punkty kontrolne modelu tekst->latenty (zwykłe lub EMA)")
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--ae_checkpoint", default="../modele/kodek_48k/ae_wsm_210_220.pt")
    p.add_argument("--stats", default="../modele/stats.pt")
    p.add_argument("--dp", default="../modele/dp_v3/duration_predictor_wsm.pt")
    p.add_argument("--ref_wav", default="../glosy/poziomka_dluga.wav", help="wzorzec głosu")
    p.add_argument("--tekst", action="append", help="zdanie (można powtórzyć); domyślnie eval_phrases z konfiguracji")
    p.add_argument("--akapit", action="store_true",
                   help="kolejne --tekst to sąsiednie fragmenty akapitu; zapisuje też akapit.wav")
    p.add_argument("--cfg", type=float, nargs="+", default=[4.0])
    p.add_argument("--kroki", type=int, nargs="+", default=[16])
    p.add_argument("--ziarno", type=int, default=0)
    p.add_argument("--rekonstruuj", nargs="*", default=[], help="nagrania do przepuszczenia przez sam kodek")
    p.add_argument("--wyjscie", default="../syntezy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = json.load(open(args.config, encoding="utf-8"))
    ttl, ae = cfg["ttl"], cfg["ae"]
    sr = int(ae.get("sample_rate", 44100))
    hop = int(ae.get("encoder", {}).get("spec_processor", {}).get("hop_length", 512))
    set_audio_format(sr, hop)
    ccf, latent_dim = int(ttl["chunk_compress_factor"]), int(ttl["latent_dim"])
    scale = float(ttl["normalizer"]["scale"])
    lang_tags = bool(cfg.get("lang_tags", False))
    teksty = args.tekst or [str(s) for s in cfg.get("eval_phrases", [])]
    os.makedirs(args.wyjscie, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        te, re_, vf, uncond, dp, enc, dec, mel, _ = build_models(ttl, ae, sr, dev, dp_ckpt_path=args.dp, dp_cfg=cfg.get("dp", {}))
    if args.akapit and (not te.context_chars or not args.tekst):
        p.error("--akapit wymaga --tekst oraz ttl.text_encoder.context_chars > 0")
    if dp is None or not os.path.exists(args.dp):
        raise SystemExit(f"Brak predyktora długości: {args.dp}")
    dec = wczytaj_kodek(args.ae_checkpoint, enc, dec, dev)
    stats = torch.load(args.stats, map_location=dev)
    mean, std = stats["mean"].to(dev), stats["std"].to(dev)
    if mean.dim() != 3:
        mean, std = mean.view(1, -1, 1), std.view(1, -1, 1)

    # rekonstrukcje samym kodekiem (porcje 512 x 6 próbek, jak przy liczeniu latentów)
    for sciezka in args.rekonstruuj:
        x, sr_x = sf.read(sciezka, dtype="float32", always_2d=True)
        x = torch.from_numpy(x.mean(1)).to(dev)
        if sr_x != sr:
            x = ensure_sr(x, sr_x, sr, device=dev).reshape(-1)
        n = x.shape[0]
        with torch.no_grad():
            y = dec(enc(mel(torch.nn.functional.pad(x, (0, (-n) % (hop * ccf))).unsqueeze(0)))).reshape(-1)[:n]
        nazwa = os.path.splitext(os.path.basename(sciezka))[0]
        os.makedirs(os.path.join(args.wyjscie, "kodek"), exist_ok=True)
        sf.write(os.path.join(args.wyjscie, "kodek", f"{nazwa}_oryginal.wav"), x.cpu().numpy(), sr)
        sf.write(os.path.join(args.wyjscie, "kodek", f"{nazwa}_rekonstrukcja.wav"), y.clamp(-1, 1).cpu().numpy(), sr)
        print(f"kodek: {nazwa} ({n / sr:.1f} s)")

    # wzorzec głosu -> latenty -> styl (jak w próbkach treningu)
    ref, sr_ref = sf.read(args.ref_wav, dtype="float32", always_2d=True)
    ref = torch.from_numpy(ref.mean(1)).to(dev)
    ref = ensure_sr(ref, sr_ref, sr, device=dev).reshape(1, -1) if sr_ref != sr else ref.view(1, -1)

    for sciezka in args.ckpt:
        krok, czy_ema = wczytaj_t2l(sciezka, te, re_, vf, uncond)
        etykieta = f"{krok // 1000}k_{'ema' if czy_ema else 'biezace'}"
        with torch.no_grad():
            z = ((compress_latents(enc(mel(ref)), factor=ccf) - mean) / std) * scale
            ref_z, ref_mask = build_reference_only(z, torch.tensor([z.shape[2]], device=dev), dev, max_frames=256)
            styl = re_(ref_z, mask=ref_mask)
            styl_dp = dp.ref_encoder(ref_z, mask=ref_mask).reshape(1, 8, 16)
        for c in args.cfg:
            for k in args.kroki:
                t0 = time.time()
                fragmenty = []
                for i, tekst in enumerate(teksty):
                    ids = text_to_indices_multilang(normalize_text(tekst), base_lang="pl")
                    if lang_tags:
                        ids = wrap_lang_tags(ids, "pl")
                    ids = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
                    context = None
                    if args.akapit:
                        from training.t2l.models.text_encoder import pack_context
                        limit = te.context_chars
                        left = text_to_indices_multilang(normalize_text(teksty[i - 1]))[-limit:] if i else []
                        right = text_to_indices_multilang(normalize_text(teksty[i + 1]))[:limit] if i + 1 < len(teksty) else []
                        context = pack_context([(ids.new_tensor(left), ids[0], ids.new_tensor(right))])
                    torch.manual_seed(args.ziarno + i)
                    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                        wav = sample_audio(vf, te, re_, dec, ids, torch.ones(1, 1, ids.shape[1], device=dev),
                                           z_ref=None, ref_enc_mask=None, mean=mean, std=std, duration_predictor=dp,
                                           steps=k, device=str(dev), latent_dim=latent_dim, chunk_compress_factor=ccf,
                                           normalizer_scale=scale, style_ttl=styl, style_dp=styl_dp,
                                           uncond_params=uncond, cfg_scale=c, context=context)
                    ustaw = f"{etykieta}_cfg{c:g}_k{k:02d}"
                    plik = os.path.join(args.wyjscie, "wg_ustawien", ustaw, f"{i + 1:02d}.wav")
                    zdanie = os.path.join(args.wyjscie, "wg_zdan", f"{i + 1:02d}", f"{ustaw}.wav")
                    for kat in (os.path.dirname(plik), os.path.dirname(zdanie)):
                        os.makedirs(kat, exist_ok=True)
                    sf.write(plik, wav.squeeze().clamp(-1, 1).cpu().numpy(), sr)
                    if os.path.exists(zdanie):
                        os.remove(zdanie)
                    os.link(plik, zdanie)
                    if args.akapit:
                        fragmenty.append(wav.reshape(-1).clamp(-1, 1).cpu())
                if args.akapit:
                    sf.write(os.path.join(os.path.dirname(plik), "akapit.wav"), torch.cat(fragmenty).numpy(), sr)
                print(f"{etykieta}  cfg {c:g}  kroki {k:>3}: {len(teksty)} zdań w {time.time() - t0:.1f} s")
    print(f"Wyniki: {args.wyjscie}")


if __name__ == "__main__":
    main()
