#!/usr/bin/env python3
"""WSM dla predyktora długości: uśrednia wagi z kilku punktów kontrolnych i porównuje je na walidacji.

Wynik ma ten sam format co `duration_predictor_final.pt` (same wagi), więc model tekst->latenty wczytuje go
przez `--dp_checkpoint`. Ocena liczy tę samą stratę co trening (L1 logarytmu długości), na klipach z
`bluetts_val.csv`, dla każdego punktu osobno i dla średniej. Latenty walidacyjne są liczone raz, a wycinki
wzorcowe losowane z tym samym ziarnem dla każdego modelu, więc różnice wynikają tylko z wag.

Uruchamiać z katalogu trening/:
  python usrednij_dp.py --punkty ../modele/dp/duration_predictor_{4500,5000,5500,6000}.pt \\
    --wyjscie ../modele/dp/duration_predictor_wsm.pt
Bez oceny (samo uśrednienie): --bez_oceny
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
from functools import partial

import torch
from torch.utils.data import DataLoader

from training.data.text2latent_dataset import Text2LatentDataset
from training.dp.data_module import collate_dp
from training.dp.trainer import _build_dp_models, _load_dp_config
from training.utils import check_stats_provenance, compress_latents


def krok(sciezka: str) -> int:
    m = re.search(r"_(\d+)\.pt$", sciezka)
    return int(m.group(1)) if m else -1


def usrednij(stany: list[dict]) -> dict:
    wynik = {}
    for k, v in stany[-1].items():
        if torch.is_floating_point(v):
            wynik[k] = (sum(s[k].double() for s in stany) / len(stany)).to(v.dtype)
        else:
            wynik[k] = v.clone()
    return wynik


@torch.no_grad()
def policz_latenty(loader, mel_spec, ae_encoder, mean, std, scale, ccf, hop, device):
    """Raz dla całej walidacji: znormalizowane latenty, długości i teksty (jak w kroku treningu)."""
    wynik = []
    for wavs, text_ids, text_masks, lengths, _ in loader:
        z = ae_encoder(mel_spec(wavs.to(device).squeeze(1)))
        z = ((compress_latents(z, factor=ccf) - mean) / std) * scale
        valid_z_len = (lengths.to(device).float() / hop / ccf).ceil().long().clamp(min=1, max=z.shape[-1])
        wynik.append((z, valid_z_len, text_ids.to(device), text_masks.to(device)))
    return wynik


@torch.no_grad()
def ocen(model, latenty, ccf, hop, sr, device, ziarno=0) -> tuple[float, float, dict]:
    """Średnia strata L1 logarytmu długości, średni błąd względny długości (w %) i, osobno dla klipów
    z cyframi i bez, średnie odchylenie ze znakiem (w %; ujemne = predyktor zaniża długość)."""
    from training.data.text_vocab import CHAR_TO_ID
    cyfry = torch.tensor([CHAR_TO_ID[c] for c in "0123456789"], device=device)
    model.eval()
    random.seed(ziarno)
    suma_l1 = suma_wzgl = n = 0.0
    grupy = {True: [0.0, 0.0, 0], False: [0.0, 0.0, 0]}   # suma ze znakiem, suma |.|, liczba
    for z, valid_z_len, text_ids, text_masks in latenty:
        B, C, _ = z.shape
        # wycinek wzorcowy 5–95% wypowiedzi, tak samo jak w trenerze
        ref = []
        for i, L in enumerate(valid_z_len.tolist()):
            lo, hi = int(L * 0.05), int(L * 0.95)
            if hi <= lo:
                s, e = 0, L
            else:
                s = random.randint(lo, hi - 1)
                e = s + random.randint(1, max(s + 1, int(L * 0.95)) - s)
            ref.append((i, s, e))
        T = max(e - s for _, s, e in ref)
        z_ref = torch.zeros(B, C, T, device=device)
        ref_mask = torch.zeros(B, 1, T, device=device)
        for i, s, e in ref:
            z_ref[i, :, : e - s] = z[i, :, s:e]
            ref_mask[i, :, : e - s] = 1.0
        log_pred = model(text_ids=text_ids, z_ref=z_ref, text_mask=text_masks, ref_mask=ref_mask, return_log=True)
        dur_gt = valid_z_len.float() * (hop * ccf) / float(sr)
        suma_l1 += (log_pred - torch.log(dur_gt.clamp(min=1e-5))).abs().sum().item()
        wzgl = torch.exp(log_pred) / dur_gt - 1
        suma_wzgl += wzgl.abs().sum().item()
        n += B
        ma_cyfry = (torch.isin(text_ids, cyfry) & (text_masks.squeeze(1) > 0)).any(dim=1)
        for g in (True, False):
            w = wzgl[ma_cyfry == g]
            grupy[g][0] += w.sum().item(); grupy[g][1] += w.abs().sum().item(); grupy[g][2] += w.numel()
    podzial = {g: (100 * s / max(k, 1), 100 * a / max(k, 1), k) for g, (s, a, k) in grupy.items()}
    return suma_l1 / n, 100 * suma_wzgl / n, podzial


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--punkty", nargs="+", required=True, help="punkty kontrolne duration_predictor_*.pt (lub wzorzec)")
    p.add_argument("--wyjscie", help="plik średniej; bez tej flagi tylko ocena podanych punktów (wystarczy jeden)")
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--ae_checkpoint", default="../modele/kodek_48k/ae_wsm_210_220.pt")
    p.add_argument("--stats_path", default="../modele/stats.pt")
    p.add_argument("--val", default="../dane/bluetts_val.csv")
    p.add_argument("--bez_oceny", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sciezki = []
    for w in args.punkty:
        sciezki += sorted(glob.glob(w)) if any(c in w for c in "*?[") else [w]
    sciezki = sorted(dict.fromkeys(sciezki), key=krok)
    if args.wyjscie and len(sciezki) < 2:
        raise SystemExit(f"Do uśrednienia potrzeba co najmniej 2 punktów kontrolnych, jest {len(sciezki)}.")
    for s in sciezki:
        if not os.path.isfile(s):
            raise SystemExit(f"Nie ma pliku: {s}")

    stany = [torch.load(s, map_location="cpu") for s in sciezki]
    srednia = None
    if args.wyjscie:
        srednia = usrednij(stany)
        os.makedirs(os.path.dirname(os.path.abspath(args.wyjscie)), exist_ok=True)
        torch.save(srednia, args.wyjscie)
        print(f"Uśredniono {len(sciezki)} punktów ({', '.join(os.path.basename(s) for s in sciezki)}) -> {args.wyjscie}")
        if args.bez_oceny:
            return

    device = torch.device(args.device)
    dp_cfg = _load_dp_config(args.config)
    full = dp_cfg["_full"]
    stl = dp_cfg.get("style_encoder", {}).get("style_token_layer", {})
    mel_spec, ae_encoder, model = _build_dp_models(
        full, int(stl.get("n_style", 8)), int(stl.get("style_value_dim", 16)), dp_cfg.get("sentence_encoder", {}),
        dp_cfg.get("style_encoder", {}), dp_cfg.get("predictor", {}), args.ae_checkpoint, device,
        edge_fill=dp_cfg.get("edge_fill", False),
    )
    ae_encoder.eval()
    ccf = int(dp_cfg.get("chunk_compress_factor", 6))
    scale = float(dp_cfg.get("normalizer", {}).get("scale", 1.0))
    spec = full["ae"]["encoder"].get("spec_processor", {})
    sr = int(full["ae"].get("sample_rate", spec.get("sample_rate", 44100)))
    hop = int(spec.get("hop_length", 512))
    stats = torch.load(args.stats_path, map_location=device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    if mean.dim() != 3:
        mean, std = mean.view(1, -1, 1), std.view(1, -1, 1)
    print(check_stats_provenance(stats, ae_encoder.state_dict()))

    limity = full.get("data", {})
    ds = Text2LatentDataset(args.val, sample_rate=sr, max_wav_len=int(sr * float(limity.get("max_wav_sec", 20))),
                            max_text_len=int(limity.get("max_text_len", 800)), load_ref=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                        collate_fn=partial(collate_dp, spk2idx=None, unknown_spk=0))
    latenty = policz_latenty(loader, mel_spec, ae_encoder, mean, std, scale, ccf, hop, device)
    print(f"Walidacja: {len(ds)} klipów")
    print(f"{'model':<34} {'strata (L1 log)':>15} {'błąd długości':>14} | {'z cyframi: błąd':>15} {'odchylenie':>10} "
          f"| {'bez cyfr: błąd':>14} {'odchylenie':>10}")

    def wiersz(nazwa, st):
        model.load_state_dict(st)
        l1, wzgl, g = ocen(model, latenty, ccf, hop, sr, device)
        (sc, ac, kc), (sb, ab, kb) = g[True], g[False]
        print(f"{nazwa:<34} {l1:>15.4f} {wzgl:>13.2f}% | {ac:>14.2f}% {sc:>+9.2f}% | {ab:>13.2f}% {sb:>+9.2f}%")
        return kc, kb

    for s, st in zip(sciezki, stany):
        kc, kb = wiersz(os.path.basename(s), st)
    if srednia is not None:
        kc, kb = wiersz(f"średnia ({len(sciezki)} punktów)", srednia)
    print(f"(klipów z cyframi: {kc}, bez cyfr: {kb}; odchylenie ujemne = predyktor zaniża długość, mowa za szybka)")


if __name__ == "__main__":
    main()
