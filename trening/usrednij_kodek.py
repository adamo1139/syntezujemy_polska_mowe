#!/usr/bin/env python3
"""WSM dla kodeka: uśrednia wagi enkodera i dekodera z kilku punktów kontrolnych i ocenia wynik.

Punkty kontrolne powinny pochodzić z fazy stałego tempa uczenia (--constant_lr), z tego samego przebiegu.
Wynik to zwykły punkt kontrolny (klucze "encoder" i "decoder"), który czytają dalsze etapy
(statystyki latentów, predyktor długości, model tekst->latenty). Nie da się z niego wznawiać treningu
(nie ma stanu optymalizatora ani dyskryminatorów) — do tego służą same punkty kontrolne.

Ocena jest ta sama co w treningu przy --heldout_every (te same odłożone klipy, te same miary), więc wyniki
można porównać z Loss/Heldout_* w TensorBoard. Dodatkowo powstaje rekonstrukcja --eval_input do odsłuchu.

Uruchamiać z katalogu trening/:
  python usrednij_kodek.py --punkty ../modele/kodek_48k/ae_2{02..20}000.pt \\
    --wyjscie ../modele/kodek_48k/ae_wsm.pt --porownaj

Opcje:
  --porownaj         ocenia też każdy punkt kontrolny osobno (do porównania ze średnią)
  --przelicz_bn N    po uśrednieniu przelicza statystyki BatchNorm na N wycinkach z danych treningowych
                     i zapisuje obie wersje (bez i z przeliczeniem), żeby można je porównać
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import re
import time

import soundfile as sf
import torch

from dataset import TTSDataset
from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.utils import LinearMelSpectrogram
from train_autoencoder import get_mel_transforms, heldout_recon


def krok(sciezka: str) -> int:
    m = re.search(r"ae_(\d+)\.pt$", sciezka)
    return int(m.group(1)) if m else -1


def zbuduj(ae_cfg: dict, device, ckpt: dict | None = None):
    """Enkoder i dekoder; budowa dekodera z ae_config punktu kontrolnego, jeśli jest (np. inna głowica niż w konfiguracji)."""
    if ckpt is not None and "ae_config" in ckpt:
        ae_cfg = {**ae_cfg, "decoder": ckpt["ae_config"]["decoder"]}
    enc = LatentEncoder(cfg=ae_cfg["encoder"]).to(device)
    dec = LatentDecoder1D(cfg=dict(ae_cfg["decoder"])).to(device)
    return enc, dec


def wczytaj_wagi(sciezka: str) -> tuple[dict, dict, int]:
    ckpt = torch.load(sciezka, map_location="cpu", weights_only=False)
    if "decoder" not in ckpt:
        raise SystemExit(f"{sciezka}: brak wag dekodera (punkt kontrolny z trybu --encoder_only?)")
    return ckpt["encoder"], ckpt["decoder"], int(ckpt.get("step", -1))


def usrednij(stany: list[dict]) -> dict:
    """Średnia tensorów zmiennoprzecinkowych; liczniki (np. num_batches_tracked) z ostatniego punktu."""
    wynik = {}
    for k, v in stany[-1].items():
        if torch.is_floating_point(v):
            s = torch.zeros_like(v, dtype=torch.float64)
            for st in stany:
                s += st[k].to(torch.float64)
            wynik[k] = (s / len(stany)).to(v.dtype)
        else:
            wynik[k] = v.clone()
    return wynik


@torch.no_grad()
def przelicz_bn(enc, dec, mel_transform, wycinki, device, partia=16):
    """Statystyki BatchNorm od nowa, jako zwykła średnia po podanych wycinkach (wagi bez zmian)."""
    bn = [m for m in list(enc.modules()) + list(dec.modules()) if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bn:
        m.reset_running_stats()
        m.momentum = None   # średnia skumulowana zamiast wykładniczej
    enc.eval(); dec.eval()
    for m in bn: m.train()
    for i in range(0, len(wycinki), partia):
        x = torch.stack(wycinki[i : i + partia]).to(device)
        dec(enc(mel_transform(x)))
    enc.eval(); dec.eval()
    for m in bn: m.momentum = 0.1


@torch.no_grad()
def ocen(enc, dec, mel_transform, miary, odlozone, device) -> dict:
    enc.eval(); dec.eval()
    return {nazwa: heldout_recon(enc, dec, mel_transform, tf, odlozone, device) for nazwa, tf in miary.items()}


@torch.no_grad()
def probka(enc, dec, mel_transform, sciezka_wej: str, sciezka_wyj: str, sr: int, device):
    wav, sr_wej = sf.read(sciezka_wej, dtype="float32", always_2d=True)
    if sr_wej != sr:
        raise SystemExit(f"{sciezka_wej}: {sr_wej} Hz, kodek ma {sr} Hz")
    x = torch.from_numpy(wav).mean(dim=1, keepdim=True).T.to(device)   # [1, T]
    enc.eval(); dec.eval()
    y = dec(enc(mel_transform(x)))[..., : x.shape[-1]].squeeze()
    szczyt = y.abs().max().item()
    if szczyt > 1.0:
        y = y / (szczyt + 1e-8)
    sf.write(sciezka_wyj, y.cpu().numpy(), sr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--punkty", nargs="+", required=True, help="punkty kontrolne ae_*.pt (można podać wzorzec w cudzysłowie)")
    p.add_argument("--wyjscie", required=True, help="plik wynikowy, np. ../modele/kodek_48k/ae_wsm.pt")
    p.add_argument("--config", default="../dane/polski_100m.json")
    p.add_argument("--eval_input", default="../glosy/poziomka_dluga.wav")
    p.add_argument("--heldout_clips", type=int, default=64)
    p.add_argument("--heldout_max_sec", type=float, default=15.0)
    p.add_argument("--porownaj", action="store_true", help="oceń też każdy punkt kontrolny osobno")
    p.add_argument("--przelicz_bn", type=int, default=0, help="przelicz statystyki BatchNorm na N wycinkach treningowych")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sciezki = []
    for wzorzec in args.punkty:
        sciezki += sorted(glob.glob(wzorzec)) if any(c in wzorzec for c in "*?[") else [wzorzec]
    sciezki = sorted(dict.fromkeys(sciezki), key=krok)
    if len(sciezki) < 2:
        raise SystemExit(f"Do uśrednienia potrzeba co najmniej 2 punktów kontrolnych, jest {len(sciezki)}.")
    for s in sciezki:
        if not os.path.isfile(s):
            raise SystemExit(f"Nie ma pliku: {s}")

    device = torch.device(args.device)
    cfg = json.load(open(args.config, encoding="utf-8"))
    ae_cfg, data_cfg = cfg["ae"], cfg["ae"]["data"]
    sr = int(data_cfg["sample_rate"])
    spec = ae_cfg["encoder"].get("spec_processor", {})
    mel_transform = LinearMelSpectrogram(sample_rate=spec.get("sample_rate", sr), n_fft=spec.get("n_fft", 2048),
                                         hop_length=spec.get("hop_length", 512), win_length=spec.get("win_length", 2048),
                                         n_mels=spec.get("n_mels", 1253)).to(device)

    # Te same odłożone klipy i miary co --heldout_every w trenerze.
    val_ds = TTSDataset(data_cfg["val_metadata"], sample_rate=sr)
    n = min(len(val_ds), args.heldout_clips)
    idx = sorted({round(i * (len(val_ds) - 1) / max(n - 1, 1)) for i in range(n)})
    maks = int(args.heldout_max_sec * sr)
    odlozone = [val_ds[i][:maks] for i in idx]
    miary = {"recon": get_mel_transforms(data_cfg, device), "logmel_full": get_mel_transforms(data_cfg, device, "logmel_fullband")}
    print(f"Odłożone klipy: {len(odlozone)}, {sum(c.shape[-1] for c in odlozone) / sr / 60:.1f} min; "
          f"punkty kontrolne: {', '.join(os.path.basename(s) for s in sciezki)}")

    enc, dec = zbuduj(ae_cfg, device, torch.load(sciezki[-1], map_location="cpu", weights_only=False))
    stany_enc, stany_dec, kroki = [], [], []
    print(f"{'model':<28} {'recon':>9} {'log-mel całe pasmo':>19}")
    for s in sciezki:
        e, d, k = wczytaj_wagi(s)
        stany_enc.append(e); stany_dec.append(d); kroki.append(k)
        if args.porownaj:
            enc.load_state_dict(e); dec.load_state_dict(d)
            w = ocen(enc, dec, mel_transform, miary, odlozone, device)
            print(f"{os.path.basename(s):<28} {w['recon']:>9.4f} {w['logmel_full']:>19.4f}", flush=True)

    enc.load_state_dict(usrednij(stany_enc)); dec.load_state_dict(usrednij(stany_dec))
    del stany_enc, stany_dec
    wersje = [("", "średnia")]
    w = ocen(enc, dec, mel_transform, miary, odlozone, device)
    print(f"{'średnia (' + str(len(sciezki)) + ' punktów)':<28} {w['recon']:>9.4f} {w['logmel_full']:>19.4f}")

    def zapisz(sciezka: str, opis: str, wynik: dict):
        _ae = {**ae_cfg, "decoder": torch.load(sciezki[-1], map_location="cpu", weights_only=False).get("ae_config", ae_cfg)["decoder"]}
        torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "ae_config": _ae, "step": max(kroki),
                    "wsm": {"punkty": [os.path.basename(s) for s in sciezki], "opis": opis, "ocena": wynik}}, sciezka)
        if args.eval_input:
            probka(enc, dec, mel_transform, args.eval_input, os.path.splitext(sciezka)[0] + "_probka.wav", sr, device)
        print(f"  zapisano {sciezka}" + (f" (+ {os.path.splitext(sciezka)[0]}_probka.wav)" if args.eval_input else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.wyjscie)), exist_ok=True)
    zapisz(args.wyjscie, "średnia wag, statystyki BatchNorm uśrednione", w)

    if args.przelicz_bn > 0:
        t0 = time.time()
        train_ds = TTSDataset(data_cfg["train_metadata"], sample_rate=sr, segment_size=data_cfg.get("segment_size"))
        los = random.Random(0)
        wycinki = [train_ds[i] for i in los.sample(range(len(train_ds)), min(args.przelicz_bn, len(train_ds)))]
        dl = max(x.shape[-1] for x in wycinki)
        wycinki = [torch.nn.functional.pad(x, (0, dl - x.shape[-1])) for x in wycinki]
        przelicz_bn(enc, dec, mel_transform, wycinki, device)
        w_bn = ocen(enc, dec, mel_transform, miary, odlozone, device)
        print(f"{'średnia + przeliczone BN':<28} {w_bn['recon']:>9.4f} {w_bn['logmel_full']:>19.4f}  ({time.time() - t0:.0f} s)")
        zapisz(os.path.splitext(args.wyjscie)[0] + "_bn.pt", f"średnia wag, BatchNorm przeliczony na {len(wycinki)} wycinkach", w_bn)


if __name__ == "__main__":
    main()
