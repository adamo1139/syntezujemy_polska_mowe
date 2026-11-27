"""Strata walidacyjna modelu tekst->latenty na klipach spoza treningu (bluetts_val.csv: 300 odłożonych rozmów).

Porównywalna między krokami, bo wszystko, co w treningu jest losowe, jest tu stałe:
  - te same klipy w tej samej kolejności i podziale na karty, te same wycinki wzorca głosu (stałe ziarno),
  - ten sam szum x_0 i stałe punkty czasu t = 0,1 / 0,3 / 0,5 / 0,7 / 0,9 dla każdego klipu,
  - pełne warunkowanie (tekst i głos, bez zrzutów CFG i bez SPFM), zwykła strata L1 flow matchingu
    (bez składników RSF, więc liczba nie skacze przy kroku włączenia RSF).
Liczona na wszystkich kartach (każda ma swoją część klipów), sumy zbierane przez all_reduce. W trybie
uśredniania wag (--avg_every) wywoływać w kroku podzielnym przez avg_every, kiedy wagi kart są identyczne.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.t2l.data_module import Text2LatentDataset, collate_text2latent
from training.t2l.sampling import build_reference_from_latents
from training.utils import compress_latents

PUNKTY_T = (0.1, 0.3, 0.5, 0.7, 0.9)


class Walidacja:
    def __init__(self, sciezka, dataset_kwargs, rank, world, device, batch_seconds, max_batch=64, seed=20261122,
                 prefiks="val"):
        self.rank, self.world, self.device, self.seed = rank, world, device, seed
        self.prefiks = prefiks   # klucze metryk: <prefiks>/loss, <prefiks>/loss_t0.1 …
        self.ds = Text2LatentDataset(sciezka, **dataset_kwargs)
        dl = self.ds.durations()
        kolejnosc = sorted(range(len(self.ds)), key=lambda i: (dl[i], i))
        moje = kolejnosc[rank::world]
        # partie po długości jak w treningu, budżet dopełnionej długości (bez gradientów, więc z zapasem)
        self.partie, cur, cur_max = [], [], 0.0
        for i in moje:
            nm = max(cur_max, dl[i])
            if cur and (nm * (len(cur) + 1) > batch_seconds or len(cur) >= max_batch):
                self.partie.append(cur); cur, nm = [], dl[i]
            cur.append(i); cur_max = nm
        if cur:
            self.partie.append(cur)
        self.n_klipow = len(self.ds)
        self.sekundy = sum(dl)

    def __call__(self, modele, przygotuj, amp_ctx, sigma_min, u_key_tile):
        """modele: (text_encoder, reference_encoder, vf_estimator) bez DDP. przygotuj: funkcja z pętli treningu
        zamieniająca partię na (z_1, valid_z_len, z_ref_full, valid_z_len_ref, is_self_ref, text_ids, text_masks)."""
        te, re_, vf = modele
        tryby = [m.training for m in modele]
        for m in modele:
            m.eval()
        np_stan = np.random.get_state()
        sumy = torch.zeros(2 * len(PUNKTY_T), dtype=torch.float64, device=self.device)
        loader = DataLoader(self.ds, batch_sampler=self.partie, num_workers=0, collate_fn=collate_text2latent)
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
                for nr, partia in enumerate(loader):
                    ziarno = self.seed + 1000 * self.rank + nr
                    np.random.seed(ziarno)
                    torch.manual_seed(ziarno)
                    z_1, valid_z_len, z_ref_full, valid_z_len_ref, is_self, text_ids, text_masks = przygotuj(partia)
                    B, C, T = z_1.shape
                    latent_mask = (torch.arange(T, device=self.device)[None] < valid_z_len[:, None]).unsqueeze(1).float()
                    z_ref, ref_mask, _, target_mask = build_reference_from_latents(
                        z_1, valid_z_len, z_ref_full, valid_z_len_ref, is_self, self.device)
                    with amp_ctx():
                        styl = re_(z_ref, mask=ref_mask)
                        context = tuple(x.to(self.device) for x in partia[9]) if len(partia) == 10 else None
                        h = te(text_ids, styl, text_mask=text_masks, context=context)
                    K = len(PUNKTY_T)
                    t = torch.tensor(PUNKTY_T, device=self.device).repeat(B)
                    z1e = z_1.repeat_interleave(K, 0)
                    lm = latent_mask.repeat_interleave(K, 0)
                    x_0 = torch.randn(B * K, C, T, device=self.device)
                    tb = t.view(-1, 1, 1)
                    x_t = ((1 - (1 - sigma_min) * tb) * x_0 + tb * z1e) * lm
                    v_cel = z1e - (1 - sigma_min) * x_0
                    with amp_ctx():
                        v = vf(noisy_latent=x_t, text_emb=h.repeat_interleave(K, 0), style_ttl=styl.repeat_interleave(K, 0),
                               latent_mask=lm, text_mask=text_masks.repeat_interleave(K, 0), current_step=t,
                               style_keys=u_key_tile(B * K))
                    maska = (lm * target_mask.repeat_interleave(K, 0)).expand(-1, C, -1)
                    blad = (F.l1_loss(v.float(), v_cel, reduction="none") * maska).sum(dim=(1, 2)).view(B, K).sum(0)
                    sumy[:K] += blad.double()
                    sumy[K:] += maska.sum(dim=(1, 2)).view(B, K).sum(0).double()
        finally:
            np.random.set_state(np_stan)
            for m, tr in zip(modele, tryby):
                m.train(tr)
        if dist.is_initialized():
            dist.all_reduce(sumy)
        K = len(PUNKTY_T)
        per_t = (sumy[:K] / sumy[K:].clamp(min=1)).tolist()
        wynik = {f"{self.prefiks}/loss": float(sumy[:K].sum() / sumy[K:].sum().clamp(min=1))}
        for tt, w in zip(PUNKTY_T, per_t):
            wynik[f"{self.prefiks}/loss_t{tt:.1f}"] = w
        return wynik
