import sys
import os
import time
import copy
import collections
import json
import math
import subprocess
import atexit
import argparse
import random
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from dataset import TTSDataset, collate_fn, collate_fn_edge
from bluecodec.audio_utils import ensure_sr
from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D, supertonic3_decoder_state
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram, compress_latents

HOP, CHUNK = 512, 3072   # latent hop; edge-aware layout: hop x compression 6

def _mod(m): return m.module if isinstance(m, DDP) else m

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_for_nan_inf(loss, name, logger):
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(f"🚨 {name} detected NaN/Inf")
        return True
    return False

def setup_logger(save_dir, rank):
    logger = logging.getLogger("train_ae")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    count = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=fmap_r[0][0].device)

def feature_loss_composite(*fmap_pairs):
    # Paper Eq. (6): one average over the layers of MPD and MRD together (D is the composite).
    # feature_loss(mpd) + feature_loss(mrd) averages each group separately, i.e. lambda_fm 0.2.
    return feature_loss([d for r, _ in fmap_pairs for d in r], [d for _, g in fmap_pairs for d in g])

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr.float() - 1.0) ** 2)
        g_loss = torch.mean((dg.float() + 1.0) ** 2)
        loss += (r_loss + g_loss)
    return loss

class SLMHead(nn.Module):
    """Dyskryminator SLM jak w StyleTTS2 (WavLMDiscriminator): na zamrożonych cechach WavLM ze wszystkich warstw
    transformera, sklejonych w kanałach [B, L*768, T]; 1×1 do `kanaly`, dwie konwolucje 5, wyjście 1 kanał."""
    def __init__(self, warstwy=12, wymiar=768, kanaly=64):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.pre = wn(nn.Conv1d(warstwy * wymiar, kanaly, 1))
        self.convs = nn.ModuleList([wn(nn.Conv1d(kanaly, kanaly * 2, 5, padding=2)),
                                    wn(nn.Conv1d(kanaly * 2, kanaly * 4, 5, padding=2))])
        self.post = wn(nn.Conv1d(kanaly * 4, 1, 3, padding=1))

    def forward(self, cechy):   # cechy: lista L tensorów [B, T, 768]
        x = F.leaky_relu(self.pre(torch.cat(cechy, dim=-1).transpose(1, 2)), 0.1)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
        return self.post(x)

class SLM:
    """Zamrożony WavLM-Base+ (torchaudio, MIT) i uczona głowica SLMHead; tylko w treningu (przy inferencji nic).
    Strata generatora: przeciwstawna (LSGAN, jak MPD/MRD) i dopasowanie cech WavLM (średnie L1 po warstwach).
    Liczone na całym segmencie (nie na krótkim wycinku dla MPD/MRD), dla pierwszych `partia` klipów porcji."""
    def __init__(self, device, sr, w_adv, w_fm, partia, kanaly=64):
        self.wavlm = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().to(device).eval().requires_grad_(False)
        self.head = SLMHead(len(self.wavlm.encoder.transformer.layers), 768, kanaly).to(device)
        self.sr, self.w_adv, self.w_fm, self.partia = sr, w_adv, w_fm, partia

    def cechy(self, wav):   # wav [B, 1, N] przy self.sr -> lista warstw [B, T, 768]
        x = torchaudio.functional.resample(wav[: self.partia].squeeze(1).float(), self.sr, 16000)
        return self.wavlm.extract_features(x)[0]

class LogMelFullBand(nn.Module):
    """Paper reconstruction mel: LOG magnitude up to sr/2 (the default loss is linear, capped at 12 kHz)."""
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, eps=1e-5):
        super().__init__()
        self.mel = MelSpectrogramNoLog(sample_rate, n_fft, win_length, hop_length, n_mels, f_min=0, f_max=sample_rate // 2)
        self.eps = eps
    def forward(self, x): return torch.log(self.mel(x).clamp_min(self.eps))

def get_mel_transforms(data_cfg, device, recon="linmel_12k"):
    # linmel_12k: linear mels up to 12 kHz (default); linmel_fullband: up to sr/2 (paper); logmel_fullband: log mels up to sr/2
    sr = data_cfg['sample_rate']
    mel_configs = [(1024, 256, 1024, 64), (2048, 512, 2048, 128), (4096, 1024, 4096, 128)]
    if recon == "logmel_fullband":
        return [LogMelFullBand(sr, n_fft, hop, win, n_mels).to(device) for (n_fft, hop, win, n_mels) in mel_configs]
    f_max = {"linmel_12k": 12000, "linmel_fullband": sr // 2}[recon]
    # by keyword: the signature is (sample_rate, n_fft, win_length, hop_length), positionally window and hop were swapped
    return [MelSpectrogramNoLog(sr, n_fft=n_fft, win_length=win, hop_length=hop, n_mels=n_mels, f_max=f_max).to(device)
            for (n_fft, hop, win, n_mels) in mel_configs]

def chunk_ceil(n): return (n + CHUNK - 1) // CHUNK * CHUNK

@dataclass
class ExtraTerms:
    """Optional generator terms; all weights 0 is the default recipe."""
    tail_w: float = 0.0
    tail_norm: tuple = None      # (mean, std) [1, 144, 1] of the compressed latents
    hiband_w: float = 0.0
    hiband_bin: int = 558        # pierwszy prążek STFT 2048 liczony do straty pasma wysokiego
    hiband_floor: float = 1e-5   # podłoga amplitudy przed logarytmem
    linie_w: float = 0.0         # strata prążków co sr/HOP (ślad sklejania klatek dekodera)
    floor_w: float = 0.0
    floor_db: float = -60.0

def tail_consistency_loss(encoder, mel_transform, audio, z, lengths, tail_norm):
    """Relative L2 between each clip's last compressed latent frame and the same frame encoded with two extra
    chunks of silence after the batch (the target, without gradient)."""
    with torch.no_grad():
        # the bare module: a DDP forward re-broadcasts buffers in place and would break the first pass's graph
        z_ref = _mod(encoder)(mel_transform(F.pad(audio.squeeze(1), (0, 2 * CHUNK))))[..., : z.shape[-1]]
    mean, std = tail_norm
    last = chunk_ceil(lengths) // CHUNK - 1
    rows = torch.arange(z.shape[0], device=z.device)
    a = ((compress_latents(z, 6) - mean) / std)[rows, :, last]
    b = ((compress_latents(z_ref, 6) - mean) / std)[rows, :, last]
    return ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-3)).mean()

def hiband_loss(y_hat, audio, n_fft=2048, min_bin=558, floor=1e-5):
    """L1 of the log-magnitude STFT from bin `min_bin` up (558 of 2048: ok. 12 kHz przy 44,1 kHz, 13 kHz przy 48 kHz).
    `floor`: amplituda, poniżej której różnice się nie liczą (niższa = kara także za cichy szum tam, gdzie oryginał
    prawie nic nie ma)."""
    win = torch.hann_window(n_fft, device=audio.device)
    def logmag(x):
        S = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=HOP, window=win, return_complex=True).abs()
        return torch.log(S[:, min_bin:, :].clamp_min(floor))
    return F.l1_loss(logmag(y_hat), logmag(audio))

_LINIE_IDX = {}

def linie_loss(y_hat, audio, sr=48000, hop=HOP, kmax=40, szer=2, tlo=100):
    """Prążki na wielokrotnościach sr/hop (93,75 Hz przy 48 kHz i hop 512): ślad sklejania kawałków dekodera.
    Widmo całego fragmentu (pochodna sygnału, okno Hanna); dla każdego k wysokość prążka = maks. amplitudy w ±szer
    prążkach wokół k·sr/hop, podzielona przez średnią w ±tlo prążkach (lokalne tło). Strata: średni NADMIAR
    logarytmu tej wysokości ponad prawdziwe nagranie (k ≤ 40, do 3,75 kHz) — prawdziwe harmoniczne głosu w tych
    miejscach są w celu, więc nie są karane. Na 4 klipach VoxCPM2 (fragmenty 1,4 s): rekonstrukcja starym dekoderem
    0,290, z modułem poprawki 0,253; synteza 0,414 / 0,352 (wersja L1 do k = 200 ich nie odróżniała)."""
    x, r = y_hat.squeeze(1).float(), audio.squeeze(1).float()
    n = x.shape[-1] - 1
    if n not in _LINIE_IDX:
        df = sr / n
        k = torch.arange(1, kmax + 1)
        c = torch.round(k * sr / hop / df).long()
        c = c[c + tlo < n // 2]
        _LINIE_IDX[n] = (c[:, None] + torch.arange(-szer, szer + 1)[None], c[:, None] + torch.arange(-tlo, tlo + 1)[None])
    szczyt_i, tlo_i = (t.to(x.device) for t in _LINIE_IDX[n])
    okno = torch.hann_window(n, device=x.device)
    def wysokosc(s):
        A = torch.fft.rfft(torch.diff(s, dim=-1) * okno).abs()
        return torch.log(A[:, szczyt_i].amax(-1).clamp_min(1e-8)) - torch.log(A[:, tlo_i].mean(-1).clamp_min(1e-8))
    return (wysokosc(x) - wysokosc(r)).clamp_min(0).mean()

@torch.no_grad()
def heldout_utmos(encoder, decoder, mel_transform, clips, device, model, sr):
    """Średni UTMOS (przewidywana ocena słuchaczy) rekonstrukcji odłożonych klipów."""
    from training.utmos import utmos
    encoder.eval(); decoder.eval()
    w = []
    for wav in clips:
        y = decoder(encoder(mel_transform(wav.to(device).unsqueeze(0)))).reshape(-1)[: wav.shape[-1]]
        w.append(utmos(model, y.clamp(-1, 1), sr))
    return sum(w) / len(w)

@torch.no_grad()
def heldout_linie(encoder, decoder, mel_transform, clips, device):
    """Średnia linie_loss na odłożonych klipach (całe klipy): nadmiar prążków co 93,75 Hz ponad oryginał."""
    encoder.eval(); decoder.eval()
    w = []
    for wav in clips:
        wav = wav.to(device).unsqueeze(0)
        y = decoder(encoder(mel_transform(wav))).reshape(1, -1)
        n = min(y.shape[-1], wav.shape[-1])
        w.append(linie_loss(y[None, :, :n], wav[None, :, :n]).item())
    return sum(w) / len(w)

def silence_floor_loss(y_hat, audio, floor_db, mask=None, frame=1024):
    """On frames whose source is below `floor_db` dBFS, the decoded level above `floor_db`, in units of 20 dB."""
    n = audio.shape[-1] // frame
    def level_db(x): return 10 * x[..., : n * frame].reshape(x.shape[0], n, frame).pow(2).mean(-1).add(1e-12).log10()
    silent = (level_db(audio) < floor_db).float()
    if mask is not None: silent = silent * mask[:, 0, : n * frame : frame]
    return (torch.relu(level_db(y_hat) - floor_db) * silent).sum() / silent.sum().clamp_min(1.0) / 20.0

def extra_losses(terms, encoder, mel_transform, audio, y_hat, z, lengths, mask):
    """{name: (weight, loss)} for the enabled ExtraTerms."""
    out = {}
    if terms.tail_w > 0: out["tail"] = (terms.tail_w, tail_consistency_loss(encoder, mel_transform, audio, z, lengths, terms.tail_norm))
    if terms.hiband_w > 0: out["hiband"] = (terms.hiband_w, hiband_loss(y_hat, audio, min_bin=terms.hiband_bin, floor=terms.hiband_floor))
    if terms.linie_w > 0: out["linie"] = (terms.linie_w, linie_loss(y_hat, audio))
    if terms.floor_w > 0: out["floor"] = (terms.floor_w, silence_floor_loss(y_hat, audio, terms.floor_db, mask))
    return out

class Prof:
    """Pomiar czasu części kroku treningu (--profile N): średnie z kroków po rozgrzewce, z synchronizacją GPU."""
    def __init__(self, device, warmup=20):
        self.device, self.warmup, self.t, self.n, self.sums, self.order = device, warmup, None, 0, {}, []
    def _now(self):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        return time.perf_counter()
    def start(self): self.t = self._now(); self.n += 1
    def mark(self, name):
        now = self._now()
        if self.n > self.warmup:
            if name not in self.sums: self.sums[name] = 0.0; self.order.append(name)
            self.sums[name] += now - self.t
        self.t = now
    def report(self, logger):
        k = max(1, self.n - self.warmup); total = sum(self.sums.values())
        logger.info(f"[profil] średnio z {k} kroków: {1000 * total / k:.1f} ms/krok ({k / max(total, 1e-9):.2f} kroku/s)")
        for name in self.order: logger.info(f"[profil]   {name:<38} {1000 * self.sums[name] / k:8.1f} ms  {100 * self.sums[name] / max(total, 1e-9):5.1f}%")


def random_crop(audio, y_hat, crop_len, limit):
    """The same random crop of both signals, within the first `limit` samples."""
    if limit <= crop_len: return audio, y_hat
    start = torch.randint(0, limit - crop_len + 1, (1,)).item()
    return audio[..., start : start + crop_len], y_hat[..., start : start + crop_len]

def recon_l1(real, fake, mel_transforms):
    # squeeze per transform: one shared view changes the order in which gradients accumulate
    loss = sum(F.l1_loss(tf(real.squeeze(1)), tf(fake.squeeze(1))) for tf in mel_transforms)
    return loss / len(mel_transforms) if mel_transforms else loss

def train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
               update_discriminator=True, lambda_recon=45.0, recon_full_segment=False, fm_composite=False, encoder_only=False, recon_only=False,
               terms=None, clip_generator=True, prof=None, slm=None, opt_slm=None):
    # recon_only: no adversarial or feature-matching terms for the generator.
    # encoder_only: the frozen decoder stays in the graph, so gradients reach the encoder through it.
    lambda_adv, lambda_fm = 1.0, 0.1
    batch, lengths = batch if isinstance(batch, (tuple, list)) else (batch, None)   # lengths: edge-aware batches
    audio = batch.to(device)
    if prof: prof.mark("czekanie na dane + kopia na GPU")
    if audio.dim() == 2: audio = audio.unsqueeze(1) 

    with torch.no_grad():
        mel = mel_transform_input(audio.squeeze(1))
    
    z = encoder(mel)
    mask = None
    if lengths is not None:
        # Keep the array's own frames (the centred STFT adds one) and supervise each clip up to ceil(L / 3072) * 3072.
        lengths = lengths.to(device)
        supervised = chunk_ceil(lengths)
        z = z[..., : audio.shape[-1] // HOP]
        mask = (torch.arange(audio.shape[-1], device=device)[None] < supervised[:, None]).float().unsqueeze(1)
    y_hat = decoder(z)
    if y_hat.dim() == 2: y_hat = y_hat.unsqueeze(1)
    
    if y_hat.shape[-1] != audio.shape[-1]:
        min_len = min(y_hat.shape[-1], audio.shape[-1])
        y_hat, audio = y_hat[..., :min_len], audio[..., :min_len]
    crop_limit = audio.shape[-1]
    if mask is not None:
        y_hat, audio = y_hat * mask, audio * mask
        if int(supervised.min()) > crop_len: crop_limit = int(supervised.min())   # D never sees the masked zeros
    audio_crop, y_hat_crop = random_crop(audio, y_hat, crop_len, crop_limit)
    if prof: prof.mark("widmo + enkoder + dekoder (w przód)")

    loss_d_total = 0.0
    if update_discriminator:
        y_hat_detached = y_hat_crop.detach()
        y_df_hat_r, y_df_hat_g, _, _ = mpd(audio_crop, y_hat_detached)
        y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audio_crop, y_hat_detached)
        loss_d_total = discriminator_loss(y_df_hat_r, y_df_hat_g) + discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        if check_for_nan_inf(loss_d_total, "Discriminator Loss", logger): return None
        opt_d.zero_grad()
        loss_d_total.backward()
        torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1.0)
        opt_d.step()
        loss_d_total = loss_d_total.item()
    slm_real = slm_fake = None
    if slm is not None:
        with torch.no_grad():
            slm_real = slm.cechy(audio)                 # cel: bez gradientu
        slm_fake = slm.cechy(y_hat)                     # raz, z gradientem przez zamrożony WavLM (krok generatora);
        if update_discriminator and slm.w_adv > 0:      # dyskryminator dostaje odłączoną kopię
            loss_slm_d = discriminator_loss([slm.head(slm_real)], [slm.head([f.detach() for f in slm_fake])])
            if check_for_nan_inf(loss_slm_d, "SLM Discriminator Loss", logger): return None
            opt_slm.zero_grad()
            loss_slm_d.backward()
            torch.nn.utils.clip_grad_norm_(slm.head.parameters(), 1.0)
            opt_slm.step()
            loss_d_total = loss_d_total + loss_slm_d.item()
    if prof: prof.mark("dyskryminatory: w przód, wstecz, krok")

    # Reconstruction on the adversarial crop, or on the whole segment with recon_full_segment.
    rec_real, rec_fake = (audio, y_hat) if recon_full_segment else (audio_crop, y_hat_crop)
    L_recon = recon_l1(rec_real, rec_fake, mel_transforms_loss)
    if prof: prof.mark("strata rekonstrukcji (mel)")

    # W kroku generatora wagi dyskryminatorów są tylko używane, nie uczone: bez tego przejście wstecz liczyłoby
    # też ich gradienty, które i tak są zerowane przed następnym krokiem dyskryminatora. Gradient generatora
    # jest identyczny. Pod DDP zostaje po staremu (reduktor DDP oczekuje gradientów wszystkich wag).
    freeze_d = not recon_only and not isinstance(mpd, DDP) and not isinstance(mrd, DDP)
    if freeze_d:
        d_params = [p for m in (mpd, mrd) + ((slm.head,) if slm is not None else ()) for p in m.parameters() if p.requires_grad]
        for p in d_params: p.requires_grad_(False)
    if not recon_only:
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio_crop, y_hat_crop)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(audio_crop, y_hat_crop)
        L_adv = generator_loss(y_df_hat_g) + generator_loss(y_ds_hat_g)
        if fm_composite: L_fm = feature_loss_composite((fmap_f_r, fmap_f_g), (fmap_s_r, fmap_s_g))
        else: L_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
        loss_g_total = (lambda_recon * L_recon) + (lambda_adv * L_adv) + (lambda_fm * L_fm)
    else:
        loss_g_total = lambda_recon * L_recon
    extras = extra_losses(terms or ExtraTerms(), encoder, mel_transform_input, audio, y_hat, z, lengths, mask)
    if slm is not None:
        if slm.w_fm > 0:
            extras["slm_fm"] = (slm.w_fm, sum(F.l1_loss(g.float(), r.float()) for g, r in zip(slm_fake, slm_real)) / len(slm_fake))
        if slm.w_adv > 0 and not recon_only:
            extras["slm_adv"] = (slm.w_adv, generator_loss([slm.head(slm_fake)]))
    for w, loss in extras.values(): loss_g_total = loss_g_total + w * loss
    if prof: prof.mark("generator: dyskryminatory w przód + straty")
    
    if check_for_nan_inf(loss_g_total, "Generator Loss", logger):
        if freeze_d:
            for p in d_params: p.requires_grad_(True)
        return None
    opt_g.zero_grad()
    loss_g_total.backward()
    if freeze_d:
        for p in d_params: p.requires_grad_(True)
    if prof: prof.mark("generator: wstecz")
    if clip_generator:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        if not encoder_only: torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
    opt_g.step()
    out = loss_g_total.item(), loss_d_total, L_recon.item(), {k: loss.item() for k, (_, loss) in extras.items()}
    if prof: prof.mark("generator: obcinanie gradientu + krok")
    return out

def freeze_batchnorm(model):
    """BatchNorm layers in eval mode (running statistics used and not updated)."""
    for m in _mod(model).modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def make_scheduler(opt, t_max, warmup=0):
    """Cosine decay to 1e-6 over `t_max` steps, after an optional linear warm-up from 1% of the lr."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)
    if not warmup: return cosine
    linear = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(opt, [linear, cosine], milestones=[warmup])

def cosine_schedulers(opt_g, opt_d, t_max, warmup=0):
    schedulers = {'scheduler_g': make_scheduler(opt_g, t_max, warmup), 'scheduler_d': make_scheduler(opt_d, t_max, warmup)}
    return schedulers, schedulers['scheduler_g']

def init_distributed(local_rank):
    """NCCL on the local GPU, or gloo on CPU when no GPU is visible. Returns (device, DDP device_ids)."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return torch.device(f'cuda:{local_rank}'), [local_rank]
    dist.init_process_group(backend='gloo', init_method='env://')
    return torch.device('cpu'), None

@torch.no_grad()
def sync_weights(modules, from_rank0=False):
    """Uśrednia wagi (i statystyki BatchNorm) podanych modułów między wszystkimi kartami jednym przesłaniem.

    from_rank0=True: zamiast średniej wszyscy dostają wagi karty 0 (wspólny start). Stany optymalizatorów
    zostają lokalne. To zwykłe okresowe uśrednianie modeli (local SGD), nie synchronizacja gradientów.
    """
    world = dist.get_world_size()
    if world < 2: return
    tensors = []
    for m in modules:
        tensors += [p.data for p in m.parameters()]
        tensors += [b for b in m.buffers() if b.is_floating_point()]
    flat = torch.cat([t.reshape(-1) for t in tensors])
    if from_rank0:
        if dist.get_rank() != 0: flat.zero_()
        dist.all_reduce(flat)
    else:
        dist.all_reduce(flat)
        flat /= world
    i = 0
    for t in tensors:
        n = t.numel()
        t.copy_(flat[i:i + n].view_as(t)); i += n
    if os.environ.get("AVG_DEBUG"):
        print(f"[avg] rank {dist.get_rank()}: {'start z karty 0' if from_rank0 else 'średnia'}, suma kontrolna {flat.double().sum().item():.6f}, {flat.numel() / 1e6:.1f}M wartości", flush=True)


def evaluate(encoder, decoder, mel_transform, input_wav_path, output_dir, step, device, target_sr, rank, keep_decoder_eval=False):
    if rank != 0: return
    encoder.eval(); decoder.eval()
    try:
        # soundfile zamiast torchaudio.load: to drugie wymaga torchcodec i ffmpeg tylko po to, żeby wczytać jeden plik
        _wav_np, sr = sf.read(input_wav_path, dtype="float32", always_2d=True)   # [T, C]
        wav = torch.from_numpy(_wav_np).T                                          # [C, T]
        if sr != target_sr: wav = ensure_sr(wav, sr, target_sr)
        wav = wav.mean(dim=0, keepdim=True).to(device)
        with torch.no_grad():
            enc_mod = encoder.module if isinstance(encoder, DDP) else encoder
            dec_mod = decoder.module if isinstance(decoder, DDP) else decoder
            y_hat = dec_mod(enc_mod(mel_transform(wav)))
            # Wyśrodkowane STFT dodaje jedną ramkę ponad długość nagrania. W treningu ten nadmiar jest obcinany
            # i nigdy nie trafia do straty, więc dekoder produkuje tam dowolnie duży skok; bez obcięcia skok
            # zaniżał głośność całej próbki przy normalizacji poniżej.
            y_hat = y_hat[..., : wav.shape[-1]]
            max_val = y_hat.abs().max().item()
            y_hat_final = y_hat.squeeze()
            if max_val > 1.0: y_hat_final /= (max_val + 1e-8)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"step_{step}_{os.path.basename(input_wav_path)}_recon.wav")
        sf.write(output_path, y_hat_final.cpu().numpy(), target_sr)
    except Exception as e: print(f"Evaluation failed: {e}")
    encoder.train()
    if not keep_decoder_eval: decoder.train()   # a frozen decoder stays in eval (BatchNorm running stats)

@torch.no_grad()
def heldout_recon(encoder, decoder, mel_transform, mel_transforms_loss, clips, device):
    encoder.eval(); decoder.eval()
    total = 0.0
    for wav in clips:
        wav = wav.to(device).unsqueeze(0)
        y_hat = decoder(encoder(mel_transform(wav))).reshape(1, -1)
        n = min(y_hat.shape[-1], wav.shape[-1])
        total += recon_l1(wav[..., :n], y_hat[..., :n], mel_transforms_loss).item()
    return total / len(clips)

@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)   # BatchNorm running stats follow the raw model

def param_groups(named, lr, no_decay_norms):
    # no_decay_norms: no weight decay on norms, biases, PReLU slopes and layer-scale gammas
    named = list(named)
    if not no_decay_norms: return [{'params': [p for _, p in named], 'lr': lr}]
    nd = lambda n, p: p.ndim <= 1 or n.endswith('gamma')
    return [{'params': [p for n, p in named if not nd(n, p)], 'lr': lr},
            {'params': [p for n, p in named if nd(n, p)], 'lr': lr, 'weight_decay': 0.0}]

class PlateauLR:
    """Warm-up, then a flat lr halved (G and D together) when the held-out recon stops improving, never below min_lr.
    Groups with g['ramp_from'] ramp from 0 over `ramp` steps: the stage-2 decoder starts with fresh Adam moments."""
    def __init__(self, opts, lr, warmup, min_lr, patience, threshold, ramp=0):
        self.opts, self.lr, self.warmup, self.min_lr = opts, lr, warmup, min_lr
        self.patience, self.threshold, self.ramp = patience, threshold, ramp
        self.scale, self.best, self.bad = 1.0, math.inf, 0

    def step(self, step):
        warm = min(1.0, (step + 1) / self.warmup) if self.warmup else 1.0
        for o in self.opts:
            for g in o.param_groups:
                f = self.scale * warm
                if g.get('ramp_from') is not None and self.ramp:
                    f *= min(1.0, max(0, step - g['ramp_from'] + 1) / self.ramp)
                g['lr'] = g['base_lr'] * f

    def report(self, val):
        if val < self.best * (1.0 - self.threshold): self.best, self.bad = val, 0
        else: self.bad += 1
        if self.bad >= self.patience: self.bad, self.scale = 0, max(self.scale * 0.5, self.min_lr / self.lr)

    def get_last_lr(self): return [g['lr'] for g in self.opts[0].param_groups]
    def state_dict(self): return {'scale': self.scale, 'best': self.best, 'bad': self.bad}
    def load_state_dict(self, sd): self.scale, self.best, self.bad = sd['scale'], sd['best'], sd['bad']

class AverageMeter:
    """avg: średnia od startu przebiegu; recent: zwykła średnia z `window` ostatnich kroków (do paska postępu)."""
    def __init__(self, window=200): self.window = window; self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0; self.recent = None; self._last = collections.deque(); self._sum = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
        self._last.append(val); self._sum += val
        if len(self._last) > self.window: self._sum -= self._last.popleft()
        self.recent = self._sum / len(self._last)

def load_checkpoint(args, ckpt, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, dec_groups, logger):
    # dec_groups (stage 2) are appended to opt_g here: a stage-2 checkpoint saved them, a stage-1 one did not
    def load_sd(model, sd):
        if any(k.startswith('module.') for k in sd.keys()): sd = {k.replace('module.', ''): v for k, v in sd.items()}
        md = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        if hasattr(model, 'module'): model.module.load_state_dict(filtered, strict=False)
        else: model.load_state_dict(filtered, strict=False)

    def load_state():
        opt_g.load_state_dict(ckpt['opt_g']); opt_d.load_state_dict(ckpt['opt_d'])
        for k, v in schedulers.items():
            if k in ckpt: v.load_state_dict(ckpt[k])
        return ckpt['step'] + 1, ckpt.get('epoch', 0)

    def opt_match(opt, sd):
        return sd and len(opt.param_groups) == len(sd['param_groups']) and \
            all(len(g['params']) == len(sg['params']) for g, sg in zip(opt.param_groups, sd['param_groups']))

    resumed = None
    if ckpt is not None:
        if args.local_rank == 0: logger.info(f"Resuming from {args.resume}...")
        load_sd(encoder, ckpt['encoder'])
        if not args.encoder_only and 'decoder' in ckpt: load_sd(decoder, ckpt['decoder'])   # encoder_only: decoder comes from --decoder
        load_sd(mpd, ckpt['mpd']); load_sd(mrd, ckpt['mrd'])
        # stage 1 -> 2: the encoder keeps its Adam moments; the decoder groups are appended after with fresh ones
        if args.stage2 and ckpt.get('stage') != 2: resumed = load_state()
    for g in dec_groups: opt_g.add_param_group(g)
    if ckpt is None or resumed: return resumed or (0, 0)

    if not args.finetune:
        if opt_match(opt_g, ckpt.get('opt_g')) and opt_match(opt_d, ckpt.get('opt_d')): return load_state()
        if args.local_rank == 0: logger.warning("Optimizer state does not match the model: fresh optimizer, step 0")
    return 0, 0

def save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, logger, ckpt_dir="checkpoints/ae",
                    decoder_source=None, extra=None):
    state = {
        "step": step, "epoch": epoch,
        "encoder": _mod(encoder).state_dict(),
        "mpd": _mod(mpd).state_dict(), "mrd": _mod(mrd).state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        **{k: v.state_dict() for k, v in schedulers.items()},
        **(extra or {}),
    }
    # A frozen decoder is not written into the checkpoint; it is recorded by its source path.
    if decoder_source is None: state["decoder"] = _mod(decoder).state_dict()
    else: state["decoder_source"] = decoder_source
    torch.save(state, os.path.join(ckpt_dir, f"ae_{step}.pt"))
    torch.save(state, os.path.join(ckpt_dir, "ae_latest.pt"))
    cleanup_checkpoints(logger, ckpt_dir)

def load_init_encoder(path, encoder, device):
    # .pt checkpoint ({'encoder': ...}) or a BlueCodec .safetensors ('encoder.*' keys, e.g. HF model.safetensors)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = {k[len("encoder."):]: v for k, v in load_file(path, device=str(device)).items() if k.startswith("encoder.")}
    else:
        sd = torch.load(path, map_location=device)["encoder"]
    _mod(encoder).load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)

def cleanup_checkpoints(logger, ckpt_dir="checkpoints/ae"):
    try:
        ckpts = []
        for f in os.listdir(ckpt_dir):
            if f.startswith("ae_") and f.endswith(".pt") and f != "ae_latest.pt":
                try: ckpts.append((int(f.replace("ae_", "").replace(".pt", "")), f))
                except ValueError: pass
        ckpts.sort(key=lambda x: x[0], reverse=True)
        for _, old in ckpts[1000:]:
            path = os.path.join(ckpt_dir, old)
            if os.path.exists(path): os.remove(path); logger.info(f"Deleted: {old}")
    except Exception as e: logger.warning(f"Cleanup error: {e}")

def start_tensorboard(logdir="checkpoints/ae/logs"):
    try:
        tb_path = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        if not os.path.exists(tb_path): tb_path = "tensorboard"
        proc = subprocess.Popen([tb_path, "--logdir", logdir, "--port", "8000", "--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(proc.kill)
        print(f"TensorBoard at http://localhost:8000")
    except Exception as e: print(f"TB error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Training checkpoint (.pt) to resume from')
    parser.add_argument('--eval_input', type=str, default=None, help='Audio file reconstructed into <checkpoint_dir>/eval at every save')
    parser.add_argument('--profile', type=int, default=0, help='Time the parts of a training step over the first N steps and log the averages')
    parser.add_argument('--constant_lr', action='store_true', help='Constant learning rate --lr from the current step (no decay)')
    parser.add_argument('--save_every', type=int, default=None, help='Checkpoint every N steps (overrides ae.train.save_interval)')
    parser.add_argument('--heldout_every', type=int, default=0,
                        help='Every N steps score a fixed set of held-out clips (TensorBoard Loss/Heldout_*); with --avg_every use a multiple of it')
    parser.add_argument('--heldout_clips', type=int, default=64, help='Number of held-out clips for --heldout_every')
    parser.add_argument('--heldout_utmos', action='store_true',
                        help='W ocenie --heldout_every także UTMOS rekonstrukcji (Loss/Heldout_UTMOS; oryginały raz: Heldout_UTMOS_oryginal)')
    parser.add_argument('--heldout_max_sec', type=float, default=15.0, help='Held-out clips are cut to this many seconds')
    parser.add_argument('--num_workers', type=int, default=None, help='Data-loading processes per GPU (overrides ae.train.num_workers)')
    parser.add_argument('--avg_every', type=int, default=0,
                        help='Multi-GPU without per-step gradient sync: each GPU trains its own copy and the copies are averaged every N steps (0 = DDP)')
    parser.add_argument('--eval_every', type=int, default=None, help='Reconstruct --eval_input every N steps (default: ae.train.save_interval)')
    parser.add_argument('--arch_config', type=str, default='config/tts.json', help='Model and data config')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--finetune', action='store_true', help='With --resume: load the weights only (fresh optimizer and schedule, step 0)')
    parser.add_argument('--lr', type=float, default=None, help='Peak learning rate (overrides ae.train.lr)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ae', help='Checkpoints, logs and eval audio')
    parser.add_argument('--total_steps', type=int, default=1500000, help='Loop length and cosine T_max')
    parser.add_argument('--batch_size', type=int, default=None, help='Per-process batch size (overrides ae.train.batch_size)')
    parser.add_argument('--segment_sec', type=float, default=None,
                        help='Długość segmentu treningowego w s (nadpisuje ae.data.segment_size, oryginał 1,4 s); '
                             'koszt kroku rośnie liniowo, więc przy dłuższym zwykle mniejsze --batch_size')
    # --- Encoder-only training against a frozen decoder ---
    parser.add_argument('--encoder_only', action='store_true', help='Freeze the decoder (from --decoder) and train the encoder + discriminators')
    parser.add_argument('--decoder', type=str, default=None, help='Frozen decoder for --encoder_only: an AE .pt checkpoint, or "official"')
    parser.add_argument('--official_onnx', type=str, default=None, help='Local official vocoder.onnx (default: downloaded)')
    parser.add_argument('--init_encoder', type=str, default=None, help='Encoder init: AE .pt checkpoint or BlueCodec .safetensors (fresh optimizer, step 0)')
    # --- Dotrenowanie samego dekodera (enkoder zamrożony: latenty, statystyki i model tekst->latenty bez zmian) ---
    parser.add_argument('--decoder_only', action='store_true',
                        help='Zamrożony enkoder, uczą się dekoder i dyskryminatory; start z --init_ae (albo --resume)')
    parser.add_argument('--init_ae', type=str, default=None, help='Kodek startowy (.pt z "encoder" i "decoder", np. ae_wsm_210_220.pt)')
    parser.add_argument('--init_disc', type=str, default=None,
                        help='Dyskryminatory startowe z punktu kontrolnego treningu (.pt z "mpd" i "mrd", np. ae_220000.pt); bez tego od zera')
    parser.add_argument('--decoder_head', choices=['patch', 'istft'], default=None,
                        help='Głowica dekodera: patch (512 próbek na klatkę, sklejane) albo istft (widmo + overlap-add, jak Vocos). '
                             'Z --init_ae starego dekodera nowa głowica startuje od zera, reszta dekodera z wag')
    parser.add_argument('--istft_n_fft', type=int, default=2048, help='Długość ramki głowicy istft (wielokrotność 512)')
    parser.add_argument('--decoder_postnet', action='store_true',
                        help='Moduł poprawki na wyjściu dekodera (STFTPostnet: maska widma, start od tożsamości); '
                             'z --head_warmup i --head_lr_scale „nową częścią” jest ten moduł, a nie głowica')
    parser.add_argument('--postnet_dim', type=int, default=384)
    parser.add_argument('--postnet_layers', type=int, default=4)
    parser.add_argument('--head_warmup', type=int, default=0,
                        help='Przez pierwsze N kroków uczy się tylko głowica dekodera (reszta dekodera stoi), potem cały dekoder')
    parser.add_argument('--body_ramp', type=int, default=1000,
                        help='Po --head_warmup tempo ciała dekodera rośnie liniowo od zera przez N kroków (świeże momenty Adama)')
    parser.add_argument('--head_lr_scale', type=float, default=1.0,
                        help='Z --decoder_only i --constant_lr: tempo głowicy dekodera = --lr × ta wartość (nowa głowica od zera)')
    parser.add_argument('--d_warmup', type=int, default=0, help='Discriminators frozen, reconstruction-only loss for the first N steps')
    parser.add_argument('--stage2', action='store_true', help='Resume an --encoder_only checkpoint (--resume) with the decoder trainable')
    # --- Paper losses (defaults reproduce the 1.5M-step recipe) ---
    parser.add_argument('--recon_logmel_fullband', action='store_true', help='Reconstruction on LOG mel up to sr/2 over the whole segment')
    parser.add_argument('--fm_composite', action='store_true', help='Feature matching averaged over MPD+MRD layers together (paper Eq. 6)')
    # --- E12b additions (all off by default) ---
    parser.add_argument('--edge_aware', action='store_true',
                        help='Half of the crops end at the clip end; batches padded to a multiple of 3072 samples and cut so clips end '
                             'in the last chunk; loss masked per clip beyond ceil(L/3072)*3072')
    parser.add_argument('--tail_consistency', type=float, default=0.0,
                        help='Weight of the last-latent-frame consistency term (needs --edge_aware and --tail_stats; freezes the encoder BatchNorm)')
    parser.add_argument('--tail_stats', type=str, default=None, help='Latent stats file (mean/std over 144 compressed channels) normalising the tail term')
    parser.add_argument('--no_bn_freeze', action='store_true', help='Keep the encoder BatchNorm in train mode with --tail_consistency')
    parser.add_argument('--hiband_w', type=float, default=0.0, help='Weight of the 12-22 kHz log-magnitude STFT L1')
    parser.add_argument('--hiband_hz', type=float, default=None,
                        help='Dolna granica pasma straty --hiband_w w Hz (domyślnie prążek 558 z 2048, jak w oryginale)')
    parser.add_argument('--d_crop', type=float, default=0.19,
                        help='Długość wycinka dla MPD/MRD w sekundach (oryginał 0,19; koszt dyskryminatorów rośnie liniowo); '
                             'nie dłużej niż segment (domyślnie 1,4 s)')
    parser.add_argument('--slm_w', type=float, default=0.0,
                        help='Waga straty przeciwstawnej dyskryminatora SLM (zamrożony WavLM-Base+, jak w StyleTTS2); 0 = wyłączony')
    parser.add_argument('--slm_fm_w', type=float, default=0.0,
                        help='Waga dopasowania cech WavLM (średnie L1 po warstwach, wyjście vs cel); 0 = wyłączone')
    parser.add_argument('--slm_partia', type=int, default=16, help='Ile klipów porcji idzie przez WavLM (pamięć)')
    parser.add_argument('--linie_w', type=float, default=0.0,
                        help='Waga straty prążków co 93,75 Hz (wysokość ponad tło względem prawdziwego nagrania)')
    parser.add_argument('--hiband_floor', type=float, default=1e-5, help='Podłoga amplitudy w stracie --hiband_w (np. 1e-7)')
    parser.add_argument('--floor_w', type=float, default=0.0, help='Weight of the silence-floor term (source frames below --floor_db must decode below it)')
    parser.add_argument('--floor_db', type=float, default=-60.0, help='Silence floor in dBFS for --floor_w')
    parser.add_argument('--drop_list', type=str, default=None, help='Text file of audio paths (one per line) to leave out')
    parser.add_argument('--decoder_pad_mode', choices=['zeros', 'replicate'], default=None,
                        help='Causal padding of the decoder (default: the config, zeros; E12b uses replicate)')
    parser.add_argument('--continue_cosine', type=int, default=None,
                        help='With --resume: keep the step counter and optimizer state, restart the cosine from --lr over N steps')
    parser.add_argument('--lr_warmup', type=int, default=0,
                        help='Z --continue_cosine albo --constant_lr: liniowa rozgrzewka tempa od 1%% --lr przez N kroków')
    args = parser.parse_args()
    if args.stage2 and args.encoder_only: raise SystemExit("--stage2 trains the decoder: drop --encoder_only")
    if args.decoder_only and (args.encoder_only or args.stage2 or args.init_encoder):
        raise SystemExit("--decoder_only wyklucza --encoder_only, --stage2 i --init_encoder")
    if args.decoder_only and not (args.init_ae or args.resume):
        raise SystemExit("--decoder_only potrzebuje --init_ae (kodek startowy) albo --resume")
    if (args.init_ae or args.init_disc) and args.resume:
        raise SystemExit("--init_ae/--init_disc tylko przy starcie; przy --resume wszystko jest w punkcie kontrolnym")
    if args.tail_consistency > 0 and args.decoder_only: raise SystemExit("--tail_consistency uczy enkoder: nie z --decoder_only")
    if args.stage2 and not args.resume: raise SystemExit("--stage2 needs --resume (a stage-1 checkpoint)")
    if args.encoder_only and not args.decoder:
        raise SystemExit("--encoder_only needs --decoder (an AE .pt checkpoint or 'official')")
    if args.tail_consistency > 0 and not (args.edge_aware and args.tail_stats):
        raise SystemExit("--tail_consistency needs --edge_aware and --tail_stats")
    if args.constant_lr and args.continue_cosine: raise SystemExit("--constant_lr i --continue_cosine wykluczają się")
    if args.continue_cosine and not args.resume:
        raise SystemExit("--continue_cosine needs --resume")
    if args.lr_warmup and not (args.continue_cosine or args.constant_lr):
        raise SystemExit("--lr_warmup needs --continue_cosine or --constant_lr")
    freeze_bn = args.tail_consistency > 0 and not args.no_bn_freeze
    ckpt_dir = args.checkpoint_dir

    if 'WORLD_SIZE' in os.environ: args.local_rank = int(os.environ['LOCAL_RANK'])
    device, ddp_ids = init_distributed(args.local_rank)
    set_seed(args.seed + args.local_rank)
    logger = setup_logger(ckpt_dir, args.local_rank)
    
    writer = None
    if args.local_rank == 0:
        logger.info(f"Training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
        start_tensorboard(os.path.join(ckpt_dir, "logs"))
        writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))

    with open(args.arch_config, "r") as f: arch_cfg = json.load(f)
    ae_cfg = arch_cfg['ae']
    data_cfg, train_cfg = ae_cfg['data'], ae_cfg['train']
    if args.save_every: train_cfg['save_interval'] = args.save_every
    # two-stage recipe switches (config/tts_st3.json); a missing key keeps the default recipe
    recon = "logmel_fullband" if args.recon_logmel_fullband else train_cfg.get('recon', 'linmel_12k')
    fm_composite = args.fm_composite or train_cfg.get('fm_composite', False)
    ema_decay = float(train_cfg.get('ema', 0.0))
    no_decay_norms = train_cfg.get('no_decay_norms', False)
    clip_generator = train_cfg.get('clip_generator', True)
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    dec_lr_scale = float(train_cfg.get('dec_lr_scale', 1.0))
    if args.stage2 and lr_schedule != 'plateau': raise SystemExit("--stage2 needs ae.train.lr_schedule 'plateau'")
    if args.continue_cosine and lr_schedule != 'cosine': raise SystemExit("--continue_cosine needs ae.train.lr_schedule 'cosine'")
    if not train_cfg.get('tf32', True):   # strict fp32 (cuDNN defaults to TF32 on Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    if args.segment_sec:
        data_cfg['segment_size'] = int(round(args.segment_sec * data_cfg['sample_rate'] / CHUNK)) * CHUNK   # pełne porcje 3072
    dataset = TTSDataset(data_cfg['train_metadata'], sample_rate=data_cfg['sample_rate'], segment_size=data_cfg.get('segment_size'),
                         end_crop_prob=0.5 if args.edge_aware else 0.0, drop_list=args.drop_list)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size or train_cfg['batch_size'], sampler=sampler, num_workers=args.num_workers if args.num_workers is not None else train_cfg['num_workers'],
                            collate_fn=collate_fn_edge if args.edge_aware else collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    decoder_cfg = dict(ae_cfg['decoder'])
    if args.decoder_pad_mode: decoder_cfg['pad_mode'] = args.decoder_pad_mode
    if args.decoder_head == 'istft':
        decoder_cfg['head'] = {**decoder_cfg['head'], 'type': 'istft', 'n_fft': args.istft_n_fft, 'hop': HOP, 'wyrownanie': 'srodek'}
    elif args.decoder_head == 'patch':
        decoder_cfg['head'] = {k: v for k, v in decoder_cfg['head'].items() if k not in ('type', 'n_fft', 'hop')}
    if args.decoder_postnet:
        decoder_cfg['postnet'] = {'dim': args.postnet_dim, 'intermediate_dim': 3 * args.postnet_dim,
                                  'num_layers': args.postnet_layers, 'ksz': 7, 'n_fft': 2048}
    # część dekodera uczona od nowa (--head_warmup zamraża resztę, --head_lr_scale mnoży jej tempo)
    nowa_czesc = 'postnet.' if decoder_cfg.get('postnet') else 'head.'
    ae_cfg = {**ae_cfg, 'decoder': decoder_cfg}   # zapisywane w punktach kontrolnych: dalsze skrypty budują ten sam dekoder
    if args.head_warmup and not args.decoder_only: raise SystemExit("--head_warmup tylko z --decoder_only")
    ckpt = torch.load(args.resume, map_location='cpu') if args.resume else None

    encoder = LatentEncoder(cfg=ae_cfg['encoder']).to(device)
    # the frozen decoder (stage 1), or the one stage 1 used (stage-1 checkpoints record only its source)
    dec_src = args.decoder if args.encoder_only else None
    if args.stage2 and 'decoder' not in ckpt: dec_src = ckpt.get('decoder_source')
    enc_ln_init = train_cfg.get('enc_ln_init') == 'official' and not args.resume and not args.init_encoder
    if dec_src == "official" or enc_ln_init: official_sd, latent_mean, latent_std = supertonic3_decoder_state(args.official_onnx)
    decoder = LatentDecoder1D(cfg=decoder_cfg).to(device)
    if dec_src == "official" and decoder_cfg.get('pad_mode') != 'replicate':
        raise SystemExit("the official decoder needs ae.decoder.pad_mode 'replicate'")
    if dec_src == "official": decoder.load_state_dict(official_sd, strict=True)
    elif dec_src: decoder.load_state_dict(torch.load(dec_src, map_location=device)['decoder'])
    decoder_source = None
    if args.encoder_only:
        # frozen, so not DDP-wrapped: it has nothing to synchronise
        decoder_source = args.decoder
        decoder.requires_grad_(False); decoder.eval()
    # stage 2 keeps the decoder's BatchNorm statistics: re-estimated on new latents they break reconstruction
    dec_bns = [m for m in decoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if args.stage2 else []

    if enc_ln_init:   # a fresh encoder starts in the official latent range (LayerNorm affine = latent std / mean)
        with torch.no_grad(): encoder.norm.norm.weight.copy_(latent_std.reshape(-1)); encoder.norm.norm.bias.copy_(latent_mean.reshape(-1))
    if args.decoder_only:
        # zamrożony, więc bez DDP; tryb eval: statystyki BatchNorm też się nie ruszają
        encoder.requires_grad_(False); encoder.eval()
    if train_cfg.get('sync_bn', False) and dist.get_world_size() > 1 and not args.decoder_only:
        # per-GPU statistics over a small batch can collapse the encoder stem
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    if args.head_warmup and dist.get_world_size() > 1 and args.avg_every == 0:
        raise SystemExit("--head_warmup na kilku kartach wymaga --avg_every (zamrażanie wag nie działa pod DDP)")
    if args.avg_every > 0 or (args.decoder_only and dist.get_world_size() == 1):
        # Bez DDP: każda karta trenuje własną kopię i nie przesyła gradientów co krok; kopie są uśredniane co N kroków.
        # Na jednej karcie DDP nic nie daje, a bez niego krok generatora nie liczy gradientów dyskryminatorów.
        mpd = MultiPeriodDiscriminator().to(device)
        mrd = MultiResolutionDiscriminator().to(device)
    else:
        if not args.decoder_only: encoder = DDP(encoder, device_ids=ddp_ids)
        if not args.encoder_only: decoder = DDP(decoder, device_ids=ddp_ids)
        mpd = DDP(MultiPeriodDiscriminator().to(device), device_ids=ddp_ids)
        mrd = DDP(MultiResolutionDiscriminator().to(device), device_ids=ddp_ids)

    spec_cfg = ae_cfg['encoder'].get('spec_processor', {})
    mel_transform_input = LinearMelSpectrogram(sample_rate=spec_cfg.get('sample_rate', data_cfg['sample_rate']), n_fft=spec_cfg.get('n_fft', 2048), hop_length=spec_cfg.get('hop_length', 512), win_length=spec_cfg.get('win_length', 2048), n_mels=spec_cfg.get('n_mels', 1253)).to(device)
    mel_transforms_loss = get_mel_transforms(data_cfg, device, recon)
    
    lr = args.lr if args.lr else float(train_cfg['lr'])
    betas = tuple(train_cfg.get('betas', (0.8, 0.99)))
    wd = float(train_cfg.get('weight_decay', 0.01))
    enc_named = list(_mod(encoder).named_parameters())
    if args.decoder_only:
        _dec = list(_mod(decoder).named_parameters())
        _cialo = param_groups([(n, p) for n, p in _dec if not n.startswith(nowa_czesc)], lr, no_decay_norms)
        for g in _cialo: g['cialo_dekodera'] = True   # --head_warmup: tempo 0 przez pierwsze kroki
        _glowica = param_groups([(n, p) for n, p in _dec if n.startswith(nowa_czesc)], lr, no_decay_norms)
        for g in _glowica: g['glowica_dekodera'] = True     # --head_lr_scale
        g_groups = _cialo + _glowica
    elif args.encoder_only or args.stage2: g_groups = param_groups(enc_named, lr, no_decay_norms)
    else: g_groups = param_groups(enc_named + list(_mod(decoder).named_parameters()), lr, no_decay_norms)
    # stage 2: decoder groups after the encoder's, so a stage-1 optimizer state still loads
    dec_groups = param_groups(_mod(decoder).named_parameters(), lr * dec_lr_scale, no_decay_norms) if args.stage2 else []
    n_enc_groups = len(g_groups)
    opt_g = torch.optim.AdamW(g_groups, lr=lr, betas=betas, weight_decay=wd)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=lr, betas=betas, weight_decay=wd)
    if lr_schedule == 'plateau':
        lr_sched = PlateauLR([opt_g, opt_d], lr, train_cfg.get('warmup_steps', 0), float(train_cfg.get('min_lr', 1e-5)),
                             train_cfg.get('plateau_patience', 3), train_cfg.get('plateau_threshold', 0.01),
                             train_cfg.get('dec_warmup_steps', 0))
        schedulers = {'lr_plateau': lr_sched}
    else:
        schedulers, lr_sched = cosine_schedulers(opt_g, opt_d, args.total_steps)
    
    step, epoch = load_checkpoint(args, ckpt, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, dec_groups, logger)
    # dyskryminator SLM (--slm_w / --slm_fm_w): własny optymalizator, stałe tempo; stan w punkcie kontrolnym pod "slm"
    slm, opt_slm = None, None
    if args.slm_w > 0 or args.slm_fm_w > 0:
        if dist.get_world_size() > 1 and args.avg_every == 0:
            raise SystemExit("--slm_w/--slm_fm_w na kilku kartach wymaga --avg_every (głowica SLM nie jest pod DDP)")
        slm = SLM(device, data_cfg['sample_rate'], args.slm_w, args.slm_fm_w, args.slm_partia)
        opt_slm = torch.optim.AdamW(slm.head.parameters(), lr=lr, betas=betas, weight_decay=wd)
        _zrodlo = ckpt if ckpt is not None and 'slm' in ckpt else None
        if _zrodlo is None and args.init_disc:
            _d = torch.load(args.init_disc, map_location='cpu')
            _zrodlo = _d if 'slm' in _d else None
        if _zrodlo is not None:
            slm.head.load_state_dict(_zrodlo['slm'])
            if 'opt_slm' in _zrodlo: opt_slm.load_state_dict(_zrodlo['opt_slm'])
        if args.local_rank == 0:
            logger.info(f"SLM: WavLM-Base+ zamrożony ({sum(p.numel() for p in slm.wavlm.parameters()) / 1e6:.1f}M), głowica "
                        f"{sum(p.numel() for p in slm.head.parameters()) / 1e6:.2f}M "
                        f"({'z punktu kontrolnego' if _zrodlo is not None else 'od zera'}); wagi: przeciwstawna {args.slm_w:g}, "
                        f"cechy {args.slm_fm_w:g}; {args.slm_partia} klipów porcji")
    if args.init_ae:
        _ae = torch.load(args.init_ae, map_location=device)
        _mod(encoder).load_state_dict({k.replace('module.', ''): v for k, v in _ae['encoder'].items()}, strict=True)
        _sd = {k.replace('module.', ''): v for k, v in _ae['decoder'].items()}
        _dek_stary = _ae.get('ae_config', {}).get('decoder', {})
        _glowica_stara = _dek_stary.get('head', {}).get('type', 'patch') + (' + postnet' if _dek_stary.get('postnet') else '')
        _glowica_nowa = decoder_cfg['head'].get('type', 'patch') + (' + postnet' if decoder_cfg.get('postnet') else '')
        if _glowica_stara == _glowica_nowa:
            _mod(decoder).load_state_dict(_sd, strict=True)
        else:
            # inna głowica: przenosimy wszystko o zgodnych kształtach (ciało dekodera, pierwsza warstwa głowicy)
            _cel = _mod(decoder).state_dict()
            _zgodne = {k: v for k, v in _sd.items() if k in _cel and _cel[k].shape == v.shape}
            _mod(decoder).load_state_dict(_zgodne, strict=False)
            if args.local_rank == 0:
                logger.info(f"Głowica {_glowica_stara} -> {_glowica_nowa}: z {args.init_ae} przeniesiono {len(_zgodne)} tensorów, "
                            f"od zera: {sorted(k for k in _cel if k not in _zgodne)}")
        if args.local_rank == 0:
            from training.utils import encoder_fingerprint
            logger.info(f"Kodek startowy {args.init_ae}: enkoder {encoder_fingerprint(_mod(encoder).state_dict())[:12]}"
                        + (" (zamrożony)" if args.decoder_only else ""))
        del _ae
    if args.init_disc:
        _d = torch.load(args.init_disc, map_location=device)
        for _m, _k in ((mpd, 'mpd'), (mrd, 'mrd')):
            _mod(_m).load_state_dict({k.replace('module.', ''): v for k, v in _d[_k].items()}, strict=True)
        if args.local_rank == 0: logger.info(f"Dyskryminatory z {args.init_disc} (krok {_d.get('step')})")
        del _d
    if args.init_encoder and not args.resume:
        load_init_encoder(args.init_encoder, encoder, device)
        if args.local_rank == 0: logger.info(f"Encoder initialised from {args.init_encoder}")
    if args.continue_cosine:
        for opt in (opt_g, opt_d):
            for g in opt.param_groups: g['lr'] = g['initial_lr'] = lr
        schedulers, lr_sched = cosine_schedulers(opt_g, opt_d, args.continue_cosine, args.lr_warmup)
        if args.local_rank == 0: logger.info(f"New schedule from step {step}: {args.lr_warmup} warm-up steps, cosine over {args.continue_cosine} steps from lr {lr}")
    if args.constant_lr:
        # Stałe tempo uczenia --lr od bieżącego kroku (pod WSM: model końcowy z uśrednienia punktów kontrolnych, bez wygaszania).
        for opt in (opt_g, opt_d):
            for g in opt.param_groups:
                g['lr'] = g['initial_lr'] = lr * (args.head_lr_scale if g.get('glowica_dekodera') else 1.0)
        # z --lr_warmup: liniowo od 1% do --lr przez N kroków (licząc od startu tego przebiegu), potem stałe
        _w = args.lr_warmup
        _f = (lambda s: min(1.0, 0.01 + 0.99 * s / _w)) if _w else (lambda _: 1.0)
        schedulers = {k: torch.optim.lr_scheduler.LambdaLR(o, _f) for k, o in (('scheduler_g', opt_g), ('scheduler_d', opt_d))}
        lr_sched = schedulers['scheduler_g']
        if args.local_rank == 0: logger.info(f"Stałe tempo uczenia {lr} od kroku {step}" + (f", rozgrzewka {_w} kroków" if _w else ""))
        if args.local_rank == 0 and args.decoder_only:
            _opis = {}
            for g in opt_g.param_groups:
                _opis[f"nowa część ({nowa_czesc.rstrip('.')})" if g.get('glowica_dekodera') else "reszta dekodera"] = g['initial_lr']
            logger.info("Tempo docelowe: " + ", ".join(f"{k} {v:g}" for k, v in _opis.items())
                        + (f"; reszta dekodera stoi przez pierwsze {args.head_warmup} kroków" if args.head_warmup else ""))
    if lr_schedule == 'plateau':
        dec_from = ckpt.get('dec_trainable_from', step) if args.stage2 else None   # the decoder lr ramps from here
        for i, g in enumerate(opt_g.param_groups):
            g['base_lr'], g['ramp_from'] = (lr, None) if i < n_enc_groups else (lr * dec_lr_scale, dec_from)
        for g in opt_d.param_groups: g['base_lr'], g['ramp_from'] = lr, None
        val_ds = TTSDataset(data_cfg['val_metadata'], sample_rate=data_cfg['sample_rate'])
        val_clips = [val_ds[i][:data_cfg.get('segment_size')] for i in range(min(len(val_ds), train_cfg.get('val_clips', 64)))]
        eval_interval = train_cfg.get('eval_interval', 10000)
    # Ocena na odłożonych klipach (--heldout_every): stała pula, te same klipy za każdym razem, tylko karta 0.
    heldout = None
    if args.heldout_every > 0 and args.local_rank == 0:
        _val_ds = TTSDataset(data_cfg['val_metadata'], sample_rate=data_cfg['sample_rate'])
        _n = min(len(_val_ds), args.heldout_clips)
        _idx = sorted({round(i * (len(_val_ds) - 1) / max(_n - 1, 1)) for i in range(_n)})   # rozłożone po całej liście walidacyjnej
        _max = int(args.heldout_max_sec * data_cfg['sample_rate'])
        heldout = {
            "clips": [_val_ds[i][:_max] for i in _idx],
            "mel": mel_transforms_loss,                                            # ta sama miara co strata Mel w treningu
            "logmel_full": get_mel_transforms(data_cfg, device, "logmel_fullband"),   # logarytmiczna, całe pasmo do 24 kHz
        }
        logger.info(f"Ocena na odłożonych klipach co {args.heldout_every} kroków: {len(heldout['clips'])} klipów, "
                    f"{sum(c.shape[-1] for c in heldout['clips']) / data_cfg['sample_rate'] / 60:.1f} min")
    terms = ExtraTerms(tail_w=args.tail_consistency, hiband_w=args.hiband_w, floor_w=args.floor_w, floor_db=args.floor_db,
                       hiband_bin=round(args.hiband_hz * 2048 / data_cfg['sample_rate']) if args.hiband_hz else 558,
                       hiband_floor=args.hiband_floor, linie_w=args.linie_w)
    if args.local_rank == 0 and args.hiband_w > 0:
        logger.info(f"Strata pasma wysokiego: waga {args.hiband_w}, od {terms.hiband_bin * data_cfg['sample_rate'] / 2048:.0f} Hz, "
                    f"podłoga {args.hiband_floor:g}")
    if args.tail_consistency > 0:
        stats = torch.load(args.tail_stats, map_location='cpu')
        terms.tail_norm = (stats['mean'].view(1, -1, 1).to(device), stats['std'].view(1, -1, 1).to(device))

    # generator EMA, used by the held-out check and eval renders; a frozen decoder is its own EMA
    enc_ema, dec_ema = encoder, decoder
    if ema_decay > 0:
        enc_ema = encoder if args.decoder_only else copy.deepcopy(_mod(encoder)).eval().requires_grad_(False)
        if not args.encoder_only: dec_ema = copy.deepcopy(_mod(decoder)).eval().requires_grad_(False)
        if step > 0:
            if 'encoder_ema' in ckpt: enc_ema.load_state_dict(ckpt['encoder_ema'])
            if 'decoder_ema' in ckpt and not args.encoder_only: dec_ema.load_state_dict(ckpt['decoder_ema'])
    # must never move: the frozen decoder (stage 1) or the decoder's BatchNorm statistics (stage 2)
    def frozen_state():
        if args.encoder_only: return decoder.state_dict()
        if args.decoder_only: return _mod(encoder).state_dict()
        return {f"{i}.{k}": v for i, m in enumerate(dec_bns) for k, v in m.named_buffers()}
    frozen_ref = {k: v.detach().cpu().clone() for k, v in frozen_state().items()}

    def set_train_modes():
        if args.decoder_only: encoder.eval(); decoder.train(); return
        encoder.train()
        if freeze_bn: freeze_batchnorm(encoder)
        if args.encoder_only: decoder.eval(); return
        decoder.train()
        for m in dec_bns: m.eval()
    set_train_modes()

    def save(step):
        assert all(torch.equal(v.detach().cpu(), frozen_ref[k]) for k, v in frozen_state().items()), "frozen state changed"
        extra = {"ae_config": ae_cfg, "stage": 1 if args.encoder_only else 2 if args.stage2 else "dekoder" if args.decoder_only else 0}
        if args.stage2: extra["dec_trainable_from"] = dec_from
        if ema_decay > 0:
            extra["encoder_ema"] = enc_ema.state_dict()
            if not args.encoder_only: extra["decoder_ema"] = dec_ema.state_dict()
        if slm is not None: extra["slm"], extra["opt_slm"] = slm.head.state_dict(), opt_slm.state_dict()
        save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, logger, ckpt_dir, decoder_source, extra)

    crop_len = int(data_cfg['sample_rate'] * args.d_crop)   # wycinek dla MPD/MRD (przepis oryginału: 0,19 s)
    if args.local_rank == 0 and args.d_crop != 0.19:
        logger.info(f"Wycinek dla MPD/MRD: {args.d_crop:g} s ({crop_len} próbek; segment {data_cfg.get('segment_size')} próbek)")
    # Pasek postępu: średnia z ostatnich 2 epok (kroków na epokę na jedną kartę, jak liczy tqdm).
    meter_window = max(1, 2 * len(dataloader))
    g_meter, d_meter, mel_meter = AverageMeter(meter_window), AverageMeter(meter_window), AverageMeter(meter_window)
    if args.local_rank == 0: logger.info(f"Pasek postępu: G/D/Mel to średnia z ostatnich {meter_window} kroków (2 epoki)")
    last_saved = None
    
    avg_modules = ([] if args.decoder_only else [encoder]) + [mpd, mrd] + ([] if args.encoder_only else [decoder])
    if slm is not None: avg_modules.append(slm.head)
    if args.avg_every > 0:
        if args.encoder_only or args.stage2 or ema_decay > 0 or lr_schedule == 'plateau':
            raise SystemExit("--avg_every działa tylko z podstawowym przepisem (bez --encoder_only, --stage2, EMA i harmonogramu plateau)")
        sync_weights(avg_modules, from_rank0=True)   # ten sam punkt startu na każdej karcie
        logger.info(f"Uśrednianie wag co {args.avg_every} kroków na {dist.get_world_size()} kartach (bez synchronizacji gradientów)")

    cialo_zamrozone = [False]   # stan zamrożenia ciała dekodera przy --head_warmup (ustawiany w pętli)
    prof = Prof(device) if args.profile > 0 and args.local_rank == 0 else None
    if prof: prof.start()

    while step < args.total_steps:
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True) if args.local_rank == 0 else dataloader
        for batch in pbar:
            if step >= args.total_steps: break
            if lr_schedule == 'plateau': lr_sched.step(step)
            if args.head_warmup:
                # Ciało dekodera naprawdę zamrożone (bez gradientów, więc bez przejścia wstecz przez nie) do kroku
                # --head_warmup, potem odmrożone z tempem rosnącym liniowo przez --body_ramp kroków.
                zamrozone = step < args.head_warmup
                if zamrozone != cialo_zamrozone[0]:
                    for n, p in _mod(decoder).named_parameters():
                        if not n.startswith(nowa_czesc): p.requires_grad_(not zamrozone)
                    cialo_zamrozone[0] = zamrozone
                    if args.local_rank == 0:
                        logger.info(f"Krok {step}: " + (f"reszta dekodera zamrożona, uczy się sama nowa część ({nowa_czesc.rstrip('.')})" if zamrozone else
                                                        f"cały dekoder w treningu (tempo reszty rośnie przez {args.body_ramp} kroków)"))
                if zamrozone:   # statystyki BatchNorm zamrożonej części też stoją (set_train_modes włącza tryb train)
                    for n, m in _mod(decoder).named_modules():
                        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not n.startswith(nowa_czesc): m.eval()
                if not zamrozone and step < args.head_warmup + args.body_ramp:
                    for g in opt_g.param_groups:
                        if g.get('cialo_dekodera'):
                            g['lr'] = g.get('initial_lr', lr) * (step - args.head_warmup + 1) / max(1, args.body_ramp)
            result = train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
                                update_discriminator=(step > args.d_warmup), recon_only=(0 < args.d_warmup and step <= args.d_warmup),
                                recon_full_segment=(recon != 'linmel_12k'),
                                fm_composite=fm_composite, encoder_only=args.encoder_only, terms=terms, clip_generator=clip_generator, prof=prof,
                                slm=slm, opt_slm=opt_slm)
            if result is None: break
            loss_g, loss_d, loss_mel, extras = result
            if lr_schedule != 'plateau':
                for s in schedulers.values(): s.step()
            if args.avg_every > 0 and step > 0 and step % args.avg_every == 0:
                sync_weights(avg_modules)   # przed zapisem i próbką, więc te zawsze dotyczą wag uśrednionych
            if ema_decay > 0:
                ema_update(enc_ema, _mod(encoder), ema_decay)
                if not args.encoder_only: ema_update(dec_ema, _mod(decoder), ema_decay)

            if lr_schedule == 'plateau' and step > 0 and step % eval_interval == 0:
                val = torch.zeros(1, device=device)   # rank 0 scores, every rank applies the same decision
                if args.local_rank == 0:
                    val[0] = heldout_recon(_mod(enc_ema), _mod(dec_ema), mel_transform_input, mel_transforms_loss, val_clips, device)
                    set_train_modes()
                dist.broadcast(val, 0)
                lr_sched.report(val.item())
                if args.local_rank == 0:
                    writer.add_scalar("Loss/Heldout_Recon", val.item(), step)
                    logger.info(f"step {step}: held-out recon {val.item():.4f}, best {lr_sched.best:.4f}, lr {lr * lr_sched.scale:.2e}")
            
            if args.local_rank == 0:
                g_meter.update(loss_g); d_meter.update(loss_d); mel_meter.update(loss_mel)
                if step % 10 == 0:
                    writer.add_scalar("Loss/Generator", loss_g, step); writer.add_scalar("Loss/Discriminator", loss_d, step); writer.add_scalar("Loss/Mel", loss_mel, step); writer.add_scalar("Training/LR", lr_sched.get_last_lr()[0], step)
                    for name, value in extras.items(): writer.add_scalar(f"Loss/{name}", value, step)
                # Pasek pokazuje średnią z ostatnich 2 epok. Oryginał pokazywał średnią od startu przebiegu, która spada
                # jeszcze długo po tym, jak bieżąca strata przestała się zmieniać.
                pbar.set_postfix({"Step": step, "G": f"{g_meter.recent:.4f}", "D": f"{d_meter.recent:.4f}", "Mel": f"{mel_meter.recent:.4f}", "LR": f"{lr_sched.get_last_lr()[0]:.2e}"})
                if step % train_cfg['save_interval'] == 0:
                    save(step); last_saved = step
                # próbka odsłuchowa niezależnie od zapisu punktu kontrolnego (--eval_every; domyślnie razem z zapisem)
                if step % (args.eval_every or train_cfg['save_interval']) == 0:
                    if args.eval_input:
                        evaluate(enc_ema, dec_ema, mel_transform_input, args.eval_input, os.path.join(ckpt_dir, "eval"), step, device,
                                 data_cfg['sample_rate'], args.local_rank, keep_decoder_eval=args.encoder_only)
                        set_train_modes()   # evaluate() leaves BatchNorm in train mode
                # przy --avg_every liczone po uśrednieniu wag, o ile --heldout_every jest wielokrotnością --avg_every
                if heldout is not None and step % args.heldout_every == 0:
                    _t = time.time()
                    _rec = heldout_recon(_mod(enc_ema), _mod(dec_ema), mel_transform_input, heldout["mel"], heldout["clips"], device)
                    _full = heldout_recon(_mod(enc_ema), _mod(dec_ema), mel_transform_input, heldout["logmel_full"], heldout["clips"], device)
                    set_train_modes()   # heldout_recon() leaves the models in eval mode
                    writer.add_scalar("Loss/Heldout_Recon", _rec, step)
                    writer.add_scalar("Loss/Heldout_LogMel_Full", _full, step)
                    _lin = heldout_linie(_mod(enc_ema), _mod(dec_ema), mel_transform_input, heldout["clips"], device)
                    set_train_modes()
                    writer.add_scalar("Loss/Heldout_Linie", _lin, step)
                    _ut = ""
                    if args.heldout_utmos:
                        if "utmos" not in heldout:
                            from training.utmos import wczytaj_utmos, utmos as _utmos_jeden
                            heldout["utmos"] = wczytaj_utmos(device)
                            heldout["utmos_oryginal"] = sum(_utmos_jeden(heldout["utmos"], c.to(device), data_cfg['sample_rate'])
                                                            for c in heldout["clips"]) / len(heldout["clips"])
                            logger.info(f"UTMOS oryginałów odłożonych klipów: {heldout['utmos_oryginal']:.3f}")
                        _u = heldout_utmos(_mod(enc_ema), _mod(dec_ema), mel_transform_input, heldout["clips"], device,
                                           heldout["utmos"], data_cfg['sample_rate'])
                        set_train_modes()
                        writer.add_scalar("Loss/Heldout_UTMOS", _u, step)
                        writer.add_scalar("Loss/Heldout_UTMOS_oryginal", heldout["utmos_oryginal"], step)
                        _ut = f", UTMOS {_u:.3f} (oryginały {heldout['utmos_oryginal']:.3f})"
                    logger.info(f"step {step}: odłożone klipy — recon {_rec:.4f}, log-mel całe pasmo {_full:.4f}, prążki {_lin:.4f}{_ut} ({time.time() - _t:.1f} s)")
            if prof:
                prof.mark("reszta pętli (harmonogram, uśrednianie, log, zapis)")
                if prof.n == args.profile:
                    prof.report(logger); prof = None
                else: prof.n += 1
            step += 1
        epoch += 1
    if args.local_rank == 0 and step >= args.total_steps and last_saved != step - 1:
        save(step - 1)   # the final weights, whatever save_interval is
    dist.destroy_process_group()

if __name__ == "__main__": main()
