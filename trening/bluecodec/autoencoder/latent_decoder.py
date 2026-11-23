from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CausalConv1d, CausalConvNeXtBlock


class CausalInputProjection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad_mode="zeros"):
        super().__init__()
        self.net = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, pad_mode=pad_mode)

    def forward(self, x):
        return self.net(x)


class FinalBatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(x)


class VocoderHead(nn.Module):
    def __init__(self, dim=512, hdim=2048, out_dim=512, kernel_size=3, pad_mode="zeros"):
        super().__init__()
        # Match the AE decoder head config:
        # layer1 causal conv -> PReLU -> layer2 1x1 conv -> transpose -> reshape.
        self.layer1 = CausalInputProjection(dim, hdim, kernel_size=kernel_size, pad_mode=pad_mode)
        self.act = nn.PReLU()
        self.layer2 = nn.Conv1d(hdim, out_dim, kernel_size=1, bias=False)

        # Keep the final projection small at initialization.
        self.layer2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)  # [B, 512, T]

        # Match notebook trace:
        # [B, 512, T] -> [B, T, 512] -> [B, T * 512]
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(x.shape[0], -1)
        return x

class ISTFTHead(nn.Module):
    """Głowica w stylu Vocos: na każdą klatkę (co `hop` próbek) widmo ramki `n_fft` (log-amplituda i faza), dźwięk przez
    odwrotne FFT, okno Hanna i sumowanie nakładających się ramek (overlap-add). Kawałki nie są sklejane „na styk”,
    tylko przenikają się na długości okna, więc granice klatek nie zostawiają okresowego śladu.

    Ramka klatki k jest wyśrodkowana na jej odcinku [k·hop, (k+1)·hop): na próbkę wpływają klatki do 2 w przód
    (przy n_fft 2048 i hop 512), czyli dekoder z tą głowicą nie jest przyczynowy (ok. 21 ms wyprzedzenia przy 48 kHz).
    Pierwsza warstwa ma ten sam kształt co w VocoderHead (`layer1`), więc da się ją przenieść ze starego dekodera."""

    def __init__(self, dim=512, hdim=2048, n_fft=2048, hop=512, kernel_size=3, pad_mode="zeros"):
        super().__init__()
        if n_fft % hop or n_fft < hop:
            raise ValueError(f"n_fft ({n_fft}) musi być wielokrotnością hop ({hop})")
        self.n_fft, self.hop = n_fft, hop
        self.layer1 = CausalInputProjection(dim, hdim, kernel_size=kernel_size, pad_mode=pad_mode)
        self.act = nn.PReLU()
        self.out = nn.Conv1d(hdim, n_fft + 2, kernel_size=1)
        nn.init.normal_(self.out.weight, 0, 0.01)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[: n_fft // 2 + 1] = -4.0   # cichy start: amplitudy ok. e^-4
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        h = self.out(self.act(self.layer1(x)))                         # [B, n_fft + 2, T]
        B, _, T = h.shape
        nb = self.n_fft // 2 + 1
        mag = torch.exp(h[:, :nb].float().clamp(max=6.0))
        faza = h[:, nb:].float()
        widmo = torch.complex(mag * torch.cos(faza), mag * torch.sin(faza))
        ramki = torch.fft.irfft(widmo, n=self.n_fft, dim=1) * self.window[None, :, None]   # [B, n_fft, T]
        L = (T - 1) * self.hop + self.n_fft
        fold = lambda r: F.fold(r, output_size=(1, L), kernel_size=(1, self.n_fft), stride=(1, self.hop)).reshape(r.shape[0], L)
        y = fold(ramki)
        obw = fold((self.window ** 2)[None, :, None].expand(1, self.n_fft, T))[0]      # suma okien^2
        y = y / obw.clamp(min=0.1)
        start = (self.n_fft - self.hop) // 2                            # ramka k wyśrodkowana na swoim odcinku
        return y[:, start: start + T * self.hop]


# Defaults based on configs/tts.json -> ae.decoder (the 1.5M decoder). The replicate variant: E12b and Supertonic-3.
DEFAULT_DECODER_CFG = {
    "idim": 24, "hdim": 512, "intermediate_dim": 2048, "ksz": 7,
    "dilation_lst": [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
    "head": {"idim": 512, "hdim": 2048, "odim": 512, "ksz": 3},
    "chunk_compress_factor": 1, "normalizer_scale": 1.0, "pad_mode": "zeros",
}
REPLICATE_DECODER_CFG = {**DEFAULT_DECODER_CFG, "pad_mode": "replicate"}

class LatentDecoder1D(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = DEFAULT_DECODER_CFG

        in_channels = cfg['idim']
        dim = cfg['hdim']
        intermediate_dim = cfg['intermediate_dim']
        kernel_size = cfg['ksz']
        dilations = cfg['dilation_lst']
        pad_mode = cfg.get('pad_mode', 'zeros')   # causal padding: zeros (1.5M decoder) or replicate (E12b)

        self.input_channels = in_channels
        self.chunk_compress_factor = int(cfg.get('chunk_compress_factor', 1))
        self.normalizer_scale = float(cfg.get('normalizer_scale', 1.0))
        self.compressed_channels = int(
            cfg.get('compressed_idim', in_channels * self.chunk_compress_factor)
        )

        # Match traced decoder buffers.
        self.register_buffer('latent_mean', torch.zeros(1, in_channels, 1), persistent=False)
        self.register_buffer('latent_std', torch.ones(1, in_channels, 1), persistent=False)

        self.embed = CausalInputProjection(in_channels, dim, kernel_size=kernel_size, pad_mode=pad_mode)

        self.convnext = nn.ModuleList([
            CausalConvNeXtBlock(
                dim=dim, 
                intermediate_dim=intermediate_dim, 
                kernel_size=kernel_size, 
                dilation=d,
                pad_mode=pad_mode,
            )
            for d in dilations
        ])

        self.final_norm = FinalBatchNorm1d(dim)

        head_cfg = cfg['head']
        head_input_dim = int(head_cfg.get('idim', dim))
        if head_input_dim != dim:
            raise ValueError(
                f"Decoder head idim={head_input_dim} must match decoder hdim={dim}"
            )

        if head_cfg.get('type', 'patch') == 'istft':
            self.head = ISTFTHead(
                dim=head_input_dim,
                hdim=head_cfg['hdim'],
                n_fft=int(head_cfg.get('n_fft', 2048)),
                hop=int(head_cfg.get('hop', head_cfg['odim'])),
                kernel_size=int(head_cfg.get('ksz', 3)),
                pad_mode=pad_mode,
            )
        else:
            self.head = VocoderHead(
                dim=head_input_dim,
                hdim=head_cfg['hdim'],
                out_dim=head_cfg['odim'],
                kernel_size=int(head_cfg.get('ksz', 3)),
                pad_mode=pad_mode,
            )

    def _prepare_latents(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected latents with shape [B, C, T], got {tuple(x.shape)}")

        if x.shape[1] == self.input_channels:
            return x

        if x.shape[1] != self.compressed_channels:
            raise ValueError(
                f"Expected latent channels to be {self.input_channels} or "
                f"{self.compressed_channels}, got {x.shape[1]}"
            )

        if self.normalizer_scale != 0.0 and self.normalizer_scale != 1.0:
            x = x / self.normalizer_scale

        bsz, _, latent_len = x.shape
        x = x.reshape(bsz, self.input_channels, self.chunk_compress_factor, latent_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(bsz, self.input_channels, latent_len * self.chunk_compress_factor)
        return x

    def load_state_dict(self, state_dict, strict=True):
        import re

        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key

            # --- embed (was input_conv) ---
            if key.startswith("input_conv.0."):
                new_key = key.replace("input_conv.0.", "embed.net.", 1)
            elif key.startswith("input_conv.1."):
                continue
            elif key.startswith("input_conv.net."):
                new_key = key.replace("input_conv.net.", "embed.net.", 1)

            # --- convnext (was blocks) ---
            elif key.startswith("blocks."):
                new_key = key.replace("blocks.", "convnext.", 1)
                m = re.match(r"convnext\.(\d+)\.dwconv\.(weight|bias)$", new_key)
                if m:
                    new_key = f"convnext.{m.group(1)}.dwconv.net.{m.group(2)}"

            # --- final_norm ---
            elif key.startswith("final_norm.") and not key.startswith("final_norm.norm."):
                new_key = key.replace("final_norm.", "final_norm.norm.", 1)

            # --- head ---
            elif key.startswith("head.conv1."):
                new_key = key.replace("head.conv1.", "head.layer1.net.", 1)
            elif key.startswith("head.conv2."):
                new_key = key.replace("head.conv2.", "head.layer2.", 1)

            if new_key == "head.layer2.bias":
                continue

            remapped[new_key] = value

        return super().load_state_dict(remapped, strict=strict)

    def forward(self, x):
        x = self._prepare_latents(x)
        x = x * self.latent_std + self.latent_mean

        x = self.embed(x)
        for block in self.convnext:
            x = block(x)

        x = self.final_norm(x)
        waveform = self.head(x)
        return waveform


# Official Supertonic-3 vocoder (OpenRAIL-M, not bundled): downloaded at a pinned revision, needs `pip install onnx`.
SUPERTONIC3_REVISION = "3cadd1ee6394adea1bd021217a0e650ede09a323"   # vocoder.onnx md5 68e5b768810cb3c2cf7a27f3ce2494e3
_SUPERTONIC3_ALIAS = {"onnx::Conv_1441": "embed.net.weight", "onnx::Conv_1442": "embed.net.bias", "onnx::PRelu_1506": "head.act.weight"}


def supertonic3_decoder_state(onnx_path=None):
    """vocoder.onnx as a LatentDecoder1D(REPLICATE_DECODER_CFG) state dict, plus the graph's latent mean and std."""
    import onnx
    from onnx import numpy_helper
    if onnx_path is None:
        from huggingface_hub import hf_hub_download
        onnx_path = hf_hub_download("Supertone/supertonic-3", "onnx/vocoder.onnx", revision=SUPERTONIC3_REVISION)
    with torch.random.fork_rng(devices=[]):   # only the shapes are needed; leave the caller's RNG alone
        ref = LatentDecoder1D(cfg=REPLICATE_DECODER_CFG).state_dict()
    off = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in onnx.load(onnx_path).graph.initializer}
    sd = {"final_norm.norm.num_batches_tracked": ref["final_norm.norm.num_batches_tracked"]}
    for k, v in off.items():
        n = k[len("tts.ae.decoder."):] if k.startswith("tts.ae.decoder.") else _SUPERTONIC3_ALIAS.get(k)
        if n is not None: sd[n] = v.reshape(ref[n].shape)
    return sd, off["tts.ae.latent_mean"], off["tts.ae.latent_std"]

