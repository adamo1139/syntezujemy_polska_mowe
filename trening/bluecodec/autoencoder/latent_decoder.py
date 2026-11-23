from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import CausalConv1d, CausalConvNeXtBlock, ConvNeXtBlock


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

    Wyrównanie ramek (`wyrownanie`):
      "srodek" (domyślne): ramka klatki k wyśrodkowana na jej odcinku [k·hop, (k+1)·hop), jak okno analizy STFT
        enkodera, z którego powstał latent klatki k; na próbkę wpływają klatki do 2 w przód (ok. 21 ms przy 48 kHz).
      "koniec": ramka kończy się razem z odcinkiem klatki (pokrywa go i n_fft/hop − 1 poprzednich).
    Test na zamrożonym ciele dekodera ae_wsm_210_220 (4 klipy VoxCPM2, 300 kroków samej głowicy, Adam 1e-3, strata
    mel z treningu kodeka; stara głowica: 1,200): "srodek" 1,451, "koniec" 2,100.
    Pierwsza warstwa ma ten sam kształt co w VocoderHead (`layer1`), więc da się ją przenieść ze starego dekodera."""

    def __init__(self, dim=512, hdim=2048, n_fft=2048, hop=512, kernel_size=3, pad_mode="zeros", wyrownanie="srodek"):
        super().__init__()
        if n_fft % hop or n_fft < hop:
            raise ValueError(f"n_fft ({n_fft}) musi być wielokrotnością hop ({hop})")
        if wyrownanie not in ("koniec", "srodek"):
            raise ValueError(f"wyrownanie: koniec albo srodek, nie {wyrownanie!r}")
        self.n_fft, self.hop, self.wyrownanie = n_fft, hop, wyrownanie
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
        # prążek stały (0 Hz) i Nyquista (sr/2) zawsze zero: żadna strata ich nie pilnuje (filtry melowe na samej
        # górze są puste), a bez dyskryminatorów głowica potrafi tam wpisać głośny ton 24 kHz
        mag = torch.cat([torch.zeros_like(mag[:, :1]), mag[:, 1:-1], torch.zeros_like(mag[:, -1:])], dim=1)
        faza = h[:, nb:].float()
        widmo = torch.complex(mag * torch.cos(faza), mag * torch.sin(faza))
        ramki = torch.fft.irfft(widmo, n=self.n_fft, dim=1) * self.window[None, :, None]   # [B, n_fft, T]
        L = (T - 1) * self.hop + self.n_fft
        fold = lambda r: F.fold(r, output_size=(1, L), kernel_size=(1, self.n_fft), stride=(1, self.hop)).reshape(r.shape[0], L)
        y = fold(ramki)
        obw = fold((self.window ** 2)[None, :, None].expand(1, self.n_fft, T))[0]      # suma okien^2
        y = y / obw.clamp(min=0.1)
        start = self.n_fft - self.hop if self.wyrownanie == "koniec" else (self.n_fft - self.hop) // 2
        return y[:, start: start + T * self.hop]


class STFTPostnet(nn.Module):
    """Poprawka na wyjściu dekodera: widmo gotowego dźwięku (STFT n_fft, hop) razem z cechami ciała dekodera trafia
    do kilku bloków ConvNeXt w czasie, które przewidują zespoloną maskę M = 1 + (a + i·b) na każdy prążek; dźwięk
    wraca przez odwrotne STFT z nakładaniem ramek. Ostatnia warstwa startuje od zer (M = 1), więc na starcie
    dekoder z tym modułem daje dokładnie to samo co bez niego.

    Wejście sieci: widmo skompresowane |X|^0.3 z fazą (część rzeczywista i urojona) oraz cechy ciała dekodera
    (wyrównane do ramek STFT). Moduł nie jest przyczynowy: patrzy num_layers · (ksz − 1) / 2 ramek w obie strony."""

    def __init__(self, feat_dim=512, n_fft=2048, hop=512, dim=384, intermediate_dim=1152, num_layers=4, ksz=7,
                 max_mask=4.0, aktywacja="gelu"):
        super().__init__()
        self.n_fft, self.hop, self.max_mask = n_fft, hop, max_mask
        nb = n_fft // 2 + 1
        self.proj_in = nn.Conv1d(2 * nb + feat_dim, dim, kernel_size=1)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim, kernel_size=ksz,
                                                   aktywacja=aktywacja)
                                     for _ in range(num_layers)])
        self.proj_out = nn.Conv1d(dim, 2 * nb, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, y, feats):
        """y: [B, N] dźwięk z głowicy; feats: [B, C, T] cechy ciała dekodera (N = T · hop)."""
        n = y.shape[-1]
        X = torch.stft(y.float(), self.n_fft, self.hop, window=self.window, center=True, return_complex=True)   # [B, nb, F]
        F_ = X.shape[-1]
        mag = X.abs()
        komp = mag.clamp_min(1e-8) ** 0.3 * X / mag.clamp_min(1e-8)          # |X|^0.3 · e^{iφ}
        f = feats.float()
        if f.shape[-1] < F_:                                                   # F = T + 1 ramek (center=True)
            f = torch.cat([f, f[..., -1:].expand(-1, -1, F_ - f.shape[-1])], dim=-1)
        h = self.proj_in(torch.cat([komp.real, komp.imag, f[..., :F_]], dim=1))
        for b in self.blocks:
            h = b(h)
        a, b_ = self.proj_out(h).chunk(2, dim=1)
        maska = torch.complex(1.0 + a, b_)
        modul = maska.abs()
        maska = maska * (modul.clamp(max=self.max_mask) / modul.clamp_min(1e-8))   # ogranicza wzmocnienie
        return torch.istft(X * maska, self.n_fft, self.hop, window=self.window, center=True, length=n)


# Defaults based on configs/tts.json -> ae.decoder (the 1.5M decoder). The replicate variant: E12b and Supertonic-3.
DEFAULT_DECODER_CFG = {
    "idim": 24, "hdim": 512, "intermediate_dim": 2048, "ksz": 7,
    "dilation_lst": [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
    "head": {"idim": 512, "hdim": 2048, "odim": 512, "ksz": 3},
    "chunk_compress_factor": 1, "normalizer_scale": 1.0, "pad_mode": "zeros",
}
REPLICATE_DECODER_CFG = {**DEFAULT_DECODER_CFG, "pad_mode": "replicate"}

class PrzedniaSiec(nn.Module):
    """[syntezujemy_polska_mowe] Transformer przy niskiej częstotliwości klatek (latenty ściśnięte ×6, 15,6 Hz przy
    48 kHz), przed rozpakowaniem do 93,75 Hz — jak pre_transformer w Code2Wav Qwen3-Omni. Parametr kosztuje tu
    6 razy mniej niż w ciele dekodera. Przyczynowy, attention w oknie `okno` klatek wstecz (RoPE, pre-RMSNorm,
    SwiGLU), więc dekoder zostaje strumieniowalny. Wyjście dodawane rezydualnie do latentów; ostatnia warstwa od
    zer, więc na starcie dekoder daje dokładnie to samo co bez niej."""

    def __init__(self, idim=24, ccf=6, dim=512, num_layers=7, heads=8, ffn=2048, okno=64):
        super().__init__()
        self.ccf, self.heads, self.okno = ccf, heads, okno
        self.inp = nn.Linear(idim * ccf, dim)
        self.warstwy = nn.ModuleList([nn.ModuleDict({
            "n1": nn.RMSNorm(dim), "qkv": nn.Linear(dim, 3 * dim, bias=False), "o": nn.Linear(dim, dim, bias=False),
            "n2": nn.RMSNorm(dim), "w13": nn.Linear(dim, 2 * ffn, bias=False), "w2": nn.Linear(ffn, dim, bias=False),
        }) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, idim * ccf)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _rope(self, x):   # x [B, H, T, d]
        T, d = x.shape[-2], x.shape[-1]
        f = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        k = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
        c, s_ = k.cos().to(x.dtype), k.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], dim=-1).flatten(-2)

    def forward(self, x):   # x [B, idim, T] przy pełnej częstotliwości klatek -> to samo
        B, C, T = x.shape
        pad = (-T) % self.ccf
        xp = F.pad(x, (0, pad), mode="replicate") if pad else x
        L = xp.shape[-1] // self.ccf
        z = xp.view(B, C, L, self.ccf).permute(0, 1, 3, 2).reshape(B, C * self.ccf, L)   # jak compress_latents
        h = self.inp(z.transpose(1, 2))                                                   # [B, L, dim]
        i = torch.arange(L, device=x.device)
        maska = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.okno)        # przyczynowo, w oknie
        for w in self.warstwy:
            q, k, v = w["qkv"](w["n1"](h)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(self._rope(q), self._rope(k), v, attn_mask=maska)
            h = h + w["o"](a.transpose(1, 2).reshape(B, L, -1))
            g, u = w["w13"](w["n2"](h)).chunk(2, dim=-1)
            h = h + w["w2"](F.silu(g) * u)
        dz = self.out(self.norm(h)).transpose(1, 2)                                       # [B, C*ccf, L]
        dx = dz.view(B, C, self.ccf, L).permute(0, 1, 3, 2).reshape(B, C, L * self.ccf)   # rozpakowanie
        return x + dx[..., :T]


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

        aktywacja = cfg.get('aktywacja', 'gelu')   # gelu | gelu_snake | snakebeta (ciało dekodera i moduł poprawki)
        pre = cfg.get('prenet')
        self.prenet = None
        if pre:
            self.prenet = PrzedniaSiec(idim=in_channels, ccf=int(pre.get('ccf', 6)), dim=int(pre.get('dim', 512)),
                                       num_layers=int(pre.get('num_layers', 7)), heads=int(pre.get('heads', 8)),
                                       ffn=int(pre.get('ffn', 2048)), okno=int(pre.get('okno', 64)))
        self.embed = CausalInputProjection(in_channels, dim, kernel_size=kernel_size, pad_mode=pad_mode)

        self.convnext = nn.ModuleList([
            CausalConvNeXtBlock(
                dim=dim, 
                intermediate_dim=intermediate_dim, 
                kernel_size=kernel_size, 
                dilation=d,
                pad_mode=pad_mode,
                aktywacja=aktywacja,
            )
            for d in dilations
        ])

        self.final_norm = FinalBatchNorm1d(dim)

        pn = cfg.get('postnet')
        self.postnet = None
        if pn:
            self.postnet = STFTPostnet(
                feat_dim=dim, n_fft=int(pn.get('n_fft', 2048)), hop=int(cfg['head'].get('hop', cfg['head']['odim'])),
                dim=int(pn.get('dim', 384)), intermediate_dim=int(pn.get('intermediate_dim', 1152)),
                num_layers=int(pn.get('num_layers', 4)), ksz=int(pn.get('ksz', 7)),
                max_mask=float(pn.get('max_mask', 4.0)), aktywacja=aktywacja,
            )

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
                wyrownanie=head_cfg.get('wyrownanie', 'srodek'),
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
        if self.prenet is not None:
            x = self.prenet(x)

        x = self.embed(x)
        for block in self.convnext:
            x = block(x)

        x = self.final_norm(x)
        waveform = self.head(x)
        if self.postnet is not None:
            waveform = self.postnet(waveform, x)
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

