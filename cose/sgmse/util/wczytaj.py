"""[syntezujemy_polska_mowe] Zamiennik torchaudio.load: torchaudio 2.9+ wymaga do wczytywania pakietu torchcodec.
Zwraca to samo co torchaudio.load: (tensor [kanały, próbki] float32, częstotliwość)."""
import soundfile as sf
import torch


def load(path, *args, **kwargs):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(x.T.copy()), sr
