# Szybkość ASR na CPU: Parakeet Ultra, Parakeet Redux, Canary-1B-v2 (2026-11-26)

Pomiar szybkości modeli ASR do filtrowania syntezy (`skrypty/czysc_asr.py`) na jednym rdzeniu CPU.

**Warunki:** i7-11700K, jeden rdzeń (`taskset -c 4`, 1 wątek PyTorch/OpenMP), partia 1 (klip po klipie),
5 klipów z próby drugiej partii (`wyniki/partia2_proba`, Ogg Opus 128 kbit/s, 48 kHz), razem 44,3 s mowy;
audio dekodowane i przepróbkowane do 16 kHz przed pomiarem (czas dekodowania nie wlicza się).
Parakeety przez Photon (`moondream`, `device="cpu"`, `cpu_threads=1`), Canary przez NeMo 3.0.0 (torch CPU).
Skrypt: `skrypty/asr_cpu.py`; surowe wyniki: `probki/asr_cpu_5.json`.

| model | parametry / źródło | RTF | szybciej niż czas rzeczywisty | wczytanie modelu |
|---|---|---|---|---|
| **Parakeet Redux** | `moondream/parakeet-redux` (Photon) | **0,081** | **12,3×** | 5 s |
| Parakeet Ultra | `moondream/parakeet-ultra` (Photon) | 0,192 | 5,2× | 11 s |
| Canary-1B-v2 | `nvidia/canary-1b-v2` (NeMo) | 0,711 | 1,4× | 115 s |

RTF = czas liczenia / długość nagrania. Na 20 klipach (219,5 s mowy) Ultra i Redux dały to samo:
RTF 0,192 i 0,089.

- Redux jest ok. 2,4× szybszy od Ultra i ok. 9× od Canary.
- Canary-1B-v2 na jednym rdzeniu ledwo wyprzedza czas rzeczywisty i długo się wczytuje (ok. 11 GB RAM) —
  do filtrowania tysięcy godzin na CPU się nie nadaje.

**Dokładności tu nie mierzymy.** CER względem tekstu podanego VoxCPM2 łączy błędy TTS (np. „Sprowadzanie”
przeczytane jako „Wprowadzanie”), błędy ASR i różnice zapisu („Pan/i”, „5-7 cm”), więc o samym ASR nic nie mówi.
Do porównania dokładności potrzebna jest prawdziwa, „brudna” polska mowa z ręcznymi transkrypcjami — np.
[BIGOS v2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) (Common Voice, FLEURS, MLS, rozmowy
telefoniczne) i [PELCRA](https://huggingface.co/datasets/amu-cai/pl-asr-pelcra-for-bigos) (mowa spontaniczna).

## Powtórzenie

```bash
U=/media/adamo/usemka/tmp      # pamięć HF i środowisko NeMo poza dyskiem głównym
taskset -c 4 python skrypty/asr_cpu.py wyniki/partia2_proba --klipy 5 --modele ultra redux --wyjscie probki/asr_cpu_5.json
taskset -c 4 $U/venv_nemo/bin/python skrypty/asr_cpu.py wyniki/partia2_proba --klipy 5 --modele canary \
  --wyjscie probki/asr_cpu_5.json
```

Środowisko NeMo: `uv venv` + `torch torchaudio` (indeks CPU PyTorch) + `nemo_toolkit[asr] morfeusz2 soundfile`.
