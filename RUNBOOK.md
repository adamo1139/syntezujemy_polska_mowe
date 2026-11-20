# Runbook: nano-vllm-voxcpm i synteza VoxCPM2 na 8× RTX 3090 Ti

Instalacja silnika `nano-vllm-voxcpm`, próba na jednej karcie, strojenie i przebieg na wielu kartach
z możliwością pominięcia dowolnych (np. 0 albo 2). Kontekst i decyzje: `PLAN.md`.

**Stan (2026-11-20):** spisane na podstawie kodu `nano-vllm-voxcpm` 2.0.5 (PyPI, 2026-10-12; repo
`a710128/nanovllm-voxcpm`, commit `b839543`). **Nie było jeszcze uruchomione na maszynie z 3090 Ti** —
na komputerze, na którym to powstało, jest tylko RTX 5080. W `skrypty/generuj.py` sprawdzony jest wybór kart,
a sama synteza nie. Pierwsze przejście kroków 2–5 potraktować jako test i poprawić runbook tam, gdzie
rzeczywistość się rozjedzie.

## 1. Wymagania

| co | wymaganie |
|---|---|
| system | Linux, sterownik NVIDIA, działające `nvidia-smi` |
| Python | 3.10–3.12 (pakiet nie wspiera 3.13) |
| narzędzia | `uv`, `git` |
| dysk | ok. 5 GB model (`model.safetensors` 4,58 GB + `audiovae.pth` 0,38 GB), kilka GB środowisko, do tego wyniki (50 h we FLAC to ok. 8–9 GB) |
| CPU | co najmniej 1 rdzeń na używaną kartę + 1–2 na proces sterujący (silnik to jeden proces na kartę) |
| RAM | każdy proces ładuje model osobno; przy 8 kartach liczyć kilkadziesiąt GB na czas ładowania (do zmierzenia) |

Silnik działa tylko na GPU (Triton + FlashAttention), bez trybu CPU.

Kontrola maszyny:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
python3 --version; uv --version; nproc; free -g; df -h .
```

Zapisz numery kart, które są zajęte albo mają zostać wolne — przydadzą się w kroku 4.

## 2. Instalacja

```bash
git clone https://github.com/adamo1139/syntezujemy_polska_mowe.git
cd syntezujemy_polska_mowe

uv venv --python 3.12 .venv
source .venv/bin/activate
```

### 2.1 Pakiety (bez kompilacji)

Wersje są przypięte w `wymagania.txt`; `flash-attn` idzie jako gotowe koło ze strony wydań
`Dao-AILab/flash-attention`, więc nie trzeba `nvcc` ani CUDA Toolkit.

```bash
uv pip install -r wymagania.txt --index-strategy unsafe-best-match
```

Co jest przypięte i dlaczego:

| pakiet | wersja | powód |
|---|---|---|
| `torch`, `torchaudio` | 2.8.0+cu128 | najnowszy torch, dla którego flash-attn 2.x ma gotowe koło (najnowszy na PyPI to 2.15 — bez koła) |
| `flash-attn` | 2.8.3.post2, koło `cu12torch2.8cxx11abiTRUE-cp312` | wydanie z 2026-10-13, ok. 256 MB |
| `nano-vllm-voxcpm` | 2.0.5 | wymaga `torch>=2.5.0`, z wyłączeniem 2.6 |

Koło musi pasować do trzech rzeczy naraz: Pythona (`cp312`), torch (`torch2.8`) i CUDA (`cu12`).
Przy innym Pythonie (3.10/3.11) podmienić `cp312` w adresie w `wymagania.txt` — wydanie ma koła dla
torch 2.4–2.8. Wydania `fa4-*` na tej samej stronie to FlashAttention 4, inny pakiet — nie brać.

### 2.2 Kontrola

```bash
python -c "
import torch, flash_attn, nanovllm_voxcpm
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| ABI cxx11:', torch._C._GLIBCXX_USE_CXX11_ABI)
print('karty:', torch.cuda.is_available(), torch.cuda.device_count())
print('flash-attn', flash_attn.__version__, '| nano-vllm-voxcpm', nanovllm_voxcpm.__version__)
"
```

Oczekiwane: `torch 2.8.0+cu128`, `ABI cxx11: True`, `karty: True` i liczba kart, `flash-attn 2.8.3.post2`.

- `karty: False` — sterownik NVIDIA za stary dla CUDA 12.8; zaktualizować sterownik.
- `ABI cxx11: False` — w `wymagania.txt` zamienić w adresie koła `cxx11abiTRUE` na `cxx11abiFALSE`.

Zapisz pełną listę wersji, żeby dało się odtworzyć środowisko:

```bash
uv pip freeze > wymagania-zamrozone.txt
```

## 3. Model

```bash
uv pip install -U "huggingface_hub[cli]"
hf download openbmb/VoxCPM2
```

Skrypt przyjmuje `--model openbmb/VoxCPM2` (domyślnie; bierze z pamięci podręcznej HF) albo ścieżkę do katalogu.
Katalog musi zawierać `config.json`, `model.safetensors` i `audiovae.pth`.

## 4. Wybór kart

Numery są takie jak w `nvidia-smi` (skrypt ustawia `CUDA_DEVICE_ORDER=PCI_BUS_ID`). Pominiętych kart procesy
w ogóle nie widzą (`CUDA_VISIBLE_DEVICES`), więc nic na nich nie zostanie zaalokowane.

```bash
python skrypty/generuj.py --lista-kart
```

| flaga | efekt |
|---|---|
| `--pomin 0,2` | wszystkie karty oprócz 0 i 2 |
| `--pomin 0` | wszystkie oprócz 0 |
| `--karty 1,3,4,5,6,7` | tylko wymienione |
| `--karty wszystkie` | wszystkie, bez pytania |
| (brak flag) | skrypt pokazuje listę kart z zajętością pamięci i pyta, które pominąć; Enter = żadnej |

W skryptach i pod `nohup`/`tmux` bez terminala trzeba podać `--karty` albo `--pomin` — skrypt nie zgaduje.
Karty z zajętą pamięcią (> 1 GB) są oznaczone na liście; użycie takiej karty kończy się ostrzeżeniem, nie odmową.

## 5. Próba na jednej karcie

```bash
python skrypty/generuj.py --karty 1 --wyjscie wyniki/proba \
  --tekst "Dzień dobry, to jest pierwsza próba syntezy." \
  --tekst "Ile wynosi stawka VAT na książki w 2026 roku?"
```

Pierwsze uruchomienie trwa dłużej (ładowanie modelu, przechwytywanie grafów CUDA). Wynik:
`wyniki/proba/audio/*.flac` (48 kHz mono) i `wyniki/proba/manifest.jsonl`. Odsłuchaj oba klipy i porównaj
barwę z `glosy/poziomka.wav`.

Głos (domyślnie `glosy/poziomka.wav` jako wzorzec barwy):

| flaga | znaczenie |
|---|---|
| `--tryb referencja` | wzorzec barwy jako osobny blok (tak działa Pytajnik); domyślne |
| `--tryb prompt` | kontynuacja nagrania z transkrypcją; przejmuje też tempo i akustykę. Transkrypcja z pliku `.txt` obok nagrania albo `--glos-tekst` |
| `--tryb oba` | jedno i drugie |
| `--glos glosy/poziomka_dluga.wav` | dłuższa referencja |
| `--glos brak` | bez wzorca: losowy mówca w każdym klipie |

Porównanie trybów i referencji (z `PLAN.md`) — te same teksty do osobnych katalogów:

```bash
for tryb in referencja prompt oba; do
  python skrypty/generuj.py --karty 1 --teksty teksty/proba.txt --limit 50 \
    --tryb $tryb --wyjscie wyniki/tryb_$tryb
done
```

## 6. Strojenie przepustowości (jedna karta)

Plik z kilkuset tekstami o docelowym rozkładzie długości (`.txt`, jeden na wiersz, albo `.jsonl` z polami
`id`, `tekst`). Skrypt na końcu wypisuje „N× czas rzeczywisty” łącznie i na kartę.

```bash
for n in 8 16 32 48; do
  python skrypty/generuj.py --karty 1 --teksty teksty/proba.txt --limit 400 \
    --rownolegle $n --wyjscie wyniki/strojenie_$n
done
```

W drugim terminalu: `watch -n1 nvidia-smi` (pamięć i obciążenie).

- `--rownolegle` — równoległe żądania **na kartę** (domyślnie 16). Punkt odniesienia autora na RTX 4090:
  ok. 100–127× przy 16–32. Na 3090 Ti spodziewane ok. 40–70× na kartę; wziąć wartość, przy której wynik
  przestaje rosnąć.
- `--pamiec` — ułamek VRAM dla silnika (domyślnie 0,8). Przy braku pamięci zmniejszyć do 0,7 albo obniżyć
  `--rownolegle`. Graf CUDA dyfuzji bierze dodatkowo ok. 2,5 GB.
- `--cfg` (domyślnie 2.0; plan: 2.0–2.5), `--kroki` (domyślnie 10).

## 7. Przebieg na wielu kartach

```bash
tmux new -s synteza
source .venv/bin/activate

python skrypty/generuj.py --pomin 0,2 \
  --teksty teksty/etap1.jsonl --wyjscie wyniki/etap1 \
  --rownolegle 16 2>&1 | tee -a wyniki/etap1.log
```

- Jeden proces sterujący; silnik uruchamia po jednym procesie na każdą wybraną kartę, a każde żądanie trafia
  do karty z najmniejszą liczbą zadań w toku. Nie trzeba dzielić tekstów na części.
- **Wznawianie:** to samo polecenie uruchomione ponownie pomija `id` obecne już w `manifest.jsonl`.
  Można też wznowić na innym zestawie kart (np. gdy jedna padnie: dopisać ją do `--pomin`).
- Nieudane teksty trafiają do `wyniki/etap1/bledy.jsonl` i nie zatrzymują przebiegu; po ponownym uruchomieniu
  są próbowane od nowa.
- Jeśli proces którejś karty zginie (np. brak pamięci), żądania do niej kończą się błędami — przerwać
  (Ctrl+C), poprawić `--rownolegle`/`--pamiec` albo pominąć kartę i wznowić.
- Ponowne próby dla złych klipów (do 3 na tekst, `PLAN.md`) należą do filtra jakości, nie do tego skryptu:
  odrzucone `id` usunąć z manifestu i uruchomić jeszcze raz.

## 8. Kontrola wyników

```bash
wc -l wyniki/etap1/manifest.jsonl
python - <<'EOF'
import json
w = [json.loads(l) for l in open("wyniki/etap1/manifest.jsonl", encoding="utf-8")]
s = [x["sekundy"] for x in w]
print(f"{len(w)} klipów, {sum(s)/3600:.2f} h, średnio {sum(s)/len(s):.1f} s, najdłuższy {max(s):.1f} s")
print("podejrzanie długie (>30 s):", sum(x > 30 for x in s))
EOF
```

Klipy dużo dłuższe, niż wynikałoby z tekstu, to zwykle generacja, która się nie zatrzymała (limit
`--max-krokow`, domyślnie 2000) — wyłapie je filtr stosunku długości z `PLAN.md`.

## 9. Problemy

| objaw | przyczyna i wyjście |
|---|---|
| `ModuleNotFoundError: flash_attn` | instalacja z `wymagania.txt` nie przeszła do końca; powtórzyć krok 2.1 |
| `undefined symbol` przy imporcie `flash_attn` | koło nie pasuje do zainstalowanego torch (wersja albo ABI); sprawdzić krok 2.2 i adres koła w `wymagania.txt` |
| uv zaczyna kompilować `flash-attn` | bierze źródła z PyPI zamiast koła; instalować przez `-r wymagania.txt`, nie `uv pip install flash-attn` |
| `ValueError: Missing parameters: ['base_lm.embed_tokens.weight', ...]` | katalog modelu ma wagi `.pt` zamiast `.safetensors`; pobrać `openbmb/VoxCPM2` jeszcze raz |
| `CUDA out of memory` przy starcie albo pod obciążeniem | karta zajęta przez coś innego (pominąć ją) albo za wysokie `--pamiec` / `--rownolegle` |
| numery kart nie zgadzają się z `nvidia-smi` | w powłoce jest ustawione własne `CUDA_VISIBLE_DEVICES`; skrypt je nadpisuje, ale `nvidia-smi` zawsze pokazuje numerację fizyczną — porównywać z `--lista-kart` |
| skrypt kończy się „Brak terminala do zadania pytania” | uruchomienie bez terminala; dodać `--karty` albo `--pomin` |
| wolno mimo wolnych kart | za mało rdzeni CPU (jeden proces na kartę kręci się w pętli) albo zapis na wolny dysk |
