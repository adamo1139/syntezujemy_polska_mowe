# Runbook: nano-vllm-voxcpm i synteza VoxCPM2 na 8× RTX 3090 Ti

Instalacja silnika `nano-vllm-voxcpm`, próba na jednej karcie, strojenie i przebieg na wielu kartach
z możliwością pominięcia dowolnych (np. 0 albo 2). Kontekst i decyzje: `PLAN.md`.

**Stan (2026-11-20):** spisane na podstawie kodu `nano-vllm-voxcpm` 2.0.5 (PyPI, 2026-10-12; repo
`a710128/nanovllm-voxcpm`, commit `b839543`). Na rigu (8× 3090 Ti) przeszły kroki 2–5: instalacja z
`wymagania.txt`, model w `HF_HOME` i próba dwóch zdań na karcie 1 (start 37 s, 48 kHz, bez błędów).
Pełny przebieg (kroki 6–8) też: 151 364 klipy, 367,7 h audio w 6005 s na 6 kartach z `--rownolegle 32`,
bez błędów. Średnio 220× czas rzeczywisty; na starcie 295×, potem spadek, bo dysk z wynikami przegrzał się
do 79 °C i dławił zapis (patrz Problemy). Jedna karta ze strojenia: 32× / 43× / 52× przy 8 / 16 / 32.
Głos mówi ok. 18,7 znaku/s, nie 14 jak w `PLAN.md`, więc „500 h” tekstu dało 367,7 h audio.

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
```

### 2.0 Pamięć podręczna uv i model poza dyskiem głównym (rig)

Na rigu dysk główny nie ma miejsca ani na pamięć podręczną uv (same koła torch i flash-attn to kilka GB),
ani na model (ok. 5 GB). Dla tego projektu oba trafiają na `/media/nvme_2tb/maked/audio`:

| co | gdzie | jak ustawione |
|---|---|---|
| pamięć podręczna uv | `/media/nvme_2tb/maked/audio/uv-cache` | `uv.toml` w katalogu projektu |
| model i reszta pamięci HF | `/media/nvme_2tb/maked/audio/hf` | `HF_HOME` dopisane do `.venv/bin/activate` |

Oba ustawienia są lokalne dla riga i nie ma ich w repo.

```bash
mkdir -p /media/nvme_2tb/maked/audio/uv-cache /media/nvme_2tb/maked/audio/hf
echo 'cache-dir = "/media/nvme_2tb/maked/audio/uv-cache"' > uv.toml
uv cache dir    # ma wypisać /media/nvme_2tb/maked/audio/uv-cache

uv venv --python 3.12 .venv
echo 'export HF_HOME=/media/nvme_2tb/maked/audio/hf' >> .venv/bin/activate
source .venv/bin/activate
echo $HF_HOME   # ma wypisać /media/nvme_2tb/maked/audio/hf
```

- `uv.toml` działa tylko dla poleceń `uv` uruchamianych z katalogu projektu — z innego miejsca uv wróci
  do `~/.cache/uv`.
- `HF_HOME` jest ustawiane przy każdym `source .venv/bin/activate`, więc pobieranie modelu (krok 3) i synteza
  korzystają z tego samego miejsca. Bez aktywacji środowiska (np. wywołanie `.venv/bin/python` wprost) zmienna
  nie jest ustawiona i model pobrałby się drugi raz do `~/.cache/huggingface`. Skrypt `generuj.py` wypisuje
  na starcie, skąd bierze model.
- Po usunięciu i odtworzeniu `.venv` wiersz z `HF_HOME` trzeba dopisać ponownie.
- Na dysku, na którym leży projekt, zostaje samo `.venv` (kilka GB). Przy pamięci podręcznej na innym dysku
  uv kopiuje pliki zamiast je linkować i wypisuje o tym ostrzeżenie — to nie błąd. Jeśli i na `.venv` brakuje
  miejsca, sklonować projekt na `/media/nvme_2tb/maked/audio`.

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
echo $HF_HOME                 # na rigu: /media/nvme_2tb/maked/audio/hf (krok 2.0)
hf download openbmb/VoxCPM2
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--openbmb--VoxCPM2"   # ok. 5 GB
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

Przy starcie pojawiają się dwa ostrzeżenia, które można pominąć: `FutureWarning` o `weight_norm` oraz
`torch._dynamo hit config.recompile_limit` dla funkcji `snake` w AudioVAE. Wynik „N× czas rzeczywisty”
z dwóch zdań nic nie mówi o przepustowości (na rigu wyszło 5×) — to mierzy dopiero krok 7.

Głos (domyślnie `glosy/poziomka.wav` jako wzorzec barwy):

| flaga | znaczenie |
|---|---|
| `--tryb referencja` | wzorzec barwy jako osobny blok (tak działa Pytajnik); domyślne |
| `--tryb prompt` | kontynuacja nagrania z transkrypcją; przejmuje też tempo i akustykę. Transkrypcja z pliku `.txt` obok nagrania albo `--glos-tekst` |
| `--tryb oba` | jedno i drugie |
| `--glos glosy/poziomka_dluga.wav` | dłuższa referencja |
| `--glos brak` | bez wzorca: losowy mówca w każdym klipie |

Porównanie trybów i referencji (z `PLAN.md`) — te same teksty (50 pierwszych
z części 9, której na razie nie syntezujemy) do osobnych katalogów:

```bash
for tryb in referencja prompt oba; do
  python skrypty/generuj.py --karty 1 --teksty teksty/wybrane.jsonl --czesc 9 --limit 50 \
    --tryb $tryb --wyjscie wyniki/tryb_$tryb
done
```

## 6. Teksty do syntezy

Źródło: `cpral/polskie-sprawy-v8-jakosc50` (`sft.jsonl`, 2,2 GB, 539 252 rozmowy). Dwa skrypty, oba
deterministyczne (to samo `--ziarno` daje ten sam plik), bez GPU, tylko biblioteka standardowa i `huggingface_hub`.

```bash
python skrypty/przygotuj_teksty.py     # pobiera dataset do HF_HOME; ok. 1 min pracy -> teksty/fragmenty.jsonl
python skrypty/wybierz_teksty.py       # 500 h w 10 częściach po ok. 50 h -> teksty/wybrane.jsonl
```

`przygotuj_teksty.py` (lokalna kopia zamiast pobierania: `--zrodlo /sciezka/sft.jsonl`):

1. odrzuca rozmowy, w których `meta.model` albo którykolwiek z `meta.turn_models` to `private`
   (33 158 rozmów; licencja CC-BY-NC, reszta jest na Apache-2.0);
2. z pozostałych 506 094 zostawia górne 10% według `jakosc.ocena` — próg 70,9, czyli 50 835 rozmów
   (`--percentyl 90`);
3. czyści markdown: usuwa nagłówki, tabele, bloki kodu, znaczniki list, pogrubienia, etykiety punktów
   („**Wsparcie**: …” → sama treść), emoji; odrzuca zdania z adresami, wzorami, szablonami w nawiasach
   kwadratowych i znakami, których nie da się przeczytać;
4. tnie odpowiedzi na fragmenty z całych sąsiednich zdań, nigdy w środku zdania. Każdy fragment losuje
   długość docelową między 3 s a 60 s (logarytmicznie równomiernie, więc krótkich jest więcej) i zbiera
   zdania, aż ją osiągnie; limit to 60 s, czyli 840 znaków przy 14 znakach/s (`--max-sekund`,
   `--cel-od-sekund`). Pytania użytkownika bierze tylko w całości;
5. usuwa powtórki i zapisuje `teksty/fragmenty.jsonl`: 520 488 fragmentów, ok. 2010 h mowy (poza repo).

Rozkład długości w `fragmenty.jsonl`: mediana 10,6 s, p90 27,9 s; 68% klipów ma do 15 s, ale 60% czasu
przypada na klipy dłuższe niż 15 s (25% czasu na dłuższe niż 30 s). Długie fragmenty powstają m.in. ze
sklejonych punktów list (bez etykiet), więc bywają wyliczeniem kilku rad pod rząd.

Liczby i skróty zostają surowe, zgodnie z `PLAN.md`.

`wybierz_teksty.py` bierze z tego 500 h (`--godziny`) i dzieli na 10 części (`--czesci`):

- tematy na zmianę, bo w górnych 10% jakości dwa tematy (rodzina, praca) to 70% fragmentów. Przy 500 h
  26 z 31 tematów wchodzi w całości, a trzy największe mają po ok. 12,5% czasu;
- przy jednych odwiedzinach rozmowy 1 pytanie użytkownika i 2 fragmenty odpowiedzi (`--pytan`, `--odpowiedzi`);
  gdy tematowi skończą się nowe rozmowy, wraca do tych samych po kolejne fragmenty;
- każda rozmowa trafia w całości do jednej części (pole `czesc`, 0–9), a każda część ma ten sam rozkład
  tematów. Części są rozłączne, więc kolejne można syntezować później, np. innymi głosami.

Wynik: 151 364 fragmenty z 30 613 rozmów (poza repo), 35% to pytania użytkownika. Klipy: 17% do 5 s,
57% 5–15 s, 21% 15–30 s, 5% ponad 30 s; w czasie odpowiednio 5%, 43%, 36% i 16%. Części mają 49,4–50,4 h.
Czas jest szacunkiem z liczby znaków — faktyczny wyjdzie po syntezie.

**Teraz syntezujemy tylko część 0** (ok. 50 h, 15 181 fragmentów) głosem „poziomka”: `generuj.py --czesc 0`.
Katalog `teksty/` nie jest w repo; na każdej maszynie oba skrypty dają identyczne pliki.

Pola wiersza: `id`, `tekst`, `zrodlo_id` (id rozmowy, do podziału na walidację i test), `rola`, `tura`,
`zdania`, `znaki`, `pytanie`, `temat`, `jakosc`, `czesc`. `generuj.py` przepisuje je wszystkie do manifestu.

## 7. Strojenie przepustowości (jedna karta)

Zestaw próbny: 500 pierwszych fragmentów części 9 (ok. 1,5 h mowy, z klipami do 60 s; część 0 zostaje nietknięta na właściwy przebieg).
Skrypt na końcu wypisuje „N× czas rzeczywisty” łącznie i na kartę.

```bash
for n in 8 16 32 48; do
  python skrypty/generuj.py --karty 1 --teksty teksty/wybrane.jsonl --czesc 9 --limit 500 \
    --rownolegle $n --wyjscie wyniki/strojenie_$n
done
```

W drugim terminalu: `watch -n1 nvidia-smi` (pamięć i obciążenie).

- `--rownolegle` — równoległe żądania **na kartę** (domyślnie 16). Punkt odniesienia autora na RTX 4090:
  ok. 100–127× przy 16–32. Na 3090 Ti spodziewane ok. 40–70× na kartę; wziąć wartość, przy której wynik
  przestaje rosnąć.
- `--pamiec` — ułamek VRAM dla silnika (domyślnie 0,8). Przy braku pamięci zmniejszyć do 0,7 albo obniżyć
  `--rownolegle`. Graf CUDA dyfuzji bierze dodatkowo ok. 2,5 GB.
- Klipy do 60 s to dłuższe sekwencje niż w benchmarku autora, więc pamięć na żądanie jest większa —
  zacząć strojenie od niższych wartości `--rownolegle` i patrzeć na VRAM.
- `--max-krokow` (domyślnie 700) ucina generację, która się nie zatrzymała: 1 krok to 0,16 s audio
  (25 Hz latentów, po 4 na krok), więc 700 kroków = 112 s; klip 60 s to ok. 375 kroków.
- `--cfg` (domyślnie 2.0; plan: 2.0–2.5), `--kroki` (domyślnie 10).

## 8. Przebieg na wielu kartach

```bash
tmux new -s synteza
source .venv/bin/activate

python skrypty/generuj.py --pomin 0,2 \
  --teksty teksty/wybrane.jsonl --czesc 0 --wyjscie wyniki/etap1 \
  --rownolegle 16 2>&1 | tee -a wyniki/etap1.log
```

- Jeden proces sterujący; silnik uruchamia po jednym procesie na każdą wybraną kartę, a każde żądanie trafia
  do karty z najmniejszą liczbą zadań w toku. Nie trzeba dzielić tekstów na części.
- **Wznawianie:** to samo polecenie uruchomione ponownie pomija `id` obecne już w `manifest.jsonl`.
  Można też wznowić na innym zestawie kart (np. gdy jedna padnie: dopisać ją do `--pomin`).
- Nieudane teksty trafiają do `wyniki/etap1/bledy.jsonl` i nie zatrzymują przebiegu; po ponownym uruchomieniu
  są próbowane od nowa.
- Jeśli proces którejś karty zginie (np. brak pamięci), skrypt sam przerywa przebieg po 50 błędach z rzędu
  (`--max-bledow-z-rzedu`) — poprawić `--rownolegle`/`--pamiec` albo pominąć kartę i wznowić.
- Wznowienie wymaga tych samych `--teksty`, `--czesc` i `--wyjscie`; karty, `--rownolegle` i `--pamiec` można
  zmienić. Zmiana `--glos`, `--tryb` albo `--cfg` w połowie da zbiór mieszany (manifest zapisuje je przy każdym
  klipie, więc da się to wykryć).
- Po twardym przerwaniu (kill -9, zanik zasilania) urwany ostatni wiersz manifestu jest obcinany przy starcie.
- Nie uruchamiać dwóch przebiegów naraz na tym samym `--wyjscie` — nie ma blokady, klipy by się dublowały.
- Ponowne próby dla złych klipów (do 3 na tekst, `PLAN.md`) należą do filtra jakości, nie do tego skryptu:
  odrzucone `id` usunąć z manifestu i uruchomić jeszcze raz.

## 9. Kontrola wyników

```bash
wc -l wyniki/etap1/manifest.jsonl
python - <<'EOF'
import json
w = [json.loads(l) for l in open("wyniki/etap1/manifest.jsonl", encoding="utf-8")]
s = [x["sekundy"] for x in w]
print(f"{len(w)} klipów, {sum(s)/3600:.2f} h, średnio {sum(s)/len(s):.1f} s, najdłuższy {max(s):.1f} s")
print("podejrzanie długie (>90 s):", sum(x > 90 for x in s))
EOF
```

Klipy dużo dłuższe, niż wynikałoby z tekstu, to zwykle generacja, która się nie zatrzymała (limit
`--max-krokow`, domyślnie 700 = 112 s) — wyłapie je filtr stosunku długości z `PLAN.md`.

## 10. Wysyłka na HF

```bash
uv pip install pyarrow
hf auth login                         # token z prawem zapisu
python skrypty/wyslij_na_hf.py --repo UZYTKOWNIK/NAZWA
```

- Pakuje `wyniki/poziomka` (`--wyniki`) do plików parquet po ok. 500 MB w `wyniki/poziomka_parquet/data/`
  i dopisuje kartę `README.md`. Kolumny: `audio` (FLAC w środku, typ Audio z `datasets`), `text`, oraz
  `id`, `sekundy`, `glos`, `zrodlo_id`, `rola`, `tura`, `temat`, `czesc`, `jakosc`, `pytanie`.
- Kopia w parquet zajmuje tyle samo co audio (367,7 h to ok. 61 GB); skrypt sprawdza wolne miejsce przed
  startem. Inny dysk: `--paczki KATALOG`.
- Repo powstaje jako **prywatne**; publiczne dopiero z `--publiczne` (albo przełączone później na stronie).
- Oba etapy da się wznowić tym samym poleceniem. Osobno: `--tylko-pakuj`, `--tylko-wyslij`.
- Karta mówi wprost, że klipy nie przeszły jeszcze filtra jakości.

## 11. Problemy

| objaw | przyczyna i wyjście |
|---|---|
| `ModuleNotFoundError: flash_attn` | instalacja z `wymagania.txt` nie przeszła do końca; powtórzyć krok 2.1 |
| `undefined symbol` przy imporcie `flash_attn` | koło nie pasuje do zainstalowanego torch (wersja albo ABI); sprawdzić krok 2.2 i adres koła w `wymagania.txt` |
| uv zaczyna kompilować `flash-attn` | bierze źródła z PyPI zamiast koła; instalować przez `-r wymagania.txt`, nie `uv pip install flash-attn` |
| `ValueError: Missing parameters: ['base_lm.embed_tokens.weight', ...]` | katalog modelu ma wagi `.pt` zamiast `.safetensors`; pobrać `openbmb/VoxCPM2` jeszcze raz |
| `CUDA out of memory` przy starcie albo pod obciążeniem | karta zajęta przez coś innego (pominąć ją) albo za wysokie `--pamiec` / `--rownolegle` |
| numery kart nie zgadzają się z `nvidia-smi` | w powłoce jest ustawione własne `CUDA_VISIBLE_DEVICES`; skrypt je nadpisuje, ale `nvidia-smi` zawsze pokazuje numerację fizyczną — porównywać z `--lista-kart` |
| skrypt kończy się „Brak terminala do zadania pytania” | uruchomienie bez terminala; dodać `--karty` albo `--pomin` |
| tempo spada w trakcie, użycie kart faluje, wysokie `wa` w `top` | dysk z wynikami się dławi: sprawdzić temperaturę (`sensors`, `sudo nvme smart-log /dev/nvme0n1`) i nawiew; przebiegu nie trzeba przerywać |
| wolno mimo wolnych kart | za mało rdzeni CPU (jeden proces na kartę kręci się w pętli) albo zapis na wolny dysk |
