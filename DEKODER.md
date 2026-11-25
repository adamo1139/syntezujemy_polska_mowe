# Poprawianie dekodera kodeka 48 kHz — podsumowanie (2026-11-23/24)

Cel: lepsze brzmienie syntezy bez zmiany latentów. Enkoder kodeka (25,6M) przez cały czas był zamrożony, więc
latenty, statystyki i model tekst→latenty pozostały ważne; zmieniał się tylko dekoder. Szczegółowy dziennik z
poleceniami: `TRENING.md`, sekcja 7.

## Dlaczego w ogóle szukaliśmy lepszego dekodera

Problemy, które było słychać w syntezie ze starym dekoderem (`kodek_48k/ae_wsm_210_220`):

1. **„Cierpkość”** — szorstkie, nieczyste brzmienie głosu, którego nie ma w nagraniach VoxCPM2. Główny powód całej
   tej pracy.
2. **Szum w górnym paśmie** — dekoder dokładał ok. 9,5 dB szumu powyżej 12 kHz, gdzie oryginały prawie nic nie mają.
3. **Ślad sklejania klatek** — prążki co 93,75 Hz (dekoder składa dźwięk z kawałków po 512 próbek), ponad dwa razy
   wyższe niż w oryginale.
4. **Brak do oryginału w ocenie**: UTMOS samej rekonstrukcji (bez modelu tekst→latenty) 2,739 wobec 2,996
   oryginałów — dekoder psuł nagranie, zanim cokolwiek zrobił model TTS.
5. **Obejście było drogie**: dobre brzmienie dawał dopiero łańcuch Sidon v0.1 + korektor, czyli dodatkowy duży
   model (w2v-BERT 2.0 + wokoder) przy każdej syntezie, ok. 0,8 s na 10 s mowy na RTX 5080 — nie do pogodzenia
   z celem małego TTS na CPU i telefonie.
6. Później, po odmrożeniu dekodera: **„schodki szumu”** wokół słów (modulacja szumu w rytmie klatek), opisane niżej.

### Diagnoza cierpkości

Synteza modelu tekst→latenty (punkt 115 tys.) miała **cierpkość**, której nie ma w nagraniach VoxCPM2.
Co ustaliliśmy (`TRENING.md`, sekcja 7):

- **Winny jest (w dużej części) dekoder kodeka**, nie model tekst→latenty: sama rekonstrukcja kodekiem
  (oryginał → enkoder → dekoder) brzmiała gorzej od oryginału. UTMOS na 4 klipach: oryginał 3,02, rekonstrukcja
  starym dekoderem 2,64, synteza 2,88.
- Ustawienia syntezy (CFG 1–4, 4–100 kroków, wagi bieżące/EMA) zmieniały barwę o ułamki dB — to nie one.
- Dekoder dokładał ok. 9,5 dB szumu powyżej 12 kHz (oryginały prawie nie mają treści powyżej 14–16 kHz), ale ręczne
  wyciszenie góry (−10 dB, obcięcie, korektor) **nie usuwało cierpkości na ucho** — źródło było niżej w paśmie.
- **Ślad sklejania klatek**: dekoder składa dźwięk z kawałków po 512 próbek, co daje prążki na k·93,75 Hz
  (miara prążków: oryginał 1,72, rekonstrukcja 3,55, synteza 4,6 ponad tło) — szorstkie, „cyfrowe” zabarwienie
  harmonicznych.

Obejście na czas poprawiania dekodera: łańcuch **Sidon v0.1 + korektor** (odtworzenie mowy od nowa z cech w2v-BERT
2.0 i stała krzywa EQ z powrotem do barwy VoxCPM2) — najlepszy na ucho; UTMOS 3,24 wobec 2,97 oryginału. Wymaga
jednak dodatkowego modelu przy każdej syntezie, więc celem stało się usunięcie przyczyny w samym dekoderze.

Nieudane próby, zanim zadziałało to, co opisane niżej:
- **Ulepszacze z zewnątrz**: AudioSR (zmienia tylko pasmo > 12 kHz), MossFormer2 SE/SR 48K (trzaski), VoiceFixer,
  Resemble Enhance (44,1 kHz), DialogueSidon2 (24 kHz, CC BY-NC), mieszanie Sidona z wejściem (filtr grzebieniowy).
- **Dotrenowanie dekodera stratą pasma wysokiego** (`--hiband_w`): LogMel_Full rósł (0,607 → 0,739 przy tempie
  8,5e-5 bez rozgrzewki; przy 2e-5 z rozgrzewką też rósł) — porzucone.
- **Głowica w stylu Vocos** (`--decoder_head istft`): usuwała prążki (1,3–1,5, poniżej oryginału), ale UTMOS stanął
  na 2,445 (val; stary dekoder 2,739) — przewidywanie fazy wymagałoby kodeka trenowanego od zera pod tę głowicę.
  Po drodze błąd: głośny ton w prążku Nyquista (24 kHz), którego nie widzi żadna strata melowa; poprawka: prążki
  0 Hz i 24 kHz zawsze zero. Lekcja: **miary melowe nie widzą fazy ani tonu Nyquista** — od tego czasu UTMOS
  w ocenie (`--heldout_utmos`, `ocen_kodek_val.py --utmos`).

## Wynik

Opublikowany dekoder: `modele/kodek_48k_postnet12_slm_dlugie/ae_wsm_26_29.pt` (średnia wag z kroków 26–29 tys.),
37,8M parametrów (ciało + głowica 25,3M, moduł poprawki widma 12,5M). Jest w `model.safetensors` modelu
[cpral/MikolAI-109M-v0.1-Polish-TTS-Preview](https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview).

Pełna walidacja (`ocen_kodek_val.py --utmos`, rekonstrukcja: oryginał → enkoder → dekoder):

| dekoder | UTMOS val | brak do oryginału | UTMOS val_liczby | brak do oryginału | log-mel | LSD 16–24 kHz |
|---|---|---|---|---|---|---|
| stary (`kodek_48k/ae_wsm_210_220`, 25,3M) | 2,739 | 0,257 | 2,624 | 0,234 | 0,586 | 24,82 |
| moduł poprawki 4 bloki (`ae_wsm_6_9`) | 2,755 | 0,241 | 2,643 | 0,215 | 0,381 | 12,04 |
| moduł poprawki 12 bloków, zamrożone ciało (`ae_wsm_17_20`) | 2,759 | 0,237 | 2,645 | 0,213 | 0,378 | 11,89 |
| **odmrożony + SLM + segment 2,8 s + modulacja (`ae_wsm_26_29`)** | **2,880** | **0,119** | **2,763** | **0,095** | **0,367** | 11,95 |

Oryginały VoxCPM2: val 2,999, val_liczby 2,858 (lista walidacyjna po filtrze ASR; wcześniejsze wiersze liczone na
liście sprzed czyszczenia, oryginał wtedy 2,996 — porównywać brak do oryginału). **Brak do oryginału spadł o połowę.**

W syntezie (model tekst→latenty EMA 225 tys., CFG 4, 16 kroków, 24 zdania próbne, to samo ziarno):

| dekoder | UTMOS syntezy | prążki | schodki szumu (śr. dB przy k·93,75 Hz) |
|---|---|---|---|
| stary | 2,983 | 0,950 | 9,2 |
| moduł 4 bloki | 3,001 | 0,797 | 8,2 |
| odmrożony, krok 16 tys. (bez straty modulacji) | 3,108 | 0,400 | 7,9 |
| **WSM 26–29 tys.** | **3,149** | **0,361** | **1,5** |

(W rekonstrukcji 7 nagrań: oryginał UTMOS 2,808, prążki 0,349, schodki 3,9 dB; WSM 26–29: 2,713, 0,466, 2,4 dB.)

## Kolejne kroki

1. **Moduł poprawki widma STFT** (`--decoder_postnet`): kilka bloków ConvNeXt na widmie wyjścia dekodera
   przewiduje zespoloną maskę |M| ≤ 4, ostatnia warstwa od zer (na starcie dekoder bez zmian). 4 bloki, potem
   12 (`--postnet_layers 12`), ciało zamrożone (`--head_warmup` większe niż liczba kroków). Wyraźnie mniej szumu
   w górnym paśmie (LSD 16–24 kHz z 24,8 do 11,9) i prążków sklejania, ale **UTMOS stanął na ~2,76** niezależnie
   od liczby bloków — ograniczenie siedziało w zamrożonej części.
2. **Dyskryminator SLM** (`--slm_w 1 --slm_fm_w 5`, jak w StyleTTS2): zamrożony WavLM-Base+ (94,4M, MIT, tylko
   w treningu) i głowica 0,8M na wszystkich warstwach; strata przeciwstawna + dopasowanie cech, na całym segmencie
   (MPD/MRD widzą tylko wycinek). Przy inferencji 0 parametrów, 0 FLOP. Trening ok. 1,6× wolniejszy.
   Sam moduł poprawki + SLM: UTMOS odłożonych klipów 2,790 (oryginały 3,020).
3. **Dłuższe segmenty i wycinek dla dyskryminatorów**: `--segment_sec 2.8` (zamiast 1,4 s; porcja 24 zamiast 64,
   ta sama ilość audio), `--d_crop 1.0` (wycinek dla MPD/MRD 1 s zamiast 0,19 s z przepisu blue-codec),
   `--slm_partia 8`. OOM po pierwszej ocenie → `torch.cuda.empty_cache()` po ocenie + `expandable_segments`.
4. **Odmrożony cały dekoder** (od kroku 6000, rampa tempa ciała): **pierwszy wyraźny skok** — UTMOS odłożonych
   klipów 2,847 po 4000 kroków (brak do oryginału 0,17 zamiast 0,23).
5. **„Schodki szumu”** (słyszalne na odsłuchu kroku 16 tys.): pomiar pokazał modulację obwiedni szumu 4–16 kHz
   w rytmie klatek głowicy (93,75 Hz = co 512 próbek) — 9–13 dB ponad tło w rekonstrukcji, 8–9 dB w syntezie, przy
   ok. 3,4 dB w oryginale; odmrożony dekoder miał jej **więcej** (strata prążków patrzyła tylko na widmo do
   3,75 kHz). Szum tła sam w sobie (ok. −56 dB w przerwach) pochodzi z danych VoxCPM2.
6. **Strata modulacji obwiedni** (`--modulacja_w 5`, od kroku 19 tys.): log-obwiednia energii 4–16 kHz (ramki co
   32 próbki), w jej widmie wysokość prążków przy k·93,75 Hz (k ≤ 6) nad lokalnym tłem; karany tylko **nadmiar**
   ponad oryginał. `Heldout_Modulacja` w ocenie. Wynik: modulacja 0,145 → 0,10, prążki 0,197 → 0,177, UTMOS
   odłożonych klipów 2,86–2,895 (bez spadku); schodki w syntezie z ~8 dB do ~1,5 dB (poziom tła).
7. **WSM** kroków 26–29 tys. (`usrednij_kodek.py`) i pełna ocena — tabela wyżej.

Ostatni przebieg (`kodek_48k_postnet12_slm_dlugie`, GPU 4 riga, wznowiony z kroku 19 tys.):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 train_autoencoder.py \
  --arch_config ../dane/polski_100m.json --checkpoint_dir ../modele/kodek_48k_postnet12_slm_dlugie \
  --resume ../modele/kodek_48k_postnet12_slm_dlugie/ae_19000.pt \
  --decoder_only --decoder_postnet --postnet_layers 12 --head_warmup 1 --head_lr_scale 5 \
  --linie_w 5 --modulacja_w 5 --slm_w 1 --slm_fm_w 5 --slm_partia 8 \
  --segment_sec 2.8 --batch_size 24 --d_crop 1.0 --constant_lr --lr 2e-5 \
  --total_steps 29000 --save_every 1000 --heldout_every 1000 --heldout_utmos --eval_input ../glosy/poziomka_dluga.wav
```

## Czy cierpkość zniknęła

Na odsłuchu (autor): dekoder z kroku 16 tys. „zdecydowanie lepiej od starszego”, ale słychać było schodki szumu
wokół tekstu; po stracie modulacji i WSM 26–29 tys. — dobrze. Liczbowo: prążki sklejania w syntezie spadły
z 0,950 (stary) do 0,361, schodki z ~9 dB do ~1,5 dB, UTMOS syntezy z 2,983 do 3,149, a brak UTMOS rekonstrukcji
do oryginału z 0,257 do 0,119. Zostaje szum tła z danych VoxCPM2 (odszumienie danych — na później) i resztka
różnicy do oryginału (UTMOS rekonstrukcji 2,880 wobec 2,999). Nie sprawdzaliśmy jeszcze, czy łańcuch z Sidonem
dalej coś daje na nowym dekoderze. Gdyby szorstkość harmonicznych wróciła, kolejny kandydat: źródło harmoniczne
NSF (jak HiFT w CosyVoice) albo aktywacja okresowa (Snake) w dekoderze v2.

## Co się nie sprawdziło / pomyłki

- Sam moduł poprawki przy zamrożonym ciele: więcej bloków (4 → 12) nie podniosło UTMOS.
- Odmrożony dekoder bez straty modulacji przerzucił ślad sklejania klatek z prążków w widmie na modulację szumu
  (schodki) — lekcja: strata działa tylko tam, gdzie patrzy.
- Pomyłka w pomiarze (poprawiona): wzorzec `*oryginal*` łapał też pliki `…_oryginal_rekonstrukcja.wav`, przez co
  „oryginał” wyszedł 2,716 i prążki 0,685 zamiast 2,808 i 0,349; wnioski o „dorównaniu oryginałowi” były błędne.

## Koszt przy inferencji

Dekoder liczy się raz na kawałek przy 93,75 klatki/s, pole wektorowe 32 razy przy 15,6 klatki/s; jeden przebieg
dekodera ≈ 3 przebiegi pola. Na 1 wątku i7-11700K (10 s mowy): stary 0,56–0,88 s, z modułem 12 bloków ok. 1,46 s
(RTF dekodera ok. 0,15). Przy obecnych ustawieniach to ok. 15% czasu syntezy; po destylacji kroków byłby ponad
połowę. INT8 w dekoderze prawie nie przyspiesza CPU i odrobinę obniża UTMOS — w bibliotece zostaje w fp32.

## Przygotowane, nieuruchomione

- **Dekoder v2** (kod w repo, `e2dc4d2`): przednia sieć przy 15,6 Hz (`--decoder_prenet`: przyczynowy transformer,
  7 warstw × 512, SwiGLU 2048, RoPE, okno 64 klatek ≈ 4 s, 29,5M, start od zer — na starcie bit w bit jak stary)
  i aktywacja okresowa (`--decoder_act gelu_snake` do dostrajania, `snakebeta` od zera). Koszt CPU: +0,11 s / 10 s
  mowy za przednią sieć, aktywacja w trybie eager +0,2 s. Plan: od `ae_wsm_26_29`, segment 4,2 s, porcja 16,
  `--d_crop 1.5`, `--slm_partia 5`; najpierw 2000 kroków samych nowych części, potem cały dekoder.
- Pomysły na później: dylatacje dla 2–4× większego zasięgu ciała (0 parametrów, ale wymaga dostrojenia albo
  treningu od zera), SwiGLU zamiast GELU-MLP, odszumienie danych treningowych (szum tła z VoxCPM2), kwantyzacja
  dekodera z QAD.
