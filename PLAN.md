# Syntezujemy polską mowę — plan

Cel: rodzina małych polskich TTS, od ok. 1M do ok. 200M parametrów, wytrenowanych na danych
syntetycznych z VoxCPM2. Najpierw dane, potem drabina architektur od najmniejszej do największej.

Notatki startowe z 2026-11-20. Liczby z pomiarów pochodzą z projektu Pytajnik
(`~/projects/misc/pytajnik/`, testy z 2026-11-10).

## Etap 1: ok. 50 h audio z VoxCPM2

### Tekst: `cpral/polskie-sprawy-v8-jakosc50`

- `sft.jsonl`, 539 252 rozmów, ok. 1,1 GB. Kolumny: `messages` (user/assistant), `meta` (temat, persona,
  model), `ocena`, `jakosc.ocena` (50–80).
- Z próbki 400 rozmów (wiersze 0, 100k, 300k, 500k):
  - **pytania użytkownika**: mediana 94 znaki, czysty tekst (12% z cyframi, 4% ze skrótami). Gotowe krótkie klipy, w tym dużo pytań.
  - **odpowiedzi asystenta**: mediana 538 znaków, p90 1852. 29% ma `**pogrubienia**`, 30% listy, 5% nagłówki,
    42% cyfry, 13% skróty WIELKIMI LITERAMI. Przed syntezą trzeba je oczyścić i pociąć.
  - tematy: od codzienności i finansów po historię, fizykę i inżynierię, więc słownictwo jest szerokie.
- Skala: 50 h = 180 000 s; przy ok. 14 znakach/s to ok. **2,5 mln znaków**, a same odpowiedzi w datasecie
  to ok. 1,3 mld znaków. Wystarczy ułamek procenta, więc można wybierać: najwyższa `jakosc.ocena`,
  różnorodność tematów, deduplikacja.
- Licencja: **Apache-2.0** dla części datasetu, której używamy (decyzja autora, 2026-11-20). VoxCPM2 też jest
  na Apache-2.0, więc dane audio i modele mogą być na Apache-2.0.

### Przygotowanie tekstu

1. Odpowiedzi: usunąć markdown (pogrubienia, nagłówki, wypunktowania jako osobne zdania), pociąć na fragmenty
   z całych zdań: mieszanka długości od ok. 1 s do 60 s (decyzja 2026-11-20; wcześniej 1–3 zdań, 2–15 s).
   Pytania użytkownika brać w całości, jeśli mieszczą się w tym zakresie.
2. **Bez normalizacji dla nauczyciela.** VoxCPM2 sam poprawnie czyta skróty i liczby (VAT, PIT, CIT, „240 tys.”),
   więc dostaje surowy tekst (po usunięciu markdownu). Rozbieżności typu „SIT”/„PID” w testach Pytajnika to
   najpewniej pomyłki STT, nie TTS.
   Normalizacja to problem **ucznia**: duże modele (100–200M) mogą uczyć się czytać surowy tekst tak jak
   nauczyciel, ale małe (1–20M, zwłaszcza na fonemach espeak-ng) prawie na pewno będą potrzebowały
   formy mówionej. Dlatego w manifeście trzymać `tekst` (surowy, wejście VoxCPM2) i dopisywać
   `tekst_mowiony` dopiero gdy będzie potrzebny — najlepiej generowany przez LLM z surowego tekstu
   i sprawdzany z tym, co nauczyciel faktycznie powiedział (to jego odczyt jest wzorcem, np. czy „PIT-36L”
   przeczytał „pit trzydzieści sześć el”).
3. Rozkład długości klipów jak w machinekind v2: dużo 2–10 s, część ≥ 10 s dla prozodii między zdaniami,
   trochę < 2 s (krótkie reakcje). Osobno zadbać o pytania (intonacja) i formy męskie/żeńskie.

### Głosy

- **Głos główny: „poziomka”** (decyzja 2026-11-20), w `glosy/`. To **nie jest wbudowany głos** — VoxCPM2 nie ma
  gotowych głosów; bez referencji losuje nowego mówcę przy każdej generacji. „poziomka” to jeden taki los:
  benchmark z 2026-10-25 (`pytamy/testy/tts_voxcpm2.py`, `reference=None`, 20 zdań), próbka 13 (liczone od 0,
  „Oczywiście, zaraz sprawdzę rozkład autobusów.”), zamrożona jako referencja (commit `98a5fb8` w repo Poziomki).
  Głos istnieje więc tylko jako ten klip — trzymać go w kopii zapasowej; każdy klip datasetu będzie jego klonem.
  - `glosy/poziomka.wav` (2,24 s, 48 kHz; oryginał, ten sam co w Pytajniku) z transkrypcją `glosy/poziomka.txt`;
  - `glosy/poziomka_dluga.wav` (5,28 s, 48 kHz; klon poziomka.wav wygenerowany przez VoxCPM2, więc „klon klonu”
    — może lekko odpływać od oryginału) z `poziomka_dluga.txt`.
    Dłuższa referencja (5–8 s) zwykle lepiej trzyma barwę; którą wziąć, sprawdzić na kilkudziesięciu klipach
    (podobieństwo mówcy, UTMOS).
- Tryby podawania głosu w VoxCPM2 (do porównania na próbach, metryki: podobieństwo do `poziomka.wav`, UTMOS, CER):
  `reference_wav_path` (wzorzec barwy jako osobny blok — tak działa Pytajnik), `prompt_wav_path` + `prompt_text`
  (kontynuacja nagrania z transkrypcją — przejmuje też tempo i akustykę) albo oba naraz; oraz krótka
  vs długa referencja. VoxCPM2 ma też wbudowane `retry_badcase` (do 3 powtórek, gdy długość audio nie pasuje do tekstu).
- Opcjonalnie 2–4 dodatkowe głosy zaprojektowane opisem (voice design VoxCPM2), zamrożone jako stały prompt.
  Dane równoległe (te same teksty, różne głosy) ułatwiają naukę mówców w jednym modelu.
- Decyzja do podjęcia: 50 h w jednym głosie (najprostsze, najlepsze dla małych modeli) czy np. 4 × 12,5 h.

### Generowanie

- **Sprzęt: 8× RTX 3090 Ti** (Ampere sm_86, 24 GB każda; decyzja 2026-11-20). Nie RTX 5080 — ta zostaje
  dla botów (Pytajnik). Ampere to dobrze wspierana architektura: bez obejść jak przy sm_120.
- **Silnik: `nano-vllm-voxcpm`** (decyzja 2026-11-20; batchowanie, obsługuje VoxCPM2). Jeden proces na kartę
  (8 instancji), teksty podzielone na 8 rozłącznych części (np. po hashu `zrodlo_id`), wyniki do jednego
  manifestu.
- Przepustowość — do zmierzenia na jednej karcie przed pełnym przebiegiem. Punkty odniesienia:
  - autor nano-vllm-voxcpm, RTX 4090 (też 24 GB): ok. 100–127× czas rzeczywisty przy 16–32 równoległych
    żądaniach (przy 64 i `gpu_memory_utilization=0.8` bywa OOM na 24 GB);
  - 3090 Ti ma wyraźnie mniej mocy obliczeniowej niż 4090, więc ostrożnie ok. 40–70× na kartę,
    **ok. 300–550× na 8 kart**: 50 h audio to rząd 6–10 minut samej syntezy;
  - dla porównania obecny serwer Pytajnika (batch 1, RTX 5080): 1,64× czas rzeczywisty.
- Przy takiej przepustowości wąskim gardłem będzie raczej filtr jakości (Canary, UTMOS, podobieństwo mówcy)
  i zapis na dysk niż sama synteza — filtr też uruchamiać na wszystkich 8 kartach.
- Parametry jak w machinekind v2: CFG 2.0–2.5, 10 kroków dyfuzji, do 3 prób na tekst.

### Filtr jakości (każdy klip)

Na wzór `machinekind/polish-synthetic-speech-voxcpm2(-v2)`, gdzie przeszło 90% prób:
- transkrypcja i CER ≤ 8%, stosunek długości 0,8–1,25. Używać **Canary-1b-v2** (lepszy od Whispera na polskim).
  Uwaga: w Pytajniku dwie wersje Canary różniły się na tym samym nagraniu („Wyjaśnij” / „Wyjaśnił”, „E,” na
  początku) — przy odrzuceniu warto sprawdzić drugim rozpoznawaczem, zanim się zwali winę na TTS;
- UTMOS ≥ 3,2;
- podobieństwo mówcy do głosu wzorcowego ≥ 0,80;
- tempo 7–22 liter/s, wysokość głosu 0,6–1,75 × wzorzec;
- brak doklejonych dźwięków i ciszy > 1 s w środku.

### Zapis

- Audio: FLAC, 48 kHz mono (oryginał VoxCPM2). 50 h to ok. 17 GB w WAV i ok. 8–9 GB we FLAC.
  Dla małych modeli i tak resample do 22,05/24 kHz w treningu.
- Opcjonalnie latenty AudioVAE VoxCPM2 (25 Hz × 64), małe na dysku, przydadzą się uczniowi w stylu VoxCPM.
- Manifest JSONL: `id, audio, sekundy, glos, tekst (surowy, wejście VoxCPM2), [tekst_mowiony później], zrodlo_id (id rozmowy), rola
  (user/assistant), temat, proba, cer, utmos, podobienstwo, tempo, f0`.
- Miejsce (2026-11-20): `/` 43 GB wolne, drugi dysk `/media/adamo/21c5…` 43 GB wolne.
- Podział: walidacja i test po rozmowach (`zrodlo_id`), nie po klipach, żeby ten sam tekst nie wyciekł.

## Etap 2: drabina modeli 1M → 200M

| rozmiar | kandydaci | uwagi |
|---|---|---|
| ok. 1–5M | mały VITS / Piper-like, Matcha-tiny + HiFi-GAN V2 (0,93M) | punkt odniesienia: Piper ma 16–60M; ile da się zachować przy 1M |
| ok. 20M | Matcha-TTS (jak Matcha-TTS-PL-v2: 20,9M + vocoder) | gotowy przepis machinekind; mówi poprawnie, ale nienaturalnie |
| ok. 50–100M | non-AR flow matching na latentach (Supertonic-like, F5/E2-like), Vocos | Supertonic 3 (ok. 99M) to najlepszy mały model w testach |
| ok. 100–200M | AR + lokalna dyfuzja (VoxCPM-like, pocket-tts-like) z MeanFlow | najlepsza prozodia, ale ryzyko gubienia słów jak w pocket-tts |

Później, osobno: uczeń z backbone Poziomki (MoE, ok. 0,45B aktywnych przy top-8) w miejscu Base LM VoxCPM2 —
większy, ale z polskim LM do prozodii.

Do decyzji przy każdym szczeblu: wejście (fonemy espeak-ng czy znaki), reprezentacja (mel czy latenty VAE),
długości (jawne, MAS, czy uwaga), vocoder (HiFi-GAN V1/V2, Vocos) i jego koszt w małych modelach.

## Ewaluacja (wspólna dla wszystkich)

- Stały zestaw testowy: odłożone klipy (po rozmowach) plus zdania z quizu podatkowego Pytajnika
  (`tts-test/zdania.txt`: skróty, liczby, pytania z wariantami A/B/C).
- WER/CER przez Canary z normalizacją liczb (num2words `pl`). Uwaga z Pytajnika: część „błędów” to artefakty
  normalizacji (np. „APIT-36”), więc patrzeć też na listę różnic, nie tylko na liczbę.
- UTMOS (naturalność) — WER nie mówi nic o tym, czy brzmi jak człowiek (Matcha: dobry WER, nienaturalna).
- Podobieństwo mówcy, RTF i czas do pierwszego dźwięku na **4 rdzeniach CPU**, rozmiar na dysku, licencja.
- Próbki do odsłuchu w jednym folderze, jeden mp3 na model (jak `pytajnik/probki-tts/`).

## Punkty odniesienia (Pytajnik, 4 rdzenie i7-11700K, 6 zdań z quizu)

| model | parametry | WER (Canary) | RTF CPU | uwagi |
|---|---|---|---|---|
| Supertonic 3 M1 | ok. 99M | 3,9% | 0,17 | najlepszy mały; bez klonowania |
| Matcha-TTS-PL-Edge-v2 fp32 | 21M | 7,8–8,8% | 0,075 | poprawny, ale nienaturalny |
| Piper darkman | ok. 20M | 6,9% | 0,03 | syntetyczny; skróty czyta źle („VAT” → „fałat”), bo espeak bez normalizacji |
| pocket-tts-polish-6l int8 | 110M | 16,7% | 0,24 | gubi fragmenty zdań |
| MOSS-TTS-Nano | 100M | 23,5% | 1,96 | za wolny |
| VoxCPM2 (nauczyciel, GPU) | 2,3B | 10,8% | 0,61–0,79 na GPU | najładniejszy; część jego „błędów” to pomyłki STT |

Próbki: `~/projects/misc/pytajnik/probki-tts/`; skrypty i wyniki: `/media/adamo/21c5…/projects/pytajnik/tts-test/`.

## Źródła i odnośniki

- VoxCPM2 (Apache-2.0): architektura AR + LocDiT (flow matching, 10 kroków, MeanFlow w kodzie, `mean_mode`),
  AudioVAE 16 → 48 kHz; trening: `scripts/train_voxcpm_finetune.py` w repo OpenBMB/VoxCPM.
- machinekind: datasety `polish-synthetic-speech-voxcpm2(-v2)` (82 h, CC-BY-SA) i modele Matcha-TTS-PL(-Edge)-v2 —
  ten sam pomysł w małej skali; ich filtr jakości jest powyżej.
- Dane naturalne do porównań i pretreningu: mc-speech (22 h, studio, CC0), pwr-azon, CML-TTS pl, MLS pl,
  VoxPopuli pl.

## Otwarte pytania

- Tylko „poziomka” czy dołożyć 2–4 głosy z voice design?
- Od jakiego rozmiaru ucznia surowy tekst wystarcza, a poniżej którego potrzebny jest `tekst_mowiony` (normalizator jako osobny moduł).
- Ile czystych danych naturalnych dołożyć dla prozodii (z ich licencjami).
