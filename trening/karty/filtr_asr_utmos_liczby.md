**Zbiór po dwóch filtrach jakości.** Z {wszystkich} klipów syntezy usunięto {usuniete} ({procent}%):

1. **ASR.** Każdy klip przepisał [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra), a tekst
   podany modelowi i transkrypcję porównano po normalizacji liczb (Morfeusz 2). Odpadły klipy z błędem znakowym
   (CER) powyżej 0,5, czyli z bełkotem, zapętleniem albo pominiętym fragmentem (`skrypty/czysc_asr.py`). Próg jest
   dużo łagodniejszy niż w zbiorach bez liczb, bo Parakeet zapisuje liczby cyframi, a przy zamianie na słowa
   odmiana liczebnika („dwudziestym szóstym”, „dwudziestego szóstego”) często różni się od tekstu, choć VoxCPM2
   przeczytał dobrze. Drobne błędy w liczbach mogły więc zostać.
2. **UTMOS.** Na tym, co zostało, policzono [UTMOS22 strong](https://github.com/sarulab-speech/UTMOS22)
   (przewidywana ocena naturalności) i usunięto klipy z oceną poniżej 2,75 (`skrypty/czysc_utmos.py`). Długie
   klipy VoxCPM2 czytał wyraźnie za szybko i UTMOS ocenia je dużo niżej (korelacja oceny z długością ok. −0,6),
   więc ten próg usuwa większość klipów dłuższych niż 20 s. Zbiór jest przez to przesunięty w stronę krótkich
   wypowiedzi: zobacz „Długość klipów”.

Kolumna `text` to dalej tekst podany modelowi, a nie transkrypcja, więc drobne przekręcenia mogły zostać.
Audio to Ogg Opus 128 kbit/s; mowa mono jest w nim praktycznie nieodróżnialna od FLAC.
