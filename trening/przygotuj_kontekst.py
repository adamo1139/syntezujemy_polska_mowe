"""Wyrównuje zdania wewnątrz pojedynczych klipów BlueTTS i zapisuje CSV dla kontekstowego T2L."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample
from transformers import pipeline
from tqdm import tqdm


def words(text):
    return re.findall(r"[^\W_]+", text.casefold())


def split_sentences(text):
    # Krótkie człony bywają skrótami albo numeracją.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])', text.strip())
    if len(sentences) < 2 or any(len(words(s)) < 2 for s in sentences):
        return []
    return sentences


def quiet_boundary(audio, sr, end, start, peak):
    # Przeszukaj przerwę między słowami i margines błędu znaczników Whisper.
    if end is None or start is None or not 0 <= end <= start <= len(audio) / sr or start - end > 1.5:
        return None
    lo = max(0, round((end - 0.08) * sr))
    hi = min(len(audio), round((start + 0.08) * sr))
    hop = max(1, round(sr * 0.01))
    frames = (hi - lo) // hop
    if frames < 4:
        return None
    power = (audio[lo:lo + frames * hop].reshape(frames, hop) ** 2).mean(axis=1)
    quiet = power < peak * 10 ** -3.5
    # Środek co najmniej 40 ms ciszy, możliwie blisko połowy pauzy.
    valid = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(4), mode="valid") == 4)
    if not len(valid):
        return None
    centers = lo + (valid + 2) * hop
    return int(centers[np.argmin(abs(centers - sr * (start + end) / 2))])


def align(model, source, sentences):
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    samples = resample(torch.from_numpy(mono), sr, 16000).numpy()
    result = model(samples, return_timestamps="word",
                   generate_kwargs={"language": "pl", "task": "transcribe"})
    timed = [(token, *chunk["timestamp"]) for chunk in result["chunks"] for token in words(chunk["text"])]
    expected = [words(s) for s in sentences]
    if [t[0] for t in timed] != [w for sentence in expected for w in sentence]:
        return []
    hop = max(1, round(sr * 0.01))
    if len(mono) < hop:
        return []
    peak = (mono[:len(mono) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1).max()
    bounds, texts, current = [0], [], sentences[0]
    for i, offset in enumerate(np.cumsum([len(s) for s in expected])[:-1]):
        cut = quiet_boundary(mono, sr, timed[offset-1][2], timed[offset][1], peak)
        if cut is not None and cut - bounds[-1] >= sr and len(mono) - cut >= sr:
            bounds.append(cut)
            texts.append(current)
            current = sentences[i+1]
        else:
            current += " " + sentences[i+1]
    if not texts:
        return []
    texts.append(current)
    bounds.append(len(audio))
    return [(text, audio[a:b], sr) for text, a, b in zip(texts, bounds, bounds[1:])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--tylko-kontekst", action="store_true", help="walidacja tylko na podzielonych klipach")
    p.add_argument("--model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    if args.csv.resolve() == args.out.resolve():
        p.error("wybierz osobny wyjściowy CSV")
    cache = args.out.resolve().parent / (args.out.stem + "_klipy")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    model = None
    accepted = 0
    with args.csv.open(encoding="utf-8") as src, tmp.open("w", newline="", encoding="utf-8") as dst:
        rows = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=[*rows.fieldnames, "left_context", "right_context"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in tqdm(rows, desc=args.csv.stem):
            name = Path(row["filename"])
            source = name if name.is_absolute() else (args.csv.parent / name).resolve()
            row["filename"] = str(source)
            sentences = split_sentences(row["whisper_phonemes"])
            clips = []
            if sentences:
                st = source.stat()
                key = hashlib.sha256(json.dumps([str(source), sentences, st.st_size, st.st_mtime_ns,
                                                args.model], ensure_ascii=False).encode()).hexdigest()
                folder = cache / key[:2]
                folder.mkdir(exist_ok=True)
                marker = folder / (key + ".json")
                cached = json.loads(marker.read_text()) if marker.exists() else None
                if cached is not None and all(Path(path).is_file() for _, path in cached):
                    clips = cached
                else:
                    if model is None:
                        model = pipeline("automatic-speech-recognition", model=args.model, device=args.device,
                                         dtype=torch.float16 if args.device == "cuda" else torch.float32)
                    for i, (text, audio, sr) in enumerate(align(model, source, sentences)):
                        path = folder / f"{key}_{i}.flac"
                        sf.write(path, audio, sr, subtype="PCM_24")
                        clips.append((text, str(path)))
                    pending = marker.with_suffix(".tmp")
                    pending.write_text(json.dumps(clips, ensure_ascii=False))
                    os.replace(pending, marker)
            if clips:
                accepted += 1
                for i, (text, path) in enumerate(clips):
                    writer.writerow(dict(row, filename=path, whisper_phonemes=text,
                                         left_context=clips[i-1][0] if i else "",
                                         right_context=clips[i+1][0] if i+1 < len(clips) else ""))
            elif not args.tylko_kontekst:
                writer.writerow(row)
    if not accepted:
        raise SystemExit(f"{args.csv}: brak potwierdzonych granic, nie publikuję manifestu")
    os.replace(tmp, args.out)
    print(f"{args.out}: oryginalne klipy z potwierdzonym podziałem: {accepted}")


if __name__ == "__main__":
    main()
