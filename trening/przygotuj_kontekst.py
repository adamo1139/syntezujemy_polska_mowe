"""Wyrównuje zdania wewnątrz pojedynczych klipów BlueTTS i zapisuje CSV dla kontekstowego T2L."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from itertools import batched
from multiprocessing import get_context
from threading import Lock
from uuid import uuid4
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from kestrel_native import resample_audio_mono
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
    # Przeszukaj przerwę między słowami i margines błędu znaczników słów.
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
    samples = mono if sr == 16000 else resample_audio_mono(
        mono, sr, 16000, max_output_values=(len(mono) * 16000 + sr - 1) // sr + 64,
    )
    result = model.transcribe(audio=np.ascontiguousarray(np.clip(samples, -1, 1)), sample_rate=16000,
                              timestamps="word")
    timed = [(token, w["start"], w["end"]) for seg in result["segments"]
             for w in seg.get("words", []) for token in words(w["word"])]
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


def previous_asr(directory):
    transcripts = {}
    for path in sorted(directory.glob("asr_*_z_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue  # an interrupted run can leave its last line incomplete
                if isinstance(item.get("asr"), str):
                    transcripts[item["id"]] = item["asr"]
    if transcripts:
        print(f"{directory}: {len(transcripts)} wcześniejszych transkrypcji", flush=True)
    return transcripts


def tasks(rows, args):
    archives = {}
    fallback_dir = args.asr_dir or args.csv.parent / "asr"
    fallback = previous_asr(fallback_dir)
    archives[fallback_dir.resolve()] = fallback
    for row in rows:
        name = Path(row["filename"])
        source = name if name.is_absolute() else (args.csv.parent / name).resolve()
        row["filename"] = str(source)
        sentences = split_sentences(row["whisper_phonemes"])
        mismatch = False
        if sentences:
            old_text = fallback.get(source.stem)
            if source.parent.name == "audio":
                directory = source.parent.parent / "asr"
                if directory not in archives:
                    archives[directory] = previous_asr(directory)
                old_text = archives[directory].get(source.stem, old_text)
            mismatch = old_text is not None and words(old_text) != words(row["whisper_phonemes"])
        yield row, sentences, mismatch


class Worker:
    def __init__(self, args, cache, stack):
        self.args = args
        self.cache = cache
        self.stack = stack
        self.lock = Lock()
        self.model = None
        self.pool = stack.enter_context(ThreadPoolExecutor(args.watki))

    def get_model(self):
        with self.lock:
            if self.model is None:
                import moondream as md
                self.model = self.stack.enter_context(md.photon(self.args.model, device=self.args.device))
            return self.model

    def one(self, task):
        row, sentences, mismatch = task
        if not sentences:
            return row, [], "bez_kontekstu"
        source = Path(row["filename"])
        st = source.stat()
        key = hashlib.sha256(json.dumps([str(source), sentences, st.st_size, st.st_mtime_ns,
                                        self.args.model, "photon-resample-v1"], ensure_ascii=False).encode()).hexdigest()
        folder = self.cache / key[:2]
        marker = folder / (key + ".json")
        if marker.exists():
            clips = json.loads(marker.read_text())
            if all(Path(path).is_file() for _, path in clips):
                return row, clips, "cache"
        if mismatch:
            return row, [], "pominiete_po_asr"
        clips = []
        folder.mkdir(exist_ok=True)
        suffix = uuid4().hex
        for i, (text, audio, sr) in enumerate(align(self.get_model(), source, sentences)):
            path = folder / f"{key}_{i}.flac"
            pending = path.with_suffix(f".{suffix}.flac")
            sf.write(pending, audio, sr, subtype="PCM_24")
            os.replace(pending, path)
            clips.append((text, str(path)))
        pending = marker.with_suffix(f".{suffix}.tmp")
        pending.write_text(json.dumps(clips, ensure_ascii=False))
        os.replace(pending, marker)
        return row, clips, "asr_ze_znacznikami"

    def batch(self, items):
        return list(self.pool.map(self.one, items))


def start_worker(card, args, cache):
    from multiprocessing.util import Finalize
    global worker
    os.environ["CUDA_VISIBLE_DEVICES"] = card
    stack = ExitStack()
    Finalize(None, stack.close, exitpriority=1)
    worker = Worker(args, cache, stack)


def worker_batch(items):
    return worker.batch(items)


def batches(items, args, cache, stack):
    chunks = batched(items, args.partia)
    if not args.karty:
        local = Worker(args, cache, stack)
        for chunk in chunks:
            yield local.batch(chunk)
        return
    executors = [stack.enter_context(ProcessPoolExecutor(1, mp_context=get_context("spawn"),
                 initializer=start_worker, initargs=(card, args, cache))) for card in args.karty]
    pending = deque()
    for i, chunk in enumerate(chunks):
        pending.append(executors[i % len(executors)].submit(worker_batch, chunk))
        if len(pending) >= 2 * len(executors):
            yield pending.popleft().result()
    for future in pending:
        yield future.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--tylko-kontekst", action="store_true", help="walidacja tylko na podzielonych klipach")
    p.add_argument("--model", choices=["moondream/parakeet-ultra", "moondream/parakeet-redux"],
                   default="moondream/parakeet-ultra")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--asr-dir", type=Path, help="dodatkowy katalog wyników (--praca) filtra dla tego zbioru")
    p.add_argument("--karty", help="fizyczne numery GPU po przecinku, po jednym procesie na kartę")
    p.add_argument("--watki", type=int, default=16, help="równoległe żądania do Photon na kartę")
    p.add_argument("--partia", type=int, default=128, help="wiersze CSV w partii jednego procesu")
    args = p.parse_args()
    if args.watki < 1 or args.partia < 1:
        p.error("--watki i --partia muszą być dodatnie")
    if args.karty:
        args.karty = [s.strip() for s in args.karty.split(",")]
        if args.device != "cuda" or not all(s.isdecimal() for s in args.karty) or len(set(args.karty)) != len(args.karty):
            p.error("--karty wymaga różnych numerów GPU oraz --device cuda")
    if args.csv.resolve() == args.out.resolve():
        p.error("wybierz osobny wyjściowy CSV")
    cache = args.out.resolve().parent / (args.out.stem + "_klipy")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".csv.tmp")
    accepted = 0
    counts = Counter()
    with ExitStack() as stack, args.csv.open(encoding="utf-8") as src, tmp.open("w", newline="", encoding="utf-8") as dst:
        rows = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=[*rows.fieldnames, "left_context", "right_context"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        with tqdm(desc=args.csv.stem, unit="klip") as progress:
            for batch in batches(tasks(rows, args), args, cache, stack):
                for row, clips, reason in batch:
                    counts[reason] += 1
                    if clips:
                        accepted += 1
                        for i, (text, path) in enumerate(clips):
                            writer.writerow(dict(row, filename=path, whisper_phonemes=text,
                                                 left_context=clips[i-1][0] if i else "",
                                                 right_context=clips[i+1][0] if i+1 < len(clips) else ""))
                    elif not args.tylko_kontekst:
                        writer.writerow(row)
                progress.set_postfix(dict(counts), refresh=False)
                progress.update(len(batch))
    if not accepted:
        raise SystemExit(f"{args.csv}: brak potwierdzonych granic, nie publikuję manifestu")
    os.replace(tmp, args.out)
    print(f"{args.out}: oryginalne klipy z potwierdzonym podziałem: {accepted}")


if __name__ == "__main__":
    main()
