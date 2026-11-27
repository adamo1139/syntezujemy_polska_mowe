"""Checkpoint MOS evaluation. Run separately with python -m training.t2l.sample_mos --out scores.csv FILE.wav ..."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

METRICS = ("dnsmos_ovrl", "utmos22", "utmosv2")


def score_checkpoint(paths, log_dir, step):
    """Run outside the trainer so evaluator dependencies, GPU memory and RNG stay isolated."""
    output = Path(log_dir, f"step_{step}_mos.csv").resolve()
    training_dir = Path(__file__).resolve().parents[2]
    cpu_python = training_dir / ".venv-mos/bin/python"
    python = os.environ.get("T2L_MOS_PYTHON") or (cpu_python if cpu_python.is_file() else sys.executable)
    command = [os.path.abspath(python), "-m", "training.t2l.sample_mos",
               "--out", str(output), *(str(Path(p).resolve()) for p in paths)]
    try:
        subprocess.run(command, cwd=training_dir, check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        with output.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        means = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric]]
            if values:
                means[f"probki/{metric}"] = sum(values) / len(values)
                means[f"probki/{metric}_n"] = len(values)
        return means
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[MOS] {error}", flush=True)
        return {}


def load_scorer(metric):
    if metric == "dnsmos_ovrl":
        from training.rl.rewards import DNSMOS
        model = DNSMOS("cpu")
        return lambda path, wav, sr: model(wav, sr)
    if metric == "utmos22":
        import torch
        from training.utmos import utmos, wczytaj_utmos
        model = wczytaj_utmos("cpu")
        return lambda path, wav, sr: utmos(model, torch.from_numpy(wav), sr)
    import utmosv2
    model = utmosv2.create_model(pretrained=True, device="cpu")
    return lambda path, wav, sr: model.predict(input_path=str(path), device="cpu", verbose=False)


def main():
    import soundfile as sf
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = [{"file": p.name} for p in args.files]
    for metric in METRICS:
        try:
            scorer = load_scorer(metric)
        except Exception as error:
            print(f"[MOS] {metric} niedostępny: {error}", flush=True)
            continue
        with torch.inference_mode():
            for path, row in zip(args.files, rows):
                try:
                    wav, sr = sf.read(path, dtype="float32")
                    score = float(scorer(path, wav, sr))
                    if not math.isfinite(score):
                        raise ValueError(f"non-finite MOS: {score}")
                    row[metric] = score
                except Exception as error:
                    print(f"[MOS] {metric} {path}: {error}", flush=True)
        del scorer
    tmp = args.out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, args.out)
    if not any(len(row) > 1 for row in rows):
        raise SystemExit("Brak ocen MOS")


if __name__ == "__main__":
    main()
