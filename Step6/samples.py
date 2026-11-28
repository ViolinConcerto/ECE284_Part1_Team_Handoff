"""Run the baseline and the alpha dataflow on additional real CIFAR-10 samples.

Export each sample with Step 1 first (PyTorch + CIFAR-10, e.g. on DataHub):
    python train.py --device cuda --data-root data --output runs/sample_N \
        --evaluate artifacts/checkpoints/best.pth --sample-index N
then pass the exported `vectors` folders (each holds tensors.npz):
    python Step6/samples.py runs/sample_*/vectors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Step3"))
from run import pack, write_words  # noqa: E402

MODES = {"baseline": [], "alpha": ["--pipeline-l0", "--stream-ofifo", "--skip-zero", "--pipeline-sfu"]}


def to_case(vectors: Path, destination: Path, name: str) -> dict:
    """Pack tensors.npz into Step3's normalized case format (channel 0 in the low bits)."""
    t = np.load(vectors / "tensors.npz")
    act, weight, out = t["activation_padded_int"][0], t["weight_int"], t["psum_relu_int"][0]
    inputs = [pack(act[:, y, x].tolist(), 4) for y in range(act.shape[1]) for x in range(act.shape[2])]
    weights = [pack(weight[o, :, ky, kx].tolist(), 4) for ky in range(3) for kx in range(3) for o in range(8)]
    golden = [pack(out[:, y, x].tolist(), 16) for y in range(out.shape[1]) for x in range(out.shape[2])]
    destination.mkdir(parents=True, exist_ok=True)
    cfg = dict(name=name, height=4, width=4, kernel=3, padding=1, stride=1, input_channels=8,
               output_channels=8, activation_bits=4, weight_bits=4, psum_bits=16,
               origin=f"{vectors}; trained-model sample")
    (destination / "case.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for filename, words, bits in (("input.txt", inputs, 32), ("weight.txt", weights, 32), ("psum.txt", golden, 128)):
        write_words(destination / filename, words, bits)
    interior = act[:, 1:-1, 1:-1]
    return dict(activation_zero_fraction=round(float(np.mean(interior == 0)), 4),
                zero_interior_words=int(np.all(interior == 0, axis=0).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vectors", type=Path, nargs="+", help="exported vectors folders")
    parser.add_argument("--sim", default="auto")
    parser.add_argument("--vivado-bin")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/samples.json")
    args = parser.parse_args()
    sim = ["--sim", args.sim] + (["--vivado-bin", args.vivado_bin] if args.vivado_bin else [])
    rows = []
    with tempfile.TemporaryDirectory(prefix="samples_") as tmp:
        for index, vectors in enumerate(args.vectors):
            case = Path(tmp) / "cases" / f"sample_{index}"
            row = dict(source=str(vectors), **to_case(vectors, case, case.name))
            for mode, flags in MODES.items():
                build = Path(tmp) / mode
                subprocess.run([sys.executable, str(ROOT / "Step3/run.py"), "--case", str(case),
                                "--repeat", "1", "--build-root", str(build), *sim, *flags],
                               check=True, capture_output=True, text=True)
                summary = json.loads(next(build.rglob(f"{case.name}/summary.json")).read_text())
                row[mode] = dict(work_cycles=summary["work_cycles_per_job"],
                                 issued_words=summary["issued_words"])
            row["speedup"] = round(row["baseline"]["work_cycles"] / row["alpha"]["work_cycles"], 3)
            rows.append(row)
            print(f"PASS {vectors}: {row['baseline']['work_cycles']} -> {row['alpha']['work_cycles']} cycles "
                  f"({row['speedup']}x), zero activations {row['activation_zero_fraction']:.1%}", flush=True)
    speedups = [r["speedup"] for r in rows]
    report = dict(samples=len(rows), speedup_min=min(speedups), speedup_mean=round(float(np.mean(speedups)), 3),
                  speedup_max=max(speedups), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} samples, all 0 mismatch; speedup {report['speedup_min']}-{report['speedup_max']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
