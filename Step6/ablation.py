"""Alpha ablation: each dataflow step is added on top of the previous one.

For every configuration this script (1) runs the full Step3 regression, (2) reruns the Step 1
sample with waves, takes the cycle and SRAM-access counts the testbench reports (cross-checked
by run.py against an independent model) and counts bit toggles of every corelet net outside the
verification phases, then writes results/ablation.json and results/ablation.png.
Toggle counts are a first-order proxy for dynamic energy; Quartus power comes from Step 5.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "Step3" / "run.py"
LADDER = [  # label, cumulative Step3 flags
    ("Baseline", []),
    ("+ L0 pipelining", ["--pipeline-l0"]),
    ("+ streaming writeback", ["--pipeline-l0", "--stream-ofifo"]),
    ("+ zero-word skipping", ["--pipeline-l0", "--stream-ofifo", "--skip-zero"]),
    ("+ pipelined SFU", ["--pipeline-l0", "--stream-ofifo", "--skip-zero", "--pipeline-sfu"]),
    ("+ operand isolation", ["--pipeline-l0", "--stream-ofifo", "--skip-zero", "--pipeline-sfu",
                             "--operand-isolation"]),
]
STAGES = [  # figure segment, testbench stages
    ("L1 load", {"1"}), ("Weight load", {"2"}), ("L0 fill", {"3"}), ("PE execute", {"4", "45"}),
    ("Psum writeback", {"5"}), ("SFU accumulate + ReLU", {"6", "7", "67"})]
GROUPS = {"array": ".processing_array.", "l0": ".input_buffer.", "ofifo": ".output_buffer.",
          "sfu": ".output_channel["}
SCOPE = "core_tb.dut.datapath."


def toggles(vcd: Path) -> dict:
    """Bit flips per corelet net group, skipping verification phases 50 (psum check) and 8."""
    names, scope = defaultdict(list), []
    lines = vcd.open()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "$scope":
            scope.append(parts[2])
        elif parts[0] == "$upscope":
            scope.pop()
        elif parts[0] == "$var":
            names[parts[3]].append(".".join(scope + [parts[4]]))
        elif parts[0] == "$enddefinitions":
            break
    watched = {code for code, paths in names.items() if any(p.startswith(SCOPE) for p in paths)}
    multiplier = {code for code in watched
                  if any(p.endswith((".mac_instance.a", ".mac_instance.b")) for p in names[code])}
    phase_code = next(code for code, paths in names.items() if "core_tb.phase" in paths)
    group_of = {code: next((g for g, key in GROUPS.items() if any(key in p for p in names[code])), "other")
                for code in watched}
    value, phase, started = {}, 0, False
    counts = defaultdict(int)
    for line in lines:
        if line[0] == "#":
            started = started or line.strip() != "#0"
            continue
        if line[0] in "bB":
            bits, code = line[1:].split()
        elif line[0] in "01xXzZ":
            bits, code = line[0], line[1:].strip()
        else:
            continue
        if code == phase_code and set(bits) <= {"0", "1"}:
            phase = int(bits, 2)
        if code not in watched:
            continue
        old, value[code] = value.get(code), bits
        if not started or old is None or phase in (50, 8):
            continue
        width = max(len(old), len(bits))
        old, bits = old.rjust(width, "0"), bits.rjust(width, "0")
        flips = sum(a != b and a in "01" and b in "01" for a, b in zip(old, bits))
        counts["corelet"] += flips
        counts[group_of[code]] += flips
        if code in multiplier:
            counts["multiplier_inputs"] += flips
    return dict(counts)


def run(flags: list[str], build: Path, sim: str, extra: list[str]) -> None:
    subprocess.run([sys.executable, str(RUN), "--sim", sim, "--build-root", str(build), *flags, *extra],
                   check=True, capture_output=True, text=True)


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, quiet, grid, neutral = "#191919", "#666666", "#e5e5e5", "#5f6b7a"
    colors = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
    rows = [r for r in results if r["label"] != "+ operand isolation"][::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    base, labeled = results[0]["work_cycles"], set()
    for y, r in enumerate(rows):
        x = 0
        for (stage, _), color in zip(STAGES, colors):
            width = r["stage_cycles"][stage]
            if width:
                left.barh(y, width, left=x, height=0.62, color=color, edgecolor="white", linewidth=1.5,
                          label=None if stage in labeled else stage)
                labeled.add(stage)
            x += width
        left.text(x + 25, y, f"{x:,}  ({base / x:.2f}x)" if x != base else f"{x:,}",
                  va="center", fontsize=9, color=ink)
    left.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9, color=ink)
    left.set_xlabel("Work cycles per inference (features.27, one sample)", fontsize=9, color=quiet)
    left.set_xlim(0, base * 1.22)
    handles, names = left.get_legend_handles_labels()
    order = [names.index(stage) for stage, _ in STAGES]
    left.legend([handles[i] for i in order], [names[i] for i in order], ncol=3, fontsize=8,
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    accesses = [r["sram_accesses"] for r in rows]
    right.barh(range(len(rows)), accesses, height=0.62, color=neutral, edgecolor="white", linewidth=1.5)
    for y, value in enumerate(accesses):
        change = f"  (\u2212{1 - value / results[0]['sram_accesses']:.0%})" if value != results[0]["sram_accesses"] else ""
        right.text(value + 15, y, f"{value:,}{change}", va="center", fontsize=9, color=ink)
    right.set_yticks(range(len(rows)), [""] * len(rows))
    right.set_xlabel("SRAM accesses per inference (L1 + psum, work stages)", fontsize=9, color=quiet)
    right.set_xlim(0, max(accesses) * 1.25)
    for ax in (left, right):
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=quiet, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="iverilog", choices=("iverilog",),
                        help="Icarus only: the toggle count needs every corelet net in the VCD")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--replot", action="store_true", help="redraw the figure from ablation.json")
    args = parser.parse_args()
    if args.replot:
        plot(json.loads((args.output / "ablation.json").read_text()), args.output / "ablation.png")
        return 0
    results = []
    with tempfile.TemporaryDirectory(prefix="ablation_") as tmp:
        for label, flags in LADDER:
            build = Path(tmp) / str(len(results))
            run(flags, build / "regress", args.sim, ["--regress"])
            regression = json.loads(next((build / "regress").rglob("regression.json")).read_text())
            run(flags, build / "waves", args.sim,
                ["--case", str(build / "regress/inputs/team_handoff"), "--repeat", "1", "--waves"])
            case_dir = next((build / "waves").rglob("core_tb.vcd")).parent
            summary = json.loads((case_dir / "summary.json").read_text())
            cycles = [row for row in summary["cycles"] if row["job"] == "0"]
            stage_cycles = {name: sum(int(row["cycles"]) for row in cycles if row["stage"] in ids)
                            for name, ids in STAGES}
            activity = toggles(case_dir / "core_tb.vcd")
            # The OFIFO changes implementation at "+ streaming writeback"; every other block
            # keeps the same RTL in every row except operand isolation.
            activity["like_for_like"] = activity["corelet"] - activity["ofifo"]
            sram = {key: summary[key] for key in ("l1_reads", "l1_writes", "psum_reads", "psum_writes")}
            results.append(dict(
                label=label, flags=flags, work_cycles=summary["work_cycles_per_job"],
                stage_cycles=stage_cycles, issued_words=summary["issued_words"],
                mac_slots=summary["issued_words"] * 64, sfu_reads=summary["sfu_reads"],
                sram=sram, sram_accesses=sum(sram.values()), toggles=activity,
                regression={c["case"]: c["work_cycles_per_job"] for c in regression["cases"]}))
            print(label, results[-1]["work_cycles"], results[-1]["sram_accesses"],
                  activity["like_for_like"], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    plot(results, args.output / "ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
