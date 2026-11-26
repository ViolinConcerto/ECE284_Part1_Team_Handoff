"""Prepare and verify the unsigned-video / signed-difference RTL experiment.

This runner uses Python integers for the independent convolution reference.  The
trained Part1 weights are an explicitly selected experiment workload, not weights
distributed with the teacher's four frames.  It never changes frozen Part1 data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PART3 = ROOT.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_decimals(path: Path, count: int) -> list[int]:
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not re.fullmatch(r"[+-]?\d+", line.strip()):
            raise ValueError(f"{path}:{number}: expected one decimal integer")
        values.append(int(line))
    if len(values) != count:
        raise ValueError(f"{path}: expected {count} scalars, found {len(values)}")
    return values


def read_binary(path: Path, width: int, count: int) -> list[int]:
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if not re.fullmatch(f"[01]{{{width}}}", line):
            raise ValueError(f"{path}:{number}: expected {width} binary digits")
        values.append(int(line, 2))
    if len(values) != count:
        raise ValueError(f"{path}: expected {count} words, found {len(values)}")
    return values


def pack(values: list[int], bits: int, signed: bool) -> int:
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if any(v < low or v > high for v in values):
        raise ValueError(f"value does not fit {'signed' if signed else 'unsigned'} {bits}-bit lane")
    return sum((v & ((1 << bits) - 1)) << (lane * bits) for lane, v in enumerate(values))


def unpack(word: int, bits: int, lanes: int = 8, signed: bool = True) -> list[int]:
    values = [(word >> (lane * bits)) & ((1 << bits) - 1) for lane in range(lanes)]
    return [v - (1 << bits) if signed and v >= (1 << (bits - 1)) else v for v in values]


def write_binary(path: Path, words: list[int], width: int) -> None:
    if any(word < 0 or word >= (1 << width) for word in words):
        raise ValueError(f"{path}: packed word exceeds {width} bits")
    path.write_text("".join(f"{word:0{width}b}\n" for word in words), encoding="ascii")


def resolve(case_path: Path, value: str) -> Path:
    return (case_path.parent / value).resolve()


def load_case(case_path: Path) -> dict:
    cfg = json.loads(case_path.read_text(encoding="utf-8-sig"))
    required = {"channels": 8, "output_channels": 8,
                "kernel": 3, "padding": 1, "stride": 1, "layout": "CHW",
                "input_bits": 8, "weight_bits": 4}
    for name, value in required.items():
        if cfg.get(name) != value:
            raise ValueError(f"video TB requires {name}={value!r}, received {cfg.get(name)!r}")
    if cfg.get("bias", False) is not False or cfg.get("groups", 1) != 1:
        raise ValueError("experiment requires bias=false and groups=1")
    for key in ("height", "width", "chunk"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["chunk"] > 64 or cfg["height"] * cfg["width"] > 256:
        raise ValueError("reference video TB requires chunk<=64 and 8*NOUT<=2048")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("name", "")):
        raise ValueError("case name must use letters, digits, '_' or '-'")
    folder = resolve(case_path, cfg["frames_dir"])
    files = [folder / f"X{i}.txt" for i in range(4)]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {folder}; see Part3/HANDOFF.md for the course data download.")
    frames = [read_decimals(p, 8 * cfg["height"] * cfg["width"]) for p in files]
    if any(not 0 <= value <= 255 for frame in frames for value in frame):
        raise ValueError("teacher frames must contain unsigned 8-bit values (0..255)")
    deltas = [[v - base for v, base in zip(frame, frames[0])] for frame in frames[1:]]
    if any(not -8 <= value <= 7 for frame in deltas for value in frame):
        raise ValueError("Xi-X0 exceeds signed 4-bit; no clipping or wrapping is permitted")
    weight_path = resolve(case_path, cfg["weights"])
    weight_words = read_binary(weight_path, 32, 9 * 8)
    weights = [[unpack(weight_words[k * 8 + co], 4) for co in range(8)] for k in range(9)]
    provenance = {str(p): sha256(p) for p in [case_path, *files, weight_path]}
    readme = folder / "README.txt"
    if readme.exists():
        provenance[str(readme)] = sha256(readme)
    metadata_path = resolve(case_path, cfg["weight_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("weight_shape_oihw") != [8, 8, 3, 3]:
        raise ValueError("handoff weight metadata has an unexpected shape")
    provenance[str(metadata_path)] = sha256(metadata_path)
    return dict(config=cfg, frames=frames, deltas=deltas, weights=weights,
                weight_words=weight_words, source_hashes=provenance,
                trained_model_sha256=metadata.get("model_sha256"))


def convolution(frame: list[int], weights: list[list[list[int]]], cfg: dict) -> list[list[int]]:
    """Direct CHW/OIHW cross-correlation, independent of the TB's gather schedule."""
    height, width = cfg["height"], cfg["width"]
    result = []
    for y in range(height):
        for x in range(width):
            values = [0] * 8
            for ky in range(3):
                iy = y + ky - 1
                if not 0 <= iy < height:
                    continue
                for kx in range(3):
                    ix = x + kx - 1
                    if not 0 <= ix < width:
                        continue
                    kernel_weights = weights[ky * 3 + kx]
                    for ci in range(8):
                        activation = frame[ci * height * width + iy * width + ix]
                        for co in range(8):
                            values[co] += activation * kernel_weights[co][ci]
            result.append(values)
    return result


def passes(data: dict, mode: str) -> list[dict]:
    result = []
    count = 4 if mode == "baseline" else 1
    for i, frame in enumerate(data["frames"][:count]):
        result += [dict(name=f"X{i}_high", frame=i, signed=False, values=[v >> 4 for v in frame]),
                   dict(name=f"X{i}_low", frame=i, signed=False, values=[v & 15 for v in frame])]
    if mode == "differential":
        result += [dict(name=f"D{i}_relative_X0", frame=i, signed=True, values=delta)
                   for i, delta in enumerate(data["deltas"], 1)]
    for p in result:
        p["expected"] = convolution(p["values"], data["weights"], data["config"])
    return result


def reconstruct(pass_results: list[list[list[int]]], mode: str) -> list[list[list[int]]]:
    def merge(high, low):
        return [[a + 16 * b for a, b in zip(lo, hi)] for lo, hi in zip(low, high)]
    if mode == "baseline":
        return [merge(pass_results[2 * i], pass_results[2 * i + 1]) for i in range(4)]
    base = merge(pass_results[0], pass_results[1])
    return [base] + [[[a + d for a, d in zip(b, delta)] for b, delta in zip(base, output)]
                     for output in pass_results[2:]]


def errors(actual, expected) -> dict:
    a = [v for frame in actual for row in frame for v in row]
    e = [v for frame in expected for row in frame for v in row]
    if len(a) != len(e):
        raise ValueError(f"result length mismatch: {len(a)} vs {len(e)}")
    difference = [x - y for x, y in zip(a, e)]
    return dict(scalar_count=len(a), mismatches=sum(v != 0 for v in difference),
                max_abs_error=max(map(abs, difference), default=0),
                mse=sum(v * v for v in difference) / len(difference),
                mean_abs_error=sum(map(abs, difference)) / len(difference),
                expected_min=min(e), expected_max=max(e), actual_min=min(a), actual_max=max(a))


def statistics(data: dict) -> dict:
    return dict(frames=[dict(frame=i, count=len(x), minimum=min(x), maximum=max(x))
                        for i, x in enumerate(data["frames"])],
                deltas=[dict(frame=i, reference_frame=0, count=len(x), minimum=min(x), maximum=max(x),
                             zeros=x.count(0), exact_minus8=x.count(-8), exact_plus7=x.count(7),
                             below_minus8=sum(v < -8 for v in x), above_plus7=sum(v > 7 for v in x),
                             clipping_changes=sum(not -8 <= v <= 7 for v in x))
                        for i, x in enumerate(data["deltas"], 1)])


def kernel_partials(values: list[int], weights, kernel_index: int, cfg: dict):
    """One 8-channel dot product per output, including zero halo contributions."""
    height, width = cfg["height"], cfg["width"]
    ky, kx = divmod(kernel_index, 3)
    result = []
    for position in range(height * width):
        oy, ox = divmod(position, width)
        iy, ix = oy + ky - 1, ox + kx - 1
        if not (0 <= iy < height and 0 <= ix < width):
            result.append([0] * 8)
        else:
            activation = [values[ci * height * width + iy * width + ix] for ci in range(8)]
            result.append([sum(a * w for a, w in zip(activation, weights[kernel_index][co]))
                           for co in range(8)])
    return result


def spatial_words(values: list[int], cfg: dict, signed: bool) -> list[int]:
    pixels = cfg["height"] * cfg["width"]
    return [pack([values[ci * pixels + p] for ci in range(8)], 4, signed) for p in range(pixels)]


def synthetic_case(name: str, frame_kind: str, template: dict) -> dict:
    """Small, explicitly synthetic signed/unsigned edge regression; never teacher evidence."""
    cfg = dict(template["config"], name=name, height=4, width=4, chunk=7)
    base = [0 if frame_kind == "zero" else (0, 255, 8, 247, 15, 240, 127, 128)[i % 8]
            for i in range(8 * 4 * 4)]
    frames = [base[:]]
    for frame_index in range(1, 4):
        if frame_kind == "zero":
            frames.append(base[:])
        else:
            delta = [7 if (i + frame_index) % 3 == 0 and v <= 248 else
                     -8 if (i + frame_index) % 3 == 1 and v >= 8 else 0
                     for i, v in enumerate(base)]
            frames.append([v + d for v, d in zip(base, delta)])
    return dict(config=cfg, frames=frames,
                deltas=[[x - y for x, y in zip(frame, base)] for frame in frames[1:]],
                weights=template["weights"], weight_words=template["weight_words"],
                source_hashes=dict(template["source_hashes"]),
                trained_model_sha256=template["trained_model_sha256"],
                synthetic=True, synthetic_description=frame_kind)


def prepare(data: dict, build: Path) -> dict:
    cfg = data["config"]
    build.mkdir(parents=True, exist_ok=True)
    conventional = passes(data, "baseline")
    differential = passes(data, "differential")
    direct = [convolution(frame, data["weights"], cfg) for frame in data["frames"]]
    for method, streams in (("conventional", conventional), ("differential", differential)):
        reconstructed = reconstruct([p["expected"] for p in streams],
                                    "baseline" if method == "conventional" else method)
        if errors(reconstructed, direct)["mismatches"]:
            raise ValueError(f"independent Python {method} decomposition is not exact")
    for filename, streams in (("conv_input.mem", conventional), ("diff_input.mem", differential)):
        write_binary(build / filename,
                     [word for p in streams for word in spatial_words(p["values"], cfg, p["signed"])], 32)
    write_binary(build / "weight.mem", data["weight_words"], 32)
    for method, streams in (("conv", conventional), ("diff", differential)):
        write_binary(build / f"expected_raw_{method}.mem",
                     [pack(row, 32, True) for k in range(9) for p in streams
                      for row in kernel_partials(p["values"], data["weights"], k, cfg)], 256)
    write_binary(build / "expected_pass_diff.mem",
                 [pack(row, 32, True) for p in differential for row in p["expected"]], 256)
    write_binary(build / "expected_frames.mem", [pack(row, 32, True) for frame in direct for row in frame], 256)
    params = {"ACT_H": cfg["height"], "ACT_W": cfg["width"], "KERNEL": cfg["kernel"],
              "PAD": cfg["padding"], "STRIDE": cfg["stride"], "CHUNK": cfg["chunk"]}
    (build / "video_config.vh").write_text(
        "".join(f"localparam integer {key} = {value};\n" for key, value in params.items()), encoding="ascii")
    pixels = cfg["height"] * cfg["width"]
    manifest = dict(
        status="PREPARED", config=cfg, synthetic=data.get("synthetic", False),
        source_hashes=data["source_hashes"], trained_model_sha256=data["trained_model_sha256"],
        input_statistics=statistics(data), scalar_outputs_per_method=4 * pixels * 8,
        expected_raw_domain="kernel-position 8-channel dot products; signed 32-bit lanes",
        expected_output_domain="full raw integer convolution before ReLU; signed 32-bit lanes",
        pass_order={"conventional": [p["name"] for p in conventional],
                    "differential": [p["name"] for p in differential]},
        storage_order="pass, spatial row-major; channel0 in least significant bits; source frames CHW",
        raw_oracle_order="kernel row-major, pass, output spatial row-major",
        measurement_contract="all stage<90 work including resets/load/gather/compute/writeback/reconstruction; stage90 verification excluded",
        host_preparation=dict(delta_subtractions=3 * pixels * 8, original_four_frame_bytes=4 * pixels * 8,
                              delta_4bit_bytes=3 * pixels * 8 // 2,
                              differential_stream_bytes_per_kernel=5 * pixels * 8 // 2,
                              conventional_stream_bytes_per_kernel=8 * pixels * 8 // 2,
                              exclusion="host subtraction, nibble splitting, oracle generation and file IO are outside RTL cycles; no host arithmetic reconstructs observed RTL outputs"),
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
               for p in sorted(build.glob("*.mem"))},
        python_decomposition_checks={"conventional": errors(reconstruct([p["expected"] for p in conventional], "baseline"), direct),
                                     "differential": errors(reconstruct([p["expected"] for p in differential], "differential"), direct)})
    dump_json(build / "preparation.json", manifest)
    return dict(manifest=manifest, direct=direct)


def main_runner():
    spec = importlib.util.spec_from_file_location("_ece284_part3_runner", PART3 / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Part3 compile helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_name(value: str) -> str:
    names = {"0": "conventional", "conv": "conventional", "baseline": "conventional",
             "conventional": "conventional", "1": "differential", "diff": "differential",
             "differential": "differential"}
    if value not in names:
        raise ValueError(f"unknown method label {value!r}")
    return names[value]


def read_metrics(build: Path, cfg: dict) -> dict:
    with (build / "cycles.csv").open(newline="", encoding="utf-8-sig") as stream:
        cycles = list(csv.DictReader(stream))
    with (build / "loads.csv").open(newline="", encoding="utf-8-sig") as stream:
        loads = list(csv.DictReader(stream))
    measured = {name: 0 for name in ("conventional", "differential")}
    verification = measured.copy()
    by_stage = {name: {} for name in measured}
    if not cycles or not loads:
        raise ValueError("missing cycle/load records")
    for row in cycles:
        name, stage, count = method_name(row["method"]), int(row["stage"]), int(row["cycles"])
        if count < 0:
            raise ValueError("negative cycle count")
        (measured if stage < 90 else verification)[name] += count
        by_stage[name][str(stage)] = by_stage[name].get(str(stage), 0) + count
    for name, pass_count in (("conventional", 8), ("differential", 5)):
        selected = [row for row in loads if method_name(row["method"]) == name]
        if len(selected) != 9 or {int(row["kernel"]) for row in selected} != set(range(9)):
            raise ValueError(f"{name}: expected one weight-load record per kernel")
        for row in selected:
            wanted = {"load_cycles": 8, "load_sessions": 1, "passes": pass_count,
                      "tile_executions": pass_count * math.ceil(cfg["height"] * cfg["width"] / cfg["chunk"])}
            for field, expected in wanted.items():
                if int(row[field]) != expected:
                    raise ValueError(f"{name} kernel{row['kernel']}: {field}={row[field]}, expected {expected}")
        if measured[name] <= 0:
            raise ValueError(f"{name}: no measured RTL work")
    return dict(work_cycles=measured, verification_cycles=verification, stage_cycles=by_stage,
                conventional_over_differential_cycles=measured["conventional"] / measured["differential"],
                theoretical_pass_work_ratio=8 / 5, load_records=loads, cycle_records=cycles,
                disclaimer="Simulation cycles only; no FPGA frequency, power, measured energy or classification-accuracy claim.")


def sources() -> list[Path]:
    return [ROOT / line.strip() for line in (ROOT / "filelist.f").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")] + [ROOT / "tb/video_tb.v"]


def compile_simulate(args, build: Path, source_files: list[Path]) -> str:
    runner = main_runner()
    if args.waves and args.sim == "vivado":
        # The shared helper hard-codes its original core_tb signal list.  Use its
        # command/error handling with the same compile flags, then simulate once
        # with this top's explicit VCD signal list.
        prefix = ["cmd.exe", "/d", "/c"] if sys.platform == "win32" else []
        suffix = ".bat" if sys.platform == "win32" else ""
        bindir = Path(args.vivado_bin)
        runner.command(prefix + [str(bindir / ("xvlog" + suffix)), "--sv", "-i", str(build),
                                 *map(str, source_files)], build, build / "compile.log")
        runner.command(prefix + [str(bindir / ("xelab" + suffix)), "video_tb", "-s", "video_tb_sim",
                                 "--debug", "typical", "--timescale", "1ns/1ps"],
                       build, build / "elaborate.log")
        names = ["clk", "reset", "compute_clear", "act_signed", "method", "phase",
                 "kernel_index", "pass_index", "chunk_start", "inst", "ofifo_valid",
                 "sfp_out", "pmem_read_data", "sfp_shift", "weight_load_cycles", "weight_load_sessions",
                 "l0_wr", "l0_rd", "execute", "load", "ofifo_rd", "acc", "sfp_clear",
                 "pmem_from_sfp", "A_xmem", "A_pmem", "D_xmem"]
        signals = " ".join(f"/video_tb/{name}" for name in names)
        script = build / "video_waves.tcl"
        script.write_text(f"open_vcd video_tb.vcd\nlog_vcd {signals}\nrun all\nclose_vcd\nquit\n", encoding="ascii")
        return runner.command(prefix + [str(bindir / ("xsim" + suffix)), "video_tb_sim",
                                         "-tclbatch", script.as_posix()], build, build / "simulation.log")
    return runner.compile_and_run(args, "video_tb", source_files, build, waves=args.waves)


def simulate(data: dict, args) -> dict:
    cfg = data["config"]
    build = args.build_dir / cfg["name"]
    prepared = prepare(data, build)
    summary = dict(prepared["manifest"])
    if args.prepare_only:
        dump_json(build / "summary.json", summary)
        print(f"PREPARED {cfg['name']}: 8 conventional / 5 differential passes; exact Python identities")
        return summary
    summary["status"] = "RUNNING"
    dump_json(build / "summary.json", summary)
    source_files = sources()
    hash_paths = [*source_files, ROOT / "filelist.f", Path(__file__), PART3 / "run.py"]
    source_hashes = {str(p): sha256(p) for p in hash_paths}
    try:
        # Remove expected output names before running so a failed simulation cannot reuse old results.
        for name in ("conventional", "differential"):
            for i in range(4):
                (build / f"{name}_frame{i}.txt").unlink(missing_ok=True)
        for filename in ("cycles.csv", "loads.csv"):
            (build / filename).unlink(missing_ok=True)
        output = compile_simulate(args, build, source_files)
        if "PASS VIDEO" not in output:
            raise RuntimeError("video TB did not report PASS VIDEO completion")
        if any(sha256(p) != digest for p, digest in ((Path(p), h) for p, h in source_hashes.items())):
            raise RuntimeError("RTL/runner sources changed during the simulation")
        results = {}
        for name in ("conventional", "differential"):
            actual = [[unpack(word, 32) for word in read_binary(build / f"{name}_frame{i}.txt", 256,
                                                               cfg["height"] * cfg["width"])] for i in range(4)]
            results[name] = errors(actual, prepared["direct"])
            results[name]["per_frame"] = [dict(frame=i, **errors([frame], [prepared["direct"][i]]))
                                           for i, frame in enumerate(actual)]
            if results[name]["mismatches"]:
                raise RuntimeError(f"{name}: nonzero raw-convolution mismatches: {results[name]}")
        summary.update(status="PASS", simulator=args.sim, rtl_source_hashes=source_hashes,
                       results=results, metrics=read_metrics(build, cfg))
        dump_json(build / "summary.json", summary)
        print(f"PASS {cfg['name']}: each method checked {4*cfg['height']*cfg['width']*8} raw scalars exactly")
        return summary
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        summary.update(status="FAIL", error=str(error), rtl_source_hashes=source_hashes)
        dump_json(build / "summary.json", summary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases/teacher.json")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--prepare-only", action="store_true", help="generate mem/oracles without compiling or simulating")
    parser.add_argument("--regress", action="store_true", help="also run explicitly synthetic unsigned-edge and zero-delta cases")
    parser.add_argument("--waves", action="store_true")
    parser.add_argument("--sim", choices=("auto", "vivado", "iverilog"), default="auto")
    parser.add_argument("--vivado-bin")
    parser.add_argument("--iverilog", default=shutil.which("iverilog"))
    parser.add_argument("--vvp", default=shutil.which("vvp"))
    args = parser.parse_args()
    args.case, args.build_dir = args.case.resolve(), args.build_dir.resolve()
    if not args.prepare_only:
        try:
            main_runner().configure_simulator(args)
        except ValueError as error:
            parser.error(str(error))
    args.build_dir.mkdir(parents=True, exist_ok=True)
    report = args.build_dir / "regression.json"
    dump_json(report, dict(status="RUNNING"))
    try:
        data = load_case(args.case)
        cases = [data]
        if args.regress:
            cases += [synthetic_case("unsigned_edges", "edges", data), synthetic_case("zero_delta", "zero", data)]
        summaries = [simulate(case, args) for case in cases]
        dump_json(report, dict(status="PREPARED" if args.prepare_only else "PASS", cases=summaries))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        dump_json(report, dict(status="FAIL", error=str(error)))
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

