"""Prepare integer vectors and run the seven-stage vanilla core testbench."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def read_words(path: Path, width: int, count: int) -> list[int]:
    words = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if not re.fullmatch(f"[01]{{{width}}}", line):
            raise ValueError(f"{path}:{number}: expected {width} binary digits")
        words.append(int(line, 2))
    if len(words) != count:
        raise ValueError(f"{path}: expected {count} words, found {len(words)}")
    return words


def write_words(path: Path, words: list[int], width: int) -> None:
    path.write_text("".join(f"{word:0{width}b}\n" for word in words), encoding="ascii")


def pack(values: list[int], bits: int) -> int:
    mask = (1 << bits) - 1
    return sum((value & mask) << (lane * bits) for lane, value in enumerate(values))


def unpack(word: int, bits: int, signed: bool = False) -> list[int]:
    values = [(word >> (lane * bits)) & ((1 << bits) - 1) for lane in range(8)]
    return [v - (1 << bits) if signed and v >= (1 << (bits - 1)) else v for v in values]


def dimensions(cfg: dict) -> tuple[int, int, int, int]:
    for key, expected in (("input_channels", 8), ("output_channels", 8),
                          ("activation_bits", 4), ("weight_bits", 4), ("psum_bits", 16)):
        if cfg.get(key) != expected:
            raise ValueError(f"reference DUT requires {key}={expected}")
    if cfg.get("groups", 1) != 1 or cfg.get("dilation", 1) != 1 or cfg.get("bias", False) is not False:
        raise ValueError("reference DUT supports groups=1, dilation=1, bias=false only")
    for key in ("height", "width", "kernel", "stride", "padding"):
        if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "padding" else 1):
            raise ValueError(f"invalid {key}: {cfg.get(key)!r}")
    ph, pw = cfg["height"] + 2 * cfg["padding"], cfg["width"] + 2 * cfg["padding"]
    oh = (ph - cfg["kernel"]) // cfg["stride"] + 1
    ow = (pw - cfg["kernel"]) // cfg["stride"] + 1
    if min(oh, ow) < 1:
        raise ValueError("kernel is larger than the padded input")
    if ph * pw > 64:
        raise ValueError("reference L0/OFIFO capacity is 64 words; larger layers need a new schedule")
    if cfg["kernel"] ** 2 * ph * pw + 2 * oh * ow > 2048:
        raise ValueError("psum address map exceeds the 2048-word SRAM")
    if 1024 + cfg["kernel"] ** 2 * 8 > 2048:
        raise ValueError("weight region exceeds input SRAM")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("name", "")):
        raise ValueError("case name must contain only letters, digits, '_' and '-'")
    return ph, pw, oh, ow


def integer_reference(cfg: dict, inputs: list[int], weights: list[int]):
    ph, pw, oh, ow = dimensions(cfg)
    x = [unpack(word, 4) for word in inputs]
    w = [unpack(word, 4, signed=True) for word in weights]
    pad, kernel, stride = cfg["padding"], cfg["kernel"], cfg["stride"]
    for y in range(ph):
        for z in range(pw):
            if y < pad or y >= ph - pad or z < pad or z >= pw - pad:
                if any(x[y * pw + z]):
                    raise ValueError("input.txt padding must contain zeros")
    raw = [[sum(x[pos][c] * w[k * 8 + o][c] for c in range(8))
            for o in range(8)] for k in range(kernel * kernel) for pos in range(ph * pw)]
    sums = []
    for y in range(oh):
        for z in range(ow):
            values = []
            for o in range(8):
                running = 0
                for ky in range(kernel):
                    for kx in range(kernel):
                        running += sum(x[(y*stride+ky)*pw+z*stride+kx][c]
                                       * w[(ky*kernel+kx)*8+o][c] for c in range(8))
                        if not -32768 <= running <= 32767:
                            raise ValueError("case overflows the signed 16-bit SFU accumulator")
                values.append(running)
            sums.append(values)
    if any(not -32768 <= value <= 32767 for row in raw + sums for value in row):
        raise ValueError("case overflows signed 16-bit psum; no silent truncation is allowed")
    return ([pack(row, 16) for row in raw], [pack(row, 16) for row in sums],
            [pack([max(0, value) for value in row], 16) for row in sums])


def prepare(case: Path, repeats: int) -> tuple[dict, Path]:
    cfg = json.loads((case / "case.json").read_text(encoding="utf-8-sig"))
    ph, pw, oh, ow = dimensions(cfg)
    inputs = read_words(case / "input.txt", 32, ph * pw)
    weights = read_words(case / "weight.txt", 32, cfg["kernel"] ** 2 * 8)
    expected = read_words(case / "psum.txt", 128, oh * ow)
    raw, sums, calculated = integer_reference(cfg, inputs, weights)
    if expected != calculated:
        pos = next(i for i, (a, b) in enumerate(zip(expected, calculated)) if a != b)
        raise ValueError(f"supplied golden disagrees with integer convolution at output word {pos}")
    build = ROOT / "build" / cfg["name"]
    build.mkdir(parents=True, exist_ok=True)
    for name, values, bits in (("input.mem", inputs, 32), ("weight.mem", weights, 32),
                               ("expected_raw.mem", raw, 128), ("expected_sum.mem", sums, 128),
                               ("expected_relu.mem", expected, 128)):
        write_words(build / name, values, bits)
    params = {"ACT_H": cfg["height"], "ACT_W": cfg["width"], "KERNEL": cfg["kernel"],
              "PAD": cfg["padding"], "STRIDE": cfg["stride"], "REPEATS": repeats}
    (build / "case_config.vh").write_text(
        "".join(f"localparam integer {key} = {value};\n" for key, value in params.items()), encoding="ascii")
    (build / "case.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg, build


def synthetic(name: str, height: int, width: int, kernel: int, pad: int, seed: int,
              stress: bool = False) -> Path:
    case = ROOT / "build" / "generated_cases" / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = dict(name=name, height=height, width=width, kernel=kernel, padding=pad, stride=1,
               input_channels=8, output_channels=8, activation_bits=4, weight_bits=4,
               psum_bits=16, origin=f"synthetic integer regression, seed={seed}")
    rng = random.Random(seed)
    ph, pw, _, _ = dimensions(cfg)
    inputs = []
    for y in range(ph):
        for x in range(pw):
            values = [0] * 8 if y < pad or y >= ph - pad or x < pad or x >= pw - pad else [
                (15 if stress else rng.randrange(16)) for _ in range(8)]
            inputs.append(pack(values, 4))
    weights = []
    for k in range(kernel * kernel):
        for o in range(8):
            if stress:
                # Include negative sums, full unsigned activations, and two's-complement -8.
                values = [[-8, 7, 0, -7, 1, -1, 6, -6][o]] * 8
            else:
                values = [rng.randrange(-8, 8) for _ in range(8)]
            weights.append(pack(values, 4))
    _, _, golden = integer_reference(cfg, inputs, weights)
    (case / "case.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for filename, values, bits in (("input.txt", inputs, 32), ("weight.txt", weights, 32),
                                    ("psum.txt", golden, 128)):
        write_words(case / filename, values, bits)
    return case


def command(args: list[str], cwd: Path, logfile: Path) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    logfile.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        print(result.stdout, end="")
        raise RuntimeError(f"command failed ({result.returncode}); see {logfile}")
    return result.stdout


def configure_simulator(args) -> None:
    suffix = ".bat" if sys.platform == "win32" else ""
    if args.sim != "iverilog":
        if args.vivado_bin:
            args.vivado_bin = str(Path(args.vivado_bin).resolve())
        else:
            executable = shutil.which("xvlog" + suffix)
            local_vivado = Path(r"E:\Xilinx\Vivado\2017.4\bin")
            if executable:
                args.vivado_bin = str(Path(executable).resolve().parent)
            elif sys.platform == "win32" and (local_vivado / "xvlog.bat").is_file():
                args.vivado_bin = str(local_vivado)
        if args.vivado_bin:
            missing = [tool for tool in ("xvlog", "xelab", "xsim")
                       if not (Path(args.vivado_bin) / (tool + suffix)).is_file()]
            if missing:
                raise ValueError(f"Missing {', '.join(missing)} in Vivado bin: {args.vivado_bin}")
        if args.sim == "vivado" and not args.vivado_bin:
            raise ValueError("Vivado not found; use --vivado-bin to select its bin directory")
        if args.sim == "auto":
            args.sim = "vivado" if args.vivado_bin else "iverilog"
    if args.sim == "iverilog" and (not args.iverilog or not args.vvp):
        raise ValueError("Install Icarus Verilog or use --sim vivado --vivado-bin <directory>")


def compile_and_run(args, top: str, sources: list[Path], build: Path, waves: bool = False) -> str:
    if args.sim == "vivado":
        bindir = Path(args.vivado_bin)
        prefix = ["cmd.exe", "/d", "/c"] if sys.platform == "win32" else []
        suffix = ".bat" if sys.platform == "win32" else ""
        command(prefix + [str(bindir / ("xvlog"+suffix)), "--sv", "-i", str(build),
                          *map(str, sources)], build, build / "compile.log")
        command(prefix + [str(bindir / ("xelab"+suffix)), top, "-s", top+"_sim",
                          "--debug", "typical", "--timescale", "1ns/1ps"],
                build, build / "elaborate.log")
        options = ["-runall"]
        if waves:
            script = build / "waves.tcl"
            names = ["clk", "reset", "phase", "job", "inst", "A_xmem", "D_xmem",
                     "A_pmem", "pmem_read_data", "sfp_out", "ofifo_valid", "ofifo_rd",
                     "l0_rd", "l0_wr", "execute", "load", "acc", "sfp_clear",
                     "sfp_relu", "pmem_from_sfp", "CEN_xmem", "WEN_xmem", "CEN_pmem", "WEN_pmem"]
            signals = " ".join(f"/{top}/{name}" for name in names)
            script.write_text(f"open_vcd core_tb.vcd\nlog_vcd {signals}\nrun all\nclose_vcd\nquit\n", encoding="ascii")
            options = ["-tclbatch", script.as_posix()]
        return command(prefix + [str(bindir / ("xsim"+suffix)), top+"_sim", *options],
                       build, build / "simulation.log")
    compiled = build / (top+".vvp")
    command([args.iverilog, "-g2012", "-Wall", "-s", top, "-I", build.as_posix(),
             "-o", compiled.as_posix(), *[s.as_posix() for s in sources]], ROOT, build / "compile.log")
    return command([args.vvp, str(compiled)] + (["+WAVES"] if waves else []),
                   build, build / "simulation.log")


def simulate(case: Path, args) -> dict:
    cfg, build = prepare(case, args.repeat)
    sources = [ROOT / line.strip() for line in (ROOT / "filelist.f").read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("//")]
    output = compile_and_run(args, "core_tb", sources+[ROOT / "tb/core_tb.v"], build, args.waves)
    if "PASS Part3" not in output:
        raise RuntimeError(f"simulation did not finish successfully: {build}")
    _, _, oh, ow = dimensions(cfg)
    expected = read_words(build / "expected_relu.mem", 128, oh * ow)
    for job in range(args.repeat):
        actual = read_words(build / f"result_{job}.txt", 128, oh * ow)
        if actual != expected:
            raise RuntimeError(f"exported SRAM results disagree for job {job}")
    with (build / "cycles.csv").open(newline="") as stream:
        cycles = list(csv.DictReader(stream))
    summary = dict(case=cfg["name"], status="PASS", simulator=args.sim, repeats=args.repeat,
                   output_words_per_job=oh * ow, scalar_outputs_per_job=oh * ow * 8,
                   raw_words_per_job=cfg["kernel"] ** 2 * (cfg["height"]+2*cfg["padding"])
                                     * (cfg["width"]+2*cfg["padding"]),
                   cycles=cycles, golden_source=str(case / "psum.txt"))
    (build / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {cfg['name']}: {oh*ow*8} scalar outputs/job, {args.repeat} jobs; {build}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases/team_handoff")
    parser.add_argument("--regress", action="store_true", help="reference + signed limits + shape cases")
    parser.add_argument("--waves", action="store_true", help="write a VCD for each case")
    parser.add_argument("--repeat", type=int, default=2, help="independent jobs in each simulation")
    parser.add_argument("--sim", choices=("auto", "vivado", "iverilog"), default="auto")
    parser.add_argument("--vivado-bin")
    parser.add_argument("--iverilog", default=shutil.which("iverilog"))
    parser.add_argument("--vvp", default=shutil.which("vvp"))
    args = parser.parse_args()
    try:
        configure_simulator(args)
    except ValueError as error:
        parser.error(str(error))
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    report = ROOT / "build/regression.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(status="RUNNING", simulator=args.sim)) + "\n", encoding="utf-8")
    try:
        unit = ROOT / "build/sfp_unit"
        unit.mkdir(parents=True, exist_ok=True)
        text = compile_and_run(args, "sfp_tb", [ROOT / "reference_dut/sfp.v", ROOT / "tb/sfp_tb.v"], unit)
        if "PASS SFP" not in text:
            raise RuntimeError("SFP unit test did not complete")
        print("PASS SFP: signed accumulation, hold, clear, ReLU")
        cases = [args.case.resolve()]
        if args.regress:
            cases += [synthetic("signed_limits", 4, 4, 3, 1, 284, stress=True),
                      synthetic("rectangle", 3, 5, 3, 1, 285),
                      synthetic("kernel1", 3, 3, 1, 0, 286),
                      synthetic("fifo_capacity", 6, 6, 3, 1, 287)]
        summaries = [simulate(case, args) for case in cases]
        report.write_text(json.dumps(
            dict(status="PASS", cases=summaries), indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.write_text(json.dumps(dict(status="FAIL", error=str(error)), indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

