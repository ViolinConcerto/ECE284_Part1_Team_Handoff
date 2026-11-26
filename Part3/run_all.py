"""Run the handoff import, arithmetic checks and both RTL workloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import run


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regress", action="store_true")
    parser.add_argument("--waves", action="store_true")
    parser.add_argument("--sim", choices=("auto", "vivado", "iverilog"), default="auto")
    parser.add_argument("--vivado-bin")
    args = parser.parse_args()
    args.iverilog, args.vvp = shutil.which("iverilog"), shutil.which("vvp")
    try:
        run.configure_simulator(args)
    except ValueError as error:
        parser.error(str(error))
    report_path = ROOT / "build/final_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report = dict(status="RUNNING", started_at_utc=datetime.now(timezone.utc).isoformat())

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def execute(script, extra=()):
        subprocess.run([sys.executable, "-u", str(ROOT / script), *extra], cwd=ROOT, check=True)

    save()
    try:
        execute("import_handoff.py")
        execute("video/tests/test_vectors.py")
        unit_build = ROOT / "video/build/unit"
        unit_build.mkdir(parents=True, exist_ok=True)
        unit_sources = [ROOT / "video/rtl" / name for name in ("mac.v", "mac_tile.v", "sfp.v")]
        unit_sources.append(ROOT / "video/tests/rtl_units_tb.v")
        output = run.compile_and_run(args, "rtl_units_tb", unit_sources, unit_build)
        if "PASS video RTL units" not in output:
            raise RuntimeError("video arithmetic unit tests did not finish")
        print("PASS video RTL units: arithmetic, weight retention, SFU")
        flags = ["--sim", args.sim]
        if args.sim == "vivado":
            flags += ["--vivado-bin", args.vivado_bin]
        if args.regress:
            flags.append("--regress")
        if args.waves:
            flags.append("--waves")
        execute("run.py", flags)
        execute("video/run_video.py", flags)
        vanilla = json.loads((ROOT / "build/regression.json").read_text(encoding="utf-8"))
        video = json.loads((ROOT / "video/build/regression.json").read_text(encoding="utf-8"))
        if vanilla["status"] != "PASS" or video["status"] != "PASS":
            raise RuntimeError("a component report did not pass")
        files = [*ROOT.glob("*.py"), *ROOT.glob("tb/*.v"), *ROOT.glob("reference_dut/*.v"),
                 *ROOT.glob("video/*.py"), *ROOT.glob("video/rtl/*.v"), *ROOT.glob("video/tb/*.v"),
                 *ROOT.glob("video/tests/*.py"), *ROOT.glob("video/tests/*.v")]
        report.update(status="PASS", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      simulator=args.sim, vanilla=vanilla, video=video,
                      source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(files)},
                      scope="Behavioral RTL simulation; no new training, Quartus fitting, FPGA frequency or power measurement.")
        save()
        print(f"PASS all checks; {report_path}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status="FAIL", error=str(error))
        save()
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
