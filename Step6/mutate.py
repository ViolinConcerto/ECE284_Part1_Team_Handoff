"""Mutation testing (alpha direction 3): inject single-point RTL faults into a copy of Step2
and check that the Step3 regression rejects each one, in the baseline and in the full alpha mode."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "Step2"
RUN = ROOT / "Step3" / "run.py"
MODES = {"baseline": [], "alpha": ["--pipeline-l0", "--stream-ofifo", "--skip-zero", "--pipeline-sfu"]}

# id, file under Step2, original text (must occur once), mutated text, fault class
MUTANTS = [
    ("M01", "course/mac.v", "assign a_pad = {1'b0, a};", "assign a_pad = {a[bw-1], a};",
     "MAC: activation sign-extended instead of zero-extended"),
    ("M02", "course/mac.v", "assign psum = product + c;", "assign psum = product;",
     "MAC: north partial sum dropped"),
    ("M03", "homework/mac_tile.v", "assign out_e = a_q;", "assign out_e = in_w;",
     "PE: east activation bypasses its register"),
    ("M04", "homework/mac_tile.v", "a_q <= in_w;", "if (inst_w[1]) a_q <= in_w;",
     "PE: activation register frozen during weight load"),
    ("M05", "homework/mac_tile.v", "load_ready_q <= 1'b0;", "load_ready_q <= 1'b1;",
     "PE: weight overwritten by later words"),
    ("M06", "homework/mac_tile.v", "inst_q[1] <= inst_w[1];", "inst_q[1] <= inst_w[0];",
     "PE: valid derived from load instead of execute"),
    ("M07", "homework/mac_row.v", "assign valid[WEST] = inst_temp[2*i + 1];",
     "assign valid[WEST] = inst_temp[2*WEST + 1];", "Row: valid tapped one column early"),
    ("M08", "course/mac_array.v", "assign out_s = temp[psum_bw*col*9-1:psum_bw*col*8];",
     "assign out_s = temp[psum_bw*col*8-1:psum_bw*col*7];", "Array: last input-channel row skipped"),
    ("M09", "course/mac_array.v", "inst_w_temp[3:2]   <= inst_w_temp[1:0];",
     "inst_w_temp[3:2]   <= inst_w;", "Array: instruction skew broken at row 2"),
    ("M10", "homework/l0.v", "rd_en <= (rd_en << 1) | rd;", "rd_en <= {row{rd}};",
     "L0: rows read without stagger"),
    ("M11", "homework/l0.v", "assign o_ready = ~|full;", "assign o_ready = 1'b1;",
     "L0: back-pressure (ready) lost"),
    ("M12", "homework/l0.v", ".in(in[i*bw +: bw])", ".in(in[(row-1-i)*bw +: bw])",
     "L0: input-channel order reversed"),
    ("M13", "homework/ofifo.v", "assign o_valid = ~|empty;", "assign o_valid = ~&empty;",
     "OFIFO: valid when any lane has data"),
    ("M14", "homework/ofifo.v", ".wr(wr[i]), .rd(rd_en),", ".wr(wr[i]), .rd(rd),",
     "OFIFO: read request not registered"),
    ("M15", "course/fifo_depth64.v", "assign full  = ((wr_ptr[5:0] == rd_ptr[5:0]) && (wr_ptr[6] != rd_ptr[6]))",
     "assign full  = ((wr_ptr[5:0] == rd_ptr[5:0]))", "FIFO64: empty reported as full"),
    ("M16", "course/fifo_depth64.v", "else if ((rd == 1) && (empty == 0)) begin",
     "else if (rd == 1) begin", "FIFO64: read underflow not blocked"),
    ("M17", "rtl/ofifo_stream.v", "wire pop = rd && o_valid;", "wire pop = rd;",
     "Stream OFIFO: pop without aligned valid"),
    ("M18", "rtl/ofifo_stream.v",
     "assign full[c] = head == tail && read_ptr[ADDR_BITS] != write_ptr[ADDR_BITS];",
     "assign full[c] = 1'b0;", "Stream OFIFO: overflow unprotected"),
    ("M19", "rtl/ofifo_stream.v", "if (wr[c] && (!full[c] || pop)) begin",
     "if (wr[c] && !full[c]) begin", "Stream OFIFO: write dropped when full and popping"),
    ("M20", "rtl/sfp.v", "accumulator <= accumulator + in;",
     "accumulator <= (accumulator + in < 0) ? 0 : accumulator + in;",
     "SFU: ReLU applied before the sum is complete"),
    ("M21", "rtl/sfp.v", "else if (clear)", "else if (1'b0)", "SFU: clear ignored"),
    ("M22", "rtl/sfp.v", "else if (relu && accumulator[psum_bw-1])",
     "else if (accumulator[psum_bw-1])", "SFU: ReLU whenever idle (no hold state)"),
    ("M23", "rtl/sfp.v", "else if (relu && accumulator[psum_bw-1])",
     "else if (relu && accumulator[psum_bw-2])", "SFU: ReLU tests the wrong sign bit"),
    ("M24", "rtl/corelet.v", "assign pmem_write_data = pmem_from_sfp ? sfp_out : ofifo_out;",
     "assign pmem_write_data = pmem_from_sfp ? ofifo_out : sfp_out;", "Corelet: writeback source swapped"),
    ("M25", "rtl/core.v", ".A(inst[30:20])", ".A({inst[30:21], 1'b0})",
     "Core: psum SRAM address LSB stuck at 0"),
    ("M26", "rtl/core.v", "assign xmem_read_data = input_word;",
     "assign xmem_read_data = 32'hFFFFFFFF;", "Core: occupancy bitmap read path broken"),
    ("M27", "course/fifo_depth64.v", "if ((wr == 1) && (full == 0)) begin // ECE284 fix",
     "if (wr == 1) begin // ECE284 fix", "FIFO64: write while full overwrites the head (original course code)"),
]
# Semantically equivalent mutants are excluded from the score, with the reason.
EQUIVALENT = {"M12": "weights and activations share the L0 lanes, so reversing both permutes the "
                     "input-channel sum without changing any dot product"}


def run_mode(rtl: Path, build: Path, mode: str, sim: list[str]) -> tuple[str, str]:
    command = [sys.executable, str(RUN), "--regress", "--repeat", "1", "--rtl", str(rtl),
               "--build-root", str(build), *sim, *MODES[mode]]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return "killed", "timeout"
    if result.returncode == 0:
        return "survived", ""
    lines = (result.stdout + result.stderr).splitlines()
    fatal = next((l for l in lines if "FATAL" in l or "mismatch" in l or "differs" in l), "")
    reason = fatal or next((l for l in reversed(lines) if l.startswith("FAIL")), "failed")
    for root in {str(ROOT), ROOT.as_posix()}:  # keep the report free of machine-specific paths
        reason = reason.replace(root + ("\\" if "\\" in root else "/"), "")
    if "compile.log" in reason:
        return "invalid", reason
    return "killed", re.sub(r"\s+", " ", reason)[:160]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="iverilog", choices=("iverilog", "vivado"))
    parser.add_argument("--vivado-bin")
    parser.add_argument("--only", nargs="*", help="mutant ids to run")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/mutation.json")
    args = parser.parse_args()
    sim = ["--sim", args.sim] + (["--vivado-bin", args.vivado_bin] if args.vivado_bin else [])
    selected = [m for m in MUTANTS if not args.only or m[0] in args.only]
    with tempfile.TemporaryDirectory(prefix="mutation_") as tmp:
        tmp = Path(tmp)
        for mode in MODES:  # the unmodified RTL must pass first
            status, reason = run_mode(STEP2, tmp / "golden", mode, sim)
            if status != "survived":
                print(f"FAIL: unmodified RTL does not pass ({mode}): {reason}", file=sys.stderr)
                return 1
        results = []
        for mid, rel, original, mutated, description in selected:
            rtl = tmp / mid / "Step2"
            shutil.copytree(STEP2, rtl)
            source = (rtl / rel).read_text(encoding="utf-8")
            if source.count(original) != 1:
                raise ValueError(f"{mid}: pattern must occur once in {rel}")
            (rtl / rel).write_text(source.replace(original, mutated), encoding="utf-8")
            outcome = {mode: run_mode(rtl, tmp / mid / "build", mode, sim) for mode in MODES}
            killed = any(status == "killed" for status, _ in outcome.values())
            results.append(dict(id=mid, file=rel, fault=description, killed=killed,
                                **{mode: status for mode, (status, _) in outcome.items()},
                                evidence=next((r for s, r in outcome.values() if s == "killed"), "")))
            print(f"{mid} {'KILLED  ' if killed else 'SURVIVED'} "
                  f"{' '.join(f'{m}={s}' for m, (s, _) in outcome.items())}  {description}", flush=True)
            shutil.rmtree(tmp / mid, ignore_errors=True)
    for r in results:
        r["equivalent"] = EQUIVALENT.get(r["id"], "")
    scored = [r for r in results if not r["equivalent"]]
    killed = sum(r["killed"] for r in scored)
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  simulator=args.sim, modes=MODES, mutants=len(results),
                  equivalent=len(results) - len(scored), killed=killed,
                  mutation_score=round(killed / len(scored), 4) if scored else None, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"mutation score {killed}/{len(scored)} (+{len(results) - len(scored)} equivalent); {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
