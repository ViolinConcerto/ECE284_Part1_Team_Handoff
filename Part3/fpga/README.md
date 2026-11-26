# Vanilla corelet: Quartus handoff

This is an authored project template, not a Quartus result. No Quartus executable was found on PATH or under the checked C:/D:/E: `intelFPGA` and `intelFPGA_lite` roots. Quartus was not installed or run. Frequency, timing closure, resources, and power remain unmeasured.

The top is `../reference_dut/corelet.v`, with its default 8×8 array, 4-bit activations/weights, and 16-bit psums. The QSF lists 12 RTL files. It excludes `core.v`, both behavioral SRAMs, every testbench, and `../video/rtl`. This handoff measures the vanilla corelet; it does not measure the differential video implementation or a complete system including SRAM.

## Fixed project choices

| Item | Setting |
|---|---|
| Project/revision | `vanilla_corelet` |
| Family | Cyclone IV GX |
| Requested device | `EP4CGX150DF31I7AD` |
| Initial clock constraint | 10.000 ns / 100 MHz |
| Fitter seed | 1 |
| Default input I/O activity | 20% |
| Activity input files | Disabled; no VCD/SAF is attached |

The QPF version identifies the intended 19.1 project format, not a successful 19.1 run. Use the instructor-approved Quartus edition/version with Cyclone IV GX device support. If another version is allowed, record that version and any QSF changes. If the exact part is unavailable, check the device package and the instructor's part name; do not silently substitute another device.

No physical pin locations, I/O standards, board loads, or external interface delays have been invented. Quartus may require additional choices for fitting or power. Record the resulting I/O standards, pin/virtual-pin treatment, voltage, temperature, cooling, and power model. Do not program a board using auto-assigned pins. Changing to virtual pins or a wrapper changes the implementation/power assumptions and must be declared.

## Run on a machine with Quartus

1. Open `vanilla_corelet.qpf` in Quartus. In Assignments → Device, check the exact family and part. Check the file list and top-level entity.
2. Inspect `vanilla_corelet.sdc`. Keep 10 ns for the first run, or deliberately change `CORELET_CLOCK_PERIOD_NS` and record it. It is a fitting constraint, not an achieved frequency.
3. Run Processing → Start Compilation and retain the full synthesis, Fitter, timing, warning, and error reports. A successful Analysis & Synthesis alone is insufficient; placement/routing must finish.
4. Open TimeQuest/Timing Analyzer with the post-fit netlist and project SDC. Generate all timing summaries, the Fmax Summary, setup/hold summaries, and Report Unconstrained Paths. Select the slow operating model(s) offered for the actual device. Record the exact process/voltage/temperature label; do not copy a fast-corner value. If several slow corners exist, retain all and identify the limiting one.
5. In the Fmax Summary retain both `Fmax` and `Restricted Fmax`, the clock name, and any notes. Retain the worst setup/hold paths and slack. Do not claim that a nominal 100 MHz constraint was met unless the corresponding checks pass.
6. In Assignments → Settings → Power Analyzer, confirm no activity files, input I/O default toggle rate 20%, and the clock derived from the SDC. Use vectorless estimation for unspecified internal nodes and record that choice. Check device power characteristics, voltage, temperature/cooling, I/O standards/loads, model status, and signal-activity confidence before running Power Analyzer.
7. Save the Power Analyzer summary and full `.pow.rpt`: total thermal power, core dynamic power, static power, I/O power, model/confidence information, and the activity-source/clock reports. Power is estimated at the selected operating clock; if reporting power at another frequency, rerun under that documented condition.

Equivalent minimal command-line flow, from this directory, after choosing an installed tool path:

```powershell
Set-Location Part3/fpga   # from the repository root
$quartusBin = 'C:\intelFPGA_lite\19.1\quartus\bin64' # replace with the actual installation
& "$quartusBin\quartus_sh.exe" --flow compile vanilla_corelet
if ($LASTEXITCODE -ne 0) { throw 'Quartus full compilation failed' }
& "$quartusBin\quartus_pow.exe" vanilla_corelet --no_input_file '--default_input_io_toggle_rate=20%'
if ($LASTEXITCODE -ne 0) { throw 'Quartus power analysis failed' }
```

The CLI power command is documented by the Standard Edition guide. Use the GUI for operating conditions and for selecting/reviewing slow-corner timing reports. Retain reports under both the project directory and `output_files`, since placement depends on tool version/settings.

## How to describe the result accurately

- 20% is a default signal transition rate relative to the applicable clock, not 20% duty cycle. At a 100 MHz clock it corresponds to 20 million transitions/second for such input data signals. The clock itself has two transitions per cycle. Inspect the activity report to ensure clocks are recognized; insufficient clock-domain information can produce zero inferred activity.
- The present SDC defines the clock but intentionally has no invented external min/max input/output delays. Report those interfaces as unconstrained. An internal register-to-register Fmax summary does not establish full corelet-interface timing closure, board timing, or the maximum externally drivable clock.
- Do not add blanket false paths to remove unconstrained-path warnings. Add real interface constraints when the SRAM/controller/board timing contract is available.
- Compare measured results only with the same RTL, tool/device, constraints, seed, I/O treatment, activity assumptions, and operating conditions. Keep source SHA256 hashes and the actual QSF/SDC with the reports.
- This folder supplies no measured metric. Fill report tables only after a successful fit and the corresponding timing/power analysis.

Suggested evidence table:

| Evidence | Collect |
|---|---|
| Reproducibility | Tool/version, device, RTL hashes, QSF/SDC, seed |
| Implementation | Synthesis/Fitter success; logic, registers, DSP and memory resources; warnings |
| Timing | Exact slow corner; Fmax and Restricted Fmax; setup/hold slack; unconstrained interfaces |
| Power | Operating clock; 20% input activity; vectorless/internal settings; total/core dynamic/static/I/O; model and confidence |

## Official syntax and interpretation references

- [Quartus Prime Standard Edition Settings File Reference Manual](https://docs.altera.com/r/docs/d815780/current): QSF setting names, including `POWER_DEFAULT_INPUT_IO_TOGGLE_RATE`, `POWER_USE_INPUT_FILES`, and `POWER_REPORT_SIGNAL_ACTIVITY`.
- [Intel/Altera Standard Edition Power Analysis guide: command line](https://docs.altera.com/r/docs/683506/18.1/intel-quartus-prime-standard-edition-user-guide-power-analysis-and-optimization/running-the-power-analyzer-from-the-command-line): `quartus_pow`, `--no_input_file`, and `--default_input_io_toggle_rate`.
- [Standard Edition guide: default toggle rate](https://docs.altera.com/r/docs/683506/18.1/quartus-prime-standard-edition-user-guide-power-analysis-and-optimization/default-toggle-rate-assignment?contentId=_VZhjmXAPRpe2nhYf7Atzw): relative activity, clock rate, and missing clock-domain inference.
- [Official Altera TimeQuest handbook](https://cdrdv2-public.intel.com/653789/quartusii_handbook_10_0_1.pdf): Fmax versus Restricted Fmax; internal register timing versus clock-port limits.

These references verify the handoff syntax and reporting interpretation. They do not replace running the selected Quartus version on this RTL.
