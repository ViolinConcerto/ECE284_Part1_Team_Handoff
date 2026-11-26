# Step 5 results: vanilla corelet on Cyclone IV GX

Run on 2026-11-26 with Quartus Prime 19.1.0 Build 670 Patches 0.02std SJ Lite Edition. Full compilation (Analysis & Synthesis, Fitter, Assembler, Timing Analyzer) and Power Analyzer completed successfully.

## Final numbers

| Metric | Value |
|---|---|
| Device | Cyclone IV GX EP4CGX150DF31I7AD |
| Top-level entity | `corelet` (8x8 array, 4-bit act/weight, 16-bit psum; SRAM excluded) |
| **Fmax, slow corner (Slow 1200mV 100C)** | **112.97 MHz** (Restricted Fmax 112.97 MHz) |
| **Total thermal power (100 MHz clock, 20% input toggle rate)** | **373.79 mW** |

## Implementation

| Item | Value |
|---|---|
| Logic elements | 17,169 / 149,760 (11%) |
| Combinational functions / dedicated registers | 12,081 / 12,098 |
| Pins | 199 real (all inputs) + 389 virtual (all outputs) |
| Memory bits / 9-bit multipliers / PLLs | 0 / 0 / 0 |
| Fitter seed | 1 |

## Timing (10.000 ns / 100 MHz constraint on `clk`)

| Model | Fmax | Setup slack | Hold slack |
|---|---|---|---|
| Slow 1200mV 100C (limiting) | 112.97 MHz | +1.148 ns | +0.395 ns |
| Slow 1200mV -40C | 126.33 MHz | +2.084 ns | +0.325 ns |
| Fast 1200mV -40C | — | +5.713 ns | +0.166 ns |

The 100 MHz constraint is met in all corners. Fmax is internal register-to-register timing. External input/output delays are not specified, so the 170 input ports are reported as unconstrained; no false paths were added.

## Power (Power Analyzer)

| Component | Power |
|---|---|
| Total thermal power | 373.79 mW |
| Core dynamic | 211.54 mW |
| Core static | 119.73 mW |
| I/O | 42.52 mW |

Settings: clock 100 MHz from the SDC; default input I/O toggle rate 20%; no VCD/SAF input files; vectorless estimation for internal nodes (default toggle rate 12.5% for remaining signals); ambient 25 °C, auto-computed junction temperature 27.0 °C, preset cooling "23 mm heat sink with 200 LFPM airflow". Quartus reports confidence "Low" because no simulation activity was supplied, which is expected for a vectorless estimate.

By block: register cells 84.43 mW, clock control 61.81 mW, combinational cells 57.85 mW, I/O 42.44 mW.

## Changes needed for Quartus

1. **`generate` blocks.** Quartus rejects module-level `for` loops without `generate`/`endgenerate` (error 10170), which Vivado accepts. Patched copies of `mac_row.v`, `mac_array.v`, `l0.v`, `ofifo.v` and `fifo_mux_2_1.v` are in `fpga_src/`. Each copy only adds the two keywords around the existing loop. The Step 2 simulation sources are unchanged.
2. **Virtual pins.** `corelet` has 588 ports, but the F896 package has 475 user I/O. All outputs (`ofifo_out`, `sfp_out`, `pmem_write_data` and the five status bits) are assigned `VIRTUAL_PIN ON`. Inputs remain real pins so the 20% input toggle rate applies to them. Output-driver power is therefore not included in the I/O power.
3. Some `inst` bits (SRAM control/address fields that `corelet` does not use) have no clock domain; Power Analyzer assumes 0 toggle rate for them. They drive no logic in `corelet`.

Raw reports (`output_files/*.rpt`) are git-ignored; force-add them if the grader needs them.
