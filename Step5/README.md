# Step 5: vanilla corelet on Cyclone IV GX

**Status: done.** Full compilation and power analysis completed with Quartus Prime 19.1 Lite. Detailed numbers, settings and notes are in [RESULTS.md](RESULTS.md).

| Result | Value |
|---|---|
| Device | Cyclone IV GX EP4CGX150DF31I7AD |
| Top-level entity | `corelet` (8x8 array, 4-bit activation/weight, 16-bit psum; SRAM excluded) |
| **Fmax, slow corner (Slow 1200mV 100C)** | **112.97 MHz** |
| **Total power (100 MHz clock, 20% input toggle rate)** | **373.79 mW** |
| Logic elements | 17,169 / 149,760 (11%) |

Screenshots: [Fmax summary](fmax_slow_100C.png), [Power Analyzer summary](power_summary.png).

## Files

| File | Role |
|---|---|
| `vanilla_corelet.qpf` / `.qsf` | Quartus project and settings (device, sources, 20% input toggle rate, virtual pins) |
| `vanilla_corelet.sdc` | 10 ns (100 MHz) clock constraint on `clk` |
| `fpga_src/` | Quartus-compatible copies of 5 Step 2 files (see below) |
| `RESULTS.md` | Final timing, power and resource results |

The other 7 source files are read directly from `../Step2/`.

## Differences from the Step 2 simulation sources

1. Quartus rejects module-level `for` loops without `generate`/`endgenerate` (error 10170), although Vivado accepts them. `fpga_src/` holds copies of `mac_row.v`, `mac_array.v`, `l0.v`, `ofifo.v` and `fifo_mux_2_1.v` with only those two keywords added. Logic is unchanged, and `Step2/` is untouched so simulation still uses the original files.
2. `corelet` has 588 ports, but the F896 package has only 475 user I/O. All outputs are assigned as virtual pins. Inputs stay real pins so the 20% input toggle rate applies to them. Output-driver power is therefore not included in the I/O power.

## Reproduce

1. Open `vanilla_corelet.qpf` in Quartus Prime 19.1 Lite with Cyclone IV device support installed.
2. Processing → Start Compilation.
3. Compilation Report → Timing Analyzer → Slow 1200mV 100C Model → Fmax Summary. The Slow 1200mV −40C model gives 126.33 MHz, so the 100C model is the limiting slow corner.
4. Processing → Power Analyzer Tool → Start. No input activity files are used; input I/O toggle rate is 20% and internal nodes use vectorless estimation. Quartus reports confidence "Low" because no simulation activity is supplied.

Compilation outputs (`db/`, `output_files/`, `*.rpt`) are git-ignored.
