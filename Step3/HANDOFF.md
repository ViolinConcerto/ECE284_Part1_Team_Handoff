# Step 3 → Step 4 testbench handoff

## Run from the repository root

```text
python Step3/run.py --sim vivado --repeat 2 --waves
```

Set `--vivado-bin` if Vivado is elsewhere. Add `--regress` for signed limits, an asymmetric shape, a 1×1 kernel and the full FIFO depth. The runner reads Step 1 vectors from `artifacts/vectors/`. Python and a simulator are enough for this run.

## Testbench

The [course template](https://github.com/VVIPLab/ece284fa26/blob/8f7df9b3e73b65d373c5e4bd738fd7162f41d046/project/core_tb.v) supplies the control word and stage outline. `core_tb.v` keeps those instruction fields and drives the seven stages through the [Step 2 core](../Step2/HANDOFF.md).

| Stage | Action |
|---|---|
| 1 | Write the activations and all nine weight matrices into L1 SRAM |
| 2 | Transfer the current weight matrix through L0 into the PEs |
| 3 | Fill L0 with the padded activations |
| 4 | Run the array and drain its pipeline |
| 5 | Write the OFIFO output to psum SRAM |
| 6 | Gather and sum nine kernel contributions in the SFU, then write the signed sum back |
| 7 | Apply ReLU to the full sum and write it back to SRAM |

Stages 2–5 repeat for each kernel position. Stage 50 in `cycles.csv` records readback checks. The frozen `features.27` sample is already padded to 6×6. There are 36 input words and 72 weight words. Both pack eight 4-bit lanes per word, with channel 0 in the low bits. The output has 16 words, each with eight signed 16-bit lanes.

## Streaming alpha

```text
python Step3/run.py --sim vivado --pipeline-l0 --stream-ofifo --regress
```

`--pipeline-l0` overlaps SRAM reads with L0 writes. `--stream-ofifo` writes each aligned eight-channel result to SRAM while the PEs are still running. In `cycles.csv`, stage 45 records the combined execution and writeback. Both switches are optional.

| `features.27`, two jobs | Simulation cycles | OFIFO depth per lane |
|---|---:|---:|
| Baseline | 6,461 | 64 |
| L0 pipeline | 5,705 | 64 |
| L0 pipeline + streamed writeback | 4,409 | 16 |

All 256 final values matched in all three runs. The combined mode passed six cases, including singleton input and 64 input words. The highest OFIFO occupancy was eight words in one lane. OFIFO data storage falls from 8,192 to 2,048 bits, excluding control. For a Cyclone IV fit of the alpha, see [Step 5](../Step5/README.md#reproduce).

## Step 4

1. Run the command above from this checkout. The runner validates and copies `input.txt`, `weight.txt` and `psum.txt` into `Step3/build/inputs/team_handoff/` and checks both pre-ReLU and post-ReLU integer results from the Step 1 export.
2. Check `Step3/build/team_handoff/simulation.log` and `summary.json`. Compare `result_0.txt` and `result_1.txt` with the prepared `psum.txt`. The result files contain the final words read from SRAM. Expect 16 words and zero mismatches per job.
3. Save the simulator version, source revision, vector provenance and results with your verification report. Keep the integer comparison exact. The model recovery error reported in Step 1 is measured after applying the two scales.

[Baseline result](results/baseline.json): Vivado 2017.4, both real jobs passed, 256 final values and 648 kernel-psum words checked. All four extra cases also passed.
