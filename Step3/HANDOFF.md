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

## Alpha modes (Step 6)

The default command above is the baseline required by Steps 3–4. Each switch below is optional, keeps the course `inst` fields, and is checked by the same regression:

| Switch | Effect | Stage in `cycles.csv` |
|---|---|---|
| `--pipeline-l0` | SRAM reads overlap L0 writes (one word per cycle) | 3 |
| `--stream-ofifo` | 16-word OFIFO; psums are written to SRAM while the PEs run (`STREAM_OFIFO=1`) | 45 |
| `--skip-zero` | L1 holds only nonzero words plus an occupancy bitmap; each kernel position issues only the nonzero words of its valid output window | 1–7 |
| `--pipeline-sfu` | psum reads overlap SFU accumulation; ReLU follows the stored sum without reloading it | 67 |
| `--operand-isolation` | `OPERAND_ISOLATION` PE variant, evaluated and not adopted | — |

```text
python Step3/run.py --sim vivado --regress --pipeline-l0 --stream-ofifo --skip-zero --pipeline-sfu
```

The testbench prints `STATS` per job (issued words, SFU reads, skipped passes and SRAM accesses of the work stages); `run.py` recomputes them independently and fails on any difference. Every run also executes the SFU and FIFO unit tests. Results, figures and the mutation report are in [Step 6](../Step6/RESULTS.md).

## Step 4

1. Run the command above from this checkout. The runner validates and copies `input.txt`, `weight.txt` and `psum.txt` into `Step3/build/inputs/team_handoff/` and checks both pre-ReLU and post-ReLU integer results from the Step 1 export.
2. Check `Step3/build/team_handoff/simulation.log` and `summary.json`. Compare `result_0.txt` and `result_1.txt` with the prepared `psum.txt`. The result files contain the final words read from SRAM. Expect 16 words and zero mismatches per job.
3. Save the simulator version, source revision, vector provenance and results with your verification report. Keep the integer comparison exact. The model recovery error reported in Step 1 is measured after applying the two scales.

[Baseline result](results/baseline.json): Vivado 2017.4, both real jobs passed, 256 final values and 648 kernel-psum words checked. All four extra cases also passed.
