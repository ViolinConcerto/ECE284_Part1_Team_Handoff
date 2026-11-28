# Step2 -> Step3 RTL handoff

## Start here

`filelist.f` lists the vanilla hardware used by the Step3 testbench. Use the frozen Part1 vectors in the repository's `artifacts/vectors/`. Steps2-4 baseline needs no GPU, PyTorch, new training or Canvas download.

| Module / path | Role |
|---|---|
| `rtl/core.v` | Connects corelet, activation/weight SRAM and psum SRAM; simulation top below the testbench |
| `rtl/corelet.v` | Connects L0, the 8x8 MAC array, OFIFO and SFU; excludes the SRAM instances |
| `rtl/sfp.v` | Signed accumulation, clear/hold and ReLU; core/corelet provide the result path back to SRAM |
| `course/` | Exact published MAC, MAC array, FIFO64/mux and SRAM source files |
| `homework/` | The student's completed HW6 PE/row and HW7 L0/OFIFO, reused for this integration |

For the later FPGA step, map **corelet**, not core or the testbench. This directory is an RTL handoff, not a Quartus timing or power result.

## Frozen arithmetic contract

| Property | Value |
|---|---|
| Layer | `features.27`, 8 input channels -> 8 output channels |
| Kernel / stride / padding | 3x3 / 1 / 1; no bias, no BN |
| Layer input / padded input | NCHW 1x8x4x4 / CHW 8x6x6 |
| Activation | Unsigned 4-bit, codes 0..15 |
| Weight | Signed 4-bit two's complement; frozen codes -7..7 |
| Partial sums / outputs | Signed 16-bit, eight lanes in each 128-bit word |
| Packing | Channel 0 in the least significant lane; pixels and kernel positions row-major |
| Final operation | Sum all nine kernel-position contributions, then ReLU |

The 8x8 array holds one kernel-position weight matrix at a time. Nine matrices are required for the 3x3 layer even though no channel tiling is needed. There is no IFIFO in this vanilla path. Keep arithmetic in integer units; Part1's scales are for software reconstruction, not an extra hardware multiply.

## Control and memory interface

The existing course instruction layout is retained:

| `inst` bits | Function |
|---|---|
| 33 | SFU accumulate |
| 32 / 31 / 30:20 | psum SRAM CEN / WEN / address |
| 19 / 18 / 17:7 | input SRAM CEN / WEN / address |
| 6 | OFIFO read |
| 5:4 | Unused IFIFO controls; keep zero |
| 3 / 2 | L0 read / write |
| 1 / 0 | Execute / weight load |

SRAM CEN and WEN are active low. A synchronous read captures the address before the consumer can latch the data. The default OFIFO registers its read request. `STREAM_OFIFO=1` selects a smaller buffer that pops all lanes together on `rd && ofifo_valid`, allowing SRAM to capture the same word on that edge.

The integration also needs explicit SFU clear, hold and ReLU control, selection of SFU versus OFIFO data for psum SRAM writes, and a way to read SRAM results back. Clearing a new output sum must not erase previously stored SRAM data. ReLU must not run between the nine signed contributions.

## Source and integration boundary

`course/` preserves the published [VVIPLab course sources](https://github.com/VVIPLab/ece284fa26) and author headers. `homework/` preserves the student's completed HW6/HW7 modules. New core, corelet and SFU integration is in `rtl/`; do not label those files as supplied staff solutions.

The published 2026 sources and `project/core_tb.v` are pinned at commit `8f7df9b3e73b65d373c5e4bd738fd7162f41d046`. The testbench includes deliberate control placeholders. This provenance does not by itself establish a simulation pass; use this checkout's fresh Step3 results.

Before handing a change to Step3, check compilation, signed activation/weight handling, weight retention, FIFO alignment, SFU hold/clear and both SRAM writeback paths. See [Step3/HANDOFF.md](../Step3/HANDOFF.md) for the simulation and Step4 verification checklist.
