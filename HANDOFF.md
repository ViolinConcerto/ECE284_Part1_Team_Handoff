# Step 1 → RTL handoff

## Start here

The frozen trained-layer inputs and expected outputs are in `artifacts/vectors/`. No training or GPU is needed to consume these files in a testbench. Preserve this entire directory and all relative paths. Start with `metadata.json` and `verification.json`.

From the repository root, run the independent NumPy checker:

```bash
python -m pip install numpy
python tools/check_handoff_vectors.py
```

Expected: PASS, all 128 integer output values match, maximum recovered-vs-captured error approximately 4.076379191531032e-6. This checks exported data and captured tensors, not the RTL implementation.

## Frozen hardware contract

| Property | Value |
|---|---|
| Selected convolution | `features.27`, 8 input channels, 8 output channels |
| Kernel / stride / padding | 3x3 / 1 / 1; no bias |
| Model input to selected layer | NCHW 1x8x4x4 |
| Exported padded activation | 8x6x6 |
| Output | 8x4x4 = 128 scalar values |
| Activation coding | unsigned 4-bit, 0..15 |
| Weight coding | signed 4-bit, two's complement; codes -7..7 |
| Output coding | signed 16-bit per channel, eight channels per 128-bit word |
| Channel ordering | channel 0 in least significant bits; channel 7 shown leftmost in binary text |
| Pixel/kernel traversal | row-major |
| Post-processing | accumulation over all 9 kernel positions, then ReLU; no BN |
| Activation scale | 0.46577975153923035 |
| Weight scale | 0.38634926080703735 |
| Test sample | CIFAR-10 test index 0, label 3 |

All 8 input/output channels fit the array without channel tiling. A 3x3 convolution still has nine kernel-position contributions; all nine must be accumulated. The output expected by the hardware is an integer, not the scale-recovered floating-point value.

`model_value = integer_output_after_relu * activation_scale * weight_scale`.

## Vector files

| File | Format / role |
|---|---|
| `activation/activation.txt` | 36 x 32-bit words, 8 padded activation channels per word |
| `weights/weight0.txt` ... `weight8.txt` | 9 kernel positions; each file has 8 x 32-bit words, one output-channel weight vector per word |
| `output/output_updated.txt` | 16 x 128-bit words, expected accumulated integer output before ReLU |
| `output/output_after_relu.txt` | 16 x 128-bit words, expected integer output after ReLU |
| `acc_address.txt` | 144 x 11-bit addresses for the legacy SFU gathering sequence |
| `input.txt`, `weight.txt`, `psum.txt` | aliases / merged weight file for the assignment's naming convention |
| `metadata.json`, `verification.json` | scales, layout, provenance, model fingerprint and observed errors |
| `tensors.npz` | integer inputs/weights/results, captured real next-layer input and recovered arrays |

Input and weight files contain exactly three whitespace-delimited header tokens before the binary words. Output/address files have no header. Decode weights and output lanes as signed two's-complement values. The activation lanes are unsigned.

## Testbench integration

From the repository root, run the current testbench with the frozen vectors:

```bash
python Step3/run.py
```

The runner imports the vector files into its build directory and compares the final SRAM output with `psum.txt`. See the [Step 3 handoff](Step3/HANDOFF.md) for the Step 4 procedure and output files.

The baseline assignment also requires weight delivery through L0, no IFIFO, and actual SRAM writeback after SFU accumulation and after ReLU. Comparing only `sfp_out` does not establish those writebacks.

## What is preserved

- `artifacts/checkpoints/best.pth`: original best checkpoint, epoch 183; includes model/optimizer/scheduler/RNG state.
- `artifacts/run/`: original results, config, 200-epoch history, architecture, curve and Part1 report draft.
- `results/Part1_Run_executed.ipynb`: completed notebook with machine-specific paths redacted.
- `artifacts/validation/`: fresh handoff consistency checks and file hashes.

Historical JSON/checkpoint fields may still contain `/home/chren/private/ECE284_Part1/...`; these are original run records. Use the current repository-relative paths for commands. `last.pth` is intentionally omitted: it is needed only to continue the original training, not to reproduce the best model or run RTL vectors. The original uploaded run archive retains it.

## Re-evaluate / export more samples

Use an environment with the project's PyTorch/torchvision dependencies and CIFAR-10. Write into a new output directory to preserve the frozen artifacts:

```bash
python train.py --device cuda --data-root data --output runs/reevaluate --evaluate artifacts/checkpoints/best.pth --sample-index 0
```

CPU evaluation is supported using `--device cpu`, but is slower. A fresh runtime or device may change floating-point rounding and reported reconstruction error. Do not replace the frozen vectors silently; export to a new directory and record sample index, model fingerprint and validation results.

## GitHub upload

Unzip this package and commit the extracted directory contents. `.gitignore` keeps the frozen vectors and best checkpoint trackable, while ignoring new local datasets/runs/caches. The best checkpoint is 71,135,885 bytes (about 67.84 MiB). GitHub browser uploads are limited to 25 MiB per file; use a Git command-line push for the full package. GitHub's ordinary Git limit is 100 MiB per file, so this checkpoint fits. See [GitHub file upload documentation](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

For a new empty repository, from the extracted project root:

```bash
git init
git add .
git status --short
git commit -m "Add Part1 trained model and RTL handoff vectors"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL` with the actual repository URL and sign into your GitHub account through your normal Git setup. For an existing repository, copy these files into that checkout and review its current changes instead of initializing a second repository. Add the team members as collaborators if the repository is private.
