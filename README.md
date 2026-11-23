# ECE284 Part1: trained W4A4 VGG16 and RTL handoff

This repository contains the Part1 code, completed training evidence, original best checkpoint and frozen hardware vectors for the team's later project stages. Read **[HANDOFF.md](HANDOFF.md)** before integrating the vectors into RTL.

| Result | Value |
|---|---|
| CIFAR-10 full-test accuracy recorded in original run | **90.94% (9094/10000)** |
| Best checkpoint | **Epoch 183 / 200** |
| Quantization | All 13 convolution inputs and weights: W4A4 |
| Selected layer | `features[27]`, 8 -> 8, 3x3, stride 1, padding 1, no bias |
| Following operation | ReLU, no BN before next convolution `features[29]` |
| Selected output | 8x4x4 |
| Maximum absolute reconstruction error | **4.076379191531032e-6**, passes <1e-5 and <1e-3 |
| Integer reconstruction scope | Selected layer, fixed CIFAR-10 test sample 0 |

The recorded accuracy was measured on all 10,000 test images after reloading the best checkpoint. The test set also selected the best checkpoint, following the course reference; it is not an untouched research holdout. The numerical error compares independently computed int64 convolution + ReLU + scale recovery with the actual next-convolution prehook input, captured by detached clone. BN and classifier parameters remain floating point. The first convolution has signed normalized RGB input; later ReLU activations are unsigned.

## Use the frozen vectors immediately

No GPU or training is needed for the exported data. With NumPy installed:

```bash
python tools/check_handoff_vectors.py
```

Use **the complete `artifacts/vectors/` directory** for the team's RTL. See `HANDOFF.md` for paths, bit packing, working directory, scales and required downstream checks.

## Layout

| Path | Contents |
|---|---|
| `models.py`, `train.py`, `verify_export.py` | Original model, QAT/resume/evaluation, independent integer reconstruction and export |
| `tests/` | Quantizer, architecture, numerical, packing, checkpoint and resume tests |
| `Part1_Run.ipynb` | Unexecuted entry point for a new run; opening it does not load the frozen checkpoint automatically |
| `results/Part1_Run_executed.ipynb` | Completed original notebook; project paths redacted |
| `artifacts/checkpoints/best.pth` | Original epoch-183 trained checkpoint |
| `artifacts/run/` | Original results/config/history/architecture/report/curve |
| `artifacts/vectors/` | Actual trained-layer stimuli and expected output |
| `artifacts/validation/` | Independent handoff checks and file hashes |
| `tools/check_handoff_vectors.py` | NumPy-only checker for the frozen vectors |

The original dataset and `last.pth` are excluded. `last.pth` is for continuing the original run; the best checkpoint is sufficient for re-evaluation and exporting more samples. All original source/test files and frozen input artifacts are preserved without changing their values. Historical absolute paths in run records identify the original machine and are not current commands.

## Train or re-evaluate

The recorded run used Python 3.11, PyTorch 2.2.1+cu121, GTX 1080 Ti, batch 128, seed 284, SGD momentum 0.9, LR 0.02, weight decay 1e-4, alpha regularization 1e-5, and milestones 80/140/180 (gamma 0.1). Requirements are not fully pinned; the run did not enable deterministic mode, so a new training run may differ.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --device cuda --data-root data --output runs/new_training --qat-epochs 200 --batch-size 128 --workers 2 --seed 284
```

Re-evaluate the supplied best model, without retraining:

```bash
python train.py --device cuda --data-root data --output runs/reevaluate --evaluate artifacts/checkpoints/best.pth --sample-index 0
```

The original 10,000-image model evaluation and a new RTL simulation were not rerun in the handoff packaging environment. Fresh artifact checks verify consistency of the supplied tensors/vectors and checkpoint contents; see `artifacts/validation/` for their exact scope. Part4 must still verify the team's actual core and testbench with these vectors.

## Source reference

Quantizer conventions were checked against the [course repository](https://github.com/VVIPLab/ece284fa26), commit `94632eae4cea0889a8d28d7b3129a74d158eb9c3`. The 2024 reference supplied architecture and hardware layout guidance; its historical accuracy is not a result of this implementation. No redistribution license is selected here.

The included 67.84 MiB checkpoint requires a command-line Git push for GitHub; the browser's per-file limit is 25 MiB. Instructions and official documentation are in `HANDOFF.md`.
