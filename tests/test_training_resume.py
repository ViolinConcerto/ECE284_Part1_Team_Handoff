"""Exercise the real training/checkpoint/resume pipeline with synthetic data.

Run ``python -m unittest discover -s tests -v`` from the package root.
The fixture contains FOUR training and TWO evaluation tensors, not CIFAR-10.
Its temporary metrics must never be treated as classification-accuracy evidence.
No dataset download, GPU, or persistent checkpoint/output is required.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


class SyntheticDataset(Dataset):
    """Fixed synthetic images, with augmentation drawing from all three RNGs."""

    def __init__(self, count: int, augment: bool) -> None:
        generator = torch.Generator().manual_seed(1234)
        self.images = torch.randn(count, 3, 32, 32, generator=generator)
        self.labels = torch.arange(count) % 10
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].clone()
        if self.augment:
            image += (
                torch.rand(()) * 0.1
                + random.random() * 0.1
                + float(np.random.rand()) * 0.1
            )
        return image, int(self.labels[index])


def synthetic_loaders(args, device):
    """Replace only external CIFAR I/O; retain real SGD/evaluation/export code."""
    generator = torch.Generator().manual_seed(args.seed)
    training = SyntheticDataset(4, augment=True)
    evaluation = SyntheticDataset(2, augment=False)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  worker_init_fn=train.seed_worker)
    return (
        DataLoader(training, shuffle=True, generator=generator, **common),
        DataLoader(evaluation, shuffle=False, **common),
        evaluation,
        generator,
    )


class PlannedInterruption(RuntimeError):
    """Simulate a stop after the complete first-epoch checkpoint was written."""


class TrainingResumeTests(unittest.TestCase):
    def assert_nested_identical(self, left, right, path="state"):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, path)
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertEqual(left.shape, right.shape, path)
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertIsInstance(right, dict, path)
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assert_nested_identical(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertIsInstance(right, type(left), path)
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self.assert_nested_identical(left_item, right_item, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    @staticmethod
    def arguments(output: Path) -> list[str]:
        return [
            "--device", "cpu", "--threads", "2", "--workers", "0",
            "--batch-size", "2", "--qat-epochs", "2", "--warmup-epochs", "0",
            "--milestones", "1", "--seed", "731", "--deterministic",
            "--no-download", "--output", str(output),
        ]

    def test_synthetic_two_epoch_training_and_resume_are_identical(self):
        """Real all-13-W4A4 model, SGD momentum, LR decay and RNG restoration."""
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_python_rng = random.getstate()
        previous_numpy_rng = np.random.get_state()
        previous_torch_rng = torch.get_rng_state()
        captured_output = io.StringIO()
        try:
            with tempfile.TemporaryDirectory(prefix="ece284_synthetic_resume_") as directory:
                root = Path(directory)
                full_output, split_output = root / "full", root / "interrupted"
                # All reports/checkpoints are temporary. Suppress their output:
                # mock-dataset metrics must not appear as real CIFAR evidence.
                with mock.patch.object(train, "cifar_loaders", synthetic_loaders), \
                     contextlib.redirect_stdout(captured_output):
                    return_code = train.main(self.arguments(full_output))
                    self.assertIn(return_code, (0, 2))
                    full = train.load_checkpoint(full_output / "last.pth")

                    original_train_epoch = train.train_epoch
                    epoch_calls = 0

                    def interrupt_before_second_epoch(*args, **kwargs):
                        nonlocal epoch_calls
                        epoch_calls += 1
                        if epoch_calls == 2:
                            raise PlannedInterruption("Synthetic test interruption")
                        return original_train_epoch(*args, **kwargs)

                    with mock.patch.object(train, "train_epoch", interrupt_before_second_epoch):
                        with self.assertRaises(PlannedInterruption):
                            train.main(self.arguments(split_output))
                    first_epoch = train.load_checkpoint(split_output / "last.pth")
                    self.assertEqual(first_epoch["epoch"], 1)
                    self.assertEqual(first_epoch["phase"], "qat")
                    self.assertTrue((split_output / "best.pth").is_file())

                    return_code = train.main(
                        self.arguments(split_output)
                        + ["--resume", str(split_output / "last.pth")]
                    )
                    self.assertIn(return_code, (0, 2))
                    resumed = train.load_checkpoint(split_output / "last.pth")

                self.assertEqual(full["epoch"], 2)
                self.assertEqual(resumed["epoch"], 2)
                self.assertTrue(resumed["quantization_enabled"])
                self.assertEqual(resumed["architecture"]["quantized_convolution_count"], 13)
                # The small counts explicitly distinguish this fixture from the
                # real 50,000/10,000-example CIFAR-10 training/evaluation splits.
                self.assertEqual(full["validation"]["total"], 2)
                for key in (
                    "model_state", "optimizer_state", "scheduler_state", "validation",
                    "best_accuracy_pct", "rng_state", "loader_rng_state",
                    "python_rng_state", "numpy_rng_state", "cuda_rng_states",
                ):
                    self.assert_nested_identical(full[key], resumed[key], key)

                # Best-checkpoint selection and safe reloading also survive the
                # interruption, even when the second epoch is not an improvement.
                full_best = train.load_checkpoint(full_output / "best.pth")
                resumed_best = train.load_checkpoint(split_output / "best.pth")
                self.assertEqual(full_best["epoch"], resumed_best["epoch"])
                self.assert_nested_identical(
                    full_best["model_state"], resumed_best["model_state"], "best.model_state"
                )
                self.assertEqual(resumed["scheduler_state"]["last_epoch"], 2)
                self.assertAlmostEqual(resumed["optimizer_state"]["param_groups"][0]["lr"], 0.002)

                with (split_output / "history.csv").open(newline="") as stream:
                    history = list(csv.DictReader(stream))
                self.assertEqual([int(row["epoch"]) for row in history], [1, 2])
                with (full_output / "history.csv").open(newline="") as stream:
                    full_history = list(csv.DictReader(stream))
                # Elapsed wall time is expected to differ; every numerical
                # training/evaluation result and recorded LR must be identical.
                without_seconds = lambda rows: [
                    {key: value for key, value in row.items() if key != "seconds"}
                    for row in rows
                ]
                self.assertEqual(without_seconds(full_history), without_seconds(history))
                for output in (full_output, split_output):
                    result = json.loads((output / "results.json").read_text())
                    self.assertEqual(result["total"], 2)
                    self.assertTrue((output / "vectors" / "verification.json").is_file())
                    self.assertTrue((output / "part1_report.txt").is_file())
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_deterministic)
            random.setstate(previous_python_rng)
            np.random.set_state(previous_numpy_rng)
            torch.set_rng_state(previous_torch_rng)


if __name__ == "__main__":
    unittest.main()
