"""Checkpoint compatibility with restrictive ``weights_only=True`` loaders.

These tests use a tiny synthetic linear model. They do not train VGG or measure
CIFAR-10 accuracy. A real MultiStepLR trajectory, including repeated milestones,
checks that safe serialization preserves scheduling and optimizer continuation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import pickle
import pickletools
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def architecture_config(self):
        return {'model': 'synthetic_checkpoint_fixture'}

    def forward(self, value):
        return self.linear(value)


def saved_globals(path):
    """Read GLOBAL opcodes without unpickling or executing checkpoint content."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(names) != 1:
            raise AssertionError('Expected one torch checkpoint pickle')
        data = archive.read(names[0])
    return {str(argument) for opcode, argument, _ in pickletools.genops(data)
            if opcode.name == 'GLOBAL'}


def explicit_safe_globals():
    # PyTorch 2.2 predates the public safe_globals/get_safe_globals API.
    getter = getattr(torch.serialization, 'get_safe_globals', None)
    return list(getter()) if callable(getter) else []


def counter_explicitly_allowed():
    entries = explicit_safe_globals()
    return any((entry[0] if isinstance(entry, tuple) else entry) is Counter
               for entry in entries)


def fixture():
    torch.manual_seed(493)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 1, 3], gamma=0.1)
    return model, optimizer, scheduler


def advance(model, optimizer, scheduler, epochs):
    images = torch.tensor([[1.0], [-0.5]])
    targets = torch.tensor([[0.75], [-0.25]])
    rates = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(images), targets).backward()
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    return rates


def checkpoint(model, optimizer, scheduler, epoch):
    return train.make_checkpoint(
        model, optimizer, scheduler, phase='qat', epoch=epoch,
        best_accuracy=0.0,
        args=SimpleNamespace(milestones=[1, 1, 3], gamma=0.1),
        generator=torch.Generator().manual_seed(824),
        validation={'loss': 0.0, 'accuracy_pct': 0.0, 'correct': 0, 'total': 2},
    )


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_new_checkpoint_contains_no_counter_and_loads_with_old_style_restriction(self):
        model, optimizer, scheduler = fixture()
        payload = checkpoint(model, optimizer, scheduler, 0)
        self.assertIs(type(payload['scheduler_state']['milestones']), dict)
        self.assertEqual(payload['scheduler_state']['milestones'], {1: 2, 3: 1})
        original_load = torch.load
        load_modes = []

        def restrictive_load(path, *args, **kwargs):
            load_modes.append(kwargs.get('weights_only'))
            if 'collections Counter' in saved_globals(path):
                raise pickle.UnpicklingError('Unsupported global: GLOBAL collections.Counter')
            return original_load(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pth'
            train.atomic_torch_save(payload, path)
            self.assertNotIn('collections Counter', saved_globals(path))
            with mock.patch.object(train.torch, 'load', restrictive_load):
                loaded = train.load_checkpoint(path)
            self.assertEqual(loaded['scheduler_state']['milestones'], {1: 2, 3: 1})
            self.assertEqual(load_modes, [True])

    def test_real_lr_and_momentum_trajectory_survive_repeated_milestone_save_resume(self):
        baseline_model, baseline_optimizer, baseline_scheduler = fixture()
        baseline_rates = advance(baseline_model, baseline_optimizer, baseline_scheduler, 5)
        for actual, expected in zip(baseline_rates, [0.002, 0.002, 0.0002, 0.0002, 0.0002]):
            self.assertAlmostEqual(actual, expected)
        for stop_after in (0, 1, 2, 3):
            with self.subTest(stop_after=stop_after), tempfile.TemporaryDirectory() as folder:
                model, optimizer, scheduler = fixture()
                before = advance(model, optimizer, scheduler, stop_after)
                path = Path(folder) / 'last.pth'
                train.atomic_torch_save(checkpoint(model, optimizer, scheduler, stop_after), path)
                saved = train.load_checkpoint(path)
                resumed_model, resumed_optimizer, resumed_scheduler = fixture()
                resumed_model.load_state_dict(saved['model_state'])
                resumed_optimizer.load_state_dict(saved['optimizer_state'])
                train.restore_scheduler_state(resumed_scheduler, saved['scheduler_state'])
                # MultiStepLR relies on Counter returning zero for absent keys.
                self.assertIsInstance(resumed_scheduler.milestones, Counter)
                self.assertEqual(resumed_scheduler.milestones[2], 0)
                self.assertEqual(resumed_scheduler.milestones[999], 0)
                self.assertEqual(resumed_scheduler.milestones[1], 2)
                after = advance(resumed_model, resumed_optimizer, resumed_scheduler, 5 - stop_after)
                self.assertEqual(before + after, baseline_rates)
                for name, value in baseline_model.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed_model.state_dict()[name]), name)
                expected_optimizer = baseline_optimizer.state_dict()
                actual_optimizer = resumed_optimizer.state_dict()
                self.assertEqual(actual_optimizer['param_groups'], expected_optimizer['param_groups'])
                for number, state in expected_optimizer['state'].items():
                    for key, tensor in state.items():
                        self.assertTrue(torch.equal(actual_optimizer['state'][number][key], tensor))

    def test_legacy_counter_checkpoint_rejection_keeps_safe_loading_enabled(self):
        model, optimizer, scheduler = fixture()
        payload = checkpoint(model, optimizer, scheduler, 0)
        payload['scheduler_state']['milestones'] = Counter([1, 1, 3])
        original_load = torch.load
        initial_safe_globals = explicit_safe_globals()
        modes, permitted = [], []

        def restrictive_old_load(path, *args, **kwargs):
            modes.append(kwargs.get('weights_only'))
            allowed = counter_explicitly_allowed()
            permitted.append(allowed)
            if 'collections Counter' in saved_globals(path) and not allowed:
                raise pickle.UnpicklingError(
                    'Weights only load failed. Unsupported global: GLOBAL collections.Counter')
            return original_load(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pth'
            torch.save(payload, path)
            self.assertIn('collections Counter', saved_globals(path))
            with mock.patch.object(train.torch, 'load', restrictive_old_load):
                with self.assertRaises(pickle.UnpicklingError):
                    train.load_checkpoint(path)
        self.assertEqual(modes, [True])
        self.assertEqual(permitted, [False])
        self.assertEqual(explicit_safe_globals(), initial_safe_globals)

    def test_unrelated_unsupported_global_never_triggers_unsafe_pickle_loading(self):
        modes = []
        initial_safe_globals = explicit_safe_globals()

        def reject_unknown(*args, **kwargs):
            modes.append(kwargs.get('weights_only'))
            raise pickle.UnpicklingError('Unsupported global: GLOBAL unrelated.CustomObject')

        with mock.patch.object(train.torch, 'load', reject_unknown):
            with self.assertRaises(pickle.UnpicklingError):
                train.load_checkpoint('not_read_by_mock.pth')
        self.assertTrue(all(mode is True for mode in modes))
        self.assertEqual(explicit_safe_globals(), initial_safe_globals)


if __name__ == '__main__':
    unittest.main()
