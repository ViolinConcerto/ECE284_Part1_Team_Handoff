"""Independent checks for QAT, downstream-hook verification and RTL vectors.

Run from the package root with ``python -m unittest discover -s tests -v``.
These checks use synthetic tensors; they do not establish CIFAR-10 accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import QuantConv2d, build_model, quantize_activation, quantize_weight
from verify_export import VerificationError, exact_conv2d_int64, verify_and_export


def loop_cross_correlation(x, w, padding=(1, 1), stride=(1, 1)):
    """Small literal oracle: no torch convolution, einsum or sliding windows."""
    n, c, h, width = x.shape
    oc, _, kh, kw = w.shape
    ph, pw = padding
    sh, sw = stride
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (width + 2 * pw - kw) // sw + 1
    result = np.zeros((n, oc, oh, ow), dtype=np.int64)
    for batch in range(n):
        for out_channel in range(oc):
            for y in range(oh):
                for xx in range(ow):
                    total = 0
                    for channel in range(c):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = y * sh + ky - ph
                                ix = xx * sw + kx - pw
                                if 0 <= iy < h and 0 <= ix < width:
                                    total += int(x[batch, channel, iy, ix]) * int(w[out_channel, channel, ky, kx])
                    result[batch, out_channel, y, xx] = total
    return result


def decode_word(word, bits, channel, signed=False):
    """Channel zero is in the least significant bits of each hardware word."""
    value = (int(word, 2) >> (bits * channel)) & ((1 << bits) - 1)
    if signed and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


class ScaledReLU(nn.Module):
    def forward(self, value):
        return value.relu() * 2


class HookFixture(nn.Module):
    """Small real quantized forward graph at the project's layer indices."""
    def __init__(self, corrupt_downstream=False):
        super().__init__()
        self.features = nn.Sequential(
            *[nn.Identity() for _ in range(27)],
            QuantConv2d(8, 8, 3, padding=1, bias=False),
            ScaledReLU() if corrupt_downstream else nn.ReLU(inplace=False),
            QuantConv2d(8, 8, 1, bias=False),
        )
        generator = torch.Generator().manual_seed(271)
        codes = torch.randint(-7, 8, (8, 8, 3, 3), generator=generator)
        with torch.no_grad():
            self.features[27].weight.copy_(codes * 0.25)
            self.features[27].set_clipping_(activation=3.75, weight=1.75)
            self.features[29].set_clipping_(activation=8.0, weight=2.0)

    def forward(self, value):
        return self.features(value)


class Part1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def test_quantizers_have_correct_lattices_and_straight_through_gradients(self):
        x = torch.tensor([-1.0, 0.15, 0.7, 2.8], requires_grad=True)
        alpha = torch.tensor(2.0, requires_grad=True)
        q = quantize_activation(x, alpha)
        codes = q.detach() / (alpha.detach() / 15)
        torch.testing.assert_close(codes, codes.round(), atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(codes.min()), 0)
        self.assertLessEqual(float(codes.max()), 15)
        q.sum().backward()
        self.assertEqual(float(x.grad[0]), 0)
        self.assertGreater(float(x.grad[1]), 0)
        self.assertEqual(float(x.grad[-1]), 0)
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertGreater(abs(float(alpha.grad)), 0)

        w = torch.tensor([-3.0, -0.3, 0.2, 3.0], requires_grad=True)
        clip = torch.tensor(2.0, requires_grad=True)
        qw = quantize_weight(w, clip)
        wcodes = qw.detach() / (clip.detach() / 7)
        torch.testing.assert_close(wcodes, wcodes.round(), atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(wcodes.min()), -7)
        self.assertLessEqual(float(wcodes.max()), 7)
        (qw * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
        self.assertGreater(abs(float(w.grad[1])), 0)
        self.assertEqual(float(w.grad[0]), 0)
        self.assertTrue(torch.isfinite(clip.grad))
        self.assertGreater(abs(float(clip.grad)), 0)

    def test_quantized_convolution_trains_weights_and_clip_parameters(self):
        torch.manual_seed(15)
        conv = QuantConv2d(8, 8, 3, padding=1, bias=False)
        conv.set_clipping_(activation=1.0, weight=0.2)
        optimizer = torch.optim.SGD(conv.parameters(), lr=0.01)
        original_weight = conv.weight.detach().clone()
        original_parameter_ids = {name: id(p) for name, p in conv.named_parameters()}
        x = (torch.rand(2, 8, 4, 4) * 2).requires_grad_()
        conv(x).square().mean().backward()
        for name in ('weight', 'raw_act_alpha', 'raw_weight_alpha'):
            parameter = dict(conv.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        optimizer.step()
        self.assertFalse(torch.equal(original_weight, conv.weight.detach()))
        self.assertEqual(original_parameter_ids, {name: id(p) for name, p in conv.named_parameters()})
        self.assertGreater(float(conv.act_alpha.detach()), 0)
        self.assertGreater(float(conv.weight_alpha.detach()), 0)

    def test_vgg_architecture_and_selected_actual_shapes(self):
        model = build_model().eval()
        convolutions = [m for m in model.features if isinstance(m, nn.Conv2d)]
        self.assertEqual(len(convolutions), 13)
        self.assertEqual(len(list(model.named_quantized_layers())), 13)
        self.assertIsInstance(model.features[0], QuantConv2d)
        selected = model.features[27]
        self.assertIsInstance(selected, QuantConv2d)
        self.assertEqual((selected.in_channels, selected.out_channels), (8, 8))
        self.assertIsNone(selected.bias)
        self.assertIsInstance(model.features[28], nn.ReLU)
        self.assertIsInstance(model.features[29], QuantConv2d)
        self.assertEqual(model.features[29].in_channels, 8)
        shapes = {}
        handle = selected.register_forward_hook(lambda _m, args, out: shapes.update(input=tuple(args[0].shape), output=tuple(out.shape)))
        with torch.no_grad():
            prediction = model(torch.randn(1, 3, 32, 32))
        handle.remove()
        self.assertEqual(shapes, {'input': (1, 8, 4, 4), 'output': (1, 8, 4, 4)})
        self.assertEqual(tuple(prediction.shape), (1, 10))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_first_convolution_quantizes_signed_normalized_rgb_inputs(self):
        first = build_model().features[0]
        self.assertIsInstance(first, QuantConv2d)
        self.assertEqual((first.a_bit, first.w_bit), (4, 4))
        first.set_clipping_(activation=3.0)
        values = torch.tensor([-7.0, -1.25, 0.0, 1.25, 7.0])
        images = values.reshape(1, 1, 1, 5).repeat(1, 3, 1, 1).requires_grad_()
        quantized = first.quantized_input(images)
        alpha = float(first.act_alpha.detach())
        expected_codes = torch.tensor([
            round(max(-1.0, min(1.0, float(value) / alpha)) * 7)
            for value in values
        ], dtype=torch.int64).reshape(1, 1, 1, 5).repeat(1, 3, 1, 1)
        torch.testing.assert_close(first.integer_input(images), expected_codes)
        torch.testing.assert_close(quantized.detach(), expected_codes.float() * (alpha / 7), atol=1e-6, rtol=0)
        self.assertLess(float(quantized.detach().min()), 0)
        self.assertGreater(float(quantized.detach().max()), 0)
        quantized.square().sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertEqual(float(images.grad[..., 0].abs().sum()), 0)
        self.assertEqual(float(images.grad[..., -1].abs().sum()), 0)
        self.assertGreater(float(images.grad[..., 1].abs().sum()), 0)
        self.assertIsNotNone(first.raw_act_alpha.grad)
        self.assertTrue(torch.isfinite(first.raw_act_alpha.grad))
        self.assertGreater(float(first.raw_act_alpha.grad), 0)

    def test_course_quantizer_forward_and_saturation_gradient_conventions(self):
        # Independent scalar formulas from the official course recipe:
        # positive activations use 15 intervals; standardized signed weights
        # use 7 intervals per sign. Alpha learns only from saturation.
        x = torch.tensor([0.0, 0.2, 1.3, 2.0, 3.2], requires_grad=True)
        activation_alpha = torch.tensor(2.0, requires_grad=True)
        coefficients = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        actual_activation = quantize_activation(x, activation_alpha)
        expected_activation = torch.tensor([
            round(min(float(v), 2.0) / 2.0 * 15) / 15 * 2.0
            for v in x.detach()
        ])
        torch.testing.assert_close(actual_activation.detach(), expected_activation)
        (actual_activation * coefficients).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0]))
        self.assertEqual(float(activation_alpha.grad), 5.0)

        conv = QuantConv2d(2, 2, 1, bias=False)
        conv.set_clipping_(weight=0.7)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[-3.0]], [[-0.2]]], [[[0.7]], [[2.4]]]]))
        raw = conv.weight.detach().numpy().astype(np.float64)
        mean = float(raw.mean())
        # torch.std() and the course use the sample standard deviation.
        std = float(raw.std(ddof=1))
        alpha = float(conv.weight_alpha.detach())
        expected_values = np.empty_like(raw)
        gradient_coefficients = torch.arange(1, 5, dtype=conv.weight.dtype).reshape_as(conv.weight)
        expected_gradient = np.empty_like(raw)
        expected_alpha_gradient = 0.0
        for index in np.ndindex(raw.shape):
            standardized = (float(raw[index]) - mean) / std
            normalized = standardized / alpha
            sign = 1 if normalized > 0 else (-1 if normalized < 0 else 0)
            expected_values[index] = round(min(abs(normalized), 1.0) * 7) / 7 * sign * alpha
            coefficient = float(gradient_coefficients[index])
            expected_gradient[index] = coefficient / std if abs(normalized) <= 1 else 0.0
            if abs(normalized) > 1:
                expected_alpha_gradient += coefficient * sign
        actual_weight = conv.quantized_weight()
        np.testing.assert_allclose(actual_weight.detach().numpy(), expected_values, atol=2e-7, rtol=0)
        (actual_weight * gradient_coefficients).sum().backward()
        np.testing.assert_allclose(conv.weight.grad.numpy(), expected_gradient, atol=2e-7, rtol=0)
        expected_raw_alpha_gradient = expected_alpha_gradient * float(torch.sigmoid(conv.raw_weight_alpha.detach()))
        self.assertAlmostEqual(float(conv.raw_weight_alpha.grad), expected_raw_alpha_gradient, places=6)

    def test_integer_reference_matches_literal_cross_correlation(self):
        generator = np.random.default_rng(61)
        for shape, wshape, padding, stride in [
            ((2, 8, 4, 4), (8, 8, 3, 3), (1, 1), (1, 1)),
            ((1, 3, 5, 7), (2, 3, 2, 3), (0, 1), (2, 2)),
        ]:
            with self.subTest(shape=shape):
                x = generator.integers(0, 16, size=shape, dtype=np.int64)
                w = generator.integers(-7, 8, size=wshape, dtype=np.int64)
                oracle = loop_cross_correlation(x, w, padding, stride)
                actual = exact_conv2d_int64(x, w, padding=padding, stride=stride)
                np.testing.assert_array_equal(actual, oracle)
        with self.assertRaises(TypeError):
            exact_conv2d_int64(x.astype(np.float32), w)

    def test_downstream_prehook_reconstruction_and_hardware_file_readback(self):
        model = HookFixture().train()
        model.features[29].eval()  # Check restoration of mixed per-module states.
        model.features[27].quantization_enabled = False
        model.features[29].quantization_enabled = False
        states = {name: m.training for name, m in model.named_modules()}
        flags = {name: m.quantization_enabled for name, m in model.named_modules() if isinstance(m, QuantConv2d)}
        generator = torch.Generator().manual_seed(313)
        images = torch.randint(0, 16, (2, 8, 4, 4), generator=generator).float() * 0.25
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            report = verify_and_export(model, images, path, sample_index=1, provenance={'evidence_type': 'synthetic'})
            self.assertEqual(report['status'], 'PASS')
            self.assertFalse(report['trained_model_evidence'])
            self.assertLess(report['max_abs_error'], 1e-5)
            self.assertTrue(report['reference_is_exact_relu_of_selected_forward_output'])
            self.assertEqual(states, {name: m.training for name, m in model.named_modules()})
            self.assertEqual(flags, {name: m.quantization_enabled for name, m in model.named_modules() if isinstance(m, QuantConv2d)})
            data = np.load(path / 'tensors.npz')
            x = data['activation_int']
            w = data['weight_int']
            oracle = loop_cross_correlation(x, w)
            np.testing.assert_array_equal(data['psum_int'], oracle)
            np.testing.assert_array_equal(data['psum_relu_int'], oracle.clip(0))
            np.testing.assert_allclose(data['recovered'], data['reference_next_input'], atol=1e-5, rtol=0)
            np.testing.assert_array_equal(data['sample_input'], images[1:2].numpy())

            activation_words = (path / 'input.txt').read_text().split()[3:]
            self.assertEqual(len(activation_words), 36)
            self.assertTrue(all(len(word) == 32 and set(word) <= {'0', '1'} for word in activation_words))
            decoded_x = np.empty((8, 6, 6), dtype=np.int64)
            for position, word in enumerate(activation_words):
                for channel in range(8):
                    decoded_x[channel, position // 6, position % 6] = decode_word(word, 4, channel)
            np.testing.assert_array_equal(decoded_x[:, 1:5, 1:5], x[0])
            np.testing.assert_array_equal(decoded_x[:, [0, 5], :], 0)
            np.testing.assert_array_equal(decoded_x[:, :, [0, 5]], 0)

            # Model the core's 9 kernel passes and its SRAM address gather.
            pmem = np.zeros((9 * 36, 8), dtype=np.int64)
            decoded_w = np.empty_like(w)
            for kernel in range(9):
                words = (path / f'weights/weight{kernel}.txt').read_text().split()[3:]
                self.assertEqual(len(words), 8)
                for output_channel, word in enumerate(words):
                    self.assertEqual(len(word), 32)
                    for channel in range(8):
                        decoded_w[output_channel, channel, kernel // 3, kernel % 3] = decode_word(word, 4, channel, signed=True)
                    for position in range(36):
                        total = 0
                        for channel in range(8):
                            total += int(decoded_x[channel, position // 6, position % 6]) * int(decoded_w[output_channel, channel, kernel // 3, kernel % 3])
                        pmem[kernel * 36 + position, output_channel] = total
            np.testing.assert_array_equal(decoded_w, w)
            address_words = (path / 'acc_address.txt').read_text().split()
            self.assertEqual(len(address_words), 16 * 9)
            self.assertTrue(all(len(word) == 11 for word in address_words))
            addresses = np.array([int(word, 2) for word in address_words]).reshape(16, 9)
            gathered = pmem[addresses].sum(axis=1).clip(0)
            expected_words = (path / 'psum.txt').read_text().split()
            decoded_psum = np.array([[decode_word(word, 16, channel, signed=True) for channel in range(8)] for word in expected_words])
            self.assertEqual(decoded_psum.shape, (16, 8))
            np.testing.assert_array_equal(gathered, oracle[0].reshape(8, 16).T.clip(0))
            np.testing.assert_array_equal(decoded_psum, gathered)
            self.assertEqual((path / 'input.txt').read_bytes(), (path / 'activation/activation.txt').read_bytes())
            self.assertEqual((path / 'psum.txt').read_bytes(), (path / 'output/output_after_relu.txt').read_bytes())
            saved = json.loads((path / 'verification.json').read_text())
            self.assertEqual(saved['model_sha256'], report['model_sha256'])

    def test_actual_downstream_change_cannot_pass_using_a_computed_reference(self):
        model = HookFixture(corrupt_downstream=True)
        images = torch.ones(1, 8, 4, 4) * 2
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(VerificationError):
                verify_and_export(model, images, folder, provenance={'evidence_type': 'synthetic'}, raise_on_failure=True)
            report = json.loads((Path(folder) / 'verification.json').read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertFalse(report['reference_is_exact_relu_of_selected_forward_output'])
            self.assertGreater(report['max_abs_error'], 1e-3)
            self.assertTrue((Path(folder) / 'tensors.npz').exists())


if __name__ == '__main__':
    unittest.main()
