"""CIFAR-10 VGG16 with an 8-to-8 convolution and W4A4 fake quantization.

This is a self-contained reconstruction of the architecture printed in the
reference notebook, not the missing original ``models.vgg_quant`` module.  It
contains 13 convolutional layers and the course's single CIFAR-10 classifier.
All 13 convolutions quantize their incoming activation and weights to four
bits in their forward pass.  The first convolution uses signed activation
codes -7..7 because normalized RGB input can be negative.  Subsequent
convolutions receive ReLU activations and use unsigned codes 0..15.  Every
convolution uses symmetric signed weight codes -7..7.  Only the final linear
classifier remains floating point.

The layer mapped to the systolic array is ``features.27``.  Its only following
operation before ``features.29`` is ReLU: there is no intervening BatchNorm.
Quantized convolutions follow the official course quantizer's detached
mean/std weight standardization and saturation-only alpha gradients.  Their
default weight/unsigned-activation clipping limits are 3/8.  The first signed
activation clipping limit is 3.  Clipping limits are learned
positive scalars, parameterized with softplus.  Floating-point warm-up disables
rounding and clipping while retaining this weight standardization.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCALE_EPS = 1e-8
FEATURE_CONFIGURATION = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    8, 8, 512, "M", 512, 512, 512, "M",
)
SELECTED_LAYER_NAME = "features.27"
NEXT_LAYER_NAME = "features.29"


def _check_bits(bits: int) -> int:
    if not isinstance(bits, int) or not 2 <= bits <= 16:
        raise ValueError("Quantization bit width must be an integer from 2 to 16.")
    return bits


def _inverse_softplus(value: float) -> float:
    """A stable inverse for positive scalar initialization."""
    if value <= SCALE_EPS:
        raise ValueError("A clipping limit must be larger than SCALE_EPS.")
    value -= SCALE_EPS
    return value + math.log(-math.expm1(-value))


def straight_through_round(value: Tensor) -> Tensor:
    """Round in the forward pass and pass an identity derivative backward."""
    return value + (torch.round(value) - value).detach()


class _ActivationFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, alpha: Tensor, levels: int) -> Tensor:
        normalized = value / alpha
        ctx.save_for_backward(normalized)
        ctx.alpha_shape = alpha.shape
        return torch.round(normalized.clamp(0.0, 1.0) * levels) / levels * alpha

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, Tensor, None]:
        (normalized,) = ctx.saved_tensors
        # The course STE passes gradients through unsaturated values and learns
        # the clipping limit only from values clipped at the upper boundary.
        value_gradient = gradient * ((normalized >= 0.0) & (normalized <= 1.0))
        alpha_gradient = (gradient * (normalized > 1.0)).sum_to_size(ctx.alpha_shape)
        return value_gradient, alpha_gradient, None


class _WeightFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, alpha: Tensor, levels: int) -> Tensor:
        normalized = value / alpha
        ctx.save_for_backward(normalized)
        ctx.alpha_shape = alpha.shape
        clipped = normalized.clamp(-1.0, 1.0)
        return torch.round(clipped.abs() * levels) / levels * clipped.sign() * alpha

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, Tensor, None]:
        (normalized,) = ctx.saved_tensors
        saturated = normalized.abs() > 1.0
        value_gradient = gradient * ~saturated
        alpha_gradient = (
            gradient * normalized.sign() * saturated
        ).sum_to_size(ctx.alpha_shape)
        return value_gradient, alpha_gradient, None


def quantize_activation(value: Tensor, alpha: Tensor, bits: int = 4) -> Tensor:
    """Course PACT-style unsigned fake quantization of a ReLU activation."""
    levels = (1 << _check_bits(bits)) - 1
    return _ActivationFakeQuant.apply(value, alpha.clamp_min(SCALE_EPS), levels)


def quantize_weight(value: Tensor, alpha: Tensor, bits: int = 4) -> Tensor:
    """Symmetric signed fake quantization (4 bits use -7 through +7)."""
    levels = (1 << (_check_bits(bits) - 1)) - 1
    return _WeightFakeQuant.apply(value, alpha.clamp_min(SCALE_EPS), levels)


def quantize_signed_activation(value: Tensor, alpha: Tensor, bits: int = 4) -> Tensor:
    """Symmetric signed fake quantization for the first normalized RGB input.

    The clipping/STE formula matches signed weight quantization, but this
    function does not standardize the input or change its normalization.
    """
    levels = (1 << (_check_bits(bits) - 1)) - 1
    return _WeightFakeQuant.apply(value, alpha.clamp_min(SCALE_EPS), levels)


class QuantConv2d(nn.Conv2d):
    """A convolution whose activation and weight quantization are in forward.

    ``quantized_input`` and ``quantized_weight`` share exactly the same
    quantizers as forward.  The integer helpers return detached int64 codes,
    which makes them suitable for independent integer verification and export.
    """

    def __init__(
        self,
        *args: Any,
        a_bit: int = 4,
        w_bit: int = 4,
        act_clip_init: float | None = None,
        weight_clip_init: float = 3.0,
        quantization_enabled: bool = True,
        signed_activation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.a_bit = _check_bits(a_bit)
        self.w_bit = _check_bits(w_bit)
        self.quantization_enabled = bool(quantization_enabled)
        self.signed_activation = bool(signed_activation)
        if act_clip_init is None:
            act_clip_init = 3.0 if self.signed_activation else 8.0
        self.raw_act_alpha = nn.Parameter(
            self.weight.new_tensor(_inverse_softplus(float(act_clip_init)))
        )
        self.raw_weight_alpha = nn.Parameter(
            self.weight.new_tensor(_inverse_softplus(float(weight_clip_init)))
        )

    @property
    def act_alpha(self) -> Tensor:
        return F.softplus(self.raw_act_alpha) + SCALE_EPS

    @property
    def weight_alpha(self) -> Tensor:
        return F.softplus(self.raw_weight_alpha) + SCALE_EPS

    def activation_scale(self) -> Tensor:
        levels = (1 << (self.a_bit - int(self.signed_activation))) - 1
        return self.act_alpha / levels

    def weight_scale(self) -> Tensor:
        return self.weight_alpha / ((1 << (self.w_bit - 1)) - 1)

    def quantized_input(self, value: Tensor) -> Tensor:
        if self.signed_activation:
            return quantize_signed_activation(value, self.act_alpha, self.a_bit)
        return quantize_activation(value, self.act_alpha, self.a_bit)

    def normalized_weight(self) -> Tensor:
        """Course weight standardization; statistics do not receive gradients."""
        mean = self.weight.detach().mean()
        std = self.weight.detach().std().clamp_min(SCALE_EPS)
        return (self.weight - mean) / std

    def quantized_weight(self) -> Tensor:
        return quantize_weight(self.normalized_weight(), self.weight_alpha, self.w_bit)

    @torch.no_grad()
    def integer_input(self, value: Tensor) -> Tensor:
        levels = (1 << (self.a_bit - int(self.signed_activation))) - 1
        minimum = -1.0 if self.signed_activation else 0.0
        return torch.round(torch.clamp(value / self.act_alpha, minimum, 1.0) * levels).to(
            dtype=torch.int64
        )

    @torch.no_grad()
    def integer_weight(self) -> Tensor:
        levels = (1 << (self.w_bit - 1)) - 1
        return torch.round(torch.clamp(self.normalized_weight() / self.weight_alpha, -1.0, 1.0) * levels).to(
            dtype=torch.int64
        )

    @torch.no_grad()
    def set_clipping_(
        self, *, activation: float | None = None, weight: float | None = None
    ) -> "QuantConv2d":
        """Set positive clipping values without replacing optimizer parameters."""
        if activation is not None:
            self.raw_act_alpha.fill_(_inverse_softplus(float(activation)))
        if weight is not None:
            self.raw_weight_alpha.fill_(_inverse_softplus(float(weight)))
        return self

    @torch.no_grad()
    def initialize_weight_clipping_(self, standard_deviations: float = 3.0) -> None:
        if standard_deviations <= 0:
            raise ValueError("standard_deviations must be positive.")
        # Quantization is applied AFTER standardization, so alpha is measured
        # in normalized standard deviations, not in raw parameter units.
        self.set_clipping_(weight=float(standard_deviations))

    def forward(self, value: Tensor) -> Tensor:
        if self.quantization_enabled:
            value = self.quantized_input(value)
            weight = self.quantized_weight()
        else:
            weight = self.normalized_weight()
        return self._conv_forward(value, weight, self.bias)

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f", a_bit={self.a_bit}, w_bit={self.w_bit}, "
            f"signed_activation={self.signed_activation}"
        )


class VGG16QAT(nn.Module):
    """The course CIFAR-10 VGG16, with ``features[27]`` squeezed to 8x8."""

    selected_layer_name = SELECTED_LAYER_NAME
    next_layer_name = NEXT_LAYER_NAME

    def __init__(self, num_classes: int = 10, quantize: bool = True) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        layers: list[nn.Module] = []
        input_channels = 3
        convolution_index = 0
        for width in FEATURE_CONFIGURATION:
            if width == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            output_channels = int(width)
            layers.append(
                QuantConv2d(
                    input_channels, output_channels, kernel_size=3,
                    stride=1, padding=1, bias=False,
                    quantization_enabled=bool(quantize),
                    signed_activation=convolution_index == 0,
                )
            )
            # The ninth convolution is features[27]; omit only its BatchNorm.
            if convolution_index != 8:
                layers.append(nn.BatchNorm2d(output_channels))
            # Explicitly avoid in-place aliasing of saved verification tensors.
            layers.append(nn.ReLU(inplace=False))
            input_channels = output_channels
            convolution_index += 1
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, self.num_classes)
        self._initialize_parameters()
        self._check_architecture()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                nn.init.zeros_(module.bias)
        self.initialize_weight_clipping_()

    def _check_architecture(self) -> None:
        selected = self.features[27]
        following = self.features[29]
        assert isinstance(selected, QuantConv2d)
        assert selected.in_channels == 8 and selected.out_channels == 8
        assert selected.bias is None
        assert isinstance(self.features[28], nn.ReLU)
        assert isinstance(following, QuantConv2d) and following.in_channels == 8
        assert sum(isinstance(module, nn.Conv2d) for module in self.features) == 13
        assert sum(isinstance(module, QuantConv2d) for module in self.features) == 13
        assert self.features[0].signed_activation

    def named_quantized_layers(self) -> Iterator[tuple[str, QuantConv2d]]:
        for name, module in self.named_modules():
            if isinstance(module, QuantConv2d):
                yield name, module

    def set_quantization(self, enabled: bool = True) -> "VGG16QAT":
        """Enable W4A4 QAT, or temporarily disable it for FP warm-up."""
        for _, module in self.named_quantized_layers():
            module.quantization_enabled = bool(enabled)
        return self

    @torch.no_grad()
    def initialize_weight_clipping_(self, standard_deviations: float = 3.0) -> None:
        """Reinitialize learned weight clips, e.g. after a floating-point warm-up."""
        for _, module in self.named_quantized_layers():
            module.initialize_weight_clipping_(standard_deviations)

    def architecture_config(self) -> dict[str, Any]:
        """Serializable architecture and precision metadata for checkpoints."""
        return {
            "model": "VGG16QAT",
            "dataset": "CIFAR10",
            "num_classes": self.num_classes,
            "input_shape": [3, 32, 32],
            "feature_configuration": list(FEATURE_CONFIGURATION),
            "convolution_count": 13,
            "quantized_convolution_count": 13,
            "activation_bits": 4,
            "weight_bits": 4,
            "activation_integer_range": [0, 15],
            "first_activation_integer_range": [-7, 7],
            "first_activation_signed": True,
            "weight_integer_range": [-7, 7],
            "floating_point_layers": ["classifier"] + [
                name for name, module in self.named_modules()
                if isinstance(module, nn.BatchNorm2d)
            ],
            "selected_layer": self.selected_layer_name,
            "next_layer": self.next_layer_name,
            "selected_layer_shape": [8, 8, 3, 3],
            "selected_output_shape": [8, 4, 4],
            "selected_batchnorm": False,
            "quantization_enabled": all(
                module.quantization_enabled for _, module in self.named_quantized_layers()
            ),
            "scale_parameterization": "softplus(raw_alpha) + 1e-8",
            "weight_standardization": "detached mean/std before weight quantization",
            "initial_weight_alpha": 3.0,
            "initial_activation_alpha": 8.0,
            "initial_first_activation_alpha": 3.0,
            "warmup": "disable rounding/clipping; retain weight standardization",
        }

    def forward(self, value: Tensor) -> Tensor:
        value = self.features(value)
        value = torch.flatten(value, 1)
        return self.classifier(value)


def build_model(quantize: bool = True, num_classes: int = 10) -> VGG16QAT:
    """Construct the architecture; this function does not load trained weights."""
    return VGG16QAT(num_classes=num_classes, quantize=quantize)


__all__ = [
    "FEATURE_CONFIGURATION", "SELECTED_LAYER_NAME", "NEXT_LAYER_NAME",
    "QuantConv2d", "VGG16QAT", "build_model", "quantize_activation",
    "quantize_weight", "quantize_signed_activation", "straight_through_round",
]
