"""Verify a quantized 8x8 layer against its actual downstream input and export RTL vectors.

The integer convolution is evaluated in NumPy int64, independently of torch's
floating point convolution. The comparison target is captured by a pre-hook on
the next convolution during a real, quantized model forward pass. A detached
clone is essential: hooks otherwise retain tensors that an in-place operation
may later change.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class VerificationError(RuntimeError):
    """Raised on request after a failed comparison has been recorded on disk."""


def model_fingerprint(model: torch.nn.Module) -> str:
    """Hash all model parameters and persistent buffers, including their layout."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def exact_conv2d_int64(
    activation: np.ndarray,
    weight: np.ndarray,
    *,
    padding: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Cross-correlate NCHW activation with OIHW weights using int64 throughout.

    This is not a floating point convolution followed by a cast. For this
    project's 4-bit, 8-input-channel, 3x3 convolution even int16 can hold each
    final sum; int64 leaves ample headroom for the independent reference.
    """
    activation = np.asarray(activation)
    weight = np.asarray(weight)
    if activation.dtype != np.int64 or weight.dtype != np.int64:
        raise TypeError("Exact convolution requires int64 activation and weight arrays.")
    if activation.ndim != 4 or weight.ndim != 4:
        raise ValueError("Expected activation NCHW and weight OIHW arrays.")
    if activation.shape[1] != weight.shape[1]:
        raise ValueError("Activation and weight input channel counts differ.")
    if any(v < 0 for v in padding) or any(v < 1 for v in stride):
        raise ValueError("Padding must be nonnegative and stride must be positive.")
    ph, pw = padding
    kh, kw = weight.shape[-2:]
    padded = np.pad(activation, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    if padded.shape[2] < kh or padded.shape[3] < kw:
        raise ValueError("Kernel is larger than the padded activation.")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kh, kw), axis=(2, 3)
    )[:, :, :: stride[0], :: stride[1], :, :]
    return np.einsum(
        "ncyxij,ocij->noyx", windows, weight, dtype=np.int64, optimize=True
    )


def _numpy(value: torch.Tensor, *, dtype: Any | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy().copy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _scalar(value: torch.Tensor, name: str) -> float:
    if value.numel() != 1:
        raise ValueError(f"{name} must be one scalar for this RTL mapping.")
    result = float(value.detach().cpu().item())
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, got {result}.")
    return result


def _packed_word(values: np.ndarray, bits: int, *, signed: bool) -> str:
    """Pack the highest channel first, so channel 0 occupies the word's LSBs."""
    flat = np.asarray(values).reshape(-1)
    minimum = -(1 << (bits - 1)) if signed else 0
    maximum = (1 << (bits - (1 if signed else 0))) - 1
    if np.any(flat < minimum) or np.any(flat > maximum):
        raise ValueError(f"Value does not fit {'signed' if signed else 'unsigned'} {bits} bits.")
    mask = (1 << bits) - 1
    return "".join(format(int(value) & mask, f"0{bits}b") for value in flat[::-1])


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _vector_exports(
    output_dir: Path,
    activation_int: np.ndarray,
    weight_int: np.ndarray,
    psum_int: np.ndarray,
) -> dict[str, Any]:
    """Emit the legacy Part3alpha TB layout and explicitly documented aliases."""
    if activation_int.shape != (1, 8, 4, 4):
        raise ValueError(
            "The supplied legacy core_tb.v requires one 8x4x4 activation; "
            f"received {activation_int.shape}."
        )
    if weight_int.shape != (8, 8, 3, 3) or psum_int.shape != (1, 8, 4, 4):
        raise ValueError("Legacy vector export requires 8x8x3x3 weights and 8x4x4 output.")

    padded = np.pad(activation_int[0], ((0, 0), (1, 1), (1, 1)))
    activation_header = [
        "#time0:channel7_to_channel0:4bits_each#",
        "#row_major_padded_6x6:36_words#",
        "#three_header_tokens_then_binary_words#",
    ]
    activation_lines = [
        _packed_word(padded[:, y, x], 4, signed=False)
        for y in range(6)
        for x in range(6)
    ]
    _write_lines(output_dir / "activation/activation.txt", activation_header + activation_lines)
    _write_lines(output_dir / "input.txt", activation_header + activation_lines)

    all_weight_lines: list[str] = []
    weight_paths: list[str] = []
    for ky in range(3):
        for kx in range(3):
            kij = ky * 3 + kx
            lines = [
                _packed_word(weight_int[oc, :, ky, kx], 4, signed=True)
                for oc in range(8)
            ]
            header = [
                "#output_channel0_to_7:one_word_each#",
                "#input_channel7_to_0:signed4_twos_complement#",
                f"#kernel_y{ky}_x{kx}#",
            ]
            name = f"weights/weight{kij}.txt"
            _write_lines(output_dir / name, header + lines)
            weight_paths.append(name)
            all_weight_lines.extend(lines)
    _write_lines(
        output_dir / "weight.txt",
        [
            "#nine_row_major_kernel_positions#",
            "#eight_output_channel_words_per_position#",
            "#input_channel7_to_0:signed4_twos_complement#",
        ]
        + all_weight_lines,
    )

    relu_int = np.maximum(psum_int, 0)
    output_lines = [
        _packed_word(relu_int[0, :, y, x], 16, signed=True)
        for y in range(4)
        for x in range(4)
    ]
    raw_lines = [
        _packed_word(psum_int[0, :, y, x], 16, signed=True)
        for y in range(4)
        for x in range(4)
    ]
    _write_lines(output_dir / "output/output_after_relu.txt", output_lines)
    _write_lines(output_dir / "output/output_updated.txt", raw_lines)
    _write_lines(output_dir / "psum.txt", output_lines)

    # The RTL stores 36 padded-position dot products for each kernel position.
    # SFU gathers the nine relevant positions for each unpadded output pixel.
    addresses = [
        (ky * 3 + kx) * 36 + (y + ky) * 6 + x + kx
        for y in range(4)
        for x in range(4)
        for ky in range(3)
        for kx in range(3)
    ]
    _write_lines(output_dir / "acc_address.txt", [format(a, "011b") for a in addresses])
    return {
        "compatibility": "Part3alpha/verilog/core_tb.v, run from export directory",
        "activation": "activation/activation.txt",
        "weights": weight_paths,
        "expected_relu_output": "output/output_after_relu.txt",
        "expected_pre_relu_output": "output/output_updated.txt",
        "accumulation_addresses": "acc_address.txt",
        "aliases": {
            "input.txt": "Byte-identical alias of activation/activation.txt",
            "weight.txt": "Three header tokens, then nine kernel blocks of eight words",
            "psum.txt": "Byte-identical alias of output/output_after_relu.txt",
        },
        "header_handling": "Skip exactly three whitespace-delimited tokens in input/weight files; output/address files have no headers.",
        "activation_words": 36,
        "activation_word_bits": 32,
        "weight_words_per_kernel": 8,
        "weight_word_bits": 32,
        "output_words": 16,
        "output_word_bits": 128,
        "address_words": 144,
        "address_word_bits": 11,
        "word_order": "Channel 0 occupies least significant bits; binary text displays channel 7 first. Pixels and kernel positions are row-major.",
        "activation_encoding": "unsigned 4-bit (0..15)",
        "weight_encoding": "signed 4-bit two's complement (-7..7)",
        "psum_encoding": "signed 16-bit two's complement",
        "psum_domain": "Integer accumulation units, after ReLU for psum.txt; multiply by activation_scale*weight_scale for model-domain values.",
    }


def verify_and_export(
    model: torch.nn.Module,
    images: torch.Tensor,
    output_dir: str | Path,
    *,
    sample_index: int = 0,
    layer_name: str = "features.27",
    next_layer_name: str = "features.29",
    provenance: Mapping[str, Any] | None = None,
    strict_tolerance: float = 1e-5,
    formal_tolerance: float = 1e-3,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Run an actual quantized forward pass, verify the selected layer, export.

    ``images`` is a normalized NCHW batch, and ``sample_index`` selects one image
    within that batch. The caller supplies any dataset index/checkpoint identity
    in ``provenance``. Mark synthetic runs with ``evidence_type='synthetic'``;
    only an explicitly identified trained model is labeled trained evidence.

    All existing module training and quantization flags are restored. A strict
    failure remains a failure in the returned/saved report even when the formal
    tolerance passes. With ``raise_on_failure``, failures are raised only after
    writing vectors and diagnostics, so the result remains inspectable.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a normalized NCHW torch.Tensor batch.")
    if not 0 <= sample_index < images.shape[0]:
        raise IndexError("sample_index is outside the supplied batch.")
    if not (math.isfinite(strict_tolerance) and 0 < strict_tolerance <= formal_tolerance):
        raise ValueError("Require 0 < strict_tolerance <= formal_tolerance.")
    if not math.isfinite(formal_tolerance):
        raise ValueError("formal_tolerance must be finite.")
    modules = dict(model.named_modules())
    if layer_name not in modules or next_layer_name not in modules:
        raise ValueError("Selected or next layer name was not found in the model.")
    layer = modules[layer_name]
    next_layer = modules[next_layer_name]
    if not isinstance(layer, torch.nn.Conv2d) or not isinstance(next_layer, torch.nn.Conv2d):
        raise ValueError("Both verification hook targets must be real convolution modules.")
    if layer.in_channels != 8 or layer.out_channels != 8:
        raise ValueError("The selected layer must have exactly eight input/output channels.")
    if layer.bias is not None or layer.groups != 1 or tuple(layer.dilation) != (1, 1):
        raise ValueError("This RTL mapping requires bias=False, groups=1, dilation=1.")
    if tuple(layer.kernel_size) != (3, 3) or tuple(layer.stride) != (1, 1) or tuple(layer.padding) != (1, 1):
        raise ValueError("This RTL mapping requires kernel=3x3, stride=1, padding=1.")
    for method in ("integer_input", "integer_weight", "quantized_input", "quantized_weight", "activation_scale", "weight_scale"):
        if not callable(getattr(layer, method, None)):
            raise TypeError(f"Selected quantized convolution is missing {method}().")
    if int(layer.a_bit) != 4 or int(layer.w_bit) != 4:
        raise ValueError("This export requires 4-bit activation and weight quantization.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_flags = {name: module.training for name, module in modules.items()}
    quantization_flags = {
        name: module.quantization_enabled
        for name, module in modules.items()
        if hasattr(module, "quantization_enabled")
    }
    selected_inputs: list[torch.Tensor] = []
    selected_outputs: list[torch.Tensor] = []
    downstream_inputs: list[torch.Tensor] = []

    def selected_pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        selected_inputs.append(args[0].detach().clone())

    def selected_post_hook(_module: torch.nn.Module, _args: Any, result: torch.Tensor) -> None:
        selected_outputs.append(result.detach().clone())

    def downstream_pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        downstream_inputs.append(args[0].detach().clone())

    handles = [
        layer.register_forward_pre_hook(selected_pre_hook),
        layer.register_forward_hook(selected_post_hook),
        next_layer.register_forward_pre_hook(downstream_pre_hook),
    ]
    sample = images[sample_index : sample_index + 1].detach().to(
        device=layer.weight.device, dtype=layer.weight.dtype
    )
    previous_tf32 = None
    if sample.device.type == "cuda":
        previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        model.eval()
        for name in quantization_flags:
            modules[name].quantization_enabled = True
        with torch.no_grad():
            model(sample)
            if not (len(selected_inputs) == len(selected_outputs) == len(downstream_inputs) == 1):
                raise ValueError("Expected exactly one selected-layer and next-layer call per forward pass.")
            selected_input = selected_inputs[0]
            activation_int = _numpy(layer.integer_input(selected_input), dtype=np.int64)
            weight_int = _numpy(layer.integer_weight(), dtype=np.int64)
            quantized_input = _numpy(layer.quantized_input(selected_input), dtype=np.float64)
            quantized_weight = _numpy(layer.quantized_weight(), dtype=np.float64)
            activation_scale = _scalar(layer.activation_scale(), "activation_scale")
            weight_scale = _scalar(layer.weight_scale(), "weight_scale")
    finally:
        for handle in handles:
            handle.remove()
        for name, state in training_flags.items():
            modules[name].training = state
        for name, state in quantization_flags.items():
            modules[name].quantization_enabled = state
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32

    if np.any(activation_int < 0) or np.any(activation_int > 15):
        raise ValueError("Activation integer codes are outside unsigned 4-bit range.")
    if np.any(weight_int < -7) or np.any(weight_int > 7):
        raise ValueError("Weight integer codes are outside the signed symmetric 4-bit range.")
    if not np.allclose(quantized_input / activation_scale, activation_int, rtol=0, atol=1e-5):
        raise ValueError("Quantized activation does not match its integer codes/scalar scale.")
    if not np.allclose(quantized_weight / weight_scale, weight_int, rtol=0, atol=1e-5):
        raise ValueError("Quantized weight does not match its integer codes/scalar scale.")

    psum_int = exact_conv2d_int64(activation_int, weight_int)
    psum_relu_int = np.maximum(psum_int, 0)
    recovered = psum_relu_int.astype(np.float64) * activation_scale * weight_scale
    recovered_pre_relu = psum_int.astype(np.float64) * activation_scale * weight_scale
    reference = _numpy(downstream_inputs[0], dtype=np.float64)
    selected_output = _numpy(selected_outputs[0], dtype=np.float64)
    if recovered.shape != reference.shape:
        raise ValueError(f"Recovered/reference shapes differ: {recovered.shape} vs {reference.shape}.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(recovered)):
        raise ValueError("Reference/recovered output contains NaN or infinity.")

    difference = recovered - reference
    max_abs_error = float(np.max(np.abs(difference)))
    strict_pass = bool(max_abs_error < strict_tolerance)
    formal_pass = bool(max_abs_error < formal_tolerance)
    fp_conv_error = float(np.max(np.abs(recovered_pre_relu - selected_output)))
    expected_relu = np.maximum(selected_output, 0)
    downstream_is_relu = bool(np.array_equal(expected_relu, reference))
    if not downstream_is_relu:
        strict_pass = False
        formal_pass = False
    vectors = _vector_exports(output_dir, activation_int, weight_int, psum_int)
    fingerprint = model_fingerprint(model)
    provenance_data = dict(provenance or {})
    evidence_type = str(provenance_data.get("evidence_type", "unidentified_model"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if strict_pass else ("FORMAL_PASS_STRICT_FAIL" if formal_pass else "FAIL"),
        "evidence_type": evidence_type,
        "trained_model_evidence": evidence_type == "trained_model",
        "accuracy_claim": "This single-layer check does not establish dataset classification accuracy; use the separate full-test-set evaluation.",
        "provenance": provenance_data,
        "model_sha256": fingerprint,
        "layer_name": layer_name,
        "next_layer_name": next_layer_name,
        "reference_source": "Actual next-convolution forward pre-hook input, captured with detach().clone() during a full quantized model forward pass.",
        "reference_is_exact_relu_of_selected_forward_output": downstream_is_relu,
        "integer_arithmetic": "numpy int64 sliding windows and einsum; no floating point convolution in integer reference",
        "recovery_arithmetic": "float64(int64_psum_after_relu) * recorded activation_scale * recorded weight_scale",
        "activation_scale": activation_scale,
        "weight_scale": weight_scale,
        "activation_bits": 4,
        "weight_bits": 4,
        "activation_integer_min": int(activation_int.min()),
        "activation_integer_max": int(activation_int.max()),
        "weight_integer_min": int(weight_int.min()),
        "weight_integer_max": int(weight_int.max()),
        "activation_shape_nchw": list(activation_int.shape),
        "weight_shape_oihw": list(weight_int.shape),
        "output_shape_nchw": list(psum_int.shape),
        "input_model_dtype": str(sample.dtype),
        "input_model_device": str(sample.device),
        "sample_index_in_batch": sample_index,
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "signed_error_sum_diagnostic_only": float(np.sum(difference)),
        "max_abs_error_pre_relu_vs_selected_fp_forward": fp_conv_error,
        "strict_tolerance": strict_tolerance,
        "formal_tolerance": formal_tolerance,
        "comparison_operator": "strictly less than (<), elementwise max absolute error",
        "strict_pass": strict_pass,
        "formal_pass": formal_pass,
        "arrays_file": "tensors.npz",
        "vectors": vectors,
    }
    if callable(getattr(model, "architecture_config", None)):
        report["architecture"] = model.architecture_config()
    np.savez_compressed(
        output_dir / "tensors.npz",
        sample_input=_numpy(sample),
        selected_input=_numpy(selected_input),
        activation_int=activation_int,
        activation_padded_int=np.pad(activation_int, ((0, 0), (0, 0), (1, 1), (1, 1))),
        weight_int=weight_int,
        psum_int=psum_int,
        psum_relu_int=psum_relu_int,
        recovered=recovered,
        recovered_pre_relu=recovered_pre_relu,
        reference_next_input=reference,
        selected_forward_output=selected_output,
        quantized_activation=quantized_input,
        quantized_weight=quantized_weight,
        activation_scale=np.asarray(activation_scale, dtype=np.float64),
        weight_scale=np.asarray(weight_scale, dtype=np.float64),
        difference=difference,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output_dir / "verification.json").write_text(encoded, encoding="utf-8")
    (output_dir / "metadata.json").write_text(encoded, encoding="utf-8")
    if raise_on_failure and not strict_pass:
        raise VerificationError(
            f"{report['status']}: max_abs_error={max_abs_error:.9g}, "
            f"strict threshold={strict_tolerance:g}; see {output_dir / 'verification.json'}."
        )
    return report
