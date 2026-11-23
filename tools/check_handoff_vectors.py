#!/usr/bin/env python3
"""Independently check ECE284 trained-layer handoff vectors using NumPy only.

Example from a handoff package root:
    python tools/check_handoff_vectors.py --vectors artifacts/vectors

This checks the supplied vectors and preserved reference tensors. It does not
run an RTL simulator, load a PyTorch checkpoint, or evaluate CIFAR-10 accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


class CheckFailure(ValueError):
    """A supplied vector or associated record failed an independent check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def read_words(path: Path, bits: int, count: int, *, headers: bool = False) -> list[str]:
    tokens = path.read_text(encoding="ascii").split()
    if headers:
        require(len(tokens) >= 3, f"{path.name}: missing three header tokens")
        require(all(t.startswith("#") for t in tokens[:3]), f"{path.name}: invalid headers")
        tokens = tokens[3:]
    require(len(tokens) == count, f"{path.name}: expected {count} words, got {len(tokens)}")
    require(
        all(len(t) == bits and set(t) <= {"0", "1"} for t in tokens),
        f"{path.name}: expected binary words of width {bits}",
    )
    return tokens


def unpack(words: list[str], lane_bits: int, lanes: int, signed: bool) -> np.ndarray:
    """Decode from the integer word's low bits, independently of exporter packing."""
    decoded = []
    mask = (1 << lane_bits) - 1
    for word in words:
        value = int(word, 2)
        row = []
        for lane in range(lanes):
            code = (value >> (lane * lane_bits)) & mask
            if signed and code >= (1 << (lane_bits - 1)):
                code -= 1 << lane_bits
            row.append(code)
        decoded.append(row)
    return np.asarray(decoded, dtype=np.int64)


def equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    require(actual.shape == expected.shape, f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
    mismatch_count = int(np.count_nonzero(actual != expected))
    require(mismatch_count == 0, f"{name}: {mismatch_count} mismatched scalar values")


def close(actual: np.ndarray, expected: np.ndarray, name: str, tolerance: float = 1e-12) -> None:
    require(actual.shape == expected.shape, f"{name}: shape mismatch")
    require(bool(np.all(np.isfinite(actual))) and bool(np.all(np.isfinite(expected))), f"{name}: nonfinite values")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    require(error <= tolerance, f"{name}: maximum absolute discrepancy {error:.12g}")


def scalar_close(actual: float, expected: float, name: str) -> None:
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f"{name}: nonfinite value")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), f"{name}: recorded {expected!r}, computed {actual!r}")


def check_vectors(directory: Path) -> dict:
    directory = directory.resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    require(metadata == verification, "metadata.json and verification.json disagree")
    require(metadata["layer_name"] == "features.27" and metadata["next_layer_name"] == "features.29", "Unexpected model layer mapping")
    require(metadata["activation_bits"] == metadata["weight_bits"] == 4, "Expected W4A4 layer")
    require(metadata["activation_shape_nchw"] == [1, 8, 4, 4], "Unexpected activation dimensions")
    require(metadata["weight_shape_oihw"] == [8, 8, 3, 3], "Unexpected kernel dimensions")
    require(metadata["output_shape_nchw"] == [1, 8, 4, 4], "Unexpected output dimensions")
    require(not metadata["architecture"]["selected_batchnorm"], "Selected layer has an intervening BatchNorm")
    fingerprint = metadata["model_sha256"]
    require(isinstance(fingerprint, str) and len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef"), "Invalid recorded model fingerprint")

    activation_words = read_words(directory / "activation/activation.txt", 32, 36, headers=True)
    activation = unpack(activation_words, 4, 8, False)  # [padded pixel, input channel]
    weights = []
    combined_weights = []
    for kij in range(9):
        words = read_words(directory / f"weights/weight{kij}.txt", 32, 8, headers=True)
        weights.append(unpack(words, 4, 8, True))  # [output channel, input channel]
        combined_weights.extend(words)
    weights = np.stack(weights)
    require(bool(np.all((weights >= -7) & (weights <= 7))), "Weights exceed symmetric signed 4-bit range -7..7")
    golden_pre_words = read_words(directory / "output/output_updated.txt", 128, 16)
    golden_relu_words = read_words(directory / "output/output_after_relu.txt", 128, 16)
    golden_pre = unpack(golden_pre_words, 16, 8, True)
    golden_relu = unpack(golden_relu_words, 16, 8, True)
    address_words = read_words(directory / "acc_address.txt", 11, 144)
    addresses = np.asarray([int(word, 2) for word in address_words], dtype=np.int64).reshape(16, 9)
    require((directory / "input.txt").read_bytes() == (directory / "activation/activation.txt").read_bytes(), "input.txt is not the promised byte-identical activation alias")
    require((directory / "psum.txt").read_bytes() == (directory / "output/output_after_relu.txt").read_bytes(), "psum.txt is not the promised byte-identical ReLU alias")
    require(read_words(directory / "weight.txt", 32, 72, headers=True) == combined_weights, "weight.txt differs from concatenated kernel files")

    # Recover geometry directly from the packed stream. The border must be the
    # one-pixel zero padding used by the selected 3x3, stride-one convolution.
    padded = activation.reshape(6, 6, 8)
    require(bool(np.all(padded[0] == 0)) and bool(np.all(padded[-1] == 0)) and bool(np.all(padded[:, 0] == 0)) and bool(np.all(padded[:, -1] == 0)), "Padded activation border is nonzero")
    activation_nchw = padded[1:5, 1:5].transpose(2, 0, 1)[None]
    weight_oihw = weights.reshape(3, 3, 8, 8).transpose(2, 3, 0, 1)

    # Explicit Python-integer sums are independent of the exporter's NumPy
    # sliding-window/einsum implementation and do not risk float accumulation.
    expected = np.zeros((16, 8), dtype=np.int64)
    for oy in range(4):
        for ox in range(4):
            for oc in range(8):
                total = 0
                for ky in range(3):
                    for kx in range(3):
                        for ic in range(8):
                            total += int(padded[oy + ky, ox + kx, ic]) * int(weights[ky * 3 + kx, oc, ic])
                expected[oy * 4 + ox, oc] = total
    equal(expected, golden_pre, "Scalar convolution vs pre-ReLU golden")
    equal(np.maximum(expected, 0), golden_relu, "Scalar convolution + ReLU vs golden")

    # Emulate the SRAM organization: each kernel position generates 36 words.
    # Read the supplied address file rather than just recomputing convolution.
    pmem = np.zeros((324, 8), dtype=np.int64)
    for kij in range(9):
        for pixel in range(36):
            for oc in range(8):
                pmem[kij * 36 + pixel, oc] = sum(int(activation[pixel, ic]) * int(weights[kij, oc, ic]) for ic in range(8))
    require(bool(np.all((addresses >= 0) & (addresses < len(pmem)))), "PMEM gather address out of range")
    cumulative = []
    for pixel in range(16):
        oy, ox = divmod(pixel, 4)
        expected_addresses = [(ky * 3 + kx) * 36 + (oy + ky) * 6 + ox + kx for ky in range(3) for kx in range(3)]
        equal(addresses[pixel], np.asarray(expected_addresses), f"PMEM addresses for output pixel {pixel}")
        running = np.zeros(8, dtype=np.int64)
        for address in addresses[pixel]:
            running = running + pmem[int(address)]
            cumulative.append(running.copy())
        equal(running, expected[pixel], f"PMEM/SFU accumulation for output pixel {pixel}")
    cumulative = np.stack(cumulative)
    for name, array in (("PE kernel partial sums", pmem), ("SFU intermediate sums", cumulative), ("Final partial sums", expected)):
        require(bool(np.all((array >= -32768) & (array <= 32767))), f"{name}: signed 16-bit overflow")

    with np.load(directory / "tensors.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name in ("activation_int", "activation_padded_int", "weight_int", "psum_int", "psum_relu_int"):
        require(np.issubdtype(arrays[name].dtype, np.integer), f"{name}: expected integer dtype")
    equal(activation_nchw, arrays["activation_int"], "Packed activation vs NPZ")
    equal(padded.transpose(2, 0, 1)[None], arrays["activation_padded_int"], "Padded activation vs NPZ")
    equal(weight_oihw, arrays["weight_int"], "Packed kernel vs NPZ")
    expected_nchw = expected.reshape(4, 4, 8).transpose(2, 0, 1)[None]
    equal(expected_nchw, arrays["psum_int"], "Scalar convolution vs NPZ")
    equal(np.maximum(expected_nchw, 0), arrays["psum_relu_int"], "Integer ReLU vs NPZ")

    scale_a = float(arrays["activation_scale"])
    scale_w = float(arrays["weight_scale"])
    require(math.isfinite(scale_a) and math.isfinite(scale_w) and scale_a > 0 and scale_w > 0, "Scales must be positive and finite")
    scalar_close(scale_a, metadata["activation_scale"], "Activation scale")
    scalar_close(scale_w, metadata["weight_scale"], "Weight scale")
    close(arrays["quantized_activation"] / scale_a, activation_nchw.astype(np.float64), "Activation quantization lattice", tolerance=1e-5)
    close(arrays["quantized_weight"] / scale_w, weight_oihw.astype(np.float64), "Weight quantization lattice", tolerance=1e-5)
    recovered_pre = expected_nchw.astype(np.float64) * scale_a * scale_w
    recovered_relu = np.maximum(expected_nchw, 0).astype(np.float64) * scale_a * scale_w
    close(recovered_pre, arrays["recovered_pre_relu"], "Reconstructed pre-ReLU values")
    close(recovered_relu, arrays["recovered"], "Reconstructed ReLU values")
    equal(np.maximum(arrays["selected_forward_output"], 0), arrays["reference_next_input"], "Preserved downstream input vs ReLU of preserved convolution output")
    difference = recovered_relu - arrays["reference_next_input"]
    close(difference, arrays["difference"], "Saved elementwise error")
    max_abs_error = float(np.max(np.abs(difference)))
    scalar_close(max_abs_error, metadata["max_abs_error"], "Maximum absolute reconstruction error")
    scalar_close(float(np.mean(np.abs(difference))), metadata["mean_abs_error"], "Mean absolute error")
    scalar_close(float(np.sqrt(np.mean(difference**2))), metadata["rmse"], "RMSE")
    pre_error = float(np.max(np.abs(recovered_pre - arrays["selected_forward_output"])))
    scalar_close(pre_error, metadata["max_abs_error_pre_relu_vs_selected_fp_forward"], "Pre-ReLU reconstruction error")
    require(metadata["reference_is_exact_relu_of_selected_forward_output"] is True, "Saved downstream reference flag is false")
    require(metadata["strict_tolerance"] == 1e-5 and metadata["formal_tolerance"] == 1e-3, "Unexpected success thresholds")
    require(max_abs_error < 1e-5 and max_abs_error < 1e-3, "Reconstruction does not meet project error thresholds")
    require(metadata["strict_pass"] is True and metadata["formal_pass"] is True and metadata["status"] == "PASS", "Recorded verification flags disagree")
    for name, data in (("Activation minimum", (int(activation.min()), metadata["activation_integer_min"])), ("Activation maximum", (int(activation.max()), metadata["activation_integer_max"])), ("Weight minimum", (int(weights.min()), metadata["weight_integer_min"])), ("Weight maximum", (int(weights.max()), metadata["weight_integer_max"]))):
        require(data[0] == data[1], f"{name} differs from metadata")

    files = [p for p in directory.rglob("*") if p.is_file()]
    return {
        "status": "PASS",
        "scope": "Independent supplied-vector arithmetic and preserved-reference consistency; no new RTL simulation, checkpoint evaluation, or accuracy measurement",
        "evidence_type": metadata["evidence_type"],
        "recorded_model_sha256": fingerprint,
        "checkpoint_fingerprint_recomputed": False,
        "scalar_integer_outputs_verified": 128,
        "integer_mismatches": 0,
        "max_abs_reconstruction_error": max_abs_error,
        "strict_error_lt_1e_5": True,
        "formal_error_lt_1e_3": True,
        "activation_integer_range": [int(activation.min()), int(activation.max())],
        "weight_integer_range": [int(weights.min()), int(weights.max())],
        "pre_relu_integer_range": [int(expected.min()), int(expected.max())],
        "post_relu_integer_range": [int(golden_relu.min()), int(golden_relu.max())],
        "pmem_partial_sum_range": [int(pmem.min()), int(pmem.max())],
        "sfu_intermediate_sum_range": [int(cumulative.min()), int(cumulative.max())],
        "pmem_words": 324,
        "gather_addresses_verified": 144,
        "all_intermediates_fit_signed_16_bits": True,
        "activation_words_32_bits": 36,
        "weight_words_32_bits": 72,
        "expected_output_words_128_bits": 16,
        "alias_files_verified": ["input.txt", "weight.txt", "psum.txt"],
        "activation_scale": scale_a,
        "weight_scale": scale_w,
        "sample_dataset_index": metadata["provenance"].get("dataset_index"),
        "sample_label": metadata["provenance"].get("label"),
        "file_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts/vectors", help="Complete exported vector directory")
    parser.add_argument("--report", type=Path, help="Optionally write the JSON result here")
    args = parser.parse_args(argv)
    try:
        result = check_vectors(args.vectors)
    except (CheckFailure, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    if result["status"] == "PASS":
        print(f"PASS: 128 integer outputs, 144 gather addresses; max_abs_error={result['max_abs_reconstruction_error']:.12g} (<1e-5).")
        print(f"Evidence: {result['evidence_type']}; no RTL or checkpoint rerun.")
    else:
        print(f"FAIL: {result['error']}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
