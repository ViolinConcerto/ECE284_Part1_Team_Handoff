"""Validate the frozen Part1 handoff and import header-free vanilla vectors.

The source checkout is only read. Integer expectations are recomputed with
run.integer_reference, rather than copied without checking their semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from run import ROOT, dimensions, integer_reference, pack, unpack, write_words


ACTIVATION_HEADERS = [
    "#time0:channel7_to_channel0:4bits_each#",
    "#row_major_padded_6x6:36_words#",
    "#three_header_tokens_then_binary_words#",
]
MERGED_WEIGHT_HEADERS = [
    "#nine_row_major_kernel_positions#",
    "#eight_output_channel_words_per_position#",
    "#input_channel7_to_0:signed4_twos_complement#",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_exact(path: Path, width: int, count: int,
               headers: list[str] | None = None) -> list[int]:
    tokens = path.read_text(encoding="ascii").split()
    prefix = headers or []
    require(tokens[:len(prefix)] == prefix, f"{path}: unexpected header tokens")
    words = tokens[len(prefix):]
    require(len(words) == count, f"{path}: expected {count} words, got {len(words)}")
    for index, word in enumerate(words):
        require(re.fullmatch(f"[01]{{{width}}}", word) is not None,
                f"{path}: data word {index} must contain exactly {width} binary digits")
    return [int(word, 2) for word in words]


def compare_words(actual: list[int], expected: list[int], label: str) -> int:
    require(len(actual) == len(expected), f"{label}: word count differs")
    errors = [(word, lane) for word, (a, e) in enumerate(zip(actual, expected))
              for lane, (av, ev) in enumerate(zip(unpack(a, 16, True), unpack(e, 16, True)))
              if av != ev]
    require(not errors, f"{label}: {len(errors)} scalar mismatches; first={errors[:1]}")
    return len(expected) * 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(source: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(source), "log", "-1", "--format=%H", "--",
                                 "artifacts/vectors", "artifacts/run/results.json"],
                                text=True, capture_output=True, timeout=10, check=True)
        revision = result.stdout.strip()
        return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None
    except (OSError, subprocess.SubprocessError):
        return None


def import_handoff(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    in_generated_cases = source == ROOT.parent and destination.is_relative_to(ROOT / "cases")
    require(not destination.is_relative_to(source) or in_generated_cases,
            "Output must be under Part3/cases or outside the Part1 source checkout")
    vectors = source / "artifacts/vectors"
    metadata = json.loads((vectors / "metadata.json").read_text(encoding="utf-8"))
    verification = json.loads((vectors / "verification.json").read_text(encoding="utf-8"))
    require(metadata == verification, "metadata.json and verification.json disagree")
    require(metadata.get("layer_name") == "features.27" and
            metadata.get("next_layer_name") == "features.29", "Unexpected selected/next layer")
    require(metadata.get("activation_shape_nchw") == [1, 8, 4, 4] and
            metadata.get("weight_shape_oihw") == [8, 8, 3, 3] and
            metadata.get("output_shape_nchw") == [1, 8, 4, 4], "Unexpected tensor shapes")
    architecture = metadata["architecture"]
    require(architecture.get("selected_batchnorm") is False, "Selected layer must have no BN")
    require(metadata.get("activation_bits") == 4 and metadata.get("weight_bits") == 4,
            "Expected W4A4 vectors")
    require(metadata.get("reference_is_exact_relu_of_selected_forward_output") is True,
            "Reference is not the selected output followed by ReLU")
    require(metadata.get("trained_model_evidence") is True and
            metadata.get("evidence_type") == "trained_model", "Expected trained-model handoff")
    model_hash = metadata.get("model_sha256", "")
    require(re.fullmatch(r"[0-9a-f]{64}", model_hash) is not None, "Invalid model fingerprint")
    for key in ("activation_scale", "weight_scale"):
        require(isinstance(metadata.get(key), (int, float)) and
                math.isfinite(metadata[key]) and metadata[key] > 0, f"Invalid {key}")
    require(metadata.get("status") == "PASS" and metadata.get("strict_pass") is True and
            metadata.get("formal_pass") is True and
            0 <= metadata["max_abs_error"] < metadata["strict_tolerance"] <= 1e-5 and
            metadata["max_abs_error"] < metadata["formal_tolerance"] <= 1e-3,
            "Recorded model-domain verification did not pass the required thresholds")

    cfg = dict(name=destination.name, height=4, width=4, kernel=3, padding=1, stride=1,
               input_channels=8, output_channels=8, activation_bits=4, weight_bits=4,
               psum_bits=16, groups=1, dilation=1, bias=False,
               origin="artifacts/vectors; frozen trained-model sample",
               model_sha256=model_hash, provenance_file="provenance.json")
    dimensions(cfg)
    inputs = read_exact(vectors / "activation/activation.txt", 32, 36, ACTIVATION_HEADERS)
    input_alias = read_exact(vectors / "input.txt", 32, 36, ACTIVATION_HEADERS)
    require(input_alias == inputs and
            (vectors / "input.txt").read_bytes() == (vectors / "activation/activation.txt").read_bytes(),
            "input.txt is not the byte-identical activation alias")
    weights = []
    for kij in range(9):
        headers = ["#output_channel0_to_7:one_word_each#",
                   "#input_channel7_to_0:signed4_twos_complement#",
                   f"#kernel_y{kij // 3}_x{kij % 3}#"]
        weights.extend(read_exact(vectors / f"weights/weight{kij}.txt", 32, 8, headers))
    merged_weights = read_exact(vectors / "weight.txt", 32, 72, MERGED_WEIGHT_HEADERS)
    require(weights == merged_weights, "Merged weight.txt disagrees with per-kernel files")
    require(all(-7 <= value <= 7 for word in weights for value in unpack(word, 4, True)),
            "Weights exceed the handoff's symmetric signed 4-bit range")
    pre_relu = read_exact(vectors / "output/output_updated.txt", 128, 16)
    post_relu = read_exact(vectors / "output/output_after_relu.txt", 128, 16)
    psum_alias = read_exact(vectors / "psum.txt", 128, 16)
    require(psum_alias == post_relu and
            (vectors / "psum.txt").read_bytes() == (vectors / "output/output_after_relu.txt").read_bytes(),
            "psum.txt is not the byte-identical post-ReLU alias")
    raw, expected_pre, expected_post = integer_reference(cfg, inputs, weights)
    pre_count = compare_words(pre_relu, expected_pre, "Pre-ReLU golden")
    post_count = compare_words(post_relu, expected_post, "Post-ReLU golden")
    addresses = read_exact(vectors / "acc_address.txt", 11, 144)
    expected_addresses = [kij * 36 + (y + kij // 3) * 6 + x + kij % 3
                          for y in range(4) for x in range(4) for kij in range(9)]
    require(addresses == expected_addresses, "SFU gather addresses disagree with the padded layout")
    gathered = [pack([sum(unpack(raw[address], 16, True)[lane]
                         for address in addresses[pixel * 9:(pixel + 1) * 9])
                      for lane in range(8)], 16) for pixel in range(16)]
    compare_words(gathered, expected_pre, "SFU gather")
    for prefix, words, bits, signed in (("activation", inputs, 4, False),
                                         ("weight", weights, 4, True)):
        values = [value for word in words for value in unpack(word, bits, signed)]
        require(metadata[f"{prefix}_integer_min"] == min(values) and
                metadata[f"{prefix}_integer_max"] == max(values),
                f"{prefix} range disagrees with metadata")

    source_files = sorted(path for path in vectors.rglob("*") if path.is_file())
    for relative in ("HANDOFF.md", "README.md", "verify_export.py", "models.py",
                     "artifacts/run/results.json", "artifacts/validation/vector_check.json"):
        path = source / relative
        require(path.is_file(), f"Missing handoff evidence: {path}")
        source_files.append(path)
    hashes = {path.relative_to(source).as_posix(): sha256(path) for path in source_files}
    frozen_check = json.loads((source / "artifacts/validation/vector_check.json").read_text())
    newline_normalizations = {}
    for name, expected_hash in frozen_check["file_sha256"].items():
        actual_hash = hashes.get("artifacts/vectors/" + name)
        if actual_hash != expected_hash and Path(name).suffix in (".txt", ".json"):
            # Git on Windows may convert tracked LF text to CRLF at checkout.
            # Record both hashes; never rewrite the preserved source checkout.
            local_bytes = (vectors / name).read_bytes()
            canonical_hash = hashlib.sha256(local_bytes.replace(b"\r\n", b"\n")).hexdigest()
            if canonical_hash == expected_hash:
                newline_normalizations[name] = dict(local_sha256=actual_hash,
                                                     frozen_lf_sha256=canonical_hash,
                                                     transform="CRLF to LF for hash comparison only")
                actual_hash = canonical_hash
        require(actual_hash == expected_hash,
                f"Source vector differs from the frozen validation hash: {name}")
    results = json.loads((source / "artifacts/run/results.json").read_text())
    require(results["test_accuracy_pct"] == metadata["provenance"]["validation"]["accuracy_pct"],
            "Run results and metadata disagree on recorded accuracy")
    provenance = dict(
        schema_version=1, imported_at_utc=datetime.now(timezone.utc).isoformat(),
        source_repository="https://github.com/ViolinConcerto/ECE284_Part1_Team_Handoff",
        source_commit=git_revision(source), source_directory=str(source),
        source_vectors_directory=str(vectors), source_sha256=hashes,
        source_frozen_hash_newline_normalizations=newline_normalizations,
        model_sha256=model_hash, sample=metadata["provenance"],
        activation_scale=metadata["activation_scale"], weight_scale=metadata["weight_scale"],
        recorded_max_abs_model_error=metadata["max_abs_error"],
        recorded_test_accuracy_pct=results["test_accuracy_pct"],
        contract=dict(input_shape_nchw=[1, 8, 4, 4], padded_shape_chw=[8, 6, 6],
                      weight_shape_oihw=[8, 8, 3, 3], output_shape_nchw=[1, 8, 4, 4],
                      activation="unsigned 4-bit", weight="signed 4-bit two's complement",
                      psum="signed 16-bit two's complement", channel_zero="least significant lane",
                      traversal="row-major", golden="integer accumulation, then ReLU; no scaling"),
        validation=dict(status="PASS", reference="Part3/run.py integer_reference",
                        activation_words=36, weight_words=72, raw_psum_words=324,
                        gather_addresses=144, pre_relu_scalar_comparisons=pre_count,
                        post_relu_scalar_comparisons=post_count, integer_mismatches=0,
                        alias_consistency=True, per_kernel_weight_consistency=True,
                        exact_header_tokens=True, frozen_source_content_hashes_match=True,
                        source_raw_hashes_match_frozen=not newline_normalizations),
        scope="Fresh integer-vector and alias checks only; model accuracy/float error are recorded source evidence. No training, checkpoint execution, or RTL simulation is performed by this importer.")
    # No output is created until every source check has passed.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "case.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for filename, words, width in (("input.txt", inputs, 32), ("weight.txt", weights, 32),
                                    ("psum.txt", post_relu, 128),
                                    ("psum_pre_relu.txt", pre_relu, 128)):
        write_words(destination / filename, words, width)
    provenance["output_sha256"] = {name: sha256(destination / name) for name in
                                   ("case.json", "input.txt", "weight.txt", "psum.txt", "psum_pre_relu.txt")}
    require(all(sha256(path) == hashes[path.relative_to(source).as_posix()] for path in source_files),
            "Source changed during import")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                encoding="utf-8")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=ROOT / "cases/team_handoff")
    args = parser.parse_args()
    try:
        result = import_handoff(args.source, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL handoff import: {error}\n")
    check = result["validation"]
    print(f"PASS handoff import: {check['pre_relu_scalar_comparisons']} pre-ReLU + "
          f"{check['post_relu_scalar_comparisons']} post-ReLU scalar checks; "
          f"{check['gather_addresses']} gather addresses; 0 mismatches; {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
