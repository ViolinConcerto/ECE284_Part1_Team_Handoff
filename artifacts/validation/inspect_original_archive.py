"""Read project checkpoints using a closed allowlist and NumPy tensor views.

This analysis utility never imports torch and never resolves arbitrary pickle
globals. Torch's tensor/storage globals are replaced by inert local descriptors.
Only the standard-library OrderedDict constructor is allowed to execute.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import pickletools
import zipfile

import numpy as np


@dataclass(frozen=True)
class StorageType:
    numpy_dtype: str
    torch_dtype: str


@dataclass
class Storage:
    data: bytes
    storage_type: StorageType
    count: int


@dataclass
class TensorDescription:
    array: np.ndarray
    torch_dtype: str


def describe_tensor(storage, offset, shape, stride, requires_grad, hooks, *metadata):
    if not isinstance(storage, Storage):
        raise pickle.UnpicklingError("Tensor did not reference an allowed storage.")
    shape, stride = tuple(shape), tuple(stride)
    if len(shape) != len(stride) or any(type(v) is not int or v < 0 for v in shape + stride):
        raise pickle.UnpicklingError("Invalid tensor shape or stride.")
    if type(offset) is not int or offset < 0:
        raise pickle.UnpicklingError("Invalid storage offset.")
    maximum = offset + sum((size - 1) * step for size, step in zip(shape, stride))
    if all(shape) and maximum >= storage.count:
        raise pickle.UnpicklingError("Tensor view is outside its storage.")
    dtype = np.dtype(storage.storage_type.numpy_dtype)
    array = np.ndarray(shape, dtype=dtype, buffer=storage.data,
                       offset=offset * dtype.itemsize,
                       strides=tuple(v * dtype.itemsize for v in stride))
    return TensorDescription(array, storage.storage_type.torch_dtype)


ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): describe_tensor,
    ("torch", "FloatStorage"): StorageType("<f4", "torch.float32"),
    ("torch", "LongStorage"): StorageType("<i8", "torch.int64"),
    ("torch", "ByteStorage"): StorageType("u1", "torch.uint8"),
}


class RestrictedDescriptorUnpickler(pickle.Unpickler):
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive, self.prefix = archive, prefix
        self.storages = {}

    def find_class(self, module, name):
        try:
            return ALLOWED_GLOBALS[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}") from None

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5:
            raise pickle.UnpicklingError("Unexpected persistent identifier.")
        kind, storage_type, key, location, count = identifier
        if kind != "storage" or not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError("Persistent object is not an allowed storage.")
        if not isinstance(key, str) or not key.isdigit() or type(count) is not int or count < 0:
            raise pickle.UnpicklingError("Invalid storage key or size.")
        if key not in self.storages:
            data = self.archive.read(self.prefix + "data/" + key)
            if len(data) != count * np.dtype(storage_type.numpy_dtype).itemsize:
                raise pickle.UnpicklingError("Storage size does not match metadata.")
            self.storages[key] = Storage(data, storage_type, count)
        return self.storages[key]


def read_checkpoint(checkpoint_bytes):
    archive = zipfile.ZipFile(io.BytesIO(checkpoint_bytes))
    pickle_name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
    prefix = pickle_name[:-len("data.pkl")]
    if archive.read(prefix + "byteorder") != b"little":
        raise ValueError("This analysis expects little-endian tensor bytes.")
    data = archive.read(pickle_name)
    globals_found = []
    for opcode, argument, _ in pickletools.genops(data):
        if opcode.name == "GLOBAL":
            module, name = argument.split(" ")
            if (module, name) not in ALLOWED_GLOBALS:
                raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")
            globals_found.append(argument)
        elif opcode.name == "STACK_GLOBAL":
            raise pickle.UnpicklingError("Unexpected STACK_GLOBAL in this protocol-2 checkpoint.")
    checkpoint = RestrictedDescriptorUnpickler(io.BytesIO(data), archive, prefix).load()
    return checkpoint, sorted(set(globals_found))


def fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, TensorDescription):
            raise TypeError(f"Non-tensor state entry: {name}")
        descriptor = json.dumps({"name": name, "dtype": tensor.torch_dtype,
                                 "shape": list(tensor.array.shape)}, sort_keys=True).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        raw = np.ascontiguousarray(tensor.array).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def source_state_layout():
    source = Path("github_prepare/ECE284_Part1_GitHub/models.py")
    tree = ast.parse(source.read_text())
    configuration = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FEATURE_CONFIGURATION"
            for target in node.targets
        ):
            configuration = ast.literal_eval(node.value)
    if configuration is None:
        raise ValueError("Feature configuration is absent from supplied model source.")
    expected = {}
    layer_index, convolution_index, input_channels = 0, 0, 3
    for channels in configuration:
        if channels == "M":
            layer_index += 1
            continue
        prefix = f"features.{layer_index}."
        expected[prefix + "weight"] = ((channels, input_channels, 3, 3), "torch.float32")
        expected[prefix + "raw_act_alpha"] = ((), "torch.float32")
        expected[prefix + "raw_weight_alpha"] = ((), "torch.float32")
        layer_index += 1
        if convolution_index != 8:
            prefix = f"features.{layer_index}."
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                expected[prefix + suffix] = ((channels,), "torch.float32")
            expected[prefix + "num_batches_tracked"] = ((), "torch.int64")
            layer_index += 1
        layer_index += 1
        convolution_index += 1
        input_channels = channels
    expected["classifier.weight"] = ((10, 512), "torch.float32")
    expected["classifier.bias"] = ((10,), "torch.float32")
    return expected


def main():
    source = Path("upload/Part1_Run_Artifacts.zip")
    with zipfile.ZipFile(source) as outer:
        metadata = json.loads(outer.read("part1/vectors/metadata.json"))
        results = json.loads(outer.read("part1/results.json"))
        with np.load(io.BytesIO(outer.read("part1/vectors/tensors.npz")), allow_pickle=False) as npz:
            vectors = {k: npz[k] for k in npz.files}
        summary = {}
        expected_layout = source_state_layout()
        for name in ("best", "last"):
            checkpoint_bytes = outer.read(f"part1/{name}.pth")
            checkpoint, globals_found = read_checkpoint(checkpoint_bytes)
            state = checkpoint["model_state"]
            summary[name] = {
                "checkpoint_bytes_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "format": checkpoint["format"],
                "phase": checkpoint["phase"],
                "epoch": checkpoint["epoch"],
                "best_accuracy_pct": checkpoint["best_accuracy_pct"],
                "validation": checkpoint["validation"],
                "quantization_enabled": checkpoint["quantization_enabled"],
                "architecture": checkpoint["architecture"],
                "model_fingerprint": fingerprint(state),
                "model_state_entries": len(state),
                "state_keys_match_model_source": set(state) == set(expected_layout),
                "state_shapes_dtypes_match_model_source": set(state) == set(expected_layout) and all(
                    (tensor.array.shape, tensor.torch_dtype) == expected_layout[key]
                    for key, tensor in state.items()
                ),
                "all_model_tensors_finite": all(np.isfinite(tensor.array).all() for tensor in state.values()),
                "allowed_pickle_globals_used": globals_found,
                "scheduler_state": checkpoint["scheduler_state"],
            }
            if name == "best":
                w = state["features.27.weight"].array
                act_raw = float(state["features.27.raw_act_alpha"].array)
                weight_raw = float(state["features.27.raw_weight_alpha"].array)
                # Softplus in float64, rounded to the model's float32 dtype.
                alpha_x = np.float32(math.log1p(math.exp(act_raw))) + np.float32(1e-8)
                alpha_w = np.float32(math.log1p(math.exp(weight_raw))) + np.float32(1e-8)
                delta_x = np.float32(alpha_x / np.float32(15))
                delta_w = np.float32(alpha_w / np.float32(7))
                # Course standardization uses sample standard deviation and
                # detached statistics. Tiny CPU/GPU reduction rounding may vary.
                normalized = (w - w.mean(dtype=np.float32)) / w.std(ddof=1, dtype=np.float32)
                weight_codes = np.rint(np.clip(normalized / alpha_w, -1, 1) * 7).astype(np.int64)
                activation_codes = np.rint(np.clip(vectors["selected_input"] / alpha_x, 0, 1) * 15).astype(np.int64)
                summary[name]["selected_layer_check"] = {
                    "weight_shape": list(w.shape),
                    "raw_act_alpha": act_raw,
                    "raw_weight_alpha": weight_raw,
                    "activation_scale_derived": float(delta_x),
                    "activation_scale_recorded": metadata["activation_scale"],
                    "weight_scale_derived": float(delta_w),
                    "weight_scale_recorded": metadata["weight_scale"],
                    "weight_integer_mismatches": int(np.count_nonzero(weight_codes != vectors["weight_int"])),
                    "activation_integer_mismatches": int(np.count_nonzero(activation_codes != vectors["activation_int"])),
                }
        summary["metadata_model_sha256"] = metadata["model_sha256"]
        summary["best_fingerprint_matches_metadata"] = summary["best"]["model_fingerprint"] == metadata["model_sha256"]
        summary["metadata_results_accuracy_match"] = metadata["provenance"]["validation"]["accuracy_pct"] == results["test_accuracy_pct"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
