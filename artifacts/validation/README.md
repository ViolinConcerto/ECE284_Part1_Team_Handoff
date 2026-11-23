# Handoff validation scope

`checkpoint_inspection.json` was generated from the supplied original `Part1_Run_Artifacts.zip` using the archived method in `inspect_original_archive.py`. It uses a closed allowlist to map the five required pickle globals into NumPy storage/tensor descriptors; arbitrary globals are rejected and no PyTorch code is imported. The script's original archive input is `upload/Part1_Run_Artifacts.zip` relative to the analysis working directory; it is included to document the exact inspection method, not as the normal team entry point.

The full 101-tensor model fingerprint reconstructed from `best.pth` exactly matches the fingerprint in the exported metadata. The checkpoint's epoch, recorded accuracy and validation count match the notebook/history/results. The checkpoint-derived selected-layer weight and activation integer codes match the exported arrays. CPU NumPy and CUDA Torch transcendental/reduction rounding can differ in the last float32 bits of scale calculations; exact scale equality is not claimed.

`vector_check.json` and the positive/negative reports document independent vector decoding and scalar integer convolution. The normal portable entry point is `python tools/check_handoff_vectors.py` from the repository root.

`source_and_artifact_checks.json` and `file_manifest.json` record packaging checks and SHA256 digests. The file manifest hashes every deliverable file except itself; it deliberately excludes the ZIP container.

The original full CIFAR-10 accuracy evaluation and new RTL simulation were not rerun here. The original dataset is not included, and this environment has no PyTorch installation. These checks establish artifact consistency and integer arithmetic, not a new end-to-end software or hardware run.
