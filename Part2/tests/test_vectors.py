"""Independent numeric and preflight checks; no training, GPU or RTL simulator.

The small CHW fixture uses asymmetric corner/center weights.  Its output oracle
is a closed-form three-term expression, not a second copy of the runner's
convolution loops.  All generated files live in a TemporaryDirectory.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("video_runner_under_test", ROOT / "run_video.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the video runner")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def binary_words(path: Path, width: int) -> list[str]:
    """Decode the produced artifacts without using the runner's file helpers."""
    words = path.read_text(encoding="ascii").splitlines()
    if any(len(word) != width or set(word) - {"0", "1"} for word in words):
        raise AssertionError(f"invalid binary words in {path}")
    return words


def lane(word: str, index: int, bits: int, signed: bool) -> int:
    end = len(word) - index * bits
    value = int(word[end - bits:end], 2)
    return value - 2 ** bits if signed and value >= 2 ** (bits - 1) else value


def fixture(folder: Path) -> tuple[Path, list[list[int]]]:
    frames_dir = folder / "frames"
    frames_dir.mkdir()
    base = [17 * channel + 4 * y + x for channel in range(8) for y in range(4) for x in range(4)]
    frames = [base, [x + 7 for x in base], [x - 8 if x >= 8 else x for x in base], base[:]]
    for i, frame in enumerate(frames):
        (frames_dir / f"X{i}.txt").write_text("".join(f"{x}\n" for x in frame), encoding="ascii")
    weights = []
    for kernel in range(9):
        for output in range(8):
            codes = [0] * 8
            if kernel == 0:
                codes[(output + 1) % 8] = 1
            elif kernel == 4:
                codes[output] = -1 if output % 2 else 1
            elif kernel == 8:
                codes[(output + 1) % 8] = -1
            weights.append("".join(f"{code & 15:04b}" for code in reversed(codes)))
    (folder / "weight.txt").write_text("\n".join(weights) + "\n", encoding="ascii")
    (folder / "metadata.json").write_text(json.dumps({"weight_shape_oihw": [8, 8, 3, 3],
                                                       "model_sha256": "explicit-synthetic-fixture"}), encoding="ascii")
    cfg = dict(name="asymmetric_fixture", frames_dir="frames", weights="weight.txt",
               weight_metadata="metadata.json", channels=8, output_channels=8, height=4, width=4,
               kernel=3, padding=1, stride=1, layout="CHW", input_bits=8, weight_bits=4,
               groups=1, bias=False, chunk=7)
    case = folder / "case.json"
    case.write_text(json.dumps(cfg), encoding="ascii")
    return case, frames


def sample(frame: list[int], channel: int, y: int, x: int) -> int:
    return frame[channel * 16 + y * 4 + x] if 0 <= y < 4 and 0 <= x < 4 else 0


def closed_form(frame: list[int], output: int, y: int, x: int) -> int:
    other = (output + 1) % 8
    sign = -1 if output % 2 else 1
    return (sign * sample(frame, output, y, x)
            + sample(frame, other, y - 1, x - 1)
            - sample(frame, other, y + 1, x + 1))


class ArithmeticTests(unittest.TestCase):
    def test_every_unsigned_byte_times_signed_weight_reconstructs(self):
        # Exercise the public high-first reconstruction, including low nibbles 8..15.
        for value in range(256):
            high, low = divmod(value, 16)
            for weight in range(-8, 8):
                high_result = [[high * weight] * 8]
                low_result = [[low * weight] * 8]
                actual = RUNNER.reconstruct([high_result, low_result] * 4, "baseline")
                self.assertEqual(actual, [[[value * weight] * 8]] * 4)
        self.assertEqual(16 * 15 + 15, 255)
        self.assertNotEqual(16 * 15 - 1, 255)  # Treating the low nibble as signed is wrong.

    def test_signed_four_bit_roundtrip_and_range_rejection(self):
        values = list(range(-8, 8))
        for offset in (0, 8):
            vector = values[offset:offset + 8]
            encoded = RUNNER.pack(vector, 4, True)
            self.assertEqual(RUNNER.unpack(encoded, 4), vector)
        for bad in (-9, 8):
            with self.assertRaisesRegex(ValueError, "does not fit"):
                RUNNER.pack([bad], 4, True)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            RUNNER.pack([-1], 4, False)

    def test_wide_accumulation_preserves_large_result_and_cancellation(self):
        # 72 unsigned-byte products can exceed signed16, even without exotic weights.
        for value in (72 * 255 * 7, 72 * 255 * -8):
            word = RUNNER.pack([value] * 8, 32, True)
            self.assertEqual(RUNNER.unpack(word, 32), [value] * 8)
            with self.assertRaisesRegex(ValueError, "does not fit"):
                RUNNER.pack([value], 16, True)
        # Widened high/low products cancel exactly to -255; no unsigned wrap/ReLU.
        actual = RUNNER.reconstruct([[[15 * -1] * 8], [[15 * -1] * 8]] * 4, "baseline")
        self.assertEqual(actual[0][0][0], -255)


class PreflightTests(unittest.TestCase):
    def test_legacy_negative_frame_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ece284_video_preflight_") as temporary:
            folder = Path(temporary)
            case, frames = fixture(folder)
            # -62 is the first value of the legacy signed teacher batch.
            frames[0][0] = -62
            (folder / "frames/X0.txt").write_text("\n".join(map(str, frames[0])) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "unsigned 8-bit"):
                RUNNER.load_case(case)

    def test_delta_outside_signed_four_bit_is_rejected_without_clipping(self):
        for difference in (-9, 8):
            with self.subTest(difference=difference), tempfile.TemporaryDirectory(prefix="ece284_video_delta_") as temporary:
                folder = Path(temporary)
                case, frames = fixture(folder)
                for i in range(4):
                    frames[i] = [128] * 128
                frames[1][0] += difference
                for i, frame in enumerate(frames):
                    (folder / f"frames/X{i}.txt").write_text("\n".join(map(str, frame)) + "\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "Xi-X0 exceeds signed 4-bit"):
                    RUNNER.load_case(case)

    def test_extra_or_malformed_decimal_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ece284_video_parse_") as temporary:
            path = Path(temporary) / "frame.txt"
            path.write_text("1\n2\n3\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "expected 2 scalars"):
                RUNNER.read_decimals(path, 2)
            path.write_text("1\n0x02\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "one decimal integer"):
                RUNNER.read_decimals(path, 2)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ece284_video_oracles_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.folder = Path(cls.temporary.name)
        cls.case, cls.frames = fixture(cls.folder)
        cls.data = RUNNER.load_case(cls.case)
        cls.build = cls.folder / "prepared"
        cls.prepared = RUNNER.prepare(cls.data, cls.build)

    def test_input_words_are_high_first_chw_and_channel_zero_lsb(self):
        words = binary_words(self.build / "conv_input.mem", 32)
        self.assertEqual(len(words), 8 * 16)
        for frame_index, frame in enumerate(self.frames):
            for position in range(16):
                for channel in range(8):
                    value = frame[channel * 16 + position]
                    self.assertEqual(lane(words[(2 * frame_index) * 16 + position], channel, 4, False), value // 16)
                    self.assertEqual(lane(words[(2 * frame_index + 1) * 16 + position], channel, 4, False), value % 16)
        diff = binary_words(self.build / "diff_input.mem", 32)
        self.assertEqual(len(diff), 5 * 16)
        self.assertEqual(diff[:32], words[:32])
        for frame_index in range(1, 4):
            for position in range(16):
                for channel in range(8):
                    index = channel * 16 + position
                    self.assertEqual(lane(diff[(frame_index + 1) * 16 + position], channel, 4, True),
                                     self.frames[frame_index][index] - self.frames[0][index])

    def test_full_raw_outputs_match_asymmetric_closed_form(self):
        words = binary_words(self.build / "expected_frames.mem", 256)
        self.assertEqual(len(words), 4 * 16)
        negative_outputs = 0
        for frame_index, frame in enumerate(self.frames):
            for position in range(16):
                y, x = divmod(position, 4)
                for output in range(8):
                    expected = closed_form(frame, output, y, x)
                    negative_outputs += expected < 0
                    self.assertEqual(lane(words[frame_index * 16 + position], output, 32, True), expected)
        self.assertGreater(negative_outputs, 0, "fixture must detect an accidental ReLU")

    def test_kernel_pass_pixel_order_sums_to_direct_raw_convolution(self):
        raw = binary_words(self.build / "expected_raw_conv.mem", 256)
        self.assertEqual(len(raw), 9 * 8 * 16)
        for frame_index, frame in enumerate(self.frames):
            for position in range(16):
                y, x = divmod(position, 4)
                for output in range(8):
                    high = sum(lane(raw[(kernel * 8 + 2 * frame_index) * 16 + position], output, 32, True)
                               for kernel in range(9))
                    low = sum(lane(raw[(kernel * 8 + 2 * frame_index + 1) * 16 + position], output, 32, True)
                              for kernel in range(9))
                    self.assertEqual(16 * high + low, closed_form(frame, output, y, x))
        diff = binary_words(self.build / "expected_raw_diff.mem", 256)
        passes = binary_words(self.build / "expected_pass_diff.mem", 256)
        self.assertEqual(len(diff), 9 * 5 * 16)
        self.assertEqual(len(passes), 5 * 16)
        for p in range(5):
            for position in range(16):
                for output in range(8):
                    total = sum(lane(diff[(kernel * 5 + p) * 16 + position], output, 32, True) for kernel in range(9))
                    self.assertEqual(total, lane(passes[p * 16 + position], output, 32, True))

    def test_all_seven_memory_files_are_hashed(self):
        manifest = json.loads((self.build / "preparation.json").read_text(encoding="utf-8"))
        names = {"conv_input.mem", "diff_input.mem", "weight.mem", "expected_raw_conv.mem",
                 "expected_raw_diff.mem", "expected_pass_diff.mem", "expected_frames.mem"}
        self.assertEqual(set(manifest["files"]), names)
        for name in names:
            raw = (self.build / name).read_bytes()
            self.assertEqual(manifest["files"][name]["bytes"], len(raw))
            self.assertEqual(manifest["files"][name]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_single_wrong_output_cannot_cancel_out(self):
        expected = self.prepared["direct"]
        changed = copy.deepcopy(expected)
        changed[0][0][0] += 1
        changed[0][0][1] -= 1
        result = RUNNER.errors(changed, expected)
        self.assertEqual(result["mismatches"], 2)
        self.assertEqual(result["max_abs_error"], 1)
        self.assertEqual(result["mse"], 2 / (4 * 16 * 8))
        changed = copy.deepcopy(expected)
        changed[0][0][0] += 1
        self.assertEqual(RUNNER.errors(changed, expected)["mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
