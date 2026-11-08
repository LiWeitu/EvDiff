import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from evaluate import frechet, sequence_readers


class EvaluationTest(unittest.TestCase):
    def test_pair_by_id_and_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt" / "seq"
            pred = root / "pred" / "seq"
            gt.mkdir(parents=True)
            pred.mkdir(parents=True)
            frames = np.arange(5 * 4 * 6, dtype=np.uint8).reshape(5, 4, 6)
            np.save(gt / "images.npy", frames)
            for index in range(1, 4):
                Image.fromarray(frames[index, 1:, 2:]).save(pred / f"frame_{index - 1:04d}.png")
            args = SimpleNamespace(
                gt_root=gt.parent, pred_root=pred.parent, index_offset=-1, pred_origin=(1, 2),
            )
            view = {"frames": [1, 4], "roi": [2, 3, 4, 6]}
            reference, prediction = sequence_readers(args, "seq", view)
            for index in range(1, 4):
                np.testing.assert_array_equal(reference(index), prediction(index))
            del reference, prediction  # Release the memory-mapped file before cleanup on Windows.
            (pred / "frame_0001.png").unlink()
            with self.assertRaisesRegex(ValueError, r"first GT IDs \[2\]"):
                sequence_readers(args, "seq", view)

    def test_frechet_low_rank(self):
        features = np.random.default_rng(7).normal(size=(13, 400))
        shift = np.linspace(-1, 1, 400)
        self.assertAlmostEqual(frechet(features, features), 0.0, places=9)
        self.assertAlmostEqual(frechet(features, features + shift), shift @ shift, places=9)


if __name__ == "__main__":
    unittest.main()
