"""Focused tests for data integrity and the persisted prediction contract."""
import tempfile
import unittest
from pathlib import Path

from common import ROOT, read_labels, read_rows


class DataTests(unittest.TestCase):
    def setUp(self):
        self.labels = read_labels(ROOT / "data/label_mapping.txt")

    def test_original_data(self):
        train = read_rows(ROOT / "data/train.txt", self.labels)
        dev = read_rows(ROOT / "data/dev.txt", self.labels)
        self.assertEqual((len(train), len(dev), len(self.labels)), (402, 63, 8))
        self.assertFalse({r["text"] for r in train} & {r["text"] for r in dev})
        self.assertEqual({r["label"] for r in train}, set(self.labels))

    def test_bad_data_fails_loudly(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as folder:
            path = Path(folder) / "rows.txt"
            for content in ["9\t未知类别", "0 没有分隔符", "0\t", ""]:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_rows(path, self.labels)
            path.write_text("1\t水果\t额外的制表符", encoding="utf-8")
            self.assertEqual(read_rows(path, self.labels)[0]["text"], "水果\t额外的制表符")

    def test_saved_model_prediction(self):
        from inference import Predictor
        predictor = Predictor(device="cpu")
        texts = ["酒店房间很干净，前台服务很好。", "苹果又甜又脆。"]
        batch = predictor.predict(texts)
        singles = [predictor.predict([text])[0] for text in texts]
        self.assertEqual([r["label"] for r in batch], [r["label"] for r in singles])
        for row in batch:
            self.assertEqual(len(row["scores"]), 8)
            self.assertAlmostEqual(sum(row["scores"].values()), 1.0, places=5)
            self.assertGreaterEqual(row["confidence"], 0)
            self.assertLessEqual(row["confidence"], 1)
        with self.assertRaises(ValueError):
            predictor.predict(["  "])


if __name__ == "__main__":
    unittest.main()
