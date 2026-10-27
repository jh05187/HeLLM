import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate
import generate
import train
from tests.test_tokenizer import TEXT


class TrainingRunTest(unittest.TestCase):
    """Train for a few steps on a local file, then load the checkpoint with the other scripts."""

    def test_train_generate_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_file = os.path.join(tmp, "stories.txt")
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(TEXT)
            out_dir = os.path.join(tmp, "out")

            for tokenizer in ("char", "bpe"):
                with self.subTest(tokenizer=tokenizer), contextlib.redirect_stdout(io.StringIO()) as out:
                    train.main([
                        "--data-file", data_file, "--tokenizer", tokenizer, "--vocab-size", "300",
                        "--block-size", "32", "--n-embd", "32", "--n-layer", "2", "--batch-size", "8",
                        "--max-steps", "60", "--warmup-steps", "10", "--eval-interval", "30",
                        "--eval-iters", "5", "--device", "cpu", "--out-dir", out_dir,
                    ])
                    checkpoint = os.path.join(out_dir, "best.pt")
                    self.assertTrue(os.path.exists(checkpoint))
                    self.assertTrue(os.path.exists(os.path.join(out_dir, "metrics.csv")))

                    with contextlib.redirect_stderr(io.StringIO()):
                        generate.main(["Lily", "--checkpoint", checkpoint, "--max-new-tokens", "20",
                                       "--device", "cpu", "--seed", "0"])
                        evaluate.main(["--checkpoint", checkpoint, "--device", "cpu"])
                    self.assertIn("Bits per byte", out.getvalue())

            first_loss, last_loss = self._losses(os.path.join(out_dir, "metrics.csv"))
            self.assertLess(last_loss, first_loss)

    @staticmethod
    def _losses(path):
        with open(path, encoding="utf-8") as f:
            rows = f.read().strip().splitlines()[1:]
        return float(rows[0].split(",")[1]), float(rows[-1].split(",")[1])

    def test_lr_schedule(self):
        args = train.parse_args(["--lr", "1e-3", "--min-lr", "1e-4", "--warmup-steps", "10", "--max-steps", "110"])
        self.assertAlmostEqual(train.lr_at(0, args), 1e-4)
        self.assertAlmostEqual(train.lr_at(9, args), 1e-3)
        self.assertAlmostEqual(train.lr_at(60, args), 5.5e-4)
        self.assertAlmostEqual(train.lr_at(110, args), 1e-4)


if __name__ == "__main__":
    unittest.main()
