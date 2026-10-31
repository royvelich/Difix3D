import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from src.dataset import PairedDataset


class FakeTokenizer:
    model_max_length = 5

    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(input_ids=torch.arange(5).unsqueeze(0))


class PairedDatasetTest(unittest.TestCase):
    def _write_image(self, path, color):
        Image.new("RGB", (6, 4), color=color).save(path)

    def test_our_manifest_builds_two_view_training_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_image(root / "corrupt.png", (255, 0, 0))
            self._write_image(root / "clean.png", (0, 255, 0))
            self._write_image(root / "ref.png", (0, 0, 255))
            manifest = {
                "metadata": {"ignored": True},
                "train": {
                    "pair-0": {
                        "image": "corrupt.png",
                        "target_image": "clean.png",
                        "ref_image": "ref.png",
                        "ref_images": ["ref.png", "unused.png"],
                    }
                },
            }
            manifest_path = root / "data.json"
            manifest_path.write_text(json.dumps(manifest))

            dataset = PairedDataset(
                manifest_path,
                "train",
                height=4,
                width=8,
                tokenizer=FakeTokenizer(),
            )
            item = dataset[0]

            self.assertEqual(
                tuple(item["conditioning_pixel_values"].shape), (2, 3, 4, 8)
            )
            self.assertEqual(
                tuple(item["output_pixel_values"].shape), (2, 3, 4, 8)
            )
            self.assertEqual(tuple(item["input_ids"].shape), (5,))
            self.assertEqual(item["caption"], "remove degradation")
            self.assertEqual(item["record_id"], "pair-0")
            torch.testing.assert_close(
                item["conditioning_pixel_values"][1],
                item["output_pixel_values"][1],
            )

    def test_missing_rank_zero_reference_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_image(root / "corrupt.png", (255, 0, 0))
            self._write_image(root / "clean.png", (0, 255, 0))
            manifest_path = root / "data.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "train": [
                            {
                                "id": "pair-0",
                                "image": "corrupt.png",
                                "target_image": "clean.png",
                            }
                        ]
                    }
                )
            )

            dataset = PairedDataset(manifest_path, "train")
            with self.assertRaisesRegex(KeyError, "ref_image"):
                dataset[0]

    def test_indexed_jsonl_supports_random_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in (
                ("corrupt.png", (255, 0, 0)),
                ("clean.png", (0, 255, 0)),
                ("ref.png", (0, 0, 255)),
            ):
                self._write_image(root / name, color)

            jsonl_path = root / "train.jsonl"
            records = [
                {
                    "record_id": f"pair-{index}",
                    "image": "corrupt.png",
                    "target_image": "clean.png",
                    "ref_image": "ref.png",
                    "prompt": f"prompt-{index}",
                }
                for index in range(2)
            ]
            offsets = []
            with jsonl_path.open("wb") as output:
                for record in records:
                    offsets.append(output.tell())
                    output.write((json.dumps(record) + "\n").encode())
            with Path(f"{jsonl_path}.idx").open("wb") as index:
                for offset in offsets:
                    index.write(struct.pack("<Q", offset))

            dataset = PairedDataset(jsonl_path, "train", height=4, width=8)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["record_id"], "pair-1")
            self.assertEqual(dataset[1]["caption"], "prompt-1")

    def test_resizes_pil_image_before_tensor_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "pattern.png"
            image = Image.new("RGB", (3, 2))
            image.putdata(
                [
                    (0, 10, 250),
                    (255, 30, 0),
                    (20, 240, 40),
                    (245, 0, 120),
                    (15, 220, 255),
                    (230, 80, 10),
                ]
            )
            image.save(image_path)

            manifest_path = root / "data.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "train": [
                            {
                                "id": "pair-0",
                                "image": "pattern.png",
                                "target_image": "pattern.png",
                                "ref_image": "pattern.png",
                            }
                        ]
                    }
                )
            )
            dataset = PairedDataset(
                manifest_path,
                "train",
                height=5,
                width=7,
            )

            actual = dataset[0]["conditioning_pixel_values"][0]
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                pil_resized = F.resize(
                    source,
                    (5, 7),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                expected = F.normalize(
                    F.to_tensor(pil_resized),
                    mean=[0.5],
                    std=[0.5],
                )

                tensor_first = F.normalize(
                    F.resize(
                        F.to_tensor(source),
                        (5, 7),
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    mean=[0.5],
                    std=[0.5],
                )

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertGreater(
                torch.max(torch.abs(actual - tensor_first)).item(),
                1e-4,
            )


if __name__ == "__main__":
    unittest.main()
