"""Exercise real image/COCO loading, including negative images and transforms."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torchvision.transforms import v2 as T

from help_funciton.marine_dataset import MarineDataset


class MarineDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Grayscale and RGB both become three-channel model input.
        Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(self.root / "object.png")
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(self.root / "empty.png")
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:5, 1:5] = 1
        encoded = mask_util.encode(np.asfortranarray(mask))
        encoded["counts"] = encoded["counts"].decode("ascii")
        self.data = {
            "images": [
                {"id": 10, "file_name": "object.png", "width": 10, "height": 8},
                {"id": 20, "file_name": "empty.png", "width": 10, "height": 8},
            ],
            "categories": [{"id": 3, "name": "bag"}, {"id": 17, "name": "bottle"}],
            "annotations": [{
                "id": 1, "image_id": 10, "category_id": 17,
                "bbox": [1, 2, 5, 4], "area": 12, "iscrowd": 1,
                "segmentation": encoded,
            }],
        }
        self.annotation_file = self.root / "instances.json"
        self.write_annotations()

    def write_annotations(self):
        self.annotation_file.write_text(json.dumps(self.data), encoding="utf-8")

    def make_dataset(self, category_ids=(17, 3), transforms=None):
        return MarineDataset(self.root, self.annotation_file, category_ids, transforms)

    def test_preserves_official_annotation_and_training_label_order(self):
        dataset = self.make_dataset()
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 8, 10))
        self.assertEqual(target["image_id"], 10)
        self.assertEqual(target["boxes"].tolist(), [[1.0, 2.0, 6.0, 6.0]])
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(target["area"].tolist(), [12.0])
        self.assertEqual(target["iscrowd"].tolist(), [1])
        self.assertEqual(tuple(target["masks"].shape), (1, 8, 10))
        self.assertEqual(dataset.coco.anns[1]["category_id"], 17)
        self.assertEqual(dataset.label_to_cat, {1: 17, 2: 3})

    def test_empty_image_retained_with_valid_tensor_shapes(self):
        dataset = self.make_dataset()
        self.assertEqual(len(dataset), 2)
        image, target = dataset[1]
        self.assertEqual(target["image_id"], 20)
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["masks"].shape), (0, 8, 10))
        for key in ["labels", "area", "iscrowd"]:
            self.assertEqual(tuple(target[key].shape), (0,))
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["masks"].dtype, torch.uint8)

    def test_transforms_keep_image_boxes_masks_aligned_including_empty_image(self):
        transform = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.ToDtype(torch.float32, scale=True), T.ToPureTensor(),
        ])
        dataset = self.make_dataset(transforms=transform)
        image, target = dataset[0]
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(target["boxes"].tolist(), [[4.0, 2.0, 9.0, 6.0]])
        self.assertTrue(torch.all(target["masks"][0, 2:5, 5:9] == 1))
        image, empty = dataset[1]
        self.assertEqual(tuple(empty["boxes"].shape), (0, 4))
        self.assertEqual(tuple(empty["masks"].shape), (0, 8, 10))

    def test_rejects_duplicate_or_missing_training_mapping(self):
        for ids in [(), (3, 3), (3,)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.make_dataset(category_ids=ids)

    def test_rejects_invalid_box_instead_of_silently_dropping_ground_truth(self):
        self.data["annotations"][0]["bbox"] = [1, 2, 0, 4]
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            self.make_dataset()[0]


if __name__ == "__main__":
    unittest.main()
