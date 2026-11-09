"""Exercise the real dataset-to-COCO evaluation path without model downloads."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from help_funciton.coco_utils import get_coco_api_from_dataset
from help_funciton.engine import evaluate
from help_funciton.marine_dataset import MarineDataset


def collate(batch):
    return tuple(zip(*batch))


def evaluation_transform(image, target):
    return image.float() / 255, target


def empty_prediction():
    return {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
        "masks": torch.empty((0, 1, 16, 16)),
    }


class SyntheticModel(torch.nn.Module):
    def __init__(self, empty=False, error=None, unknown_label=False, wrong_count=False):
        super().__init__()
        self.empty = empty
        self.error = error
        self.unknown_label = unknown_label
        self.wrong_count = wrong_count

    def forward(self, images):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Evaluation must disable training and gradients")
        if self.error:
            raise RuntimeError(self.error)
        if self.wrong_count:
            return []
        predictions = []
        for image in images:
            image_id = round(float(image[0, 0, 0]) * 255)
            if self.empty or image_id in (30, 40):
                predictions.append(empty_prediction())
                continue
            mask = torch.zeros((1, 1, 16, 16))
            mask[:, :, 3:11, 2:10] = 1
            predictions.append({
                # This deliberately differs from the tight raster mask box.
                "boxes": torch.tensor([[2., 3., 12., 13.]]),
                "labels": torch.tensor([99 if self.unknown_label else {10: 1, 20: 2}[image_id]]),
                "scores": torch.tensor([0.95]),
                "masks": mask,
            })
        return predictions


class EvaluationEngineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        image_ids = (10, 20, 30, 40)
        for image_id in image_ids:
            Image.fromarray(np.full((16, 16, 3), image_id, dtype=np.uint8)).save(self.root / f"{image_id}.png")
        annotations = [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [2, 3, 10, 10],
                "area": 64,
                "segmentation": [[2, 3, 10, 3, 10, 11, 2, 11]],
                "iscrowd": crowd,
            }
            for annotation_id, image_id, category_id, crowd in (
                (1, 10, 42, 0), (2, 20, 7, 0), (3, 40, 42, 1)
            )
        ]
        source = {
            "info": {"description": "synthetic regression fixture"},
            "images": [{"id": i, "height": 16, "width": 16, "file_name": f"{i}.png"} for i in image_ids],
            "categories": [{"id": 7, "name": "bag"}, {"id": 42, "name": "bottle"}, {"id": 101, "name": "absent"}],
            "annotations": annotations,
        }
        annotation_file = self.root / "annotations.json"
        annotation_file.write_text(json.dumps(source), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.dataset = MarineDataset(self.root, annotation_file, [42, 7, 101], transforms=evaluation_transform)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False, collate_fn=collate)

    def evaluate_quietly(self, model, loader=None):
        with redirect_stdout(io.StringIO()):
            return evaluate(model, self.loader if loader is None else loader, "cpu", iou_types=["bbox", "segm"])

    def test_dataset_preserves_annotation_geometry_mapping_crowds_and_empty_images(self):
        image, target = self.dataset[0]
        self.assertEqual(image.shape, (3, 16, 16))
        self.assertTrue(image.is_floating_point())
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(target["boxes"].tolist(), [[2., 3., 12., 13.]])
        self.assertEqual(target["area"].tolist(), [64.])
        self.assertEqual(int(target["masks"].sum()), 64)
        self.assertEqual(self.dataset[1][1]["labels"].tolist(), [2])
        self.assertEqual(self.dataset[3][1]["iscrowd"].tolist(), [1])
        _, negative = self.dataset[2]
        self.assertEqual(negative["boxes"].shape, (0, 4))
        self.assertEqual(negative["masks"].shape, (0, 16, 16))
        self.assertEqual(negative["labels"].numel(), 0)

    def test_coco_view_remaps_labels_and_preserves_nested_subset_without_mutation(self):
        original = copy.deepcopy(self.dataset.coco.dataset)
        nested = Subset(Subset(self.dataset, [3, 2, 0]), [2, 1])
        with redirect_stdout(io.StringIO()):
            coco = get_coco_api_from_dataset(nested)
        self.assertEqual(set(coco.getImgIds()), {10, 30})
        self.assertEqual({c["id"]: c["name"] for c in coco.dataset["categories"]}, {1: "bottle", 2: "bag", 3: "absent"})
        self.assertEqual(len(coco.dataset["annotations"]), 1)
        annotation = coco.dataset["annotations"][0]
        self.assertEqual(annotation["category_id"], 1)
        self.assertEqual(annotation["bbox"], [2, 3, 10, 10])
        self.assertEqual(annotation["area"], 64)
        self.assertEqual(self.dataset.coco.dataset, original)

    def test_perfect_predictions_score_one_and_restore_model_and_thread_state(self):
        model = SyntheticModel()
        threads = torch.get_num_threads()
        result = self.evaluate_quietly(model)
        self.assertTrue(model.training)
        self.assertEqual(torch.get_num_threads(), threads)
        for coco_eval in result.coco_eval.values():
            np.testing.assert_allclose(coco_eval.stats[:3], 1., atol=1e-12)
            self.assertEqual(set(coco_eval.params.imgIds), {10, 20, 30, 40})

    def test_all_empty_predictions_score_zero_and_preserve_eval_mode(self):
        model = SyntheticModel(empty=True).eval()
        threads = torch.get_num_threads()
        result = self.evaluate_quietly(model)
        self.assertFalse(model.training)
        self.assertEqual(torch.get_num_threads(), threads)
        for coco_eval in result.coco_eval.values():
            np.testing.assert_allclose(coco_eval.stats[:3], 0., atol=1e-12)

    def test_subset_evaluation_only_contains_requested_images(self):
        loader = DataLoader(Subset(self.dataset, [0, 2]), batch_size=2, collate_fn=collate)
        result = self.evaluate_quietly(SyntheticModel(), loader)
        self.assertEqual(set(result.coco_gt.getImgIds()), {10, 30})
        for coco_eval in result.coco_eval.values():
            self.assertEqual(set(coco_eval.params.imgIds), {10, 30})
            self.assertAlmostEqual(coco_eval.stats[0], 1.)

    def test_inference_error_restores_model_mode_and_threads(self):
        model = SyntheticModel(error="synthetic inference failure")
        threads = torch.get_num_threads()
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            self.evaluate_quietly(model)
        self.assertTrue(model.training)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_invalid_predictions_raise_instead_of_silently_scoring(self):
        for model, message in ((SyntheticModel(unknown_label=True), "outside the dataset label space"),
                               (SyntheticModel(wrong_count=True), "one prediction per image")):
            with self.subTest(message=message):
                threads = torch.get_num_threads()
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate_quietly(model)
                self.assertTrue(model.training)
                self.assertEqual(torch.get_num_threads(), threads)

    def test_empty_loader_is_rejected(self):
        loader = DataLoader(Subset(self.dataset, []), collate_fn=collate)
        model = SyntheticModel()
        threads = torch.get_num_threads()
        with self.assertRaisesRegex(ValueError, "empty data loader"):
            self.evaluate_quietly(model, loader)
        self.assertTrue(model.training)
        self.assertEqual(torch.get_num_threads(), threads)


if __name__ == "__main__":
    unittest.main()
