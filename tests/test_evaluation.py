"""Regression checks for reports using small, deterministic COCO examples.

Run with ``python -m unittest discover -s tests -v``. No dataset or weights
are downloaded, and the synthetic scores are not project benchmark results.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from help_funciton.coco_eval import CocoEvaluator
from help_funciton.evaluation import (
    export_evaluation_results,
    per_class_metrics,
    plot_model_comparison,
    plot_precision_recall,
    plot_predictions,
    summarize_evaluators,
)


CLASS_NAMES = {2: "bottle", 9: "bag", 17: "absent"}


def empty_prediction():
    return {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
        "masks": torch.empty((0, 1, 16, 16)),
    }


def make_evaluator(perfect=True):
    """Two objects, one negative image, and one category without any GT."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:11, 2:10] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    coco = COCO()
    coco.dataset = {
        "info": {},
        "images": [{"id": i, "height": 16, "width": 16} for i in (101, 202, 303)],
        "categories": [{"id": i, "name": n} for i, n in CLASS_NAMES.items()],
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [2, 3, 8, 8],
                "area": 64,
                "iscrowd": 0,
                "segmentation": rle,
            }
            for annotation_id, image_id, category_id in ((1, 101, 2), (2, 202, 9))
        ],
    }
    predictions = {i: empty_prediction() for i in (101, 202, 303)}
    if perfect:
        for image_id, label in ((101, 2), (202, 9)):
            predictions[image_id] = {
                "boxes": torch.tensor([[2.0, 3.0, 10.0, 11.0]]),
                "labels": torch.tensor([label]),
                "scores": torch.tensor([0.95]),
                "masks": torch.tensor(mask.copy(), dtype=torch.float32)[None, None],
            }
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
        evaluator = CocoEvaluator(coco, ["bbox", "segm"])
        evaluator.update(predictions)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


class SmallDataset:
    def __init__(self, count=1, empty=False):
        self.count = count
        self.empty = empty
        self.accessed = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.accessed.append(index)
        image = torch.zeros((3, 16, 16), dtype=torch.float32)
        if self.empty:
            target = {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "masks": torch.empty((0, 16, 16), dtype=torch.uint8),
            }
        else:
            masks = torch.zeros((1, 16, 16), dtype=torch.uint8)
            masks[:, 3:11, 2:10] = 1
            target = {
                "boxes": torch.tensor([[2.0, 3.0, 10.0, 11.0]]),
                "labels": torch.tensor([2]),
                "masks": masks,
            }
        target["image_id"] = index
        return image, target


class EmptyModel(torch.nn.Module):
    def forward(self, images):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Visualisation must use eval mode without gradients")
        return [empty_prediction() for _ in images]


class FailingModel(torch.nn.Module):
    def forward(self, images):
        raise RuntimeError("synthetic inference failure")


class ScoredModel(torch.nn.Module):
    def forward(self, images):
        masks = torch.zeros((3, 1, 16, 16))
        masks[0, :, 1:7, 1:7] = 1
        masks[1, :, 3:11, 2:10] = 1
        masks[2, :, 6:14, 6:14] = 1
        return [{
            "boxes": torch.tensor([[1., 1., 7., 7.], [2., 3., 10., 11.], [6., 6., 14., 14.]]),
            "labels": torch.tensor([2, 2, 9]),
            "scores": torch.tensor([0.2, 0.8, 0.9]),
            "masks": masks,
        } for _ in images]


class EvaluationReportsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perfect = make_evaluator()
        cls.empty = make_evaluator(perfect=False)

    def tearDown(self):
        plt.close("all")

    def test_standard_summary_perfect_and_empty_predictions(self):
        summary = summarize_evaluators({"perfect": self.perfect, "empty": self.empty})
        self.assertEqual(set(summary["iou_type"]), {"bbox", "segm"})
        self.assertEqual(len(summary), 4)
        for model_name, expected in (("perfect", 1.0), ("empty", 0.0)):
            rows = summary.loc[summary["model"] == model_name]
            for metric in ("mAP", "AP50", "AP75", "AP_small", "AR1", "AR10", "AR100"):
                np.testing.assert_allclose(rows[metric], expected, atol=1e-12)
            self.assertTrue(rows["AP_medium"].isna().all())
            self.assertTrue(rows["AP_large"].isna().all())
            self.assertTrue(rows["AR_medium"].isna().all())
            self.assertTrue(rows["AR_large"].isna().all())

    def test_per_class_uses_category_ids_and_marks_absent_class_undefined(self):
        table = per_class_metrics(self.perfect, CLASS_NAMES)
        self.assertEqual(len(table), 6)
        for category_id, name in CLASS_NAMES.items():
            rows = table.loc[table["category_id"] == category_id]
            self.assertEqual(set(rows["class_name"]), {name})
            self.assertEqual(set(rows["iou_type"]), {"bbox", "segm"})
            values = rows[["AP", "AP50", "AR100"]].to_numpy()
            if category_id == 17:
                self.assertTrue(np.isnan(values).all())
            else:
                np.testing.assert_allclose(values, 1.0, atol=1e-12)

    def test_curves_and_comparison_render_with_undefined_values(self):
        for iou_type in ("bbox", "segm"):
            figure = plot_precision_recall(self.perfect, CLASS_NAMES, iou_type=iou_type)
            figure.canvas.draw()
            self.assertTrue(figure.axes)
        figure = plot_model_comparison(summarize_evaluators({"perfect": self.perfect, "empty": self.empty}))
        figure.canvas.draw()
        self.assertTrue(figure.axes)

    def test_export_writes_readable_tables_and_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = export_evaluation_results({"model / example": self.perfect}, directory, CLASS_NAMES)
            self.assertIsInstance(result["summary"], pd.DataFrame)
            self.assertIsInstance(result["per_class"], pd.DataFrame)
            paths = [Path(path) for path in result["paths"]]
            self.assertGreaterEqual(sum(path.suffix == ".csv" for path in paths), 2)
            self.assertGreaterEqual(sum(path.suffix == ".png" for path in paths), 3)
            for path in paths:
                self.assertTrue(path.resolve().is_relative_to(Path(directory).resolve()))
                self.assertGreater(path.stat().st_size, 0)
                if path.suffix == ".csv":
                    self.assertFalse(pd.read_csv(path).empty)

    def test_prediction_plot_handles_empty_predictions_and_ground_truth(self):
        model = EmptyModel()
        for training in (True, False):
            for empty_gt in (True, False):
                model.train(training)
                figure = plot_predictions(model, SmallDataset(empty=empty_gt), "cpu", CLASS_NAMES, indices=[0])
                figure.canvas.draw()
                self.assertEqual(model.training, training)
                self.assertEqual(len(figure.axes), 2)

    def test_prediction_sample_is_reproducible(self):
        dataset = SmallDataset(count=10)
        model = EmptyModel()
        plot_predictions(model, dataset, "cpu", CLASS_NAMES)
        first = dataset.accessed.copy()
        dataset.accessed.clear()
        plot_predictions(model, dataset, "cpu", CLASS_NAMES)
        self.assertTrue(first)
        self.assertEqual(first, dataset.accessed)

    def test_prediction_plot_filters_scores_and_draws_highest_scoring_mask(self):
        model = ScoredModel()
        figure = plot_predictions(
            model, SmallDataset(), "cpu", CLASS_NAMES, indices=[0],
            score_threshold=0.5, max_detections=1,
        )
        figure.canvas.draw()
        prediction_axis = figure.axes[1]
        self.assertEqual(len(prediction_axis.patches), 1)
        self.assertEqual(prediction_axis.patches[0].get_x(), 6)
        self.assertEqual([text.get_text() for text in prediction_axis.texts], ["bag 0.90"])
        self.assertEqual(len(prediction_axis.images), 2)  # RGB image and retained mask.
        self.assertGreater(np.asarray(prediction_axis.images[1].get_array())[7, 7, 3], 0)
        self.assertTrue(model.training)

    def test_prediction_plot_restores_mode_after_inference_failure(self):
        model = FailingModel()
        model.train()
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            plot_predictions(model, SmallDataset(), "cpu", CLASS_NAMES, indices=[0])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
