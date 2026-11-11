"""Regression checks for reports using small, deterministic COCO examples.

Run with ``python -m unittest discover -s tests -v``. No dataset or weights
are downloaded, and the synthetic scores are not project benchmark results.
"""

import copy
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


def make_evaluator(perfect=True, iou_types=("bbox", "segm"), image_ids=None,
                   modify_gt=None, modify_predictions=None):
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
    if modify_predictions is not None:
        modify_predictions(predictions)
    if "segm" not in iou_types:
        predictions = {image_id: {key: value for key, value in prediction.items() if key != "masks"}
                       for image_id, prediction in predictions.items()}
    if image_ids is not None:
        predictions = {image_id: predictions[image_id] for image_id in image_ids}
    if modify_gt is not None:
        modify_gt(coco.dataset)
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
        evaluator = CocoEvaluator(coco, list(iou_types))
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
            np.testing.assert_array_equal(rows[["gt_instances", "gt_images"]], 0 if category_id == 17 else 1)
            if category_id == 17:
                self.assertTrue(np.isnan(values).all())
            else:
                np.testing.assert_allclose(values, 1.0, atol=1e-12)

    def test_per_class_support_excludes_crowds_and_unevaluated_images(self):
        def add_annotations(data):
            extra = copy.deepcopy(data["annotations"][0])
            extra.update(id=3, bbox=[1, 1, 8, 8])
            crowd = copy.deepcopy(extra)
            crowd.update(id=4, image_id=303, iscrowd=1)
            data["annotations"].extend([extra, crowd])
        evaluator = make_evaluator(image_ids=(101, 303), modify_gt=add_annotations)
        table = per_class_metrics(evaluator, CLASS_NAMES)
        for iou_type in ("bbox", "segm"):
            rows = table.loc[table["iou_type"] == iou_type].set_index("category_id")
            self.assertEqual(rows.loc[2, "gt_instances"], 2)
            self.assertEqual(rows.loc[2, "gt_images"], 1)
            self.assertEqual(rows.loc[9, "gt_instances"], 0)
            self.assertEqual(rows.loc[9, "gt_images"], 0)
            self.assertTrue(np.isnan(rows.loc[9, "AP"]))

    def test_ranked_false_positives_duplicates_and_misses_change_ap_and_recall(self):
        def one_class(data):
            data["annotations"][1]["category_id"] = 2

        def predictions_for(case):
            def modify(predictions):
                predictions[202]["labels"] = torch.tensor([2])
                predictions[202]["scores"] = torch.tensor([0.8])
                if case in ("high FP", "low FP"):
                    predictions[303] = copy.deepcopy(predictions[101])
                    predictions[303]["scores"] = torch.tensor([0.99 if case == "high FP" else 0.1])
                elif case == "duplicate":
                    predictions[101] = {key: torch.cat([value, value])
                                        for key, value in predictions[101].items()}
                    predictions[101]["scores"] = torch.tensor([0.95, 0.9])
                elif case == "miss":
                    predictions[202] = empty_prediction()
            return modify

        # COCO samples recall at 101 points. A single detected object out of
        # two reaches the first 51; a duplicate is a FP before the second TP.
        expected = {"high FP": (2 / 3, 1), "low FP": (1, 1),
                    "duplicate": ((51 + 50 * 2 / 3) / 101, 1), "miss": (51 / 101, 0.5)}
        for case, (ap, recall) in expected.items():
            with self.subTest(case=case):
                evaluator = make_evaluator(iou_types=("bbox",), modify_gt=one_class,
                                           modify_predictions=predictions_for(case))
                summary = summarize_evaluators({case: evaluator}).iloc[0]
                self.assertAlmostEqual(summary["mAP"], ap)
                self.assertAlmostEqual(summary["AP50"], ap)
                self.assertAlmostEqual(summary["AR100"], recall)
                table = per_class_metrics(evaluator, CLASS_NAMES).set_index("category_id")
                self.assertAlmostEqual(table.loc[2, "AP"], ap)
                self.assertEqual(table.loc[2, "gt_instances"], 2)
                self.assertEqual(table.loc[2, "gt_images"], 2)

    def test_bbox_only_and_mask_models_share_detection_comparison(self):
        faster = make_evaluator(iou_types=("bbox",))
        summary = summarize_evaluators({"Faster R-CNN": faster, "Mask R-CNN": self.perfect})
        self.assertEqual(list(summary.loc[summary["model"] == "Faster R-CNN", "iou_type"]), ["bbox"])
        self.assertEqual(len(summary), 3)
        np.testing.assert_allclose(summary["mAP"], 1.)
        self.assertEqual(set(per_class_metrics(faster, CLASS_NAMES)["iou_type"]), {"bbox"})
        figure = plot_model_comparison(summary)
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(len(figure.axes[0].patches), 4)
        self.assertEqual(len(figure.axes[1].patches), 2)

    def test_comparison_rejects_different_actually_evaluated_images(self):
        subset = make_evaluator(iou_types=("bbox",), image_ids=(101, 202))
        # The underlying COCO datasets are equal; only the evaluated IDs differ.
        self.assertEqual(set(subset.coco_gt.getImgIds()), set(self.perfect.coco_gt.getImgIds()))
        with self.assertRaisesRegex(ValueError, "evaluated image IDs"):
            summarize_evaluators({"all images": self.perfect, "subset": subset})

    def test_comparison_rejects_different_taxonomy_or_ground_truth(self):
        modifiers = {
            "taxonomy": lambda data: data["categories"][0].update(name="different class"),
            "bbox": lambda data: data["annotations"][0].update(bbox=[3, 3, 8, 8]),
            "area": lambda data: data["annotations"][0].update(area=128),
            "crowd": lambda data: data["annotations"][0].update(iscrowd=1),
        }
        for name, modifier in modifiers.items():
            with self.subTest(name=name):
                changed = make_evaluator(iou_types=("bbox",), modify_gt=modifier)
                with self.assertRaisesRegex(ValueError, "ground truth differ"):
                    summarize_evaluators({"original": self.perfect, "changed": changed})

    def test_comparison_ignores_annotation_order_and_ids(self):
        def reorder(data):
            data["annotations"].reverse()
            for index, annotation in enumerate(data["annotations"], start=50):
                annotation["id"] = index
        equivalent = make_evaluator(iou_types=("bbox",), modify_gt=reorder)
        summary = summarize_evaluators({"original": self.perfect, "equivalent": equivalent})
        np.testing.assert_allclose(summary["mAP"], 1.)

    def test_comparison_checks_masks_only_for_shared_segmentation_metrics(self):
        def move_mask(data):
            data["annotations"][0]["segmentation"] = [[1, 1, 9, 1, 9, 9, 1, 9]]
        changed = make_evaluator(modify_gt=move_mask)
        with self.assertRaisesRegex(ValueError, "for segm"):
            summarize_evaluators({"original": self.perfect, "changed": changed})
        bbox_only = make_evaluator(iou_types=("bbox",), modify_gt=move_mask)
        summary = summarize_evaluators({"original": self.perfect, "bbox only": bbox_only})
        np.testing.assert_allclose(summary["mAP"], 1.)

    def test_standard_summary_rejects_mislabeled_coco_configurations(self):
        changes = {
            "area ranges": ("areaRng", [[0, 1e10], [0, 100], [100, 1000], [1000, 1e10]]),
            "area labels": ("areaRngLbl", ["all", "medium", "small", "large"]),
            "recall thresholds": ("recThrs", np.linspace(0, 1, 11)),
            "IoU thresholds": ("iouThrs", np.array([0.5])),
            "detection limits": ("maxDets", [1, 10, 50]),
        }
        for name, (attribute, value) in changes.items():
            for parameter_source in ("current", "accumulated"):
                with self.subTest(name=name, parameters=parameter_source):
                    changed = copy.deepcopy(self.perfect)
                    coco_eval = changed.coco_eval["bbox"]
                    params = coco_eval.params if parameter_source == "current" else coco_eval.eval["params"]
                    setattr(params, attribute, value)
                    with self.assertRaises(ValueError):
                        summarize_evaluators({"changed": changed})

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

    def test_mixed_model_export_keeps_segmentation_exclusive_to_mask_model(self):
        faster = make_evaluator(iou_types=("bbox",))
        with tempfile.TemporaryDirectory() as directory:
            result = export_evaluation_results(
                {"Faster R-CNN": faster, "Mask R-CNN": self.perfect}, directory, CLASS_NAMES,
            )
            summary = pd.read_csv(Path(directory) / "summary.csv")
            per_class = pd.read_csv(Path(directory) / "per_class_metrics.csv")
            self.assertEqual(set(summary.loc[summary["model"] == "Faster R-CNN", "iou_type"]), {"bbox"})
            self.assertEqual(set(per_class.loc[per_class["model"] == "Faster R-CNN", "iou_type"]), {"bbox"})
            self.assertEqual(set(summary.loc[summary["model"] == "Mask R-CNN", "iou_type"]), {"bbox", "segm"})
            names = {Path(path).name for path in result["paths"]}
            self.assertIn("01_Faster_R-CNN_bbox_pr.png", names)
            self.assertNotIn("01_Faster_R-CNN_segm_pr.png", names)
            self.assertIn("02_Mask_R-CNN_segm_pr.png", names)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in result["paths"]))

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
