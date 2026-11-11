"""Execute the notebook's restart-to-report workflow on synthetic local data.

Only detector factories are replaced: dataset loading, checkpoint verification,
COCO evaluation, report exports and qualitative figures use the real notebook
code. No training, downloaded weights or benchmark claims are involved.
"""

import ast
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch

from help_funciton.checkpoints import save_model_checkpoint


NOTEBOOK = Path(__file__).resolve().parents[1] / "Marine_Debris_Blank_Notebook.ipynb"
SETUP_CELLS = (
    "2ec63a33",  # Reuse/audit the committed splits without changing validation.
    "4ef8ef68", "0161fca1",  # Define Faster and Mask R-CNN factories.
    "28f8d7be", "4906d879", "bc27481d",  # Dataset, transforms and setup.
    "8a398320",  # Held-out test dataset and loader.
)
REPORT_CELLS = ("marine-cell-035", "marine-cell-038", "marine-cell-040", "marine-cell-042")


class TinyDetector(torch.nn.Module):
    """Perfect synthetic detections with the same public output contract."""

    def __init__(self, masks, confidence=0.2):
        super().__init__()
        self.confidence = torch.nn.Parameter(torch.tensor(confidence))
        self.masks = masks
        self.roi_heads = SimpleNamespace(score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        self.transform = SimpleNamespace(min_size=(16,), max_size=16)

    def forward(self, images):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Notebook inference must disable training and gradients")
        predictions = []
        for image in images:
            image_id = round(float(image[0, 0, 0]) * 255)
            positive = image_id in (10, 20)
            prediction = {
                "boxes": torch.tensor([[2., 3., 10., 11.]], device=image.device)
                if positive else torch.empty((0, 4), device=image.device),
                "labels": torch.tensor([1 if image_id == 10 else 2], device=image.device)
                if positive else torch.empty((0,), dtype=torch.int64, device=image.device),
                "scores": self.confidence.expand(1) if positive else self.confidence.expand(0),
            }
            if self.masks:
                mask = torch.zeros((int(positive), 1, 16, 16), device=image.device)
                mask[:, :, 3:11, 2:10] = 1
                prediction["masks"] = mask
            predictions.append(prediction)
        return predictions


class NotebookWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        cls.sources = {cell.id: cell.source for cell in cls.notebook.cells if cell.cell_type == "code"}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.working_directory = chdir(self.root)
        self.working_directory.__enter__()
        self.addCleanup(self.working_directory.__exit__, None, None, None)
        self.addCleanup(plt.close, "all")
        self.output = io.StringIO()
        self.namespace = {"__name__": "__main__"}
        self.factory_calls = []
        self._write_fixture()
        # A hidden pretrained-weight download must fail rather than use network.
        blocker = patch("torch.hub.download_url_to_file", side_effect=AssertionError("No downloads allowed"))
        self.download = blocker.start()
        self.addCleanup(blocker.stop)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Deliberate validation overlap in fixture.
            for cell_id in SETUP_CELLS:
                self.run_cell(cell_id)
        self.namespace["device"] = torch.device("cpu")
        self.namespace["create_faster_rcnn_model"] = self._factory("fasterrcnn_resnet50_fpn_v2", False)
        self.namespace["create_model"] = self._factory("maskrcnn_resnet50_fpn_v2", True)

    def run_cell(self, cell_id):
        with redirect_stdout(self.output), patch("IPython.display.display"):
            exec(compile(self.sources[cell_id], f"notebook:{cell_id}", "exec"), self.namespace)

    def _factory(self, architecture, masks):
        def factory(num_classes, pretrained=True):
            self.factory_calls.append((architecture, num_classes, pretrained))
            if pretrained:
                raise AssertionError("Restart evaluation must construct models without pretrained weights")
            return TinyDetector(masks)
        return factory

    def _write_fixture(self):
        base = self.root / "instance_version"
        (base / "train").mkdir(parents=True)
        (base / "val").mkdir()
        categories = [{"id": 42, "name": "bottle"}, {"id": 7, "name": "bag"}]

        def split(rows, directory):
            images, annotations = [], []
            for image_id, file_name, category_id in rows:
                Image.fromarray(np.full((16, 16, 3), image_id, dtype=np.uint8)).save(base / directory / file_name)
                images.append({"id": image_id, "file_name": file_name, "width": 16, "height": 16})
                if category_id is not None:
                    annotations.append({
                        "id": image_id, "image_id": image_id, "category_id": category_id,
                        "bbox": [2, 3, 8, 8], "area": 64, "iscrowd": 0,
                        "segmentation": [[2, 3, 10, 3, 10, 11, 2, 11]],
                    })
            return {"info": {}, "images": images, "annotations": annotations, "categories": categories}

        fixtures = {
            "instances_train.json": split([(100, "vid_train_frame001.png", 42)], "train"),
            "instances_test.json": split([
                (10, "vid_test1_frame010.png", 42),
                (20, "vid_test2_frame020.png", 7),
                (30, "vid_test3_frame030.png", None),
            ], "train"),
            "instances_val_trashcan.json": split([
                (21, "vid_test1_frame021.png", 42),
                (200, "vid_train_frame002.png", 42),
                (201, "vid_clean_frame001.png", 7),
            ], "val"),
        }
        for filename, data in fixtures.items():
            (base / filename).write_text(json.dumps(data), encoding="utf-8")

    def _save_checkpoint(self, faster=True):
        architecture = "fasterrcnn_resnet50_fpn_v2" if faster else "maskrcnn_resnet50_fpn_v2"
        path = Path("checkpoints") / ("faster_rcnn_sgd.pth" if faster else "mask_rcnn_sgd.pth")
        metadata = save_model_checkpoint(
            TinyDetector(masks=not faster, confidence=0.95), path,
            architecture=architecture, category_ids=self.namespace["category_id"],
            annotation_files=self.namespace["checkpoint_annotation_files"],
            training_config={"epochs": 15, "seed": 42},
        )
        return metadata

    def test_notebook_schema_and_all_code_cells_compile(self):
        nbformat.validate(self.notebook)
        for cell_id, source in self.sources.items():
            ast.parse(source, filename=f"notebook:{cell_id}")

    def test_split_statistics_are_quiet_and_preserve_audit_records(self):
        files = list((self.root / "instance_version").glob("*.json"))
        before = {path: path.read_bytes() for path in files}
        self.output.seek(0)
        self.output.truncate(0)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            self.run_cell("2ec63a33")
        self.assertEqual(emitted, [])
        printed = json.loads(self.output.getvalue().split("\n", 1)[1])
        audit = self.namespace["split_audit"]
        self.assertEqual(printed, {key: value for key, value in audit.items() if key != "warnings"})
        self.assertEqual(len(audit["warnings"]), 2)
        self.assertEqual(before, {path: path.read_bytes() for path in files})

    def test_restart_loads_correct_architectures_and_exports_complete_mixed_report(self):
        self._save_checkpoint(faster=True)
        self._save_checkpoint(faster=False)
        self.assertFalse(self.namespace["trained_model_names"])
        for cell_id in REPORT_CELLS:
            self.run_cell(cell_id)

        self.assertEqual(self.factory_calls, [
            ("fasterrcnn_resnet50_fpn_v2", 3, False),
            ("maskrcnn_resnet50_fpn_v2", 3, False),
        ])
        self.download.assert_not_called()
        self.assertEqual(len(self.namespace["dataset_val"]), 3)
        self.assertEqual({int(self.namespace["dataset_val"][i][1]["image_id"]) for i in range(3)}, {21, 200, 201})
        self.assertEqual(self.namespace["split_audit"]["overlap"]["validation_test"]["validation_images"], 1)
        self.assertIn("Skipping Mask R-CNN / AdamW", self.output.getvalue())
        self.assertIn("Skipping Mask R-CNN / SGD + flip", self.output.getvalue())

        output_dir = self.root / "evaluation_results" / "held_out_test"
        summary = pd.read_csv(output_dir / "summary.csv")
        self.assertEqual(set(zip(summary["model"], summary["iou_type"])), {
            ("Faster R-CNN / SGD", "bbox"),
            ("Mask R-CNN / SGD", "bbox"),
            ("Mask R-CNN / SGD", "segm"),
        })
        np.testing.assert_allclose(summary[["mAP", "AP50", "AP75", "AR100"]], 1.)
        per_class = pd.read_csv(output_dir / "per_class_metrics.csv")
        self.assertEqual(set(per_class["class_name"]), {"bottle", "bag"})
        self.assertTrue((per_class["gt_instances"] == 1).all())
        metadata = json.loads((output_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(metadata["models"]), {"Faster R-CNN / SGD", "Mask R-CNN / SGD"})
        for name, source in metadata["checkpoint_sources"].items():
            self.assertEqual(source["verification"], "verified")
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(metadata["evaluated_image_ids"][name], [10, 20, 30])
            self.assertAlmostEqual(float(self.namespace["models_to_evaluate"][name].confidence.detach()), 0.95)
        self.assertEqual(metadata["inference_settings"]["Faster R-CNN / SGD"]["iou_types"], ["bbox"])
        self.assertTrue((output_dir / "model_comparison.png").is_file())
        self.assertTrue((output_dir / "01_qualitative.png").is_file())
        self.assertTrue((output_dir / "02_qualitative.png").is_file())
        self.assertEqual(len(list(output_dir.glob("*_pr.png"))), 3)
        self.assertFalse(plt.get_fignums())

    def test_untrained_variables_without_checkpoints_are_not_evaluated(self):
        self.namespace["faster_model"] = TinyDetector(masks=False)
        self.namespace["model"] = TinyDetector(masks=True)
        with self.assertRaisesRegex(RuntimeError, "Train at least one model"):
            self.run_cell("marine-cell-035")
        self.assertFalse(self.namespace["models_to_evaluate"])
        self.assertFalse(self.factory_calls)
        self.download.assert_not_called()

    def test_stale_in_memory_split_metadata_is_rejected_before_evaluation(self):
        metadata = self._save_checkpoint(faster=True)
        self.namespace["faster_model"] = TinyDetector(masks=False, confidence=0.95)
        self.namespace["trained_model_names"].add("Faster R-CNN / SGD")
        self.namespace["trained_metadata"]["Faster R-CNN / SGD"] = metadata
        test_json = self.root / "instance_version" / "instances_test.json"
        test_json.write_text(test_json.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with patch("help_funciton.engine.evaluate") as evaluate:
            # The setup cell imported evaluate into the notebook namespace.
            self.namespace["evaluate"] = evaluate
            with self.assertRaisesRegex(ValueError, "annotation_sha256"):
                self.run_cell("marine-cell-035")
            evaluate.assert_not_called()
        self.assertFalse(self.factory_calls)


if __name__ == "__main__":
    unittest.main()
