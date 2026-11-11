"""Checkpoint provenance failures must not silently change model evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from help_funciton.checkpoints import (
    load_model_checkpoint, save_model_checkpoint, validate_checkpoint_metadata,
)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "weights.pth"
        files = {name: self.root / f"{name}.json" for name in ("train", "test")}
        for name, path in files.items():
            path.write_text(name, encoding="utf-8")
        self.context = dict(architecture="fasterrcnn_resnet50_fpn_v2", category_ids=[42, 7],
                            annotation_files=files)
        self.trained = torch.nn.Linear(2, 3)
        self.metadata = save_model_checkpoint(
            self.trained, self.path, **self.context,
            training_config={"epochs": 15, "seed": 42},
        )

    def test_round_trip_restores_weights_and_provenance(self):
        restored = torch.nn.Linear(2, 3)
        result = load_model_checkpoint(restored, self.path, **self.context)
        self.assertEqual(result["verification"], "verified")
        self.assertEqual(result["metadata"]["training_config"]["epochs"], 15)
        for key, tensor in self.trained.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], tensor)

    def test_rejects_same_size_label_reordering_and_wrong_architecture_before_loading(self):
        for changed in ({"category_ids": [7, 42]}, {"architecture": "maskrcnn_resnet50_fpn_v2"}):
            with self.subTest(changed=changed):
                restored = torch.nn.Linear(2, 3)
                before = restored.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_model_checkpoint(restored, self.path, **(self.context | changed))
                torch.testing.assert_close(restored.weight, before)

    def test_rejects_changed_split(self):
        self.context["annotation_files"]["test"].write_text("different test", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "annotation_sha256"):
            load_model_checkpoint(torch.nn.Linear(2, 3), self.path, **self.context)

    def test_rejects_stale_in_memory_metadata_after_split_changes(self):
        self.assertEqual(validate_checkpoint_metadata(self.metadata, **self.context), self.metadata)
        self.context["annotation_files"]["train"].write_text("new training split", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "annotation_sha256"):
            validate_checkpoint_metadata(self.metadata, **self.context)

    def test_failed_save_preserves_previous_checkpoint_and_removes_partial_file(self):
        previous_bytes = self.path.read_bytes()
        previous_files = set(self.root.iterdir())

        def interrupted_save(payload, destination):
            Path(destination).write_bytes(b"partially written checkpoint")
            raise OSError("simulated storage failure")

        with patch("help_funciton.checkpoints.torch.save", side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, "simulated storage failure"):
                save_model_checkpoint(
                    self.trained, self.path, **self.context,
                    training_config={"epochs": 16, "seed": 42},
                )
        self.assertEqual(self.path.read_bytes(), previous_bytes)
        self.assertEqual(set(self.root.iterdir()), previous_files)
        restored = torch.nn.Linear(2, 3)
        self.assertEqual(load_model_checkpoint(restored, self.path, **self.context)["verification"], "verified")
        torch.testing.assert_close(restored.weight, self.trained.weight)

    def test_legacy_bare_and_wrapped_weights_are_explicitly_unverified(self):
        for checkpoint in (self.trained.state_dict(), {"model_state_dict": self.trained.state_dict()}):
            torch.save(checkpoint, self.path)
            restored = torch.nn.Linear(2, 3)
            with self.assertWarnsRegex(UserWarning, "Legacy checkpoint"):
                result = load_model_checkpoint(restored, self.path, **self.context)
            self.assertEqual(result["verification"], "legacy_unverified")
            torch.testing.assert_close(restored.weight, self.trained.weight)

    def test_incomplete_metadata_is_not_treated_as_legacy(self):
        torch.save({"format_version": 1, "model_state_dict": self.trained.state_dict()}, self.path)
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_model_checkpoint(torch.nn.Linear(2, 3), self.path, **self.context)


if __name__ == "__main__":
    unittest.main()
