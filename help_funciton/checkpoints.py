"""Evaluation checkpoints with verifiable training labels and split provenance."""

import hashlib
from pathlib import Path
import tempfile
import warnings

import torch


def file_sha256(path):
    """Hash large weights/annotation files without loading them all into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(architecture, category_ids, annotation_files):
    category_ids = list(category_ids)
    if not category_ids or len(set(category_ids)) != len(category_ids):
        raise ValueError("category_ids must contain unique training category IDs.")
    if set(annotation_files) != {"train", "test"}:
        raise ValueError("annotation_files must identify the train and test JSON files.")
    return {
        "architecture": architecture,
        "category_ids_in_training_order": category_ids,
        "annotation_sha256": {name: file_sha256(path) for name, path in annotation_files.items()},
    }


def validate_checkpoint_metadata(metadata, *, architecture, category_ids, annotation_files):
    """Validate provenance for file-loaded or still-in-memory trained models.

    Rechecking the current annotation files prevents stale training metadata
    from being accepted after a split changes during a notebook session.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Unsupported or incomplete checkpoint metadata.")
    expected = _provenance(architecture, category_ids, annotation_files)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match this evaluation; use matching weights/splits.")
    return metadata


def save_model_checkpoint(model, path, *, architecture, category_ids,
                          annotation_files, training_config):
    """Save final trained weights, label order, split hashes and training details.

    This is an evaluation checkpoint, not an optimizer/resume checkpoint.
    ``training_config`` should contain JSON-compatible experiment settings.
    """
    metadata = _provenance(architecture, category_ids, annotation_files)
    metadata["training_config"] = dict(training_config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "metadata": metadata,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    # Save beside the destination so replacement is atomic on the same file
    # system. A failed write leaves an earlier good checkpoint intact.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return metadata


def load_model_checkpoint(model, path, *, architecture, category_ids, annotation_files):
    """Load trained weights after checking architecture, label order and split.

    Historical bare state_dict files (including faster_rcnn_sgd.pth) remain
    supported, but their training provenance cannot be verified. They emit a
    warning and are explicitly labelled legacy_unverified in the report.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or a checkpoint dictionary.")
    metadata = checkpoint.get("metadata")
    if "format_version" in checkpoint or metadata is not None:
        if checkpoint.get("format_version") != 1 or not isinstance(metadata, dict):
            raise ValueError("Unsupported or incomplete checkpoint metadata.")
        validate_checkpoint_metadata(
            metadata, architecture=architecture, category_ids=category_ids,
            annotation_files=annotation_files,
        )
        verification = "verified"
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing model_state_dict.")
    else:
        _provenance(architecture, category_ids, annotation_files)
        warnings.warn(
            "Legacy checkpoint has no verifiable architecture, label order or split metadata. "
            "Confirm its training provenance before interpreting test results.",
            UserWarning, stacklevel=2,
        )
        verification = "legacy_unverified"
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return {"verification": verification, "metadata": metadata}
