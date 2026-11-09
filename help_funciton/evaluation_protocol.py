"""Check the data split used for an evaluation without changing its contents."""

import warnings


def audit_splits(train, test, validation=None):
    """Accept COCO dictionaries; reject train/test leakage and report val overlap.

    COCO image IDs may be reused by separately supplied validation annotations,
    so cross-file validation overlap is checked by filename and source video.
    """
    splits = {"train": train, "test": test}
    if validation is not None:
        splits["validation"] = validation
    expected = {c["id"]: c["name"] for c in train["categories"]}
    summary = {"splits": {}, "overlap": {}, "warnings": []}
    files, videos = {}, {}
    for name, data in splits.items():
        if {c["id"]: c["name"] for c in data["categories"]} != expected:
            raise ValueError(f"{name} categories do not match the training taxonomy.")
        ids = [im["id"] for im in data["images"]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name} must contain nonempty, unique image IDs.")
        files[name] = {im["file_name"] for im in data["images"]}
        videos[name] = {file.split("_frame")[0] for file in files[name]}
        annotated = {a["image_id"] for a in data["annotations"]}
        summary["splits"][name] = {
            "images": len(ids), "annotations": len(data["annotations"]),
            "videos": len(videos[name]), "empty_images": len(set(ids) - annotated),
        }
    if {im["id"] for im in train["images"]} & {im["id"] for im in test["images"]}:
        raise ValueError("Train/test image IDs overlap.")
    if files["train"] & files["test"] or videos["train"] & videos["test"]:
        raise ValueError("Train/test filenames or source videos overlap.")
    if validation is not None:
        for other in ("train", "test"):
            shared_files = files["validation"] & files[other]
            shared_videos = videos["validation"] & videos[other]
            summary["overlap"][f"validation_{other}"] = {
                "filenames": len(shared_files), "videos": len(shared_videos),
                "validation_images": sum(
                    im["file_name"].split("_frame")[0] in shared_videos
                    for im in validation["images"]
                ),
            }
            if shared_files or shared_videos:
                message = (
                    f"Validation and {other} share {len(shared_files)} filenames and "
                    f"{len(shared_videos)} source videos. These splits are not fully "
                    "video-independent; report this limitation and do not tune on test results."
                )
                summary["warnings"].append(message)
                warnings.warn(message, UserWarning, stacklevel=2)
    return summary
