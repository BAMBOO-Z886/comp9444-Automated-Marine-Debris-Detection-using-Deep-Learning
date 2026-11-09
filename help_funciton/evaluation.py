"""Tables and figures for the notebook's held-out COCO evaluation.

Metric values use the [0, 1] scale. COCO's undefined values (-1, for example
when a class has no ground truth) are represented by NaN, never by zero.
The confidence threshold in ``plot_predictions`` affects display only.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch


COCO_METRICS = (
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)
CLASS_COLUMNS = ("iou_type", "category_id", "class_name", "AP", "AP50", "AR100")


def _valid_mean(values):
    values = np.asarray(values, dtype=float)
    valid = values[(values >= 0) & np.isfinite(values)]
    return float(valid.mean()) if valid.size else np.nan


def _evaluation_axes(coco_eval):
    """Find the all-area / 100-detection axes in accumulated COCO arrays."""
    if not coco_eval.eval or "precision" not in coco_eval.eval:
        raise ValueError("Run evaluation and accumulate predictions before reporting metrics.")
    params = coco_eval.eval["params"]
    if not params.useCats:
        raise ValueError("Per-class reports require category-aware evaluation (useCats=1).")
    try:
        area_index = list(params.areaRngLbl).index("all")
        detection_index = list(params.maxDets).index(100)
    except ValueError as exc:
        raise ValueError("Reports require COCO's 'all' area range and maxDets=100.") from exc
    return params, area_index, detection_index


def summarize_evaluators(evaluators):
    """Return the 12 standard bbox/segm COCO metrics for each named model.

    ``evaluators`` maps model names to objects returned by ``engine.evaluate``.
    COCO evaluation must use the standard IoUs (0.50:0.05:0.95), area ranges,
    and maxDets=[1, 10, 100], as configured by the existing CocoEvaluator.
    """
    rows = []
    for model_name, evaluator in evaluators.items():
        for iou_type, coco_eval in evaluator.coco_eval.items():
            if iou_type not in ("bbox", "segm"):
                continue
            stats = np.asarray(coco_eval.stats, dtype=float)
            if stats.shape != (12,):
                raise ValueError(f"{model_name}/{iou_type}: run COCO summarize() before reporting.")
            params = coco_eval.params
            if (list(params.maxDets) != [1, 10, 100]
                    or len(params.iouThrs) != 10
                    or not np.allclose(params.iouThrs, np.linspace(0.5, 0.95, 10))):
                raise ValueError("Summary labels require standard COCO IoUs and maxDets=[1, 10, 100].")
            row = {"model": model_name, "iou_type": iou_type}
            row.update(zip(COCO_METRICS, np.where(stats >= 0, stats, np.nan)))
            rows.append(row)
    return pd.DataFrame(rows, columns=("model", "iou_type", *COCO_METRICS))


def per_class_metrics(evaluator, class_names):
    """Return AP, AP50 and AR100 using category IDs from the evaluator.

    Pass training-label IDs as keys in ``class_names`` for MarineDataset.
    COCO array layout: precision[T,R,K,A,M], recall[T,K,A,M]. See
    https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    """
    rows = []
    for iou_type, coco_eval in evaluator.coco_eval.items():
        if iou_type not in ("bbox", "segm"):
            continue
        params, area_index, detection_index = _evaluation_axes(coco_eval)
        precision = coco_eval.eval["precision"]
        recall = coco_eval.eval["recall"]
        iou50 = np.flatnonzero(np.isclose(params.iouThrs, 0.5))
        if not len(iou50):
            raise ValueError("AP50 requires an IoU=0.50 evaluation threshold.")
        for category_index, category_id in enumerate(params.catIds):
            rows.append({
                "iou_type": iou_type,
                "category_id": int(category_id),
                "class_name": class_names.get(int(category_id), str(category_id)),
                "AP": _valid_mean(precision[:, :, category_index, area_index, detection_index]),
                "AP50": _valid_mean(precision[iou50, :, category_index, area_index, detection_index]),
                "AR100": _valid_mean(recall[:, category_index, area_index, detection_index]),
            })
    return pd.DataFrame(rows, columns=CLASS_COLUMNS)


def plot_precision_recall(evaluator, class_names, iou_type="bbox", iou_threshold=0.5):
    """Plot COCO interpolated per-class PR curves at one measured IoU."""
    if iou_type not in evaluator.coco_eval:
        raise ValueError(f"No {iou_type!r} evaluation is available.")
    coco_eval = evaluator.coco_eval[iou_type]
    params, area_index, detection_index = _evaluation_axes(coco_eval)
    matches = np.flatnonzero(np.isclose(params.iouThrs, iou_threshold))
    if not len(matches):
        raise ValueError(f"IoU={iou_threshold} was not evaluated.")
    precision = coco_eval.eval["precision"][matches[0], :, :, area_index, detection_index]
    fig, ax = plt.subplots(figsize=(9, 5))
    for category_index, category_id in enumerate(params.catIds):
        values = precision[:, category_index]
        valid = values >= 0
        if valid.any():
            ax.plot(np.asarray(params.recThrs)[valid], values[valid],
                    label=class_names.get(int(category_id), str(category_id)))
    ax.set(xlabel="Recall", ylabel="Interpolated precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{iou_type}: precision-recall at IoU={iou_threshold:.2f} (maxDets=100)")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    else:
        ax.text(0.5, 0.5, "No classes with ground truth", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary):
    """Compare mAP and AP50 on the same split; no ranking or model selection."""
    if summary.empty:
        raise ValueError("No model results are available to plot.")
    iou_types = list(summary["iou_type"].unique())
    fig, axes = plt.subplots(1, len(iou_types), figsize=(7 * len(iou_types), 4), squeeze=False)
    for ax, iou_type in zip(axes[0], iou_types):
        frame = summary.loc[summary["iou_type"] == iou_type]
        positions = np.arange(len(frame))
        ax.bar(positions - 0.18, frame["mAP"], width=0.36, label="mAP (IoU 0.50:0.95)")
        ax.bar(positions + 0.18, frame["AP50"], width=0.36, label="AP50")
        ax.set_xticks(positions, frame["model"], rotation=20, ha="right")
        ax.set(ylabel="Average precision", ylim=(0, 1.05), title=iou_type)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _draw_instances(ax, image, instances, class_names, prediction=False):
    image_array = _numpy(image).transpose(1, 2, 0)
    if np.issubdtype(image_array.dtype, np.floating):
        image_array = np.clip(image_array, 0, 1)
    if image_array.shape[-1] == 1:
        image_array = image_array[..., 0]
    ax.imshow(image_array)
    labels = _numpy(instances["labels"])
    boxes = _numpy(instances["boxes"])
    masks = _numpy(instances["masks"]) if "masks" in instances else None
    colors = plt.get_cmap("tab20")
    for index, (box, label) in enumerate(zip(boxes, labels)):
        color = colors(int(label) % 20)
        if masks is not None:
            mask = masks[index]
            if mask.ndim == 3:
                mask = mask[0]
            overlay = np.zeros((*mask.shape, 4))
            overlay[..., :3] = color[:3]
            overlay[..., 3] = (mask > 0.5) * 0.3
            ax.imshow(overlay)
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=1.5, edgecolor=color, facecolor="none"))
        label_text = class_names.get(int(label), str(int(label)))
        if prediction:
            label_text += f" {_numpy(instances['scores'])[index]:.2f}"
        ax.text(x1, y1, label_text, color="white", fontsize=8,
                bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "edgecolor": "none"})
    if not len(labels):
        ax.text(0.02, 0.95, "No detections" if prediction else "No annotations",
                transform=ax.transAxes, color="white", va="top",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2})
    ax.axis("off")


@torch.inference_mode()
def plot_predictions(model, dataset, device, class_names, indices=None,
                     score_threshold=0.5, max_detections=20):
    """Show ground truth and predictions, with a deterministic sample by default.

    ``dataset`` must use evaluation transforms (RGB tensors in [0, 1]). Model
    parameters must already be on ``device``. Filtering here never changes the
    predictions used to compute COCO metrics.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1.")
    if not isinstance(max_detections, int) or max_detections < 1:
        raise ValueError("max_detections must be a positive integer.")
    if not len(dataset):
        raise ValueError("Cannot visualize an empty dataset.")
    if indices is None:
        indices = np.random.default_rng(0).choice(len(dataset), min(3, len(dataset)), replace=False).tolist()
    else:
        indices = list(indices)
    if not indices or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("indices must contain valid dataset indices.")

    fig, axes = plt.subplots(len(indices), 2, figsize=(13, 4 * len(indices)), squeeze=False)
    was_training = model.training
    try:
        model.eval()
        for row, index in enumerate(indices):
            image, target = dataset[index]
            prediction = {key: value.detach().cpu()
                          for key, value in model([image.to(device)])[0].items()}
            keep = torch.where(prediction["scores"] >= score_threshold)[0]
            keep = keep[torch.argsort(prediction["scores"][keep], descending=True)][:max_detections]
            prediction = {key: value[keep] for key, value in prediction.items()}
            _draw_instances(axes[row, 0], image, target, class_names)
            _draw_instances(axes[row, 1], image, prediction, class_names, prediction=True)
            image_id = target.get("image_id", index)
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            axes[row, 0].set_title(f"Image {image_id}: ground truth")
            axes[row, 1].set_title(f"Prediction (score >= {score_threshold:.2f}, top {max_detections})")
        fig.tight_layout()
    except Exception:
        plt.close(fig)
        raise
    finally:
        model.train(was_training)
    return fig


def export_evaluation_results(evaluators, output_dir, class_names):
    """Save summary/per-class CSVs and PR/comparison PNGs, returning their paths.

    Returns a dict with ``summary`` and ``per_class`` DataFrames and ``paths``.
    Undefined CSV values are blank. Existing report files in this directory
    are replaced, so use a separate directory for each experiment or split.
    """
    summary = summarize_evaluators(evaluators)
    if summary.empty:
        raise ValueError("No completed bbox or segm evaluations were provided.")
    per_class = pd.concat(
        [per_class_metrics(evaluator, class_names).assign(model=name)
         for name, evaluator in evaluators.items()], ignore_index=True,
    )
    per_class = per_class[["model", *CLASS_COLUMNS]]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "summary.csv", output_dir / "per_class_metrics.csv"]
    summary.to_csv(paths[0], index=False, na_rep="")
    per_class.to_csv(paths[1], index=False, na_rep="")

    def save_figure(figure, path):
        try:
            figure.savefig(path, dpi=160, bbox_inches="tight")
            paths.append(path)
        finally:
            plt.close(figure)

    save_figure(plot_model_comparison(summary), output_dir / "model_comparison.png")
    for index, (name, evaluator) in enumerate(evaluators.items(), start=1):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "model"
        for iou_type in evaluator.coco_eval:
            if iou_type in ("bbox", "segm"):
                save_figure(plot_precision_recall(evaluator, class_names, iou_type=iou_type),
                            output_dir / f"{index:02d}_{stem}_{iou_type}_pr.png")
    return {"summary": summary, "per_class": per_class, "paths": paths}
