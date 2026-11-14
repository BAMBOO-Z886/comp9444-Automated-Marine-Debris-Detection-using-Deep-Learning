"""Tables and figures from the existing COCO evaluators."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch



METRICS = (
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)


def _valid_mean(values):
    values = np.asarray(values)
    values = values[values >= 0]
    return values.mean() if len(values) else np.nan


def summarize_evaluators(evaluators):
    rows = []
    for name, evaluator in evaluators.items():
        for iou_type, result in evaluator.coco_eval.items():
            row = {"model": name, "iou_type": iou_type}
            row.update(zip(METRICS, np.where(result.stats >= 0, result.stats, np.nan)))
            rows.append(row)
    return pd.DataFrame(rows)


def per_class_metrics(evaluator, class_names):
    rows = []
    for iou_type, result in evaluator.coco_eval.items():
        params = result.eval["params"]
        precision = result.eval["precision"]
        recall = result.eval["recall"]
        area = list(params.areaRngLbl).index("all")
        detections = list(params.maxDets).index(100)
        iou50 = np.flatnonzero(np.isclose(params.iouThrs, 0.5))[0]
        ground_truth = result.cocoGt.dataset["annotations"]
        for index, category_id in enumerate(params.catIds):
            anns = [a for a in ground_truth if a["category_id"] == category_id
                    and a["image_id"] in params.imgIds and not a.get("iscrowd", 0)]
            rows.append({
                "iou_type": iou_type,
                "category_id": category_id,
                "class_name": class_names[category_id],
                "gt_instances": len(anns),
                "gt_images": len({a["image_id"] for a in anns}),
                "AP": _valid_mean(precision[:, :, index, area, detections]),
                "AP50": _valid_mean(precision[iou50, :, index, area, detections]),
                "AR100": _valid_mean(recall[:, index, area, detections]),
            })
    return pd.DataFrame(rows)


def plot_precision_recall(evaluator, class_names, iou_type="bbox"):
    result = evaluator.coco_eval[iou_type]
    params = result.eval["params"]
    area = list(params.areaRngLbl).index("all")
    detections = list(params.maxDets).index(100)
    iou50 = np.flatnonzero(np.isclose(params.iouThrs, 0.5))[0]
    precision = result.eval["precision"][iou50, :, :, area, detections]
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, category_id in enumerate(params.catIds):
        values = precision[:, index]
        valid = values >= 0
        if valid.any():
            ax.plot(params.recThrs[valid], values[valid], label=class_names[category_id])
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{iou_type} precision-recall (IoU=0.50)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary):
    types = summary["iou_type"].unique()
    fig, axes = plt.subplots(1, len(types), figsize=(7 * len(types), 4), squeeze=False)
    for ax, iou_type in zip(axes[0], types):
        rows = summary[summary["iou_type"] == iou_type]
        x = np.arange(len(rows))
        ax.bar(x - 0.18, rows["mAP"], width=0.36, label="mAP")
        ax.bar(x + 0.18, rows["AP50"], width=0.36, label="AP50")
        ax.set_xticks(x, rows["model"], rotation=20, ha="right")
        ax.set(ylabel="Average precision", ylim=(0, 1.05), title=iou_type)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _draw_instances(ax, image, objects, class_names, predictions=False):
    ax.imshow(image.detach().cpu().permute(1, 2, 0))
    labels = objects["labels"].detach().cpu().numpy()
    boxes = objects["boxes"].detach().cpu().numpy()
    masks = objects.get("masks")
    if masks is not None:
        masks = masks.detach().cpu().numpy()
    for index, (label, box) in enumerate(zip(labels, boxes)):
        color = plt.get_cmap("tab20")(int(label) % 20)
        if masks is not None:
            mask = masks[index].squeeze() > 0.5
            overlay = np.zeros((*mask.shape, 4))
            overlay[..., :3] = color[:3]
            overlay[..., 3] = mask * 0.3
            ax.imshow(overlay)
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor=color, facecolor="none", linewidth=1.5))
        caption = class_names.get(int(label), str(label))
        if predictions:
            caption += f" {objects['scores'][index]:.2f}"
        ax.text(x1, y1, caption, color="white", fontsize=8,
                bbox={"facecolor": color, "pad": 1})
    ax.axis("off")


@torch.inference_mode()
def plot_predictions(model, dataset, device, class_names, indices=(0, 1, 2), score_threshold=0.5):
    model.eval()
    fig, axes = plt.subplots(len(indices), 2, figsize=(13, 4 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        image, target = dataset[index]
        prediction = {key: value.cpu() for key, value in model([image.to(device)])[0].items()}
        keep = prediction["scores"] >= score_threshold
        prediction = {key: value[keep] for key, value in prediction.items()}
        _draw_instances(axes[row, 0], image, target, class_names)
        _draw_instances(axes[row, 1], image, prediction, class_names, predictions=True)
        axes[row, 0].set_title(f"Image {target['image_id']}: ground truth")
        axes[row, 1].set_title(f"Prediction (score >= {score_threshold})")
    fig.tight_layout()
    return fig


def export_evaluation_results(evaluators, output_dir, class_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_evaluators(evaluators)
    per_class = pd.concat([
        per_class_metrics(evaluator, class_names).assign(model=name)
        for name, evaluator in evaluators.items()
    ], ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    fig = plot_model_comparison(summary)
    fig.savefig(output_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    for index, (name, evaluator) in enumerate(evaluators.items(), start=1):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
        for iou_type in evaluator.coco_eval:
            fig = plot_precision_recall(evaluator, class_names, iou_type)
            fig.savefig(output_dir / f"{index:02d}_{stem}_{iou_type}_pr.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)
    return {"summary": summary, "per_class": per_class}
