import math
from numbers import Integral
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import help_funciton.utils as utils
from help_funciton.coco_eval import CocoEvaluator
from help_funciton.coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def _check_evaluation_coverage(expected_ids, seen_ids, errors):
    """Check all ranks together, allowing DistributedSampler padding across ranks."""
    coverage = utils.all_gather({"image_ids": list(seen_ids), "errors": sorted(errors)})
    all_errors = sorted({error for rank in coverage for error in rank["errors"]})
    if all_errors:
        raise ValueError("Invalid evaluation image IDs: " + "; ".join(all_errors))
    evaluated_ids = {image_id for rank in coverage for image_id in rank["image_ids"]}
    missing = expected_ids - evaluated_ids
    unexpected = evaluated_ids - expected_ids
    if missing or unexpected:
        raise ValueError(
            f"Evaluation did not cover the intended dataset: {len(missing)} missing "
            f"image IDs {sorted(missing)[:10]}, {len(unexpected)} unexpected image IDs "
            f"{sorted(unexpected)[:10]}. Disable drop_last and use a Subset dataset "
            "for intentional partial evaluation instead of a partial sampler."
        )


@torch.inference_mode()
def evaluate(model, data_loader, device, *, iou_types=None):
    """Evaluate unfiltered predictions in the dataset's training-label space.

    Faster R-CNN reports bounding-box metrics; Mask R-CNN also reports masks. Custom
    detection models may pass iou_types explicitly. No display score threshold
    is applied: COCO AP uses the full ranked predictions returned by the model.
    Every image in the loader's dataset must be evaluated. For a smaller test,
    wrap the dataset in Subset instead of using drop_last or a partial sampler.
    """
    if len(data_loader) == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    device = torch.device(device)
    n_threads = torch.get_num_threads()
    was_training = model.training
    try:
        torch.set_num_threads(1)
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        coco = get_coco_api_from_dataset(data_loader.dataset)
        if iou_types is None:
            iou_types = _get_iou_types(model)
        if not iou_types or any(t not in ("bbox", "segm", "keypoints") for t in iou_types):
            raise ValueError("iou_types must contain bbox, segm or keypoints.")
        coco_evaluator = CocoEvaluator(coco, iou_types)
        valid_labels = set(coco.getCatIds())
        expected_ids = set(coco.getImgIds())
        seen_ids, coverage_errors = set(), set()

        for images, targets in metric_logger.log_every(data_loader, 100, "Test:"):
            images = [img.to(device) for img in images]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_time = time.time()
            outputs = model(images)
            if len(outputs) != len(targets):
                raise ValueError("The model must return one prediction per image.")
            outputs = [{k: v.to("cpu") for k, v in output.items()} for output in outputs]
            model_time = time.time() - model_time
            for output in outputs:
                unknown = set(output["labels"].tolist()) - valid_labels
                if unknown:
                    raise ValueError(f"Prediction labels {sorted(unknown)} are outside the dataset label space.")
            res = {}
            for target, output in zip(targets, outputs):
                image_id = target.get("image_id")
                if isinstance(image_id, torch.Tensor) and image_id.numel() == 1:
                    image_id = image_id.item()
                if not isinstance(image_id, Integral) or isinstance(image_id, bool):
                    coverage_errors.add("each target image_id must be a scalar integer")
                    continue
                image_id = int(image_id)
                if image_id not in expected_ids:
                    coverage_errors.add(f"image_id {image_id} is outside the dataset")
                    continue
                if image_id in seen_ids:
                    coverage_errors.add(f"duplicate image_id {image_id} on the same rank")
                    continue
                seen_ids.add(image_id)
                res[image_id] = output
            evaluator_time = time.time()
            if res:
                coco_evaluator.update(res)
            metric_logger.update(model_time=model_time, evaluator_time=time.time() - evaluator_time)

        # Raise after every rank reaches this point, before COCO silently drops
        # duplicates or reports AP on only the images visited by the sampler.
        _check_evaluation_coverage(expected_ids, seen_ids, coverage_errors)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator
    finally:
        model.train(was_training)
        torch.set_num_threads(n_threads)
