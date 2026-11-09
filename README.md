# comp9444-Automated-Marine-Debris-Detection-using-Deep-Learning

## Evaluation

The evaluation workflow is in `Marine_Debris_Blank_Notebook.ipynb`, sections
5.2-7.1. It compares the existing SGD, AdamW and augmented SGD Mask R-CNN
experiments on the same committed held-out test split.

```sh
python -m pip install -r requirements.txt
jupyter lab Marine_Debris_Blank_Notebook.ipynb
```

For GPU use, install a compatible PyTorch/torchvision pair for your platform
from https://pytorch.org/get-started/locally/ before installing the remaining
requirements. Run the notebook from the repository root with `instance_version/`
available. No trained checkpoints or measured model scores are included.

1. Run section 3.4 to reuse and audit the committed train/test split. It does
   not overwrite the split unless `REGENERATE_SPLIT=True`; a changed split
   requires retraining all compared models.
2. Run the model definition, dataset import, transforms, dataset setup and test
   dataset setup cells in sections 4-5.1. Train the desired models in section
   5.1, or skip the training loops and set `MODEL_CHECKPOINTS` in section 5.2.
   Checkpoints may be raw `state_dict` files or dictionaries containing
   `model_state_dict`; they must use the exact split and training category order.
   For example, after training use `torch.save(model.state_dict(), "sgd.pth")`
   and set `MODEL_CHECKPOINTS = {"Mask R-CNN / SGD": "sgd.pth"}`.
3. Run sections 5.2-7.1. Each available model is evaluated once on test images.
   The report shows which models were actually evaluated; unavailable models
   are not assigned placeholder scores.

Outputs under `evaluation_results/held_out_test/`:

- `summary.csv`: 12 standard COCO metrics for boxes and masks separately.
- `per_class_metrics.csv`: per-class AP, AP50 and AR100.
- PR curves, model comparison and matching qualitative sample PNGs.
- `evaluation_metadata.json`: annotation hashes, split audit, category mapping,
  inference settings, library versions and supplied checkpoint hashes.

Metrics use a 0-1 scale. Undefined COCO metrics are NaN (blank in CSV), not zero.
The display confidence threshold only affects qualitative figures. Evaluation
passes all predictions returned by the model to COCO; the model's built-in
score/NMS settings still apply. Ground truth preserves original COCO boxes,
segmentation areas and crowd flags, remapped to the shared training labels.
Images with no annotations remain valid examples and their false positives count.

The original COCO-pretrained baseline uses a different class vocabulary and is
excluded from TrashCAN class-aware numerical comparisons. The placeholder
Faster R-CNN section does not imply an implemented model or benchmark.

### Split limitation

The committed train/test sets have 4,970/1,095 images and disjoint source videos.
The official validation set shares 97 source videos with train and 16 with test
(160 validation images come from test videos). The notebook audits and reports
this overlap. Validation-based tuning is therefore not fully independent of
test video sources; report the limitation with results. A strict three-way
video-independent benchmark requires a new split and retraining. Do not select
hyperparameters or checkpoints using final test results.

### Verification

```sh
python -m unittest discover -s tests -v
```

Tests use small synthetic COCO annotations and temporary images; they do not
download model weights or train models. They cover perfect/empty predictions,
negative images, sparse category mappings, subset evaluation, COCO metric
extraction, plots/export and evaluation state restoration.

Metric definitions follow the
[official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
