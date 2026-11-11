"""TrashCan instance data with one shared training-label vocabulary."""

from pathlib import Path

import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image


class MarineDataset(Dataset):
    """Load COCO annotations while mapping category IDs to labels 1..K.

    Pass the *training* ``category_ids`` in the same order to every split.
    ``coco`` retains the original category IDs and annotations; returned targets
    use the mapped labels. Evaluation must use the same mapping for its ground
    truth. Images without annotations remain valid negative examples.
    """

    def __init__(self, image_dir, annotation_file, category_ids, transforms=None):
        self.image_dir = str(image_dir)
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(self.coco.getImgIds())
        self.category_ids = list(category_ids)
        self.transforms = transforms
        if not self.category_ids or len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("category_ids must contain unique training category IDs.")
        self.cat_to_label = {
            category_id: index + 1
            for index, category_id in enumerate(self.category_ids)
        }
        self.label_to_cat = {label: cat for cat, label in self.cat_to_label.items()}
        annotation_categories = {
            ann["category_id"] for ann in self.coco.dataset.get("annotations", [])
        }
        unknown = annotation_categories - self.cat_to_label.keys()
        if unknown:
            raise ValueError(
                "Annotations contain categories missing from the training mapping: "
                f"{sorted(unknown)}"
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = int(self.image_ids[idx])
        info = self.coco.loadImgs(image_id)[0]
        image = read_image(
            str(Path(self.image_dir) / info["file_name"]), mode=ImageReadMode.RGB
        )
        height, width = image.shape[-2:]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))

        boxes, labels, masks, areas, crowds = [], [], [], [], []
        for ann in annotations:
            # Keep the supplied COCO box and segmentation area. Reconstructing
            # either from the raster mask changes the evaluation ground truth.
            x, y, box_width, box_height = ann["bbox"]
            box = torch.tensor(
                [x, y, x + box_width, y + box_height], dtype=torch.float32
            )
            if not torch.isfinite(box).all() or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Annotation {ann['id']} has an invalid bounding box.")
            mask = torch.as_tensor(self.coco.annToMask(ann), dtype=torch.uint8)
            if tuple(mask.shape) != (height, width):
                raise ValueError(
                    f"Image {image_id} dimensions do not match its COCO annotations."
                )
            boxes.append(box)
            labels.append(self.cat_to_label[ann["category_id"]])
            masks.append(mask)
            areas.append(float(ann.get("area", mask.sum().item())))
            crowds.append(int(ann.get("iscrowd", 0)))

        boxes = torch.stack(boxes) if boxes else torch.empty((0, 4), dtype=torch.float32)
        masks = (
            torch.stack(masks)
            if masks
            else torch.empty((0, height, width), dtype=torch.uint8)
        )
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=(height, width)
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(masks),
            "image_id": image_id,
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        image = tv_tensors.Image(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
