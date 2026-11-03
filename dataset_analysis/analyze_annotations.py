"""Reproduce statistics directly from the four official TrashCan COCO files."""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sources" / "original" / "dataset"
OUT = ROOT / "analysis"


def video_id(name):
    match = re.search(r"vid_(\d+)_frame", name)
    return match.group(1) if match else None


def profile(data):
    images = data["images"]
    annotations = data["annotations"]
    categories = {c["id"]: c["name"] for c in data["categories"]}
    by_image = {i["id"]: i for i in images}
    counts = Counter(a["category_id"] for a in annotations)
    image_counts = Counter(a["image_id"] for a in annotations)
    sizes = Counter()
    issues = Counter()
    overshoot = []
    for ann in annotations:
        if ann["image_id"] not in by_image:
            issues["orphan_image_reference"] += 1
            continue
        im = by_image[ann["image_id"]]
        if ann["category_id"] not in categories:
            issues["unknown_category_reference"] += 1
        x, y, w, h = ann["bbox"]
        if not all(math.isfinite(v) for v in [x, y, w, h]) or w <= 0 or h <= 0:
            issues["nonpositive_or_nonfinite_bbox"] += 1
        margin = max(-x, -y, x + w - im["width"], y + h - im["height"], 0)
        if margin > 1e-6:
            issues["bbox_out_of_bounds"] += 1
            overshoot.append(margin)
        if not ann.get("segmentation"):
            issues["empty_segmentation"] += 1
        area = ann["area"]
        sizes["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    empty = sum(image_counts[i["id"]] == 0 for i in images)
    resolution = Counter(f"{i['width']}x{i['height']}" for i in images)
    video_ids = {video_id(i["file_name"]) for i in images}
    return {
        "images": len(images),
        "annotations": len(annotations),
        "classes": len(categories),
        "class_counts": {categories[c]: counts[c] for c in categories},
        "empty_annotation_images": empty,
        "annotations_per_image": len(annotations) / len(images),
        "resolutions": dict(resolution),
        "video_count": len(video_ids - {None}),
        "unparsed_video_ids": sum(video_id(i["file_name"]) is None for i in images),
        "duplicate_image_ids": len(images) - len(by_image),
        "duplicate_annotation_ids": len(annotations) - len({a['id'] for a in annotations}),
        "duplicate_image_filenames": len(images) - len({i['file_name'] for i in images}),
        "coco_area_size_counts": dict(sizes),
        "quality_issues": dict(issues),
        "max_bbox_overshoot_pixels": max(overshoot, default=0),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"method": "Counts from official COCO JSON annotations, not README tables. Size uses annotation area and thresholds 32^2,96^2 at original resolution. Filenames identify video groups; no full-image hash or perceptual duplicate audit was performed.", "versions": {}}
    loaded = {}
    for version in ["instance", "material"]:
        splits = {}
        source_hashes = {}
        for split in ["train", "val"]:
            path = DATA / f"{version}_version" / f"instances_{split}_trashcan.json"
            raw = path.read_bytes()
            splits[split] = json.loads(raw)
            source_hashes[split] = hashlib.sha256(raw).hexdigest()
        loaded[version] = splits
        combined = {"images": splits['train']['images'] + splits['val']['images'], "annotations": splits['train']['annotations'] + splits['val']['annotations'], "categories": splits['train']['categories']}
        profiles = {s: profile(d) for s, d in splits.items()}
        profiles['combined'] = profile(combined)
        filenames = {s: {i['file_name'] for i in d['images']} for s, d in splits.items()}
        videos = {s: {video_id(i['file_name']) for i in d['images']} - {None} for s, d in splits.items()}
        profiles['split_checks'] = {
            "shared_image_filenames": sorted(filenames['train'] & filenames['val']),
            "shared_video_ids": sorted(videos['train'] & videos['val']),
            "same_class_schema": splits['train']['categories'] == splits['val']['categories'],
        }
        profiles['sha256'] = source_hashes
        results['versions'][version] = profiles
        with (OUT / f"{version}_class_counts.csv").open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['class', 'train_objects', 'val_objects', 'total_objects'])
            for name, total in profiles['combined']['class_counts'].items():
                writer.writerow([name, profiles['train']['class_counts'][name], profiles['val']['class_counts'][name], total])
    name_sets = {v: {i['file_name'] for d in splits.values() for i in d['images']} for v, splits in loaded.items()}
    results['cross_version'] = {
        "same_image_filename_set": name_sets['instance'] == name_sets['material'],
        "shared_unique_filenames": len(name_sets['instance'] & name_sets['material']),
        "distinct_unique_filenames": len(name_sets['instance'] | name_sets['material']),
        "train_in_instance_val_in_material": len({i['file_name'] for i in loaded['instance']['train']['images']} & {i['file_name'] for i in loaded['material']['val']['images']}),
    }
    (OUT / "statistics.json").write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
