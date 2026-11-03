# Project 014: Dataset(s) and Data Analysis

本文件提供可直接放入课程展示的两页英文内容、英文讲稿、中文讲解及可核查的统计。数据集信息和统计已完成；训练方案及额外评估均为建议，尚未训练模型。建议以 TrashCan-Instance 为主要实验版本，最终版本选择仍由小组确定。

## Slide 5 — Dataset(s)

### English slide copy

**TrashCan 1.0: underwater marine debris detection**

- **Source:** 7,212 underwater images derived from JAMSTEC's J-EDI ROV videos [1, 2].
- **Annotations:** COCO JSON with object categories, bounding boxes and instance segmentation masks [2].
- **Label schemes:** object types in TrashCan-Instance, materials in TrashCan-Material. Both include marine life and ROVs.
- **Image sizes:** 480 × 270 or 480 × 360 pixels [3].

| Label scheme | Total classes | Training images | Validation images | Object annotations* |
|---|---:|---:|---:|---:|
| Instance | 22 | 6,065 | 1,147 | 12,128 |
| Material | 16 | 6,008 | 1,204 | 12,336 |

*Training + validation. Both schemes use the same 7,212 image filenames. Counts come from the original COCO files [3].*

**Suggested starting point:** TrashCan-Instance for detecting specific debris types.

Footer: Hong et al. (2020), TrashCan 1.0. DOI: 10.13020/g1gx-y834.

### Suggested layout

保留原模板的标题。上半部分放前三条说明，下半部分放表格；图像尺寸和版本建议可口头补充。若加入样例图，使用下载的真实图像，并保留 JAMSTEC 水印。不要用生成图片替代真实数据样例。

### English speaker notes (about 60–75 seconds)

We use TrashCan 1.0, which contains 7,212 underwater images derived from JAMSTEC's J-EDI videos. The dataset provides COCO annotations for object detection and instance segmentation, so it supports both Faster R-CNN and Mask R-CNN.

There are two label schemes. The Instance version groups debris by object type, such as bags, bottles and cups. It contains 22 classes, including marine life and ROVs, with 6,065 training images and 1,147 validation images. The Material version groups debris by material, such as plastic and metal, and contains 16 classes.

Both versions use the same image collection, so we count 7,212 images in total. Our suggested starting point is the Instance version because its labels directly describe the objects we aim to detect. The native images have two aspect ratios, which we should preserve during resizing.

### 中文讲解

我们真正使用的数据集名称是 **TrashCan 1.0**。J-EDI 是原始视频来源，不能把二者当作两个独立的训练数据集。图像从水下遥控机器人 ROV 的视频中选取，因此画面也会出现机器人设备、鱼类、海星、植物等。

Instance 版本按物体类型分类，例如袋子、瓶子和杯子；Material 版本按材质分类，例如塑料和金属。两个版本总类别数分别是 22 和 16，其中垃圾类分别为 14 和 8，其余都是 8 类非垃圾对象。数据中的“实例分割”表示每一个物体有独立的像素区域标签；Instance 版本则是按物体类型组织类别，两处的 instance 不要混淆。

这里的 12,128／12,336 是物体标注数量，不是图片数量。同一张图像可能有多个物体。两版本的图像文件名集合相同，但物体标注数量、类别及训练／验证划分不同。

## Slide 6 — Data Analysis

### English slide copy

**TrashCan-Instance: data characteristics and training implications**

- **Class imbalance:** unknown debris accounts for **46.2%** of training debris annotations. Bags have **727** training objects, while cups have only **47** [3].
- **Small objects:** **23.1%** of training object annotations have mask area below **32² pixels** at native resolution [3].
- **Visual difficulty:** inspected examples show low contrast, suspended particles and uneven illumination [4].
- **Split dependence:** **113 video IDs** appear in both official splits. A separate evaluation with disjoint video groups would better assess generalisation to unseen videos [3].

**Proposed response:** class-aware sampling, aspect-ratio-preserving augmentation, per-class AP and a video-disjoint evaluation.

Footer: Counts computed from the official Instance COCO files. Visual observations refer to inspected examples. Proposed responses have not yet been evaluated.

### Figure

![Training debris class distribution](dataset_work/analysis/instance_training_class_distribution.png)

图中统计的是 **Instance 训练集的 14 个垃圾类别的物体标注数**，不包含 8 个非垃圾类别。橙色的 unknown instance 是一个有效的兜底标签，不代表标注文件缺失。46.2% 的分母是训练集全部 **4,766 个垃圾物体标注**，不是全部图像，也不是全部 9,540 个物体标注。

若一页空间有限，可让图表占主要区域，将四条分析缩成右侧要点，并将具体预处理建议放进讲稿。完整 14 类图表适合大面积展示，不建议缩成角落的小图。

### English speaker notes (about 75–90 seconds)

Our analysis of the official Instance annotations identifies several challenges. First, the training distribution is highly uneven. Unknown debris accounts for 46.2 percent of debris annotations, and there are 727 bags but only 47 cups. This suggests that we should examine performance for individual classes and consider sampling methods for rare classes.

Second, 23.1 percent of training annotations are small according to the mask-area threshold shown here. The inspected images also contain low contrast, suspended particles and uneven lighting, which can make boundaries harder to distinguish. We therefore propose preserving aspect ratio and applying augmentations consistently to images, boxes and masks.

Finally, the official training and validation sets share 113 video IDs. Their image filenames do not overlap, but related frames may still look similar. We can retain the official split for comparison with published work and add an evaluation in which no video appears in both training and evaluation. These are findings from the data and proposed experiments, rather than completed model results.

### 中文讲解

这一页应回答“数据有什么特点，这些特点为什么影响建模”。

1. **类别不平衡。** Instance 训练集共有 4,766 个垃圾物体标注，其中 2,203 个属于 `trash_unknown_instance`，占 46.2%。明确类型中，袋子有 727 个，杯子只有 47 个，约相差 15.5 倍。因此不能只看总体分数，应看每类 AP；类别采样或损失调整需要作为实验检验，不能预先宣称有效。
2. **小目标。** 按原始标注的 `area` 字段，以 32² 像素为阈值计算，训练集全部 9,540 个物体标注中有 2,201 个属于 small，占 23.1%。这个统计使用掩码面积，包括非垃圾类别，不能写成“23.1% 的垃圾图片是小目标”。保持宽高比，并检查缩放是否让小目标更难辨认。
3. **图像难点。** 查看过的少量真实样例存在对比度低、悬浮颗粒和光照不均。这里属于定性观察，尚未测量全数据集的模糊、暗图或遮挡比例。可测试颜色／亮度增强，但是否改善效果应由对照实验决定。
4. **视频相关性。** 根据文件名 `vid_XXXXXX_frameXXXXXXX.jpg` 提取视频 ID，Instance 的训练／验证集共享 113 个 ID。文件名没有重复，但同视频帧可能相似，存在验证偏乐观的风险。这个检查不能证明模型成绩已经受到多大影响，也不等于检测到了重复图像。建议保留官方划分作为 benchmark，另做视频组隔离评估并单独报告结果。

## Supporting statistics and checks

以下用于回答老师提问或撰写报告，不必全部放进两页 PPT。全部数字来自本地计算 [3]。

| Statistic | Instance | Material |
|---|---:|---:|
| Images, training / validation | 6,065 / 1,147 | 6,008 / 1,204 |
| Objects, training / validation | 9,540 / 2,588 | 9,741 / 2,595 |
| Classes, debris / other | 14 / 8 | 8 / 8 |
| Objects labelled as debris, both splits | 6,006 | 6,214 |
| Image dimensions 480 × 270 / 480 × 360, both splits | 3,967 / 3,245 | 3,967 / 3,245 |
| Distinct video IDs, both splits | 312 | 312 |
| Video IDs present in both splits | 113 | 127 |
| Exact filename overlap between training and validation | 0 | 0 |
| Training images with no object annotations | 129 | 0 |
| Validation images with no object annotations | 0 | 0 |
| Duplicate image IDs / annotation IDs within each version | 0 / 0 | 0 / 0 |
| Missing referenced image / class records | 0 / 0 | 0 / 0 |

Additional preparation notes:

- **Empty annotations:** 129 Instance training images have no object annotations. Review these images before treating them as verified background examples. This analysis has not inspected all 129 images and does not establish whether they contain unlabelled debris.
- **Boundary coordinates:** 3,286 Instance boxes and 3,346 Material boxes extend slightly beyond the image boundary. The largest overshoot is about 0.998 pixels. This is consistent with a coordinate-conversion edge effect, but its cause has not been established. Clip coordinates consistently in the data loader, review segmentation coordinates, and log any changes. These are preparation recommendations; no annotations have been modified.
- **README discrepancies:** some total counts in the bundled README do not equal their train-plus-validation entries. For example, the Instance README lists 3,447 ROV objects, but its split counts and the JSON give 2,633 + 684 = 3,317. Use the actual JSON counts for experiments.
- **Different splits across schemes:** 731 filenames occur in Instance training and Material validation. A cross-version comparison should disclose the different official splits. A controlled comparison can use a shared, explicitly documented split.
- **Validation distribution:** small objects account for 35.4% of Instance validation annotations versus 23.1% in training, using the same mask-area definition. This differs from evidence about small-object model performance, which requires training and evaluation.
- **Evaluation:** report box AP for detection and mask AP for segmentation separately, with the chosen IoU thresholds. Per-class AP is especially useful for rare debris categories. If reporting debris-only AP in addition to all-class AP, state the class subset explicitly.
- **Test data:** the distributed COCO files provide training and validation annotations. Do not call the validation set an independent test set. An additional held-out test set requires an explicit split and should remain unused during model selection.
- **Augmentation:** resize while preserving aspect ratio, and transform boxes and masks with the image. Fit any dataset-derived normalization on training images only. Apply augmentation and sampling only to training data.
- **Version choice:** starting with Instance is a recommendation. Neither version is universally better, and the published baselines are not this group's experimental results.
- **Scope of inspection:** all four COCO JSON files and the ZIP member manifest were read. Six sample images were downloaded. A full-image integrity, content-duplicate, near-duplicate or visual-quality audit has not been performed.

## Access and provenance

The teacher's item URL maps to handle `11299/214865` and DOI `10.13020/g1gx-y834`. The normal webpage returned HTTP 403 to the browsing tool, but the repository's public REST API returned the record and original files successfully. The source record and file manifest are saved under `dataset_work/sources/`.

The official `dataset.zip` contains images, COCO annotations and original annotation data. Its listed size is 553,029,970 bytes, about 527.4 MiB. The two checkpoint ZIPs are approximately 9.55 GB each and contain pretrained network configurations and checkpoints. The checkpoint ZIPs are unnecessary for dataset inspection and were not downloaded. This work used byte-range reads to retrieve the four COCO JSON files, READMEs and six example images from the official archive.

Repository metadata permits academic teaching/research use and states that commercial use requires JAMSTEC permission. Consult the distributed license for full terms.

### Sources

1. Teacher-provided official record: https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7 . DOI: https://doi.org/10.13020/g1gx-y834 . Public metadata: https://conservancy.umn.edu/server/api/core/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7 . Local copy: `dataset_work/sources/drum_item.json`.
2. Hong, J., Fulton, M., & Sattar, J. (2020). *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris*. arXiv:2007.08097. https://arxiv.org/abs/2007.08097 . Official lab description: https://irvlab.cs.umn.edu/resources/trashcan . The paper establishes provenance, collection and annotation methods. Counts in this file come from the actual annotations.
3. Original archive: https://conservancy.umn.edu/server/api/core/bitstreams/fffa78cc-3abf-42b9-b019-bc15a953a8e9/content . Four source files: `instance_version/instances_train_trashcan.json`, `instance_version/instances_val_trashcan.json`, `material_version/instances_train_trashcan.json`, `material_version/instances_val_trashcan.json`. Extracted under `dataset_work/sources/original/dataset/`. Reproduction script: `dataset_work/analyze_annotations.py`. Results with SHA-256 hashes: `dataset_work/analysis/statistics.json`. Class tables: `dataset_work/analysis/instance_class_counts.csv` and `material_class_counts.csv`.
4. Qualitative examples from the official archive: `dataset_work/sources/samples/`. In particular, `vid_000005_frame0000010.jpg` shows suspended particles and low contrast, while `vid_000145_frame0000067.jpg` shows uneven illumination and an obscured seabed. These are examples, not estimates of dataset-wide prevalence.

### Reproduce the analysis

Run from the COMP9444 workspace:

```powershell
python Project/dataset_work/fetch_sources.py --extract --samples
python Project/dataset_work/analyze_annotations.py
python Project/dataset_work/plot_analysis.py
```

The first command needs network access. The remaining commands use the saved original annotations. The figure script requires matplotlib. The acquisition script requires requests.
