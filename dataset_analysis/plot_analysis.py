"""Export a presentation-sized figure from the audited training annotations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parent
stats = json.loads((ROOT / "analysis" / "statistics.json").read_text())
counts = stats["versions"]["instance"]["train"]["class_counts"]
data = sorted(((name.removeprefix("trash_").replace("_", " "), value) for name, value in counts.items() if name.startswith("trash_")), key=lambda pair: pair[1])
fig, ax = plt.subplots(figsize=(12, 7.1), dpi=180)
fig.patch.set_facecolor("white")
colors = ["#E3A008" if name == "unknown instance" else "#176B87" for name, _ in data]
bars = ax.barh([name for name, _ in data], [value for _, value in data], color=colors, height=.68)
ax.bar_label(bars, labels=[f"{value:,}" for _, value in data], padding=5, fontsize=12)
ax.set_xlim(0, max(v for _, v in data) * 1.15)
ax.set_xlabel("Annotated objects in the official training split", fontsize=12, labelpad=12)
ax.set_title("TrashCan-Instance: training debris class distribution", loc="left", fontsize=19, fontweight="bold", pad=22)
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.tick_params(axis="both", labelsize=12, length=0)
ax.tick_params(axis="y", pad=9)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#DDE4E8", linewidth=.7)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(.02, .025, "Source: official dataset.zip / instance_version / instances_train_trashcan.json.\nCounts refer to object annotations, not images. 14 debris classes; 22 classes including ROV and marine life.", fontsize=10, color="#475569")
fig.subplots_adjust(left=.22, right=.98, top=.87, bottom=.16)
output = ROOT / "analysis" / "instance_training_class_distribution.png"
fig.savefig(output, facecolor="white")
print(output)
