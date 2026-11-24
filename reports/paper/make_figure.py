"""Methodology workflow figure (single-column, real content only)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("fig1_pipeline.png")

STAGES = [
    ("Census 2011 sources\nD1, D2, D5", "#1f4e79"),
    ("Harmonise geography\n640/640 keys, 628 reliable", "#24688f"),
    ("Validate target\nT2 vs count-based T3", "#2d7fa6"),
    ("Build features\nA: 43, B: 77", "#3b93b5"),
    ("Nested CV\n5 outer x 4 inner, grouped", "#4aa3bf"),
    ("Error analysis and\ntail-aware experiments", "#5fb0c6"),
]

COLS = 2
W, H = 6.6, 4.4
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

box_w, box_h = 2.85, 1.0
x0, y_top = 0.28, H - 0.35
h_gap = (W - 2 * x0 - COLS * box_w) / (COLS - 1)
v_gap = 0.42

positions = []
for index, (label, colour) in enumerate(STAGES):
    row, col = divmod(index, COLS)
    x = x0 + col * (box_w + h_gap)
    y = y_top - box_h - row * (box_h + v_gap)
    positions.append((x, y))
    ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                linewidth=1.1, edgecolor="#12374f", facecolor=colour))
    ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center",
            fontsize=7.6, color="white", linespacing=1.4)

for index in range(len(STAGES) - 1):
    (x1, y1), (x2, y2) = positions[index], positions[index + 1]
    if index % COLS == 0:  # left box -> right box, same row
        start, end = (x1 + box_w + 0.03, y1 + box_h / 2), (x2 - 0.03, y2 + box_h / 2)
    else:                  # right box -> left box of next row
        start = (x1 + box_w / 2, y1 - 0.03)
        end = (x2 + box_w / 2, y2 + box_h + 0.03)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color="#12374f",
                                 connectionstyle="arc3,rad=0.0"))

ax.text(x0, 0.16, "Frozen at every stage: seed 42, documented commands, hashed artefacts",
        fontsize=7.0, color="#333333", ha="left", style="italic")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
