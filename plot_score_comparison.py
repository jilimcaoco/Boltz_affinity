#!/usr/bin/env python3
"""Compare affinity scores between master_scores.csv and AA2AR_subposes_scores.csv."""

import csv
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/home/limcaoco/turbo/limcaoco/Boltz_affinity"

# ── Parse subposes scores ──
with open(f"{BASE}/AA2AR_subposes_scores.csv", "r") as f:
    content = f.read().replace("\r\n", "\n").replace("\r", "\n")
reader = csv.DictReader(io.StringIO(content))
subposes = {}
for row in reader:
    chembl_id = row["ligand_name"].split()[0]
    subposes[chembl_id] = float(row["affinity_score"])

# ── Parse master scores ──
master = {}
with open(f"{BASE}/master_scores.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cid = row["compound_ID"]
        if cid in subposes:
            master[cid] = float(row["Affinity Pred Value"])

# ── Align data ──
compounds = sorted(subposes.keys())
master_vals = np.array([master[c] for c in compounds])
subposes_vals = np.array([subposes[c] for c in compounds])

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: scatter / correlation
ax = axes[0]
ax.scatter(master_vals, subposes_vals, s=80, c="#2563eb", edgecolors="black", linewidths=0.5, zorder=3)
for i, cid in enumerate(compounds):
    ax.annotate(cid.replace("CHEMBL", ""), (master_vals[i], subposes_vals[i]),
                fontsize=7, textcoords="offset points", xytext=(5, 5), color="gray")

# Identity line
lo = min(master_vals.min(), subposes_vals.min()) - 0.2
hi = max(master_vals.max(), subposes_vals.max()) + 0.2
ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6, label="y = x")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Master Score (SMILES-based)", fontsize=11)
ax.set_ylabel("Subposes Score (MOL2 pose-based)", fontsize=11)
ax.set_title("Affinity Score Correlation", fontsize=13)
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Correlation coefficient
r = np.corrcoef(master_vals, subposes_vals)[0, 1]
ax.text(0.05, 0.92, f"r = {r:.3f}  (n={len(compounds)})",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

# Right panel: paired dot plot (lollipop)
ax2 = axes[1]
y_pos = np.arange(len(compounds))
short_labels = [c.replace("CHEMBL", "") for c in compounds]

for i in range(len(compounds)):
    ax2.plot([master_vals[i], subposes_vals[i]], [y_pos[i], y_pos[i]],
             color="gray", linewidth=1.5, zorder=1)

ax2.scatter(master_vals, y_pos, s=70, c="#ef4444", edgecolors="black",
            linewidths=0.5, zorder=2, label="Master (SMILES)")
ax2.scatter(subposes_vals, y_pos, s=70, c="#2563eb", edgecolors="black",
            linewidths=0.5, zorder=2, label="Subposes (MOL2)")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_labels, fontsize=9)
ax2.set_xlabel("Affinity Pred Value", fontsize=11)
ax2.set_title("Per-Compound Score Comparison", fontsize=13)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, axis="x", alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
out_path = f"{BASE}/score_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Print summary stats
diffs = subposes_vals - master_vals
print(f"\nMean diff (subposes - master): {diffs.mean():.4f}")
print(f"Std  diff: {diffs.std():.4f}")
print(f"Pearson r: {r:.4f}")
