"""
Weight of Evidence Density Plot
Author: https://github.com/deburky
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import fastwoe
from fastwoe import FastWoe

print(f"fastwoe version: {fastwoe.__version__}")

# ---------------------------
# 1. Load data
# ---------------------------
ROOT_DIR = Path(__file__).parent.parent
data_path = ROOT_DIR / "data" / "BankCaseStudyData.csv"
df = pd.read_csv(data_path)

y = df["Final_Decision"].map({"Accept": 0, "Decline": 1}).values
x = df["Application_Score"].values

x_good = x[y == 0]
x_bad = x[y == 1]

# ---------------------------
# 2. Setup
# ---------------------------
x_grid = np.linspace(x.min() - 1, x.max() + 1, 500).reshape(-1, 1)
eps = 1e-6

colors = [
    "#d3aa3d",  # Normal parametric
    "#97d2f1",  # KDE
    "#1e9575",  # GMM
    "#e3e162",  # Histogram
    "#ef7b7b",  # GAM
    "#6a4c93",  # Isotonic
    "#4a90e2",  # FastWoe (tree)
    "#6a4c93",  # FastWoe (faiss_kmeans)
]

# ---------------------------
# 3. WOE Calculations
# ---------------------------

# Normal parametric fit
mu_good, std_good = np.mean(x_good), np.std(x_good)
mu_bad, std_bad = np.mean(x_bad), np.std(x_bad)
f_good_norm = norm.pdf(x_grid, mu_good, std_good)
f_bad_norm = norm.pdf(x_grid, mu_bad, std_bad)
woe_norm = np.log((f_bad_norm + eps) / (f_good_norm + eps))

# KDE
kde_good = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(x_good.reshape(-1, 1))
kde_bad = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(x_bad.reshape(-1, 1))
f_good_kde = np.exp(kde_good.score_samples(x_grid))
f_bad_kde = np.exp(kde_bad.score_samples(x_grid))
woe_kde = np.log((f_bad_kde + eps) / (f_good_kde + eps))

# GMM
gmm_good = GaussianMixture(n_components=2, random_state=42).fit(x_good.reshape(-1, 1))
gmm_bad = GaussianMixture(n_components=2, random_state=42).fit(x_bad.reshape(-1, 1))
f_good_gmm = np.exp(gmm_good.score_samples(x_grid))
f_bad_gmm = np.exp(gmm_bad.score_samples(x_grid))
woe_gmm = np.log((f_bad_gmm + eps) / (f_good_gmm + eps))

# Histogram / binned WOE
bins = np.histogram_bin_edges(x, bins="fd")
good_hist, _ = np.histogram(x_good, bins=bins, density=True)
bad_hist, _ = np.histogram(x_bad, bins=bins, density=True)
woe_hist = np.log((bad_hist + eps) / (good_hist + eps))
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# GAM WOE
gam = LogisticGAM(s(0, n_splines=20)).fit(x.reshape(-1, 1), y)
log_odds = gam._modelmat(x_grid) @ gam.coef_  # pylint: disable=protected-access
prior_odds = np.log(y.mean() / (1 - y.mean()))
woe_gam = log_odds - prior_odds

# FastWoe WOE Tree
encoder = FastWoe(binning_method="tree")
encoder.fit(x, y)
woe_fastwoe = encoder.transform(x_grid.reshape(-1, 1))

# FastWoe WOE (faiss_kmeans)
encoder = FastWoe(binning_method="faiss_kmeans")
encoder.fit(x, y)
woe_fastwoe_faiss = encoder.transform(x_grid.reshape(-1, 1))

# FastWoe WOE (Tree with monotonic constraints)
encoder = FastWoe(
    binning_method="tree", monotonic_cst={"Application_Score": -1}
)  # Use meaningful feature name - should work automatically now!
encoder.fit(x, y)
woe_fastwoe_mono = encoder.transform(x_grid.reshape(-1, 1))

# Debug: Check if monotonic and non-monotonic results are different
print(
    f"Tree without constraints - first 5 WOE values: {woe_fastwoe.values.flatten()[:5]}"
)
print(
    f"Tree with monotonic constraints - first 5 WOE values: {woe_fastwoe_mono.values.flatten()[:5]}"
)
print(f"Are they identical? {np.allclose(woe_fastwoe.values, woe_fastwoe_mono.values)}")

# ---------------------------
# 4. Plot
# ---------------------------
fig, ax1 = plt.subplots(figsize=(11, 6))

# WOE curves
# ax1.plot(x_grid, woe_norm, color=colors[0], label="Normal Parametric", linewidth=2)
# ax1.plot(x_grid, woe_kde, color=colors[1], label="KDE", linewidth=2)
ax1.plot(
    x_grid,
    woe_fastwoe_mono,
    color=colors[5],
    label="FastWoe (tree) - Monotonic",
    linewidth=2,
)
ax1.step(
    bin_centers, woe_hist, color=colors[1], label="Histogram", where="mid", linewidth=2
)
ax1.plot(x_grid, woe_gam, color=colors[2], label="GAM", linewidth=2)
ax1.plot(x_grid, woe_fastwoe, color=colors[3], label="FastWoe (tree)", linewidth=2)
ax1.plot(
    x_grid,
    woe_fastwoe_faiss,
    color=colors[4],
    label="FastWoe (faiss k-means)",
    linewidth=2,
)

ax1.axhline(0, color="black", linestyle="--", linewidth=1)

# Secondary axis for counts (hidden)
ax2 = ax1.twinx()
counts, _ = np.histogram(x, bins=bins)
ax2.bar(
    bin_centers,
    counts,
    width=(bins[1] - bins[0]) * 0.8,
    alpha=0.2,
    color="gray",
    align="center",
)
ax2.get_yaxis().set_visible(False)

# Title and labels
plt.suptitle(
    "Weight of Evidence (WOE) by Method — Application Score",
    fontsize=22,
    y=0.9,
)
ax1.set_xlabel("Application Score", fontsize=14)
ax1.set_ylabel("WOE(x)", fontsize=14)
ax1.tick_params(axis="both", which="major", labelsize=14)

# disable upper and right spines
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# You might also want to remove spines from ax2 if they're visible:
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend above plot, 3 columns, no frame
ax1.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.04),  # Centers horizontally, positions above plot
    ncol=3,
    fontsize=11,
    frameon=False,  # No frame like the first example
)

ax1.grid(False)
fig.tight_layout(rect=(0, 0, 1, 0.91))

# Create a temporary directory to save the image
with TemporaryDirectory() as temp_dir:
    image_path = os.path.join(temp_dir, "woe_density.png")
    plt.savefig(image_path)
    plt.show()
    print(f"WOE density plot saved to: {image_path}")
