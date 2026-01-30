"""
Plotting utilities for niche maps.

This module exposes `plot_niche_continuous_and_binary` which plots a
continuous niche map and a corresponding binary thresholded map side-by-side.
"""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def plot_niche_continuous_and_binary(
    niche_maps: np.ndarray,
    k: int = 0,
    threshold: Any = "p95",   # "p95" default, or give a numeric value like 1.5
    p: float = 99.5,
    cmap_cont: str = "RdBu_r",
) -> plt.Figure:
    """Plot continuous and binary versions of a niche map slice.

    Parameters
    ----------
    niche_maps : array-like
        3D array-like object with shape (H, W, K) or similar where the last
        axis indexes niche maps. Will be converted with ``np.asarray``.
    k : int, optional
        Index of the niche to plot (default 0).
    threshold : str or float, optional
        If a string starting with "p" (e.g. "p80"), interprets as a
        percentile for thresholding; otherwise converted to float and used
        directly as the numeric threshold.
    p : float, optional
        Percentile used to set the symmetric color scale (default 99.5).
    cmap_cont : str, optional
        Matplotlib colormap for the continuous plot.

    Returns
    -------
    matplotlib.figure.Figure
        The created Figure object.
    """

    arr = np.asarray(niche_maps)
    if arr.ndim < 3:
        raise ValueError("niche_maps must be at least 3D with last axis indexing niches")
    if not (0 <= k < arr.shape[-1]):
        raise IndexError(f"k out of range: got {k}, valid range 0..{arr.shape[-1]-1}")

    M = arr[..., k]

    # robust symmetric scale for continuous plot
    try:
        v = np.nanpercentile(np.abs(M), p)
    except Exception:
        v = np.nanmax(np.abs(M))
    if not np.isfinite(v) or v == 0:
        v = np.nanmax(np.abs(M)) if np.nanmax(np.abs(M)) != 0 else 1.0

    # choose threshold
    if isinstance(threshold, str) and threshold.startswith("p"):
        q = float(threshold[1:])  # e.g. "p95" -> 95.0
        t = np.nanpercentile(M, q)
        thr_label = f"p{q:g}={t:.3g}"
    else:
        t = float(threshold)
        thr_label = f"{t:.3g}"

    # binarize (True where M >= threshold)
    B = (M >= t)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(M, cmap=cmap_cont, vmin=-v, vmax=v)
    axes[0].set_title(f"Niche {k} (continuous)")
    axes[0].axis("off")
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label(f"Intensity (±p{p})")

    # reversed black/white (True=black, False=white)
    axes[1].imshow(B.astype(np.uint8), cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title(f"Niche {k} (binary ≥ {thr_label})")
    axes[1].axis("off")

    return fig


# examples:
# plot_niche_continuous_and_binary(Z_big_niche_maps, k=3)           # default p95
# plot_niche_continuous_and_binary(Z_big_niche_maps, k=3, threshold="p90")
# plot_niche_continuous_and_binary(Z_big_niche_maps, k=3, threshold=1.2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def plot_niche_continuous_and_percentile_categories(
    niche_maps,
    k=0,
    percentiles=(90, 95, 98, 99),
    p=99.5,
    cmap_cont="RdBu_r",
    cat_colors=("#6baed6", "#fd8d3c", "#de2d26", "#67000d"),
):
    M = niche_maps[:, :, k]

    # robust symmetric scale for continuous plot
    v = np.nanpercentile(np.abs(M), p)
    if not np.isfinite(v) or v == 0:
        v = np.nanmax(np.abs(M)) if np.nanmax(np.abs(M)) != 0 else 1.0

    percentiles = sorted(percentiles)
    thresholds = [np.nanpercentile(M, q) for q in percentiles]

    # categorical encoding
    # 0 = below p90
    # 1 = p90–p95
    # 2 = p95–p98
    # 3 = p98–p99
    # 4 = ≥ p99
    C = np.zeros_like(M, dtype=int)
    for i, t in enumerate(thresholds):
        C[M >= t] = i + 1

    cmap_cat = ListedColormap(["white", *cat_colors])
    norm = BoundaryNorm(range(len(cat_colors) + 2), cmap_cat.N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # continuous
    im0 = axes[0].imshow(M, cmap=cmap_cont, vmin=-v, vmax=v)
    axes[0].set_title(f"Niche {k} (continuous)")
    axes[0].axis("off")
    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label(f"Intensity (±p{p})")

    # categorical
    axes[1].imshow(C, cmap=cmap_cat, norm=norm)
    axes[1].set_title(f"Niche {k} (percentile categories)")
    axes[1].axis("off")

    # legend
    labels = [
        f"p{percentiles[0]}–p{percentiles[1]} ({thresholds[0]:.3g})",
        f"p{percentiles[1]}–p{percentiles[2]} ({thresholds[1]:.3g})",
        f"p{percentiles[2]}–p{percentiles[3]} ({thresholds[2]:.3g})",
        f"≥ p{percentiles[3]} ({thresholds[3]:.3g})",
    ]

    patches = [
        mpatches.Patch(color=cat_colors[i], label=labels[i])
        for i in range(len(cat_colors))
    ]

    axes[1].legend(
        handles=patches,
        loc="lower left",
        frameon=True,
        fontsize=9,
    )

    plt.show()