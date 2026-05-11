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
    ax=None,
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

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    else:
        if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) != 2:
            raise ValueError("ax must be None or a sequence of two matplotlib Axes for this function")
        axes = ax
        fig = axes[0].figure

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _draw_percentile_block_legend(
    ax,
    percentiles,
    cat_colors,
    legend_show_thresholds=False,
    thresholds=None,
):
    """Draw a compact equal-height percentile legend on the provided axes."""
    percentiles = sorted(percentiles)
    n = len(percentiles)
    strip = np.arange(n)[:, None]
    cmap = ListedColormap(cat_colors)

    ax.imshow(
        strip,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(0, 1, 0, n),
    )

    ax.set_xticks([])

    tick_positions = np.arange(n)
    if legend_show_thresholds and thresholds is not None:
        ticklabels = [f"p{q}\n{t:.3g}" for q, t in zip(percentiles, thresholds)]
    else:
        ticklabels = [f"p{q}" for q in percentiles]

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(
        axis="y",
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        length=3,
        width=1,
        pad=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n)

    for spine in ax.spines.values():
        spine.set_visible(True)


def plot_percentile_bar_only(
    percentiles=(90, 95, 98, 99),
    cat_colors=("#6baed6", "#fd8d3c", "#de2d26", "#67000d"),
    legend_show_thresholds=False,
    thresholds=None,
    ax=None,
):
    """Plot only the compact equal-height percentile legend."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.25, 2))
    else:
        fig = ax.figure

    _draw_percentile_block_legend(
        ax=ax,
        percentiles=percentiles,
        cat_colors=cat_colors,
        legend_show_thresholds=legend_show_thresholds,
        thresholds=thresholds,
    )

    return fig


def plot_niche_continuous_only(
    niche_maps,
    k=0,
    p=99.5,
    cmap_cont="RdBu_r",
    module_labels=None,
    ax=None,
    show_colorbar=True,
):
    M = niche_maps[:, :, k]

    v = np.nanpercentile(np.abs(M), p)
    if not np.isfinite(v) or v == 0:
        vmax = np.nanmax(np.abs(M))
        v = vmax if vmax != 0 else 1.0

    if module_labels is not None and k in module_labels:
        title_suffix = f"{k}: {module_labels[k]}"
    else:
        title_suffix = f"{k}"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    else:
        fig = ax.figure

    im = ax.imshow(
        M,
        cmap=cmap_cont,
        vmin=-v,
        vmax=v,
        interpolation="bilinear",
    )
    ax.set_title(f"Niche {title_suffix} (continuous)")
    ax.axis("off")

    if show_colorbar:   # <-- conditional colorbar
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"Intensity (±p{p})")

    return fig


def plot_niche_percentile_categories_only(
    niche_maps,
    k=0,
    percentiles=(90, 95, 98, 99),
    cat_colors=("#6baed6", "#fd8d3c", "#de2d26", "#67000d"),
    module_labels=None,
    show_percentile_legend=True,
    legend_show_thresholds=False,
    ax=None,
):
    M = niche_maps[:, :, k]

    percentiles = sorted(percentiles)
    thresholds = [np.nanpercentile(M, q) for q in percentiles]

    C = np.zeros_like(M, dtype=int)
    for i, t in enumerate(thresholds):
        C[M >= t] = i + 1

    cmap_cat = ListedColormap(["white", *cat_colors])
    norm = BoundaryNorm(range(len(cat_colors) + 2), cmap_cat.N)

    if module_labels is not None and k in module_labels:
        title_suffix = f"{k}: {module_labels[k]}"
    else:
        title_suffix = f"{k}"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    else:
        fig = ax.figure

    ax.imshow(
        C,
        cmap=cmap_cat,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(f"Niche {title_suffix} (percentile categories)")
    ax.axis("off")

    if show_percentile_legend:
        cax = inset_axes(
            ax,
            width="4%",
            height="85%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.08, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        _draw_percentile_block_legend(
            ax=cax,
            percentiles=percentiles,
            cat_colors=cat_colors,
            legend_show_thresholds=legend_show_thresholds,
            thresholds=thresholds,
        )

    return fig


def plot_niche_continuous_and_percentile_categories(
    niche_maps,
    k=0,
    percentiles=(90, 95, 98, 99),
    p=99.5,
    cmap_cont="RdBu_r",
    cat_colors=("#6baed6", "#fd8d3c", "#de2d26", "#67000d"),
    module_labels=None,
    show_percentile_legend=True,
    legend_show_thresholds=False,
    ax=None,
):
    M = niche_maps[:, :, k]

    # ---- robust symmetric scale for continuous plot
    v = np.nanpercentile(np.abs(M), p)
    if not np.isfinite(v) or v == 0:
        vmax = np.nanmax(np.abs(M))
        v = vmax if vmax != 0 else 1.0

    percentiles = sorted(percentiles)
    thresholds = [np.nanpercentile(M, q) for q in percentiles]

    # ---- categorical encoding
    C = np.zeros_like(M, dtype=int)
    for i, t in enumerate(thresholds):
        C[M >= t] = i + 1

    cmap_cat = ListedColormap(["white", *cat_colors])
    norm = BoundaryNorm(range(len(cat_colors) + 2), cmap_cat.N)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    else:
        if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) != 2:
            raise ValueError("ax must be None or a sequence of two matplotlib Axes for this function")
        axes = ax
        fig = axes[0].figure

    # ---- Title handling
    if module_labels is not None and k in module_labels:
        title_suffix = f"{k}: {module_labels[k]}"
    else:
        title_suffix = f"{k}"

    # =========================================================
    # Continuous map
    # =========================================================
    im0 = axes[0].imshow(
        M,
        cmap=cmap_cont,
        vmin=-v,
        vmax=v,
        interpolation="bilinear",
    )
    axes[0].set_title(f"Niche {title_suffix} (continuous)")
    axes[0].axis("off")

    cb = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label(f"Intensity (±p{p})")

    # =========================================================
    # Categorical map
    # =========================================================
    axes[1].imshow(
        C,
        cmap=cmap_cat,
        norm=norm,
        interpolation="nearest",
    )
    axes[1].set_title(f"Niche {title_suffix} (percentile categories)")
    axes[1].axis("off")

    # =========================================================
    # Vertical percentile block legend
    # =========================================================
    
    # ---- Place bar directly to the RIGHT of categorical plot
    if show_percentile_legend:
        cax = inset_axes(
            axes[1],
            width="4%",           # thin vertical bar
            height="85%",         # match plot height visually
            loc="lower left",
            bbox_to_anchor=(1.02, 0.08, 1, 1),   # push just outside plot
            bbox_transform=axes[1].transAxes,
            borderpad=0,
        )

        _draw_percentile_block_legend(
            ax=cax,
            percentiles=percentiles,
            cat_colors=cat_colors,
            legend_show_thresholds=legend_show_thresholds,
            thresholds=thresholds,
        )

    plt.show()
    return fig