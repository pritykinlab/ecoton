"""
Plotting utilities for colocalization modules.

This module provides functions for visualizing gene-gene interaction graphs
using UMAP embeddings with cluster labels.
"""

import numpy as np
import umap
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from adjustText import adjust_text
import textwrap


def plot_colocalization_modules(
    G_ig,
    module_labels,  # <-- can be dict {module_id: label} OR list (see below)
    myclusters=None,
    title="Colocalization Modules",
    figsize=(4, 4),
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    label_font_size=8,
    wrap_width=18,
    ax=None,
):
    """
    Plot UMAP embedding of colocalization graph with module labels.

    Parameters
    ----------
    G_ig : ig.Graph
        igraph Graph object with 'cluster' vertex attribute and 'weight' edge attribute.
    module_labels : dict[int, str] OR list[str]
        Prefer dict keyed by true module id (e.g. {0:"...", 22:"...", 24:"B cell"}).
        If you pass a list, labels are assumed to be aligned to module id == list index.
    myclusters : list of int, optional
        Cluster IDs to highlight. If None, highlights all clusters present in module_labels.
    title : str, default "Colocalization Modules"
        Plot title.
    figsize : tuple, default (4, 4)
        Figure size if ax is None.
    n_neighbors : int, default 15
        UMAP n_neighbors parameter.
    min_dist : float, default 0.1
        UMAP min_dist parameter.
    random_state : int, default 42
        Random state for UMAP.
    label_font_size : int, default 8
        Font size for cluster labels.
    wrap_width : int, default 18
        Character width for text wrapping.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """

    # ===============================
    # STEP 0. Normalize module_labels
    # ===============================
    # If module_labels is a list, treat as {idx: label}
    if isinstance(module_labels, (list, tuple, np.ndarray)):
        module_labels = {i: str(lbl) for i, lbl in enumerate(module_labels)}

    # ===============================
    # STEP 1. Ensure weights exist
    # ===============================
    if "weight" not in G_ig.es.attributes():
        print("No edge weights found — setting all weights to 1.0")
        G_ig.es["weight"] = [1.0] * G_ig.ecount()

    print(f"Graph has {G_ig.vcount()} nodes and {G_ig.ecount()} edges")

    # ===============================
    # STEP 2. Use full graph and all cluster labels
    # ===============================
    # `G_ig.vs['cluster']` may be an int per-vertex or a list of archetype
    # indices (soft membership). Choose the first membership as the primary
    # cluster for coloring; absent membership -> -1.
    raw_membership = np.array(G_ig.vs["cluster"], dtype=object)
    membership = []
    for x in raw_membership:
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) > 0:
                try:
                    membership.append(int(x[0]))
                except Exception:
                    membership.append(-1)
            else:
                membership.append(-1)
        else:
            try:
                membership.append(int(x))
            except Exception:
                membership.append(-1)

    membership = np.array(membership, dtype=int)

    G_sub = G_ig  # FULL GRAPH, no filtering
    membership_sub = membership

    print(f"Using FULL graph: {G_sub.vcount()} nodes, {G_sub.ecount()} edges")

    # ===============================
    # STEP 3. Distance matrix from weighted adjacency
    # ===============================
    A_sub = np.array(G_sub.get_adjacency(attribute="weight").data, dtype=float)
    max_weight = A_sub.max()
    distance_matrix = max_weight - A_sub

    # ===============================
    # STEP 4. UMAP
    # ===============================
    print("Running UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state
    )
    embedding = umap_model.fit_transform(distance_matrix)

    print(f"UMAP embedding shape: {embedding.shape}")

    # ===============================
    # STEP 5. Colors
    # ===============================
    unique_clusters = sorted(np.unique(membership_sub))
    palette = sns.husl_palette(n_colors=len(unique_clusters))
    color_map = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    # ===============================
    # STEP 6. DataFrame
    # ===============================
    df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "cluster": membership_sub
    })

    # ===============================
    # STEP 7. KDE label placement
    # ===============================
    print("Computing density peaks...")
    label_positions = {}

    # FIX: default to actual module ids from module_labels, not range(len(...))
    if myclusters is None:
        myclusters = sorted(module_labels.keys())

    for cluster_id in unique_clusters:
        if cluster_id not in myclusters:
            continue

        subset = df[df["cluster"] == cluster_id][["x", "y"]].values.T

        if subset.shape[1] <= 3:
            continue

        kde = gaussian_kde(subset)
        densities = kde(subset)
        max_idx = np.argmax(densities)
        label_positions[cluster_id] = subset[:, max_idx]

    # ===============================
    # STEP 8. Plot
    # ===============================
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.axis("off")

    gray = (0.75, 0.75, 0.75)
    is_target = np.isin(membership_sub, myclusters)

    # Background (non-target) points
    ax.scatter(
        df.loc[~is_target, "x"],
        df.loc[~is_target, "y"],
        c=[gray] * (~is_target).sum(),
        s=8,
        alpha=0.8,
        zorder=1
    )

    # Highlighted clusters
    ax.scatter(
        df.loc[is_target, "x"],
        df.loc[is_target, "y"],
        c=[color_map[c] for c in membership_sub[is_target]],
        s=8,
        alpha=0.9,
        zorder=2
    )

    # ===============================
    # STEP 9. Text Labels
    # ===============================
    texts = []
    xs = []
    ys = []

    for cid, (lx, ly) in label_positions.items():
        # FIX: look up by true module id
        if cid not in module_labels:
            continue

        label = f"{cid}. {module_labels[cid]}"
        wrapped = "\n".join(textwrap.wrap(label, wrap_width))

        t = ax.text(
            lx, ly,
            wrapped,
            fontsize=label_font_size,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=10
        )
        texts.append(t)
        xs.append(lx)
        ys.append(ly)

    # ===============================
    # STEP 10. Adjust Text w/ Small Arrow
    # ===============================
    adjust_text(
        texts,
        x=xs,
        y=ys,
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            lw=0.5,
            shrinkA=3,
            shrinkB=4
        ),
        force_points=0.15,
        force_text=0.4,
        expand_points=(1.05, 1.05),
        expand_text=(1.08, 1.08),
        only_move={"text": "xy"}
    )

    ax.set_title(title)
    plt.tight_layout()

    return ax

def plot_gene_program_from_W(
    W_df,
    archetype_idx: int,
    threshold=0.3,
    top_n=25,
    title=None,
    figsize=(2.0, 5.0),
    gene_fontsize=10,
    axis_fontsize=9,
    title_fontsize=11,
    bar_height=0.55,
    include_neg1=True,          # <- set True if UMAP had -1 in unique_clusters
    unique_clusters=None,        # <- BEST: pass sorted(np.unique(membership_sub)) from UMAP
):
    col = f"archetype_{archetype_idx}"
    if col not in W_df.columns:
        raise KeyError(f"Column '{col}' not found in W_df.columns")

    # --- determine cluster ordering exactly like plot_colocalization_modules ---
    arch_ids = sorted(
        int(c.split("_")[-1]) for c in W_df.columns if c.startswith("archetype_")
    )

    if unique_clusters is not None:
        cluster_order = list(unique_clusters)
    else:
        cluster_order = ([-1] + arch_ids) if include_neg1 else arch_ids

    palette = sns.husl_palette(n_colors=len(cluster_order))
    color_map = {cid: palette[i] for i, cid in enumerate(cluster_order)}

    if archetype_idx not in color_map:
        raise KeyError(
            f"archetype_idx={archetype_idx} not in color map. "
            f"cluster_order={cluster_order[:10]}{'...' if len(cluster_order)>10 else ''}"
        )

    bar_color = color_map[archetype_idx]

    # --- filter + sort genes ---
    series = W_df[col]
    s = series[series >= threshold].sort_values(ascending=True).tail(top_n)

    genes = s.index.tolist()
    vals  = s.values
    y = np.arange(len(s))

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, vals, height=bar_height, color=bar_color)

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=gene_fontsize)

    xmax = float(vals.max()) if len(vals) else 1.0
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("Weight", fontsize=axis_fontsize)

    if title is None:
        title = f"A{archetype_idx} Top Genes"
    ax.set_title(title, fontsize=title_fontsize, pad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=axis_fontsize)
    ax.tick_params(axis="y", length=0)

    ax.xaxis.set_ticks_position("bottom")
    ax.grid(axis="x", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig, ax