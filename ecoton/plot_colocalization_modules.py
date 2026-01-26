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
    module_labels,
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
    module_labels : list of str
        Labels for each cluster/module.
    myclusters : list of int, optional
        Cluster IDs to highlight. If None, highlights all clusters.
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
    # STEP 1. Ensure weights exist
    # ===============================
    if "weight" not in G_ig.es.attributes():
        print("No edge weights found — setting all weights to 1.0")
        G_ig.es["weight"] = [1.0] * G_ig.ecount()

    print(f"Graph has {G_ig.vcount()} nodes and {G_ig.ecount()} edges")

    # ===============================
    # STEP 2. Use full graph and all cluster labels
    # ===============================
    raw_membership = np.array(G_ig.vs["cluster"], dtype=object)
    membership = []

    for x in raw_membership:
        try:
            membership.append(int(x))
        except:
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

    if myclusters is None:
        myclusters = list(range(len(module_labels)))

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
        if cid >= len(module_labels):
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