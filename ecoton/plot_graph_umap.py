"""
Graph UMAP plotting helpers.

Split into:
- compute_graph_umap: computes and returns the embedding (and a few helpful extras)
- plot_graph_umap: plots a provided embedding (or computes one if not provided)
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from adjustText import adjust_text
import textwrap


def compute_graph_umap(
    G_ig,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
):
    """
    Compute a UMAP embedding derived from the graph's weighted adjacency.

    Parameters
    ----------
    G_ig : igraph.Graph
        Graph with 'weight' edge attribute and optional 'cluster' vertex attribute.
    n_neighbors : int
    min_dist : float
    random_state : int

    Returns
    -------
    embedding : np.ndarray, shape (n_vertices, 2)
        2D UMAP embedding.
    membership : np.ndarray, shape (n_vertices,)
        Integer cluster labels per vertex (falls back to -1 if missing/non-int).
    """
    if "weight" not in G_ig.es.attributes():
        G_ig.es["weight"] = [1.0] * G_ig.ecount()

    raw_membership = np.array(G_ig.vs.get("cluster", [-1] * G_ig.vcount()), dtype=object)
    membership = []
    for x in raw_membership:
        try:
            membership.append(int(x))
        except Exception:
            membership.append(-1)
    membership = np.array(membership, dtype=int)

    A_sub = np.array(G_ig.get_adjacency(attribute="weight").data, dtype=float)
    max_weight = A_sub.max()
    distance_matrix = max_weight - A_sub

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    embedding = umap_model.fit_transform(distance_matrix)

    return embedding, membership


def plot_graph_umap(
    G_ig,
    module_labels,
    embedding=None,
    membership=None,
    myclusters=None,
    title="Graph UMAP",
    figsize=(4, 4),
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    label_font_size=8,
    wrap_width=18,
    ax=None,
):
    """
    Plot a UMAP embedding derived from the graph's weighted adjacency.

    Parameters
    ----------
    G_ig : igraph.Graph
        Graph with 'weight' edge attribute and optional 'cluster' vertex attribute.
    module_labels : list[str]
        Labels for clusters/modules.
    embedding : np.ndarray, optional
        Precomputed embedding of shape (n_vertices, 2). If None, it is computed.
    membership : np.ndarray, optional
        Cluster labels per vertex of shape (n_vertices,). If None, it is derived
        (and computed along with embedding if embedding is None).
    myclusters : list[int], optional
        Cluster IDs to highlight; if None, all clusters are highlighted.
    Other parameters mirror those in the original `plot_graph_umap`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # If not provided, compute embedding (and membership)
    if embedding is None or membership is None:
        embedding, membership = compute_graph_umap(
            G_ig,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    unique_clusters = sorted(np.unique(membership))
    palette = sns.husl_palette(n_colors=len(unique_clusters))
    color_map = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    df = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "cluster": membership})

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.axis("off")
    gray = (0.75, 0.75, 0.75)
    is_target = np.isin(membership, myclusters)

    ax.scatter(
        df.loc[~is_target, "x"],
        df.loc[~is_target, "y"],
        c=[gray] * (~is_target).sum(),
        s=8,
        alpha=0.8,
        zorder=1,
    )
    ax.scatter(
        df.loc[is_target, "x"],
        df.loc[is_target, "y"],
        c=[color_map[c] for c in membership[is_target]],
        s=8,
        alpha=0.9,
        zorder=2,
    )

    texts, xs, ys = [], [], []
    for cid, (lx, ly) in label_positions.items():
        if cid >= len(module_labels):
            continue
        label = f"{cid}. {module_labels[cid]}"
        wrapped = "\n".join(textwrap.wrap(label, wrap_width))
        t = ax.text(
            lx, ly, wrapped,
            fontsize=label_font_size,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=10,
        )
        texts.append(t); xs.append(lx); ys.append(ly)

    adjust_text(
        texts,
        x=xs,
        y=ys,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=3, shrinkB=4),
        force_points=0.15,
        force_text=0.4,
        expand_points=(1.05, 1.05),
        expand_text=(1.08, 1.08),
        only_move={"text": "xy"},
    )

    ax.set_title(title)
    plt.tight_layout()
    return ax