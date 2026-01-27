"""
Analytic null models for metatranscript interactions.

This module provides functions for computing statistical significance
of gene-gene interactions in metatranscript data using analytic null models.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
import igraph as ig
from typing import Optional

def build_edges_from_metatranscripts(meta_df, radius=10.0):
    coords = meta_df[['x_centroid', 'y_centroid']].values
    sizes = meta_df['size'].values
    ids = meta_df['meta_id'].values
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    if pairs.size == 0:
        return pd.DataFrame(columns=['source', 'target', 'weight', 'gene_source', 'gene_target'])
    src = pairs[:, 0]
    tgt = pairs[:, 1]
    weights = sizes[src] * sizes[tgt]
    edges = pd.DataFrame({
        'source': ids[src],
        'target': ids[tgt],
        'weight': weights,
        'gene_source': meta_df['feature_name'].values[src],
        'gene_target': meta_df['feature_name'].values[tgt],
    })
    return edges

def analytic_null_metatranscripts(
    metadf: pd.DataFrame,
    edges_df: Optional[pd.DataFrame] = None,
    *,
    gene_col_meta: str = "feature_name",
    meta_id_col: str = "meta_id",
    size_col: str = "size",
    src_col: str = "source",
    tgt_col: str = "target",
    gene_src_col: str = "gene_source",
    gene_tgt_col: str = "gene_target",
    weight_col: str = "weight",
    undirected: bool = True,
    drop_self_gene_edges: bool = False,
    eps: float = 1e-9,
    z_thresh: float = 1.96,
    require_positive_pmi: bool = True,
    recompute_weight_from_sizes: bool = False,
    return_igraph: bool = True,
    stats_df: bool = True,
    igraph_weight: str = "PMI",
    radius: float = 10.0,
    verbose: bool = False,
):
    """
    Implements the size-aware analytic null over metatranscripts.

    Observed gene-gene interaction mass:
        O_ij = sum_{meta-edge e} w_e over edges where (gene_source, gene_target) == (i, j)

    Null expectation:
        E_ij = f_i * f_j * S
    where:
        f_i = total transcripts of gene i / total transcripts overall
        S   = sum_{meta-edge e} (s_u * s_v) over ALL meta-edges (u,v)

    Parameters
    ----------
    metadf : pd.DataFrame
        Metatranscript dataframe with gene, meta_id, and size columns.
    edges_df : pd.DataFrame or None
        Edges between metatranscripts with source, target, weight, etc. If
        ``None`` the function will build edges by calling
        ``build_edges_from_metatranscripts(metadf)``. When building edges the
        parameter ``radius`` (see below) is passed through to control the
        neighborhood radius used by the KDTree.
    radius : float, default 10.0
        Radius (same units as `x_centroid`/`y_centroid`) used when constructing
        edges via `build_edges_from_metatranscripts` if `edges_df` is None.
    gene_col_meta : str, default "feature_name"
        Column name for gene names in metadf.
    meta_id_col : str, default "meta_id"
        Column name for metatranscript IDs in metadf.
    size_col : str, default "size"
        Column name for metatranscript sizes in metadf.
    src_col : str, default "source"
        Column name for source metatranscript IDs in edges_df.
    tgt_col : str, default "target"
        Column name for target metatranscript IDs in edges_df.
    gene_src_col : str, default "gene_source"
        Column name for source gene names in edges_df.
    gene_tgt_col : str, default "gene_target"
        Column name for target gene names in edges_df.
    weight_col : str, default "weight"
        Column name for edge weights in edges_df.
    undirected : bool, default True
        Whether to treat edges as undirected.
    drop_self_gene_edges : bool, default False
        Whether to remove edges between same gene.
    eps : float, default 1e-9
        Small value to avoid division by zero.
    z_thresh : float, default 1.96
        Z-score threshold for significance.
    require_positive_pmi : bool, default True
        Whether to require positive PMI for significance.
    recompute_weight_from_sizes : bool, default False
        Whether to recompute weights from sizes.

    Returns
    -------
    If ``return_igraph`` is False (default):

    stats_df : pd.DataFrame
        DataFrame with columns: g_a, g_b, O, E, PMI, Z, keep
    globals : dict
        Dictionary with global statistics: S, total_transcripts, gene_totals, etc.

    If ``return_igraph`` is True (default), the function returns a tuple
    ``(stats_df, globals, G)`` where ``G`` is an ``igraph.Graph`` built from
    the filtered significant edges. Set ``stats_df`` to ``False`` to exclude
    the DataFrame from the return tuple (only ``globals`` and ``G`` will be
    returned). The ``igraph_weight`` parameter controls which column from the
    stats table is used as the graph edge weight; valid options are ``'PMI'``
    or ``'Z'``. If ``verbose`` is True, concise progress messages will be
    printed to stdout.
    """

    # if edges not provided, build them from metadf
    if edges_df is None:
        if verbose:
            print("[analytic_null_metatranscripts] building edges from metadf using radius=", radius)
        edges_df = build_edges_from_metatranscripts(metadf, radius=radius)
        if verbose:
            print(f"[analytic_null_metatranscripts] built {len(edges_df)} edges")

    # ---- 1) Global gene frequencies f_i from metatranscripts table ----
    # total transcripts per gene = sum of 'size' across meta clusters of that gene
    gene_totals = metadf.groupby(gene_col_meta, sort=False)[size_col].sum().astype(np.float64)
    total_transcripts = float(gene_totals.sum())
    f = (gene_totals / (total_transcripts + eps)).astype(np.float64)  # Series: gene -> freq
    if verbose:
        print(f"[analytic_null_metatranscripts] total_transcripts={total_transcripts:.3f}, n_genes={len(gene_totals)}")

    # ---- 2) Attach sizes to each meta-edge (source/target) ----
    meta_sizes = metadf[[meta_id_col, size_col]].drop_duplicates(subset=[meta_id_col])
    meta_sizes = meta_sizes.set_index(meta_id_col)[size_col].astype(np.float64)

    e = edges_df.copy()

    # ensure gene labels exist (use edges_df columns if present; otherwise map from metadf)
    if gene_src_col not in e.columns or gene_tgt_col not in e.columns:
        meta_gene = metadf[[meta_id_col, gene_col_meta]].drop_duplicates(subset=[meta_id_col])
        meta_gene = meta_gene.set_index(meta_id_col)[gene_col_meta]
        e[gene_src_col] = e[src_col].map(meta_gene)
        e[gene_tgt_col] = e[tgt_col].map(meta_gene)

    # add sizes
    e["s_src"] = e[src_col].map(meta_sizes)
    e["s_tgt"] = e[tgt_col].map(meta_sizes)
    # drop edges where size or gene is missing
    e = e.dropna(subset=[gene_src_col, gene_tgt_col, "s_src", "s_tgt"])

    # choose observed per-edge contribution
    if recompute_weight_from_sizes or (weight_col not in e.columns):
        e["w_obs"] = (e["s_src"] * e["s_tgt"]).astype(np.float64)
    else:
        e["w_obs"] = e[weight_col].astype(np.float64)

    # ---- 3) Canonicalize to undirected gene pairs if desired ----
    if undirected:
        g1 = e[gene_src_col].astype(str).to_numpy()
        g2 = e[gene_tgt_col].astype(str).to_numpy()
        gmin = np.minimum(g1, g2)
        gmax = np.maximum(g1, g2)
        e["g_a"] = gmin
        e["g_b"] = gmax
    else:
        e["g_a"] = e[gene_src_col].astype(str)
        e["g_b"] = e[gene_tgt_col].astype(str)

    if drop_self_gene_edges:
        e = e[e["g_a"] != e["g_b"]]

    # ---- 4) Observed O_ij: sum observed contributions over meta-edges ----
    O = (
        e.groupby(["g_a", "g_b"], sort=False)["w_obs"]
         .sum()
         .astype(np.float64)
         .rename("O")
         .reset_index()
    )

    # ---- 5) Compute S = sum_{all meta-edges} s_u*s_v  (the mass under the null) ----
    # IMPORTANT: S should reflect the same notion of "edge instance" you used to build O.
    # If your observed weights already are s_u*s_v, then S should be sum of s_u*s_v across edges.
    S = float((e["s_src"] * e["s_tgt"]).sum())

    # ---- 6) Expected E_ij = f_i*f_j*S ----
    # map gene frequencies into O table
    O["f_a"] = O["g_a"].map(f).astype(np.float64)
    O["f_b"] = O["g_b"].map(f).astype(np.float64)

    # if a gene didn't appear in metadf totals (shouldn't happen), fill 0
    O[["f_a", "f_b"]] = O[["f_a", "f_b"]].fillna(0.0)

    O["E"] = (O["f_a"] * O["f_b"] * S).astype(np.float64)

    # ---- 7) PMI and Z-score ----
    O["PMI"] = np.log((O["O"] + eps) / (O["E"] + eps))
    O["Z"] = (O["O"] - O["E"]) / np.sqrt(O["E"] + eps)

    # ---- 8) Filtering ----
    keep = np.ones(len(O), dtype=bool)
    if require_positive_pmi:
        keep &= (O["PMI"].to_numpy() > 0)
    keep &= (O["Z"].to_numpy() > z_thresh)

    O["keep"] = keep
    n_keep = int(O["keep"].sum()) if len(O) > 0 else 0
    if verbose:
        print(f"[analytic_null_metatranscripts] filtering applied: kept {n_keep} / {len(O)} gene pairs")

    globals_ = {
        "S": S,
        "total_transcripts": total_transcripts,
        "gene_totals": gene_totals,
        "gene_freq": f,
        "n_meta_edges_used": int(len(e)),
        "n_gene_pairs_observed": int(len(O)),
    }

    # validate igraph_weight
    if igraph_weight not in ("PMI", "Z"):
        raise ValueError("igraph_weight must be 'PMI' or 'Z'")

    if return_igraph:
        if verbose:
            print(f"[analytic_null_metatranscripts] building igraph using weight={igraph_weight}")
        G = stats_df_to_igraph(O, weight_col=igraph_weight, keep_col="keep", gene_a_col="g_a", gene_b_col="g_b")
        if verbose:
            print(f"[analytic_null_metatranscripts] igraph: n_vertices={len(G.vs)}, n_edges={len(G.es)}")
        if stats_df:
            return O, globals_, G
        return globals_, G

    if stats_df:
        return O, globals_

    return globals_


def stats_df_to_igraph(
    stats_df: pd.DataFrame,
    *,
    weight_col: str = "PMI",     # or "Z"
    keep_col: str = "keep",
    gene_a_col: str = "g_a",
    gene_b_col: str = "g_b",
):
    """
    Convert analytic-null gene-gene stats DataFrame to an undirected igraph Graph.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame from analytic_null_metatranscripts with gene pairs and stats.
    weight_col : str, default "PMI"
        Column to use for edge weights in the graph.
    keep_col : str, default "keep"
        Column indicating which edges to include.
    gene_a_col : str, default "g_a"
        Column for first gene in pair.
    gene_b_col : str, default "g_b"
        Column for second gene in pair.

    Returns
    -------
    ig.Graph
        Undirected graph with genes as vertices and significant edges.
    """

    # ---- 1) Filter significant edges ----
    df = stats_df[stats_df[keep_col]].copy()
    if df.empty:
        raise ValueError("No edges passed the significance filter.")

    # ---- 2) Build vertex set ----
    genes = pd.Index(
        pd.unique(df[[gene_a_col, gene_b_col]].values.ravel())
    )
    gene_to_vid = {g: i for i, g in enumerate(genes)}

    # ---- 3) Build edge list ----
    edges = [
        (gene_to_vid[a], gene_to_vid[b])
        for a, b in zip(df[gene_a_col], df[gene_b_col])
    ]

    weights = df[weight_col].astype(float).tolist()

    # ---- 4) Create igraph ----
    G = ig.Graph(
        n=len(genes),
        edges=edges,
        directed=False
    )

    # ---- 5) Assign attributes ----
    G.vs["name"] = genes.tolist()
    G.es["weight"] = weights

    # Optional: store additional edge stats
    for col in ["O", "E", "PMI", "Z"]:
        if col in df.columns:
            G.es[col] = df[col].astype(float).tolist()

    return G



