"""
Analytic null models for metatranscript interactions.

This module provides functions for computing statistical significance
of gene-gene interactions in metatranscript data using analytic null models.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import igraph as ig


def analytic_null_metatranscripts(
    metadf: pd.DataFrame,
    edges_df: pd.DataFrame,
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
    edges_df : pd.DataFrame
        Edges between metatranscripts with source, target, weight, etc.
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
    stats_df : pd.DataFrame
        DataFrame with columns: g_a, g_b, O, E, PMI, Z, keep
    globals : dict
        Dictionary with global statistics: S, total_transcripts, gene_totals, etc.
    """

    # ---- 1) Global gene frequencies f_i from metatranscripts table ----
    # total transcripts per gene = sum of 'size' across meta clusters of that gene
    gene_totals = metadf.groupby(gene_col_meta, sort=False)[size_col].sum().astype(np.float64)
    total_transcripts = float(gene_totals.sum())
    f = (gene_totals / (total_transcripts + eps)).astype(np.float64)  # Series: gene -> freq

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

    globals_ = {
        "S": S,
        "total_transcripts": total_transcripts,
        "gene_totals": gene_totals,
        "gene_freq": f,
        "n_meta_edges_used": int(len(e)),
        "n_gene_pairs_observed": int(len(O)),
    }

    return O, globals_


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



