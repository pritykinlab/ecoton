"""
Demo pipeline showing how to run the ecoton analysis end-to-end on example data.

This script creates a small synthetic transcriptome, computes metatranscripts,
applies the analytic null, converts results into a graph, runs archetypal
analysis to get archetypes (modules), computes niche maps for archetypes,
and runs a small niche-statistics example. Finally it plots the colocalization
modules using the archetype labels.

Run:
    python scripts/run_demo_pipeline.py

Note: This is a demo with synthetic data for quick smoke-testing. For real
data, replace the synthetic-data block with a load of your transcripts
DataFrame (columns: feature_name, x_location, y_location, qv, codeword_category).
"""

import numpy as np
import pandas as pd

from ecoton import (
    ComputeMetatranscripts,
    analytic_null_metatranscripts,
    stats_df_to_igraph,
    ProcessColocalizationGraph,
    bin_transcripts,
    bins_from_niche_threshold,
    cells_in_selected_bins,
    plot_modules,
)

from ecoton.compute_niche_maps import create_niche_maps_by_archetype_all_at_once

from scipy.spatial import cKDTree as KDTree

def make_synthetic_transcripts(n_genes=8, points_per_gene=200, cluster_std=3.0, seed=0):
    """Make a synthetic transcripts DataFrame suitable for the demo."""
    from sklearn.datasets import make_blobs

    rng = np.random.RandomState(seed)
    centers = rng.uniform(-200, 200, size=(n_genes, 2))

    X, labels = make_blobs(n_samples=n_genes * points_per_gene,
                           centers=centers,
                           cluster_std=cluster_std,
                           random_state=seed)

    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    gene_for_point = [gene_names[l] for l in labels]

    df = pd.DataFrame({
        "feature_name": gene_for_point,
        "x_location": X[:, 0],
        "y_location": X[:, 1],
        # high quality and predesigned category so ComputeMetatranscripts keeps them
        "qv": 30.0,
        "codeword_category": "predesigned_gene",
    })
    return df


def build_metatranscripts_from_clusters(transcripts_df, clustering_results):
    """Turn ComputeMetatranscripts output into a metatranscripts DataFrame.

    clustering_results is the dict returned by ComputeMetatranscripts: gene -> pd.Series
    where the series maps transcript index -> cluster_label for clustered points.
    """
    rows = []
    meta_id = 0
    for gene, ser in clustering_results.items():
        if ser is None or len(ser) == 0:
            continue
        # ser index are original transcript indices; values are cluster labels
        df_gene = transcripts_df.loc[ser.index]
        df_gene = df_gene.assign(cluster=ser.values)
        grouped = df_gene.groupby('cluster')
        for cluster_label, g in grouped:
            x_cent = float(g['x_location'].mean())
            y_cent = float(g['y_location'].mean())
            size = int(g.shape[0])
            rows.append({
                'feature_name': gene,
                'cluster': int(cluster_label),
                'x_centroid': x_cent,
                'y_centroid': y_cent,
                'size': size,
                'meta_id': int(meta_id),
            })
            meta_id += 1

    meta_df = pd.DataFrame(rows)
    return meta_df


def build_edges_from_metatranscripts(meta_df, radius=15.0):
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


def main():
    print('Preparing synthetic transcripts...')
    transcripts = make_synthetic_transcripts(n_genes=8, points_per_gene=200, cluster_std=4.0)

    print('Computing metatranscripts (DBSCAN per gene) ...')
    clustering = ComputeMetatranscripts(transcripts, mode='xenium_5k', min_transcripts=10, batch_size=50)

    print('Building metatranscripts table from clusters...')
    metadf = build_metatranscripts_from_clusters(transcripts, clustering)
    print(f'Metatranscripts found: {len(metadf)}')

    print('Building meta-edge list via KDTree...')
    edges_df = build_edges_from_metatranscripts(metadf, radius=25.0)
    print(f'Edges found: {len(edges_df)}')

    print('Running analytic null on metatranscripts...')
    stats_df, globals_ = analytic_null_metatranscripts(metadf, edges_df, recompute_weight_from_sizes=True)
    print('Top gene pairs by Z:')
    print(stats_df.sort_values('Z', ascending=False).head())

    print('Converting stats to igraph...')
    G_ig = stats_df_to_igraph(stats_df, weight_col='Z')

    print('Processing colocalization graph (archetypes)...')
    pcs = ProcessColocalizationGraph(G_ig, flavor='archetypes', K=5)
    # expect keys: modules_Z, modules_H, Z_df, H_df
    Z_df = pcs.get('Z_df')
    H_df = pcs.get('H_df')
    print('Archetypes (Z_df columns):', list(Z_df.columns) if Z_df is not None else None)

    print('Computing niche maps for archetypes...')
    coords = transcripts[['x_location', 'y_location']].values.astype(np.float32)
    gene_labels = transcripts['feature_name'].values
    niche_maps = create_niche_maps_by_archetype_all_at_once(
        coords=coords,
        gene_labels=gene_labels,
        W=Z_df,
        bin_size=8.0,
        smoothing_radius=12.0,
        weight_threshold=0.1,
    )
    print('niche_maps shape:', niche_maps.shape)

    print('Binning transcripts to compute cells_by_bin...')
    out = bin_transcripts(transcripts, bin_size=8.0, x_col='x_location', y_col='y_location', gene_col='feature_name', cell_col='feature_name', return_matrix=False, return_matrix_split_assignment=False, return_cells=True)
    grid_meta = out['grid_meta']
    cells_by_bin = out['cells_by_bin']

    print('Selecting high-niche bins for archetype 0...')
    selected_bin_ids, mask, t = bins_from_niche_threshold(niche_maps, grid_meta={**grid_meta, 'height': niche_maps.shape[0], 'width': niche_maps.shape[1]}, k=0, threshold='p90')
    cells = cells_in_selected_bins(selected_bin_ids, cells_by_bin)
    print(f'Cells in selected bins (archetype 0, threshold p90): {len(cells)}')

    print('Plotting colocalization modules (using simple archetype labels)...')
    arche_labels = [f'Archetype {i}' for i in range(Z_df.shape[1])] if Z_df is not None else [f'Archetype {i}' for i in range(5)]
    ax = plot_modules(G_ig, arche_labels, myclusters=list(range(len(arche_labels))))
    # show the plot if running interactively
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
