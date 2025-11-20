def ComputeColocalizationGraph(permutations):
    # Load permutation results
    # permutations is a dictionary of permutations from different slices
    """
    Combine permutation results from multiple slices and compute a co-localization graph.
    Args:
    permutations (dict): Dictionary where keys are slice names and values are DataFrames with permutation results.
        Each DataFrame should contain columns: ['gene_i', 'gene_j', 'z', 'observed_weight'].
    Returns:
    G_ig (igraph.Graph): The co-localization graph with genes as nodes and edges weighted by observed co-localization.
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import norm
    from functools import reduce

    # Suppose you have 10 slice result DataFrames in a dict
    # key = slice name, value = DataFrame
    # Example:
    # results = {'89_region_1': df1, '89_region_2': df2, ...}

    # 1. Reduce into one combined dataframe
    results = permutations
    dfs = []
    for slice_name, df in results.items():
        temp = df[['gene_i', 'gene_j', 'z', 'observed_weight']].copy()
        temp = temp.rename(columns={'z': f'z_{slice_name}'})
        dfs.append(temp) 

    # Outer merge on (gene_i, gene_j)
    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['gene_i', 'gene_j'], how='outer'), dfs)

    # Fill missing z with 0
    combined_df = combined_df.fillna(0)

    # 2. Calculate Stouffer combined Z
    z_cols = [col for col in combined_df.columns if col.startswith('z_')]
    k = len(z_cols)

    combined_df['z_combined'] = combined_df[z_cols].sum(axis=1) / np.sqrt(k)

    # --- One-sided p-value (enrichment only) ---
    combined_df['p_one_sided'] = 1 - norm.cdf(combined_df['z_combined'])

    # 4. FDR correction
    from statsmodels.stats.multitest import multipletests
    combined_df['fdr'] = multipletests(combined_df['p_one_sided'], method='fdr_bh')[1]

    # 5. Mark significant results
    combined_df['significant'] = combined_df['fdr'] < 0.05

    # Optional: sort by combined z
    combined_df = combined_df.sort_values('z_combined', ascending=False)

    combined_df.head() 
    filtered_df = combined_df
    all_df = filtered_df[filtered_df['significant'] == True] 

    final_network = all_df

    import igraph as ig

    # From final_network DataFrame
    # final_network has: ['gene1', 'gene2', 'observed_weight']

    # Combine unique gene names
    all_genes = pd.Index(
        pd.concat([final_network['gene_i'], final_network['gene_j']]).unique()
    )

    # Map gene names to indices
    gene_to_id = {gene: i for i, gene in enumerate(all_genes)}

    # Build edge list as tuples of integer IDs
    edges = [(gene_to_id[a], gene_to_id[b]) for a, b in zip(final_network['gene_i'], final_network['gene_j'])]
    weights = final_network['observed_weight'].values

    # Build igraph graph
    G_ig = ig.Graph(edges=edges, directed=False)
    G_ig.es['weight'] = weights
    G_ig.vs['name'] = all_genes.tolist()  # <-- Add gene names explicitly

    return G_ig



