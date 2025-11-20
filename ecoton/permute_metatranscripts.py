def PermuteMetatranscripts(mytranscripts, 
                           results, 
                           transcript_threshold_factor=1000, 
                           min_transcripts=50,
                           n_permutations=1000):
    
    """
    Given the clustered transcripts (results from compute_metatranscripts.py),
    compute a gene-gene co-localization matrix and assess significance via permutation testing.
    Parameters:
    mytranscripts (pd.DataFrame): DataFrame containing transcript data with columns:
        - 'qv': Quality value
        - 'codeword_category': Category of the codeword
        - 'feature_name': Gene name
        - 'x_location': X coordinate
        - 'y_location': Y coordinate
    results (dict): Output from ComputeMetatranscripts function.
    transcript_threshold_factor (float): Factor to determine max transcripts per gene.
    min_transcripts (int): Minimum transcripts per gene to consider.
    n_permutations (int): Number of permutations for significance testing.
    Returns:
    pd.DataFrame: DataFrame containing gene-gene co-localization statistics and significance.
    """

    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests


    # Quality filter
    hq_transcripts = mytranscripts[mytranscripts['qv'] >= 20.0]
    hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

    num_transcripts = hq_transcripts.shape[0]
    max_transcripts = num_transcripts / transcript_threshold_factor

    # Step 1: Count transcripts per gene
    gene_counts = hq_transcripts['feature_name'].astype('category').value_counts()

    # Step 2: Get genes within desired range (>=50 and <100000)
    valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index

    # Step 3: Subset the dataframe to only those genes
    filtered_df = hq_transcripts[hq_transcripts['feature_name'].isin(valid_genes)]

    # Final gene list
    genes = list(filtered_df['feature_name'].astype('category').unique())

    results # the results from compute_metatranscripts.py

    clusters = pd.Series(-1, index= filtered_df.index)

    for results in results.values():
        clusters[results.index] = results

    filtered_df['cluster'] = clusters

    clustered_df = filtered_df[filtered_df['cluster'] != -1]

    metatranscripts = (
        clustered_df
        .groupby(['feature_name', 'cluster'], as_index=False)
        .agg(
            x_centroid=('x_location', 'mean'),
            y_centroid=('y_location', 'mean'),
            size=('x_location', 'size')   # cluster size = number of transcripts
        )
    )

    # Add an ID for each metatranscript
    metatranscripts['meta_id'] = np.arange(len(metatranscripts))

    meta_df = metatranscripts

    meta_df.to_csv('metatranscripts_debug.csv')

    # Load your dataframe
    # meta_df has: feature_name, cluster, x_centroid, y_centroid, size, meta_id

    coords = meta_df[['x_centroid', 'y_centroid']].values
    sizes = meta_df['size'].values
    genes = meta_df['feature_name'].values
    meta_ids = meta_df['meta_id'].values

    # Build KDTree
    tree = cKDTree(coords)

    # Choose a radius (distance threshold)
    radius = 10.0  # adjust based on tissue scale, units should match your coordinates

    # Query all pairs of nodes within radius
    pairs = tree.query_pairs(r=radius, output_type='ndarray')  
    # shape: (n_edges, 2), each row is [i, j] where i < j

    # Extract size and gene info for edge endpoints
    edge_sources = pairs[:, 0]
    edge_targets = pairs[:, 1]

    edge_weights = sizes[edge_sources] * sizes[edge_targets]

    edges_df = pd.DataFrame({
        'source': meta_ids[edge_sources],
        'target': meta_ids[edge_targets],
        'weight': edge_weights,
        'gene_source': genes[edge_sources],
        'gene_target': genes[edge_targets]
    })

    print(edges_df.head())

    all_genes = pd.Index(pd.concat([edges_df['gene_source'], edges_df['gene_target']]).unique())
    gene_to_id = pd.Series(np.arange(len(all_genes)), index=all_genes)

    # Convert gene names to integer codes
    g1 = edges_df['gene_source'].map(gene_to_id).values
    g2 = edges_df['gene_target'].map(gene_to_id).values
    weights = edges_df['weight'].values

    import numpy as np

    # Pre-factorize gene names into ints
    gene_codes, unique_genes = pd.factorize(genes)
    source_idx = pairs[:, 0]
    target_idx = pairs[:, 1]

    # Integer labels for source and target endpoints
    g1 = gene_codes[source_idx]
    g2 = gene_codes[target_idx]

    # Symmetrize: always (min, max) for undirected edges
    gmin = np.minimum(g1, g2)
    gmax = np.maximum(g1, g2)

    # Edge weights
    edge_weights = sizes[source_idx] * sizes[target_idx]

    # Precompute observed counts
    pair_ids = gmin * len(unique_genes) + gmax  # unique id for each gene pair
    obs_counts = np.bincount(pair_ids, weights=edge_weights, minlength=len(unique_genes)**2)

    # We only care about nonzero pairs
    nonzero_pairs = np.flatnonzero(obs_counts)
    obs_values = obs_counts[nonzero_pairs]
    
    rng = np.random.default_rng(42)

    # Store null counts for each observed pair only
    null_matrix = np.zeros((len(nonzero_pairs), n_permutations), dtype=np.float32)

    for p in range(n_permutations):
        # Shuffle gene labels
        permuted = rng.permutation(gene_codes)
        g1p = permuted[source_idx]
        g2p = permuted[target_idx]
        
        gminp = np.minimum(g1p, g2p)
        gmaxp = np.maximum(g1p, g2p)
        perm_ids = gminp * len(unique_genes) + gmaxp
        
        # Count for all edges in one go
        perm_counts = np.bincount(perm_ids, weights=edge_weights, minlength=len(unique_genes)**2)
        
        # Store only counts for nonzero observed pairs
        null_matrix[:, p] = perm_counts[nonzero_pairs]
        if (p+1) % 100 == 0:
            print(f"Permutation {p+1}/{n_permutations} done")

    mu = null_matrix.mean(axis=1)
    sigma = null_matrix.std(axis=1, ddof=1)
    sigma = np.maximum(sigma, 1e-12)  # guard

    z = (obs_values - mu) / sigma
    p_right = norm.sf(z)                # one-sided (more-than-null co-localization)
    p_two_sided = 2 * norm.sf(np.abs(z))

    G = len(unique_genes) 

    # --------- Decode (i, j) from pair ids and assemble a DataFrame ---------
    i_idx, j_idx = np.divmod(nonzero_pairs, G)  # since id = i*G + j with i<=j

    results_df = pd.DataFrame({
        "gene_i": unique_genes[i_idx],
        "gene_j": unique_genes[j_idx],
        "observed_weight": obs_values,
        "null_mean": mu,
        "null_std": sigma,
        "z": z,
        "p_one_sided": p_right,
        "p_two_sided": p_two_sided,
    })

    # Use one-sided or two-sided p-values depending on your hypothesis
    pvals = results_df["p_one_sided"].values  # or "p_two_sided"

    # Run Benjamini-Hochberg correction
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Add results back to the dataframe
    results_df["fdr"] = pvals_fdr
    results_df["significant"] = reject

    # Sort by adjusted p-value
    results_df = results_df.sort_values("fdr").reset_index(drop=True)

    print(results_df.head())

    return results_df