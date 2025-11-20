def ComputeCellOverlaps(mytranscripts, modules, num_modules=36, min_samples=50):
    
    """
    Compute the overlap of gene modules within cells based on transcript data.
    
    Args:
    mytranscripts (pd.DataFrame): DataFrame containing transcript data with columns:
        - 'cell_id': Identifier for the cell
        - 'feature_name': Gene name
        - 'x_location', 'y_location': Spatial coordinates
        - 'qv': Quality value
        - 'codeword_category': Category of the codeword
    modules (list of list): List of gene modules, where each module is a list of gene names.
    num_modules (int): Number of modules to consider for clustering.
    Returns:
    pd.DataFrame: DataFrame with cells as rows and module proportions as columns.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    # Quality filter
    hq_transcripts = mytranscripts[mytranscripts['qv'] >= 20.0]
    hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

    # Step 3: Subset the dataframe to only those genes
    filtered_df = hq_transcripts

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    module_clusters = pd.Series(-1, index=hq_transcripts.index)

    # ---- Parameters ----
    eps = 10
    modules_to_use = list(range(num_modules))  # first 25 modules

    for i, mod_idx in enumerate(modules_to_use):
        print(i)
        gene_list = modules[mod_idx]
        
        # Filter to current module genes
        sub = hq_transcripts[hq_transcripts['feature_name'].isin(gene_list)].copy()

        from sklearn.neighbors import NearestNeighbors
        coords = sub[['x_location', 'y_location']] # the transcripts of the genes in the module
        nbrs = NearestNeighbors(radius=10, algorithm='kd_tree').fit(coords)
        n_neighbors = np.array([len(v)-1 for v in nbrs.radius_neighbors(coords)[1]])
        min_samples = np.percentile(n_neighbors, 80)
        min_samples_actual = int(max(min_samples, 50))

        # DBSCAN clustering
        coords = sub[['x_location', 'y_location']].values
        db = DBSCAN(eps=eps, min_samples=min_samples_actual).fit(coords)
        sub['cluster'] = db.labels_
        clustered = sub[sub['cluster'] != -1]
        if clustered.index.empty:
            continue
        module_clusters[clustered.index] = i

    hq_transcripts['module_clusters'] = module_clusters.astype('category')

    # Ensure categorical types for performance
    hq_transcripts['cell_id'] = hq_transcripts['cell_id'].astype('category')
    hq_transcripts['module_clusters'] = hq_transcripts['module_clusters'].astype('category')

    # --- Step 1: Total transcripts per cell (including unclustered) ---
    total_transcripts = (
        hq_transcripts
        .groupby('cell_id')
        .size()
        .reset_index(name='total_count')
    )

    # --- Step 2: Count transcripts per cell *per module*, excluding unclustered (-1) ---
    clustered_counts = (
        hq_transcripts[hq_transcripts['module_clusters'] != -1]
        .groupby(['cell_id', 'module_clusters'])
        .size()
        .reset_index(name='count')
    )

    # --- Step 3: Merge to get proportions ---
    clustered_counts = clustered_counts.merge(total_transcripts, on='cell_id', how='left')
    clustered_counts['proportion'] = clustered_counts['count'] / clustered_counts['total_count']

    # --- Step 4: Pivot to wide format ---
    cell_module_props = clustered_counts.pivot(
        index='cell_id', columns='module_clusters', values='proportion'
    ).fillna(0)

    # Optional: make column names clearer
    cell_module_props.columns = [f'module_{int(c)}' for c in cell_module_props.columns]

    # Reset index if you want cell_id as a column
    cell_module_props = cell_module_props.reset_index()

    print(cell_module_props.head())

    cell_module_props.index = cell_module_props['cell_id']
    cell_module_props = cell_module_props[1:]
    cell_module_props = cell_module_props.iloc[:, 2:]

    return cell_module_props