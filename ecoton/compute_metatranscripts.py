
def ComputeMetatranscripts(mytranscripts, 
                           transcript_threshold_factor = 1000, 
                           min_transcripts=50, 
                           nofilter=False,
                           mode='xenium_5k',
                           batch_size=200):
    """
    Compute metatranscripts using DBSCAN clustering.

    Parameters:
    mytranscripts (pd.DataFrame): DataFrame containing transcript data with columns:
        - 'qv': Quality value
        - 'codeword_category': Category of the codeword
        - 'feature_name': Gene name
        - 'x_location': X coordinate
        - 'y_location': Y coordinate

    Returns:
    dict: A dictionary with gene names as keys and clustered transcript indices as values.
    """
    import pandas as pd
    from sklearn.cluster import DBSCAN
    from joblib import Parallel, delayed

    def is_real_gene(x):
        x = x.decode() if isinstance(x, bytes) else x
        if ("BLANK" in x) or ("NegControl" in x) or ("antisense" in x) or ("Blank" in x) or ("SystemControl" in x) or ("Negative" in x) or ("NegPrb" in x):
            return 'something_else'
        if "_" in x:
            return 'something_else'
        return 'predesigned_gene'
    
    gene_name = None 
    x_coord = None 
    y_coord = None

    if mode == 'xenium_5k':
        print(f"Mode: {mode}")
        gene_name = 'feature_name'
        x_coord = 'x_location'
        y_coord = 'y_location'
        # Quality filter
        hq_transcripts = mytranscripts[mytranscripts['qv'] >= 20.0]
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        # Step 1: Count transcripts per gene
        gene_counts = hq_transcripts['feature_name'].astype('category').value_counts()

        # Step 2: Get genes within desired range (>=50 and <100000)
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        # Step 3: Subset the dataframe to only those genes
        filtered_df = hq_transcripts[hq_transcripts['feature_name'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        # Final gene list
        genes = list(filtered_df['feature_name'].astype('category').unique())
        sub_transcripts = filtered_df

    if mode == 'xenium':
        print(f"Mode: {mode}")
        gene_name = 'feature_name'
        x_coord = 'x_location'
        y_coord = 'y_location'
        # Quality filter
        mytranscripts['codeword_category'] = mytranscripts['feature_name'].apply(is_real_gene)
        hq_transcripts = mytranscripts[mytranscripts['qv'] >= 20.0]
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        # Step 1: Count transcripts per gene
        gene_counts = hq_transcripts['feature_name'].astype('category').value_counts()

        # Step 2: Get genes within desired range (>=50 and <100000)
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        # Step 3: Subset the dataframe to only those genes
        filtered_df = hq_transcripts[hq_transcripts['feature_name'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        # Final gene list
        genes = list(filtered_df['feature_name'].astype('category').unique())
        sub_transcripts = filtered_df

    if mode == 'merfish':
        print(f"Mode: {mode}")
        gene_name = 'gene'
        x_coord = 'global_x'
        y_coord = 'global_y'
        mytranscripts['codeword_category'] = mytranscripts['gene'].apply(is_real_gene)
        
        # Quality filter
        hq_transcripts = mytranscripts

        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']
        filtered_df = hq_transcripts

        # Final gene list
        genes = list(filtered_df['gene'].astype('category').unique())
        sub_transcripts = filtered_df

    if mode == 'cosmx':
        print(f"Mode: {mode}")
        gene_name = 'target'
        x_coord = 'x_global_px'
        y_coord = 'y_global_px'
        mytranscripts['codeword_category'] = mytranscripts['target'].apply(is_real_gene)
        # Quality filter
        hq_transcripts = mytranscripts
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        # Step 1: Count transcripts per gene
        gene_counts = hq_transcripts['target'].astype('category').value_counts()

        # Step 2: Get genes within desired range (>=50 and <100000)
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        # Step 3: Subset the dataframe to only those genes
        filtered_df = hq_transcripts[hq_transcripts['target'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        # Final gene list
        genes = list(filtered_df['target'].astype('category').unique())
        sub_transcripts = filtered_df    

    if mode == 'cosmx_prefiltered':
        print(f"Mode: {mode}")
        gene_name = 'target'
        x_coord = 'x_global_px'
        y_coord = 'y_global_px'
        sub_transcripts = mytranscripts
        genes = list(sub_transcripts['target'].astype('category').unique())

    # ------------------------------
    # Parameters
    # ------------------------------
    eps = 5
    min_samples = 3
    n_jobs = -1       # number of parallel workers

    # ------------------------------
    # DBSCAN function for a batch of genes
    # ------------------------------
    def cluster_gene_batch(gene_batch):
        """Cluster multiple genes in one function call to reduce overhead."""
        batch_results = {}
        for gene in gene_batch:
            gene_data = sub_transcripts[sub_transcripts[gene_name] == gene]
            coords = gene_data[[x_coord, y_coord]]

            if coords.shape[0] == 0:
                batch_results[gene] = pd.Series(dtype=int)
                continue

            db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(coords)
            myseries = pd.Series(db.labels_, index=gene_data.index)
            batch_results[gene] = myseries[myseries != -1]  # keep only clustered points
        return batch_results

    # ------------------------------
    # Create batches of genes
    # ------------------------------
    gene_batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]

    print(f"Total genes: {len(genes)}")
    print(f"Total batches: {len(gene_batches)} (batch size = {batch_size})")

    # ------------------------------
    # Run clustering in parallel
    # ------------------------------
    results_batches = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(cluster_gene_batch)(batch) for batch in gene_batches
    )

    # Merge batch dictionaries
    results = {}
    for batch_dict in results_batches:
        results.update(batch_dict)

    return results