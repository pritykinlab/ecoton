import pandas as pd
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
import numpy as np

def is_real_gene(x):
    x = x.decode() if isinstance(x, bytes) else x
    if ("BLANK" in x) or ("NegControl" in x) or ("antisense" in x) or ("Blank" in x) or ("SystemControl" in x) or ("Negative" in x) or ("NegPrb" in x):
        return 'something_else'
    if "_" in x:
        return 'something_else'
    return 'predesigned_gene'

def bytes_to_str(df):
    for c in df.columns:
        if df[c].dtype == object:
            if df[c].apply(lambda x: isinstance(x, (bytes, bytearray))).any():
                df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
    return df


def prepare_transcripts(mytranscripts,
                        mode='xenium_5k',
                        transcript_threshold_factor=None,
                        min_transcripts=None,
                        nofilter=False,
                        ):
    """Preprocess transcripts and return a preparation object (dict).

    Returned dict contains keys: 'params', 'filtered_transcripts', 'genes',
    'gene_name', 'x_coord', 'y_coord'.
    """
    gene_name = None
    x_coord = None
    y_coord = None

    # Work on a copy where we add any derived columns
    df = mytranscripts

    if mode == 'xenium_5k':
        print(f"Mode: {mode}")
        gene_name = 'feature_name'
        x_coord = 'x_location'
        y_coord = 'y_location'

        if transcript_threshold_factor is None:
            transcript_threshold_factor = 1000
        if min_transcripts is None:
            min_transcripts = 50
        if nofilter is None:
            nofilter = False

        # Quality filter
        hq_transcripts = df[df['qv'] >= 20.0]
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        gene_counts = hq_transcripts['feature_name'].astype('category').value_counts()
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        filtered_df = hq_transcripts[hq_transcripts['feature_name'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        genes = list(filtered_df['feature_name'].astype('category').unique())
        sub_transcripts = filtered_df

    elif mode == 'xenium':
        print(f"Mode: {mode}")
        gene_name = 'feature_name'
        x_coord = 'x_location'
        y_coord = 'y_location'

        if transcript_threshold_factor is None:
            transcript_threshold_factor = 0.5 # no max filtering
        if min_transcripts is None:
            min_transcripts = 25
        if nofilter is None:
            nofilter = False

        df = bytes_to_str(df)
        df['codeword_category'] = df['feature_name'].apply(is_real_gene)
        hq_transcripts = df[df['qv'] >= 20.0]
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        gene_counts = hq_transcripts['feature_name'].astype('category').value_counts()
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        filtered_df = hq_transcripts[hq_transcripts['feature_name'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        genes = list(filtered_df['feature_name'].astype('category').unique())
        sub_transcripts = filtered_df

    elif mode == 'merfish':
        print(f"Mode: {mode}")
        gene_name = 'gene'
        x_coord = 'global_x'
        y_coord = 'global_y'

        if transcript_threshold_factor is None:
            transcript_threshold_factor = 0.5 # no max filtering
        if min_transcripts is None:
            min_transcripts = 25
        if nofilter is None:
            nofilter = False

        df['codeword_category'] = df['gene'].apply(is_real_gene)
        hq_transcripts = df
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']
        filtered_df = hq_transcripts
        genes = list(filtered_df['gene'].astype('category').unique())
        sub_transcripts = filtered_df

    elif mode == 'cosmx':
        print(f"Mode: {mode}")
        gene_name = 'target'
        x_coord = 'x_global_px'
        y_coord = 'y_global_px'

        if transcript_threshold_factor is None:
            transcript_threshold_factor = 0.5 # no max filtering
        if min_transcripts is None:
            min_transcripts = 25
        if nofilter is None:
            nofilter = False

        df['codeword_category'] = df['target'].apply(is_real_gene)
        hq_transcripts = df
        hq_transcripts = hq_transcripts[hq_transcripts['codeword_category'] == 'predesigned_gene']

        num_transcripts = hq_transcripts.shape[0]
        max_transcripts = num_transcripts / transcript_threshold_factor

        gene_counts = hq_transcripts['target'].astype('category').value_counts()
        valid_genes = gene_counts[(gene_counts >= min_transcripts) & (gene_counts < max_transcripts)].index
        if nofilter:
            valid_genes = gene_counts.index

        filtered_df = hq_transcripts[hq_transcripts['target'].isin(valid_genes)]
        if nofilter:
            filtered_df = hq_transcripts

        genes = list(filtered_df['target'].astype('category').unique())
        sub_transcripts = filtered_df

    elif mode == 'cosmx_prefiltered':
        print(f"Mode: {mode}")
        gene_name = 'target'
        x_coord = 'x_global_px'
        y_coord = 'y_global_px'
        sub_transcripts = df
        genes = list(sub_transcripts['target'].astype('category').unique())

    else:
        raise ValueError(f"Unknown mode: {mode}")

    prep = {
        'params': {
            'mode': mode,
            'transcript_threshold_factor': transcript_threshold_factor,
            'min_transcripts': min_transcripts,
            'nofilter': nofilter
        },
        'gene_name': gene_name,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'filtered_transcripts': sub_transcripts,
        'genes': genes
    }

    return prep


def build_metatranscripts(prep_obj, eps=5, min_samples=3, batch_size=200, n_jobs=-1):
    """Run DBSCAN-based metatranscript building using the prepared object.

    This function adds a 'metatranscripts' entry to `prep_obj` and returns it.
    """
    sub_transcripts = prep_obj['filtered_transcripts']
    genes = prep_obj['genes']
    gene_name = prep_obj['gene_name']
    x_coord = prep_obj['x_coord']
    y_coord = prep_obj['y_coord']

    def cluster_gene_batch(gene_batch):
        batch_results = {}
        for gene in gene_batch:
            gene_data = sub_transcripts[sub_transcripts[gene_name] == gene]
            coords = gene_data[[x_coord, y_coord]]

            if coords.shape[0] == 0:
                batch_results[gene] = pd.Series(dtype=int)
                continue

            db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(coords)
            myseries = pd.Series(db.labels_, index=gene_data.index)
            batch_results[gene] = myseries[myseries != -1]
        return batch_results

    gene_batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]

    print(f"Total genes: {len(genes)}")
    print(f"Total batches: {len(gene_batches)} (batch size = {batch_size})")

    results_batches = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(cluster_gene_batch)(batch) for batch in gene_batches
    )

    results = {}
    for batch_dict in results_batches:
        results.update(batch_dict)

    prep_obj['metatranscripts'] = results

    # finalize clusters into metatranscript summary and clustered transcripts
    prep_obj = finalize_metatranscripts(prep_obj)
    return prep_obj


def ComputeMetatranscripts_old(mytranscripts,
                               transcript_threshold_factor=1000,
                               min_transcripts=50,
                               nofilter=False,
                               mode='xenium_5k',
                               batch_size=200):
    """Legacy wrapper that preserves original behavior (kept for now).

    This function calls the new `prepare_transcripts` and `build_metatranscripts`
    but returns only the metatranscript results to emulate the previous API.
    """
    prep = prepare_transcripts(mytranscripts,
                               transcript_threshold_factor=transcript_threshold_factor,
                               min_transcripts=min_transcripts,
                               nofilter=nofilter,
                               mode=mode)
    prep = build_metatranscripts(prep, batch_size=batch_size)
    return prep['metatranscripts']


def ComputeMetatranscripts(mytranscripts,
                           transcript_threshold_factor=None,
                           min_transcripts=None,
                           nofilter=None,
                           mode='xenium_5k',
                           batch_size=200):
    """Orchestrator: prepare transcripts then build metatranscripts.

    Returns the preparation object with an added 'metatranscripts' key.
    """
    prep = prepare_transcripts(mytranscripts,
                               transcript_threshold_factor=transcript_threshold_factor,
                               min_transcripts=min_transcripts,
                               nofilter=nofilter,
                               mode=mode)
    prep = build_metatranscripts(prep, batch_size=batch_size)
    return prep


def finalize_metatranscripts(prep_obj):
    """Take `prep_obj` with 'metatranscripts' and 'filtered_transcripts',
    compute cluster assignments, build a metatranscript summary DataFrame,
    and attach results back to prep_obj.
    """
    results = prep_obj.get('metatranscripts', {})
    filtered_df = prep_obj['filtered_transcripts']
    gene_name = prep_obj['gene_name']
    x_coord = prep_obj['x_coord']
    y_coord = prep_obj['y_coord']

    clusters = pd.Series(-1, index=filtered_df.index)
    for s in results.values():
        clusters.loc[s.index] = s.values

    filtered_df = filtered_df.copy()
    filtered_df['cluster'] = clusters

    clustered_df = filtered_df[filtered_df['cluster'] != -1]

    metatranscripts = (
        clustered_df
        .groupby([gene_name, 'cluster'], as_index=False)
        .agg(
            x_centroid=(x_coord, 'mean'),
            y_centroid=(y_coord, 'mean'),
            size=(x_coord, 'size')
        )
    )

    metatranscripts['meta_id'] = np.arange(len(metatranscripts))

    prep_obj['clustered_transcripts'] = clustered_df
    prep_obj['metatranscripts_df'] = metatranscripts
    return prep_obj