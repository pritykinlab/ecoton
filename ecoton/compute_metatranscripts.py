import pandas as pd
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
            transcript_threshold_factor = 500
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
        x_coord = 'x_global_um'
        y_coord = 'y_global_um'

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

        filtered_df = filtered_df.copy()
        filtered_df[x_coord] = filtered_df['x_global_px'].astype(np.float32) * 0.18
        filtered_df[y_coord] = filtered_df['y_global_px'].astype(np.float32) * 0.18

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
    

import numpy as np
import pandas as pd

def build_binned_metatranscripts_one_gene(
    sub: pd.DataFrame,
    gene: str,
    bin_um: float = 8.0,
    min_points: int = 3,
    gene_name = "feature_name",
    x_col: str = "x_location",
    y_col: str = "y_location",
    # adaptive parameters
    adaptive_min_points: bool = True,
    global_bin_counts_series: bool = None,  # MultiIndex (bin_x,bin_y) -> global_n
    global_low_count_threshold: int = 50,
    low_count_min_points: int = 2,
):
    if len(sub) == 0:
        return pd.DataFrame(columns=[gene_name,"bin_x","bin_y","n","x_centroid","y_centroid"])

    x = sub[x_col].to_numpy(dtype=np.float32, copy=False)
    y = sub[y_col].to_numpy(dtype=np.float32, copy=False)

    bin_x = np.floor(x / bin_um).astype(np.int32)
    bin_y = np.floor(y / bin_um).astype(np.int32)

    tmp = pd.DataFrame({"bin_x": bin_x, "bin_y": bin_y, "x": x, "y": y})

    meta = (
        tmp.groupby(["bin_x", "bin_y"], sort=False)
           .agg(n=("bin_x", "size"), x_centroid=("x", "mean"), y_centroid=("y", "mean"))
           .reset_index()
    )

    # ----- adaptive filtering WITHOUT merge -----
    if adaptive_min_points:
        if global_bin_counts_series is None:
            raise ValueError("Need global_bin_counts_series when adaptive_min_points=True")

        # Fast lookup using MultiIndex
        idx = pd.MultiIndex.from_arrays([meta["bin_x"].values, meta["bin_y"].values])
        global_n = global_bin_counts_series.reindex(idx).to_numpy()

        required = np.where(
            global_n < global_low_count_threshold,
            low_count_min_points,
            min_points
        ).astype(np.int32)

        meta = meta[meta["n"].to_numpy() >= required].copy()
    else:
        meta = meta[meta["n"] >= min_points].copy()

    if len(meta) == 0:
        meta = meta.copy()
        meta[gene_name] = pd.Series([], dtype="category")
        return meta[[gene_name, "bin_x", "bin_y", "n", "x_centroid", "y_centroid"]]

    meta.insert(0, gene_name, gene)

    meta["bin_x"] = meta["bin_x"].astype(np.int32)
    meta["bin_y"] = meta["bin_y"].astype(np.int32)
    meta["n"] = meta["n"].astype(np.int32)
    meta["x_centroid"] = meta["x_centroid"].astype(np.float32)
    meta["y_centroid"] = meta["y_centroid"].astype(np.float32)

    return meta[[gene_name,"bin_x","bin_y","n","x_centroid","y_centroid"]]

def build_metatranscripts(
    df: pd.DataFrame,
    genes,
    bin_um: float = 8.0,
    min_points: int = 3,
    print_every: int = 200,
    gene_name = "feature_name",
    x_col: str = "x_location",
    y_col: str = "y_location",
    adaptive_min_points: bool = True,
    global_low_count_threshold: int = 50,
    low_count_min_points: int = 2,
):
    grouped = df.groupby(gene_name, sort=False)

    # ---- Build GLOBAL bin lookup ONCE ----
    x_all = df[x_col].to_numpy(dtype=np.float32, copy=False)
    y_all = df[y_col].to_numpy(dtype=np.float32, copy=False)

    bin_x_all = np.floor(x_all / bin_um).astype(np.int32)
    bin_y_all = np.floor(y_all / bin_um).astype(np.int32)

    global_bin_counts_series = (
        pd.DataFrame({"bin_x": bin_x_all, "bin_y": bin_y_all})
        .groupby(["bin_x","bin_y"], sort=False)
        .size()
    )
    # --------------------------------------

    metas = []
    total_meta = 0

    for i, gene in enumerate(genes, start=1):
        if gene not in grouped.groups:
            continue

        sub = grouped.get_group(gene)[[x_col, y_col]]

        meta = build_binned_metatranscripts_one_gene(
            sub=sub,
            gene=gene,
            bin_um=bin_um,
            min_points=min_points,
            gene_name=gene_name,
            x_col=x_col,
            y_col=y_col,
            adaptive_min_points=adaptive_min_points,
            global_bin_counts_series=global_bin_counts_series,
            global_low_count_threshold=global_low_count_threshold,
            low_count_min_points=low_count_min_points,
        )

        if len(meta):
            metas.append(meta)
            total_meta += len(meta)

        if i % print_every == 0:
            print(f"[{i}/{len(genes)}] genes | metatranscripts so far: {total_meta:,}")

    meta_df = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame(
        columns=[gene_name,"bin_x","bin_y","n","x_centroid","y_centroid"]
    )

    meta_df[gene_name] = meta_df[gene_name].astype("category")
    meta_df["bin_x"] = meta_df["bin_x"].astype(np.int32)
    meta_df["bin_y"] = meta_df["bin_y"].astype(np.int32)
    meta_df["n"] = meta_df["n"].astype(np.int32)
    meta_df["x_centroid"] = meta_df["x_centroid"].astype(np.float32)
    meta_df["y_centroid"] = meta_df["y_centroid"].astype(np.float32)
    meta_df["meta_id"] = meta_df.index.astype(np.int32)

    return meta_df

def ComputeMetatranscripts(mytranscripts,
                           transcript_threshold_factor=None,
                           min_transcripts=None,
                           nofilter=None,
                           mode='xenium_5k',
                           adaptive_min_points: bool = True,
                           global_low_count_threshold: int = 50,
                           low_count_min_points: int = 2,
                           bin_um: float = 8.0,
                           min_points: int = 3,
                           ):
    """Orchestrator: prepare transcripts then build metatranscripts.

    Returns the preparation object with an added 'metatranscripts' key.
    """
    prep = prepare_transcripts(mytranscripts,
                               transcript_threshold_factor=transcript_threshold_factor,
                               min_transcripts=min_transcripts,
                               nofilter=nofilter,
                               mode=mode)
    
    prep['metatranscripts'] = build_metatranscripts(prep['filtered_transcripts'], prep['genes'], 
                                            bin_um=bin_um, 
                                            min_points=min_points,
                                            gene_name=prep['gene_name'],
                                            x_col=prep['x_coord'],
                                            y_col=prep['y_coord'],
                                            adaptive_min_points=adaptive_min_points,
                                            global_low_count_threshold=global_low_count_threshold,
                                            low_count_min_points=low_count_min_points,
                                            )
    return prep