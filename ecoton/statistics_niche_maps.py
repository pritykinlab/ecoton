"""
Statistics and utilities for niche maps and transcript binning.

This module provides functions for binning spatial transcript data,
computing niche maps, and analyzing cell distributions in selected bins.
"""

import numpy as np
import pandas as pd
from scipy import sparse

def _compute_bin_id_and_grid_meta(df, bin_size, x_col, y_col, verbose=False):
    x = df[x_col].to_numpy(np.float32, copy=False)
    y = df[y_col].to_numpy(np.float32, copy=False)

    x_min = float(np.nanmin(x)); y_min = float(np.nanmin(y))
    x_max = float(np.nanmax(x)); y_max = float(np.nanmax(y))

    width  = int(np.ceil((x_max - x_min) / bin_size)) + 1
    height = int(np.ceil((y_max - y_min) / bin_size)) + 1
    P = width * height

    gx = ((x - x_min) / bin_size).astype(np.int32)
    gy = ((y - y_min) / bin_size).astype(np.int32)
    gx = np.clip(gx, 0, width - 1)
    gy = np.clip(gy, 0, height - 1)

    bin_id = gy.astype(np.int64) * width + gx.astype(np.int64)

    if verbose:
        print("[binning] computed bin_id")
        print(f"  bin_size = {bin_size}")
        print(f"  x range  = [{x_min:.3f}, {x_max:.3f}]")
        print(f"  y range  = [{y_min:.3f}, {y_max:.3f}]")
        print(f"  grid     = width={width}, height={height}, P={P}")
        print(f"  n_transcripts = {bin_id.size}")

    grid_meta = {"x_min": x_min, "y_min": y_min, "width": width, "height": height, "P": P}
    return bin_id, grid_meta


def _parse_cell_ids(
    df,
    cell_col,
    unassigned_tokens=("UNASSIGNED",),
    keep_original_strings=True,
    verbose=False,
):
    cell = df[cell_col].to_numpy(copy=False)

    if np.issubdtype(cell.dtype, np.number):
        cell_num = cell.astype(np.float64, copy=False)
        unassigned = np.isnan(cell_num) | (cell_num == -1)
        assigned = ~unassigned
        cell_assigned_str = cell_num[assigned].astype(np.int64).astype(str)

        if verbose:
            print("[cells] detected numeric cell ids")
            print(f"  n_assigned   = {assigned.sum()}")
            print(f"  n_unassigned = {unassigned.sum()}")

    else:
        s = pd.Series(cell, copy=False).astype("string")
        s_stripped = s.str.strip()
        s_norm = s_stripped.str.upper()

        unassigned = s_norm.isna() | (s_norm == "") | (s_norm == "-1")
        for tok in unassigned_tokens:
            unassigned |= (s_norm == str(tok).strip().upper())

        assigned = ~unassigned

        if keep_original_strings:
            cell_assigned_str = s_stripped[assigned].astype(str).to_numpy()
        else:
            cell_assigned_str = s_norm[assigned].astype(str).to_numpy()

        if verbose:
            print("[cells] detected string cell ids")
            print(f"  n_assigned   = {assigned.sum()}")
            print(f"  n_unassigned = {unassigned.sum()}")
            print(f"  unassigned_tokens = {unassigned_tokens}")

    if isinstance(assigned, pd.Series):
        assigned = assigned.to_numpy()
    if isinstance(unassigned, pd.Series):
        unassigned = unassigned.to_numpy()

    return assigned, unassigned, np.asarray(cell_assigned_str)


def bin_transcripts(
    df,
    bin_size=10.0,
    x_col="x_location",
    y_col="y_location",
    gene_col="feature_name",
    cell_col="cell_id",
    dtype=np.int32,

    keep_empty_bins=False,
    return_matrix=True,
    return_matrix_split_assignment=False,

    unassigned_tokens=("UNASSIGNED",),
    keep_original_cell_strings=True,

    return_cells=True,
    verbose=True,
):
    """
    Bin spatial transcript data into a grid and compute matrices and cell mappings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transcript data with columns for x, y, gene, and cell.
    bin_size : float, default 10.0
        Size of each bin in spatial units.
    x_col : str, default "x_location"
        Column name for x coordinates.
    y_col : str, default "y_location"
        Column name for y coordinates.
    gene_col : str, default "feature_name"
        Column name for gene names.
    cell_col : str, default "cell_id"
        Column name for cell IDs.
    dtype : np.dtype, default np.int32
        Data type for count matrices.
    keep_empty_bins : bool, default False
        Whether to include bins with no transcripts in the output matrices.
    return_matrix : bool, default True
        Whether to return the classic bin x gene matrix.
    return_matrix_split_assignment : bool, default False
        Whether to return the split matrix with assigned/unassigned blocks.
    unassigned_tokens : tuple of str, default ("UNASSIGNED",)
        Tokens indicating unassigned transcripts.
    keep_original_cell_strings : bool, default True
        Whether to preserve original cell ID string formatting.
    return_cells : bool, default True
        Whether to return cell mappings and counts.
    verbose : bool, default True
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'grid_meta': dict with grid metadata (x_min, y_min, width, height)
        - 'X': sparse matrix (bins x genes) if return_matrix=True
        - 'bin_index': array of bin IDs if return_matrix=True
        - 'gene_names': array of gene names if return_matrix=True
        - 'X_split': sparse matrix (bins x 2*genes) if return_matrix_split_assignment=True
        - 'bin_index_split': array of bin IDs for split matrix
        - 'gene_names_split': array of gene names for split matrix
        - 'col_names_split': pd.Index of column names for split matrix
        - 'cells_by_bin': dict mapping bin IDs to sets of cell IDs if return_cells=True
        - 'bin_counts': pd.DataFrame with transcript counts per bin if return_cells=True
    """
    if verbose:
        print("============================================================")
        print("[bin_transcripts] starting")
        print(f"  df shape = {df.shape}")
        print(f"  return_matrix={return_matrix}")
        print(f"  return_matrix_split_assignment={return_matrix_split_assignment}")
        print(f"  return_cells={return_cells}")
        print("============================================================")

    bin_id, grid_meta = _compute_bin_id_and_grid_meta(df, bin_size, x_col, y_col, verbose=verbose)
    P = grid_meta["P"]

    out = {"grid_meta": {k: grid_meta[k] for k in ("x_min", "y_min", "width", "height")}}

    need_cells = return_cells or return_matrix_split_assignment
    if need_cells:
        assigned_mask, unassigned_mask, cell_assigned_str = _parse_cell_ids(
            df,
            cell_col=cell_col,
            unassigned_tokens=unassigned_tokens,
            keep_original_strings=keep_original_cell_strings,
            verbose=verbose,
        )

    # A) Classic matrix
    if return_matrix:
        if verbose:
            print("[matrix] building classic bin x gene sparse matrix...")

        genes = df[gene_col].to_numpy(copy=False)
        gene_codes, gene_names = pd.factorize(genes, sort=True)
        ok = gene_codes >= 0

        n_na_gene = int((~ok).sum())
        if verbose and n_na_gene > 0:
            print(f"[matrix] dropped {n_na_gene} transcripts with NA gene")

        b = bin_id[ok]
        g = gene_codes[ok].astype(np.int64, copy=False)

        G = len(gene_names)
        data = np.ones(b.shape[0], dtype=dtype)

        if keep_empty_bins:
            row = b
            n_rows = P
            bin_index = np.arange(P, dtype=np.int64)
            if verbose:
                print(f"[matrix] keep_empty_bins=True -> n_rows={n_rows} (full grid)")
        else:
            bin_index, row = np.unique(b, return_inverse=True)
            n_rows = bin_index.size
            if verbose:
                print(f"[matrix] keep_empty_bins=False -> n_rows={n_rows} (compacted)")

        X = sparse.coo_matrix((data, (row, g)), shape=(n_rows, G), dtype=dtype).tocsr()

        if verbose:
            print("[matrix] done")
            print(f"  X shape = {X.shape} (bins x genes)")
            print(f"  nnz     = {X.nnz}")
            print(f"  n_genes = {G}")

        out.update({"X": X, "bin_index": bin_index, "gene_names": gene_names})

    # B) Split matrix
    if return_matrix_split_assignment:
        if verbose:
            print("[matrix_split] building bin x (gene|ASSIGNED + gene|UNASSIGNED) matrix...")

        genes = df[gene_col].to_numpy(copy=False)
        gene_codes, gene_names = pd.factorize(genes, sort=True)
        ok = gene_codes >= 0

        n_na_gene = int((~ok).sum())
        if verbose and n_na_gene > 0:
            print(f"[matrix_split] dropped {n_na_gene} transcripts with NA gene")

        b = bin_id[ok]
        g = gene_codes[ok].astype(np.int64, copy=False)
        unassigned = np.asarray(unassigned_mask, dtype=bool)[ok]

        G = len(gene_names)
        col = g + (G * unassigned.astype(np.int64))
        data = np.ones(b.shape[0], dtype=dtype)

        if keep_empty_bins:
            row = b
            n_rows = P
            bin_index_split = np.arange(P, dtype=np.int64)
            if verbose:
                print(f"[matrix_split] keep_empty_bins=True -> n_rows={n_rows} (full grid)")
        else:
            bin_index_split, row = np.unique(b, return_inverse=True)
            n_rows = bin_index_split.size
            if verbose:
                print(f"[matrix_split] keep_empty_bins=False -> n_rows={n_rows} (compacted)")

        X_split = sparse.coo_matrix(
            (data, (row, col)),
            shape=(n_rows, 2 * G),
            dtype=dtype
        ).tocsr()

        # ✅ robust column names (no Index + string ufunc)
        gene_list = gene_names.astype(str).tolist()
        col_names_split = pd.Index(
            [f"{gn}|ASSIGNED" for gn in gene_list] +
            [f"{gn}|UNASSIGNED" for gn in gene_list]
        )

        if verbose:
            print("[matrix_split] done")
            print(f"  X_split shape = {X_split.shape} (bins x 2*genes)")
            print(f"  nnz           = {X_split.nnz}")
            print(f"  n_genes       = {G}")
            print(f"  assigned block   = [:, 0:{G}]")
            print(f"  unassigned block = [:, {G}:{2*G}]")

        out.update({
            "X_split": X_split,
            "bin_index_split": bin_index_split,
            "gene_names_split": gene_names,
            "col_names_split": col_names_split,
        })

    # C) cells + counts
    if return_cells:
        if verbose:
            print("[cells_by_bin] building bin_counts and cells_by_bin...")

        n_total = np.bincount(bin_id, minlength=P).astype(np.int32, copy=False)
        n_unassigned = np.bincount(bin_id[unassigned_mask], minlength=P).astype(np.int32, copy=False)

        present = n_total > 0
        present_bins = np.flatnonzero(present)

        bin_counts = pd.DataFrame(
            {"n_total": n_total[present_bins], "n_unassigned": n_unassigned[present_bins]},
            index=pd.Index(present_bins, name="bin_id"),
        )

        b_assigned = bin_id[assigned_mask]
        if b_assigned.size == 0:
            cells_by_bin = {}
            if verbose:
                print("[cells_by_bin] no assigned transcripts -> cells_by_bin is empty")
        else:
            b = np.asarray(b_assigned)
            c = np.asarray(cell_assigned_str)

            order = np.lexsort((c, b))
            b = b[order]
            c = c[order]

            keep = np.empty(b.size, dtype=bool)
            keep[0] = True
            keep[1:] = (b[1:] != b[:-1]) | (c[1:] != c[:-1])
            b = b[keep]
            c = c[keep]

            cuts = np.flatnonzero(b[1:] != b[:-1]) + 1
            starts = np.r_[0, cuts]
            ends = np.r_[cuts, b.size]

            cells_by_bin = {int(b[s]): set(c[s:e].tolist()) for s, e in zip(starts, ends)}

            if verbose:
                print("[cells_by_bin] done")
                print(f"  present bins    = {present_bins.size}")
                print(f"  bins w/ cells   = {len(cells_by_bin)}")

        out.update({"cells_by_bin": cells_by_bin, "bin_counts": bin_counts})

    if verbose:
        print("[bin_transcripts] finished")
        print("============================================================")

    return out


def bins_from_niche_threshold(
    niche_maps,
    grid_meta,
    k=0,
    threshold="p80",
):
    """
    Returns
    -------
    selected_bin_ids : np.ndarray[int64]
        Original bin_id values (y*width+x) where niche >= threshold.
    mask : np.ndarray[bool]
        (H,W) threshold mask used.
    t : float
        numeric threshold used
    """
    M = niche_maps[:, :, k]
    H, W = M.shape

    # sanity check vs grid_meta
    if (grid_meta["width"] != W) or (grid_meta["height"] != H):
        raise ValueError(
            f"niche map shape is (H={H}, W={W}) but grid_meta is "
            f"(height={grid_meta['height']}, width={grid_meta['width']}). "
            "Make sure you used the same bin_size and x/y mins."
        )

    # compute threshold value
    if isinstance(threshold, str) and threshold.startswith("p"):
        q = float(threshold[1:])
        t = float(np.nanpercentile(M, q))
    else:
        t = float(threshold)

    mask = (M >= t)

    ys, xs = np.nonzero(mask)
    selected_bin_ids = (ys.astype(np.int64) * grid_meta["width"] + xs.astype(np.int64))

    return selected_bin_ids, mask, t


def cells_in_selected_bins(
    selected_bin_ids,
    cells_by_bin,
    return_bin_to_cells=False,
):
    """
    Returns
    -------
    cells : set[str]
        Union of cell IDs appearing in any selected bin.
    (optional) bin_to_cells : dict[int,set[str]]
        Only bins that had any cells in cells_by_bin.
    """
    cells_union = set()
    if return_bin_to_cells:
        bin_to_cells = {}

    for b in selected_bin_ids:
        s = cells_by_bin.get(int(b))
        if s:
            cells_union.update(s)
            if return_bin_to_cells:
                bin_to_cells[int(b)] = s

    if return_bin_to_cells:
        return cells_union, bin_to_cells
    return cells_union
