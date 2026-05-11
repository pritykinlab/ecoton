"""
Statistics and utilities for niche maps and transcript binning.

This module provides functions for binning spatial transcript data,
computing niche maps, and analyzing cell distributions in selected bins.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

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

    # normalise unassigned_tokens: accept a bare scalar, str, or any iterable
    if isinstance(unassigned_tokens, (str, bytes)):
        unassigned_tokens = (unassigned_tokens,)
    elif not hasattr(unassigned_tokens, "__iter__"):
        unassigned_tokens = (unassigned_tokens,)
    else:
        unassigned_tokens = tuple(unassigned_tokens)

    if np.issubdtype(cell.dtype, np.number):
        if np.issubdtype(cell.dtype, np.integer):
            cell_int = cell.astype(np.int64, copy=False)
            unassigned = (cell_int == -1)
            for tok in unassigned_tokens:
                try:
                    unassigned |= (cell_int == int(tok))
                except (ValueError, TypeError):
                    pass
            assigned = ~unassigned
            cell_assigned_str = cell_int[assigned].astype(str)
        else:
            cell_num = cell.astype(np.float64, copy=False)
            unassigned = np.isnan(cell_num) | (cell_num == -1)
            # also honour user-supplied unassigned_tokens (e.g. 0)
            for tok in unassigned_tokens:
                try:
                    unassigned |= (cell_num == float(tok))
                except (ValueError, TypeError):
                    pass
            assigned = ~unassigned

            assigned_vals = cell_num[assigned]
            if assigned_vals.size and np.all(np.isfinite(assigned_vals)) and np.all(assigned_vals == np.floor(assigned_vals)):
                cell_assigned_str = assigned_vals.astype(np.int64).astype(str)
            else:
                cell_assigned_str = np.asarray([str(v) for v in assigned_vals])

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
    bin_size=8.0,
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


def bins_from_niche_threshold_with_support(
    niche_maps,
    grid_meta,
    binning_output,
    gene_list,
    k=0,
    threshold="p80",
    min_gene_types=2,
    min_gene_transcripts=5,
    case_sensitive=True,
    return_support_table=False,
):
    """
    Threshold a niche map and keep only bins with sufficient support
    from a provided gene list.

    Requires ``binning_output`` from ``bin_transcripts(..., return_matrix=True)``.

    Support criteria per bin (for transcripts whose gene is in gene_list):
    - at least ``min_gene_types`` unique genes
    - at least ``min_gene_transcripts`` total transcripts

    Returns
    -------
    selected_bin_ids_supported : np.ndarray[int64]
        Original bin_id values (y*width+x) that pass niche threshold and support checks.
    mask_supported : np.ndarray[bool]
        (H,W) boolean mask for supported selected bins.
    t : float
        Numeric niche threshold used.
    support_table : pd.DataFrame, optional
        Returned only when ``return_support_table=True``. Indexed by selected bin_id with
        columns: n_gene_types, n_gene_transcripts, passes_support.
    """
    selected_bin_ids, mask, t = bins_from_niche_threshold(
        niche_maps=niche_maps,
        grid_meta=grid_meta,
        k=k,
        threshold=threshold,
    )

    if min_gene_types < 1:
        raise ValueError("min_gene_types must be >= 1")
    if min_gene_transcripts < 1:
        raise ValueError("min_gene_transcripts must be >= 1")

    if binning_output is None:
        raise ValueError("binning_output must not be None")
    if gene_list is None:
        raise ValueError("gene_list must not be None")
    gene_list = list(gene_list)
    if len(gene_list) == 0:
        raise ValueError("gene_list must contain at least one gene")

    if "X" not in binning_output or "bin_index" not in binning_output or "gene_names" not in binning_output:
        raise ValueError(
            "binning_output must contain keys 'X', 'bin_index', and 'gene_names'. "
            "Pass the output of bin_transcripts(..., return_matrix=True)."
        )

    width = int(grid_meta["width"])

    if case_sensitive:
        gene_set = set(gene_list)
    else:
        gene_set = {str(g).upper() for g in gene_list}

    support_table = pd.DataFrame(index=pd.Index(selected_bin_ids, name="bin_id"))
    support_table["n_gene_types"] = 0
    support_table["n_gene_transcripts"] = 0

    X = binning_output["X"]
    bin_index = np.asarray(binning_output["bin_index"], dtype=np.int64)
    gene_names = np.asarray(binning_output["gene_names"])

    if case_sensitive:
        gene_name_mask = np.isin(gene_names, list(gene_set))
    else:
        gene_name_mask = np.isin(np.asarray([str(g).upper() for g in gene_names]), list(gene_set))

    if np.any(gene_name_mask) and selected_bin_ids.size > 0:
        selected_bin_ids_arr = np.asarray(selected_bin_ids, dtype=np.int64)

        order = np.argsort(bin_index)
        bin_sorted = bin_index[order]
        pos = np.searchsorted(bin_sorted, selected_bin_ids_arr)
        in_range = pos < bin_sorted.size
        found = np.zeros(selected_bin_ids_arr.size, dtype=bool)
        found[in_range] = (bin_sorted[pos[in_range]] == selected_bin_ids_arr[in_range])

        n_gene_types = np.zeros(selected_bin_ids_arr.size, dtype=np.int64)
        n_gene_transcripts = np.zeros(selected_bin_ids_arr.size, dtype=np.int64)

        if np.any(found):
            row_idx = order[pos[found]]
            X_sub = X[row_idx][:, gene_name_mask]
            if sparse.issparse(X_sub):
                n_gene_transcripts[found] = np.asarray(X_sub.sum(axis=1)).ravel().astype(np.int64)
                n_gene_types[found] = np.asarray((X_sub > 0).sum(axis=1)).ravel().astype(np.int64)
            else:
                n_gene_transcripts[found] = np.sum(X_sub, axis=1).astype(np.int64)
                n_gene_types[found] = np.sum(X_sub > 0, axis=1).astype(np.int64)

        support_table["n_gene_types"] = n_gene_types
        support_table["n_gene_transcripts"] = n_gene_transcripts

    support_table = support_table.fillna(0)
    support_table["n_gene_types"] = support_table["n_gene_types"].astype(np.int64)
    support_table["n_gene_transcripts"] = support_table["n_gene_transcripts"].astype(np.int64)

    support_table["passes_support"] = (
        (support_table["n_gene_types"] >= int(min_gene_types))
        & (support_table["n_gene_transcripts"] >= int(min_gene_transcripts))
    )

    pass_mask = support_table["passes_support"].to_numpy(dtype=bool)
    selected_bin_ids_supported = np.asarray(selected_bin_ids, dtype=np.int64)[pass_mask]

    mask_supported = np.zeros_like(mask, dtype=bool)
    if selected_bin_ids_supported.size > 0:
        ys = selected_bin_ids_supported // width
        xs = selected_bin_ids_supported % width
        mask_supported[ys, xs] = True

    if return_support_table:
        return selected_bin_ids_supported, mask_supported, t, support_table
    return selected_bin_ids_supported, mask_supported, t


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

def _geojson_to_shapely(gj, feature_index=0):
    """Convert a parsed GeoJSON dict to a shapely geometry."""
    try:
        import shapely.geometry as sg
    except ImportError:
        raise ImportError("shapely is required. Install with: pip install shapely")

    gj_type = gj.get("type", "")
    if gj_type == "FeatureCollection":
        features = gj.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection has no features.")
        if feature_index >= len(features):
            raise ValueError(
                f"feature_index={feature_index} is out of range "
                f"(FeatureCollection has {len(features)} features)."
            )
        geom = features[feature_index].get("geometry")
        if geom is None:
            raise ValueError(f"Feature at index {feature_index} has no geometry.")
    elif gj_type == "Feature":
        geom = gj.get("geometry")
        if geom is None:
            raise ValueError("GeoJSON Feature has no geometry.")
    else:
        geom = gj  # assume it is already a geometry object

    return sg.shape(geom)


def _parse_polygon_input(polygon, feature_index=0):
    """Parse polygon input from a GeoJSON file path, dict, or coordinate array."""
    try:
        import shapely.geometry as sg
    except ImportError:
        raise ImportError("shapely is required. Install with: pip install shapely")

    import json

    # shapely geometry passed directly (handles holes, MultiPolygon, etc.)
    if hasattr(polygon, "geom_type"):
        return polygon

    if isinstance(polygon, str):
        with open(polygon, "r") as f:
            gj = json.load(f)
        return _geojson_to_shapely(gj, feature_index=feature_index)
    elif isinstance(polygon, dict):
        return _geojson_to_shapely(polygon, feature_index=feature_index)
    else:
        coords = np.asarray(polygon, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "polygon as array-like must have shape (N, 2) with columns [x, y]."
            )
        return sg.Polygon(coords)


def bins_in_polygon(
    polygon,
    grid_meta,
    bin_size,
    bin_ids=None,
    use_bin_center=True,
    feature_index=0,
    x_offset=0.0,
    y_offset=0.0,
    scale=1.0,
    verbose=False,
):
    """
    Return bin IDs whose representative point falls inside a polygon.

    Parameters
    ----------
    polygon : str, dict, or array-like
        Polygon specification. Accepted forms:

        - **str** – path to a GeoJSON file.
        - **dict** – parsed GeoJSON object (Polygon geometry, Feature, or
          FeatureCollection).
        - **array-like of shape (N, 2)** – (x, y) coordinate pairs defining
          the polygon ring. Coordinates must be in the same space as the
          transcript data.

    grid_meta : dict
        Grid metadata returned by ``bin_transcripts`` (keys: ``x_min``,
        ``y_min``, ``width``, ``height``).
    bin_size : float
        Bin size (spatial units) used when calling ``bin_transcripts``.
    bin_ids : array-like or None, default None
        Restrict the search to this subset of bin IDs (e.g. the
        ``bin_index`` array from ``bin_transcripts``). If ``None``, every
        bin in the full grid is tested.
    use_bin_center : bool, default True
        If ``True``, test the geometric centre of each bin.
        If ``False``, test the lower-left corner of each bin.
    feature_index : int, default 0
        When *polygon* is a FeatureCollection, selects which feature to use.
    x_offset : float, default 0.0
        Shift added to polygon x coordinates *before* scaling.
        Use to align annotations drawn in a different origin (e.g. full-slide
        pixel space) to the transcript coordinate origin.
    y_offset : float, default 0.0
        Shift added to polygon y coordinates *before* scaling.
    scale : float or (sx, sy), default 1.0
        Scale factor applied *after* the offset. Pass a single float to scale
        both axes equally, or a ``(sx, sy)`` tuple for anisotropic scaling.
        Example: if the polygon was drawn in pixels and transcripts are in
        microns with pixel_size=0.2125 µm/px, use ``scale=0.2125``.
        The full transform applied to polygon coordinates is:
        ``x_new = (x_poly + x_offset) * sx``
        ``y_new = (y_poly + y_offset) * sy``

    Returns
    -------
    selected_bin_ids : np.ndarray[int64]
        Bin IDs (``gy * width + gx``) whose representative point lies inside
        the polygon.

    Notes
    -----
    Requires `shapely <https://shapely.readthedocs.io/>`_. Install with::

        pip install shapely

    The polygon coordinates must be in the same coordinate system as the
    ``x_location`` / ``y_location`` columns passed to ``bin_transcripts``.

    Examples
    --------
    >>> result = bin_transcripts(df, bin_size=8.0, return_matrix=True)
    >>> inside = bins_in_polygon(
    ...     "region.geojson",
    ...     grid_meta=result["grid_meta"],
    ...     bin_size=8.0,
    ...     bin_ids=result["bin_index"],
    ... )
    >>> cells = cells_in_selected_bins(inside, result["cells_by_bin"])
    """
    try:
        import shapely
        import shapely.geometry as sg
    except ImportError:
        raise ImportError(
            "shapely is required for bins_in_polygon. "
            "Install it with:  pip install shapely"
        )

    poly_geom = _parse_polygon_input(polygon, feature_index=feature_index)

    # Apply optional coordinate transform: new = (old + offset) * scale
    if x_offset != 0.0 or y_offset != 0.0 or scale != 1.0:
        import shapely.affinity as _sa
        if isinstance(scale, (list, tuple)):
            sx, sy = float(scale[0]), float(scale[1])
        else:
            sx = sy = float(scale)
        if x_offset != 0.0 or y_offset != 0.0:
            poly_geom = _sa.translate(poly_geom, xoff=float(x_offset), yoff=float(y_offset))
        if sx != 1.0 or sy != 1.0:
            poly_geom = _sa.scale(poly_geom, xfact=sx, yfact=sy, origin=(0, 0))
        if verbose:
            print(f"[bins_in_polygon] applied transform: offset=({x_offset}, {y_offset}), scale=({sx}, {sy})")

    x_min = float(grid_meta["x_min"])
    y_min = float(grid_meta["y_min"])
    width = int(grid_meta["width"])
    height = int(grid_meta["height"])

    if verbose:
        px_min, py_min, px_max, py_max = poly_geom.bounds
        grid_x_max = x_min + width * bin_size
        grid_y_max = y_min + height * bin_size
        print("[bins_in_polygon] polygon bounds (x_min, y_min, x_max, y_max):")
        print(f"  x: [{px_min:.3f}, {px_max:.3f}]")
        print(f"  y: [{py_min:.3f}, {py_max:.3f}]")
        print(f"[bins_in_polygon] grid spatial extent:")
        print(f"  x: [{x_min:.3f}, {grid_x_max:.3f}]")
        print(f"  y: [{y_min:.3f}, {grid_y_max:.3f}]")
        x_overlap = px_min < grid_x_max and px_max > x_min
        y_overlap = py_min < grid_y_max and py_max > y_min
        print(f"[bins_in_polygon] overlap check: x={x_overlap}, y={y_overlap}")
        if not (x_overlap and y_overlap):
            print("  WARNING: polygon does not overlap grid extent — expect empty result")
            print("  Hint: check for coordinate system mismatch or y-axis flip")

    if bin_ids is None:
        candidate_bin_ids = np.arange(width * height, dtype=np.int64)
    else:
        candidate_bin_ids = np.asarray(bin_ids, dtype=np.int64)

    gx = (candidate_bin_ids % width).astype(np.float64)
    gy = (candidate_bin_ids // width).astype(np.float64)

    offset = 0.5 if use_bin_center else 0.0
    x_coords = x_min + (gx + offset) * bin_size
    y_coords = y_min + (gy + offset) * bin_size

    if verbose:
        print(f"[bins_in_polygon] candidate bins: {candidate_bin_ids.size}")
        print(f"  bin x_coords range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"  bin y_coords range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")

    # shapely 2.x: vectorised and fast
    if hasattr(shapely, "contains_xy"):
        inside = shapely.contains_xy(poly_geom, x_coords, y_coords)
    else:
        # shapely 1.x fallback: prepared geometry for speed
        from shapely.prepared import prep
        prepared_poly = prep(poly_geom)
        inside = np.array(
            [prepared_poly.contains(sg.Point(x, y)) for x, y in zip(x_coords, y_coords)],
            dtype=bool,
        )

    if verbose:
        print(f"[bins_in_polygon] bins inside polygon: {inside.sum()} / {inside.size}")

    return candidate_bin_ids[inside]


def flatten_valid(N, mask=None):
    """
    Flatten a 2D niche map and remove NaNs.
    """
    N = np.asarray(N)
    if mask is not None:
        x = N[mask]
    else:
        x = N.ravel()
    x = x[np.isfinite(x)]
    return x


def _find_signal_start_index(
    x_sorted,
    slope_quantile=0.2,
    min_run=25,
    smooth_window=11,
):
    """
    Find where the sorted curve begins to leave a flat baseline.

    Parameters
    ----------
    x_sorted : np.ndarray
        Sorted values.
    slope_quantile : float
        Use this quantile of positive slopes as a reference scale.
        Smaller = earlier start, larger = later start.
    min_run : int
        Require at least this many consecutive points above threshold.
    smooth_window : int
        Simple moving-average window for slope smoothing.

    Returns
    -------
    start_idx : int
        Index in x_sorted where the signal begins.
    """
    n = len(x_sorted)
    if n < 10:
        return 0

    # slope of sorted curve
    dx = np.gradient(x_sorted.astype(float))

    # smooth slope a bit
    smooth_window = max(3, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window) / smooth_window
    dx_smooth = np.convolve(dx, kernel, mode="same")

    # reference threshold from positive slopes
    pos = dx_smooth[dx_smooth > 0]
    if len(pos) == 0:
        return 0

    slope_thr = np.quantile(pos, slope_quantile)

    # require a sustained run above threshold
    above = dx_smooth > slope_thr

    run = 0
    for i, flag in enumerate(above):
        if flag:
            run += 1
            if run >= min_run:
                return max(0, i - min_run + 1)
        else:
            run = 0

    return 0


def _knee_on_sorted_segment(x_sorted_segment):
    """
    Compute knee index for a sorted 1D segment using max distance to endpoint line.

    Returns
    -------
    knee_idx : int
        Knee index within the provided segment.
    xs : np.ndarray
        Normalized x coordinate in [0, 1] for the segment.
    ys : np.ndarray
        Normalized y coordinate in [0, 1] for the segment.
    dists : np.ndarray
        Per-point distance to the endpoint line.
    """
    n = len(x_sorted_segment)
    if n < 2:
        return 0, np.array([0.0]), np.array([0.0]), np.array([0.0])

    xs = np.linspace(0, 1, n)
    ys = (x_sorted_segment - x_sorted_segment.min()) / (x_sorted_segment.max() - x_sorted_segment.min() + 1e-12)

    p1 = np.array([xs[0], ys[0]])
    p2 = np.array([xs[-1], ys[-1]])
    line_vec = p2 - p1
    line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-12)

    points = np.column_stack([xs, ys])
    vecs = points - p1
    proj = np.outer(np.dot(vecs, line_vec), line_vec)
    perp = vecs - proj
    dists = np.linalg.norm(perp, axis=1)

    knee_idx = int(np.argmax(dists))
    return knee_idx, xs, ys, dists


def _right_distance_fraction_index(dists, peak_idx, distance_fraction=0.9):
    """
    Find a right-side boundary of a knee distance peak at a target fraction.

    This gives a conservative threshold downstream of the main knee without
    choosing an arbitrary percentile delta.
    """
    dists = np.asarray(dists, dtype=float)
    peak_idx = int(peak_idx)
    if dists.size == 0:
        return 0

    peak_idx = int(np.clip(peak_idx, 0, dists.size - 1))
    peak_dist = dists[peak_idx]
    if peak_dist <= 0:
        return peak_idx

    target_dist = float(distance_fraction) * peak_dist
    tail = dists[peak_idx:]
    crossing = np.where(tail <= target_dist)[0]
    if crossing.size > 0:
        return int(min(peak_idx + crossing[0], dists.size - 1))

    return int(dists.size - 1)


def _darken_color(color, factor=0.65):
    """Return a darker version of any Matplotlib-compatible color."""
    rgb = np.array(to_rgb(color))
    return to_hex(np.clip(rgb * factor, 0, 1))


def knee_from_sorted_curve(
    N,
    mask=None,
    start_mode="auto",   # "auto", "zero", or integer index
    normalize_y=False,
    figsize=(3, 3),
    dpi=150,
    plot_top_zoom=False,
    zoom_top_fraction=0.05,
    curve_color="#1f77b4",
    line_color="#d62728",
    second_line_color="#ff7f0e",
    sharper_first_line_color=None,
    sharper_second_line_color=None,
    lw=2.2,
    show=True,
    ax=None,
    slope_quantile=0.2,
    min_run=25,
    smooth_window=11,
    detect_second_knee=True,
    detect_sharper_first_knee=True,
    sharper_first_knee_distance_fraction=0.95,
    detect_sharper_second_knee=True,
    sharper_second_knee_distance_fraction=None,
):
    """
    Publication-style single-panel knee plot for a 2D niche map.

    The knee is computed starting from the point where signal begins,
    rather than necessarily from x > 0.

    Parameters
    ----------
    N : np.ndarray
        2D niche intensity map.
    mask : np.ndarray or None
        Optional boolean tissue mask.
    start_mode : {"auto", "zero"} or int
        How to choose where the knee computation starts:
        - "auto": detect end of flat baseline using slope
        - "zero": start at first value > 0
        - int: explicit start index in full sorted array
    normalize_y : bool
        If True, plot normalized intensities in [0, 1].
        If False, plot raw sorted intensities.
    slope_quantile : float
        Only used for start_mode="auto".
    min_run : int
        Only used for start_mode="auto".
    smooth_window : int
        Only used for start_mode="auto".

    Returns
    -------
    threshold : float
        Intensity at the first knee.
    occupancy : float
        Fraction of valid pixels above first-knee threshold.
    info : dict
        Diagnostic information, including sharper first/second knee fields when enabled.
    """
    # Full data for plotting + occupancy
    x_all = flatten_valid(N, mask=mask)
    if x_all.size < 10:
        raise ValueError("Not enough valid pixels to estimate a knee.")

    if sharper_first_line_color is None:
        sharper_first_line_color = _darken_color(curve_color)
    if sharper_second_line_color is None:
        sharper_second_line_color = _darken_color(second_line_color)

    x_all_sorted = np.sort(x_all)
    n_all = len(x_all_sorted)
    xs_all = np.linspace(0, 1, n_all)

    # Choose where the fit begins
    if start_mode == "auto":
        start_idx = _find_signal_start_index(
            x_all_sorted,
            slope_quantile=slope_quantile,
            min_run=min_run,
            smooth_window=smooth_window,
        )
    elif start_mode == "zero":
        start_idx = np.searchsorted(x_all_sorted, 0.0, side="right")
        start_idx = min(start_idx, n_all - 1)
    elif isinstance(start_mode, (int, np.integer)):
        start_idx = int(np.clip(start_mode, 0, n_all - 1))
    else:
        raise ValueError("start_mode must be 'auto', 'zero', or an integer index.")

    x_fit_sorted = x_all_sorted[start_idx:]
    if x_fit_sorted.size < 10:
        raise ValueError("Not enough values after choosing start index to estimate a knee.")

    n_fit = len(x_fit_sorted)
    knee_idx_fit, xs_fit, ys_fit, dists = _knee_on_sorted_segment(x_fit_sorted)
    threshold = x_fit_sorted[knee_idx_fit]

    # Occupancy on all valid pixels (first knee)
    occupancy = np.mean(x_all > threshold)
    knee_pct = 100 * (1 - occupancy)

    # Map first knee threshold back onto full sorted curve
    knee_idx_all = np.searchsorted(x_all_sorted, threshold, side="left")
    knee_idx_all = np.clip(knee_idx_all, 0, n_all - 1)
    knee_x_all = xs_all[knee_idx_all]
    start_x_all = xs_all[start_idx]

    # Optional second knee: search from first knee to tail of fit domain
    second_threshold = None
    second_occupancy = None
    second_knee_pct = None
    second_knee_idx_fit = None
    second_knee_idx_all = None
    second_knee_x_all = None
    second_dists = None
    second_idx_local = None

    if detect_second_knee:
        second_start_fit = knee_idx_fit
        x_second_fit_sorted = x_fit_sorted[second_start_fit:]
        if x_second_fit_sorted.size >= 10:
            second_idx_local, _, _, second_dists = _knee_on_sorted_segment(x_second_fit_sorted)
            second_knee_idx_fit = int(second_start_fit + second_idx_local)
            second_threshold = x_fit_sorted[second_knee_idx_fit]
            second_occupancy = np.mean(x_all > second_threshold)
            second_knee_pct = 100 * (1 - second_occupancy)
            second_knee_idx_all = int(np.searchsorted(x_all_sorted, second_threshold, side="left"))
            second_knee_idx_all = int(np.clip(second_knee_idx_all, 0, n_all - 1))
            second_knee_x_all = xs_all[second_knee_idx_all]

    # Optional sharper first knee: conservative first-knee threshold from first-knee geometry.
    sharper_first_threshold = None
    sharper_first_occupancy = None
    sharper_first_knee_pct = None
    sharper_first_knee_idx_fit = None
    sharper_first_knee_idx_all = None
    sharper_first_knee_x_all = None

    if detect_sharper_first_knee:
        sharper_first_knee_idx_fit = _right_distance_fraction_index(
            dists,
            knee_idx_fit,
            distance_fraction=sharper_first_knee_distance_fraction,
        )
        if sharper_first_knee_idx_fit <= knee_idx_fit and knee_idx_fit < n_fit - 1:
            sharper_first_knee_idx_fit = int(knee_idx_fit + 1)
        sharper_first_threshold = x_fit_sorted[sharper_first_knee_idx_fit]
        sharper_first_occupancy = np.mean(x_all > sharper_first_threshold)
        sharper_first_knee_pct = 100 * (1 - sharper_first_occupancy)
        sharper_first_knee_idx_all = int(np.searchsorted(x_all_sorted, sharper_first_threshold, side="left"))
        sharper_first_knee_idx_all = int(np.clip(sharper_first_knee_idx_all, 0, n_all - 1))
        sharper_first_knee_x_all = xs_all[sharper_first_knee_idx_all]

    # Optional sharper second knee: conservative version of the second knee.
    sharper_second_threshold = None
    sharper_second_occupancy = None
    sharper_second_knee_pct = None
    sharper_second_knee_idx_fit = None
    sharper_second_knee_idx_all = None
    sharper_second_knee_x_all = None

    if detect_sharper_second_knee and second_knee_idx_fit is not None and second_dists is not None:
        sharper_second_fraction = (
            sharper_first_knee_distance_fraction
            if sharper_second_knee_distance_fraction is None
            else sharper_second_knee_distance_fraction
        )
        sharper_second_idx_local = _right_distance_fraction_index(
            second_dists,
            second_idx_local,
            distance_fraction=sharper_second_fraction,
        )
        if sharper_second_idx_local <= second_idx_local and second_idx_local < len(second_dists) - 1:
            sharper_second_idx_local = int(second_idx_local + 1)
        sharper_second_knee_idx_fit = int(second_start_fit + sharper_second_idx_local)
        sharper_second_threshold = x_fit_sorted[sharper_second_knee_idx_fit]
        sharper_second_occupancy = np.mean(x_all > sharper_second_threshold)
        sharper_second_knee_pct = 100 * (1 - sharper_second_occupancy)
        sharper_second_knee_idx_all = int(np.searchsorted(x_all_sorted, sharper_second_threshold, side="left"))
        sharper_second_knee_idx_all = int(np.clip(sharper_second_knee_idx_all, 0, n_all - 1))
        sharper_second_knee_x_all = xs_all[sharper_second_knee_idx_all]

    # Plot values
    if normalize_y:
        y_all_plot = (x_all_sorted - x_all_sorted.min()) / (x_all_sorted.max() - x_all_sorted.min() + 1e-12)
        y_label = "Normalized intensity"
        knee_y_all = y_all_plot[knee_idx_all]

        # reference line over fit domain only
        ref_x = [start_x_all, 1.0]
        ref_y = [0.0, 1.0]
    else:
        y_all_plot = x_all_sorted
        y_label = "Intensity"
        knee_y_all = x_all_sorted[knee_idx_all]

        ref_x = [start_x_all, 1.0]
        ref_y = [x_fit_sorted[0], x_fit_sorted[-1]]

    # Create figure if needed
    created_fig = False
    if ax is None:
        if plot_top_zoom:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(figsize[0] * 2.1, figsize[1]),
                dpi=dpi,
                gridspec_kw={"width_ratios": [1.0, 1.0]},
            )
            ax = axes[0]
            zoom_ax = axes[1]
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            zoom_ax = None
        created_fig = True
    else:
        fig = ax.figure
        zoom_ax = None

    # Full curve
    ax.plot(xs_all, y_all_plot, color=curve_color, lw=lw)

    # Show fit start
    ax.axvline(start_x_all, linestyle=":", color="0.4", lw=1.2, alpha=0.9)

    # Reference dotted line corresponding to fit domain
    ax.plot(ref_x, ref_y, linestyle="--", color=line_color, lw=1.8, alpha=0.9)

    # Knee guides
    ax.axvline(knee_x_all, linestyle="--", color=line_color, lw=1.5, alpha=0.85)
    ax.axhline(knee_y_all, linestyle="--", color=line_color, lw=1.5, alpha=0.85)

    # Knee marker
    ax.scatter(knee_x_all, knee_y_all, s=55, color=curve_color, zorder=5)

    # Second knee guides/marker
    if second_knee_idx_all is not None:
        second_knee_y_all = y_all_plot[second_knee_idx_all]
        ax.axvline(second_knee_x_all, linestyle="-.", color=second_line_color, lw=1.5, alpha=0.9)
        ax.axhline(second_knee_y_all, linestyle="-.", color=second_line_color, lw=1.5, alpha=0.9)
        ax.scatter(second_knee_x_all, second_knee_y_all, s=48, color=second_line_color, zorder=5)

    # Sharper first knee guides/marker
    if sharper_first_knee_idx_all is not None:
        sharper_first_knee_y_all = y_all_plot[sharper_first_knee_idx_all]
        ax.axvline(sharper_first_knee_x_all, linestyle=(0, (3, 1, 1, 1)), color=sharper_first_line_color, lw=1.5, alpha=0.95)
        ax.axhline(sharper_first_knee_y_all, linestyle=(0, (3, 1, 1, 1)), color=sharper_first_line_color, lw=1.5, alpha=0.95)
        ax.scatter(sharper_first_knee_x_all, sharper_first_knee_y_all, s=44, color=sharper_first_line_color, zorder=5)

    # Sharper second knee guides/marker
    if sharper_second_knee_idx_all is not None:
        sharper_second_knee_y_all = y_all_plot[sharper_second_knee_idx_all]
        ax.axvline(sharper_second_knee_x_all, linestyle=(0, (5, 1, 1, 1)), color=sharper_second_line_color, lw=1.5, alpha=0.95)
        ax.axhline(sharper_second_knee_y_all, linestyle=(0, (5, 1, 1, 1)), color=sharper_second_line_color, lw=1.5, alpha=0.95)
        ax.scatter(sharper_second_knee_x_all, sharper_second_knee_y_all, s=44, color=sharper_second_line_color, zorder=5)

    percentile_parts = [f"K1 p{knee_pct:.1f}"]
    if sharper_first_knee_pct is not None:
        percentile_parts.append(f"K1S p{sharper_first_knee_pct:.1f}")
    if second_knee_pct is not None:
        percentile_parts.append(f"K2 p{second_knee_pct:.1f}")
    if sharper_second_knee_pct is not None:
        percentile_parts.append(f"K2S p{sharper_second_knee_pct:.1f}")
    percentile_title = "Sorted niche intensity distribution\n" + " | ".join(percentile_parts)

    # Labels and title
    ax.set_title(percentile_title, fontsize=7.5, pad=6)
    ax.set_xlabel("Normalized sorted index", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)

    # Cleaner style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, width=1)
    ax.set_box_aspect(1)

    # Optional right-hand zoom into the high-intensity tail.
    if zoom_ax is not None:
        zoom_start_idx = int(np.clip(np.floor((1.0 - float(zoom_top_fraction)) * n_all), 0, n_all - 1))
        zoom_start_x = xs_all[zoom_start_idx]

        zoom_ax.plot(xs_all[zoom_start_idx:], y_all_plot[zoom_start_idx:], color=curve_color, lw=lw)
        zoom_ax.axvline(knee_x_all, linestyle="--", color=line_color, lw=1.5, alpha=0.85)
        zoom_ax.axhline(knee_y_all, linestyle="--", color=line_color, lw=1.5, alpha=0.85)
        zoom_ax.scatter(knee_x_all, knee_y_all, s=45, color=curve_color, zorder=5)

        if second_knee_idx_all is not None:
            second_knee_y_all = y_all_plot[second_knee_idx_all]
            zoom_ax.axvline(second_knee_x_all, linestyle="-.", color=second_line_color, lw=1.4, alpha=0.9)
            zoom_ax.axhline(second_knee_y_all, linestyle="-.", color=second_line_color, lw=1.4, alpha=0.9)
            zoom_ax.scatter(second_knee_x_all, second_knee_y_all, s=40, color=second_line_color, zorder=5)

        if sharper_first_knee_idx_all is not None:
            sharper_first_knee_y_all = y_all_plot[sharper_first_knee_idx_all]
            zoom_ax.axvline(sharper_first_knee_x_all, linestyle=(0, (3, 1, 1, 1)), color=sharper_first_line_color, lw=1.4, alpha=0.95)
            zoom_ax.axhline(sharper_first_knee_y_all, linestyle=(0, (3, 1, 1, 1)), color=sharper_first_line_color, lw=1.4, alpha=0.95)
            zoom_ax.scatter(sharper_first_knee_x_all, sharper_first_knee_y_all, s=38, color=sharper_first_line_color, zorder=5)

        if sharper_second_knee_idx_all is not None:
            sharper_second_knee_y_all = y_all_plot[sharper_second_knee_idx_all]
            zoom_ax.axvline(sharper_second_knee_x_all, linestyle=(0, (5, 1, 1, 1)), color=sharper_second_line_color, lw=1.4, alpha=0.95)
            zoom_ax.axhline(sharper_second_knee_y_all, linestyle=(0, (5, 1, 1, 1)), color=sharper_second_line_color, lw=1.4, alpha=0.95)
            zoom_ax.scatter(sharper_second_knee_x_all, sharper_second_knee_y_all, s=38, color=sharper_second_line_color, zorder=5)

        zoom_ax.set_xlim(zoom_start_x, 1.0)
        zoom_ax.set_title(f"Top {100 * zoom_top_fraction:.1f}% zoom\n" + " | ".join(percentile_parts), fontsize=7.5, pad=6)
        zoom_ax.set_xlabel("Normalized sorted index", fontsize=9)
        zoom_ax.set_ylabel(y_label, fontsize=9)
        zoom_ax.spines["top"].set_visible(False)
        zoom_ax.spines["right"].set_visible(False)
        zoom_ax.tick_params(axis="both", labelsize=8, width=1)
        zoom_ax.set_box_aspect(1)

    fig.tight_layout()

    if show and created_fig:
        plt.show()

    info = {
        "start_idx": start_idx,
        "start_x_all": start_x_all,
        "knee_index_fit": int(knee_idx_fit),
        "knee_index_all": int(knee_idx_all),
        "knee_x_all": knee_x_all,
        "threshold": threshold,
        "occupancy": occupancy,
        "approx_percentile": knee_pct,
        "second_knee_enabled": detect_second_knee,
        "second_knee_index_fit": second_knee_idx_fit,
        "second_knee_index_all": second_knee_idx_all,
        "second_knee_x_all": second_knee_x_all,
        "second_threshold": second_threshold,
        "second_occupancy": second_occupancy,
        "second_approx_percentile": second_knee_pct,
        "sharper_first_knee_enabled": detect_sharper_first_knee,
        "sharper_first_knee_method": "right_distance_fraction",
        "sharper_first_knee_distance_fraction": sharper_first_knee_distance_fraction,
        "sharper_first_knee_index_fit": sharper_first_knee_idx_fit,
        "sharper_first_knee_index_all": sharper_first_knee_idx_all,
        "sharper_first_knee_x_all": sharper_first_knee_x_all,
        "sharper_first_threshold": sharper_first_threshold,
        "sharper_first_occupancy": sharper_first_occupancy,
        "sharper_first_approx_percentile": sharper_first_knee_pct,
        "sharper_second_knee_enabled": detect_sharper_second_knee,
        "sharper_second_knee_method": "right_distance_fraction_from_second_knee",
        "sharper_second_knee_distance_fraction": (
            sharper_second_fraction if "sharper_second_fraction" in locals() else None
        ),
        "sharper_second_knee_index_fit": sharper_second_knee_idx_fit,
        "sharper_second_knee_index_all": sharper_second_knee_idx_all,
        "sharper_second_knee_x_all": sharper_second_knee_x_all,
        "sharper_second_threshold": sharper_second_threshold,
        "sharper_second_occupancy": sharper_second_occupancy,
        "sharper_second_approx_percentile": sharper_second_knee_pct,
        "plot_top_zoom": plot_top_zoom,
        "zoom_top_fraction": zoom_top_fraction,
        "x_all_sorted": x_all_sorted,
        "xs_all": xs_all,
        "x_fit_sorted": x_fit_sorted,
        "xs_fit": xs_fit,
        "ys_fit": ys_fit,
        "distances": dists,
        "second_distances": second_dists,
    }

    return threshold, occupancy, info
