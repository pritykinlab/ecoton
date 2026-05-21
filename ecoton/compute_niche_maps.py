"""
Efficient niche map computation utilities.

Provides functions to build blurred, z-scored niche maps per archetype
from transcript coordinates, gene labels, and archetype weight matrix `W`.
"""

import numpy as np
from scipy import ndimage, sparse


def _select_genes_for_niche_maps(W, weight_threshold, union_driver_genes, selected_genes=None):
    if selected_genes is not None:
        selected_genes = W.index.intersection(np.asarray(selected_genes))
        print(f"Using provided gene set: {len(selected_genes)} / {len(W.index)}")
        return selected_genes

    if union_driver_genes:
        if weight_threshold is None:
            driver_mask = (W.values != 0).any(axis=1)
        else:
            driver_mask = (W.values > weight_threshold).any(axis=1)
        genes_use = W.index[driver_mask]
        print(f"Using union of driver genes: {len(genes_use)} / {len(W.index)}")
        return genes_use

    print(f"Using all genes in W: {len(W.index)}")
    return W.index


def _prepare_spatial_grid(coords, bin_size):
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    width = int(np.ceil((x_max - x_min) / bin_size)) + 1
    height = int(np.ceil((y_max - y_min) / bin_size)) + 1

    grid_x = ((coords[:, 0] - x_min) / bin_size).astype(np.int32)
    grid_y = ((coords[:, 1] - y_min) / bin_size).astype(np.int32)
    grid_x = np.clip(grid_x, 0, width - 1)
    grid_y = np.clip(grid_y, 0, height - 1)

    flat_pixels = grid_y * width + grid_x
    return height, width, flat_pixels


def _build_blurred_gene_maps(coords, gene_labels, gene_index, bin_size, smoothing_radius):
    height, width, flat_pixels = _prepare_spatial_grid(coords, bin_size)
    P = height * width
    G = len(gene_index)

    gene_ids = gene_index.get_indexer(gene_labels).astype(np.int32)
    ok = gene_ids >= 0

    if not np.any(ok):
        return np.zeros((height, width, G), dtype=np.float32), height, width

    pixel_gene = sparse.coo_matrix(
        (np.ones(int(ok.sum()), dtype=np.float32), (flat_pixels[ok], gene_ids[ok])),
        shape=(P, G),
    ).toarray()

    gene_maps = pixel_gene.reshape(height, width, G)
    sigma_pixels = smoothing_radius / bin_size
    blurred = ndimage.gaussian_filter(
        gene_maps,
        sigma=(sigma_pixels, sigma_pixels, 0.0),
        mode="constant",
    ).astype(np.float32)
    return blurred, height, width


def _compute_slice_niche_maps(blurred, W_sub, mean_g, std_g, eps):
    good = std_g > eps
    norm = np.zeros_like(blurred, dtype=np.float32)
    norm[:, :, good] = (blurred[:, :, good] - mean_g[good]) / std_g[good]
    niche_flat = norm.reshape(-1, blurred.shape[2]) @ W_sub
    return niche_flat.reshape(blurred.shape[0], blurred.shape[1], W_sub.shape[1]).astype(np.float32)


def _build_weight_matrix(W, genes_use, weight_threshold):
    W_sub = W.loc[genes_use, :].to_numpy(dtype=np.float32, copy=False)
    if weight_threshold is not None:
        W_sub = np.where(W_sub > weight_threshold, W_sub, 0.0).astype(np.float32)
    return W_sub


def _build_normalization_stats_dict(
    genes_use,
    mean_g,
    std_g,
    *,
    weight_threshold,
    union_driver_genes,
    selected_genes,
    n_slices=None,
    n_pooled_pixels=None,
):
    return {
        "genes_use": np.asarray(genes_use),
        "mean_g": np.asarray(mean_g, dtype=np.float32),
        "std_g": np.asarray(std_g, dtype=np.float32),
        "weight_threshold": weight_threshold,
        "union_driver_genes": union_driver_genes,
        "selected_genes": None if selected_genes is None else np.asarray(selected_genes),
        "n_slices": n_slices,
        "n_pooled_pixels": n_pooled_pixels,
    }


def _resolve_normalization_context(
    W,
    *,
    weight_threshold,
    union_driver_genes,
    selected_genes,
    normalization_stats=None,
):
    if normalization_stats is None:
        genes_use = _select_genes_for_niche_maps(
            W,
            weight_threshold=weight_threshold,
            union_driver_genes=union_driver_genes,
            selected_genes=selected_genes,
        )
        return W.index.__class__(genes_use), genes_use, None, None

    required = ("genes_use", "mean_g", "std_g")
    missing = [key for key in required if key not in normalization_stats]
    if missing:
        raise ValueError(f"normalization_stats is missing required keys: {missing}")

    genes_use = np.asarray(normalization_stats["genes_use"])
    mean_g = np.asarray(normalization_stats["mean_g"], dtype=np.float32)
    std_g = np.asarray(normalization_stats["std_g"], dtype=np.float32)
    if len(genes_use) != len(mean_g) or len(genes_use) != len(std_g):
        raise ValueError("normalization_stats genes_use, mean_g, and std_g must have the same length.")

    gene_index = W.index.__class__(genes_use)
    missing_in_W = gene_index.get_indexer(genes_use) < 0
    if np.any(missing_in_W):
        missing_genes = np.asarray(genes_use)[missing_in_W].tolist()
        raise ValueError(f"normalization_stats contains genes that are not present in W.index: {missing_genes[:10]}")

    print(f"Using provided normalization stats: {len(genes_use)} genes")
    return gene_index, genes_use, mean_g, std_g


def _normalize_slice_input(slice_item):
    if isinstance(slice_item, dict):
        coords = slice_item.get("coords")
        gene_labels = slice_item.get("gene_labels")
        slice_name = slice_item.get("slice_name")
    else:
        if len(slice_item) == 2:
            coords, gene_labels = slice_item
            slice_name = None
        elif len(slice_item) == 3:
            coords, gene_labels, slice_name = slice_item
        else:
            raise ValueError(
                "Each slice item must be (coords, gene_labels), (coords, gene_labels, slice_name), "
                "or a dict with keys 'coords' and 'gene_labels'."
            )

    coords = np.asarray(coords)
    gene_labels = np.asarray(gene_labels)
    if gene_labels.shape[0] != coords.shape[0]:
        raise ValueError("gene_labels must have the same length as coords for each slice.")
    return coords, gene_labels, slice_name


def create_niche_maps_by_archetype_all_at_once(
    coords,             # (N,2) float32
    gene_labels,        # (N,) array-like of str (gene name for each transcript)
    W,                  # pandas.DataFrame: index=gene labels, columns=archetypes
    bin_size=8.0,
    smoothing_radius=8.0,
    weight_threshold=0.3,
    union_driver_genes=True,   # if True, only compute maps for genes used by any archetype
    selected_genes=None,
    normalization_stats=None,
    return_stats=False,
    eps=1e-9,
):
    """
    Efficient spatial reconstruction:
      - Build binned per-gene maps once (for union of driver genes or all W genes)
      - Blur all genes at once using a single 3D gaussian_filter
      - Z-score each gene map once
      - Weighted sum for all archetypes via matrix multiply

    Returns
    -------
    niche_maps : (height, width, n_arch) float32
    """

    # Basic checks
    if not hasattr(W, "index") or not hasattr(W, "columns"):
        raise TypeError("W must be a pandas DataFrame with .index (genes) and .columns (archetypes).")

    coords = np.asarray(coords)
    gene_labels = np.asarray(gene_labels)
    if gene_labels.shape[0] != coords.shape[0]:
        raise ValueError("gene_labels must have the same length as coords (N transcripts).")

    height, width, _ = _prepare_spatial_grid(coords, bin_size)
    P = height * width

    n_arch = W.shape[1]

    print(f"--- Spatial Projection (Radius={smoothing_radius}µm) ---")
    print(f"Grid size: {height} × {width}  (pixels={P})")
    print(f"Archetypes: {n_arch}")

    # ------------------------------------------------------------------
    # 2. Choose which genes to build maps for
    # ------------------------------------------------------------------
    gene_index, genes_use, mean_g, std_g = _resolve_normalization_context(
        W,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
        normalization_stats=normalization_stats,
    )

    if len(genes_use) == 0:
        print("⚠️ No genes selected for mapping (check weight_threshold).")
        niche_maps = np.zeros((height, width, n_arch), dtype=np.float32)
        if return_stats:
            stats = _build_normalization_stats_dict(
                genes_use,
                np.array([]),
                np.array([]),
                weight_threshold=weight_threshold,
                union_driver_genes=union_driver_genes,
                selected_genes=selected_genes,
                n_slices=1,
                n_pooled_pixels=P,
            )
            return niche_maps, stats
        return niche_maps

    gene_ids = gene_index.get_indexer(gene_labels).astype(np.int32)
    if not np.any(gene_ids >= 0):
        print("⚠️ None of the selected genes appear in gene_labels.")
        niche_maps = np.zeros((height, width, n_arch), dtype=np.float32)
        if return_stats:
            if mean_g is None:
                mean_g = np.zeros(len(genes_use), dtype=np.float32)
                std_g = np.zeros(len(genes_use), dtype=np.float32)
            stats = _build_normalization_stats_dict(
                genes_use,
                mean_g,
                std_g,
                weight_threshold=weight_threshold,
                union_driver_genes=union_driver_genes,
                selected_genes=selected_genes,
                n_slices=1,
                n_pooled_pixels=P,
            )
            return niche_maps, stats
        return niche_maps

    blurred, _, _ = _build_blurred_gene_maps(
        coords,
        gene_labels,
        gene_index,
        bin_size,
        smoothing_radius,
    )

    # ------------------------------------------------------------------
    # 5. Z-score each gene map ONCE
    # ------------------------------------------------------------------
    if mean_g is None:
        mean_g = blurred.mean(axis=(0, 1))
        std_g = blurred.std(axis=(0, 1))

    # ------------------------------------------------------------------
    # 6. Weighted sums for ALL archetypes (cheap)
    # ------------------------------------------------------------------
    W_sub = _build_weight_matrix(W, genes_use, weight_threshold)

    niche_maps = _compute_slice_niche_maps(blurred, W_sub, mean_g, std_g, eps)
    if return_stats:
        stats = _build_normalization_stats_dict(
            genes_use,
            mean_g,
            std_g,
            weight_threshold=weight_threshold,
            union_driver_genes=union_driver_genes,
            selected_genes=selected_genes,
            n_slices=1,
            n_pooled_pixels=P,
        )
        return niche_maps, stats
    return niche_maps


def compute_niche_map_normalization_stats_across_slices(
    coords_list,
    gene_labels_list,
    W,
    bin_size=8.0,
    smoothing_radius=8.0,
    weight_threshold=0.3,
    union_driver_genes=True,
    selected_genes=None,
):
    """Compute reusable pooled gene-wise normalization statistics across slices."""
    if not hasattr(W, "index") or not hasattr(W, "columns"):
        raise TypeError("W must be a pandas DataFrame with .index (genes) and .columns (archetypes).")

    if len(coords_list) != len(gene_labels_list):
        raise ValueError("coords_list and gene_labels_list must have the same length.")
    if len(coords_list) == 0:
        raise ValueError("coords_list must contain at least one slice.")

    coords_list = [np.asarray(coords) for coords in coords_list]
    gene_labels_list = [np.asarray(gene_labels) for gene_labels in gene_labels_list]
    for slice_idx, (coords, gene_labels) in enumerate(zip(coords_list, gene_labels_list)):
        if gene_labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"gene_labels_list[{slice_idx}] must have the same length as coords_list[{slice_idx}]."
            )

    _, genes_use, _, _ = _resolve_normalization_context(
        W,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
        normalization_stats=None,
    )

    if len(genes_use) == 0:
        pooled_pixels = sum(_prepare_spatial_grid(coords, bin_size)[0] * _prepare_spatial_grid(coords, bin_size)[1] for coords in coords_list)
        return _build_normalization_stats_dict(
            genes_use,
            np.array([]),
            np.array([]),
            weight_threshold=weight_threshold,
            union_driver_genes=union_driver_genes,
            selected_genes=selected_genes,
            n_slices=len(coords_list),
            n_pooled_pixels=pooled_pixels,
        )

    gene_index = W.index.__class__(genes_use)
    pooled_sum = np.zeros(len(genes_use), dtype=np.float64)
    pooled_sumsq = np.zeros(len(genes_use), dtype=np.float64)
    pooled_pixels = 0

    for coords, gene_labels in zip(coords_list, gene_labels_list):
        blurred, height, width = _build_blurred_gene_maps(
            coords,
            gene_labels,
            gene_index,
            bin_size,
            smoothing_radius,
        )
        blurred64 = blurred.astype(np.float64, copy=False)
        pooled_sum += blurred64.sum(axis=(0, 1))
        pooled_sumsq += np.square(blurred64).sum(axis=(0, 1))
        pooled_pixels += height * width

    mean_g = pooled_sum / max(pooled_pixels, 1)
    var_g = pooled_sumsq / max(pooled_pixels, 1) - np.square(mean_g)
    std_g = np.sqrt(np.maximum(var_g, 0.0))

    return _build_normalization_stats_dict(
        genes_use,
        mean_g,
        std_g,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
        n_slices=len(coords_list),
        n_pooled_pixels=pooled_pixels,
    )


def create_niche_maps_by_archetype_pooled_across_slices(
    coords_list,
    gene_labels_list,
    W,
    bin_size=8.0,
    smoothing_radius=8.0,
    weight_threshold=0.3,
    union_driver_genes=True,
    selected_genes=None,
    eps=1e-9,
    return_stats=False,
):
    """
    Compute per-slice niche maps using a shared gene panel and pooled gene-wise
    normalization across all slices.

    This keeps each slice on its own spatial grid, but makes the resulting niche
    scores directly comparable by computing one mean/std per gene from the full
    collection of blurred slice maps.

    Returns
    -------
    niche_maps_list : list[np.ndarray]
        List of per-slice niche maps, each shaped (height, width, n_arch).
    stats : dict, optional
        Returned when ``return_stats`` is True. Contains the shared genes and
        pooled normalization statistics used for all slices.
    """
    if not hasattr(W, "index") or not hasattr(W, "columns"):
        raise TypeError("W must be a pandas DataFrame with .index (genes) and .columns (archetypes).")

    if len(coords_list) != len(gene_labels_list):
        raise ValueError("coords_list and gene_labels_list must have the same length.")
    if len(coords_list) == 0:
        raise ValueError("coords_list must contain at least one slice.")

    coords_list = [np.asarray(coords) for coords in coords_list]
    gene_labels_list = [np.asarray(gene_labels) for gene_labels in gene_labels_list]
    for slice_idx, (coords, gene_labels) in enumerate(zip(coords_list, gene_labels_list)):
        if gene_labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"gene_labels_list[{slice_idx}] must have the same length as coords_list[{slice_idx}]."
            )

    n_arch = W.shape[1]
    print(f"--- Pooled Spatial Projection (Radius={smoothing_radius}µm) ---")
    print(f"Slices: {len(coords_list)}")
    print(f"Archetypes: {n_arch}")

    stats = compute_niche_map_normalization_stats_across_slices(
        coords_list,
        gene_labels_list,
        W,
        bin_size=bin_size,
        smoothing_radius=smoothing_radius,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
    )
    genes_use = stats["genes_use"]

    if len(genes_use) == 0:
        niche_maps_list = []
        for coords in coords_list:
            height, width, _ = _prepare_spatial_grid(coords, bin_size)
            niche_maps_list.append(np.zeros((height, width, n_arch), dtype=np.float32))
        if return_stats:
            return niche_maps_list, stats
        return niche_maps_list

    gene_index = W.index.__class__(genes_use)
    W_sub = _build_weight_matrix(W, genes_use, weight_threshold)
    mean_g = stats["mean_g"]
    std_g = stats["std_g"]

    niche_maps_list = []
    for coords, gene_labels in zip(coords_list, gene_labels_list):
        blurred, _, _ = _build_blurred_gene_maps(
            coords,
            gene_labels,
            gene_index,
            bin_size,
            smoothing_radius,
        )
        niche_maps_list.append(_compute_slice_niche_maps(blurred, W_sub, mean_g, std_g, eps))

    if return_stats:
        stats = dict(stats)
        stats["weight_matrix"] = W_sub
        return niche_maps_list, stats

    return niche_maps_list


def compute_niche_map_normalization_stats_across_slices_streaming(
    slice_iterator_factory,
    W,
    bin_size=8.0,
    smoothing_radius=8.0,
    weight_threshold=0.3,
    union_driver_genes=True,
    selected_genes=None,
):
    """Compute reusable pooled gene-wise normalization statistics from a streaming slice source."""
    if not hasattr(W, "index") or not hasattr(W, "columns"):
        raise TypeError("W must be a pandas DataFrame with .index (genes) and .columns (archetypes).")
    if not callable(slice_iterator_factory):
        raise TypeError("slice_iterator_factory must be callable and return a fresh iterable per pass.")

    _, genes_use, _, _ = _resolve_normalization_context(
        W,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
        normalization_stats=None,
    )

    gene_index = W.index.__class__(genes_use)
    pooled_sum = np.zeros(len(genes_use), dtype=np.float64)
    pooled_sumsq = np.zeros(len(genes_use), dtype=np.float64)
    pooled_pixels = 0
    slice_count = 0

    for slice_item in slice_iterator_factory():
        coords, gene_labels, _ = _normalize_slice_input(slice_item)
        if len(genes_use) == 0:
            height, width, _ = _prepare_spatial_grid(coords, bin_size)
        else:
            blurred, height, width = _build_blurred_gene_maps(
                coords,
                gene_labels,
                gene_index,
                bin_size,
                smoothing_radius,
            )
            blurred64 = blurred.astype(np.float64, copy=False)
            pooled_sum += blurred64.sum(axis=(0, 1))
            pooled_sumsq += np.square(blurred64).sum(axis=(0, 1))
        pooled_pixels += height * width
        slice_count += 1

    if slice_count == 0:
        raise ValueError("slice_iterator_factory produced no slices.")

    if len(genes_use) == 0:
        return _build_normalization_stats_dict(
            genes_use,
            np.array([]),
            np.array([]),
            weight_threshold=weight_threshold,
            union_driver_genes=union_driver_genes,
            selected_genes=selected_genes,
            n_slices=slice_count,
            n_pooled_pixels=pooled_pixels,
        )

    mean_g = pooled_sum / max(pooled_pixels, 1)
    var_g = pooled_sumsq / max(pooled_pixels, 1) - np.square(mean_g)
    std_g = np.sqrt(np.maximum(var_g, 0.0))

    return _build_normalization_stats_dict(
        genes_use,
        mean_g,
        std_g,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
        n_slices=slice_count,
        n_pooled_pixels=pooled_pixels,
    )


def create_niche_maps_by_archetype_pooled_across_slices_streaming(
    slice_iterator_factory,
    W,
    bin_size=8.0,
    smoothing_radius=8.0,
    weight_threshold=0.3,
    union_driver_genes=True,
    selected_genes=None,
    eps=1e-9,
    return_stats=False,
    output_callback=None,
):
    """
    Compute pooled per-slice niche maps without holding all slice inputs in memory.

    Parameters
    ----------
    slice_iterator_factory : callable
        Zero-argument callable that returns a fresh iterable on each call. Each
        item must be either ``(coords, gene_labels)``,
        ``(coords, gene_labels, slice_name)``, or a dict containing ``coords``
        and ``gene_labels``.
    output_callback : callable, optional
        If provided, called as ``output_callback(niche_map, slice_name, slice_idx)``
        during the second pass. When set, niche maps are not stored in memory.

    Returns
    -------
    niche_maps_list : list[np.ndarray], optional
        Returned when ``output_callback`` is None.
    stats : dict, optional
        Returned when ``return_stats`` is True.
    """
    if not hasattr(W, "index") or not hasattr(W, "columns"):
        raise TypeError("W must be a pandas DataFrame with .index (genes) and .columns (archetypes).")
    if not callable(slice_iterator_factory):
        raise TypeError("slice_iterator_factory must be callable and return a fresh iterable per pass.")

    n_arch = W.shape[1]
    print(f"--- Streaming Pooled Spatial Projection (Radius={smoothing_radius}µm) ---")
    print(f"Archetypes: {n_arch}")

    stats = compute_niche_map_normalization_stats_across_slices_streaming(
        slice_iterator_factory,
        W,
        bin_size=bin_size,
        smoothing_radius=smoothing_radius,
        weight_threshold=weight_threshold,
        union_driver_genes=union_driver_genes,
        selected_genes=selected_genes,
    )
    genes_use = stats["genes_use"]

    gene_index = W.index.__class__(genes_use)
    W_sub = _build_weight_matrix(W, genes_use, weight_threshold)
    slice_count = stats["n_slices"]
    pooled_pixels = stats["n_pooled_pixels"]
    print(f"Slices: {slice_count}")

    if len(genes_use) == 0:
        niche_maps_list = [] if output_callback is None else None
        for slice_idx, slice_item in enumerate(slice_iterator_factory()):
            coords, _, slice_name = _normalize_slice_input(slice_item)
            height, width, _ = _prepare_spatial_grid(coords, bin_size)
            niche_map = np.zeros((height, width, n_arch), dtype=np.float32)
            if output_callback is None:
                niche_maps_list.append(niche_map)
            else:
                output_callback(niche_map, slice_name, slice_idx)
        if return_stats:
            stats = dict(stats)
            stats["weight_matrix"] = W_sub
            if output_callback is not None:
                return stats
            return niche_maps_list, stats
        return niche_maps_list

    mean_g = stats["mean_g"]
    std_g = stats["std_g"]

    niche_maps_list = [] if output_callback is None else None
    for slice_idx, slice_item in enumerate(slice_iterator_factory()):
        coords, gene_labels, slice_name = _normalize_slice_input(slice_item)
        blurred, _, _ = _build_blurred_gene_maps(
            coords,
            gene_labels,
            gene_index,
            bin_size,
            smoothing_radius,
        )
        niche_map = _compute_slice_niche_maps(blurred, W_sub, mean_g, std_g, eps)
        if output_callback is None:
            niche_maps_list.append(niche_map)
        else:
            output_callback(niche_map, slice_name, slice_idx)

    if return_stats:
        stats = dict(stats)
        stats["weight_matrix"] = W_sub
        if output_callback is not None:
            return stats
        return niche_maps_list, stats

    return niche_maps_list




