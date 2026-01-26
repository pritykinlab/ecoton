"""
Efficient niche map computation utilities.

Provides functions to build blurred, z-scored niche maps per archetype
from transcript coordinates, gene labels, and archetype weight matrix `W`.
"""

import numpy as np
from scipy import ndimage, sparse


def create_niche_maps_by_archetype_all_at_once(
    coords,             # (N,2) float32
    gene_labels,        # (N,) array-like of str (gene name for each transcript)
    W,                  # pandas.DataFrame: index=gene labels, columns=archetypes
    bin_size=10.0,
    smoothing_radius=30.0,
    weight_threshold=0.1,
    union_driver_genes=True,   # if True, only compute maps for genes used by any archetype
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

    gene_labels = np.asarray(gene_labels)
    if gene_labels.shape[0] != coords.shape[0]:
        raise ValueError("gene_labels must have the same length as coords (N transcripts).")

    # ------------------------------------------------------------------
    # 1. Spatial Grid
    # ------------------------------------------------------------------
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    width  = int(np.ceil((x_max - x_min) / bin_size)) + 1
    height = int(np.ceil((y_max - y_min) / bin_size)) + 1

    grid_x = ((coords[:, 0] - x_min) / bin_size).astype(np.int32)
    grid_y = ((coords[:, 1] - y_min) / bin_size).astype(np.int32)
    grid_x = np.clip(grid_x, 0, width - 1)
    grid_y = np.clip(grid_y, 0, height - 1)

    flat_pixels = grid_y * width + grid_x
    P = height * width
    sigma_pixels = smoothing_radius / bin_size

    n_arch = W.shape[1]

    print(f"--- Spatial Projection (Radius={smoothing_radius}µm) ---")
    print(f"Grid size: {height} × {width}  (pixels={P})")
    print(f"Archetypes: {n_arch}")

    # ------------------------------------------------------------------
    # 2. Choose which genes to build maps for
    # ------------------------------------------------------------------
    if union_driver_genes:
        # union of genes with any weight above threshold in any archetype
        driver_mask = (W.values > weight_threshold).any(axis=1)
        genes_use = W.index[driver_mask]
        print(f"Using union of driver genes: {len(genes_use)} / {len(W.index)}")
    else:
        genes_use = W.index
        print(f"Using all genes in W: {len(genes_use)}")

    if len(genes_use) == 0:
        print("⚠️ No genes selected for mapping (check weight_threshold).")
        return np.zeros((height, width, n_arch), dtype=np.float32)

    # Map transcripts -> gene column index (only for selected genes)
    # Use pandas Index.get_indexer (fast) but keep it generic:
    gene_index = W.index.__class__(genes_use)  # Index of same type
    gene_ids = gene_index.get_indexer(gene_labels).astype(np.int32)  # -1 if not present

    ok = gene_ids >= 0
    if not np.any(ok):
        print("⚠️ None of the selected genes appear in gene_labels.")
        return np.zeros((height, width, n_arch), dtype=np.float32)

    pix = flat_pixels[ok]
    gid = gene_ids[ok]
    G = len(genes_use)

    # ------------------------------------------------------------------
    # 3. Build pixel × gene count matrix ONCE
    # ------------------------------------------------------------------
    # (pix, gid) count each transcript
    pixel_gene = sparse.coo_matrix(
        (np.ones(len(pix), dtype=np.float32), (pix, gid)),
        shape=(P, G)
    ).toarray()  # dense because we need gaussian blur; keep float32

    # Reshape to (H, W, G)
    gene_maps = pixel_gene.reshape(height, width, G)

    # ------------------------------------------------------------------
    # 4. Blur ALL genes in one pass
    # ------------------------------------------------------------------
    # Blur only spatial axes; leave gene axis untouched
    blurred = ndimage.gaussian_filter(
        gene_maps,
        sigma=(sigma_pixels, sigma_pixels, 0.0),
        mode="constant"
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Z-score each gene map ONCE
    # ------------------------------------------------------------------
    # Compute mean/std over spatial dims for each gene
    # shape: (G,)
    mean_g = blurred.mean(axis=(0, 1))
    std_g  = blurred.std(axis=(0, 1))

    # Avoid divide-by-zero: genes with ~0 variance get zeroed
    good = std_g > eps

    norm = np.zeros_like(blurred, dtype=np.float32)
    norm[:, :, good] = (blurred[:, :, good] - mean_g[good]) / std_g[good]

    # ------------------------------------------------------------------
    # 6. Weighted sums for ALL archetypes (cheap)
    # ------------------------------------------------------------------
    # Build weights matrix aligned to genes_use
    # W_sub: (G, A)
    W_sub = W.loc[genes_use, :].to_numpy(dtype=np.float32, copy=False)

    # Apply threshold per archetype by zeroing small weights
    if weight_threshold is not None:
        W_sub = np.where(W_sub > weight_threshold, W_sub, 0.0).astype(np.float32)

    # Matrix multiply: (P, G) @ (G, A) -> (P, A)
    norm_flat = norm.reshape(P, G)
    niche_flat = norm_flat @ W_sub  # float32 matmul

    niche_maps = niche_flat.reshape(height, width, n_arch).astype(np.float32)
    return niche_maps




