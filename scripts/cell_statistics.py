"""Compute per-niche cell subsets and assignment stats from an ecoton workflow pickle.

This script loads `workflow.pkl` produced by `ecoton.cli`, bins transcripts,
and for each niche (default first 25) computes:
- selected bin IDs at a given niche threshold
- cell subset (union of cell IDs in selected bins)
- assigned / unassigned transcript counts and proportions

Outputs
-------
- niche_cell_subsets.pkl
- niche_assignment_stats.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import ecoton


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute per-niche cell subsets and assignment proportions from workflow.pkl"
    )
    parser.add_argument(
        "--workflow-pkl",
        type=Path,
        required=True,
        help="Path to workflow.pkl produced by ecoton.cli",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data/cell_statistics"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=8.0,
        help="Bin size used for niche maps in the original CLI run (µm)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="p98",
        help="Threshold string or numeric value used per niche (e.g. p98 or 1.5)",
    )
    parser.add_argument(
        "--n-niches",
        type=int,
        default=25,
        help="Number of niches to process (starting from k=0)",
    )
    parser.add_argument(
        "--cell-col",
        type=str,
        default="cell_id",
        help="Cell ID column in filtered_transcripts",
    )
    parser.add_argument(
        "--unassigned-token",
        action="append",
        default=["UNASSIGNED", "-1"],
        help="Token treated as unassigned; can be passed multiple times",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details",
    )
    return parser.parse_args(argv)


def _coerce_threshold(threshold_raw: str):
    if isinstance(threshold_raw, str) and threshold_raw.lower().startswith("p"):
        return threshold_raw.lower()
    try:
        return float(threshold_raw)
    except Exception:
        return threshold_raw


def _parse_unassigned_tokens(tokens):
    parsed = []
    for tok in tokens:
        try:
            parsed.append(int(tok))
            continue
        except Exception:
            pass
        parsed.append(tok)
    return tuple(parsed)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.workflow_pkl.open("rb") as f:
        workflow = pickle.load(f)

    if not isinstance(workflow, dict):
        raise TypeError("workflow.pkl must contain a dict")
    if "results" not in workflow or "niche_maps" not in workflow:
        raise KeyError("workflow.pkl must contain keys 'results' and 'niche_maps'")

    results = workflow["results"]
    niche_maps = np.asarray(workflow["niche_maps"])

    required_result_keys = ["filtered_transcripts", "x_coord", "y_coord", "gene_name"]
    missing = [k for k in required_result_keys if k not in results]
    if missing:
        raise KeyError(f"workflow['results'] missing required keys: {missing}")

    transcripts = results["filtered_transcripts"]
    x_col = results["x_coord"]
    y_col = results["y_coord"]
    gene_col = results["gene_name"]

    if args.cell_col not in transcripts.columns:
        raise KeyError(f"cell column '{args.cell_col}' not found in filtered_transcripts")

    threshold = _coerce_threshold(args.threshold)
    unassigned_tokens = _parse_unassigned_tokens(args.unassigned_token)

    if args.verbose:
        print(f"Loading workflow: {args.workflow_pkl}")
        print(f"transcripts shape: {transcripts.shape}")
        print(f"niche_maps shape: {niche_maps.shape}")
        print(f"bin_size: {args.bin_size}")
        print(f"threshold: {threshold}")
        print(f"unassigned_tokens: {unassigned_tokens}")

    resp = ecoton.bin_transcripts(
        transcripts,
        bin_size=args.bin_size,
        x_col=x_col,
        y_col=y_col,
        cell_col=args.cell_col,
        gene_col=gene_col,
        unassigned_tokens=unassigned_tokens,
        return_cells=True,
        return_matrix_split_assignment=True,
        return_matrix=False,
        verbose=args.verbose,
    )

    X_split = resp["X_split"]
    bin_index = resp["bin_index_split"]
    grid_meta = resp["grid_meta"]
    cells_by_bin = resp["cells_by_bin"]
    G = len(resp["gene_names_split"])

    K_available = niche_maps.shape[2]
    K = min(max(0, int(args.n_niches)), K_available)

    if K == 0:
        raise ValueError(f"No niches to process. n_niches={args.n_niches}, available={K_available}")

    grid_meta_for_niche = {**grid_meta, "height": niche_maps.shape[0], "width": niche_maps.shape[1]}

    cell_subsets = {}
    rows = []

    for k in range(K):
        selected_bin_ids, _, t = ecoton.bins_from_niche_threshold(
            niche_maps=niche_maps,
            grid_meta=grid_meta_for_niche,
            k=k,
            threshold=threshold,
        )

        selected_bins = set(map(int, selected_bin_ids))
        row_mask = np.isin(bin_index, list(selected_bins))
        X_sel = X_split[row_mask, :]

        assigned_ct = float(X_sel[:, :G].sum())
        unassigned_ct = float(X_sel[:, G:].sum())
        total_ct = assigned_ct + unassigned_ct

        prop_assigned = (assigned_ct / total_ct) if total_ct > 0 else np.nan
        prop_unassigned = (unassigned_ct / total_ct) if total_ct > 0 else np.nan

        cells_subset = ecoton.cells_in_selected_bins(selected_bin_ids, cells_by_bin)
        cell_subsets[k] = sorted(cells_subset)

        rows.append(
            {
                "niche_k": int(k),
                "threshold_value": float(t),
                "n_selected_bins": int(len(selected_bin_ids)),
                "n_cells_subset": int(len(cells_subset)),
                "assigned_transcripts": int(assigned_ct),
                "unassigned_transcripts": int(unassigned_ct),
                "total_transcripts": int(total_ct),
                "prop_assigned": float(prop_assigned) if np.isfinite(prop_assigned) else np.nan,
                "prop_unassigned": float(prop_unassigned) if np.isfinite(prop_unassigned) else np.nan,
            }
        )

        if args.verbose:
            print(
                f"k={k:02d} bins={len(selected_bin_ids):6d} cells={len(cells_subset):6d} "
                f"assigned={int(assigned_ct):8d} unassigned={int(unassigned_ct):8d} "
                f"prop_unassigned={prop_unassigned:.4f}"
            )

    stats_df = pd.DataFrame(rows).sort_values("niche_k").reset_index(drop=True)

    subsets_path = args.output_dir / "niche_cell_subsets.pkl"
    stats_csv_path = args.output_dir / "niche_assignment_stats.csv"

    with subsets_path.open("wb") as f:
        pickle.dump(cell_subsets, f)

    stats_df.to_csv(stats_csv_path, index=False)

    print(f"Saved cell subsets: {subsets_path}")
    print(f"Saved stats table:  {stats_csv_path}")


if __name__ == "__main__":
    main()