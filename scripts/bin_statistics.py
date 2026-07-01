"""Persist the full ecoton.bin_transcripts response and niche-level stats.

This script loads ``workflow.pkl`` produced by ``ecoton.cli``, runs
``ecoton.bin_transcripts(...)`` on ``results['filtered_transcripts']``, and
saves the returned response dict for downstream analysis. When ``niche_maps``
are present, it also computes per-niche cell subsets and assigned/unassigned
transcript proportions.

Outputs
-------
- bin_transcripts_response.pkl
- niche_cell_subsets.pkl
- niche_assignment_stats.csv
- niche_component_stats.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import ecoton


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run ecoton.bin_transcripts on workflow.pkl and save the full response"
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
        default=Path("processed_data/bin_statistics"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="bin_transcripts_response.pkl",
        help="Filename for the pickled ecoton.bin_transcripts response",
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
        "--component-connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="Connectivity used for niche connected components (4 or 8)",
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
        "--keep-empty-bins",
        action="store_true",
        help="Preserve empty bins in returned matrices",
    )
    parser.add_argument(
        "--return-matrix",
        action="store_true",
        help="Include the classic bin x gene matrix in the response",
    )
    parser.add_argument(
        "--no-return-matrix-split-assignment",
        action="store_true",
        help="Do not include the split assigned/unassigned matrix in the response",
    )
    parser.add_argument(
        "--no-return-cells",
        action="store_true",
        help="Do not include cells_by_bin and bin_counts in the response",
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
    if "results" not in workflow:
        raise KeyError("workflow.pkl must contain key 'results'")

    results = workflow["results"]
    niche_maps = workflow.get("niche_maps")

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
    return_matrix = args.return_matrix
    return_matrix_split_assignment = not args.no_return_matrix_split_assignment
    return_cells = not args.no_return_cells

    if args.verbose:
        print(f"Loading workflow: {args.workflow_pkl}")
        print(f"transcripts shape: {transcripts.shape}")
        if niche_maps is not None:
            print(f"niche_maps shape: {np.asarray(niche_maps).shape}")
        print(f"bin_size: {args.bin_size}")
        print(f"threshold: {threshold}")
        print(f"n_niches: {args.n_niches}")
        print(f"component_connectivity: {args.component_connectivity}")
        print(f"unassigned_tokens: {unassigned_tokens}")
        print(f"keep_empty_bins: {args.keep_empty_bins}")
        print(f"return_matrix: {return_matrix}")
        print(f"return_matrix_split_assignment: {return_matrix_split_assignment}")
        print(f"return_cells: {return_cells}")

    resp = ecoton.bin_transcripts(
        transcripts,
        bin_size=args.bin_size,
        x_col=x_col,
        y_col=y_col,
        cell_col=args.cell_col,
        gene_col=gene_col,
        unassigned_tokens=unassigned_tokens,
        keep_empty_bins=args.keep_empty_bins,
        return_matrix=return_matrix,
        return_matrix_split_assignment=return_matrix_split_assignment,
        return_cells=return_cells,
        verbose=args.verbose,
    )

    output_path = args.output_dir / args.output_name
    with output_path.open("wb") as f:
        pickle.dump(resp, f)

    print(f"Saved bin_transcripts response: {output_path}")

    can_compute_niche_stats = (
        niche_maps is not None and return_matrix_split_assignment and return_cells
    )
    if not can_compute_niche_stats:
        if args.verbose:
            print(
                "Skipping niche cell statistics because one or more prerequisites are missing: "
                "workflow['niche_maps'], return_matrix_split_assignment=True, return_cells=True"
            )
        return

    niche_maps = np.asarray(niche_maps)
    K_available = niche_maps.shape[2]
    K = min(max(0, int(args.n_niches)), K_available)

    if K == 0:
        raise ValueError(f"No niches to process. n_niches={args.n_niches}, available={K_available}")

    cell_subsets = {}
    rows = []
    component_rows = []
    grid_meta_for_niche = {
        **resp["grid_meta"],
        "height": int(niche_maps.shape[0]),
        "width": int(niche_maps.shape[1]),
    }

    for k in range(K):
        stats = ecoton.niche_unassigned_transcript_stats(
            niche_maps=niche_maps,
            grid_meta=resp["grid_meta"],
            binning_output=resp,
            k=k,
            threshold=threshold,
            return_selected_bin_ids=True,
        )

        selected_bin_ids = stats.pop("selected_bin_ids")
        cells_subset = ecoton.cells_in_selected_bins(selected_bin_ids, resp["cells_by_bin"])
        cell_subsets[k] = sorted(cells_subset)

        _, component_summary = ecoton.connected_components_from_selected_bins(
            selected_bin_ids,
            grid_meta=grid_meta_for_niche,
            connectivity=args.component_connectivity,
        )

        n_connected_components = int(component_summary.shape[0])
        if n_connected_components > 0:
            largest_component_id = int(component_summary["component_n_bins"].idxmax())
            largest_component_n_bins = int(component_summary.loc[largest_component_id, "component_n_bins"])
            largest_component_fraction = (
                float(largest_component_n_bins / stats["n_selected_bins"])
                if stats["n_selected_bins"] > 0
                else np.nan
            )
            mean_component_n_bins = float(component_summary["component_n_bins"].mean())
            largest_component_extent = float(component_summary.loc[largest_component_id, "extent"])
            largest_component_compactness = float(component_summary.loc[largest_component_id, "compactness"])
        else:
            largest_component_id = np.nan
            largest_component_n_bins = 0
            largest_component_fraction = np.nan
            mean_component_n_bins = np.nan
            largest_component_extent = np.nan
            largest_component_compactness = np.nan

        if n_connected_components > 0:
            component_summary_reset = component_summary.reset_index()
            for _, component_row in component_summary_reset.iterrows():
                component_rows.append(
                    {
                        "niche_k": int(k),
                        "threshold_value": float(stats["threshold_value"]),
                        "component_id": int(component_row["component_id"]),
                        "component_n_bins": int(component_row["component_n_bins"]),
                        "bbox_width_bins": int(component_row["bbox_width_bins"]),
                        "bbox_height_bins": int(component_row["bbox_height_bins"]),
                        "bbox_area_bins": int(component_row["bbox_area_bins"]),
                        "extent": float(component_row["extent"]),
                        "perimeter_edges": int(component_row["perimeter_edges"]),
                        "compactness": float(component_row["compactness"]),
                        "centroid_x_bin": float(component_row["centroid_x_bin"]),
                        "centroid_y_bin": float(component_row["centroid_y_bin"]),
                        "aspect_ratio": float(component_row["aspect_ratio"]),
                        "touches_border": bool(component_row["touches_border"]),
                    }
                )

        rows.append(
            {
                "niche_k": int(k),
                "threshold_value": float(stats["threshold_value"]),
                "n_selected_bins": int(stats["n_selected_bins"]),
                "n_connected_components": n_connected_components,
                "largest_component_id": largest_component_id,
                "largest_component_n_bins": largest_component_n_bins,
                "largest_component_fraction": largest_component_fraction,
                "mean_component_n_bins": mean_component_n_bins,
                "largest_component_extent": largest_component_extent,
                "largest_component_compactness": largest_component_compactness,
                "n_cells_subset": int(len(cells_subset)),
                "assigned_transcripts": int(stats["assigned_transcripts"]),
                "unassigned_transcripts": int(stats["unassigned_transcripts"]),
                "total_transcripts": int(stats["total_transcripts"]),
                "prop_assigned": float(stats["prop_assigned"]) if np.isfinite(stats["prop_assigned"]) else np.nan,
                "prop_unassigned": float(stats["prop_unassigned"]) if np.isfinite(stats["prop_unassigned"]) else np.nan,
            }
        )

        if args.verbose:
            print(
                f"k={k:02d} bins={stats['n_selected_bins']:6d} comps={n_connected_components:4d} "
                f"cells={len(cells_subset):6d} "
                f"assigned={stats['assigned_transcripts']:8d} unassigned={stats['unassigned_transcripts']:8d} "
                f"prop_unassigned={stats['prop_unassigned']:.4f}"
            )

    stats_df = pd.DataFrame(rows).sort_values("niche_k").reset_index(drop=True)
    component_stats_df = pd.DataFrame(component_rows).sort_values(["niche_k", "component_id"]).reset_index(drop=True)

    subsets_path = args.output_dir / "niche_cell_subsets.pkl"
    stats_csv_path = args.output_dir / "niche_assignment_stats.csv"
    component_stats_csv_path = args.output_dir / "niche_component_stats.csv"

    with subsets_path.open("wb") as f:
        pickle.dump(cell_subsets, f)

    stats_df.to_csv(stats_csv_path, index=False)
    component_stats_df.to_csv(component_stats_csv_path, index=False)

    print(f"Saved cell subsets: {subsets_path}")
    print(f"Saved stats table:  {stats_csv_path}")
    print(f"Saved component stats: {component_stats_csv_path}")


if __name__ == "__main__":
    main()