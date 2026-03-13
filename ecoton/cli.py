"""Command-line interface for running the Ecoton workflow."""

import argparse
import os
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytic_metatranscripts import analytic_null_metatranscripts
from .compute_metatranscripts import ComputeMetatranscripts
from .compute_niche_maps import create_niche_maps_by_archetype_all_at_once
from .plot_colocalization_modules import plot_colocalization_modules
from .process_colocalization_graph import ProcessColocalizationGraph

try:
    import psutil
except ImportError:
    psutil = None


def init_runtime_tracking() -> dict:
    return {
        "script_start_perf_counter": time.perf_counter(),
        "script_start_unix": time.time(),
        "steps": {},
        "memory_tracking": {
            "method": (
                "psutil.Process().memory_info().rss"
                if psutil is not None
                else "unavailable (install psutil)"
            )
        },
    }


def run_timed(runtime_tracking, step_name, func, *args, **kwargs):
    """Run a callable and record runtime + RAM usage in runtime_tracking."""
    proc = psutil.Process(os.getpid()) if psutil is not None else None

    mem_before_bytes = None
    mem_after_bytes = None
    mem_delta_bytes = None
    mem_peak_bytes = None

    stop_evt = threading.Event()

    def _sample_peak_memory():
        nonlocal mem_peak_bytes
        while not stop_evt.is_set():
            rss = proc.memory_info().rss
            if mem_peak_bytes is None or rss > mem_peak_bytes:
                mem_peak_bytes = rss
            stop_evt.wait(0.05)

    sampler = None
    if proc is not None:
        mem_before_bytes = proc.memory_info().rss
        mem_peak_bytes = mem_before_bytes
        sampler = threading.Thread(target=_sample_peak_memory, daemon=True)
        sampler.start()

    t0 = time.perf_counter()
    try:
        out = func(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        if proc is not None:
            stop_evt.set()
            if sampler is not None:
                sampler.join(timeout=0.2)
            mem_after_bytes = proc.memory_info().rss
            if mem_peak_bytes is None or mem_after_bytes > mem_peak_bytes:
                mem_peak_bytes = mem_after_bytes
            mem_delta_bytes = mem_after_bytes - mem_before_bytes

    runtime_tracking["steps"][step_name] = {
        "seconds": dt,
        "minutes": dt / 60.0,
        "ram_before_mb": None if mem_before_bytes is None else mem_before_bytes / (1024 ** 2),
        "ram_after_mb": None if mem_after_bytes is None else mem_after_bytes / (1024 ** 2),
        "ram_delta_mb": None if mem_delta_bytes is None else mem_delta_bytes / (1024 ** 2),
        "ram_peak_mb": None if mem_peak_bytes is None else mem_peak_bytes / (1024 ** 2),
    }
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Ecoton workflow in script mode.")
    parser.add_argument(
        "--transcripts-path",
        type=Path,
        default=Path("../spatial_5k/data/transcripts.parquet"),
        help="Path to transcripts file.",
    )
    parser.add_argument(
        "--transcripts-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Format of transcripts input file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory for output artifacts.",
    )
    parser.add_argument("--mode", type=str, default="xenium_5k")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--organism", type=str, default="Human")
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--smoothing-radius", type=float, default=8.0)
    parser.add_argument("--weight-threshold", type=float, default=0.3)
    parser.add_argument("--min-points", "--min_points", dest="min_points", type=int, default=3)
    parser.add_argument(
        "--save-module-plot",
        action="store_true",
        help="Save selected archetypal module plot to output dir.",
    )
    return parser.parse_args(argv)


def run_workflow(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)

    runtime_tracking = init_runtime_tracking()

    print(f"Loading transcripts: {args.transcripts_path}")
    if args.transcripts_format == "parquet":
        load_step = "load_transcripts_parquet"
        load_fn = pd.read_parquet
    else:
        load_step = "load_transcripts_csv"
        load_fn = pd.read_csv

    transcripts_df = run_timed(
        runtime_tracking,
        load_step,
        load_fn,
        args.transcripts_path,
    )

    print("Computing metatranscripts")
    results = run_timed(
        runtime_tracking,
        "compute_metatranscripts",
        ComputeMetatranscripts,
        transcripts_df,
        mode=args.mode,
        min_points=args.min_points,
    )

    print("Computing analytic null metatranscripts")
    analytic_response = run_timed(
        runtime_tracking,
        "analytic_null_metatranscripts",
        analytic_null_metatranscripts,
        results["metatranscripts"],
        gene_col_meta=results["gene_name"],
        verbose=True,
    )

    print("Processing colocalization graph")
    colocalization_response = run_timed(
        runtime_tracking,
        "process_colocalization_graph",
        ProcessColocalizationGraph,
        G_ig=analytic_response[2],
        K=args.k,
        seed=args.seed,
        run_enrich=True,
        organism=args.organism,
    )

    if args.save_module_plot:
        print("Saving module plot")
        module_labels = np.arange(args.k).astype(str).tolist()
        nums = list(range(args.k))
        ax = plot_colocalization_modules(
            analytic_response[2],
            module_labels,
            nums,
            title="Selected Archetypal Modules",
        )
        fig = ax.figure
        fig.savefig(args.output_dir / "selected_archetypal_modules.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    transcripts = results["filtered_transcripts"]
    coords = transcripts[[results["x_coord"], results["y_coord"]]].values.astype(np.float32)
    gene_labels = transcripts[results["gene_name"]].values
    W_df = colocalization_response["W_df"]

    print("Creating niche maps")
    niche_maps = run_timed(
        runtime_tracking,
        "create_niche_maps_by_archetype_all_at_once",
        create_niche_maps_by_archetype_all_at_once,
        coords=coords,
        gene_labels=gene_labels,
        W=W_df,
        bin_size=args.bin_size,
        smoothing_radius=args.smoothing_radius,
        weight_threshold=args.weight_threshold,
    )

    runtime_tracking["total_seconds_through_niche_maps"] = (
        time.perf_counter() - runtime_tracking["script_start_perf_counter"]
    )
    runtime_tracking["total_minutes_through_niche_maps"] = (
        runtime_tracking["total_seconds_through_niche_maps"] / 60.0
    )

    workflow_out = {
        "results": results,
        "analytic_response": analytic_response,
        "colocalization_response": colocalization_response,
        "niche_maps": niche_maps,
    }

    workflow_pickle = args.output_dir / "workflow.pkl"
    runtime_pickle = args.output_dir / "runtime_tracking.pkl"

    print(f"Pickling workflow outputs: {workflow_pickle}")
    with workflow_pickle.open("wb") as f:
        pickle.dump(workflow_out, f)

    print(f"Pickling runtime tracking: {runtime_pickle}")
    with runtime_pickle.open("wb") as f:
        pickle.dump(runtime_tracking, f)

    print("Done")
    return workflow_out, runtime_tracking


def main(argv=None):
    args = parse_args(argv)
    run_workflow(args)
