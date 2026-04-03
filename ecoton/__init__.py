from importlib import import_module

__version__ = "0.2.4"
__all__ = [
    "ComputeMetatranscripts",
    
    "ProcessColocalizationGraph",
    "ComputeCellOverlaps",
    "bin_transcripts",
    "bins_from_niche_threshold",
    "cells_in_selected_bins",
    "knee_from_sorted_curve",
    "analytic_null_metatranscripts",
    "combine_O_matrices_stouffer",
    "stats_df_to_igraph",
    "plot_colocalization_modules",
    "plot_gene_program_from_W",
    "plot_graph_umap",
    "compute_graph_umap",
    "plot_niche_continuous_and_binary",
    "plot_niche_continuous_only",
    "plot_niche_percentile_categories_only",
    "plot_niche_continuous_and_percentile_categories",
    "create_niche_maps_by_archetype_all_at_once",
    "enrichr_with_local_gmt",
]

_EXPORT_TO_MODULE = {
    "ComputeMetatranscripts": ".compute_metatranscripts",
    
    "ProcessColocalizationGraph": ".process_colocalization_graph",
    "ComputeCellOverlaps": ".compute_cell_overlaps",
    "bin_transcripts": ".statistics_niche_maps",
    "bins_from_niche_threshold": ".statistics_niche_maps",
    "cells_in_selected_bins": ".statistics_niche_maps",
    "knee_from_sorted_curve": ".statistics_niche_maps",
    "analytic_null_metatranscripts": ".analytic_metatranscripts",
    "combine_O_matrices_stouffer": ".analytic_metatranscripts",
    "stats_df_to_igraph": ".analytic_metatranscripts",
    "plot_colocalization_modules": ".plot_colocalization_modules",
    "plot_gene_program_from_W": ".plot_colocalization_modules",
    "plot_graph_umap": ".plot_graph_umap",
    "compute_graph_umap": ".plot_graph_umap",
    "plot_niche_continuous_and_binary": ".plot_niche_maps",
    "plot_niche_continuous_only": ".plot_niche_maps",
    "plot_niche_percentile_categories_only": ".plot_niche_maps",
    "plot_niche_continuous_and_percentile_categories": ".plot_niche_maps",
    "create_niche_maps_by_archetype_all_at_once": ".compute_niche_maps",
    "enrichr_with_local_gmt": ".enrichment",
}


def __getattr__(name):
    module_path = _EXPORT_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))

