"""Local GMT-based enrichment utilities for ecoton.

This module provides functions to prepare local GMT pathway collections
and run a simple hypergeometric enrichment (no internet required).

Functions:
- prepare_pathways(): copy/clean GMT files into ecoton/pathways on demand.
- load_gmt(): parse a GMT file into a dict of name -> (desc, set_of_genes).
- run_enrich(): run hypergeometric enrichment for a gene list against
  local GMT files. Use `species='human'` to select files starting with
  `c5` or `h`, and `species='mouse'` to select files starting with `m`.

This keeps the package self-contained and avoids external Enrichr calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas should be present in envs that use ecoton
    pd = None


PKG_ROOT = Path(__file__).resolve().parents[1]
# package directory (the ecoton python package folder)
PKG_DIR = Path(__file__).resolve().parent
# store prepared pathway GMTs inside the package at `ecoton/resources/`
PATHWAYS_DIR = PKG_DIR / "resources"


def _ensure_pathways_dir() -> Path:
    PATHWAYS_DIR.mkdir(exist_ok=True)
    return PATHWAYS_DIR


def _source_gmt_files() -> List[Path]:
    """Return candidate source GMT files located in the package root.

    We look next to the package for files with extension `.gmt` or `.txt` that
    were downloaded from MSigDB. These are used as the source to prepare
    cleaned copies in `ecoton/pathways/`.
    """
    # source files live next to the top-level package folder (PKG_ROOT)
    # — these are the original downloaded files. We copy them into
    # `ecoton/resources/` on prepare_pathways().
    root = PKG_ROOT
    candidates = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in {".gmt", ".txt"}:
            # Heuristic: named like c5., h.all., m5., mh.all., etc.
            if p.name.lower().startswith(("c5", "h", "m", "mh")):
                candidates.append(p)
    return sorted(candidates)


def prepare_pathways(copy_sources: bool = True) -> Path:
    """Ensure `ecoton/resources/` exists and (optionally) copy & clean
    GMT files from the package root into it.

    The function performs minimal cleaning: strips whitespace, collapses
    multiple separators into tabs, and preserves the second column (URL/desc)
    if present.
    Returns the pathways directory path.
    """
    outdir = _ensure_pathways_dir()
    sources = _source_gmt_files()
    if not sources:
        return outdir

    for src in sources:
        dst = outdir / src.name
        if dst.exists():
            continue
        if not copy_sources:
            continue
        # clean and rewrite
        with src.open("r", encoding="utf-8") as fh_in, dst.open("w", encoding="utf-8") as fh_out:
            for line in fh_in:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                # some files use spaces; fall back to splitting on whitespace
                if len(parts) < 3:
                    parts = line.strip().split()
                if len(parts) >= 3:
                    name = parts[0].strip()
                    desc = parts[1].strip()
                    genes = [g.strip() for g in parts[2:] if g.strip()]
                elif len(parts) == 2:
                    name = parts[0].strip()
                    desc = ""
                    genes = [g.strip() for g in parts[1].split() if g.strip()]
                else:
                    continue
                # write cleaned line with tabs
                fh_out.write("\t".join([name, desc] + genes) + "\n")

    return outdir


def load_gmt(path: Path) -> Dict[str, Tuple[str, Set[str]]]:
    """Parse a GMT file into a mapping: term_name -> (description, set(genes))."""
    terms: Dict[str, Tuple[str, Set[str]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                # fallback: try whitespace split
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            name = parts[0]
            # Some GMTs (from MSigDB) put a URL/link in column 2 — drop it.
            raw_desc = parts[1].strip()
            if raw_desc.lower().startswith("http") or "gsea-msigdb" in raw_desc.lower() or raw_desc.startswith("www."):
                desc = ""
            else:
                desc = raw_desc
            genes = {g for g in parts[2:] if g}
            terms[name] = (desc, genes)
    return terms




# NOTE: removed older hypergeometric/local-run logic. The package exposes
# `prepare_pathways`, `load_gmt`, and `enrichr_with_local_gmt` which uses
# `gseapy.enrichr` with in-memory gene-sets built from local GMT files.


# Public API
# `enrichr_with_local_gmt` is defined below.


def enrichr_with_local_gmt(
    gene_list: Iterable[str],
    species: str = "human",
    pathways_dir: Optional[Path] = None,
    background: Optional[Iterable[str]] = None,
    top_n: int = 200,
    organism: Optional[str] = None,
) -> Optional["pd.DataFrame"]:
    """Run gseapy.enrichr against local GMT collections.

    This function collects local GMT files for the chosen species, concatenates
    them into a temporary GMT file, and calls `gseapy.enrichr` with that file
    as the `gene_sets` argument. Returns the top `top_n` rows from `enr.results`
    or None if enrichment failed.
    """
    try:
        import gseapy as gp
    except Exception:
        raise

    if pathways_dir is None:
        pathways_dir = prepare_pathways()

    species_l = species.lower() if species is not None else ""
    files = []
    for p in Path(pathways_dir).iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if species_l.startswith("human"):
            if name.startswith("c5") or name.startswith("h"):
                files.append(p)
        elif species_l.startswith("mouse"):
            if name.startswith("m"):
                files.append(p)
        else:
            files.append(p)

    if not files:
        return None

    # Build an in-memory gene set dict: term_name -> list_of_genes
    gene_sets = {}
    for f in files:
        terms = load_gmt(f)
        for term, (_desc, genes) in terms.items():
            # gseapy expects lists for gene sets
            gene_sets[term] = sorted(genes)

    # call enrichr with an in-memory dict of gene sets
    enr = gp.enrichr(
        gene_list=list(gene_list),
        gene_sets=gene_sets,
        organism=(organism or ("Human" if species_l.startswith("human") else "Mouse")),
        cutoff=1.0,
        background=(None if background is None else list(background)),
        outdir=None,
    )

    if enr is None or getattr(enr, "results", None) is None or enr.results.empty:
        return None

    df = enr.results
    # prefer p-value column naming compatibility
    sort_col = "P-value" if "P-value" in df.columns else ("pvalue" if "pvalue" in df.columns else df.columns[0])
    df = df.sort_values(by=sort_col)
    if top_n is not None:
        df = df.head(top_n)
    return df.reset_index(drop=True)


__all__ = ["prepare_pathways", "load_gmt", "enrichr_with_local_gmt"]
