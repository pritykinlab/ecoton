def ProcessColocalizationGraph(
    G_ig,
    K=25,                 # number of archetypes
    z_threshold=0.30,     # threshold for Z (soft memberships → broad modules)
    h_threshold=0.005,    # threshold for H (archetype gene programs → sharp modules)
    seed=42,
    # enrichment options
    run_enrich=False,
    msigdb_libraries=None,
    organism='Mouse',
    top_n_terms=20,
    compute_centrality=False,
    module_source='W'
):
    """
    Process the co-localization graph using Archetypal Analysis.

    Identifies gene modules, optionally computes centrality metrics,
    and optionally performs enrichment analysis.
    """

    import pandas as pd
    import numpy as np
    import igraph as ig
    import random

    np.random.seed(seed)
    random.seed(seed)

    result = {}

    # ---------- CENTRALITY (optional) ----------
    if compute_centrality:
        deg_cent = G_ig.strength(weights=G_ig.es['weight'])
        deg_unweighted = G_ig.degree()
        inv_weights = [1 / w for w in G_ig.es['weight']]
        bet_cent = G_ig.betweenness(weights=inv_weights)
        close_cent = G_ig.closeness(weights=inv_weights)
        eig_cent = G_ig.eigenvector_centrality(weights=G_ig.es['weight'])

        df_centrality = pd.DataFrame({
            "Gene": G_ig.vs['name'],
            "Degree": deg_cent,
            "Degree Unweighted": deg_unweighted,
            "Betweenness": bet_cent,
            "Closeness": close_cent,
            "Eigenvector": eig_cent
        })

        result["centrality"] = df_centrality

    # ========================================================
    #  ARCHETYPAL ANALYSIS
    # ========================================================
    print(f"→ Running Archetypal Analysis (K={K})...")

    from sklearn.preprocessing import normalize
    from archetypes import AA

    A = np.array(G_ig.get_adjacency(attribute="weight").data, dtype=float)
    A = (A + A.T) / 2
    X = normalize(A, norm='l2', axis=1)

    model = AA(n_archetypes=K, random_state=seed).fit(X)
    W = model.A_
    H = model.B_

    genes = np.array(G_ig.vs["name"])
    W_df = pd.DataFrame(W, index=genes, columns=[f"archetype_{k}" for k in range(K)])
    H_df = pd.DataFrame(H, index=[f"archetype_{k}" for k in range(K)], columns=genes)

    # W-based (soft) modules
    modules_W = [
        W_df.index[W_df[f"archetype_{k}"] >= z_threshold].tolist()
        for k in range(K)
    ]

    # H-based (sharp) modules
    modules_H = [
        H_df.columns[H_df.loc[f"archetype_{k}"] >= h_threshold].tolist()
        for k in range(K)
    ]

    result.update({
        "modules_W": modules_W,
        "modules_H": modules_H,
        "W_df": W_df,
        "H_df": H_df,
    })

    # save soft-module membership per vertex: list of archetype indices
    membership = {g: [] for g in genes}
    for k, mod in enumerate(modules_W):
        for g in mod:
            membership.setdefault(g, []).append(k)

    # align to graph vertex order and store
    G_ig.vs["cluster"] = [membership.get(v, []) for v in G_ig.vs["name"]]

    # ----------------- Enrichment (optional) -----------------
    if run_enrich:
        import gseapy as gp

        if msigdb_libraries is None:
            msigdb_libraries = [
                "MSigDB_Hallmark_2020",
                "GO_Biological_Process_2025",
                "GO_Molecular_Function_2025",
                "GO_Cellular_Component_2025"
            ]

        if module_source == 'H':
            modules_for_enrich = result.get("modules_H", None)
        else:
            modules_for_enrich = result.get("modules_W", None)

        enrichment_results = {}
        if modules_for_enrich is not None:
            background = list(G_ig.vs['name'])
            for idx, mod in enumerate(modules_for_enrich):
                if len(mod) == 0:
                    enrichment_results[idx] = None
                    continue

                enr = gp.enrichr(
                    gene_list=mod,
                    gene_sets=msigdb_libraries,
                    organism=organism,
                    cutoff=1.0,
                    background=background
                )

                if enr is None or enr.results is None or enr.results.empty:
                    enrichment_results[idx] = None
                else:
                    top_terms = enr.results.sort_values(by='P-value').head(top_n_terms)
                    enrichment_results[idx] = top_terms

        result['enrichment'] = enrichment_results

    return result

def ProcessColocalizationGraphExtra(
    G_ig,
    flavor="leiden",
    resolution=4.0,
    n_components=20,
    top_k=200,
    seed=42,
    compute_centrality=False,
):
    """
    Extra processing for benchmarking flavors: 'leiden' or 'nmf'.

    Parameters
    ----------
    G_ig : igraph.Graph
        Graph with 'weight' edge attribute and vertex names in G_ig.vs['name'].
    flavor : {"leiden", "nmf"}
        Which extra method to run.
    resolution : float
        Leiden resolution parameter (used when flavor='leiden').
    n_components : int
        Number of NMF components (used when flavor='nmf').
    top_k : int
        Number of top genes to keep per component (used when flavor='nmf').
    seed : int
        Random seed for reproducibility.
    compute_centrality : bool
        Whether to compute and return centrality metrics (default False).

    Returns
    -------
    dict
        For leiden:
          - "modules_leiden": list[list[str]]
        For nmf:
          - "modules_nmf": list[list[str]]
          - "W": np.ndarray
        Optionally:
          - "centrality": pd.DataFrame
    """
    import pandas as pd
    import numpy as np
    import igraph as ig
    import random

    np.random.seed(seed)
    random.seed(seed)

    result = {}

    # ---------- CENTRALITY (optional) ----------
    if compute_centrality:
        deg_cent = G_ig.strength(weights=G_ig.es['weight'])
        deg_unweighted = G_ig.degree()
        inv_weights = [1 / w for w in G_ig.es['weight']]
        bet_cent = G_ig.betweenness(weights=inv_weights)
        close_cent = G_ig.closeness(weights=inv_weights)
        eig_cent = G_ig.eigenvector_centrality(weights=G_ig.es['weight'])

        df_centrality = pd.DataFrame({
            "Gene": G_ig.vs['name'],
            "Degree": deg_cent,
            "Degree Unweighted": deg_unweighted,
            "Betweenness": bet_cent,
            "Closeness": close_cent,
            "Eigenvector": eig_cent
        })
        result["centrality"] = df_centrality

    # ========================================================
    #  FLAVOR: LEIDEN
    # ========================================================
    if flavor.lower() == "leiden":
        print("→ Running Leiden clustering...")
        import leidenalg as la

        partition = la.find_partition(
            G_ig,
            la.RBConfigurationVertexPartition,
            weights=G_ig.es['weight'],
            resolution_parameter=resolution,
            seed=seed
        )

        G_ig.vs['cluster'] = partition.membership
        modules = [[G_ig.vs[i]['name'] for i in comm] for comm in partition]
        result.update({"modules_leiden": modules})

    # ========================================================
    #  FLAVOR: NMF
    # ========================================================
    elif flavor.lower() == "nmf":
        print(f"→ Running NMF clustering (n_components={n_components}, top_k={top_k})...")

        from sklearn.decomposition import NMF

        A = np.array(G_ig.get_adjacency(attribute="weight").data, dtype=float)

        nmf_model = NMF(
            n_components=n_components,
            init="nndsvd",
            random_state=seed,
            max_iter=1000
        )
        W = nmf_model.fit_transform(A)

        gene_names = np.array(G_ig.vs['name'])

        modules_nmf = [
            list(gene_names[np.argsort(W[:, k])[::-1][:top_k]])
            for k in range(n_components)
        ]

        result.update({"modules_nmf": modules_nmf, "W": W})

    else:
        raise ValueError("flavor must be one of {'leiden', 'nmf'}")

    return result