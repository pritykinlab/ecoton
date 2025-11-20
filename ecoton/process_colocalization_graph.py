
def ProcessColocalizationGraph(
    G_ig,
    flavor="leiden",
    resolution=4.0,
    n_components=20,
    top_k=200,
    K=50,                 # number of archetypes
    z_threshold=0.30,     # threshold for Z (soft memberships → broad modules)
    h_threshold=0.005,     # threshold for H (archetype gene programs → sharp modules)
    seed=42
):
    """
    Process the co-localization graph to identify gene modules and compute centrality measures.

    Parameters
    ----------
    flavor : {"leiden", "nmf", "archetypes"}
    resolution : float      (for leiden)
    n_components : int      (for nmf)
    top_k : int             (for nmf)
    K : int                 (for archetypes)
    z_threshold : float     (for archetypes)
    h_threshold : float     (for archetypes)
    """

    import pandas as pd
    import numpy as np
    import igraph as ig

    # ---------- CENTRALITY (same for all flavors) ----------
    deg_cent = G_ig.strength(weights=G_ig.es['weight'])
    deg_unweighted = G_ig.degree()
    inv_weights = [1/w for w in G_ig.es['weight']]
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

        return {
            "modules_leiden": modules,
            "centrality": df_centrality
        }

    # ========================================================
    #  FLAVOR: NMF
    # ========================================================
    elif flavor.lower() == "nmf":
        print(f"→ Running NMF clustering (n_components={n_components}, top_k={top_k})...")

        from sklearn.decomposition import NMF

        A = np.array(G_ig.get_adjacency(attribute="weight").data, dtype=float)
        nmf_model = NMF(n_components=n_components, init="nndsvd", random_state=seed, max_iter=1000)
        W = nmf_model.fit_transform(A)

        gene_names = np.array(G_ig.vs['name'])

        modules_nmf = [
            list(gene_names[np.argsort(W[:, k])[::-1][:top_k]])
            for k in range(n_components)
        ]

        return {
            "modules_nmf": modules_nmf,
            "W": W,
            "centrality": df_centrality
        }

    # ========================================================
    #  FLAVOR: ARCHETYPAL ANALYSIS
    # ========================================================
    elif flavor.lower() == "archetypes":
        print(f"→ Running Archetypal Analysis (K={K})...")

        from sklearn.preprocessing import normalize
        from archetypes import AA

        A = np.array(G_ig.get_adjacency(attribute="weight").data, dtype=float)
        A = (A + A.T) / 2
        X = normalize(A, norm='l2', axis=1)

        model = AA(n_archetypes=K, random_state=seed).fit(X)
        Z = model.A_
        H = model.B_

        genes = np.array(G_ig.vs["name"])
        Z_df = pd.DataFrame(Z, index=genes, columns=[f"archetype_{k}" for k in range(K)])
        H_df = pd.DataFrame(H, index=[f"archetype_{k}" for k in range(K)], columns=genes)

        # Z-based (soft) modules
        modules_Z = [
            Z_df.index[Z_df[f"archetype_{k}"] >= z_threshold].tolist()
            for k in range(K)
        ]

        # H-based (sharp) modules
        modules_H = [
            H_df.columns[H_df.loc[f"archetype_{k}"] >= h_threshold].tolist()
            for k in range(K)
        ]

        return {
            "modules_Z": modules_Z,
            "modules_H": modules_H,
            "Z_df": Z_df,
            "H_df": H_df,
            "centrality": df_centrality
        }

    else:
        raise ValueError("flavor must be one of {'leiden', 'nmf', 'archetypes'}")
