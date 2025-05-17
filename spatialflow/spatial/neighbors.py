import scanpy as sc
import squidpy as sq
import logging
import numpy as np
from anndata import AnnData

logger = logging.getLogger('spatialflow.spatial.neighbors')

def build_spatial_graph(adata, n_neighs=6, coord_type='generic', **kwargs):
    """
    Build spatial neighbors graph
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates
    n_neighs : int, optional
        Number of neighbors, by default 6
    coord_type : str, optional
        Type of coordinates, by default 'generic'
    **kwargs : dict
        Additional arguments to pass to the spatial_neighbors function
    
    Returns
    -------
    adata : AnnData
        AnnData object with spatial neighbors graph
    """
    try:
        logger.info("Building spatial neighbors graph")
        logger.info(f"Using {n_neighs} neighbors for spatial neighbors")
        spatial_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['cluster_resolution', 'n_perms']}
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type=coord_type,
                              spatial_key="spatial", **spatial_kwargs)
        return adata
    except Exception as e:
        logger.error(f"Error building spatial neighbors graph: {str(e)}")
        raise

def refine_spatial_graph(adata, condition_key=None, min_weight=0.1, **kwargs):
    """
    Refine the spatial neighbors graph based on additional criteria
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object with spatial neighbors graph
    condition_key : str, optional
        Key in adata.obs for condition-based refinement
    min_weight : float, optional
        Minimum weight for edges
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with refined spatial neighbors graph
    """
    logger.info("Refining spatial neighbors graph")
    if 'spatial_connectivities' not in adata.obsp:
        logger.error("No spatial neighbors graph found. Run build_spatial_graph first.")
        raise ValueError("No spatial neighbors graph found. Run build_spatial_graph first.")
    connectivity = adata.obsp['spatial_connectivities'].copy()
    if condition_key is not None:
        if condition_key not in adata.obs:
            logger.error(f"Condition key {condition_key} not found in adata.obs")
            raise ValueError(f"Condition key {condition_key} not found in adata.obs")
        
        logger.info(f"Refining graph based on condition {condition_key}")
        conditions = adata.obs[condition_key].values
        import scipy.sparse as sp
        new_connectivity = connectivity.copy()
        rows, cols = connectivity.nonzero()
        for i, j in zip(rows, cols):
            if conditions[i] != conditions[j]:
                new_connectivity[i, j] *= min_weight
        adata.obsp['spatial_connectivities'] = new_connectivity
        if 'spatial_distances' in adata.obsp:
            adata.obsp['spatial_distances'] = new_connectivity.copy()
            adata.obsp['spatial_distances'].data = 1.0 / adata.obsp['spatial_distances'].data
    
    return adata

def compute_local_neighborhood_stats(adata, layer=None, genes=None, n_jobs=None):
    """
    Compute statistics for local neighborhoods
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object with spatial neighbors graph
    layer : str, optional
        Layer to use for gene expression
    genes : list, optional
        Genes to include in the analysis
    n_jobs : int, optional
        Number of jobs for parallel processing
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added neighborhood statistics
    """
    logger.info("Computing local neighborhood statistics")
    if 'spatial_connectivities' not in adata.obsp:
        logger.error("No spatial neighbors graph found. Run build_spatial_graph first.")
        raise ValueError("No spatial neighbors graph found. Run build_spatial_graph first.")
    
    try:
        import networkx as nx
        from scipy import sparse
        conn_matrix = adata.obsp['spatial_connectivities']
        G = nx.from_scipy_sparse_matrix(conn_matrix)
        degrees = dict(G.degree())
        adata.obs['spatial_degree'] = [degrees[i] for i in range(adata.n_obs)]
        clustering = nx.clustering(G)
        adata.obs['spatial_clustering'] = [clustering[i] for i in range(adata.n_obs)]
        try:
            betweenness = nx.betweenness_centrality(G, k=min(100, adata.n_obs), normalized=True)
            adata.obs['spatial_betweenness'] = [betweenness[i] for i in range(adata.n_obs)]
        except:
            logger.warning("Could not compute betweenness centrality. Graph may be too large.")
        if genes is not None:
            if layer is not None and layer in adata.layers:
                X = adata.layers[layer]
            else:
                X = adata.X
            from scipy.stats import entropy
            import numpy as np
            gene_indices = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
            
            if not gene_indices:
                logger.warning("None of the specified genes found in the data")
            else:
                neighbors = dict(nx.all_pairs_shortest_path_length(G, cutoff=1))
                local_heterogeneity = np.zeros((adata.n_obs, len(gene_indices)))
                
                for spot_idx in range(adata.n_obs):
                    try:
                        neighbor_indices = list(neighbors[spot_idx].keys())
                    except KeyError:
                        neighbor_indices = [spot_idx]
                    for i, gene_idx in enumerate(gene_indices):
                        if sparse.issparse(X):
                            expr_values = X[neighbor_indices, gene_idx].toarray().flatten()
                        else:
                            expr_values = X[neighbor_indices, gene_idx]
                        bins = min(10, len(expr_values))
                        if bins > 1:
                            hist, _ = np.histogram(expr_values, bins=bins)
                            if np.sum(hist) > 0:
                                probs = hist / np.sum(hist)
                                local_heterogeneity[spot_idx, i] = entropy(probs)
                for i, gene_idx in enumerate(gene_indices):
                    gene_name = adata.var_names[gene_idx]
                    adata.obs[f'heterogeneity_{gene_name}'] = local_heterogeneity[:, i]
                adata.obs['spatial_heterogeneity'] = np.mean(local_heterogeneity, axis=1)
        
        logger.info("Local neighborhood statistics computed successfully")
        
    except Exception as e:
        logger.error(f"Error computing local neighborhood statistics: {str(e)}")
        raise
    
    return adata

def build_knn_graph(adata, n_neighbors=15, n_pcs=None, use_rep=None, **kwargs):
    """
    Build a K-nearest neighbors graph based on expression space
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    n_neighbors : int, optional
        Number of neighbors
    n_pcs : int, optional
        Number of principal components to use
    use_rep : str, optional
        Key for representation to use as alternative to PCs
    **kwargs
        Additional parameters for scanpy.pp.neighbors
    
    Returns
    -------
    adata : AnnData
        The AnnData object with KNN graph
    """
    logger.info(f"Building KNN graph with {n_neighbors} neighbors")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, **kwargs)
    
    return adata

def compute_neighbor_enrichment(adata, cluster_key, n_perms=1000, **kwargs):
    """
    Compute neighborhood enrichment between clusters
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object with spatial neighbors graph
    cluster_key : str
        Key in adata.obs for cluster assignments
    n_perms : int, optional
        Number of permutations for significance testing
    **kwargs
        Additional parameters for squidpy.gr.nhood_enrichment
    
    Returns
    -------
    adata : AnnData
        The AnnData object with neighborhood enrichment results
    """
    logger.info(f"Computing neighborhood enrichment for clusters in {cluster_key}")
    if 'spatial_connectivities' not in adata.obsp:
        logger.error("No spatial neighbors graph found. Run build_spatial_graph first.")
        raise ValueError("No spatial neighbors graph found. Run build_spatial_graph first.")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    try:
        sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_perms=n_perms, **kwargs)
        
        logger.info("Neighborhood enrichment computed successfully")
        
    except Exception as e:
        logger.error(f"Error computing neighborhood enrichment: {str(e)}")
        raise
    
    return adata
