import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.spatial.statistics')

def calculate_spatial_stats(
    adata,
    method='moran',
    n_genes=100,
    **kwargs
):
    """
    Calculate spatial statistics for the data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    method : str, optional
        Method to use for spatial statistics, by default 'moran'
    n_genes : int, optional
        Number of genes to use, by default 100
    **kwargs
        Additional arguments to pass to sq.gr.spatial_autocorr
        
    Returns
    -------
    AnnData
        Annotated data matrix with spatial statistics.
    """
    logger.info(f"Calculating spatial statistics using {method} index")
    if not adata.var_names.is_unique:
        logger.warning("Variable names are not unique. Making them unique.")
        adata.var_names_make_unique()
    if 'highly_variable' in adata.var:
        logger.info(f"Using top {n_genes} highly variable genes")
        genes = adata.var_names[adata.var['highly_variable']].tolist()[:n_genes]
    else:
        logger.warning("No highly variable genes found. Using top expressed genes.")
        gene_means = adata.X.mean(axis=0)
        top_genes_idx = np.argsort(gene_means)[-n_genes:]
        genes = adata.var_names[top_genes_idx].tolist()
    
    try:
        sq.gr.spatial_autocorr(
            adata,
            genes=genes
        )
        logger.info(f"Spatial statistics calculated for {len(genes)} genes")
    except Exception as e:
        logger.error(f"Error calculating spatial statistics: {str(e)}")
        raise
        
    return adata

def calculate_ripley_statistics(adata, cluster_key, mode="K", dist=None, 
                              n_simulations=100, n_steps=50, **kwargs):
    """
    Calculate Ripley's statistics for spatial clustering
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    mode : str, optional
        Which statistic to compute. Options: "K", "L", "F"
    dist : np.ndarray, optional
        Distribution of distances to evaluate
    n_simulations : int, optional
        Number of simulations for confidence intervals
    n_steps : int, optional
        Number of steps for distance range
    **kwargs
        Additional parameters for squidpy.gr.ripley
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added Ripley's statistics
    """
    logger.info(f"Calculating Ripley's {mode} statistics for {cluster_key}")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    try:
        sq.gr.ripley(
            adata,
            cluster_key=cluster_key,
            mode=mode,
            dist=dist,
            n_simulations=n_simulations,
            n_steps=n_steps,
            **kwargs
        )
        
        logger.info(f"Ripley's {mode} statistics calculated successfully")
        
    except Exception as e:
        logger.error(f"Error calculating Ripley's {mode} statistics: {str(e)}")
        raise
    
    return adata

def calculate_spatial_hotspots(adata, genes, method="getis-ord", n_jobs=None, **kwargs):
    """
    Calculate spatial hotspots for genes of interest
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to calculate hotspots for
    method : str, optional
        Method to use for hotspot detection. Options: "getis-ord", "local-moran"
    n_jobs : int, optional
        Number of jobs for parallel processing
    **kwargs
        Additional parameters for the hotspot detection method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added hotspot results
    """
    logger.info(f"Calculating spatial hotspots using {method} method")
    if 'spatial_connectivities' not in adata.obsp:
        logger.error("No spatial neighbors graph found. Build a spatial graph first.")
        raise ValueError("No spatial neighbors graph found. Build a spatial graph first.")
    valid_genes = [gene for gene in genes if gene in adata.var_names]
    if not valid_genes:
        logger.error("None of the specified genes found in the data")
        raise ValueError("None of the specified genes found in the data")
    adj_matrix = adata.obsp['spatial_connectivities']
    weights = adj_matrix.copy()
    weights_array = weights.toarray()
    row_sums = weights_array.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights_array = weights_array / row_sums[:, np.newaxis]
    from scipy import sparse
    weights = sparse.csr_matrix(weights_array)
    hotspot_results = {}
    
    try:
        if method == "getis-ord":
            from scipy import stats
            
            for gene in valid_genes:
                gene_idx = adata.var_names.get_loc(gene)
                x = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
                local_sum = weights @ x
                n = len(x)
                x_mean = np.mean(x)
                x_std = np.std(x)
                gi_star = (local_sum - x_mean * np.sum(weights_array, axis=1)) / (x_std * np.sqrt((n * np.sum(weights_array**2, axis=1) - np.sum(weights_array, axis=1)**2) / (n - 1)))
                p_values = 2 * (1 - stats.norm.cdf(np.abs(gi_star)))
                hotspot_results[gene] = {
                    'gi_star': gi_star,
                    'p_value': p_values
                }
                adata.obs[f'hotspot_{gene}_gi_star'] = gi_star
                adata.obs[f'hotspot_{gene}_pval'] = p_values
                adata.obs[f'hotspot_{gene}_significant'] = p_values < 0.05
                n_hotspots = np.sum(p_values < 0.05)
                logger.info(f"Gene {gene}: {n_hotspots} significant hotspots (p < 0.05)")
                
        elif method == "local-moran":
            for gene in valid_genes:
                gene_idx = adata.var_names.get_loc(gene)
                x = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
                z = (x - np.mean(x)) / np.std(x)
                local_moran = z * (weights @ z)
                n = len(x)
                expected_i = -1 / (n - 1)
                from scipy import stats
                z_scores = (local_moran - expected_i) / np.std(local_moran)
                p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
                hotspot_results[gene] = {
                    'local_moran': local_moran,
                    'p_value': p_values
                }
                adata.obs[f'hotspot_{gene}_local_moran'] = local_moran
                adata.obs[f'hotspot_{gene}_pval'] = p_values
                adata.obs[f'hotspot_{gene}_significant'] = p_values < 0.05
                categories = np.array(['NS'] * n, dtype=object)
                sig_mask = p_values < 0.05
                hh_mask = (z > 0) & (weights @ z > 0) & sig_mask
                ll_mask = (z < 0) & (weights @ z < 0) & sig_mask
                hl_mask = (z > 0) & (weights @ z < 0) & sig_mask
                lh_mask = (z < 0) & (weights @ z > 0) & sig_mask
                
                categories[hh_mask] = 'HH'  # High-High cluster
                categories[ll_mask] = 'LL'  # Low-Low cluster
                categories[hl_mask] = 'HL'  # High-Low outlier
                categories[lh_mask] = 'LH'  # Low-High outlier
                
                adata.obs[f'hotspot_{gene}_category'] = pd.Categorical(categories)
                n_hotspots = np.sum(sig_mask)
                logger.info(f"Gene {gene}: {n_hotspots} significant local clusters/outliers (p < 0.05)")
                logger.info(f"  High-High clusters: {np.sum(hh_mask)}")
                logger.info(f"  Low-Low clusters: {np.sum(ll_mask)}")
                logger.info(f"  High-Low outliers: {np.sum(hl_mask)}")
                logger.info(f"  Low-High outliers: {np.sum(lh_mask)}")
        
        else:
            logger.error(f"Unsupported hotspot detection method: {method}")
            raise ValueError(f"Unsupported hotspot detection method: {method}")
        adata.uns[f'hotspots_{method}'] = hotspot_results
        
    except Exception as e:
        logger.error(f"Error calculating spatial hotspots: {str(e)}")
        raise
    
    return adata

def calculate_spatial_context(adata, cluster_key, context_radius=None, n_rings=3):
    """
    Calculate the spatial context of each spot (composition of surrounding clusters)
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    context_radius : float, optional
        Radius to consider for context, in same units as spatial coordinates
    n_rings : int, optional
        Number of concentric rings to divide the neighborhood into
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added spatial context results
    """
    logger.info(f"Calculating spatial context for {cluster_key}")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    clusters = adata.obs[cluster_key].values
    unique_clusters = adata.obs[cluster_key].cat.categories.tolist()
    coords = adata.obsm['spatial']
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(coords))
    if context_radius is None:
        sorted_distances = np.sort(dist_matrix, axis=1)
        k = min(10, dist_matrix.shape[1] - 1)
        context_radius = np.median(sorted_distances[:, k])
        logger.info(f"Using context radius: {context_radius:.2f}")
    radii = np.linspace(0, context_radius, n_rings + 1)
    context_results = []
    
    for i in range(adata.n_obs):
        spot_context = {}
        
        for r in range(n_rings):
            ring_min = radii[r]
            ring_max = radii[r + 1]
            ring_mask = (dist_matrix[i] > ring_min) & (dist_matrix[i] <= ring_max)
            ring_clusters = clusters[ring_mask]
            ring_counts = pd.Series(ring_clusters).value_counts()
            ring_total = len(ring_clusters)
            
            for cluster in unique_clusters:
                if cluster in ring_counts:
                    spot_context[f'context_ring{r+1}_{cluster}'] = ring_counts[cluster] / ring_total
                else:
                    spot_context[f'context_ring{r+1}_{cluster}'] = 0.0
        
        context_results.append(spot_context)
    context_df = pd.DataFrame(context_results, index=adata.obs.index)
    for col in context_df.columns:
        adata.obs[col] = context_df[col]
    from scipy.stats import entropy
    
    for r in range(n_rings):
        context_columns = [col for col in context_df.columns if col.startswith(f'context_ring{r+1}_')]
        context_values = context_df[context_columns].values
        ring_entropy = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            values = context_values[i]
            if np.sum(values) > 0:
                ring_entropy[i] = entropy(values)
        
        adata.obs[f'context_ring{r+1}_entropy'] = ring_entropy
    
    logger.info(f"Spatial context calculated successfully")
    
    return adata