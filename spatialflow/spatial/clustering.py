import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.spatial.clustering')

def run_spatial_clustering(adata, resolution=0.5, cluster_key="spatial_clusters", 
                           neighbors_key="spatial_connectivities", method="leiden", **kwargs):
    """
    Run clustering on spatial data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    method : str, optional
        Clustering method. Options: "leiden", "louvain", "kmeans", "spectral"
    resolution : float, optional
        Resolution parameter for community detection
    cluster_key : str, optional
        Key to add to adata.obs for cluster assignments
    neighbors_key : str, optional
        Key in adata.uns for neighbors graph
    n_pcs : int, optional
        Number of PCs to use for clustering (if not using neighbors graph)
    **kwargs
        Additional parameters for the clustering method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added clustering results
    """
    try:
        logger.info(f"Running spatial clustering using {method} method")
        
        if method == "leiden":
            logger.info(f"Running Leiden clustering with resolution {resolution}")
            _run_leiden_clustering(adata, resolution, cluster_key, neighbors_key, **kwargs)
        elif method == "louvain":
            logger.info(f"Running Louvain clustering with resolution {resolution}")
            _run_louvain_clustering(adata, resolution, cluster_key, neighbors_key, **kwargs)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        return adata
    except Exception as e:
        logger.error(f"Error in spatial clustering: {str(e)}")
        raise

def _run_leiden_clustering(adata, resolution=0.5, cluster_key="spatial_leiden", 
                          neighbors_key="spatial_connectivities", **kwargs):
    """
    Run Leiden clustering on the data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    resolution : float, optional
        Resolution parameter for Leiden clustering, by default 0.5
    cluster_key : str, optional
        Key to store the cluster assignments in adata.obs, by default "spatial_leiden"
    neighbors_key : str, optional
        Key for the neighbors graph in adata.obsp, by default "spatial_connectivities"
    **kwargs
        Additional arguments to pass to sc.tl.leiden
        
    Returns
    -------
    None
    """
    logger.info(f"Running Leiden clustering with resolution {resolution}")
    if neighbors_key not in adata.obsp:
        raise ValueError(f"Neighbors graph {neighbors_key} not found. Build a graph first.")
    leiden_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['key_added', 'use_pca', 'use_spatial', 'n_pcs', 
                                'cluster_resolution', 'coord_type', 'n_neighs', 'n_perms']}
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=cluster_key,
        adjacency=adata.obsp[neighbors_key],
        **leiden_kwargs
    )
    
    logger.info(f"Leiden clustering completed. Found {len(set(adata.obs[cluster_key]))} clusters")

def _run_louvain_clustering(adata, resolution=0.5, cluster_key='spatial_louvain', 
                          neighbors_key='spatial_neighbors', **kwargs):
    """
    Run Louvain clustering
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    resolution : float, optional
        Resolution parameter for Louvain clustering
    cluster_key : str, optional
        Key to add to adata.obs for cluster assignments
    neighbors_key : str, optional
        Key in adata.uns for neighbors graph
    **kwargs
        Additional parameters for sc.tl.louvain
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added clustering results
    """
    logger.info(f"Running Louvain clustering with resolution {resolution}")
    if f'{neighbors_key}_connectivities' not in adata.obsp:
        logger.error(f"Neighbors graph {neighbors_key} not found. Build a graph first.")
        raise ValueError(f"Neighbors graph {neighbors_key} not found. Build a graph first.")
    sc.tl.louvain(adata, resolution=resolution, neighbors_key=neighbors_key, 
                key_added=cluster_key, **kwargs)
    
    return adata

def _run_kmeans_clustering(adata, cluster_key='kmeans', n_pcs=None, n_clusters=None, **kwargs):
    """
    Run K-means clustering
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key to add to adata.obs for cluster assignments
    n_pcs : int, optional
        Number of PCs to use for clustering
    n_clusters : int, optional
        Number of clusters to identify
    **kwargs
        Additional parameters for KMeans
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added clustering results
    """
    from sklearn.cluster import KMeans
    if n_clusters is None:
        if 'leiden' in adata.obs:
            n_clusters = adata.obs['leiden'].nunique()
            logger.info(f"Using {n_clusters} clusters based on leiden clustering")
        else:
            n_clusters = int(np.sqrt(adata.n_obs / 2))
            logger.info(f"Using {n_clusters} clusters based on rule of thumb")
    
    logger.info(f"Running K-means clustering with {n_clusters} clusters")
    if n_pcs is not None and 'X_pca' in adata.obsm:
        logger.info(f"Using {n_pcs} PCs for K-means clustering")
        X = adata.obsm['X_pca'][:, :n_pcs]
    elif 'X_pca' in adata.obsm:
        logger.info("Using all available PCs for K-means clustering")
        X = adata.obsm['X_pca']
    elif 'spatial' in adata.obsm:
        logger.info("Using spatial coordinates for K-means clustering")
        X = adata.obsm['spatial']
    else:
        logger.info("Using full gene expression matrix for K-means clustering")
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    kmeans_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['key_added', 'use_pca', 'use_spatial', 'n_pcs']}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kmeans_kwargs)
    adata.obs[cluster_key] = pd.Categorical(kmeans.fit_predict(X).astype(str))
    
    return adata

def _run_spectral_clustering(adata, cluster_key='spectral', neighbors_key='spatial_neighbors', 
                           n_pcs=None, n_clusters=None, **kwargs):
    """
    Run spectral clustering
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key to add to adata.obs for cluster assignments
    neighbors_key : str, optional
        Key in adata.uns for neighbors graph
    n_pcs : int, optional
        Number of PCs to use for clustering
    n_clusters : int, optional
        Number of clusters to identify
    **kwargs
        Additional parameters for SpectralClustering
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added clustering results
    """
    from sklearn.cluster import SpectralClustering
    if n_clusters is None:
        if 'leiden' in adata.obs:
            n_clusters = adata.obs['leiden'].nunique()
            logger.info(f"Using {n_clusters} clusters based on leiden clustering")
        else:
            n_clusters = int(np.sqrt(adata.n_obs / 2))
            logger.info(f"Using {n_clusters} clusters based on rule of thumb")
    
    logger.info(f"Running spectral clustering with {n_clusters} clusters")
    if f'{neighbors_key}_connectivities' in adata.obsp:
        logger.info(f"Using {neighbors_key} graph for spectral clustering")
        affinity = 'precomputed'
        connectivity = adata.obsp[f'{neighbors_key}_connectivities']
        connectivity = 0.5 * (connectivity + connectivity.T)
        
        X = connectivity.toarray() if hasattr(connectivity, 'toarray') else connectivity
    elif n_pcs is not None and 'X_pca' in adata.obsm:
        logger.info(f"Using {n_pcs} PCs for spectral clustering")
        affinity = 'nearest_neighbors'
        X = adata.obsm['X_pca'][:, :n_pcs]
    elif 'X_pca' in adata.obsm:
        logger.info("Using all available PCs for spectral clustering")
        affinity = 'nearest_neighbors'
        X = adata.obsm['X_pca']
    elif 'spatial' in adata.obsm:
        logger.info("Using spatial coordinates for spectral clustering")
        affinity = 'nearest_neighbors'
        X = adata.obsm['spatial']
    else:
        logger.info("Using full gene expression matrix for spectral clustering")
        affinity = 'nearest_neighbors'
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, 
                                 random_state=42, **kwargs)
    adata.obs[cluster_key] = pd.Categorical(spectral.fit_predict(X).astype(str))
    
    return adata

def run_spatial_domain_segmentation(adata, method="watershed", cluster_key='spatial_domains', 
                                  genes=None, smooth=True, **kwargs):
    """
    Run spatial domain segmentation
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    method : str, optional
        Segmentation method. Options: "watershed", "hmrf"
    cluster_key : str, optional
        Key to add to adata.obs for domain assignments
    genes : list, optional
        List of genes to use for segmentation
    smooth : bool, optional
        Whether to smooth gene expression before segmentation
    **kwargs
        Additional parameters for the segmentation method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added domain segmentation results
    """
    logger.info(f"Running spatial domain segmentation using {method} method")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    if genes is None:
        if 'moranI' in adata.uns:
            try:
                genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:20].tolist()
                logger.info(f"Using top 20 spatially variable genes for segmentation")
            except:
                logger.warning("Could not get top spatially variable genes")
                if 'highly_variable' in adata.var.columns:
                    genes = adata.var_names[adata.var['highly_variable']].tolist()[:20]
                    logger.info(f"Using top 20 highly variable genes for segmentation")
                else:
                    logger.error("No genes selected for segmentation")
                    raise ValueError("No genes selected for segmentation")
        elif 'highly_variable' in adata.var.columns:
            genes = adata.var_names[adata.var['highly_variable']].tolist()[:20]
            logger.info(f"Using top 20 highly variable genes for segmentation")
        else:
            logger.error("No genes selected for segmentation")
            raise ValueError("No genes selected for segmentation")
    genes = [gene for gene in genes if gene in adata.var_names]
    if not genes:
        logger.error("No valid genes for segmentation")
        raise ValueError("No valid genes for segmentation")
    
    if method == "watershed":
        _run_watershed_segmentation(adata, genes, cluster_key, smooth, **kwargs)
    
    elif method == "hmrf":
        _run_hmrf_segmentation(adata, genes, cluster_key, **kwargs)
    
    else:
        logger.error(f"Unsupported segmentation method: {method}")
        raise ValueError(f"Unsupported segmentation method: {method}")
    if cluster_key in adata.obs:
        n_domains = adata.obs[cluster_key].nunique()
        logger.info(f"Identified {n_domains} spatial domains with {method} segmentation")
    
    return adata

def _run_watershed_segmentation(adata, genes, cluster_key='spatial_domains', 
                              smooth=True, min_distance=10, **kwargs):
    """
    Run watershed-based spatial domain segmentation
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to use for segmentation
    cluster_key : str, optional
        Key to add to adata.obs for domain assignments
    smooth : bool, optional
        Whether to smooth gene expression before segmentation
    min_distance : int, optional
        Minimum distance between domain centers
    **kwargs
        Additional parameters for watershed segmentation
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added domain segmentation results
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    
    logger.info("Running watershed segmentation")
    expr_matrix = np.zeros((adata.n_obs, len(genes)))
    for i, gene in enumerate(genes):
        gene_idx = adata.var_names.get_loc(gene)
        expr_matrix[:, i] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
    coords = adata.obsm['spatial'].copy()
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    grid_size = 100
    x_scale = (grid_size - 1) / (x_max - x_min)
    y_scale = (grid_size - 1) / (y_max - y_min)
    coords_scaled = np.zeros_like(coords)
    coords_scaled[:, 0] = (coords[:, 0] - x_min) * x_scale
    coords_scaled[:, 1] = (coords[:, 1] - y_min) * y_scale
    coords_scaled = coords_scaled.astype(int)
    grid = np.zeros((grid_size, grid_size, len(genes)))
    for i in range(adata.n_obs):
        x, y = coords_scaled[i]
        grid[y, x] = expr_matrix[i]
    if smooth:
        for i in range(len(genes)):
            grid[:, :, i] = ndi.gaussian_filter(grid[:, :, i], sigma=1.0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    grid_flat = grid.reshape(-1, len(genes))
    grid_pca = pca.fit_transform(grid_flat).reshape(grid_size, grid_size)
    from skimage import filters
    gradient = filters.sobel(grid_pca)
    local_max = peak_local_max(grid_pca, min_distance=min_distance, 
                              indices=False, **kwargs)
    markers = ndi.label(local_max)[0]
    segmentation = watershed(gradient, markers)
    domains = np.zeros(adata.n_obs, dtype=int)
    for i in range(adata.n_obs):
        x, y = coords_scaled[i]
        domains[i] = segmentation[y, x]
    adata.obs[cluster_key] = pd.Categorical(domains.astype(str))
    
    return adata

def _run_hmrf_segmentation(adata, genes, cluster_key='spatial_domains', 
                         beta=2.0, n_domains=None, **kwargs):
    """
    Run HMRF-based spatial domain segmentation
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to use for segmentation
    cluster_key : str, optional
        Key to add to adata.obs for domain assignments
    beta : float, optional
        Spatial regularization parameter
    n_domains : int, optional
        Number of domains to identify
    **kwargs
        Additional parameters for HMRF segmentation
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added domain segmentation results
    """
    logger.info("Running HMRF segmentation")
    if n_domains is None:
        if 'leiden' in adata.obs:
            n_domains = adata.obs['leiden'].nunique()
            logger.info(f"Using {n_domains} domains based on leiden clustering")
        else:
            n_domains = int(np.sqrt(adata.n_obs / 2))
            logger.info(f"Using {n_domains} domains based on rule of thumb")
    expr_matrix = np.zeros((adata.n_obs, len(genes)))
    for i, gene in enumerate(genes):
        gene_idx = adata.var_names.get_loc(gene)
        expr_matrix[:, i] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
    coords = adata.obsm['spatial'].copy()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_domains, random_state=42)
    domains = kmeans.fit_predict(expr_matrix)
    from sklearn.neighbors import kneighbors_graph
    n_neighbors = min(6, adata.n_obs - 1)
    spatial_graph = kneighbors_graph(coords, n_neighbors=n_neighbors, 
                                   mode='connectivity')
    max_iter = kwargs.get('max_iter', 10)
    
    for iter in range(max_iter):
        centers = np.zeros((n_domains, len(genes)))
        for k in range(n_domains):
            mask = domains == k
            if np.sum(mask) > 0:
                centers[k] = np.mean(expr_matrix[mask], axis=0)
        old_domains = domains.copy()
        for i in range(adata.n_obs):
            data_term = np.zeros(n_domains)
            for k in range(n_domains):
                data_term[k] = np.sum((expr_matrix[i] - centers[k])**2)
            spatial_term = np.zeros(n_domains)
            neighbors = spatial_graph[i].nonzero()[1]
            for k in range(n_domains):
                different_domain = np.sum(domains[neighbors] != k)
                spatial_term[k] = beta * different_domain
            total_energy = data_term + spatial_term
            domains[i] = np.argmin(total_energy)
        changes = np.sum(old_domains != domains)
        logger.info(f"Iteration {iter+1}: {changes} changes")
        
        if changes == 0:
            logger.info("HMRF converged")
            break
    adata.obs[cluster_key] = pd.Categorical(domains.astype(str))
    
    return adata

def identify_spatial_regions(adata, cluster_key=None, region_key='spatial_regions', 
                           n_neighbors=10, n_regions=None, **kwargs):
    """
    Identify higher-level spatial regions by merging similar clusters
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    region_key : str, optional
        Key to add to adata.obs for region assignments
    n_neighbors : int, optional
        Number of neighbors to consider for computing cluster similarity
    n_regions : int, optional
        Number of regions to identify
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added region assignments
    """
    logger.info("Identifying spatial regions")
    if cluster_key is None:
        cluster_keys = [key for key in adata.obs.columns if 
                      any(key.startswith(prefix) for prefix in 
                         ['leiden', 'louvain', 'kmeans', 'spectral', 'spatial_domains'])]
        
        if not cluster_keys:
            logger.error("No cluster assignments found in AnnData object")
            raise ValueError("No cluster assignments found in AnnData object")
        
        cluster_key = cluster_keys[0]
        logger.info(f"Using {cluster_key} as basis for spatial regions")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    clusters = adata.obs[cluster_key].cat.codes.values
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    if 'spatial_neighbors' in adata.uns:
        logger.info("Using spatial neighbors graph for region identification")
        adjacency = adata.obsp['spatial_connectivities']
        cluster_adjacency = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    cluster_adjacency[i, j] = 1
                else:
                    spots_i = np.where(clusters == i)[0]
                    spots_j = np.where(clusters == j)[0]
                    connections = 0
                    for spot_i in spots_i:
                        for spot_j in spots_j:
                            if adjacency[spot_i, spot_j] > 0:
                                connections += 1
                    if len(spots_i) > 0 and len(spots_j) > 0:
                        cluster_adjacency[i, j] = connections / (len(spots_i) * len(spots_j))
    else:
        logger.info("Using spatial coordinates for region identification")
        coords = adata.obsm['spatial']
        cluster_coords = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            mask = clusters == i
            cluster_coords[i] = np.mean(coords[mask], axis=0)
        from scipy.spatial.distance import pdist, squareform
        cluster_distances = squareform(pdist(cluster_coords))
        max_dist = np.max(cluster_distances)
        cluster_adjacency = 1 - cluster_distances / max_dist
    from scipy.cluster.hierarchy import linkage, fcluster
    cluster_distances = 1 - cluster_adjacency
    cluster_distances = 0.5 * (cluster_distances + cluster_distances.T)
    condensed_distances = squareform(cluster_distances)
    Z = linkage(condensed_distances, method='ward')
    if n_regions is None:
        n_regions = max(2, n_clusters // 3)
        logger.info(f"Using {n_regions} regions based on number of clusters")
    region_assignments = fcluster(Z, n_regions, criterion='maxclust') - 1
    spot_regions = np.zeros_like(clusters)
    for i in range(n_clusters):
        spot_regions[clusters == i] = region_assignments[i]
    adata.obs[region_key] = pd.Categorical(spot_regions.astype(str))
    n_actual_regions = adata.obs[region_key].nunique()
    logger.info(f"Identified {n_actual_regions} spatial regions")
    
    return adata

def refine_cluster_boundaries(adata, cluster_key='leiden', refined_key=None, 
                            smoothing_factor=0.3, iterations=3, **kwargs):
    """
    Refine cluster boundaries using spatial smoothing
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    refined_key : str, optional
        Key to add to adata.obs for refined cluster assignments
    smoothing_factor : float, optional
        Factor controlling the amount of smoothing (0-1)
    iterations : int, optional
        Number of smoothing iterations
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added refined cluster assignments
    """
    logger.info(f"Refining cluster boundaries for {cluster_key}")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if refined_key is None:
        refined_key = f"{cluster_key}_refined"
    if 'spatial_neighbors' not in adata.uns:
        logger.error("Spatial neighbors graph not found. Build a graph first.")
        raise ValueError("Spatial neighbors graph not found. Build a graph first.")
    adjacency = adata.obsp['spatial_connectivities']
    clusters = adata.obs[cluster_key].cat.codes.values
    unique_clusters = adata.obs[cluster_key].cat.categories
    n_clusters = len(unique_clusters)
    one_hot = np.zeros((adata.n_obs, n_clusters))
    for i in range(adata.n_obs):
        one_hot[i, clusters[i]] = 1
    smoothed = one_hot.copy()
    
    for _ in range(iterations):
        neighborhood_avg = adjacency @ smoothed
        row_sums = adjacency.sum(axis=1).A.flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        neighborhood_avg = neighborhood_avg / row_sums[:, np.newaxis]
        smoothed = (1 - smoothing_factor) * one_hot + smoothing_factor * neighborhood_avg
    refined_clusters = np.argmax(smoothed, axis=1)
    adata.obs[refined_key] = pd.Categorical(
        [unique_clusters[i] for i in refined_clusters], 
        categories=unique_clusters
    )
    changes = np.sum(refined_clusters != clusters)
    change_percentage = changes / adata.n_obs * 100
    logger.info(f"Refined {changes} spots ({change_percentage:.1f}%)")
    
    return adata
