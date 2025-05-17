import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger('spatialflow.spatial.trajectories')

def run_spatial_trajectory_analysis(adata, method="principal_curve", 
                                  start_spot=None, end_spot=None, 
                                  n_points=100, cluster_key=None,
                                  key_added='spatial_trajectory', **kwargs):
    """
    Run spatial trajectory analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    method : str, optional
        Method for spatial trajectory analysis. Options: "principal_curve", "shortest_path", "custom"
    start_spot : str or int, optional
        Index of starting spot for trajectory
    end_spot : str or int, optional
        Index of ending spot for trajectory
    n_points : int, optional
        Number of points for trajectory interpolation
    cluster_key : str, optional
        Key in adata.obs for cluster assignments, used for trajectory between clusters
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    **kwargs
        Additional parameters for trajectory analysis
    
    Returns
    -------
    adata : AnnData
        The AnnData object with spatial trajectory results
    """
    logger.info(f"Running spatial trajectory analysis using {method} method")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    if method == "principal_curve":
        adata = _run_principal_curve(adata, n_points, key_added, **kwargs)
    
    elif method == "shortest_path":
        adata = _run_shortest_path(adata, start_spot, end_spot, n_points, key_added, **kwargs)
    
    elif method == "custom":
        if 'trajectory_function' not in kwargs:
            logger.error("Custom trajectory method requires 'trajectory_function' parameter")
            raise ValueError("Custom trajectory method requires 'trajectory_function' parameter")
        
        trajectory_function = kwargs.pop('trajectory_function')
        adata = trajectory_function(adata, **kwargs)
    
    else:
        logger.error(f"Unsupported trajectory method: {method}")
        raise ValueError(f"Unsupported trajectory method: {method}")
    if f'{key_added}_pseudotime' in adata.obs:
        _compute_trajectory_gene_trends(adata, key_added, **kwargs)
    
    return adata

def _run_principal_curve(adata, n_points=100, key_added='spatial_trajectory', 
                       n_dim=2, use_expression=True, **kwargs):
    """
    Run principal curve analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    n_points : int, optional
        Number of points for trajectory interpolation
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    n_dim : int, optional
        Number of dimensions for PCA
    use_expression : bool, optional
        Whether to use gene expression data
    **kwargs
        Additional parameters for principal curve
    
    Returns
    -------
    adata : AnnData
        The AnnData object with principal curve results
    """
    try:
        from princurve import principal_curve
    except ImportError:
        logger.error("princurve package not installed. Please install it with 'pip install princurve'")
        raise ImportError("princurve package required for principal curve analysis")
    
    logger.info("Running principal curve analysis")
    coords = adata.obsm['spatial'].copy()
    if use_expression:
        if 'X_pca' in adata.obsm:
            logger.info("Using PCA components for expression data")
            expr_data = adata.obsm['X_pca'][:, :n_dim]
        else:
            logger.info("Computing PCA for expression data")
            sc.pp.pca(adata, n_comps=n_dim)
            expr_data = adata.obsm['X_pca'][:, :n_dim]
        from sklearn.preprocessing import StandardScaler
        coords_scaled = StandardScaler().fit_transform(coords)
        expr_scaled = StandardScaler().fit_transform(expr_data)
        fitting_data = np.hstack([coords_scaled, expr_scaled])
    else:
        fitting_data = coords
    curve_result = principal_curve(fitting_data, smoother='lowess', **kwargs)
    pseudotime = curve_result['lambda']
    if use_expression:
        curve_points_full = curve_result['s']
        curve_points = curve_points_full[:, :2]  # Get spatial dimensions only
        curve_points = curve_points * coords.std(axis=0) + coords.mean(axis=0)
    else:
        curve_points = curve_result['s']
    pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    from scipy.interpolate import interp1d
    sort_idx = np.argsort(pseudotime)
    curve_sorted = curve_points[sort_idx]
    t = np.linspace(0, 1, len(curve_sorted))
    fx = interp1d(t, curve_sorted[:, 0], kind='cubic')
    fy = interp1d(t, curve_sorted[:, 1], kind='cubic')
    t_new = np.linspace(0, 1, n_points)
    curve_interp = np.column_stack([fx(t_new), fy(t_new)])
    adata.obs[f'{key_added}_pseudotime'] = pseudotime_norm
    adata.uns[key_added] = {
        'method': 'principal_curve',
        'curve_points': curve_interp,
        'use_expression': use_expression,
        'n_dim': n_dim
    }
    logger.info("Principal curve analysis completed")
    logger.info(f"Pseudotime range: [{pseudotime_norm.min():.3f}, {pseudotime_norm.max():.3f}]")
    
    return adata

def _run_shortest_path(adata, start_spot=None, end_spot=None, n_points=100, 
                     key_added='spatial_trajectory', **kwargs):
    """
    Run shortest path analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    start_spot : str or int, optional
        Index of starting spot for trajectory
    end_spot : str or int, optional
        Index of ending spot for trajectory
    n_points : int, optional
        Number of points for trajectory interpolation
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    **kwargs
        Additional parameters for shortest path
    
    Returns
    -------
    adata : AnnData
        The AnnData object with shortest path results
    """
    import networkx as nx
    
    logger.info("Running shortest path analysis")
    if 'spatial_neighbors' not in adata.uns:
        logger.info("Building spatial neighbors graph")
        from scipy.spatial import Delaunay
        coords = adata.obsm['spatial']
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((simplex[i], simplex[j]))
        import scipy.sparse as sp
        adj_matrix = sp.lil_matrix((adata.n_obs, adata.n_obs))
        
        for i, j in edges:
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist
        adata.uns['spatial_neighbors'] = {}
        adata.obsp['spatial_connectivities'] = adj_matrix.tocsr()
        distances = sp.lil_matrix((adata.n_obs, adata.n_obs))
        distances[adj_matrix.nonzero()] = adj_matrix.data
        adata.obsp['spatial_distances'] = distances.tocsr()
    adj_matrix = adata.obsp['spatial_connectivities']
    try:
        G = nx.from_scipy_sparse_array(adj_matrix, edge_attribute='weight')
    except AttributeError:
        try:
            G = nx.from_scipy_sparse_matrix(adj_matrix, edge_attribute='weight')
        except AttributeError:
            raise ImportError("Your version of networkx doesn't support conversion from scipy sparse matrices. Please upgrade networkx.")
    if start_spot is None or end_spot is None:
        if 'X_pca' in adata.obsm and kwargs.get('use_pca', False):
            pc1 = adata.obsm['X_pca'][:, 0]
            min_idx = np.argmin(pc1)
            max_idx = np.argmax(pc1)
            
            start_spot = min_idx if start_spot is None else start_spot
            end_spot = max_idx if end_spot is None else end_spot
        else:
            coords = adata.obsm['spatial']
            x_min_idx = np.argmin(coords[:, 0])
            x_max_idx = np.argmax(coords[:, 0])
            
            start_spot = x_min_idx if start_spot is None else start_spot
            end_spot = x_max_idx if end_spot is None else end_spot
    if isinstance(start_spot, str):
        start_spot = np.where(adata.obs.index == start_spot)[0][0]
    
    if isinstance(end_spot, str):
        end_spot = np.where(adata.obs.index == end_spot)[0][0]
    
    logger.info(f"Computing shortest path from spot {start_spot} to spot {end_spot}")
    path = nx.shortest_path(G, source=start_spot, target=end_spot, weight='weight')
    coords = adata.obsm['spatial']
    path_coords = coords[path]
    from scipy.interpolate import interp1d
    distances = np.sqrt(np.sum(np.diff(path_coords, axis=0)**2, axis=1))
    cum_distances = np.concatenate([[0], np.cumsum(distances)])
    t = cum_distances / cum_distances[-1]
    fx = interp1d(t, path_coords[:, 0], kind='linear')
    fy = interp1d(t, path_coords[:, 1], kind='linear')
    t_new = np.linspace(0, 1, n_points)
    path_interp = np.column_stack([fx(t_new), fy(t_new)])
    pseudotime = np.zeros(adata.n_obs)
    
    for i in range(adata.n_obs):
        dists = np.sqrt(np.sum((path_coords - coords[i])**2, axis=1))
        min_idx = np.argmin(dists)
        if min_idx == 0:
            pseudotime[i] = dists[0]
        else:
            pseudotime[i] = cum_distances[min_idx] + dists[min_idx]
    pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    adata.obs[f'{key_added}_pseudotime'] = pseudotime_norm
    adata.uns[key_added] = {
        'method': 'shortest_path',
        'curve_points': path_interp,
        'path_indices': path,
        'start_spot': start_spot,
        'end_spot': end_spot
    }
    logger.info("Shortest path analysis completed")
    logger.info(f"Path length: {len(path)} spots")
    logger.info(f"Path distance: {cum_distances[-1]:.3f}")
    
    return adata

def _compute_trajectory_gene_trends(adata, key_added='spatial_trajectory', 
                                 n_top_genes=50, **kwargs):
    """
    Compute gene expression trends along the trajectory
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    n_top_genes : int, optional
        Number of top genes to compute trends for
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata.uns in place
    """
    logger.info(f"Computing gene trends along the trajectory for {n_top_genes} genes")
    pseudotime = adata.obs[f'{key_added}_pseudotime']
    gene_trends = []
    if 'highly_variable' in adata.var:
        var_genes = adata.var_names[adata.var['highly_variable']].tolist()
        logger.info(f"Using {len(var_genes)} highly variable genes")
    else:
        var_genes = adata.var_names.tolist()
    from scipy.stats import spearmanr
    
    for gene in var_genes:
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            expr = adata.X[:, gene_idx].toarray().flatten()
        else:
            expr = adata.X[:, gene_idx].copy()
        corr, pval = spearmanr(pseudotime, expr)
        gene_trends.append({
            'gene': gene,
            'correlation': corr,
            'p_value': pval
        })
    gene_trends_df = pd.DataFrame(gene_trends)
    from statsmodels.stats.multitest import multipletests
    gene_trends_df['p_adjusted'] = multipletests(gene_trends_df['p_value'], method='fdr_bh')[1]
    gene_trends_df['abs_correlation'] = np.abs(gene_trends_df['correlation'])
    gene_trends_df = gene_trends_df.sort_values('abs_correlation', ascending=False)
    adata.uns[f'{key_added}_trends'] = gene_trends_df
    sig_genes = gene_trends_df[gene_trends_df['p_adjusted'] < 0.05]
    logger.info(f"Found {len(sig_genes)} genes with significant trends (FDR < 0.05)")
    top_pos = gene_trends_df[gene_trends_df['correlation'] > 0].head(5)
    top_neg = gene_trends_df[gene_trends_df['correlation'] < 0].head(5)
    
    logger.info("Top increasing genes along trajectory:")
    for i, (_, row) in enumerate(top_pos.iterrows()):
        logger.info(f"  {i+1}. {row['gene']}: {row['correlation']:.3f} (p-adj={row['p_adjusted']:.2e})")
    
    logger.info("Top decreasing genes along trajectory:")
    for i, (_, row) in enumerate(top_neg.iterrows()):
        logger.info(f"  {i+1}. {row['gene']}: {row['correlation']:.3f} (p-adj={row['p_adjusted']:.2e})")
    _fit_smoothed_gene_trends(adata, gene_trends_df, key_added, n_top_genes, **kwargs)

def _fit_smoothed_gene_trends(adata, gene_trends_df, key_added='spatial_trajectory', 
                            n_top_genes=50, n_points=100, **kwargs):
    """
    Fit smoothed gene expression trends along the trajectory
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    gene_trends_df : pandas.DataFrame
        DataFrame with gene trend statistics
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    n_top_genes : int, optional
        Number of top genes to compute trends for
    n_points : int, optional
        Number of points for trend interpolation
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata.uns in place
    """
    logger.info(f"Fitting smoothed gene trends for top {n_top_genes} genes")
    pseudotime = adata.obs[f'{key_added}_pseudotime'].values
    top_genes = gene_trends_df.head(n_top_genes)['gene'].tolist()
    trend_matrix = np.zeros((n_points, len(top_genes)))
    t_points = np.linspace(0, 1, n_points)
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    for i, gene in enumerate(top_genes):
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            expr = adata.X[:, gene_idx].toarray().flatten()
        else:
            expr = adata.X[:, gene_idx]
        smoothed = lowess(expr, pseudotime, frac=0.2, it=2)
        from scipy.interpolate import interp1d
        f = interp1d(smoothed[:, 0], smoothed[:, 1], bounds_error=False, fill_value='extrapolate')
        trend = f(t_points)
        trend = (trend - trend.min()) / (trend.max() - trend.min() + 1e-10)
        trend_matrix[:, i] = trend
    adata.uns[f'{key_added}_smoothed_trends'] = {
        'genes': top_genes,
        'trends': trend_matrix,
        'pseudotime_points': t_points
    }
    
    logger.info("Smoothed gene trends computed successfully")

def compute_branched_trajectory(adata, cluster_key, start_cluster=None, 
                              branch_clusters=None, key_added='branched_trajectory', 
                              **kwargs):
    """
    Compute a branched trajectory based on clusters
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    start_cluster : str, optional
        Starting cluster for the trajectory
    branch_clusters : list, optional
        List of terminal cluster for the trajectory branches
    key_added : str, optional
        Key to add to adata.uns for trajectory results
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with branched trajectory results
    """
    logger.info("Computing branched trajectory")
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    clusters = adata.obs[cluster_key]
    unique_clusters = clusters.cat.categories.tolist()
    if start_cluster is None:
        from networkx.algorithms.centrality import closeness_centrality
        G = _build_cluster_graph(adata, cluster_key)
        centrality = closeness_centrality(G)
        start_cluster = max(centrality, key=centrality.get)
        logger.info(f"Selected {start_cluster} as start cluster based on centrality")
    if branch_clusters is None:
        from collections import Counter
        G = _build_cluster_graph(adata, cluster_key)
        degrees = dict(G.degree())
        branch_clusters = [c for c, d in degrees.items() if d == 1]
        
        if not branch_clusters:
            import networkx as nx
            distances = nx.single_source_shortest_path_length(G, start_cluster)
            max_dist = max(distances.values())
            branch_clusters = [c for c, d in distances.items() if d == max_dist]
        
        logger.info(f"Selected {len(branch_clusters)} branch clusters: {branch_clusters}")
    coords = adata.obsm['spatial']
    centroids = {}
    
    for cluster in unique_clusters:
        mask = clusters == cluster
        centroids[cluster] = coords[mask].mean(axis=0)
    branches = []
    
    for i, end_cluster in enumerate(branch_clusters):
        logger.info(f"Computing trajectory branch from {start_cluster} to {end_cluster}")
        G = _build_cluster_graph(adata, cluster_key)
        
        import networkx as nx
        try:
            path = nx.shortest_path(G, source=start_cluster, target=end_cluster)
            logger.info(f"Branch path: {' -> '.join(path)}")
            path_coords = np.array([centroids[c] for c in path])
            from scipy.interpolate import interp1d
            distances = np.sqrt(np.sum(np.diff(path_coords, axis=0)**2, axis=1))
            cum_distances = np.concatenate([[0], np.cumsum(distances)])
            t = cum_distances / cum_distances[-1]
            fx = interp1d(t, path_coords[:, 0], kind='linear')
            fy = interp1d(t, path_coords[:, 1], kind='linear')
            n_points = kwargs.get('n_points', 100)
            t_new = np.linspace(0, 1, n_points)
            path_interp = np.column_stack([fx(t_new), fy(t_new)])
            branches.append({
                'branch_id': i,
                'start_cluster': start_cluster,
                'end_cluster': end_cluster,
                'path': path,
                'coords': path_interp
            })
        
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {start_cluster} to {end_cluster}")
    pseudotime = np.zeros((adata.n_obs, len(branches)))
    branch_assignment = np.zeros(adata.n_obs, dtype=int)
    
    for i, branch in enumerate(branches):
        branch_coords = branch['coords']
        
        for j in range(adata.n_obs):
            dists = np.sqrt(np.sum((branch_coords - coords[j])**2, axis=1))
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            pt = min_idx / (len(branch_coords) - 1)
            pseudotime[j, i] = pt
            if i == 0 or min_dist < min(np.sqrt(np.sum((branch['coords'] - coords[j])**2, axis=1)) for branch in branches[:i]):
                branch_assignment[j] = i
    adata.obs[f'{key_added}_branch'] = pd.Categorical([str(b) for b in branch_assignment])
    
    for i in range(len(branches)):
        adata.obs[f'{key_added}_branch{i}_pseudotime'] = pseudotime[:, i]
    adata.uns[key_added] = {
        'method': 'branched',
        'branches': branches,
        'start_cluster': start_cluster,
        'branch_clusters': branch_clusters,
        'cluster_key': cluster_key
    }
    
    logger.info(f"Branched trajectory analysis completed with {len(branches)} branches")
    
    return adata

def _build_cluster_graph(adata, cluster_key):
    """
    Build a graph of cluster connections
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    
    Returns
    -------
    networkx.Graph
        Graph of cluster connections
    """
    import networkx as nx
    clusters = adata.obs[cluster_key]
    unique_clusters = clusters.cat.categories.tolist()
    G = nx.Graph()
    G.add_nodes_from(unique_clusters)
    if 'spatial_neighbors' not in adata.uns:
        from scipy.spatial import Delaunay
        coords = adata.obsm['spatial']
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((simplex[i], simplex[j]))
        import scipy.sparse as sp
        adj_matrix = sp.lil_matrix((adata.n_obs, adata.n_obs))
        
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    else:
        adj_matrix = adata.obsp['spatial_connectivities']
    cluster_connections = {}
    
    for i in range(adata.n_obs):
        c1 = clusters.iloc[i]
        neighbors = adj_matrix[i].nonzero()[1]
        
        for j in neighbors:
            c2 = clusters.iloc[j]
            if c1 == c2:
                continue
            key = tuple(sorted([c1, c2]))
            if key in cluster_connections:
                cluster_connections[key] += 1
            else:
                cluster_connections[key] = 1
    for (c1, c2), count in cluster_connections.items():
        G.add_edge(c1, c2, weight=count)
    
    return G

def compute_diffusion_pseudotime(adata, root_spots=None, key_added='dpt', 
                              use_rep='X_diffmap', **kwargs):
    """
    Compute diffusion pseudotime
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    root_spots : str, int, or list, optional
        Index or indices of root spots
    key_added : str, optional
        Key to add to adata.obs for DPT results
    use_rep : str, optional
        Representation to use for DPT
    **kwargs
        Additional parameters for sc.tl.dpt
    
    Returns
    -------
    adata : AnnData
        The AnnData object with DPT results
    """
    logger.info("Computing diffusion pseudotime")
    if use_rep == 'X_diffmap' and 'X_diffmap' not in adata.obsm:
        logger.info("Computing diffusion maps")
        sc.pp.neighbors(adata, **kwargs)
        sc.tl.diffmap(adata, **kwargs)
    if root_spots is None:
        if 'X_diffmap' in adata.obsm:
            dc1 = adata.obsm['X_diffmap'][:, 0]
            root_spots = [np.argmin(dc1)]
            logger.info(f"Selected spot {root_spots[0]} as root based on diffusion map")
        else:
            root_spots = [0]
            logger.info(f"Using spot {root_spots[0]} as root (default)")
    if not isinstance(root_spots, list):
        root_spots = [root_spots]
    for i, spot in enumerate(root_spots):
        if isinstance(spot, str):
            root_spots[i] = np.where(adata.obs.index == spot)[0][0]
    sc.tl.dpt(adata, n_dcs=10, root=root_spots[0], **kwargs)
    if key_added != 'dpt':
        adata.obs[f'{key_added}_pseudotime'] = adata.obs['dpt_pseudotime']
        adata.uns[key_added] = {
            'method': 'dpt',
            'root_spots': root_spots,
            'use_rep': use_rep
        }
    
    logger.info("Diffusion pseudotime computed successfully")
    
    return adata

def run_velocity_analysis(adata, velocity_data=None, key_added='velocity', **kwargs):
    """
    Run RNA velocity analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    velocity_data : AnnData, optional
        AnnData object with spliced/unspliced counts
    key_added : str, optional
        Key to add to adata.obsm for velocity results
    **kwargs
        Additional parameters for velocity calculation
    
    Returns
    -------
    adata : AnnData
        The AnnData object with velocity results
    """
    try:
        import scvelo as scv
    except ImportError:
        logger.error("scVelo package not installed. Please install it with 'pip install scvelo'")
        raise ImportError("scVelo package required for velocity analysis")
    
    logger.info("Running RNA velocity analysis")
    if velocity_data is None:
        logger.error("No velocity data provided")
        raise ValueError("Velocity data required for velocity analysis")
    common_barcodes = set(adata.obs.index) & set(velocity_data.obs.index)
    
    if not common_barcodes:
        logger.error("No common barcodes between spatial and velocity data")
        raise ValueError("No common barcodes between spatial and velocity data")
    
    logger.info(f"Found {len(common_barcodes)} common barcodes")
    adata_sub = adata[list(common_barcodes)].copy()
    velocity_data_sub = velocity_data[list(common_barcodes)].copy()
    velocity_data_sub = velocity_data_sub[adata_sub.obs.index]
    required_layers = ['spliced', 'unspliced']
    missing_layers = [layer for layer in required_layers if layer not in velocity_data_sub.layers]
    
    if missing_layers:
        logger.error(f"Missing required layers: {missing_layers}")
        raise ValueError(f"Velocity data must contain 'spliced' and 'unspliced' layers")
    for layer in required_layers:
        adata_sub.layers[layer] = velocity_data_sub.layers[layer]
    logger.info("Running scVelo preprocessing")
    scv.pp.filter_and_normalize(adata_sub, **kwargs.get('pp_kwargs', {}))
    scv.pp.moments(adata_sub, **kwargs.get('moments_kwargs', {}))
    
    logger.info("Computing RNA velocity")
    scv.tl.velocity(adata_sub, **kwargs.get('velocity_kwargs', {}))
    scv.tl.velocity_graph(adata_sub, **kwargs.get('graph_kwargs', {}))
    logger.info("Projecting velocity to embedding")
    if 'X_umap' in adata_sub.obsm:
        scv.tl.velocity_embedding(adata_sub, basis='umap')
    
    if 'X_pca' in adata_sub.obsm:
        scv.tl.velocity_embedding(adata_sub, basis='pca')
    
    if 'spatial' in adata_sub.obsm:
        scv.tl.velocity_embedding(adata_sub, basis='spatial')
    logger.info("Computing velocity pseudotime")
    scv.tl.velocity_pseudotime(adata_sub)
    transfer_keys = {
        'obsm': ['velocity_umap', 'velocity_pca', 'velocity_spatial'],
        'obs': ['velocity_pseudotime'],
        'uns': ['velocity_graph', 'velocity_params'],
        'varm': ['velocity'],
        'layers': ['velocity']
    }
    adata_result = adata.copy()
    for layer in required_layers:
        layer_data = np.zeros((adata.n_obs, adata.n_vars))
        
        common_barcodes_idx = [adata.obs.index.get_loc(bc) for bc in common_barcodes]
        
        for i, bc_idx in enumerate(common_barcodes_idx):
            layer_data[bc_idx] = velocity_data_sub.layers[layer][i]
        
        adata_result.layers[layer] = layer_data
    for key_type, keys in transfer_keys.items():
        for key in keys:
            if hasattr(adata_sub, key_type) and key in getattr(adata_sub, key_type):
                if key_type == 'obs':
                    data = getattr(adata_sub, key_type)[key]
                    adata_result.obs[key] = np.nan
                    
                    for i, bc in enumerate(adata_sub.obs.index):
                        idx = adata_result.obs.index.get_loc(bc)
                        adata_result.obs[key].iloc[idx] = data.iloc[i]
                
                elif key_type == 'obsm':
                    if key in adata_sub.obsm:
                        data = np.zeros((adata.n_obs, adata_sub.obsm[key].shape[1]))
                        
                        for i, bc in enumerate(adata_sub.obs.index):
                            idx = adata_result.obs.index.get_loc(bc)
                            data[idx] = adata_sub.obsm[key][i]
                        
                        adata_result.obsm[key] = data
                
                elif key_type == 'varm':
                    if key in adata_sub.varm:
                        data = np.zeros((adata.n_vars, adata_sub.varm[key].shape[1]))
                        
                        for i, gene in enumerate(adata_sub.var_names):
                            if gene in adata_result.var_names:
                                idx = adata_result.var_names.get_loc(gene)
                                data[idx] = adata_sub.varm[key][i]
                        
                        adata_result.varm[key] = data
                
                elif key_type == 'uns':
                    adata_result.uns[key] = getattr(adata_sub, 'uns')[key]
                
                elif key_type == 'layers':
                    if key in adata_sub.layers:
                        data = np.zeros((adata.n_obs, adata.n_vars))
                        
                        for i, bc in enumerate(adata_sub.obs.index):
                            idx = adata_result.obs.index.get_loc(bc)
                            data[idx] = adata_sub.layers[key][i]
                        
                        adata_result.layers[key] = data
    if key_added != 'velocity':
        if 'velocity_umap' in adata_result.obsm:
            adata_result.obsm[f'{key_added}_umap'] = adata_result.obsm['velocity_umap'].copy()
        
        if 'velocity_pca' in adata_result.obsm:
            adata_result.obsm[f'{key_added}_pca'] = adata_result.obsm['velocity_pca'].copy()
        
        if 'velocity_spatial' in adata_result.obsm:
            adata_result.obsm[f'{key_added}_spatial'] = adata_result.obsm['velocity_spatial'].copy()
        
        if 'velocity_pseudotime' in adata_result.obs:
            adata_result.obs[f'{key_added}_pseudotime'] = adata_result.obs['velocity_pseudotime'].copy()
    adata_result.uns[f'{key_added}_info'] = {
        'method': 'scvelo',
        'common_barcodes': len(common_barcodes),
        'total_barcodes': adata.n_obs,
        'has_spatial_projection': 'velocity_spatial' in adata_result.obsm
    }
    
    logger.info("RNA velocity analysis completed")
    
    return adata_result
