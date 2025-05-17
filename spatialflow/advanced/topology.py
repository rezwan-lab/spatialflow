import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import logging
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import warnings

logger = logging.getLogger('spatialflow.advanced.topology')

def run_tda_analysis(adata, method="persistent_homology", n_components=2, key_added=None, **kwargs):
    """
    Run topological data analysis (TDA)
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates
    method : str, optional
        TDA method to use. Options: 'persistent_homology', 'mapper', 'neighborhood_graph'
    n_components : int, optional
        Number of dimensions for dimension reduction before TDA
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional arguments for specific TDA methods
        
    Returns
    -------
    adata : AnnData
        AnnData object with added TDA results
    """
    logger.info(f"Running topological data analysis using {method} method")
    
    if key_added is None:
        key_added = f"tda_{method}"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    coords = adata.obsm['spatial']
    if method in ['mapper', 'persistent_homology'] and kwargs.get('use_expression', False):
        logger.info("Using gene expression data for TDA")
        if 'X_pca' not in adata.obsm:
            logger.info(f"Computing PCA with {n_components} components")
            sc.pp.pca(adata, n_comps=n_components)
        X = adata.obsm['X_pca'][:, :n_components]
        
        if kwargs.get('use_spatial', True):
            X = np.hstack([X, coords])
            logger.info("Combined gene expression PCA and spatial coordinates")
    else:
        X = coords
        logger.info("Using only spatial coordinates for TDA")
    if method == "persistent_homology":
        _run_persistent_homology(adata, X, key_added, **kwargs)
    elif method == "mapper":
        _run_mapper(adata, X, key_added, **kwargs)
    elif method == "neighborhood_graph":
        _run_neighborhood_graph(adata, X, key_added, **kwargs)
    else:
        logger.error(f"Unsupported TDA method: {method}")
        raise ValueError(f"Unsupported TDA method: {method}")
    
    return adata

def _run_persistent_homology(adata, X, key_added, max_dim=1, **kwargs):
    """
    Run persistent homology analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix for TDA
    key_added : str
        Key for storing results in adata.uns
    max_dim : int, optional
        Maximum homology dimension
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        import gudhi
        from gudhi.rips_complex import RipsComplex
        from gudhi.persistence_graphical_tools import plot_persistence_barcode
    except ImportError:
        try:
            from ripser import ripser
        except ImportError:
            logger.error("Neither gudhi nor ripser package is installed. "
                        "Please install one with 'pip install gudhi' or 'pip install ripser'")
            raise ImportError("gudhi or ripser required for persistent homology")
    
    logger.info(f"Running persistent homology analysis with max dimension {max_dim}")
    if kwargs.get('precomputed_distances', False):
        distances = X
    else:
        distances = squareform(pdist(X, metric=kwargs.get('metric', 'euclidean')))
    try:
        rips = RipsComplex(distance_matrix=distances)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dim+1)
        persistence = simplex_tree.persistence()
        diagrams = {}
        for dim in range(max_dim + 1):
            pairs = [(b, d) for (d_tmp, b, d) in persistence if d_tmp == dim and d != float('inf')]
            diagrams[dim] = np.array(pairs) if pairs else np.zeros((0, 2))
        adata.uns[key_added] = {
            'method': 'persistent_homology_gudhi',
            'diagrams': diagrams,
            'parameters': {
                'max_dim': max_dim,
                'metric': kwargs.get('metric', 'euclidean')
            }
        }
        fig_dir = kwargs.get('fig_dir', None)
        if fig_dir is not None:
            for dim in range(max_dim + 1):
                plot_persistence_barcode(persistence, alpha=0.8, legend=True, axes=plt.gca())
                plt.title(f"Persistence Barcode (Dimension {dim})")
                plt.tight_layout()
                fig_path = f"{fig_dir}/persistence_barcode_dim{dim}.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved persistence barcode plot for dimension {dim} to {fig_path}")
        
    except (ImportError, NameError):
        logger.info("Using ripser for persistent homology")
        
        result = ripser(distances, maxdim=max_dim, distance_matrix=True)
        diagrams = {dim: dgm for dim, dgm in enumerate(result['dgms'])}
        adata.uns[key_added] = {
            'method': 'persistent_homology_ripser',
            'diagrams': diagrams,
            'parameters': {
                'max_dim': max_dim,
                'metric': kwargs.get('metric', 'euclidean')
            }
        }
        fig_dir = kwargs.get('fig_dir', None)
        if fig_dir is not None:
            try:
                from persim import plot_diagrams
                
                plt.figure(figsize=(10, 8))
                plot_diagrams(result['dgms'], show=False)
                plt.title("Persistence Diagrams")
                plt.tight_layout()
                fig_path = f"{fig_dir}/persistence_diagrams.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved persistence diagram plot to {fig_path}")
            except ImportError:
                logger.warning("persim package not installed, could not create persistence diagram plot")
    persistence_stats = {}
    for dim, dgm in diagrams.items():
        if len(dgm) > 0:
            persistence_lengths = dgm[:, 1] - dgm[:, 0]
            persistence_stats[dim] = {
                'count': len(dgm),
                'mean_persistence': float(np.mean(persistence_lengths)),
                'max_persistence': float(np.max(persistence_lengths)),
                'total_persistence': float(np.sum(persistence_lengths)),
                'persistence_entropy': float(_calculate_persistence_entropy(persistence_lengths))
            }
        else:
            persistence_stats[dim] = {
                'count': 0,
                'mean_persistence': 0.0,
                'max_persistence': 0.0,
                'total_persistence': 0.0,
                'persistence_entropy': 0.0
            }
    adata.uns[key_added]['statistics'] = persistence_stats
    if max_dim >= 1:
        for dim in range(max_dim + 1):
            adata.obs[f'betti_{dim}'] = persistence_stats[dim]['count']
        for dim in range(max_dim + 1):
            adata.obs[f'total_persistence_{dim}'] = persistence_stats[dim]['total_persistence']
    
    logger.info("Persistent homology analysis completed")
    logger.info(f"Topological features added to adata.obs: {[f'betti_{dim}' for dim in range(max_dim + 1)]}")
    logger.info(f"Persistence statistics stored in adata.uns['{key_added}']['statistics']")

def _calculate_persistence_entropy(persistence_lengths):
    """
    Calculate the entropy of persistence lengths
    
    Parameters
    ----------
    persistence_lengths : numpy.ndarray
        Array of persistence lengths (death - birth)
        
    Returns
    -------
    float
        Persistence entropy
    """
    if len(persistence_lengths) == 0:
        return 0.0
    total = np.sum(persistence_lengths)
    if total == 0:
        return 0.0
    
    probabilities = persistence_lengths / total
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

def _run_mapper(adata, X, key_added, n_intervals=10, overlap=0.3, clusterer=None, **kwargs):
    """
    Run Mapper algorithm for topological data analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix for TDA
    key_added : str
        Key for storing results in adata.uns
    n_intervals : int, optional
        Number of intervals for the filter function
    overlap : float, optional
        Percentage of overlap between intervals
    clusterer : object, optional
        Clustering algorithm to use (should have fit_predict method)
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        import kmapper as km
    except ImportError:
        logger.error("kmapper package not installed. Please install it with 'pip install kmapper'")
        raise ImportError("kmapper required for Mapper algorithm")
    
    logger.info("Running Mapper algorithm")
    if clusterer is None:
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
    filter_function = kwargs.get('filter_function', None)
    if filter_function is None:
        from sklearn.decomposition import PCA
        
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components)
        filter_function = lambda X: pca.fit_transform(X)
    mapper = km.KeplerMapper(verbose=0)
    projected_data = mapper.fit_transform(X, projection=filter_function)
    graph = mapper.map(
        projected_data,
        X,
        cover=km.Cover(n_cubes=n_intervals, perc_overlap=overlap),
        clusterer=clusterer
    )
    adata.uns[key_added] = {
        'method': 'mapper',
        'graph': {
            'nodes': {str(k): list(v) for k, v in graph['nodes'].items()},
            'links': graph['links'],
            'meta': graph['meta']
        },
        'parameters': {
            'n_intervals': n_intervals,
            'overlap': overlap,
            'clusterer': str(clusterer)
        }
    }
    node_assignments = {}
    for node_id, members in graph['nodes'].items():
        for member in members:
            if member not in node_assignments:
                node_assignments[member] = []
            node_assignments[member].append(node_id)
    mapper_labels = np.zeros(adata.n_obs, dtype=int)
    for i in range(adata.n_obs):
        if i in node_assignments:
            mapper_labels[i] = int(node_assignments[i][0])
    adata.obs[f'mapper_clusters'] = pd.Categorical(mapper_labels.astype(str))
    fig_dir = kwargs.get('fig_dir', None)
    if fig_dir is not None:
        html = mapper.visualize(
            graph,
            title=f"Mapper Graph",
            color_function=mapper_labels
        )
        html_path = f"{fig_dir}/mapper_graph.html"
        with open(html_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Saved Mapper graph visualization to {html_path}")
        if 'spatial' in adata.obsm:
            plt.figure(figsize=(10, 8))
            coords = adata.obsm['spatial']
            
            plt.scatter(coords[:, 0], coords[:, 1], c=mapper_labels, cmap='tab20', s=30, alpha=0.8)
            plt.title("Mapper Clusters")
            plt.xlabel("Spatial X")
            plt.ylabel("Spatial Y")
            plt.colorbar(label="Cluster")
            plt.tight_layout()
            fig_path = f"{fig_dir}/mapper_clusters.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved Mapper clusters plot to {fig_path}")
    
    logger.info(f"Mapper analysis completed with {len(graph['nodes'])} nodes and {len(graph['links'])} edges")
    logger.info(f"Mapper clusters added to adata.obs['mapper_clusters']")

def _run_neighborhood_graph(adata, X, key_added, n_neighbors=10, **kwargs):
    """
    Run neighborhood graph analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix for TDA
    key_added : str
        Key for storing results in adata.uns
    n_neighbors : int, optional
        Number of neighbors for graph construction
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph
    
    logger.info(f"Running neighborhood graph analysis with {n_neighbors} neighbors")
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    G = nx.from_scipy_sparse_matrix(knn_graph)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    connected_components = list(nx.connected_components(G))
    diameters = {}
    for i, component in enumerate(connected_components):
        if len(component) > 1:
            subgraph = G.subgraph(component)
            try:
                diameters[i] = nx.diameter(subgraph)
            except nx.NetworkXError:
                diameters[i] = float('inf')
    adata.uns[key_added] = {
        'method': 'neighborhood_graph',
        'graph': G,
        'n_components': len(connected_components),
        'component_sizes': [len(c) for c in connected_components],
        'diameters': diameters,
        'parameters': {
            'n_neighbors': n_neighbors
        }
    }
    adata.obs[f'degree_centrality'] = pd.Series(degree_centrality)
    adata.obs[f'betweenness_centrality'] = pd.Series(betweenness_centrality)
    adata.obs[f'closeness_centrality'] = pd.Series(closeness_centrality)
    component_labels = np.zeros(adata.n_obs, dtype=int)
    for i, component in enumerate(connected_components):
        for node in component:
            component_labels[node] = i
    
    adata.obs[f'graph_component'] = pd.Categorical(component_labels.astype(str))
    fig_dir = kwargs.get('fig_dir', None)
    if fig_dir is not None and 'spatial' in adata.obsm:
        plt.figure(figsize=(10, 8))
        coords = adata.obsm['spatial']
        plt.scatter(coords[:, 0], coords[:, 1], c=component_labels, cmap='tab20', s=30, alpha=0.8)
        for (u, v) in G.edges():
            plt.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], 'k-', alpha=0.1, linewidth=0.5)
        
        plt.title("Neighborhood Graph Components")
        plt.xlabel("Spatial X")
        plt.ylabel("Spatial Y")
        plt.colorbar(label="Component")
        plt.tight_layout()
        fig_path = f"{fig_dir}/neighborhood_graph.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved neighborhood graph plot to {fig_path}")
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
        titles = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            scatter = axs[i].scatter(coords[:, 0], coords[:, 1], c=adata.obs[metric], cmap='viridis', s=30, alpha=0.8)
            axs[i].set_title(title)
            axs[i].set_xlabel("Spatial X")
            axs[i].set_ylabel("Spatial Y")
            plt.colorbar(scatter, ax=axs[i])
        
        plt.tight_layout()
        fig_path = f"{fig_dir}/centrality_metrics.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved centrality metrics plot to {fig_path}")
    
    logger.info(f"Neighborhood graph analysis completed with {len(connected_components)} connected components")
    logger.info(f"Centrality metrics added to adata.obs: {['degree_centrality', 'betweenness_centrality', 'closeness_centrality']}")
    logger.info(f"Component labels added to adata.obs['graph_component']")

def run_topological_features(adata, method="persistent_homology", use_expression=True,
                            n_components=10, key_added=None, **kwargs):
    """
    Extract topological features from spatial data
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates
    method : str, optional
        Method to use for extracting topological features
    use_expression : bool, optional
        Whether to use gene expression data
    n_components : int, optional
        Number of components for dimension reduction
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional arguments
        
    Returns
    -------
    adata : AnnData
        AnnData object with extracted topological features
    """
    logger.info(f"Extracting topological features using {method} method")
    
    if key_added is None:
        key_added = f"topo_features_{method}"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    coords = adata.obsm['spatial']
    if use_expression:
        if 'X_pca' not in adata.obsm:
            logger.info(f"Computing PCA with {n_components} components")
            sc.pp.pca(adata, n_comps=n_components)
        X = adata.obsm['X_pca'][:, :min(n_components, adata.obsm['X_pca'].shape[1])]
    else:
        X = coords
    if method == "persistent_homology":
        _extract_persistent_homology_features(adata, X, key_added, **kwargs)
    elif method == "graph_features":
        _extract_graph_features(adata, X, key_added, **kwargs)
    elif method == "filtration":
        _extract_filtration_features(adata, X, key_added, **kwargs)
    else:
        logger.error(f"Unsupported feature extraction method: {method}")
        raise ValueError(f"Unsupported feature extraction method: {method}")
    
    return adata

def _extract_persistent_homology_features(adata, X, key_added, max_dim=1, **kwargs):
    """
    Extract features from persistent homology
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix
    key_added : str
        Key for storing results in adata.uns
    max_dim : int, optional
        Maximum homology dimension
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        from ripser import ripser
    except ImportError:
        logger.error("ripser package not installed. Please install it with 'pip install ripser'")
        raise ImportError("ripser required for persistent homology features")
    
    logger.info(f"Extracting persistent homology features with max dimension {max_dim}")
    result = ripser(X, maxdim=max_dim, distance_matrix=kwargs.get('distance_matrix', False))
    diagrams = result['dgms']
    features = {}
    
    for dim in range(len(diagrams)):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            features[f'dim{dim}_count'] = 0
            features[f'dim{dim}_sum_persistence'] = 0
            features[f'dim{dim}_max_persistence'] = 0
            features[f'dim{dim}_mean_persistence'] = 0
            features[f'dim{dim}_std_persistence'] = 0
            continue
        finite_dgm = dgm[dgm[:, 1] != np.inf]
        
        if len(finite_dgm) == 0:
            persistence = np.zeros(0)
        else:
            persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
        features[f'dim{dim}_count'] = len(dgm)
        features[f'dim{dim}_sum_persistence'] = np.sum(persistence) if len(persistence) > 0 else 0
        features[f'dim{dim}_max_persistence'] = np.max(persistence) if len(persistence) > 0 else 0
        features[f'dim{dim}_mean_persistence'] = np.mean(persistence) if len(persistence) > 0 else 0
        features[f'dim{dim}_std_persistence'] = np.std(persistence) if len(persistence) > 0 else 0
        if len(finite_dgm) > 0:
            features[f'dim{dim}_mean_birth'] = np.mean(finite_dgm[:, 0])
            features[f'dim{dim}_mean_death'] = np.mean(finite_dgm[:, 1])
            features[f'dim{dim}_std_birth'] = np.std(finite_dgm[:, 0])
            features[f'dim{dim}_std_death'] = np.std(finite_dgm[:, 1])
        else:
            features[f'dim{dim}_mean_birth'] = 0
            features[f'dim{dim}_mean_death'] = 0
            features[f'dim{dim}_std_birth'] = 0
            features[f'dim{dim}_std_death'] = 0
        if len(persistence) > 0:
            features[f'dim{dim}_persistence_entropy'] = _calculate_persistence_entropy(persistence)
        else:
            features[f'dim{dim}_persistence_entropy'] = 0
    for feature, value in features.items():
        adata.obs[f'topo_{feature}'] = value
    adata.uns[key_added] = {
        'method': 'persistent_homology_features',
        'features': features,
        'parameters': {
            'max_dim': max_dim
        }
    }
    
    logger.info(f"Added {len(features)} topological features to adata.obs")
    logger.info(f"Features added with prefix 'topo_'")

def _extract_graph_features(adata, X, key_added, n_neighbors=5, **kwargs):
    """
    Extract features from neighborhood graphs
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix
    key_added : str
        Key for storing results in adata.uns
    n_neighbors : int, optional
        Number of neighbors for graph construction
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph
    
    logger.info(f"Extracting graph features with {n_neighbors} neighbors")
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')
    G = nx.from_scipy_sparse_matrix(knn_graph)
    features = {}
    features['degree'] = dict(G.degree())
    features['clustering'] = nx.clustering(G)
    features['degree_centrality'] = nx.degree_centrality(G)
    features['betweenness_centrality'] = nx.betweenness_centrality(G)
    features['closeness_centrality'] = nx.closeness_centrality(G)
    features['local_efficiency'] = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subgraph = G.subgraph(neighbors)
            if nx.number_of_nodes(subgraph) > 0:
                features['local_efficiency'][node] = nx.local_efficiency(subgraph)
            else:
                features['local_efficiency'][node] = 0
        else:
            features['local_efficiency'][node] = 0
    for feature, values in features.items():
        if isinstance(values, dict):
            adata.obs[f'graph_{feature}'] = pd.Series(values)
    adata.uns[key_added] = {
        'method': 'graph_features',
        'graph': G,
        'parameters': {
            'n_neighbors': n_neighbors
        }
    }
    
    logger.info(f"Added graph features to adata.obs: {[f'graph_{feature}' for feature in features.keys()]}")

def _extract_filtration_features(adata, X, key_added, max_radius=None, n_steps=20, **kwargs):
    """
    Extract features from filtration
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    X : numpy.ndarray
        Data matrix
    key_added : str
        Key for storing results in adata.uns
    max_radius : float, optional
        Maximum radius for filtration
    n_steps : int, optional
        Number of steps in filtration
    **kwargs
        Additional arguments
        
    Returns
    -------
    None
        Modifies adata in place
    """
    import networkx as nx
    from sklearn.neighbors import radius_neighbors_graph
    
    logger.info("Extracting filtration features")
    distances = squareform(pdist(X, metric='euclidean'))
    if max_radius is None:
        max_radius = np.percentile(distances, 50)  # Median distance
    
    radii = np.linspace(0, max_radius, n_steps)
    features = {
        'connected_components': np.zeros(n_steps),
        'largest_component_size': np.zeros(n_steps),
        'average_degree': np.zeros(n_steps),
        'average_clustering': np.zeros(n_steps),
        'edge_density': np.zeros(n_steps)
    }
    for i, radius in enumerate(radii):
        adj_matrix = (distances <= radius).astype(int)
        np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
        G = nx.from_numpy_array(adj_matrix)
        connected_components = list(nx.connected_components(G))
        features['connected_components'][i] = len(connected_components)
        
        if connected_components:
            largest_component_size = max(len(c) for c in connected_components)
            features['largest_component_size'][i] = largest_component_size / adata.n_obs
        else:
            features['largest_component_size'][i] = 0

            if G.number_of_edges() > 0:
                features['average_degree'][i] = np.mean([d for _, d in G.degree()])
                features['edge_density'][i] = 2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                clustering = nx.clustering(G)
                if clustering:
                    features['average_clustering'][i] = np.mean(list(clustering.values()))
                else:
                    features['average_clustering'][i] = 0
            else:
                features['average_degree'][i] = 0
                features['edge_density'][i] = 0
                features['average_clustering'][i] = 0
        thresholds = [0.25, 0.5, 0.75]  # Thresholds as fraction of max radius
        threshold_features = {}
        
        for threshold in thresholds:
            idx = np.argmin(np.abs(radii - threshold * max_radius))
            radius = radii[idx]
            
            for feature, values in features.items():
                threshold_features[f'{feature}_r{int(threshold*100)}'] = values[idx]
        for feature, values in features.items():
            from sklearn.linear_model import LinearRegression
            
            X_reg = radii.reshape(-1, 1)
            y_reg = values.reshape(-1, 1)
            
            model = LinearRegression().fit(X_reg, y_reg)
            threshold_features[f'{feature}_slope'] = float(model.coef_[0][0])
        for feature, value in threshold_features.items():
            adata.obs[f'filt_{feature}'] = value
        adata.uns[key_added] = {
            'method': 'filtration_features',
            'features': features,
            'threshold_features': threshold_features,
            'radii': radii.tolist(),
            'parameters': {
                'max_radius': max_radius,
                'n_steps': n_steps
            }
        }
        fig_dir = kwargs.get('fig_dir', None)
        if fig_dir is not None:
            plt.figure(figsize=(12, 8))
            
            for feature, values in features.items():
                plt.plot(radii, values, 'o-', label=feature)
            
            plt.xlabel('Radius')
            plt.ylabel('Feature Value')
            plt.title('Filtration Features')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = f"{fig_dir}/filtration_features.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved filtration features plot to {fig_path}")
        
        logger.info(f"Added filtration features to adata.obs with prefix 'filt_'")
        logger.info(f"Total of {len(threshold_features)} features added")

    def run_topology_comparison(adata, cluster_key, tda_method="persistent_homology", key_added=None, **kwargs):
        """
        Compare topological features across clusters
        
        Parameters
        ----------
        adata : AnnData
            AnnData object with spatial coordinates
        cluster_key : str
            Key in adata.obs for cluster assignments
        tda_method : str, optional
            TDA method to use for comparison
        key_added : str, optional
            Key for storing results in adata.uns
        **kwargs
            Additional arguments
            
        Returns
        -------
        adata : AnnData
            AnnData object with added comparison results
        """
        logger.info(f"Comparing topological features across clusters defined by {cluster_key}")
        
        if key_added is None:
            key_added = f"topology_comparison_{cluster_key}"
        if cluster_key not in adata.obs.columns:
            logger.error(f"Cluster key {cluster_key} not found in adata.obs")
            raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
        clusters = adata.obs[cluster_key].cat.categories
        if 'spatial' not in adata.obsm:
            logger.error("No spatial coordinates found in AnnData object")
            raise ValueError("No spatial coordinates found in AnnData object")
        coords = adata.obsm['spatial']
        comparison_results = {
            'method': tda_method,
            'clusters': {},
            'pairwise_distances': {}
        }
        for cluster in clusters:
            logger.info(f"Running TDA for cluster {cluster}")
            mask = adata.obs[cluster_key] == cluster
            if np.sum(mask) < 5:
                logger.warning(f"Cluster {cluster} has fewer than 5 spots, skipping")
                continue
            cluster_coords = coords[mask]
            if tda_method == "persistent_homology":
                try:
                    from ripser import ripser
                    
                    result = ripser(cluster_coords, maxdim=kwargs.get('max_dim', 1))
                    comparison_results['clusters'][cluster] = {
                        'diagrams': result['dgms'],
                        'n_spots': int(np.sum(mask))
                    }
                    
                except ImportError:
                    logger.error("ripser package not installed. Please install it with 'pip install ripser'")
                    raise ImportError("ripser required for persistent homology comparison")
                    
            elif tda_method == "graph_features":
                import networkx as nx
                from sklearn.neighbors import kneighbors_graph
                n_neighbors = min(kwargs.get('n_neighbors', 5), np.sum(mask) - 1)
                knn_graph = kneighbors_graph(cluster_coords, n_neighbors=n_neighbors, mode='distance')
                G = nx.from_scipy_sparse_matrix(knn_graph)
                features = {
                    'n_spots': int(np.sum(mask)),
                    'n_edges': G.number_of_edges(),
                    'average_degree': np.mean([d for _, d in G.degree()]),
                    'density': nx.density(G),
                    'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
                    'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
                    'average_clustering': nx.average_clustering(G)
                }
                
                comparison_results['clusters'][cluster] = features
                
            else:
                logger.error(f"Unsupported TDA method: {tda_method}")
                raise ValueError(f"Unsupported TDA method: {tda_method}")
        if tda_method == "persistent_homology":
            from scipy.spatial.distance import directed_hausdorff
            
            for i, cluster1 in enumerate(clusters):
                if cluster1 not in comparison_results['clusters']:
                    continue
                    
                comparison_results['pairwise_distances'][cluster1] = {}
                
                for j, cluster2 in enumerate(clusters):
                    if cluster2 not in comparison_results['clusters']:
                        continue
                    if i == j:
                        comparison_results['pairwise_distances'][cluster1][cluster2] = 0.0
                        continue
                    diagrams1 = comparison_results['clusters'][cluster1]['diagrams']
                    diagrams2 = comparison_results['clusters'][cluster2]['diagrams']
                    distances = {}
                    
                    for dim in range(min(len(diagrams1), len(diagrams2))):
                        dgm1 = diagrams1[dim]
                        dgm2 = diagrams2[dim]
                        if len(dgm1) == 0 or len(dgm2) == 0:
                            distances[f'dim{dim}'] = float('nan')
                            continue
                        hausdorff1, _, _ = directed_hausdorff(dgm1, dgm2)
                        hausdorff2, _, _ = directed_hausdorff(dgm2, dgm1)
                        hausdorff = max(hausdorff1, hausdorff2)
                        
                        distances[f'dim{dim}'] = float(hausdorff)
                    
                    comparison_results['pairwise_distances'][cluster1][cluster2] = distances
        
        elif tda_method == "graph_features":
            for i, cluster1 in enumerate(clusters):
                if cluster1 not in comparison_results['clusters']:
                    continue
                    
                comparison_results['pairwise_distances'][cluster1] = {}
                
                for j, cluster2 in enumerate(clusters):
                    if cluster2 not in comparison_results['clusters']:
                        continue
                    if i == j:
                        comparison_results['pairwise_distances'][cluster1][cluster2] = 0.0
                        continue
                    features1 = comparison_results['clusters'][cluster1]
                    features2 = comparison_results['clusters'][cluster2]
                    numeric_features1 = [features1[k] for k in ['average_degree', 'density', 'average_clustering'] 
                                        if np.isfinite(features1[k])]
                    numeric_features2 = [features2[k] for k in ['average_degree', 'density', 'average_clustering'] 
                                        if np.isfinite(features2[k])]
                    if numeric_features1 and numeric_features2:
                        dist = np.sqrt(np.sum([(a - b)**2 for a, b in zip(numeric_features1, numeric_features2)]))
                    else:
                        dist = float('nan')
                    
                    comparison_results['pairwise_distances'][cluster1][cluster2] = float(dist)
        adata.uns[key_added] = comparison_results
        fig_dir = kwargs.get('fig_dir', None)
        if fig_dir is not None:
            if tda_method == "persistent_homology":
                for dim in range(kwargs.get('max_dim', 1) + 1):
                    dim_key = f'dim{dim}'
                    has_dim = all(cluster1 in comparison_results['pairwise_distances'] and 
                                 all(dim_key in comparison_results['pairwise_distances'][cluster1].get(cluster2, {}) 
                                    for cluster2 in comparison_results['pairwise_distances']) 
                                 for cluster1 in comparison_results['pairwise_distances'])
                    
                    if not has_dim:
                        continue
                    dist_matrix = np.zeros((len(clusters), len(clusters)))
                    
                    for i, cluster1 in enumerate(clusters):
                        if cluster1 not in comparison_results['pairwise_distances']:
                            continue
                            
                        for j, cluster2 in enumerate(clusters):
                            if cluster2 not in comparison_results['pairwise_distances'][cluster1]:
                                continue
                                
                            try:
                                dist_matrix[i, j] = comparison_results['pairwise_distances'][cluster1][cluster2][dim_key]
                            except (KeyError, TypeError):
                                dist_matrix[i, j] = np.nan
                    plt.figure(figsize=(10, 8))
                    plt.imshow(dist_matrix, cmap='viridis')
                    plt.colorbar(label='Distance')
                    plt.xticks(np.arange(len(clusters)), clusters, rotation=90)
                    plt.yticks(np.arange(len(clusters)), clusters)
                    plt.title(f'Topological Distance (Dimension {dim})')
                    plt.tight_layout()
                    fig_path = f"{fig_dir}/topo_distance_dim{dim}.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved topology distance heatmap for dimension {dim} to {fig_path}")
            
            elif tda_method == "graph_features":
                dist_matrix = np.zeros((len(clusters), len(clusters)))
                
                for i, cluster1 in enumerate(clusters):
                    if cluster1 not in comparison_results['pairwise_distances']:
                        continue
                        
                    for j, cluster2 in enumerate(clusters):
                        if cluster2 not in comparison_results['pairwise_distances'][cluster1]:
                            continue
                            
                        dist_matrix[i, j] = comparison_results['pairwise_distances'][cluster1][cluster2]
                plt.figure(figsize=(10, 8))
                plt.imshow(dist_matrix, cmap='viridis')
                plt.colorbar(label='Distance')
                plt.xticks(np.arange(len(clusters)), clusters, rotation=90)
                plt.yticks(np.arange(len(clusters)), clusters)
                plt.title('Graph Feature Distance Between Clusters')
                plt.tight_layout()
                fig_path = f"{fig_dir}/graph_feature_distance.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved graph feature distance heatmap to {fig_path}")
        
        logger.info(f"Topology comparison completed. Results stored in adata.uns['{key_added}']")
        
        return adata