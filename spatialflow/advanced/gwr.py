import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
import warnings

logger = logging.getLogger('spatialflow.advanced.gwr')

def run_gwr_analysis(adata, response_var, explanatory_vars=None, cluster_key=None,
                   adaptive=True, bw_min=None, bw_max=None, key_added=None, **kwargs):
    """
    Run Geographically Weighted Regression (GWR) analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    response_var : str
        Variable to use as response (can be a gene name or obs column)
    explanatory_vars : list, optional
        Variables to use as explanatory variables (genes or obs columns)
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    adaptive : bool, optional
        Whether to use adaptive bandwidth
    bw_min : float, optional
        Minimum bandwidth to test
    bw_max : float, optional
        Maximum bandwidth to test
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters for GWR
    
    Returns
    -------
    adata : AnnData
        The AnnData object with GWR results
    """
    try:
        import mgwr
        from mgwr.gwr import GWR, MGWR
        from mgwr.sel_bw import Sel_BW
    except ImportError:
        logger.error("mgwr package not installed. Please install it with 'pip install mgwr'")
        raise ImportError("mgwr package required for GWR analysis")
    
    logger.info("Running Geographically Weighted Regression analysis")
    if key_added is None:
        key_added = "gwr_results"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    coords = adata.obsm['spatial']
    if response_var in adata.var_names:
        logger.info(f"Using gene {response_var} as response variable")
        
        gene_idx = adata.var_names.get_loc(response_var)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
    
    elif response_var in adata.obs.columns:
        logger.info(f"Using observation variable {response_var} as response")
        y = adata.obs[response_var].values
    
    else:
        logger.error(f"Response variable {response_var} not found in genes or observations")
        raise ValueError(f"Response variable {response_var} not found in genes or observations")
    if explanatory_vars is None:
        if cluster_key is not None and cluster_key in adata.obs.columns:
            logger.info(f"Using cluster membership from {cluster_key} as explanatory variables")
            clusters = pd.get_dummies(adata.obs[cluster_key]).values
            var_names = adata.obs[cluster_key].cat.categories.tolist()
            X = np.column_stack([np.ones(adata.n_obs), clusters])
            var_names = ['intercept'] + [f"cluster_{c}" for c in var_names]
        
        else:
            if 'moranI' in adata.uns:
                logger.info("Using top spatially variable genes as explanatory variables")
                top_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:3].tolist()
                if response_var in top_genes:
                    top_genes.remove(response_var)
                X = np.zeros((adata.n_obs, len(top_genes) + 1))
                X[:, 0] = 1  # Constant term
                
                for i, gene in enumerate(top_genes):
                    gene_idx = adata.var_names.get_loc(gene)
                    if hasattr(adata.X, 'toarray'):
                        X[:, i+1] = adata.X[:, gene_idx].toarray().flatten()
                    else:
                        X[:, i+1] = adata.X[:, gene_idx].copy()
                
                var_names = ['intercept'] + top_genes
            
            else:
                logger.info("Using random genes as explanatory variables")
                all_genes = adata.var_names.tolist()
                if response_var in all_genes:
                    all_genes.remove(response_var)
                np.random.seed(kwargs.get('seed', 42))
                rand_genes = np.random.choice(all_genes, size=2, replace=False)
                X = np.zeros((adata.n_obs, 3))
                X[:, 0] = 1  # Constant term
                
                for i, gene in enumerate(rand_genes):
                    gene_idx = adata.var_names.get_loc(gene)
                    if hasattr(adata.X, 'toarray'):
                        X[:, i+1] = adata.X[:, gene_idx].toarray().flatten()
                    else:
                        X[:, i+1] = adata.X[:, gene_idx].copy()
                
                var_names = ['intercept'] + rand_genes.tolist()
    
    else:
        logger.info(f"Using {len(explanatory_vars)} explanatory variables")
        X = np.ones((adata.n_obs, len(explanatory_vars) + 1))  # Add constant term
        var_names = ['intercept']
        
        for i, var in enumerate(explanatory_vars):
            if var in adata.var_names:
                gene_idx = adata.var_names.get_loc(var)
                if hasattr(adata.X, 'toarray'):
                    X[:, i+1] = adata.X[:, gene_idx].toarray().flatten()
                else:
                    X[:, i+1] = adata.X[:, gene_idx].copy()
                
                var_names.append(var)
            
            elif var in adata.obs.columns:
                try:
                    X[:, i+1] = adata.obs[var].values.astype(float)
                    var_names.append(var)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {var} to numeric, skipping")
            
            else:
                logger.warning(f"Variable {var} not found in genes or observations, using zeros")
                var_names.append(f"unknown_{var}")
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X[:, 1:], y)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X[:, 1:])
    _, s, _ = np.linalg.svd(X_scaled)
    condition_number = max(s) / min(s)
    
    if condition_number > 30:
        logger.warning(f"High condition number ({condition_number:.1f}) indicates possible multicollinearity")
    y_std = (y - np.mean(y)) / np.std(y)
    X_std = np.zeros_like(X)
    X_std[:, 0] = 1  # Keep constant term
    for i in range(1, X.shape[1]):
        X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if bw_min is None or bw_max is None:
            avg_nn_dist = _calculate_avg_nn_distance(coords, 10)
            
            if bw_min is None:
                bw_min = avg_nn_dist
            
            if bw_max is None:
                max_dist = np.max(pdist(coords))
                bw_max = max_dist / 3
        
        logger.info(f"Selecting bandwidth in range [{bw_min:.3f}, {bw_max:.3f}]")
        
        try:
            bw = Sel_BW(coords, y_std, X_std, fixed=not adaptive)
            use_mgwr = kwargs.get('use_mgwr', False)
            if use_mgwr:
                logger.info("Selecting optimal bandwidths for MGWR")
                bws = bw.search(bw_min=bw_min, bw_max=bw_max)
                logger.info(f"Selected bandwidths: {bws}")
            else:
                logger.info("Selecting optimal bandwidth for GWR")
                bw_optimal = bw.search(bw_min=bw_min, bw_max=bw_max)
                logger.info(f"Selected bandwidth: {bw_optimal:.3f}")
            if use_mgwr:
                logger.info("Running MGWR with variable bandwidths")
                gwr_model = MGWR(coords, y_std, X_std, bws)
            else:
                logger.info("Running GWR with fixed bandwidth")
                gwr_model = GWR(coords, y_std, X_std, bw_optimal, fixed=not adaptive)
            gwr_results = gwr_model.fit()
            logger.info(f"Model fit complete. R²: {gwr_results.R2:.4f}, Adj. R²: {gwr_results.adj_R2:.4f}")
            params = gwr_results.params
            t_vals = gwr_results.tvalues
            p_vals = 2 * (1 - abs(gwr_results.tvalues).ppf()) 
            for i, var in enumerate(var_names):
                adata.obs[f'gwr_{var}_coef'] = params[:, i]
                adata.obs[f'gwr_{var}_tval'] = t_vals[:, i]
                adata.obs[f'gwr_{var}_pval'] = p_vals[:, i]
            adata.obs['gwr_localR2'] = gwr_results.localR2
            adata.uns[key_added] = {
                'method': 'mgwr' if use_mgwr else 'gwr',
                'response': response_var,
                'explanatory': var_names[1:],  # Skip intercept
                'adaptive': adaptive,
                'bandwidth': bws if use_mgwr else bw_optimal,
                'condition_number': condition_number,
                'summary': {
                    'aic': float(gwr_results.aic),
                    'aicc': float(gwr_results.aicc),
                    'bic': float(gwr_results.bic),
                    'R2': float(gwr_results.R2),
                    'adj_R2': float(gwr_results.adj_R2),
                    'sigma2': float(gwr_results.sigma2)
                }
            }
            from sklearn.linear_model import LinearRegression
            global_model = LinearRegression().fit(X_std, y_std)
            global_r2 = global_model.score(X_std, y_std)
            adata.uns[key_added]['global_R2'] = float(global_r2)
            adata.uns[key_added]['global_coef'] = global_model.coef_.tolist() if len(global_model.coef_.shape) == 1 else global_model.coef_[:,0].tolist()
            adata.uns[key_added]['global_intercept'] = float(global_model.intercept_)
            logger.info(f"Global model R²: {global_r2:.4f}, GWR R²: {gwr_results.R2:.4f}")
            logger.info(f"Improvement from GWR: {gwr_results.R2 - global_r2:.4f}")
            _compute_coefficient_hotspots(adata, var_names, key_added)
        
        except Exception as e:
            logger.error(f"Error during GWR analysis: {str(e)}")
            raise
    
    return adata

def _calculate_avg_nn_distance(coords, k=10):
    """
    Calculate the average distance to k nearest neighbors
    
    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates array
    k : int, optional
        Number of nearest neighbors
    
    Returns
    -------
    float
        Average distance to k nearest neighbors
    """
    dist_matrix = squareform(pdist(coords))
    sorted_dist = np.sort(dist_matrix, axis=1)
    knn_dist = sorted_dist[:, 1:k+1]
    avg_dist = np.mean(knn_dist)
    
    return avg_dist

def _compute_coefficient_hotspots(adata, var_names, key_added):
    """
    Compute hotspots of GWR coefficients
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    var_names : list
        Names of variables in the GWR model
    key_added : str
        Key for storing results in adata.uns
    
    Returns
    -------
    None
        Modifies adata in place
    """
    logger.info("Computing coefficient hotspots")
    coef_vars = var_names[1:]
    if 'spatial_neighbors' not in adata.uns:
        logger.info("Building spatial neighbors graph")
        from sklearn.neighbors import kneighbors_graph
        coords = adata.obsm['spatial']
        n_neighbors = min(6, adata.n_obs - 1)
        knn_graph = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity')
        adata.uns['spatial_neighbors'] = {}
        adata.obsp['spatial_connectivities'] = knn_graph
    adjacency = adata.obsp['spatial_connectivities']
    for var in coef_vars:
        coef_val = adata.obs[f'gwr_{var}_coef'].values
        local_moran = _compute_local_morans_i(coef_val, adjacency)
        hotspot_type = _classify_local_morans(coef_val, local_moran)
        adata.obs[f'gwr_{var}_localmoran'] = local_moran
        adata.obs[f'gwr_{var}_hotspot'] = pd.Categorical(hotspot_type)
        counts = pd.Series(hotspot_type).value_counts()
        logger.info(f"Coefficient hotspots for {var}:")
        for ht, count in counts.items():
            if ht != 'NS':
                logger.info(f"  {ht}: {count} spots")
    adata.uns[key_added]['hotspots'] = {
        'variables': coef_vars,
        'types': ['HH', 'LL', 'HL', 'LH', 'NS']
    }

def _compute_local_morans_i(values, adjacency):
    """
    Compute local Moran's I for spatial autocorrelation
    
    Parameters
    ----------
    values : numpy.ndarray
        Values to compute autocorrelation for
    adjacency : scipy.sparse.spmatrix
        Spatial weights matrix
    
    Returns
    -------
    numpy.ndarray
        Local Moran's I values
    """
    z = (values - np.mean(values)) / np.std(values)
    if hasattr(adjacency, 'toarray'):
        W = adjacency.toarray()
    else:
        W = adjacency.copy()
    W_row_sum = W.sum(axis=1)
    W_row_sum[W_row_sum == 0] = 1  # Avoid division by zero
    W = W / W_row_sum[:, np.newaxis]
    z_lag = W @ z
    local_moran = z * z_lag
    
    return local_moran

def _classify_local_morans(values, local_moran, p_threshold=0.05):
    """
    Classify local Moran's I into hotspot types
    
    Parameters
    ----------
    values : numpy.ndarray
        Original values
    local_moran : numpy.ndarray
        Local Moran's I values
    p_threshold : float, optional
        P-value threshold for significance
    
    Returns
    -------
    list
        Hotspot classification for each location
    """
    z = (values - np.mean(values)) / np.std(values)
    significant = np.abs(local_moran) > np.std(local_moran) * 1.96  # ~95% confidence
    hotspot_type = ['NS'] * len(values)  # NS = Not Significant
    
    for i in range(len(values)):
        if significant[i]:
            if z[i] > 0 and local_moran[i] > 0:
                hotspot_type[i] = 'HH'  # High-High cluster
            elif z[i] < 0 and local_moran[i] > 0:
                hotspot_type[i] = 'LL'  # Low-Low cluster
            elif z[i] > 0 and local_moran[i] < 0:
                hotspot_type[i] = 'HL'  # High-Low outlier
            elif z[i] < 0 and local_moran[i] < 0:
                hotspot_type[i] = 'LH'  # Low-High outlier
    
    return hotspot_type

def run_mgwr_analysis(adata, response_var, explanatory_vars, key_added=None, **kwargs):
    """
    Run Multiscale Geographically Weighted Regression (MGWR) analysis
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    response_var : str
        Variable to use as response (can be a gene name or obs column)
    explanatory_vars : list
        Variables to use as explanatory variables (genes or obs columns)
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters for MGWR
    
    Returns
    -------
    adata : AnnData
        The AnnData object with MGWR results
    """
    if key_added is None:
        key_added = "mgwr_results"
    kwargs['use_mgwr'] = True
    return run_gwr_analysis(adata, response_var, explanatory_vars, key_added=key_added, **kwargs)

def run_local_bivariate_analysis(adata, var1, var2, method="corr", bandwidth=None, 
                               key_added=None, **kwargs):
    """
    Run local bivariate analysis between two variables
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    var1 : str
        First variable (gene name or obs column)
    var2 : str
        Second variable (gene name or obs column)
    method : str, optional
        Method for bivariate analysis. Options: "corr", "regression", "covariance"
    bandwidth : float, optional
        Bandwidth for local analysis. If None, estimated from data.
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with local bivariate analysis results
    """
    logger.info(f"Running local bivariate analysis between {var1} and {var2}")
    if key_added is None:
        key_added = f"bivar_{method}"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    coords = adata.obsm['spatial']
    if var1 in adata.var_names:
        gene_idx = adata.var_names.get_loc(var1)
        if hasattr(adata.X, 'toarray'):
            x1 = adata.X[:, gene_idx].toarray().flatten()
        else:
            x1 = adata.X[:, gene_idx].copy()
    elif var1 in adata.obs.columns:
        try:
            x1 = adata.obs[var1].values.astype(float)
        except (ValueError, TypeError):
            logger.error(f"Could not convert {var1} to numeric")
            raise ValueError(f"Could not convert {var1} to numeric")
    else:
        logger.error(f"Variable {var1} not found in genes or observations")
        raise ValueError(f"Variable {var1} not found in genes or observations")
    if var2 in adata.var_names:
        gene_idx = adata.var_names.get_loc(var2)
        if hasattr(adata.X, 'toarray'):
            x2 = adata.X[:, gene_idx].toarray().flatten()
        else:
            x2 = adata.X[:, gene_idx].copy()
    elif var2 in adata.obs.columns:
        try:
            x2 = adata.obs[var2].values.astype(float)
        except (ValueError, TypeError):
            logger.error(f"Could not convert {var2} to numeric")
            raise ValueError(f"Could not convert {var2} to numeric")
    else:
        logger.error(f"Variable {var2} not found in genes or observations")
        raise ValueError(f"Variable {var2} not found in genes or observations")
    x1_std = (x1 - np.mean(x1)) / np.std(x1)
    x2_std = (x2 - np.mean(x2)) / np.std(x2)
    if bandwidth is None:
        avg_nn_dist = _calculate_avg_nn_distance(coords, 10)
        bandwidth = avg_nn_dist * 2
        logger.info(f"Using bandwidth: {bandwidth:.3f}")
    sq_dists = squareform(pdist(coords))
    weights = np.exp(-0.5 * (sq_dists / bandwidth)**2)
    if method == "corr":
        local_stat = _calculate_local_correlation(x1_std, x2_std, weights)
        stat_name = "correlation"
    
    elif method == "regression":
        local_stat = _calculate_local_regression(x1_std, x2_std, weights)
        stat_name = "regression_beta"
    
    elif method == "covariance":
        local_stat = _calculate_local_covariance(x1_std, x2_std, weights)
        stat_name = "covariance"
    
    else:
        logger.error(f"Unsupported method: {method}")
        raise ValueError(f"Unsupported method: {method}")
    result_key = f"local_{stat_name}_{var1}_{var2}"
    adata.obs[result_key] = local_stat
    if method == "corr":
        from scipy.stats import pearsonr
        global_stat, global_pval = pearsonr(x1, x2)
    elif method == "regression":
        from scipy.stats import linregress
        slope, _, _, global_pval, _ = linregress(x1, x2)
        global_stat = slope
    elif method == "covariance":
        global_stat = np.cov(x1, x2)[0, 1]
        global_pval = None
    stat_std = np.std(local_stat)
    z_scores = (local_stat - global_stat) / (stat_std + 1e-10)
    adata.obs[f"{result_key}_zscore"] = z_scores
    alpha = kwargs.get('alpha', 0.05)
    z_threshold = abs(np.percentile(z_scores, (alpha/2)*100))
    
    sig_pos = z_scores > z_threshold
    sig_neg = z_scores < -z_threshold
    
    classification = np.array(['NS'] * len(z_scores), dtype=object)
    classification[sig_pos] = 'SigPos'
    classification[sig_neg] = 'SigNeg'
    
    adata.obs[f"{result_key}_sig"] = pd.Categorical(classification)
    adata.uns[key_added] = {
        'method': method,
        'var1': var1,
        'var2': var2,
        'bandwidth': bandwidth,
        'result_key': result_key,
        'global_stat': float(global_stat),
        'global_pval': float(global_pval) if global_pval is not None else None,
        'z_threshold': z_threshold,
        'alpha': alpha,
        'sig_pos_count': int(np.sum(sig_pos)),
        'sig_neg_count': int(np.sum(sig_neg)),
        'ns_count': int(np.sum(classification == 'NS'))
    }
    logger.info(f"Local {stat_name} analysis completed")
    logger.info(f"Global {stat_name}: {global_stat:.3f}")
    logger.info(f"Local {stat_name} range: [{min(local_stat):.3f}, {max(local_stat):.3f}]")
    logger.info(f"Significant positive: {np.sum(sig_pos)} spots")
    logger.info(f"Significant negative: {np.sum(sig_neg)} spots")
    logger.info(f"Not significant: {np.sum(classification == 'NS')} spots")
    
    return adata

def _calculate_local_correlation(x1, x2, weights):
    """
    Calculate locally weighted correlation coefficient
    
    Parameters
    ----------
    x1 : numpy.ndarray
        First variable
    x2 : numpy.ndarray
        Second variable
    weights : numpy.ndarray
        Weight matrix
    
    Returns
    -------
    numpy.ndarray
        Local correlation coefficients
    """
    n = len(x1)
    local_corr = np.zeros(n)
    
    for i in range(n):
        w = weights[i]
        w = w / np.sum(w)
        x1_mean = np.sum(w * x1)
        x2_mean = np.sum(w * x2)
        cov = np.sum(w * (x1 - x1_mean) * (x2 - x2_mean))
        var1 = np.sum(w * (x1 - x1_mean)**2)
        var2 = np.sum(w * (x2 - x2_mean)**2)
        if var1 > 0 and var2 > 0:
            local_corr[i] = cov / np.sqrt(var1 * var2)
        else:
            local_corr[i] = 0
    
    return local_corr

def _calculate_local_regression(x1, x2, weights):
    """
    Calculate locally weighted regression coefficient
    
    Parameters
    ----------
    x1 : numpy.ndarray
        Independent variable
    x2 : numpy.ndarray
        Dependent variable
    weights : numpy.ndarray
        Weight matrix
    
    Returns
    -------
    numpy.ndarray
        Local regression coefficients
    """
    n = len(x1)
    local_beta = np.zeros(n)
    
    for i in range(n):
        w = weights[i]
        w = w + 1e-10
        w = w / np.sum(w)
        X = np.column_stack([np.ones(n), x1])
        XtW = X.T * w
        XtWX = XtW @ X
        XtWy = XtW @ x2
        
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            local_beta[i] = beta[1]  # Slope term
        except np.linalg.LinAlgError:
            local_beta[i] = 0
    
    return local_beta

def _calculate_local_covariance(x1, x2, weights):
    """
    Calculate locally weighted covariance
    
    Parameters
    ----------
    x1 : numpy.ndarray
        First variable
    x2 : numpy.ndarray
        Second variable
    weights : numpy.ndarray
        Weight matrix
    
    Returns
    -------
    numpy.ndarray
        Local covariances
    """
    n = len(x1)
    local_cov = np.zeros(n)
    
    for i in range(n):
        w = weights[i]
        w = w / np.sum(w)
        x1_mean = np.sum(w * x1)
        x2_mean = np.sum(w * x2)
        local_cov[i] = np.sum(w * (x1 - x1_mean) * (x2 - x2_mean))
    
    return local_cov

def run_spatial_regression_grid(adata, response_var, explanatory_vars, bandwidth=None,
                              resolution=50, key_added=None, **kwargs):
    """
    Run a grid of spatially varying regressions for visualization
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    response_var : str
        Variable to use as response (can be a gene name or obs column)
    explanatory_vars : list
        Variables to use as explanatory variables (genes or obs columns)
    bandwidth : float, optional
        Bandwidth for local regression. If None, estimated from data.
    resolution : int, optional
        Resolution of the grid
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with grid regression results
    """
    logger.info(f"Running spatial regression grid with resolution {resolution}")
    if key_added is None:
        key_added = "grid_regression"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    coords = adata.obsm['spatial']
    if response_var in adata.var_names:
        logger.info(f"Using gene {response_var} as response variable")
        
        gene_idx = adata.var_names.get_loc(response_var)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
    
    elif response_var in adata.obs.columns:
        logger.info(f"Using observation variable {response_var} as response")
        y = adata.obs[response_var].values
    
    else:
        logger.error(f"Response variable {response_var} not found in genes or observations")
        raise ValueError(f"Response variable {response_var} not found in genes or observations")
    X = np.ones((adata.n_obs, len(explanatory_vars) + 1))  # Add constant term
    var_names = ['intercept']
    
    for i, var in enumerate(explanatory_vars):
        if var in adata.var_names:
            gene_idx = adata.var_names.get_loc(var)
            if hasattr(adata.X, 'toarray'):
                X[:, i+1] = adata.X[:, gene_idx].toarray().flatten()
            else:
                X[:, i+1] = adata.X[:, gene_idx].copy()
            
            var_names.append(var)
        
        elif var in adata.obs.columns:
            try:
                X[:, i+1] = adata.obs[var].values.astype(float)
                var_names.append(var)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {var} to numeric, using zeros")
                var_names.append(f"unknown_{var}")
        
        else:
            logger.warning(f"Variable {var} not found in genes or observations, using zeros")
            var_names.append(f"unknown_{var}")
    y_std = (y - np.mean(y)) / np.std(y)
    X_std = np.zeros_like(X)
    X_std[:, 0] = 1  # Keep constant term
    for i in range(1, X.shape[1]):
        X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
    if bandwidth is None:
        avg_nn_dist = _calculate_avg_nn_distance(coords, 10)
        bandwidth = avg_nn_dist * 3
        logger.info(f"Using bandwidth: {bandwidth:.3f}")
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([xx.flatten(), yy.flatten()])
    n_params = X_std.shape[1]
    grid_params = np.zeros((len(grid_coords), n_params))
    
    for i, point in enumerate(grid_coords):
        sq_dists = np.sum((coords - point)**2, axis=1)
        weights = np.exp(-0.5 * sq_dists / bandwidth**2)
        weights = weights / np.sum(weights)
        weights = weights + 1e-10
        W = np.diag(weights)
        XtW = X_std.T @ W
        XtWX = XtW @ X_std
        XtWy = XtW @ y_std
        
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            grid_params[i] = beta
        except np.linalg.LinAlgError:
            grid_params[i] = np.nan
    grid_params_shaped = {}
    for i, var in enumerate(var_names):
        grid_params_shaped[var] = grid_params[:, i].reshape(resolution, resolution)
    adata.uns[key_added] = {
        'method': 'grid_regression',
        'response': response_var,
        'explanatory': explanatory_vars,
        'bandwidth': bandwidth,
        'resolution': resolution,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'params': grid_params_shaped
    }
    
    logger.info("Spatial regression grid completed")
    logger.info(f"Grid parameters: {n_params} parameters at {resolution}x{resolution} = {n_params * resolution**2} total values")
    
    return adata
