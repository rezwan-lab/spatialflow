import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger('spatialflow.advanced.bayesian')

def run_bayesian_analysis(adata, method="gp", genes=None, 
                        seed=42, key_added=None, **kwargs):
    """
    Run Bayesian analysis on spatial data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    method : str, optional
        Bayesian analysis method. Options: "gp", "bayes_smooth", "tradeseq", "hmrf"
    genes : list, optional
        List of genes to analyze. If None, uses highly variable genes.
    seed : int, optional
        Random seed for reproducibility
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters for the specific method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with Bayesian analysis results
    """
    logger.info(f"Running Bayesian analysis using {method} method")
    np.random.seed(seed)
    if key_added is None:
        key_added = f"bayesian_{method}"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    if genes is None:
        if 'highly_variable' in adata.var:
            genes = adata.var_names[adata.var.highly_variable].tolist()[:20]
            logger.info(f"Using {len(genes)} highly variable genes")
        else:
            if hasattr(adata.X, 'toarray'):
                gene_var = np.var(adata.X.toarray(), axis=0)
            else:
                gene_var = np.var(adata.X, axis=0)
            
            high_var_idx = np.argsort(gene_var)[-20:]
            genes = adata.var_names[high_var_idx].tolist()
            logger.info(f"Using {len(genes)} genes with highest variance")
    if method == "gp":
        _run_gp_regression(adata, genes, key_added, **kwargs)
    elif method == "bayes_smooth":
        _run_bayesian_smoothing(adata, genes, key_added, **kwargs)
    elif method == "tradeseq":
        _run_tradeseq(adata, genes, key_added, **kwargs)
    elif method == "hmrf":
        _run_hmrf(adata, genes, key_added, **kwargs)
    else:
        logger.error(f"Unsupported method: {method}")
        raise ValueError(f"Unsupported method: {method}")
    
    return adata

def _run_gp_regression(adata, genes, key_added, n_samples=100, 
                     length_scale=None, **kwargs):
    """
    Run Gaussian Process regression for spatial gene expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    key_added : str
        Key for storing results in adata.uns
    n_samples : int, optional
        Number of posterior samples to draw
    length_scale : float, optional
        Length scale for the GP kernel. If None, estimated from data.
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        import gpflow
        import tensorflow as tf
    except ImportError:
        logger.error("gpflow and tensorflow packages not installed. Please install them with 'pip install gpflow tensorflow'")
        raise ImportError("gpflow and tensorflow packages required for GP regression")
    
    logger.info("Running Gaussian Process regression")
    tf.random.set_seed(kwargs.get('seed', 42))
    coords = adata.obsm['spatial']
    coords_norm = coords.copy()
    for i in range(coords.shape[1]):
        coords_norm[:, i] = (coords[:, i] - coords[:, i].min()) / (coords[:, i].max() - coords[:, i].min())
    grid_size = kwargs.get('grid_size', 50)
    x_range = np.linspace(0, 1, grid_size)
    y_range = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_coords = np.column_stack([xx.flatten(), yy.flatten()])
    gene_results = {}
    for gene in genes:
        logger.info(f"Fitting GP for gene {gene}")
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / (y_std + 1e-10)
        if length_scale is None:
            dist_matrix = squareform(pdist(coords_norm))
            median_dist = np.median(dist_matrix[dist_matrix > 0])
            ls = median_dist
            logger.info(f"Estimated length scale: {ls:.3f}")
        else:
            ls = length_scale
        kernel = gpflow.kernels.SquaredExponential(lengthscales=ls)
        model = gpflow.models.GPR(
            data=(coords_norm, y_norm[:, None]),
            kernel=kernel,
            mean_function=None,
            noise_variance=0.1
        )
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        logger.info(f"Optimized length scale: {model.kernel.lengthscales.numpy():.3f}")
        logger.info(f"Optimized noise variance: {model.likelihood.variance.numpy():.3f}")
        from gpflow.utilities import sample_mvn
        mean, cov = model.predict_f(grid_coords, full_cov=True)
        samples = sample_mvn(mean, cov, n_samples)
        samples = samples * y_std + y_mean
        gp_mean = mean.numpy() * y_std + y_mean
        gp_std = np.sqrt(np.diag(cov.numpy())) * y_std
        gp_mean_grid = gp_mean.reshape(grid_size, grid_size)
        gp_std_grid = gp_std.reshape(grid_size, grid_size)
        gene_results[gene] = {
            'mean': gp_mean_grid,
            'std': gp_std_grid,
            'samples': samples,
            'parameters': {
                'length_scale': model.kernel.lengthscales.numpy(),
                'noise_variance': model.likelihood.variance.numpy(),
                'scale': y_std,
                'offset': y_mean
            }
        }
        from scipy.interpolate import LinearNDInterpolator
        mean_interp = LinearNDInterpolator(
            grid_coords, 
            gp_mean.flatten(),
            fill_value=np.mean(gp_mean)
        )
        std_interp = LinearNDInterpolator(
            grid_coords, 
            gp_std.flatten(),
            fill_value=np.mean(gp_std)
        )
        smoothed_expr = mean_interp(coords_norm)
        smoothed_std = std_interp(coords_norm)
        adata.obs[f"{gene}_gp_mean"] = smoothed_expr
        adata.obs[f"{gene}_gp_std"] = smoothed_std
    adata.uns[key_added] = {
        'method': 'gp',
        'genes': genes,
        'grid_size': grid_size,
        'n_samples': n_samples,
        'results': gene_results,
        'x_range': x_range,
        'y_range': y_range
    }
    
    logger.info("Gaussian Process regression completed")

def _run_bayesian_smoothing(adata, genes, key_added, n_samples=1000, **kwargs):
    """
    Run Bayesian spatial smoothing for gene expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    key_added : str
        Key for storing results in adata.uns
    n_samples : int, optional
        Number of MCMC samples to draw
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        import pymc3 as pm
        import theano.tensor as tt
    except ImportError:
        logger.error("pymc3 package not installed. Please install it with 'pip install pymc3'")
        raise ImportError("pymc3 package required for Bayesian smoothing")
    
    logger.info("Running Bayesian spatial smoothing")
    coords = adata.obsm['spatial']
    coords_norm = coords.copy()
    for i in range(coords.shape[1]):
        coords_norm[:, i] = (coords[:, i] - coords[:, i].min()) / (coords[:, i].max() - coords[:, i].min())
    if 'spatial_neighbors' not in adata.uns:
        logger.info("Building spatial neighbors graph")
        from sklearn.neighbors import kneighbors_graph
        n_neighbors = min(kwargs.get('n_neighbors', 6), adata.n_obs - 1)
        knn_graph = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity')
        adata.uns['spatial_neighbors'] = {}
        adata.obsp['spatial_connectivities'] = knn_graph
    adjacency = adata.obsp['spatial_connectivities']
    gene_results = {}
    for gene in genes:
        logger.info(f"Running Bayesian smoothing for gene {gene}")
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
        with pm.Model() as model:
            alpha = pm.HalfNormal('alpha', sigma=1.0)
            sigma = pm.HalfNormal('sigma', sigma=1.0)
            n_spots = adata.n_obs
            adj_list = [[] for _ in range(n_spots)]
            for i in range(n_spots):
                neighbors = adjacency[i].nonzero()[1]
                adj_list[i] = list(neighbors)
            phi = pm.MvNormal('phi', 
                            mu=0, 
                            cov=tt.eye(n_spots) / alpha,
                            shape=n_spots)
            mu = phi
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(n_samples, tune=500, chains=2, 
                            target_accept=0.9, return_inferencedata=False)
        post_samples = trace['phi']
        post_mean = post_samples.mean(axis=0)
        post_std = post_samples.std(axis=0)
        post_q05 = np.percentile(post_samples, 5, axis=0)
        post_q95 = np.percentile(post_samples, 95, axis=0)
        gene_results[gene] = {
            'mean': post_mean,
            'std': post_std,
            'q05': post_q05,
            'q95': post_q95,
            'parameters': {
                'alpha': trace['alpha'].mean(),
                'sigma': trace['sigma'].mean()
            }
        }
        adata.obs[f"{gene}_bayes_mean"] = post_mean
        adata.obs[f"{gene}_bayes_std"] = post_std
    adata.uns[key_added] = {
        'method': 'bayes_smooth',
        'genes': genes,
        'n_samples': n_samples,
        'results': gene_results
    }
    
    logger.info("Bayesian spatial smoothing completed")

def _run_tradeseq(adata, genes, key_added, **kwargs):
    """
    Run TradeSeq for trajectory-based differential expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    key_added : str
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    logger.info("Running TradeSeq analysis")
    trajectory_key = kwargs.get('trajectory_key', 'spatial_trajectory')
    if trajectory_key not in adata.uns:
        logger.error(f"No trajectory information found with key '{trajectory_key}'")
        raise ValueError(f"No trajectory information found with key '{trajectory_key}'")
    pseudotime_key = f"{trajectory_key}_pseudotime"
    if pseudotime_key not in adata.obs:
        logger.error(f"No pseudotime found with key '{pseudotime_key}'")
        raise ValueError(f"No pseudotime found with key '{pseudotime_key}'")
    pseudotime = adata.obs[pseudotime_key].values
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray()
    else:
        counts = adata.X.copy()
    pseudotime_norm = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        numpy2ri.activate()
        base = importr('base')
        stats = importr('stats')
        
        try:
            tradeseq = importr('tradeSeq')
        except:
            logger.error("TradeSeq R package not installed. Please install it with "
                       "BiocManager::install('tradeSeq') in R")
            raise ImportError("TradeSeq R package required for TradeSeq analysis")
        r_counts = numpy2ri.py2rpy(counts)
        r_pseudotime = numpy2ri.py2rpy(pseudotime_norm)
        logger.info("Setting up tradeSeq model")
        n_knots = kwargs.get('n_knots', 5)
        r_code = f"""
        pseudotime <- c({','.join(map(str, pseudotime_norm))})
        cellWeights <- rep(1, length(pseudotime))
        sce <- fitGAM(counts = counts, pseudotime = pseudotime, cellWeights = cellWeights, nknots = {n_knots})
        associationTest <- associationTest(sce)
        pvals <- associationTest$pvalue
        waldStats <- associationTest$waldStat
        startVsEndTest <- startVsEndTest(sce)
        startVsEnd_pvals <- startVsEndTest$pvalue
        """
        ro.r(r_code)
        pvals = ro.r('pvals')
        waldStats = ro.r('waldStats')
        startVsEnd_pvals = ro.r('startVsEnd_pvals')
        results = pd.DataFrame({
            'gene': adata.var_names,
            'pvalue': pvals,
            'waldStat': waldStats,
            'startVsEnd_pvalue': startVsEnd_pvals
        })
        from statsmodels.stats.multitest import multipletests
        results['padj'] = multipletests(results['pvalue'], method='fdr_bh')[1]
        results['startVsEnd_padj'] = multipletests(results['startVsEnd_pvalue'], method='fdr_bh')[1]
        adata.uns[key_added] = {
            'method': 'tradeseq',
            'genes': genes,
            'results': results,
            'parameters': {
                'n_knots': n_knots,
                'trajectory_key': trajectory_key,
                'pseudotime_key': pseudotime_key
            }
        }
        sig_genes = results[results['padj'] < 0.05]
        logger.info(f"Found {len(sig_genes)} genes with trajectory association (FDR < 0.05)")
        
        sig_sve = results[results['startVsEnd_padj'] < 0.05]
        logger.info(f"Found {len(sig_sve)} genes differentially expressed between trajectory start and end (FDR < 0.05)")
        
    except ImportError:
        logger.error("rpy2 package not installed. Please install it with 'pip install rpy2'")
        raise ImportError("rpy2 package required for TradeSeq analysis")
    
    logger.info("TradeSeq analysis completed")

def _run_hmrf(adata, genes, key_added, n_domains=None, beta=2.0, **kwargs):
    """
    Run Hidden Markov Random Field (HMRF) for spatial domain identification
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    key_added : str
        Key for storing results in adata.uns
    n_domains : int, optional
        Number of domains to identify. If None, determined automatically.
    beta : float, optional
        Spatial regularization parameter
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    logger.info("Running HMRF analysis")
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray()
    else:
        counts = adata.X.copy()
    gene_indices = [adata.var_names.get_loc(gene) for gene in genes]
    counts_subset = counts[:, gene_indices]
    if n_domains is None:
        if 'leiden' in adata.obs:
            n_domains = adata.obs['leiden'].nunique()
            logger.info(f"Using {n_domains} domains based on leiden clustering")
        else:
            n_domains = int(np.sqrt(adata.n_obs / 2))
            logger.info(f"Using {n_domains} domains based on rule of thumb")
    if 'spatial_neighbors' not in adata.uns:
        logger.info("Building spatial neighbors graph")
        coords = adata.obsm['spatial']
        from sklearn.neighbors import kneighbors_graph
        
        n_neighbors = min(kwargs.get('n_neighbors', 6), adata.n_obs - 1)
        knn_graph = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity')
        adata.uns['spatial_neighbors'] = {}
        adata.obsp['spatial_connectivities'] = knn_graph
    adjacency = adata.obsp['spatial_connectivities']
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_domains, random_state=kwargs.get('seed', 42))
    domains = kmeans.fit_predict(counts_subset)
    centers = np.zeros((n_domains, counts_subset.shape[1]))
    for k in range(n_domains):
        mask = domains == k
        centers[k] = counts_subset[mask].mean(axis=0)
    def compute_energy(domains, centers, beta):
        data_term = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            data_term[i] = np.sum((counts_subset[i] - centers[domains[i]])**2)
        spatial_term = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neighbors = adjacency[i].nonzero()[1]
            for j in neighbors:
                if domains[i] != domains[j]:
                    spatial_term[i] += 1
        spatial_term *= beta
        total_energy = data_term + spatial_term
        
        return total_energy, data_term, spatial_term
    n_iter = kwargs.get('n_iter', 10)
    convergence_threshold = kwargs.get('convergence_threshold', 0.01)
    
    energy_history = []
    
    for iteration in range(n_iter):
        logger.info(f"HMRF iteration {iteration+1}/{n_iter}")
        total_energy, data_term, spatial_term = compute_energy(domains, centers, beta)
        energy_history.append(np.sum(total_energy))
        domains_old = domains.copy()
        changes = 0
        
        for i in range(adata.n_obs):
            domain_energies = np.zeros(n_domains)
            
            for k in range(n_domains):
                domains[i] = k
                spot_energy, _, _ = compute_energy(domains, centers, beta)
                domain_energies[k] = spot_energy[i]
            best_domain = np.argmin(domain_energies)
            domains[i] = best_domain
            if domains_old[i] != domains[i]:
                changes += 1
        for k in range(n_domains):
            mask = domains == k
            if np.sum(mask) > 0:
                centers[k] = counts_subset[mask].mean(axis=0)
        if iteration > 0:
            energy_change = (energy_history[-2] - energy_history[-1]) / energy_history[-2]
            logger.info(f"Energy change: {energy_change:.4f}, Changes: {changes}/{adata.n_obs}")
            
            if energy_change < convergence_threshold:
                logger.info(f"Converged after {iteration+1} iterations")
                break
    adata.obs[f"{key_added}_domains"] = pd.Categorical(domains.astype(str))
    adata.uns[key_added] = {
        'method': 'hmrf',
        'genes': genes,
        'n_domains': n_domains,
        'beta': beta,
        'energy_history': energy_history,
        'centers': centers,
        'parameters': {
            'n_iter': n_iter,
            'convergence_threshold': convergence_threshold
        }
    }
    
    logger.info("HMRF analysis completed")

def run_spatial_uncertainty_quantification(adata, genes=None, method="bootstrap",
                                         n_samples=100, key_added=None, **kwargs):
    """
    Quantify uncertainty in spatial patterns of gene expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list, optional
        List of genes to analyze. If None, uses highly variable genes.
    method : str, optional
        Method for uncertainty quantification. Options: "bootstrap", "bayesian"
    n_samples : int, optional
        Number of bootstrap or posterior samples
    key_added : str, optional
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    adata : AnnData
        The AnnData object with uncertainty quantification results
    """
    logger.info(f"Running spatial uncertainty quantification using {method} method")
    if key_added is None:
        key_added = f"uncertainty_{method}"
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    if genes is None:
        if 'highly_variable' in adata.var:
            genes = adata.var_names[adata.var.highly_variable].tolist()[:10]
            logger.info(f"Using {len(genes)} highly variable genes")
        else:
            if hasattr(adata.X, 'toarray'):
                gene_var = np.var(adata.X.toarray(), axis=0)
            else:
                gene_var = np.var(adata.X, axis=0)
            
            high_var_idx = np.argsort(gene_var)[-10:]
            genes = adata.var_names[high_var_idx].tolist()
            logger.info(f"Using {len(genes)} genes with highest variance")
    if method == "bootstrap":
        _run_bootstrap_uq(adata, genes, n_samples, key_added, **kwargs)
    elif method == "bayesian":
        _run_bayesian_uq(adata, genes, n_samples, key_added, **kwargs)
    else:
        logger.error(f"Unsupported method: {method}")
        raise ValueError(f"Unsupported method: {method}")
    
    return adata

def _run_bootstrap_uq(adata, genes, n_samples, key_added, **kwargs):
    """
    Perform bootstrap-based uncertainty quantification
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    n_samples : int
        Number of bootstrap samples
    key_added : str
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    logger.info(f"Running bootstrap uncertainty quantification for {len(genes)} genes")
    coords = adata.obsm['spatial']
    coords_norm = coords.copy()
    for i in range(coords.shape[1]):
        coords_norm[:, i] = (coords[:, i] - coords[:, i].min()) / (coords[:, i].max() - coords[:, i].min())
    grid_size = kwargs.get('grid_size', 30)
    x_range = np.linspace(0, 1, grid_size)
    y_range = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_coords = np.column_stack([xx.flatten(), yy.flatten()])
    gene_results = {}
    for gene in genes:
        logger.info(f"Bootstrap analysis for gene {gene}")
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
        bootstrap_means = np.zeros((grid_size * grid_size, n_samples))
        for i in range(n_samples):
            bootstrap_idx = np.random.choice(adata.n_obs, size=adata.n_obs, replace=True)
            coords_bootstrap = coords_norm[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            smoothed_values = _fit_smooth_function(coords_bootstrap, y_bootstrap, grid_coords, **kwargs)
            bootstrap_means[:, i] = smoothed_values
        bs_mean = np.mean(bootstrap_means, axis=1)
        bs_std = np.std(bootstrap_means, axis=1)
        bs_q05 = np.percentile(bootstrap_means, 5, axis=1)
        bs_q95 = np.percentile(bootstrap_means, 95, axis=1)
        bs_mean_grid = bs_mean.reshape(grid_size, grid_size)
        bs_std_grid = bs_std.reshape(grid_size, grid_size)
        bs_q05_grid = bs_q05.reshape(grid_size, grid_size)
        bs_q95_grid = bs_q95.reshape(grid_size, grid_size)
        gene_results[gene] = {
            'mean': bs_mean_grid,
            'std': bs_std_grid,
            'q05': bs_q05_grid,
            'q95': bs_q95_grid
        }
        bs_cv = bs_std / (bs_mean + 1e-10)
        bs_cv_grid = bs_cv.reshape(grid_size, grid_size)
        gene_results[gene]['cv'] = bs_cv_grid
        from scipy.interpolate import LinearNDInterpolator
        mean_interp = LinearNDInterpolator(
            grid_coords, 
            bs_mean,
            fill_value=np.mean(bs_mean)
        )
        
        std_interp = LinearNDInterpolator(
            grid_coords, 
            bs_std,
            fill_value=np.mean(bs_std)
        )
        
        cv_interp = LinearNDInterpolator(
            grid_coords, 
            bs_cv,
            fill_value=np.mean(bs_cv)
        )
        smoothed_mean = mean_interp(coords_norm)
        smoothed_std = std_interp(coords_norm)
        smoothed_cv = cv_interp(coords_norm)
        adata.obs[f"{gene}_bs_mean"] = smoothed_mean
        adata.obs[f"{gene}_bs_std"] = smoothed_std
        adata.obs[f"{gene}_bs_cv"] = smoothed_cv
    adata.uns[key_added] = {
        'method': 'bootstrap',
        'genes': genes,
        'n_samples': n_samples,
        'grid_size': grid_size,
        'results': gene_results,
        'x_range': x_range,
        'y_range': y_range
    }
    
    logger.info("Bootstrap uncertainty quantification completed")

def _fit_smooth_function(coords, values, grid_coords, **kwargs):
    """
    Fit a smoothing function to data and predict on a grid
    
    Parameters
    ----------
    coords : numpy.ndarray
        Data coordinates
    values : numpy.ndarray
        Data values
    grid_coords : numpy.ndarray
        Grid coordinates for prediction
    **kwargs
        Additional parameters
    
    Returns
    -------
    numpy.ndarray
        Predicted values on grid
    """
    smooth_method = kwargs.get('smooth_method', 'kernel')
    
    if smooth_method == 'kernel':
        from sklearn.neighbors import KernelDensity
        bandwidth = kwargs.get('bandwidth', 0.1)
        X = np.column_stack([coords, values[:, np.newaxis]])
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(X)
        grid_values = np.zeros(grid_coords.shape[0])
        
        for i, point in enumerate(grid_coords):
            X_grid = np.column_stack([np.tile(point, (100, 1)), np.linspace(min(values), max(values), 100)[:, np.newaxis]])
            log_dens = kde.score_samples(X_grid)
            grid_values[i] = X_grid[np.argmax(log_dens), 2]
        
        return grid_values
    
    elif smooth_method == 'rbf':
        from scipy.interpolate import Rbf
        rbf = Rbf(coords[:, 0], coords[:, 1], values, function='thin_plate')
        grid_values = rbf(grid_coords[:, 0], grid_coords[:, 1])
        
        return grid_values
    
    elif smooth_method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        def distance_weights(coords, point):
            dist = np.sqrt(np.sum((coords - point)**2, axis=1))
            weights = np.exp(-dist / 0.1)
            weights /= np.sum(weights)
            return weights
        grid_values = np.zeros(grid_coords.shape[0])
        
        for i, point in enumerate(grid_coords):
            weights = distance_weights(coords, point)
            if np.sum(weights > 0) >= 5:  # Need at least 5 points with non-zero weight
                grid_values[i] = np.average(values, weights=weights)
            else:
                grid_values[i] = np.mean(values)
        
        return grid_values
    
    else:
        logger.error(f"Unsupported smoothing method: {smooth_method}")
        raise ValueError(f"Unsupported smoothing method: {smooth_method}")

def _run_bayesian_uq(adata, genes, n_samples, key_added, **kwargs):
    """
    Perform Bayesian uncertainty quantification
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list
        List of genes to analyze
    n_samples : int
        Number of posterior samples
    key_added : str
        Key for storing results in adata.uns
    **kwargs
        Additional parameters
    
    Returns
    -------
    None
        Modifies adata in place
    """
    try:
        import pymc3 as pm
        import theano.tensor as tt
    except ImportError:
        logger.error("pymc3 package not installed. Please install it with 'pip install pymc3'")
        raise ImportError("pymc3 package required for Bayesian uncertainty quantification")
    
    logger.info(f"Running Bayesian uncertainty quantification for {len(genes)} genes")
    coords = adata.obsm['spatial']
    coords_norm = coords.copy()
    for i in range(coords.shape[1]):
        coords_norm[:, i] = (coords[:, i] - coords[:, i].min()) / (coords[:, i].max() - coords[:, i].min())
    grid_size = kwargs.get('grid_size', 30)
    x_range = np.linspace(0, 1, grid_size)
    y_range = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_coords = np.column_stack([xx.flatten(), yy.flatten()])
    gene_results = {}
    for gene in genes:
        logger.info(f"Bayesian analysis for gene {gene}")
        gene_idx = adata.var_names.get_loc(gene)
        if hasattr(adata.X, 'toarray'):
            y = adata.X[:, gene_idx].toarray().flatten()
        else:
            y = adata.X[:, gene_idx].copy()
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / (y_std + 1e-10)
        with pm.Model() as model:
            ls = pm.Gamma('ls', alpha=2, beta=1)
            eta = pm.HalfCauchy('eta', beta=5)
            sigma = pm.HalfCauchy('sigma', beta=5)
            def rbf_cov(x1, x2, ls, eta):
                """RBF covariance function"""
                d = tt.sqrt(tt.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2))
                return eta * tt.exp(-d**2 / (2 * ls**2))
            mu = pm.gp.mean.Zero()
            cov_func = pm.gp.cov.Covariance(rbf_cov, input_dim=2)
            gp = pm.gp.Marginal(mean_func=mu, cov_func=cov_func)
            gp.marginal_likelihood('y', X=coords_norm, y=y_norm, noise=sigma)
            if kwargs.get('sample_prior', False):
                prior_pred = gp.conditional('prior_pred', grid_coords)
            post_pred = gp.conditional('post_pred', grid_coords)
            trace = pm.sample(n_samples, tune=500, chains=2, target_accept=0.9,
                            return_inferencedata=False)
            ppc = pm.sample_posterior_predictive(trace, var_names=['post_pred'], samples=100)
        post_samples = ppc['post_pred']
        post_samples = post_samples * y_std + y_mean
        post_mean = post_samples.mean(axis=0)
        post_std = post_samples.std(axis=0)
        post_q05 = np.percentile(post_samples, 5, axis=0)
        post_q95 = np.percentile(post_samples, 95, axis=0)
        post_mean_grid = post_mean.reshape(grid_size, grid_size)
        post_std_grid = post_std.reshape(grid_size, grid_size)
        post_q05_grid = post_q05.reshape(grid_size, grid_size)
        post_q95_grid = post_q95.reshape(grid_size, grid_size)
        post_cv = post_std / (post_mean + 1e-10)
        post_cv_grid = post_cv.reshape(grid_size, grid_size)
        gene_results[gene] = {
            'mean': post_mean_grid,
            'std': post_std_grid,
            'q05': post_q05_grid,
            'q95': post_q95_grid,
            'cv': post_cv_grid,
            'parameters': {
                'lengthscale': float(np.mean(trace['ls'])),
                'amplitude': float(np.mean(trace['eta'])),
                'noise': float(np.mean(trace['sigma']))
            }
        }
        from scipy.interpolate import LinearNDInterpolator
        mean_interp = LinearNDInterpolator(
            grid_coords, 
            post_mean,
            fill_value=np.mean(post_mean)
        )
        
        std_interp = LinearNDInterpolator(
            grid_coords, 
            post_std,
            fill_value=np.mean(post_std)
        )
        
        cv_interp = LinearNDInterpolator(
            grid_coords, 
            post_cv,
            fill_value=np.mean(post_cv)
        )
        smoothed_mean = mean_interp(coords_norm)
        smoothed_std = std_interp(coords_norm)
        smoothed_cv = cv_interp(coords_norm)
        adata.obs[f"{gene}_bayes_mean"] = smoothed_mean
        adata.obs[f"{gene}_bayes_std"] = smoothed_std
        adata.obs[f"{gene}_bayes_cv"] = smoothed_cv
    adata.uns[key_added] = {
        'method': 'bayesian',
        'genes': genes,
        'n_samples': n_samples,
        'grid_size': grid_size,
        'results': gene_results,
        'x_range': x_range,
        'y_range': y_range
    }
    
    logger.info("Bayesian uncertainty quantification completed")
