import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.core.feature_selection')

def run_feature_selection(adata, n_top_genes=2000, flavor='seurat', 
                         min_mean=0.0125, max_mean=3, min_disp=0.5, 
                         span=0.3, use_raw=True, **kwargs):
    """
    Run feature selection to identify highly variable genes
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    n_top_genes : int, optional
        Number of highly variable genes to select
    flavor : str, optional
        Flavor for highly variable gene identification.
        One of 'seurat', 'cell_ranger', 'seurat_v3'
    min_mean : float, optional
        Minimum mean expression to consider
    max_mean : float, optional
        Maximum mean expression to consider
    min_disp : float, optional
        Minimum dispersion to consider
    span : float, optional
        Span parameter for trend fitting
    use_raw : bool, optional
        Whether to use raw data or not
    **kwargs
        Additional arguments passed to sc.pp.highly_variable_genes
        
    Returns
    -------
    adata : AnnData
        AnnData object with updated .var containing highly variable genes
    """
    logger.info(f"Running feature selection to identify {n_top_genes} highly variable genes")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        span=span,
        batch_key=kwargs.get('batch_key', None),
        n_bins=kwargs.get('n_bins', 20),
        subset=kwargs.get('subset', False),
        inplace=True
    )
    n_hvg = adata.var.highly_variable.sum()
    logger.info(f"Identified {n_hvg} highly variable genes")
    
    return adata

def run_spatial_feature_selection(adata, n_top_genes=2000, use_moran=True,
                                spatial_key='spatial_neighbors', **kwargs):
    """
    Run spatially-aware feature selection to identify genes with spatial patterns
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    n_top_genes : int, optional
        Number of top spatially variable genes to select
    use_moran : bool, optional
        Whether to use Moran's I for identifying spatial genes
    spatial_key : str, optional
        Key in adata.uns for the spatial neighbors graph
    **kwargs
        Additional arguments passed to squidpy.gr.spatial_autocorr
        
    Returns
    -------
    adata : AnnData
        AnnData object with updated .var containing spatially variable genes
    """
    import squidpy as sq
    
    logger.info(f"Running spatial feature selection")
    if spatial_key not in adata.uns and 'spatial_connectivities' not in adata.obsp:
        logger.warning("Spatial neighbors graph not found. Running sq.gr.spatial_neighbors first.")
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    if use_moran:
        logger.info("Computing Moran's I spatial autocorrelation")
        if 'highly_variable' in adata.var:
            hvg = adata.var_names[adata.var.highly_variable].tolist()
            genes = hvg[:min(1000, len(hvg))]  # Use at most 1000 genes for computation efficiency
        else:
            if hasattr(adata.X, 'toarray'):
                variances = np.var(adata.X.toarray(), axis=0)
            else:
                variances = np.var(adata.X, axis=0)
            top_var_idx = np.argsort(variances)[-1000:]  # Use top 1000 variable genes
            genes = adata.var_names[top_var_idx].tolist()
        sq.gr.spatial_autocorr(
            adata,
            genes=genes,
            mode='moran',
            n_perms=kwargs.get('n_perms', 100),
            n_jobs=kwargs.get('n_jobs', -1)
        )
        if "moranI" in adata.uns:
            spatial_genes = adata.uns["moranI"].sort_values("I", ascending=False)
            adata.var['spatially_variable'] = False
            if len(spatial_genes) > 0:
                pval_cols = [col for col in spatial_genes.columns if 'p' in col.lower()]
                if pval_cols:
                    pval_col = pval_cols[0]
                    sig_genes = spatial_genes[spatial_genes[pval_col] < 0.05].index
                    adata.var.loc[sig_genes, 'spatially_variable'] = True
                    logger.info(f"Identified {len(sig_genes)} spatially variable genes with p < 0.05")
                top_genes = spatial_genes.index[:n_top_genes]
                adata.var['spatially_variable_rank'] = np.nan
                for i, gene in enumerate(top_genes):
                    if gene in adata.var_names:
                        adata.var.loc[gene, 'spatially_variable_rank'] = i + 1
                
                logger.info(f"Ranked top {len(top_genes)} spatially variable genes by Moran's I")
    
    return adata

def select_genes_by_custom_metric(adata, metric='cv', n_top_genes=2000, key_added=None):
    """
    Select genes based on a custom metric
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    metric : str or callable, optional
        Metric to use for ranking genes.
        If str, one of 'cv' (coefficient of variation), 'var' (variance),
        'mean' (mean expression).
        If callable, should take a gene expression vector and return a score.
    n_top_genes : int, optional
        Number of top genes to select
    key_added : str, optional
        Key to add to adata.var
        
    Returns
    -------
    adata : AnnData
        AnnData object with updated .var containing selected genes
    """
    logger.info(f"Selecting genes by {metric} metric")
    
    if key_added is None:
        key_added = f"selected_by_{metric}" if isinstance(metric, str) else "selected_by_custom"
    if metric == 'cv':
        if hasattr(adata.X, 'toarray'):
            gene_mean = np.mean(adata.X.toarray(), axis=0)
            gene_std = np.std(adata.X.toarray(), axis=0)
        else:
            gene_mean = np.mean(adata.X, axis=0)
            gene_std = np.std(adata.X, axis=0)
        
        gene_cv = np.divide(gene_std, gene_mean, out=np.zeros_like(gene_mean), where=gene_mean!=0)
        scores = gene_cv
        
    elif metric == 'var':
        if hasattr(adata.X, 'toarray'):
            scores = np.var(adata.X.toarray(), axis=0)
        else:
            scores = np.var(adata.X, axis=0)
            
    elif metric == 'mean':
        if hasattr(adata.X, 'toarray'):
            scores = np.mean(adata.X.toarray(), axis=0)
        else:
            scores = np.mean(adata.X, axis=0)
            
    elif callable(metric):
        scores = np.zeros(adata.n_vars)
        for i in range(adata.n_vars):
            if hasattr(adata.X, 'toarray'):
                gene_expr = adata.X[:, i].toarray().flatten()
            else:
                gene_expr = adata.X[:, i]
            scores[i] = metric(gene_expr)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    top_idx = np.argsort(scores)[-n_top_genes:][::-1]  # Reverse to get descending order
    adata.var[f"{key_added}_score"] = scores
    adata.var[key_added] = False
    adata.var.loc[adata.var_names[top_idx], key_added] = True
    
    logger.info(f"Selected {n_top_genes} genes using {metric} metric")
    
    return adata

def run_dimension_reduction(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack', 
                           method='pca', **kwargs):
    """
    Run dimension reduction on the data
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    n_comps : int, optional
        Number of components to compute
    use_highly_variable : bool, optional
        Whether to use highly variable genes only
    svd_solver : str, optional
        SVD solver to use
    method : str, optional
        Dimension reduction method. One of 'pca', 'nmf', 'scvi'
    **kwargs
        Additional arguments passed to the dimension reduction function
        
    Returns
    -------
    adata : AnnData
        AnnData object with computed representations in .obsm
    """
    logger.info(f"Running {method.upper()} dimension reduction with {n_comps} components")
    
    if method.lower() == 'pca':
        sc.pp.pca(
            adata,
            n_comps=n_comps,
            use_highly_variable=use_highly_variable,
            svd_solver=svd_solver,
            **kwargs
        )
        
        if 'X_pca' in adata.obsm:
            logger.info(f"PCA computed successfully. Results stored in adata.obsm['X_pca']")
            if 'pca' not in adata.uns or 'variance_ratio' not in adata.uns['pca']:
                logger.warning("PCA variance ratio not found in adata.uns. Calculating manually.")
                if hasattr(adata.X, 'toarray'):
                    data = adata.X.toarray()
                else:
                    data = adata.X
                
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_comps)
                pca.fit(data)
                
                if 'pca' not in adata.uns:
                    adata.uns['pca'] = {}
                adata.uns['pca']['variance_ratio'] = pca.explained_variance_ratio_
        
    elif method.lower() == 'nmf':
        from sklearn.decomposition import NMF
        
        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X.copy()
        if np.any(data < 0):
            logger.warning("Data contains negative values. Shifting to non-negative range for NMF.")
            data = data - np.min(data) + 1e-6
        if use_highly_variable and 'highly_variable' in adata.var:
            mask = adata.var['highly_variable']
            data_subset = data[:, mask]
        else:
            data_subset = data
        nmf = NMF(n_components=n_comps, random_state=kwargs.get('random_state', 0))
        adata.obsm['X_nmf'] = nmf.fit_transform(data_subset)
        if use_highly_variable:
            hvg_indices = np.where(mask)[0]
            components = np.zeros((adata.n_vars, n_comps))
            components[hvg_indices] = nmf.components_.T
            adata.varm['nmf_components'] = components
        else:
            adata.varm['nmf_components'] = nmf.components_.T
        
        logger.info(f"NMF computed successfully. Results stored in adata.obsm['X_nmf']")
        
    elif method.lower() == 'scvi':
        try:
            import scvi
            logger.info("Using scVI for dimensionality reduction")
            if use_highly_variable and 'highly_variable' in adata.var:
                adata_scvi = adata[:, adata.var.highly_variable].copy()
            else:
                adata_scvi = adata.copy()
            scvi.model.SCVI.setup_anndata(adata_scvi)
            model = scvi.model.SCVI(
                adata_scvi,
                n_latent=n_comps,
                **kwargs
            )
            model.train()
            latent = model.get_latent_representation()
            adata.obsm['X_scvi'] = latent
            
            logger.info(f"scVI computed successfully. Results stored in adata.obsm['X_scvi']")
            
        except ImportError:
            logger.error("scVI not installed. Please install scvi-tools with 'pip install scvi-tools'")
            raise ImportError("scVI required for SCVI dimensionality reduction")
            
    else:
        raise ValueError(f"Unsupported dimension reduction method: {method}")
    
    return adata