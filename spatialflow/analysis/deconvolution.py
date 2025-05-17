import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import NMF
from anndata import AnnData

logger = logging.getLogger('spatialflow.analysis.deconvolution')

def run_deconvolution(adata, method="simple_nmf", ref_dataset=None, marker_genes=None, 
                     n_cell_types=None, random_state=42, **kwargs):
    """
    Run cell type deconvolution on spatial data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    method : str, optional
        Deconvolution method. Options: "simple_nmf", "manual_markers", "reference_based"
    ref_dataset : AnnData, optional
        Reference single-cell dataset with cell type annotations (required for reference_based method)
    marker_genes : dict, optional
        Dictionary mapping cell types to marker genes (required for manual_markers method)
    n_cell_types : int, optional
        Number of cell types to deconvolve (for simple_nmf method)
    random_state : int, optional
        Random state for reproducibility
    **kwargs
        Additional parameters for the specific deconvolution method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with deconvolution results
    """
    logger.info(f"Running cell type deconvolution using {method} method")
    
    if method == "simple_nmf":
        return _run_simple_nmf_deconvolution(adata, n_cell_types, random_state, **kwargs)
    
    elif method == "manual_markers":
        if marker_genes is None:
            marker_genes = _get_default_marker_genes()
            logger.info("Using default marker genes for deconvolution")
        
        return _run_manual_marker_deconvolution(adata, marker_genes, **kwargs)
    
    elif method == "reference_based":
        if ref_dataset is None:
            logger.error("Reference dataset is required for reference-based deconvolution")
            raise ValueError("Reference dataset is required for reference-based deconvolution")
        
        return _run_reference_based_deconvolution(adata, ref_dataset, **kwargs)
    
    else:
        logger.error(f"Unsupported deconvolution method: {method}")
        raise ValueError(f"Unsupported deconvolution method: {method}")

def _run_simple_nmf_deconvolution(adata, n_cell_types=None, random_state=42, 
                                 min_expression=1e-5, max_iter=200, key_added='deconv_nmf'):
    """
    Run simple NMF-based deconvolution
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    n_cell_types : int, optional
        Number of cell types to deconvolve
    random_state : int, optional
        Random state for reproducibility
    min_expression : float, optional
        Minimum expression value
    max_iter : int, optional
        Maximum number of iterations
    key_added : str, optional
        Key to add to adata.obsm for deconvolution results
    
    Returns
    -------
    adata : AnnData
        The AnnData object with deconvolution results
    """
    logger.info("Running simple NMF-based deconvolution")
    adata_copy = adata.copy()
    if n_cell_types is None:
        if 'leiden' in adata.obs:
            n_cell_types = adata.obs['leiden'].nunique()
            logger.info(f"Using {n_cell_types} cell types based on leiden clusters")
        else:
            n_cell_types = 5
            logger.info(f"Using default of {n_cell_types} cell types")
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()
    X = np.maximum(X, min_expression)
    logger.info(f"Running NMF with {n_cell_types} components")
    model = NMF(n_components=n_cell_types, random_state=random_state, max_iter=max_iter)
    W = model.fit_transform(X)  # Spot x CellType
    H = model.components_  # CellType x Gene
    W_norm = W / W.sum(axis=1, keepdims=True)
    column_names = [f'celltype_{i+1}' for i in range(n_cell_types)]
    adata.obsm[key_added] = pd.DataFrame(W_norm, index=adata.obs.index, columns=column_names)
    for i, col in enumerate(column_names):
        adata.obs[f'prop_{col}'] = W_norm[:, i]
    adata.varm[f'{key_added}_loadings'] = H.T
    top_markers = {}
    for i in range(n_cell_types):
        cell_type = column_names[i]
        gene_loadings = H[i]
        top_genes_idx = np.argsort(gene_loadings)[-20:]
        top_genes = adata.var_names[top_genes_idx].tolist()
        top_markers[cell_type] = top_genes
    adata.uns[f'{key_added}_markers'] = top_markers
    logger.info("Top markers for each deconvolved cell type:")
    for cell_type, markers in top_markers.items():
        logger.info(f"{cell_type}: {', '.join(markers[:5])}...")
    
    return adata

def _run_manual_marker_deconvolution(adata, marker_genes, min_expression=1e-5, 
                                   key_added='deconv_markers'):
    """
    Run deconvolution based on pre-defined marker genes
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    marker_genes : dict
        Dictionary mapping cell types to marker genes
    min_expression : float, optional
        Minimum expression value
    key_added : str, optional
        Key to add to adata.obsm for deconvolution results
    
    Returns
    -------
    adata : AnnData
        The AnnData object with deconvolution results
    """
    logger.info("Running marker-based deconvolution")
    filtered_markers = {}
    for cell_type, genes in marker_genes.items():
        valid_genes = [gene for gene in genes if gene in adata.var_names]
        if len(valid_genes) > 0:
            filtered_markers[cell_type] = valid_genes
    if not filtered_markers:
        logger.error("No valid marker genes found in the dataset")
        raise ValueError("No valid marker genes found in the dataset")
    for cell_type, genes in filtered_markers.items():
        logger.info(f"{cell_type}: {len(genes)} marker genes available")
    scores = {}
    for cell_type, genes in filtered_markers.items():
        gene_indices = [adata.var_names.get_loc(gene) for gene in genes]
        if hasattr(adata.X, 'toarray'):
            expr = adata.X[:, gene_indices].toarray()
        else:
            expr = adata.X[:, gene_indices].copy()
        expr = np.maximum(expr, min_expression)
        score = expr.mean(axis=1)
        scores[cell_type] = score
    score_df = pd.DataFrame(scores, index=adata.obs.index)
    props = score_df.div(score_df.sum(axis=1), axis=0)
    adata.obsm[key_added] = props
    for cell_type in props.columns:
        adata.obs[f'prop_{cell_type}'] = props[cell_type]
    adata.uns[f'{key_added}_markers'] = filtered_markers
    
    return adata

def _run_reference_based_deconvolution(adata, ref_dataset, cell_type_key='cell_type', 
                                     min_expression=1e-5, key_added='deconv_reference'):
    """
    Run reference-based deconvolution using a reference single-cell dataset
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object for spatial data
    ref_dataset : AnnData
        Reference single-cell dataset with cell type annotations
    cell_type_key : str, optional
        Key in ref_dataset.obs for cell type annotations
    min_expression : float, optional
        Minimum expression value
    key_added : str, optional
        Key to add to adata.obsm for deconvolution results
    
    Returns
    -------
    adata : AnnData
        The AnnData object with deconvolution results
    """
    logger.info("Running reference-based deconvolution")
    if cell_type_key not in ref_dataset.obs:
        logger.error(f"Cell type key {cell_type_key} not found in reference dataset")
        raise ValueError(f"Cell type key {cell_type_key} not found in reference dataset")
    cell_types = ref_dataset.obs[cell_type_key].cat.categories.tolist()
    logger.info(f"Reference dataset has {len(cell_types)} cell types")
    common_genes = list(set(adata.var_names) & set(ref_dataset.var_names))
    logger.info(f"Found {len(common_genes)} common genes between spatial and reference dataset")
    
    if len(common_genes) < 100:
        logger.warning("Very few common genes found. Deconvolution may be inaccurate.")
    spatial_data = adata[:, common_genes].copy()
    ref_data = ref_dataset[:, common_genes].copy()
    cell_type_profiles = {}
    
    for cell_type in cell_types:
        cells = ref_data[ref_data.obs[cell_type_key] == cell_type]
        if hasattr(cells.X, 'toarray'):
            avg_expr = cells.X.toarray().mean(axis=0)
        else:
            avg_expr = cells.X.mean(axis=0)
        avg_expr = np.maximum(avg_expr, min_expression)
        cell_type_profiles[cell_type] = avg_expr
    profile_matrix = np.vstack([cell_type_profiles[cell_type] for cell_type in cell_types])
    if hasattr(spatial_data.X, 'toarray'):
        spatial_expr = spatial_data.X.toarray()
    else:
        spatial_expr = spatial_data.X.copy()
    spatial_expr = np.maximum(spatial_expr, min_expression)
    from scipy.optimize import nnls
    
    proportions = np.zeros((spatial_data.n_obs, len(cell_types)))
    
    for i in range(spatial_data.n_obs):
        spot_expr = spatial_expr[i]
        props, _ = nnls(profile_matrix.T, spot_expr)
        if props.sum() > 0:
            props = props / props.sum()
        
        proportions[i] = props
    prop_df = pd.DataFrame(proportions, index=adata.obs.index, columns=cell_types)
    adata.obsm[key_added] = prop_df
    for cell_type in prop_df.columns:
        adata.obs[f'prop_{cell_type}'] = prop_df[cell_type]
    adata.varm[f'{key_added}_profiles'] = profile_matrix.T
    adata.obs[f'{key_added}_dominant'] = prop_df.idxmax(axis=1)
    
    return adata

def _get_default_marker_genes():
    """
    Get default marker genes for common cell types
    
    Returns
    -------
    dict
        Dictionary mapping cell types to marker genes
    """
    return {
        'B_cells': ['MS4A1', 'CD19', 'CD79A', 'CD79B', 'BANK1', 'CD22', 'FCER2', 'CR2'],
        'T_cells': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B', 'CCR7', 'IL7R'],
        'Plasma_cells': ['MZB1', 'JCHAIN', 'IGHA1', 'IGHA2', 'IGHG1', 'IGHG2', 'IGHM', 'SDC1'],
        'Macrophages': ['CD68', 'CD163', 'CD14', 'MARCO', 'SIGLEC1', 'APOE', 'C1QA', 'C1QB', 'C1QC'],
        'Dendritic_cells': ['FCER1A', 'CD1C', 'CD1A', 'CLEC10A', 'CLEC9A', 'XCR1', 'IDO1', 'CCL22'],
        'Endothelial_cells': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'ENG', 'EMCN', 'KDR', 'PLVAP'],
        'Fibroblasts': ['DCN', 'COL1A1', 'COL3A1', 'COL6A1', 'LUM', 'PDGFRA', 'FAP', 'THY1']
    }

def compute_cell_type_enrichment(adata, cluster_key, deconv_key='deconv_nmf', 
                               cell_type_key=None, use_obs=True):
    """
    Compute cell type enrichment for clusters
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    cluster_key : str
        Key in adata.obs for cluster assignments
    deconv_key : str, optional
        Key in adata.obsm for deconvolution results
    cell_type_key : str, optional
        Prefix for cell type proportions in adata.obs if use_obs is True
    use_obs : bool, optional
        Whether to use columns in adata.obs or data in adata.obsm
    
    Returns
    -------
    adata : AnnData
        The AnnData object with enrichment results
    """
    logger.info(f"Computing cell type enrichment for clusters in {cluster_key}")
    if use_obs:
        if cell_type_key is None:
            cell_type_key = 'prop_'
        cell_type_cols = [col for col in adata.obs.columns if col.startswith(cell_type_key)]
        
        if not cell_type_cols:
            logger.error(f"No columns with prefix {cell_type_key} found in adata.obs")
            raise ValueError(f"No columns with prefix {cell_type_key} found in adata.obs")
        cell_types = [col[len(cell_type_key):] for col in cell_type_cols]
        proportions = adata.obs[cell_type_cols].values
        
    else:
        if deconv_key not in adata.obsm:
            logger.error(f"Deconvolution key {deconv_key} not found in adata.obsm")
            raise ValueError(f"Deconvolution key {deconv_key} not found in adata.obsm")
        
        proportions = adata.obsm[deconv_key].values
        cell_types = adata.obsm[deconv_key].columns.tolist()
    if cluster_key not in adata.obs:
        logger.error(f"Cluster key {cluster_key} not found in adata.obs")
        raise ValueError(f"Cluster key {cluster_key} not found in adata.obs")
    clusters = adata.obs[cluster_key].values
    unique_clusters = adata.obs[cluster_key].cat.categories.tolist()
    enrichment = np.zeros((len(unique_clusters), len(cell_types)))
    p_values = np.zeros((len(unique_clusters), len(cell_types)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = clusters == cluster
        cluster_props = proportions[cluster_mask]
        other_props = proportions[~cluster_mask]
        for j, cell_type in enumerate(cell_types):
            cluster_mean = cluster_props[:, j].mean()
            other_mean = other_props[:, j].mean()
            if other_mean > 0:
                enrichment[i, j] = cluster_mean / other_mean
            else:
                enrichment[i, j] = np.inf if cluster_mean > 0 else 1.0
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(cluster_props[:, j], other_props[:, j], equal_var=False)
            p_values[i, j] = p_val
    enrichment_df = pd.DataFrame(enrichment, index=unique_clusters, columns=cell_types)
    p_values_df = pd.DataFrame(p_values, index=unique_clusters, columns=cell_types)
    adata.uns[f'{cluster_key}_cell_type_enrichment'] = {
        'enrichment': enrichment_df,
        'p_values': p_values_df
    }
    logger.info("Top enriched cell types for each cluster:")
    for i, cluster in enumerate(unique_clusters):
        top_idx = np.argsort(enrichment[i])[::-1]
        top_cell_types = [cell_types[j] for j in top_idx[:3]]
        top_enrichment = [enrichment[i, j] for j in top_idx[:3]]
        logger.info(f"{cluster}: " + ", ".join([f"{ct} ({e:.2f}x)" for ct, e in zip(top_cell_types, top_enrichment)]))
    
    return adata
