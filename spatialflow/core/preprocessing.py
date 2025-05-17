import scanpy as sc
import numpy as np
import logging
from anndata import AnnData

logger = logging.getLogger('spatialflow.core.preprocessing')

def run_preprocessing(adata, min_genes=200, min_cells=3, target_sum=1e4, 
                     log_transform=True, normalize=True, **kwargs):
    """
    Run standard preprocessing steps on an AnnData object
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess
    min_genes : int
        Minimum number of genes expressed for a spot to be kept
    min_cells : int
        Minimum number of spots a gene must be expressed in to be kept
    target_sum : float
        Target sum for normalization
    log_transform : bool
        Whether to log-transform the data
    normalize : bool
        Whether to normalize the data
    **kwargs
        Additional parameters
        
    Returns
    -------
    adata : AnnData
        The preprocessed AnnData object
    """
    logger.info("Starting preprocessing")
    adata = adata.copy()
    logger.info("Calculating QC metrics")
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    logger.info(f"Filtering spots with less than {min_genes} genes")
    original_n_obs = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    logger.info(f"Removed {original_n_obs - adata.n_obs} spots with less than {min_genes} genes")
    
    logger.info(f"Filtering genes expressed in less than {min_cells} spots")
    original_n_vars = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"Removed {original_n_vars - adata.n_vars} genes expressed in less than {min_cells} spots")
    if normalize:
        logger.info(f"Normalizing total counts per spot to {target_sum}")
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log_transform:
        logger.info("Log-transforming the data")
        sc.pp.log1p(adata)
    
    logger.info(f"Preprocessing complete. Final shape: {adata.shape}")
    
    return adata

def filter_spots_by_image(adata, library_id=None, tissue_mask=None):
    """
    Filter spots based on image information
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to filter
    library_id : str, optional
        Library ID for the image data
    tissue_mask : numpy.ndarray, optional
        Binary mask indicating tissue areas
        
    Returns
    -------
    adata : AnnData
        The filtered AnnData object
    """
    if 'spatial' not in adata.uns:
        logger.warning("No spatial image information found in AnnData object. Skipping image-based filtering.")
        return adata
    if library_id is None:
        library_ids = list(adata.uns['spatial'].keys())
        if len(library_ids) == 0:
            logger.warning("No library IDs found in spatial image information")
            return adata
        library_id = library_ids[0]
        logger.info(f"Using library ID: {library_id}")
    if library_id not in adata.uns['spatial']:
        logger.warning(f"Library ID {library_id} not found in spatial image information")
        return adata
    if tissue_mask is None:
        if 'images' in adata.uns['spatial'][library_id]:
            if 'tissue_hires' in adata.uns['spatial'][library_id]['images']:
                logger.info("Using existing tissue_hires image as mask")
                tissue_mask = adata.uns['spatial'][library_id]['images']['tissue_hires']
    if tissue_mask is not None:
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found in AnnData object")
            return adata
        
        coords = adata.obsm['spatial']
        scalefactors = adata.uns['spatial'][library_id]['scalefactors']
        if 'tissue_hires_scalef' in scalefactors:
            scale = scalefactors['tissue_hires_scalef']
            pixel_coords = coords * scale
            pixel_coords = pixel_coords.astype(int)
            in_tissue = np.zeros(adata.n_obs, dtype=bool)
            
            for i, (x, y) in enumerate(pixel_coords):
                if 0 <= x < tissue_mask.shape[1] and 0 <= y < tissue_mask.shape[0]:
                    in_tissue[i] = tissue_mask[y, x] > 0
            logger.info(f"Filtering spots based on tissue mask. Keeping {np.sum(in_tissue)} out of {adata.n_obs} spots.")
            adata = adata[in_tissue].copy()
    
    return adata

def regress_out_covariates(adata, covariates, n_jobs=None):
    """
    Regress out covariates from the data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    covariates : list or str
        Names of categorical or continuous covariates to regress out
    n_jobs : int, optional
        Number of jobs for parallel processing
        
    Returns
    -------
    adata : AnnData
        The AnnData object with covariates regressed out
    """
    if isinstance(covariates, str):
        covariates = [covariates]
    missing_covs = [cov for cov in covariates if cov not in adata.obs]
    if missing_covs:
        logger.warning(f"Covariates not found in AnnData object: {missing_covs}")
        covariates = [cov for cov in covariates if cov in adata.obs]
    
    if len(covariates) == 0:
        logger.warning("No valid covariates to regress out")
        return adata
    
    logger.info(f"Regressing out covariates: {covariates}")
    sc.pp.regress_out(adata, covariates, n_jobs=n_jobs)
    
    return adata

def batch_correction(adata, batch_key, n_pcs=50):
    """
    Perform batch correction
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    batch_key : str
        Name of the batch annotation in adata.obs
    n_pcs : int
        Number of principal components to use
        
    Returns
    -------
    adata_corrected : AnnData
        The batch-corrected AnnData object
    """
    try:
        import harmonypy
    except ImportError:
        logger.error("harmonypy package not installed. Please install it with pip install harmonypy")
        raise ImportError("harmonypy package required for batch correction")
    
    if batch_key not in adata.obs:
        logger.error(f"Batch key {batch_key} not found in AnnData object")
        raise ValueError(f"Batch key {batch_key} not found in AnnData object")
    if 'X_pca' not in adata.obsm:
        logger.info("Computing PCA for batch correction")
        sc.pp.pca(adata, n_comps=n_pcs)
    elif adata.obsm['X_pca'].shape[1] < n_pcs:
        logger.info(f"Recomputing PCA with {n_pcs} components for batch correction")
        sc.pp.pca(adata, n_comps=n_pcs)
    logger.info(f"Running Harmony batch correction using {batch_key}")
    pca_matrix = adata.obsm['X_pca']
    batch_labels = adata.obs[batch_key].values
    harmony_object = harmonypy.run_harmony(pca_matrix, adata.obs, batch_key)
    corrected_pca = harmony_object.Z_corr
    adata.obsm['X_pca_harmony'] = corrected_pca
    logger.info("Computing UMAP on batch-corrected PCA")
    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    
    return adata

def filter_by_total_counts(adata, min_counts=None, max_counts=None):
    """
    Filter spots by total counts
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    min_counts : float, optional
        Minimum total counts
    max_counts : float, optional
        Maximum total counts
        
    Returns
    -------
    adata : AnnData
        The filtered AnnData object
    """
    if 'total_counts' not in adata.obs:
        logger.info("Calculating QC metrics")
        sc.pp.calculate_qc_metrics(adata, inplace=True)
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    if min_counts is not None:
        min_mask = adata.obs['total_counts'] >= min_counts
        keep_mask = keep_mask & min_mask
        logger.info(f"Filtering spots with less than {min_counts} total counts: {np.sum(~min_mask)} spots will be removed")
    if max_counts is not None:
        max_mask = adata.obs['total_counts'] <= max_counts
        keep_mask = keep_mask & max_mask
        logger.info(f"Filtering spots with more than {max_counts} total counts: {np.sum(~max_mask)} spots will be removed")
    if not np.all(keep_mask):
        adata = adata[keep_mask].copy()
        logger.info(f"Filtered out {np.sum(~keep_mask)} spots by total counts. Remaining: {adata.n_obs}")
    
    return adata

def filter_by_genes_detected(adata, min_genes=None, max_genes=None):
    """
    Filter spots by number of genes detected
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    min_genes : int, optional
        Minimum number of genes detected
    max_genes : int, optional
        Maximum number of genes detected
        
    Returns
    -------
    adata : AnnData
        The filtered AnnData object
    """
    if 'n_genes_by_counts' not in adata.obs:
        logger.info("Calculating QC metrics")
        sc.pp.calculate_qc_metrics(adata, inplace=True)
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    if min_genes is not None:
        min_mask = adata.obs['n_genes_by_counts'] >= min_genes
        keep_mask = keep_mask & min_mask
        logger.info(f"Filtering spots with less than {min_genes} genes detected: {np.sum(~min_mask)} spots will be removed")
    if max_genes is not None:
        max_mask = adata.obs['n_genes_by_counts'] <= max_genes
        keep_mask = keep_mask & max_mask
        logger.info(f"Filtering spots with more than {max_genes} genes detected: {np.sum(~max_mask)} spots will be removed")
    if not np.all(keep_mask):
        adata = adata[keep_mask].copy()
        logger.info(f"Filtered out {np.sum(~keep_mask)} spots by genes detected. Remaining: {adata.n_obs}")
    
    return adata
