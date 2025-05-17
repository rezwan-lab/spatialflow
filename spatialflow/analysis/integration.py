import scanpy as sc
import numpy as np
import pandas as pd
import logging
from anndata import AnnData
from pathlib import Path

logger = logging.getLogger('spatialflow.analysis.integration')

def run_integration(adata, external_data, method="harmony", batch_key='batch', 
                  integration_key='integrated', **kwargs):
    """
    Run integration of spatial data with external datasets
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData or str or dict
        External data to integrate with. Can be an AnnData object, 
        a path to a file, or a dictionary mapping dataset names to either AnnData objects or file paths
    method : str, optional
        Integration method. Options: "harmony", "scanorama", "scvi", "ingest", "mnn"
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    **kwargs
        Additional parameters for the integration method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with integrated data
    """
    logger.info(f"Running integration using {method} method")
    if isinstance(external_data, str):
        logger.info(f"Loading external data from {external_data}")
        external_data = _load_external_data(external_data)
    
    elif isinstance(external_data, dict):
        logger.info(f"Loading {len(external_data)} external datasets")
        loaded_data = {}
        for name, data in external_data.items():
            if isinstance(data, str):
                loaded_data[name] = _load_external_data(data)
            else:
                loaded_data[name] = data
        external_data = loaded_data
    if method == "harmony":
        adata = _run_harmony_integration(adata, external_data, batch_key, integration_key, **kwargs)
    
    elif method == "scanorama":
        adata = _run_scanorama_integration(adata, external_data, batch_key, integration_key, **kwargs)
    
    elif method == "scvi":
        adata = _run_scvi_integration(adata, external_data, batch_key, integration_key, **kwargs)
    
    elif method == "ingest":
        adata = _run_ingest_integration(adata, external_data, batch_key, integration_key, **kwargs)
    
    elif method == "mnn":
        adata = _run_mnn_integration(adata, external_data, batch_key, integration_key, **kwargs)
    
    else:
        logger.error(f"Unsupported integration method: {method}")
        raise ValueError(f"Unsupported integration method: {method}")
    
    return adata

def _load_external_data(file_path):
    """
    Load external data from a file
    
    Parameters
    ----------
    file_path : str
        Path to the data file
    
    Returns
    -------
    AnnData
        AnnData object with the external data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    
    if suffix == '.h5ad':
        return sc.read_h5ad(file_path)
    
    elif suffix in ['.h5', '.hdf5']:
        return sc.read_10x_h5(file_path)
    
    elif suffix == '.csv':
        return sc.read_csv(file_path)
    
    elif suffix == '.mtx':
        return sc.read_mtx(file_path)
    
    else:
        logger.error(f"Unsupported file format: {suffix}")
        raise ValueError(f"Unsupported file format: {suffix}")

def _run_harmony_integration(adata, external_data, batch_key='batch', 
                          integration_key='X_harmony', n_pcs=30, **kwargs):
    """
    Run Harmony integration
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData or dict
        External data to integrate with
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    n_pcs : int, optional
        Number of principal components to use
    **kwargs
        Additional parameters for Harmony
    
    Returns
    -------
    adata : AnnData
        The integrated AnnData object
    """
    try:
        import harmonypy
    except ImportError:
        logger.error("harmonypy package not installed. Please install it with 'pip install harmonypy'")
        raise ImportError("harmonypy package required for Harmony integration")
    
    logger.info("Running Harmony integration")
    if isinstance(external_data, dict):
        logger.info(f"Combining spatial data with {len(external_data)} external datasets")
        combined = _combine_datasets(adata, external_data, batch_key)
    else:
        logger.info("Combining spatial data with one external dataset")
        external_data.obs[batch_key] = 'external'
        adata.obs[batch_key] = 'spatial'
        combined = adata.concatenate(external_data, batch_key=batch_key)
    logger.info("Normalizing and log-transforming combined data")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    logger.info("Identifying highly variable genes")
    sc.pp.highly_variable_genes(combined, n_top_genes=3000, batch_key=batch_key)
    logger.info(f"Running PCA with {n_pcs} components")
    sc.pp.pca(combined, n_comps=n_pcs, use_highly_variable=True)
    logger.info("Running Harmony batch correction")
    pca_matrix = combined.obsm['X_pca']
    harmony_result = harmonypy.run_harmony(pca_matrix, combined.obs, batch_key)
    combined.obsm[integration_key] = harmony_result.Z_corr
    logger.info("Computing neighbors on integrated components")
    sc.pp.neighbors(combined, use_rep=integration_key)
    logger.info("Running UMAP on integrated components")
    sc.tl.umap(combined)
    logger.info("Transferring integration results to the original object")
    spatial_mask = combined.obs[batch_key] == 'spatial'
    adata_idx_to_combined_idx = {}
    combined_spatial_indices = np.where(spatial_mask)[0]
    
    for i, barcode in enumerate(adata.obs.index):
        combined_idx = combined.obs.index.get_loc(barcode)
        adata_idx_to_combined_idx[i] = combined_idx
    adata.obsm[integration_key] = combined.obsm[integration_key][spatial_mask]
    if 'X_umap' in combined.obsm:
        adata.obsm['X_umap_integrated'] = combined.obsm['X_umap'][spatial_mask]
    adata.uns['integration'] = {
        'method': 'harmony',
        'n_pcs': n_pcs,
        'batch_key': batch_key,
        'n_batches': combined.obs[batch_key].nunique(),
        'external_data_shape': combined.shape[0] - adata.shape[0]
    }
    adata.uns['combined'] = combined
    
    return adata

def _run_scanorama_integration(adata, external_data, batch_key='batch', 
                            integration_key='X_scanorama', n_pcs=30, **kwargs):
    """
    Run Scanorama integration
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData or dict
        External data to integrate with
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    n_pcs : int, optional
        Number of principal components to use
    **kwargs
        Additional parameters for Scanorama
    
    Returns
    -------
    adata : AnnData
        The integrated AnnData object
    """
    try:
        import scanorama
    except ImportError:
        logger.error("scanorama package not installed. Please install it with 'pip install scanorama'")
        raise ImportError("scanorama package required for Scanorama integration")
    
    logger.info("Running Scanorama integration")
    datasets = []
    labels = []
    adata.obs[batch_key] = 'spatial'
    if 'log1p' not in adata.uns.get('preprocessed', {}):
        logger.info("Normalizing and log-transforming spatial data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if hasattr(adata.X, 'toarray'):
        datasets.append(adata.X.toarray())
    else:
        datasets.append(adata.X.copy())
    labels.append('spatial')
    if isinstance(external_data, dict):
        logger.info(f"Adding {len(external_data)} external datasets")
        
        for name, data in external_data.items():
            if 'log1p' not in data.uns.get('preprocessed', {}):
                logger.info(f"Normalizing and log-transforming {name} data")
                sc.pp.normalize_total(data, target_sum=1e4)
                sc.pp.log1p(data)
            if hasattr(data.X, 'toarray'):
                datasets.append(data.X.toarray())
            else:
                datasets.append(data.X.copy())
            labels.append(name)
            data.obs[batch_key] = name
    else:
        logger.info("Adding one external dataset")
        if 'log1p' not in external_data.uns.get('preprocessed', {}):
            logger.info("Normalizing and log-transforming external data")
            sc.pp.normalize_total(external_data, target_sum=1e4)
            sc.pp.log1p(external_data)
        if hasattr(external_data.X, 'toarray'):
            datasets.append(external_data.X.toarray())
        else:
            datasets.append(external_data.X.copy())
        labels.append('external')
        external_data.obs[batch_key] = 'external'
    if isinstance(external_data, dict):
        all_var_names = [adata.var_names] + [data.var_names for data in external_data.values()]
        common_genes = set(all_var_names[0])
        for var_names in all_var_names[1:]:
            common_genes &= set(var_names)
        common_genes = list(common_genes)
    else:
        common_genes = list(set(adata.var_names) & set(external_data.var_names))
    
    logger.info(f"Found {len(common_genes)} common genes")
    
    if len(common_genes) < 100:
        logger.warning("Very few common genes found. Integration may be inaccurate.")
    datasets_subset = []
    for i, dataset in enumerate(datasets):
        if i == 0:
            common_indices = [adata.var_names.get_loc(gene) for gene in common_genes]
            datasets_subset.append(dataset[:, common_indices])
        elif isinstance(external_data, dict):
            data = list(external_data.values())[i-1]
            common_indices = [data.var_names.get_loc(gene) for gene in common_genes]
            datasets_subset.append(dataset[:, common_indices])
        else:
            common_indices = [external_data.var_names.get_loc(gene) for gene in common_genes]
            datasets_subset.append(dataset[:, common_indices])
    logger.info("Running Scanorama batch correction")
    integrated, corrected = scanorama.integrate_scanpy(datasets_subset, labels, **kwargs)
    if isinstance(external_data, dict):
        logger.info(f"Combining spatial data with {len(external_data)} external datasets")
        combined = _combine_datasets(adata, external_data, batch_key)
    else:
        logger.info("Combining spatial data with one external dataset")
        combined = adata.concatenate(external_data, batch_key=batch_key)
    combined_common = combined[:, common_genes].copy()
    combined_common.obsm[integration_key] = np.vstack(integrated)
    logger.info("Computing neighbors on integrated components")
    sc.pp.neighbors(combined_common, use_rep=integration_key)
    logger.info("Running UMAP on integrated components")
    sc.tl.umap(combined_common)
    logger.info("Transferring integration results to the original object")
    spatial_mask = combined_common.obs[batch_key] == 'spatial'
    adata.obsm[integration_key] = combined_common.obsm[integration_key][spatial_mask]
    if 'X_umap' in combined_common.obsm:
        adata.obsm['X_umap_integrated'] = combined_common.obsm['X_umap'][spatial_mask]
    adata.uns['integration'] = {
        'method': 'scanorama',
        'batch_key': batch_key,
        'n_batches': combined_common.obs[batch_key].nunique(),
        'n_common_genes': len(common_genes),
        'external_data_shape': combined_common.shape[0] - adata.shape[0]
    }
    adata.uns['integration']['common_genes'] = common_genes
    adata.uns['combined'] = combined_common
    
    return adata

def _run_scvi_integration(adata, external_data, batch_key='batch', 
                        integration_key='X_scvi', n_latent=30, **kwargs):
    """
    Run scVI integration
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData or dict
        External data to integrate with
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    n_latent : int, optional
        Number of latent dimensions for scVI
    **kwargs
        Additional parameters for scVI
    
    Returns
    -------
    adata : AnnData
        The integrated AnnData object
    """
    try:
        import scvi
    except ImportError:
        logger.error("scvi-tools package not installed. Please install it with 'pip install scvi-tools'")
        raise ImportError("scvi-tools package required for scVI integration")
    
    logger.info("Running scVI integration")
    adata.obs[batch_key] = 'spatial'
    if isinstance(external_data, dict):
        logger.info(f"Combining spatial data with {len(external_data)} external datasets")
        for name, data in external_data.items():
            data.obs[batch_key] = name
        
        combined = _combine_datasets(adata, external_data, batch_key)
    else:
        logger.info("Combining spatial data with one external dataset")
        external_data.obs[batch_key] = 'external'
        
        combined = adata.concatenate(external_data, batch_key=batch_key)
    logger.info("Setting up scVI model")
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(combined, batch_key=batch_key)
    logger.info(f"Training scVI model with {n_latent} latent dimensions")
    model = scvi.model.SCVI(combined, n_latent=n_latent, **kwargs)
    model.train()
    logger.info("Getting latent representation")
    latent = model.get_latent_representation()
    combined.obsm[integration_key] = latent
    logger.info("Computing neighbors on integrated components")
    sc.pp.neighbors(combined, use_rep=integration_key)
    logger.info("Running UMAP on integrated components")
    sc.tl.umap(combined)
    logger.info("Transferring integration results to the original object")
    spatial_mask = combined.obs[batch_key] == 'spatial'
    adata.obsm[integration_key] = combined.obsm[integration_key][spatial_mask]
    if 'X_umap' in combined.obsm:
        adata.obsm['X_umap_integrated'] = combined.obsm['X_umap'][spatial_mask]
    adata.uns['integration'] = {
        'method': 'scvi',
        'batch_key': batch_key,
        'n_batches': combined.obs[batch_key].nunique(),
        'n_latent': n_latent,
        'external_data_shape': combined.shape[0] - adata.shape[0]
    }
    adata.uns['combined'] = combined
    adata.uns['scvi_model'] = model
    
    return adata

def _run_ingest_integration(adata, external_data, batch_key='batch', 
                          integration_key='X_emb', **kwargs):
    """
    Run ingest integration (mapping spatial data onto reference embedding)
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData
        External reference data to integrate with
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    **kwargs
        Additional parameters for sc.tl.ingest
    
    Returns
    -------
    adata : AnnData
        The integrated AnnData object
    """
    logger.info("Running ingest integration")
    if isinstance(external_data, dict):
        logger.error("Ingest method requires a single reference dataset, not a dictionary")
        raise ValueError("Ingest method requires a single reference dataset, not a dictionary")
    adata.obs[batch_key] = 'spatial'
    external_data.obs[batch_key] = 'reference'
    required_embeddings = ['X_pca', 'X_umap']
    missing_embeddings = [emb for emb in required_embeddings if emb not in external_data.obsm]
    
    if missing_embeddings:
        logger.info(f"Computing missing embeddings in reference: {missing_embeddings}")
        if 'log1p' not in external_data.uns.get('preprocessed', {}):
            logger.info("Normalizing and log-transforming reference data")
            sc.pp.normalize_total(external_data, target_sum=1e4)
            sc.pp.log1p(external_data)
        if 'X_pca' in missing_embeddings:
            logger.info("Computing PCA for reference data")
            sc.pp.pca(external_data)
        if 'neighbors' not in external_data.uns and 'X_umap' in missing_embeddings:
            logger.info("Computing neighbors for reference data")
            sc.pp.neighbors(external_data)
        if 'X_umap' in missing_embeddings:
            logger.info("Computing UMAP for reference data")
            sc.tl.umap(external_data)
    if 'log1p' not in adata.uns.get('preprocessed', {}):
        logger.info("Normalizing and log-transforming spatial data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    logger.info("Running ingest integration")
    sc.tl.ingest(adata, external_data, **kwargs)
    if integration_key != 'X_emb':
        adata.obsm[integration_key] = adata.obsm['X_emb'].copy()
    adata.uns['integration'] = {
        'method': 'ingest',
        'batch_key': batch_key,
        'reference_shape': external_data.shape
    }
    
    return adata

def _run_mnn_integration(adata, external_data, batch_key='batch', 
                       integration_key='X_mnn', n_pcs=30, **kwargs):
    """
    Run MNN (Mutual Nearest Neighbors) integration
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data : AnnData or dict
        External data to integrate with
    batch_key : str, optional
        Key in adata.obs for batch assignment
    integration_key : str, optional
        Key to add to adata.obsm for integration results
    n_pcs : int, optional
        Number of principal components to use
    **kwargs
        Additional parameters for MNN
    
    Returns
    -------
    adata : AnnData
        The integrated AnnData object
    """
    try:
        import mnnpy
    except ImportError:
        logger.error("mnnpy package not installed. Please install it with 'pip install mnnpy'")
        raise ImportError("mnnpy package required for MNN integration")
    
    logger.info("Running MNN integration")
    adata.obs[batch_key] = 'spatial'
    if isinstance(external_data, dict):
        logger.info(f"Combining spatial data with {len(external_data)} external datasets")
        for name, data in external_data.items():
            data.obs[batch_key] = name
        
        combined = _combine_datasets(adata, external_data, batch_key)
    else:
        logger.info("Combining spatial data with one external dataset")
        external_data.obs[batch_key] = 'external'
        
        combined = adata.concatenate(external_data, batch_key=batch_key)
    logger.info("Normalizing and log-transforming combined data")
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    logger.info("Identifying highly variable genes")
    sc.pp.highly_variable_genes(combined, n_top_genes=3000, batch_key=batch_key)
    batches = combined.obs[batch_key].unique().tolist()
    batch_adatas = []
    
    for batch in batches:
        batch_adatas.append(combined[combined.obs[batch_key] == batch])
    logger.info("Running MNN batch correction")
    corrected_data = mnnpy.mnn_correct(
        *[adata.X for adata in batch_adatas],
        var_subset=combined.var.highly_variable,
        batch_categories=[adata.obs[batch_key].iloc[0] for adata in batch_adatas],
        **kwargs
    )
    combined.uns['mnn'] = {}
    combined.uns['mnn']['merged_data'] = corrected_data[0]
    logger.info(f"Running PCA with {n_pcs} components")
    corrected_matrix = corrected_data[0]
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pcs)
    if hasattr(corrected_matrix, 'toarray'):
        corrected_matrix = corrected_matrix.toarray()
    
    pca_res = pca.fit_transform(corrected_matrix)
    combined.obsm[integration_key] = pca_res
    logger.info("Computing neighbors on integrated components")
    sc.pp.neighbors(combined, use_rep=integration_key)
    logger.info("Running UMAP on integrated components")
    sc.tl.umap(combined)
    logger.info("Transferring integration results to the original object")
    spatial_mask = combined.obs[batch_key] == 'spatial'
    adata.obsm[integration_key] = combined.obsm[integration_key][spatial_mask]
    if 'X_umap' in combined.obsm:
        adata.obsm['X_umap_integrated'] = combined.obsm['X_umap'][spatial_mask]
    adata.uns['integration'] = {
        'method': 'mnn',
        'batch_key': batch_key,
        'n_batches': combined.obs[batch_key].nunique(),
        'n_pcs': n_pcs,
        'external_data_shape': combined.shape[0] - adata.shape[0]
    }
    adata.uns['combined'] = combined
    
    return adata

def _combine_datasets(adata, external_data_dict, batch_key='batch'):
    """
    Combine spatial data with multiple external datasets
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    external_data_dict : dict
        Dictionary mapping dataset names to AnnData objects
    batch_key : str, optional
        Key in adata.obs for batch assignment
    
    Returns
    -------
    AnnData
        Combined AnnData object
    """
    adata.obs[batch_key] = 'spatial'
    adatas = [adata]
    for name, data in external_data_dict.items():
        data.obs[batch_key] = name
        adatas.append(data)
    return adata.concatenate(adatas[1:], batch_key=batch_key)

def transfer_labels(adata, reference_data, label_key='cell_type', 
                  transfer_key=None, method='knn', n_neighbors=30, **kwargs):
    """
    Transfer cell type labels from reference data to spatial data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    reference_data : AnnData
        Reference data with cell type annotations
    label_key : str, optional
        Key in reference_data.obs for cell type labels
    transfer_key : str, optional
        Key to add to adata.obs for transferred labels
    method : str, optional
        Method for label transfer. Options: 'knn', 'svm', 'logistic'
    n_neighbors : int, optional
        Number of neighbors for KNN classifier
    **kwargs
        Additional parameters for the classification method
    
    Returns
    -------
    adata : AnnData
        The AnnData object with transferred labels
    """
    if method == 'knn':
        return _transfer_labels_knn(adata, reference_data, label_key, transfer_key, n_neighbors, **kwargs)
    elif method == 'svm':
        return _transfer_labels_svm(adata, reference_data, label_key, transfer_key, **kwargs)
    elif method == 'logistic':
        return _transfer_labels_logistic(adata, reference_data, label_key, transfer_key, **kwargs)
    else:
        logger.error(f"Unsupported label transfer method: {method}")
        raise ValueError(f"Unsupported label transfer method: {method}")

def _transfer_labels_knn(adata, reference_data, label_key='cell_type', 
                       transfer_key=None, n_neighbors=30, use_rep='X_pca', **kwargs):
    """
    Transfer cell type labels using K-nearest neighbors
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    reference_data : AnnData
        Reference data with cell type annotations
    label_key : str, optional
        Key in reference_data.obs for cell type labels
    transfer_key : str, optional
        Key to add to adata.obs for transferred labels
    n_neighbors : int, optional
        Number of neighbors for KNN classifier
    use_rep : str, optional
        Representation to use for KNN
    **kwargs
        Additional parameters for KNN classifier
    
    Returns
    -------
    adata : AnnData
        The AnnData object with transferred labels
    """
    logger.info(f"Transferring labels using KNN with {n_neighbors} neighbors")
    if transfer_key is None:
        transfer_key = f"{label_key}_transferred"
    if label_key not in reference_data.obs:
        logger.error(f"Label key {label_key} not found in reference data")
        raise ValueError(f"Label key {label_key} not found in reference data")
    if use_rep not in adata.obsm or use_rep not in reference_data.obsm:
        logger.error(f"Representation {use_rep} not found in both datasets")
        if use_rep == 'X_pca':
            logger.info("Computing PCA for both datasets")
            for data, name in [(adata, 'spatial'), (reference_data, 'reference')]:
                if 'log1p' not in data.uns.get('preprocessed', {}):
                    logger.info(f"Normalizing and log-transforming {name} data")
                    sc.pp.normalize_total(data, target_sum=1e4)
                    sc.pp.log1p(data)
            sc.pp.pca(adata)
            sc.pp.pca(reference_data)
        else:
            raise ValueError(f"Representation {use_rep} not found in both datasets")
    X_spatial = adata.obsm[use_rep]
    X_reference = reference_data.obsm[use_rep]
    y_reference = reference_data.obs[label_key].values
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    knn.fit(X_reference, y_reference)
    y_pred = knn.predict(X_spatial)
    y_proba = knn.predict_proba(X_spatial)
    max_proba = np.max(y_proba, axis=1)
    adata.obs[transfer_key] = pd.Categorical(y_pred)
    adata.obs[f"{transfer_key}_probability"] = max_proba
    class_counts = adata.obs[transfer_key].value_counts()
    logger.info(f"Transferred labels: {class_counts.to_dict()}")
    adata.uns['label_transfer'] = {
        'method': 'knn',
        'n_neighbors': n_neighbors,
        'use_rep': use_rep,
        'reference_labels': reference_data.obs[label_key].cat.categories.tolist(),
        'class_counts': class_counts.to_dict()
    }
    
    return adata

def _transfer_labels_svm(adata, reference_data, label_key='cell_type', 
                       transfer_key=None, use_rep='X_pca', **kwargs):
    """
    Transfer cell type labels using Support Vector Machine
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    reference_data : AnnData
        Reference data with cell type annotations
    label_key : str, optional
        Key in reference_data.obs for cell type labels
    transfer_key : str, optional
        Key to add to adata.obs for transferred labels
    use_rep : str, optional
        Representation to use for SVM
    **kwargs
        Additional parameters for SVM classifier
    
    Returns
    -------
    adata : AnnData
        The AnnData object with transferred labels
    """
    logger.info("Transferring labels using SVM")
    if transfer_key is None:
        transfer_key = f"{label_key}_transferred"
    if label_key not in reference_data.obs:
        logger.error(f"Label key {label_key} not found in reference data")
        raise ValueError(f"Label key {label_key} not found in reference data")
    if use_rep not in adata.obsm or use_rep not in reference_data.obsm:
        logger.error(f"Representation {use_rep} not found in both datasets")
        if use_rep == 'X_pca':
            logger.info("Computing PCA for both datasets")
            for data, name in [(adata, 'spatial'), (reference_data, 'reference')]:
                if 'log1p' not in data.uns.get('preprocessed', {}):
                    logger.info(f"Normalizing and log-transforming {name} data")
                    sc.pp.normalize_total(data, target_sum=1e4)
                    sc.pp.log1p(data)
            sc.pp.pca(adata)
            sc.pp.pca(reference_data)
        else:
            raise ValueError(f"Representation {use_rep} not found in both datasets")
    X_spatial = adata.obsm[use_rep]
    X_reference = reference_data.obsm[use_rep]
    y_reference = reference_data.obs[label_key].values
    from sklearn.svm import SVC
    svm = SVC(probability=True, **kwargs)
    svm.fit(X_reference, y_reference)
    y_pred = svm.predict(X_spatial)
    y_proba = svm.predict_proba(X_spatial)
    max_proba = np.max(y_proba, axis=1)
    adata.obs[transfer_key] = pd.Categorical(y_pred)
    adata.obs[f"{transfer_key}_probability"] = max_proba
    class_counts = adata.obs[transfer_key].value_counts()
    logger.info(f"Transferred labels: {class_counts.to_dict()}")
    adata.uns['label_transfer'] = {
        'method': 'svm',
        'use_rep': use_rep,
        'reference_labels': reference_data.obs[label_key].cat.categories.tolist(),
        'class_counts': class_counts.to_dict()
    }
    
    return adata

def _transfer_labels_logistic(adata, reference_data, label_key='cell_type', 
                            transfer_key=None, use_rep='X_pca', **kwargs):
    """
    Transfer cell type labels using Logistic Regression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    reference_data : AnnData
        Reference data with cell type annotations
    label_key : str, optional
        Key in reference_data.obs for cell type labels
    transfer_key : str, optional
        Key to add to adata.obs for transferred labels
    use_rep : str, optional
        Representation to use for logistic regression
    **kwargs
        Additional parameters for logistic regression classifier
    
    Returns
    -------
    adata : AnnData
        The AnnData object with transferred labels
    """
    logger.info("Transferring labels using Logistic Regression")
    if transfer_key is None:
        transfer_key = f"{label_key}_transferred"
    if label_key not in reference_data.obs:
        logger.error(f"Label key {label_key} not found in reference data")
        raise ValueError(f"Label key {label_key} not found in reference data")
    if use_rep not in adata.obsm or use_rep not in reference_data.obsm:
        logger.error(f"Representation {use_rep} not found in both datasets")
        if use_rep == 'X_pca':
            logger.info("Computing PCA for both datasets")
            for data, name in [(adata, 'spatial'), (reference_data, 'reference')]:
                if 'log1p' not in data.uns.get('preprocessed', {}):
                    logger.info(f"Normalizing and log-transforming {name} data")
                    sc.pp.normalize_total(data, target_sum=1e4)
                    sc.pp.log1p(data)
            sc.pp.pca(adata)
            sc.pp.pca(reference_data)
        else:
            raise ValueError(f"Representation {use_rep} not found in both datasets")
    X_spatial = adata.obsm[use_rep]
    X_reference = reference_data.obsm[use_rep]
    y_reference = reference_data.obs[label_key].values
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(max_iter=1000, **kwargs)
    logreg.fit(X_reference, y_reference)
    y_pred = logreg.predict(X_spatial)
    y_proba = logreg.predict_proba(X_spatial)
    max_proba = np.max(y_proba, axis=1)
    adata.obs[transfer_key] = pd.Categorical(y_pred)
    adata.obs[f"{transfer_key}_probability"] = max_proba
    class_counts = adata.obs[transfer_key].value_counts()
    logger.info(f"Transferred labels: {class_counts.to_dict()}")
    adata.uns['label_transfer'] = {
        'method': 'logistic_regression',
        'use_rep': use_rep,
        'reference_labels': reference_data.obs[label_key].cat.categories.tolist(),
        'class_counts': class_counts.to_dict()
    }
    
    return adata

def integrate_with_image_data(adata, image_path, library_id=None, 
                             extraction_method='cnn', n_features=50, **kwargs):
    """
    Integrate spatial transcriptomics data with histology image data
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    image_path : str or Path
        Path to the histology image file
    library_id : str, optional
        Library ID for Visium data
    extraction_method : str, optional
        Method for feature extraction. Options: 'cnn', 'hog', 'color', 'custom'
    n_features : int, optional
        Number of features to extract from the image
    **kwargs
        Additional parameters for feature extraction
    
    Returns
    -------
    adata : AnnData
        The AnnData object with added image features
    """
    try:
        from skimage import io, color, feature
    except ImportError:
        logger.error("scikit-image package not installed. Please install it with 'pip install scikit-image'")
        raise ImportError("scikit-image package required for image processing")
    
    logger.info(f"Integrating spatial data with image data using {extraction_method} method")
    image_path = Path(image_path)
    
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    logger.info(f"Loading image from {image_path}")
    img = io.imread(image_path)
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}
    
    if library_id is None:
        library_id = "visium_data"
    
    if library_id not in adata.uns['spatial']:
        adata.uns['spatial'][library_id] = {}
    
    if 'images' not in adata.uns['spatial'][library_id]:
        adata.uns['spatial'][library_id]['images'] = {}
    
    adata.uns['spatial'][library_id]['images']['hires'] = img
    if 'scalefactors' not in adata.uns['spatial'][library_id]:
        adata.uns['spatial'][library_id]['scalefactors'] = {
            'tissue_hires_scalef': 1.0,
            'spot_diameter_fullres': 50.0
        }
    if extraction_method == 'cnn':
        features = _extract_cnn_features(adata, img, n_features, **kwargs)
    elif extraction_method == 'hog':
        features = _extract_hog_features(adata, img, **kwargs)
    elif extraction_method == 'color':
        features = _extract_color_features(adata, img, **kwargs)
    elif extraction_method == 'custom':
        if 'extraction_function' not in kwargs:
            logger.error("Custom extraction method requires 'extraction_function' parameter")
            raise ValueError("Custom extraction method requires 'extraction_function' parameter")
        
        extraction_function = kwargs.pop('extraction_function')
        features = extraction_function(adata, img, **kwargs)
    else:
        logger.error(f"Unsupported feature extraction method: {extraction_method}")
        raise ValueError(f"Unsupported feature extraction method: {extraction_method}")
    if features is not None:
        adata.obsm['image_features'] = features
        if 'run_combined_pca' in kwargs and kwargs['run_combined_pca']:
            _run_combined_dim_reduction(adata, features, n_comps=kwargs.get('n_comps', 30))
    adata.uns['image_integration'] = {
        'method': extraction_method,
        'n_features': features.shape[1] if features is not None else 0,
        'image_path': str(image_path),
        'library_id': library_id
    }
    
    return adata

def _extract_cnn_features(adata, img, n_features=50, model_name='vgg16', **kwargs):
    """
    Extract CNN features from image patches
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    img : numpy.ndarray
        The image array
    n_features : int, optional
        Number of features to extract
    model_name : str, optional
        CNN model to use. Options: 'vgg16', 'resnet50', 'inception_v3'
    **kwargs
        Additional parameters for feature extraction
    
    Returns
    -------
    numpy.ndarray
        Array of image features
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import vgg16, resnet50, inception_v3
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.image import img_to_array
    except ImportError:
        logger.error("TensorFlow not installed. Please install with 'pip install tensorflow'")
        raise ImportError("TensorFlow required for CNN feature extraction")
    
    logger.info(f"Extracting CNN features using {model_name} model")
    coords = adata.obsm['spatial']
    library_id = kwargs.get('library_id', list(adata.uns['spatial'].keys())[0])
    scalefactors = adata.uns['spatial'][library_id]['scalefactors']
    if 'tissue_hires_scalef' in scalefactors:
        coords_pixels = coords * scalefactors['tissue_hires_scalef']
    else:
        coords_pixels = coords
    if 'spot_diameter_fullres' in scalefactors:
        spot_size = scalefactors['spot_diameter_fullres']
    else:
        spot_size = 50.0
    if model_name == 'vgg16':
        base_model = vgg16.VGG16(weights='imagenet', include_top=False)
        preprocess_input = vgg16.preprocess_input
    elif model_name == 'resnet50':
        base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet50.preprocess_input
    elif model_name == 'inception_v3':
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        preprocess_input = inception_v3.preprocess_input
    else:
        logger.error(f"Unsupported CNN model: {model_name}")
        raise ValueError(f"Unsupported CNN model: {model_name}")
    input_shape = base_model.input_shape[1:3]
    model = Model(inputs=base_model.input, outputs=base_model.output)
    all_features = []
    for i in range(coords_pixels.shape[0]):
        x, y = coords_pixels[i].astype(int)
        half_size = int(spot_size / 2)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        patch = img[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            all_features.append(np.zeros((1, n_features)))
            continue
        from skimage.transform import resize
        patch_resized = resize(patch, input_shape)
        patch_array = img_to_array(patch_resized)
        patch_batch = np.expand_dims(patch_array, axis=0)
        patch_preprocessed = preprocess_input(patch_batch)
        features = model.predict(patch_preprocessed)
        features = features.reshape((1, -1))
        all_features.append(features)
    all_features = np.vstack(all_features)
    if all_features.shape[1] > n_features:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_features)
        all_features = pca.fit_transform(all_features)
    
    return all_features

def _extract_hog_features(adata, img, **kwargs):
    """
    Extract HOG (Histogram of Oriented Gradients) features from image patches
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    img : numpy.ndarray
        The image array
    **kwargs
        Additional parameters for feature extraction
    
    Returns
    -------
    numpy.ndarray
        Array of image features
    """
    from skimage import feature, color, transform
    
    logger.info("Extracting HOG features")
    coords = adata.obsm['spatial']
    library_id = kwargs.get('library_id', list(adata.uns['spatial'].keys())[0])
    scalefactors = adata.uns['spatial'][library_id]['scalefactors']
    if 'tissue_hires_scalef' in scalefactors:
        coords_pixels = coords * scalefactors['tissue_hires_scalef']
    else:
        coords_pixels = coords
    if 'spot_diameter_fullres' in scalefactors:
        spot_size = scalefactors['spot_diameter_fullres']
    else:
        spot_size = 50.0
    pixels_per_cell = kwargs.get('pixels_per_cell', (8, 8))
    cells_per_block = kwargs.get('cells_per_block', (3, 3))
    orientations = kwargs.get('orientations', 9)
    all_features = []
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    for i in range(coords_pixels.shape[0]):
        x, y = coords_pixels[i].astype(int)
        half_size = int(spot_size / 2)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        patch = img_gray[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            all_features.append(np.zeros(108))  # Default HOG size
            continue
        patch_resized = transform.resize(patch, (64, 64))
        hog_features = feature.hog(
            patch_resized,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False
        )
        all_features.append(hog_features)
    all_features = np.vstack(all_features)
    
    return all_features

def _extract_color_features(adata, img, **kwargs):
    """
    Extract color features from image patches
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    img : numpy.ndarray
        The image array
    **kwargs
        Additional parameters for feature extraction
    
    Returns
    -------
    numpy.ndarray
        Array of image features
    """
    from skimage import color
    
    logger.info("Extracting color features")
    if len(img.shape) < 3 or img.shape[2] < 3:
        logger.error("Image must be RGB for color feature extraction")
        raise ValueError("Image must be RGB for color feature extraction")
    coords = adata.obsm['spatial']
    library_id = kwargs.get('library_id', list(adata.uns['spatial'].keys())[0])
    scalefactors = adata.uns['spatial'][library_id]['scalefactors']
    if 'tissue_hires_scalef' in scalefactors:
        coords_pixels = coords * scalefactors['tissue_hires_scalef']
    else:
        coords_pixels = coords
    if 'spot_diameter_fullres' in scalefactors:
        spot_size = scalefactors['spot_diameter_fullres']
    else:
        spot_size = 50.0
    all_features = []
    img_hsv = color.rgb2hsv(img)
    img_lab = color.rgb2lab(img)
    for i in range(coords_pixels.shape[0]):
        x, y = coords_pixels[i].astype(int)
        half_size = int(spot_size / 2)
        x_min = max(0, x - half_size)
        x_max = min(img.shape[1], x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(img.shape[0], y + half_size)
        if x_min >= x_max or y_min >= y_max:
            all_features.append(np.zeros(15))  # Default color feature size
            continue
        patch_rgb = img[y_min:y_max, x_min:x_max]
        patch_hsv = img_hsv[y_min:y_max, x_min:x_max]
        patch_lab = img_lab[y_min:y_max, x_min:x_max]
        if patch_rgb.size == 0:
            all_features.append(np.zeros(15))  # Default color feature size
            continue
        features = []
        for c in range(3):  # R, G, B channels
            features.append(np.mean(patch_rgb[:, :, c]))
            features.append(np.std(patch_rgb[:, :, c]))
        for c in range(2):  # H, S channels
            features.append(np.mean(patch_hsv[:, :, c]))
            features.append(np.std(patch_hsv[:, :, c]))
        for c in range(1, 3):  # a, b channels
            features.append(np.mean(patch_lab[:, :, c]))
            features.append(np.std(patch_lab[:, :, c]))
        gray = color.rgb2gray(patch_rgb)
        features.append(np.std(gray))
        all_features.append(features)
    all_features = np.vstack(all_features)
    
    return all_features

def _run_combined_dim_reduction(adata, image_features, n_comps=30):
    """
    Run dimension reduction on combined gene expression and image features
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial data
    image_features : numpy.ndarray
        Array of image features
    n_comps : int, optional
        Number of components for dimension reduction
    
    Returns
    -------
    None
        Modifies adata in place
    """
    logger.info("Running dimension reduction on combined features")
    if hasattr(adata.X, 'toarray'):
        gene_expr = adata.X.toarray()
    else:
        gene_expr = adata.X.copy()
    from sklearn.preprocessing import StandardScaler
    scaler_gene = StandardScaler()
    scaler_img = StandardScaler()
    
    gene_expr_scaled = scaler_gene.fit_transform(gene_expr)
    image_features_scaled = scaler_img.fit_transform(image_features)
    combined_features = np.hstack([gene_expr_scaled, image_features_scaled])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comps)
    combined_pca = pca.fit_transform(combined_features)
    adata.obsm['X_combined_pca'] = combined_pca
    from sklearn.neighbors import NearestNeighbors
    from umap import UMAP
    nn = NearestNeighbors(n_neighbors=15)
    nn.fit(combined_pca)
    umap = UMAP(random_state=42)
    combined_umap = umap.fit_transform(combined_pca)
    adata.obsm['X_combined_umap'] = combined_umap
    adata.uns['combined_dim_reduction'] = {
        'n_comps': n_comps,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'gene_expr_weight': gene_expr.shape[1] / combined_features.shape[1],
        'image_weight': image_features.shape[1] / combined_features.shape[1]
    }
