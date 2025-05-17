import scanpy as sc
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger('spatialflow.core.data_loader')

def load_data(input_path, format="10x_visium", **kwargs):
    """
    Load data from various formats into an AnnData object
    
    Parameters
    ----------
    input_path : str
        Path to the input data file
    format : str
        Format of the input data
        Supported formats: 10x_visium, anndata
    **kwargs : dict
        Additional arguments to pass to the reader function
    
    Returns
    -------
    adata : AnnData
        AnnData object containing the loaded data
    """
    logger.info(f"Loading data from {input_path} (format: {format})")
    
    if format == "10x_visium":
        adata = read_10x_visium_h5(input_path, **kwargs)
    elif format == "anndata":
        import scanpy as sc
        adata = sc.read_h5ad(input_path)
    else:
        raise ValueError(f"Unsupported data format: {format}")
    
    return adata

def read_10x_visium_directory(path, count_file='filtered_feature_bc_matrix.h5', 
                             spatial_file='spatial/tissue_positions_list.csv'):
    """
    Read a 10x Visium dataset from directory structure
    
    Parameters
    ----------
    path : str or Path
        Path to the Visium directory
    count_file : str
        Name of the count matrix file within the directory
    spatial_file : str
        Name of the spatial coordinates file within the directory
        
    Returns
    -------
    adata : AnnData
        AnnData object with the Visium data
    """
    path = Path(path)
    counts_path = path / count_file
    if not counts_path.exists():
        raise FileNotFoundError(f"Count matrix file not found: {counts_path}")
    
    adata = sc.read_10x_h5(counts_path)
    spatial_path = path / spatial_file
    if spatial_path.exists():
        spatial_df = pd.read_csv(spatial_path, header=None)
        if spatial_df.shape[1] >= 5:
            barcodes = spatial_df.iloc[:, 0].values
            coords = spatial_df.iloc[:, 2:4].values
            mask = np.isin(adata.obs.index, barcodes)
            adata = adata[mask].copy()
            coord_dict = {bc: coord for bc, coord in zip(barcodes, coords)}
            coords_array = np.array([coord_dict[bc] for bc in adata.obs.index])
            adata.obsm['spatial'] = coords_array
    else:
        logger.warning(f"Spatial file not found: {spatial_path}")
    
    return adata

def read_10x_visium_h5(path, **kwargs):
    """
    Read a 10x Visium dataset from an H5 file
    
    Parameters
    ----------
    path : str or Path
        Path to the H5 file
        
    Returns
    -------
    adata : AnnData
        AnnData object with the Visium data
    """
    adata = sc.read_10x_h5(path)
    if 'spatial' not in adata.uns:
        logger.warning("No spatial information found in the H5 file. Is this a Visium dataset?")
    
    return adata

def read_csv_to_anndata(path, sep=',', gene_columns=None, spatial_columns=None, **kwargs):
    """
    Read a CSV file into an AnnData object
    
    Parameters
    ----------
    path : str or Path
        Path to the CSV file
    sep : str
        Separator in the CSV file
    gene_columns : list or None
        List of column names to use as gene expression data
        If None, all columns except those specified in spatial_columns will be used
    spatial_columns : list or None
        List of column names to use as spatial coordinates
        If None, will look for columns named 'x', 'y', 'X', 'Y', 'x_coord', 'y_coord'
        
    Returns
    -------
    adata : AnnData
        AnnData object with the data from the CSV file
    """
    df = pd.read_csv(path, sep=sep, **kwargs)
    if spatial_columns is None:
        potential_coords = ['x', 'y', 'X', 'Y', 'x_coord', 'y_coord']
        spatial_columns = [col for col in potential_coords if col in df.columns]
        
        if len(spatial_columns) >= 2:
            logger.info(f"Using columns {spatial_columns[:2]} as spatial coordinates")
            spatial_columns = spatial_columns[:2]
        else:
            logger.warning("No spatial coordinate columns detected")
            spatial_columns = []
    if gene_columns is None:
        gene_columns = [col for col in df.columns if col not in spatial_columns]
        logger.info(f"Using {len(gene_columns)} columns as gene expression data")
    X = df[gene_columns].values
    var = pd.DataFrame(index=gene_columns)
    obs = pd.DataFrame(index=df.index)
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if len(spatial_columns) >= 2:
        adata.obsm['spatial'] = df[spatial_columns[:2]].values
    
    return adata

def add_image_data(adata, image_path, library_id=None):
    """
    Add image data to an AnnData object
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to add image data to
    image_path : str or Path
        Path to the image file
    library_id : str, optional
        Library ID to use for the image data
        
    Returns
    -------
    adata : AnnData
        AnnData object with added image data
    """
    from skimage import io
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = io.imread(image_path)
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}
    
    if library_id is None:
        library_id = "visium_library"
    
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
    
    logger.info(f"Added image with shape {img.shape} to AnnData object")
    
    return adata

def merge_datasets(adatas, batch_key=None, batch_categories=None):
    """
    Merge multiple AnnData objects into one
    
    Parameters
    ----------
    adatas : list of AnnData
        List of AnnData objects to merge
    batch_key : str, optional
        Name of the batch annotation to add
    batch_categories : list, optional
        Names of the batches
        
    Returns
    -------
    adata : AnnData
        Merged AnnData object
    """
    if len(adatas) == 0:
        raise ValueError("No datasets provided to merge")
    
    if len(adatas) == 1:
        return adatas[0].copy()
    
    if batch_key is None:
        batch_key = 'batch'
    
    if batch_categories is None:
        batch_categories = [f'batch_{i}' for i in range(len(adatas))]
    
    if len(batch_categories) != len(adatas):
        raise ValueError("Number of batch categories must match number of datasets")
    for i, adata in enumerate(adatas):
        adata.obs[batch_key] = batch_categories[i]
    merged = adatas[0].concatenate(
        adatas[1:],
        batch_key=batch_key,
        batch_categories=batch_categories,
        join='outer'
    )
    
    logger.info(f"Merged {len(adatas)} datasets into one with shape {merged.shape}")
    
    return merged
