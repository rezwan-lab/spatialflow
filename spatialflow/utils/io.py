import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import json
import logging
from pathlib import Path
import h5py
import shutil
import datetime
import warnings

logger = logging.getLogger('spatialflow.utils.io')

def save_results(adata, output_dir, save_formats=None, compress=True, **kwargs):
    """
    Save analysis results in various formats
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    output_dir : str or Path
        Directory to save results
    save_formats : list, optional
        List of formats to save. Options: 'h5ad', 'csv', 'loom', 'zarr', 'mtx'
        If None, saves only in 'h5ad' format
    compress : bool, optional
        Whether to compress the data
    **kwargs
        Additional arguments for saving
        
    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_formats is None:
        save_formats = ['h5ad']
    valid_formats = ['h5ad', 'csv', 'loom', 'zarr', 'mtx', 'json']
    for fmt in save_formats:
        if fmt not in valid_formats:
            logger.warning(f"Unsupported save format: {fmt}. Skipping.")
            save_formats.remove(fmt)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}
    for fmt in save_formats:
        try:
            if fmt == 'h5ad':
                file_path = output_dir / f"spatialflow_results_{timestamp}.h5ad"
                
                try:
                    adata.write(file_path, compression='gzip' if compress else None)
                except Exception as e:
                    logger.warning(f"Could not save with full compression: {str(e)}")
                    logger.warning("Trying with minimal attributes")
                    minimal_adata = sc.AnnData(
                        X=adata.X,
                        obs=adata.obs,
                        var=adata.var,
                        obsm={'spatial': adata.obsm['spatial']} if 'spatial' in adata.obsm else {}
                    )
                    if 'spatial' in adata.uns:
                        minimal_adata.uns['spatial'] = adata.uns['spatial']
                    if 'spatial_connectivities' in adata.obsp:
                        minimal_adata.obsp['spatial_connectivities'] = adata.obsp['spatial_connectivities']
                    minimal_adata.write(file_path, compression='gzip' if compress else None)
                
                saved_paths['h5ad'] = file_path
                logger.info(f"Saved AnnData object to {file_path}")
            
            elif fmt == 'csv':
                csv_dir = output_dir / f"spatialflow_csv_{timestamp}"
                csv_dir.mkdir(exist_ok=True)
                if hasattr(adata.X, 'toarray'):
                    expr_df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
                else:
                    expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
                
                expr_path = csv_dir / "expression_matrix.csv"
                expr_df.to_csv(expr_path)
                obs_path = csv_dir / "observation_metadata.csv"
                adata.obs.to_csv(obs_path)
                var_path = csv_dir / "variable_metadata.csv"
                adata.var.to_csv(var_path)
                if 'spatial' in adata.obsm:
                    coords_df = pd.DataFrame(
                        adata.obsm['spatial'],
                        index=adata.obs_names,
                        columns=['x_coord', 'y_coord']
                    )
                    coords_path = csv_dir / "spatial_coordinates.csv"
                    coords_df.to_csv(coords_path)
                readme_path = csv_dir / "README.txt"
                with open(readme_path, 'w') as f:
                    f.write("SpatialFlow Analysis Results\n")
                    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Files:\n")
                    f.write("- expression_matrix.csv: Gene expression matrix (spots x genes)\n")
                    f.write("- observation_metadata.csv: Metadata for spots\n")
                    f.write("- variable_metadata.csv: Metadata for genes\n")
                    if 'spatial' in adata.obsm:
                        f.write("- spatial_coordinates.csv: Spatial coordinates for spots\n")
                
                saved_paths['csv'] = csv_dir
                logger.info(f"Saved CSV files to {csv_dir}")
            
            elif fmt == 'loom':
                file_path = output_dir / f"spatialflow_results_{timestamp}.loom"
                adata.write_loom(file_path)
                
                saved_paths['loom'] = file_path
                logger.info(f"Saved Loom file to {file_path}")
            
            elif fmt == 'zarr':
                file_path = output_dir / f"spatialflow_results_{timestamp}.zarr"
                adata.write_zarr(file_path, chunks=True if compress else None)
                
                saved_paths['zarr'] = file_path
                logger.info(f"Saved Zarr directory to {file_path}")
            
            elif fmt == 'mtx':
                mtx_dir = output_dir / f"spatialflow_mtx_{timestamp}"
                mtx_dir.mkdir(exist_ok=True)
                sc.write(mtx_dir / "matrix", adata)
                if 'spatial' in adata.obsm:
                    coords_df = pd.DataFrame(
                        adata.obsm['spatial'],
                        index=adata.obs_names,
                        columns=['x_coord', 'y_coord']
                    )
                    coords_path = mtx_dir / "spatial_coordinates.csv"
                    coords_df.to_csv(coords_path)
                
                saved_paths['mtx'] = mtx_dir
                logger.info(f"Saved MTX files to {mtx_dir}")
            
            elif fmt == 'json':
                json_dir = output_dir / f"spatialflow_json_{timestamp}"
                json_dir.mkdir(exist_ok=True)
                obs_dict = {}
                for col in adata.obs.columns:
                    if pd.api.types.is_categorical_dtype(adata.obs[col]):
                        obs_dict[col] = adata.obs[col].astype(str).to_dict()
                    elif pd.api.types.is_numeric_dtype(adata.obs[col]):
                        obs_dict[col] = {str(idx): float(val) if pd.notnull(val) else None 
                                       for idx, val in adata.obs[col].items()}
                    else:
                        obs_dict[col] = {str(idx): str(val) if pd.notnull(val) else None 
                                       for idx, val in adata.obs[col].items()}
                obs_path = json_dir / "observation_metadata.json"
                with open(obs_path, 'w') as f:
                    json.dump(obs_dict, f, indent=2)
                if 'spatial' in adata.obsm:
                    coords_dict = {
                        str(idx): [float(coords[0]), float(coords[1])]
                        for idx, coords in zip(adata.obs_names, adata.obsm['spatial'])
                    }
                    
                    coords_path = json_dir / "spatial_coordinates.json"
                    with open(coords_path, 'w') as f:
                        json.dump(coords_dict, f, indent=2)
                uns_dict = {}
                for key, value in adata.uns.items():
                    try:
                        json.dumps(value)
                        uns_dict[key] = value
                    except (TypeError, OverflowError):
                        logger.warning(f"Could not JSON-serialize uns['{key}']. Skipping.")
                
                if uns_dict:
                    uns_path = json_dir / "unstructured_data.json"
                    with open(uns_path, 'w') as f:
                        json.dump(uns_dict, f, indent=2)
                
                saved_paths['json'] = json_dir
                logger.info(f"Saved JSON files to {json_dir}")
        
        except Exception as e:
            logger.error(f"Error saving in {fmt} format: {str(e)}")
    manifest_path = output_dir / f"spatialflow_manifest_{timestamp}.json"
    manifest = {
        'timestamp': timestamp,
        'formats': {fmt: str(path) for fmt, path in saved_paths.items()},
        'dataset_shape': list(adata.shape),
        'obs_columns': list(adata.obs.columns),
        'var_columns': list(adata.var.columns),
        'obsm_keys': list(adata.obsm.keys()),
        'obsp_keys': list(adata.obsp.keys()),
        'uns_keys': list(adata.uns.keys())
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created manifest file at {manifest_path}")
    
    return saved_paths

def read_results(input_path, format=None, **kwargs):
    """
    Read analysis results from file
    
    Parameters
    ----------
    input_path : str or Path
        Path to the file or directory with results
    format : str, optional
        Format of the input. If None, inferred from file extension
        Options: 'h5ad', 'csv', 'loom', 'zarr', 'mtx'
    **kwargs
        Additional arguments for reading
        
    Returns
    -------
    adata : AnnData
        AnnData object with loaded results
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if format is None:
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix == '.h5ad':
                format = 'h5ad'
            elif suffix == '.loom':
                format = 'loom'
            elif suffix == '.csv':
                format = 'csv'
        elif input_path.is_dir():
            if (input_path / '.zgroup').exists():
                format = 'zarr'
            elif (input_path / 'matrix.mtx').exists() or (input_path / 'matrix.mtx.gz').exists():
                format = 'mtx'
            elif (input_path / 'expression_matrix.csv').exists():
                format = 'csv'
    
    if format is None:
        raise ValueError(f"Could not infer format from path: {input_path}")
    
    logger.info(f"Reading results in {format} format from {input_path}")
    if format == 'h5ad':
        adata = sc.read_h5ad(input_path)
    
    elif format == 'loom':
        adata = sc.read_loom(input_path)
    
    elif format == 'zarr':
        adata = sc.read_zarr(input_path)
    
    elif format == 'mtx':
        adata = sc.read_10x_mtx(input_path, **kwargs)
        coords_path = input_path / "spatial_coordinates.csv"
        if coords_path.exists():
            coords_df = pd.read_csv(coords_path, index_col=0)
            common_indices = coords_df.index.intersection(adata.obs_names)
            if len(common_indices) < len(adata.obs_names):
                logger.warning(f"Only {len(common_indices)} out of {len(adata.obs_names)} spots have spatial coordinates")
                adata = adata[common_indices].copy()
            adata.obsm['spatial'] = coords_df.loc[adata.obs_names].values
    
    elif format == 'csv':
        if input_path.is_file():
            expr_df = pd.read_csv(input_path, index_col=0)
            adata = ad.AnnData(X=expr_df.values, obs=pd.DataFrame(index=expr_df.index), var=pd.DataFrame(index=expr_df.columns))
        
        else:
            expr_path = input_path / "expression_matrix.csv"
            if not expr_path.exists():
                raise FileNotFoundError(f"Expression matrix file not found: {expr_path}")
            expr_df = pd.read_csv(expr_path, index_col=0)
            obs_path = input_path / "observation_metadata.csv"
            obs = pd.DataFrame(index=expr_df.index)
            if obs_path.exists():
                obs = pd.read_csv(obs_path, index_col=0)
                common_indices = obs.index.intersection(expr_df.index)
                if len(common_indices) < len(expr_df.index):
                    logger.warning(f"Only {len(common_indices)} out of {len(expr_df.index)} spots have metadata")
                    expr_df = expr_df.loc[common_indices]
            var_path = input_path / "variable_metadata.csv"
            var = pd.DataFrame(index=expr_df.columns)
            if var_path.exists():
                var = pd.read_csv(var_path, index_col=0)
                common_indices = var.index.intersection(expr_df.columns)
                if len(common_indices) < len(expr_df.columns):
                    logger.warning(f"Only {len(common_indices)} out of {len(expr_df.columns)} genes have metadata")
                    expr_df = expr_df[common_indices]
            adata = ad.AnnData(X=expr_df.values, obs=obs.loc[expr_df.index], var=var.loc[expr_df.columns])
            coords_path = input_path / "spatial_coordinates.csv"
            if coords_path.exists():
                coords_df = pd.read_csv(coords_path, index_col=0)
                common_indices = coords_df.index.intersection(adata.obs_names)
                if len(common_indices) < len(adata.obs_names):
                    logger.warning(f"Only {len(common_indices)} out of {len(adata.obs_names)} spots have spatial coordinates")
                    adata = adata[common_indices].copy()
                adata.obsm['spatial'] = coords_df.loc[adata.obs_names].values
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Successfully loaded AnnData object with shape {adata.shape}")
    
    return adata

def export_figures(adata, output_dir, formats=None, figsize=(10, 8), dpi=300, **kwargs):
    """
    Export figures from analysis results
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    output_dir : str or Path
        Directory to save figures
    formats : list, optional
        List of file formats to save figures in. Options: 'png', 'pdf', 'svg'
        If None, saves in 'png' format
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        Resolution for raster graphics
    **kwargs
        Additional arguments for plotting
        
    Returns
    -------
    dict
        Dictionary with paths to saved figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ['png']
    valid_formats = ['png', 'pdf', 'svg', 'jpg']
    for fmt in formats:
        if fmt not in valid_formats:
            logger.warning(f"Unsupported figure format: {fmt}. Skipping.")
            formats.remove(fmt)
    figure_paths = {}
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    scatter_figs = {}
    if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:
        for var, title in [('total_counts', 'Total UMI Counts'), ('n_genes_by_counts', 'Genes Detected per Spot')]:
            if kwargs.get('spatial_scatter', True):
                fig = plt.figure(figsize=figsize)
                sc.pl.spatial(adata, color=var, title=title, size=kwargs.get('spot_size', 1.5), show=False)
                paths = []
                for fmt in formats:
                    fig_path = output_dir / f"spatial_{var}.{fmt}"
                    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    paths.append(fig_path)
                
                plt.close(fig)
                scatter_figs[var] = paths
                logger.info(f"Saved spatial scatter plot for {var}")
    cluster_keys = []
    for col in adata.obs.columns:
        if 'cluster' in col.lower() or any(x in col.lower() for x in ['leiden', 'louvain', 'kmeans', 'domain']):
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                cluster_keys.append(col)
    
    for key in cluster_keys:
        if kwargs.get('spatial_scatter', True):
            fig = plt.figure(figsize=figsize)
            sc.pl.spatial(adata, color=key, title=f"Clustering: {key}", size=kwargs.get('spot_size', 1.5), show=False)
            paths = []
            for fmt in formats:
                fig_path = output_dir / f"spatial_clusters_{key}.{fmt}"
                plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                paths.append(fig_path)
            
            plt.close(fig)
            scatter_figs[f"clusters_{key}"] = paths
            logger.info(f"Saved spatial scatter plot for clusters: {key}")
    if 'moranI' in adata.uns:
        top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:5].tolist()
        
        for gene in top_spatial_genes:
            if gene in adata.var_names and kwargs.get('spatial_scatter', True):
                fig = plt.figure(figsize=figsize)
                sc.pl.spatial(adata, color=gene, title=f"Gene Expression: {gene}", size=kwargs.get('spot_size', 1.5), show=False)
                paths = []
                for fmt in formats:
                    fig_path = output_dir / f"spatial_gene_{gene}.{fmt}"
                    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    paths.append(fig_path)
                
                plt.close(fig)
                scatter_figs[f"gene_{gene}"] = paths
                logger.info(f"Saved spatial scatter plot for gene: {gene}")
    
    figure_paths['spatial_scatter'] = scatter_figs
    dr_figs = {}
    
    if 'X_pca' in adata.obsm:
        fig = plt.figure(figsize=figsize)
        if cluster_keys:
            color = cluster_keys[0]
        else:
            color = 'total_counts' if 'total_counts' in adata.obs else None
        
        sc.pl.pca(adata, color=color, show=False)
        paths = []
        for fmt in formats:
            fig_path = output_dir / f"pca.{fmt}"
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            paths.append(fig_path)
        
        plt.close(fig)
        dr_figs['pca'] = paths
        logger.info("Saved PCA plot")
    
    figure_paths['dimensionality_reduction'] = dr_figs
    qc_figs = {}
    
    if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axs[0], show=False)
        sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axs[1], show=False)
        plt.tight_layout()
        paths = []
        for fmt in formats:
            fig_path = output_dir / f"qc_metrics.{fmt}"
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            paths.append(fig_path)
        
        plt.close(fig)
        qc_figs['qc_metrics'] = paths
        logger.info("Saved QC metrics plot")
    
    figure_paths['qc'] = qc_figs
    heatmap_figs = {}
    if 'highly_variable' in adata.var and adata.var.highly_variable.sum() > 0:
        fig = plt.figure(figsize=figsize)
        
        hvg = adata.var.index[adata.var.highly_variable]
        n_hvg = min(50, len(hvg))
        
        if cluster_keys:
            sc.pl.heatmap(adata, hvg[:n_hvg], groupby=cluster_keys[0], show=False)
        else:
            sc.pl.heatmap(adata, hvg[:n_hvg], show=False)
        paths = []
        for fmt in formats:
            fig_path = output_dir / f"heatmap_hvg.{fmt}"
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            paths.append(fig_path)
        
        plt.close(fig)
        heatmap_figs['hvg'] = paths
        logger.info("Saved highly variable genes heatmap")
    if 'moranI' in adata.uns:
        fig = plt.figure(figsize=figsize)
        
        top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:20].tolist()
        
        if cluster_keys:
            sc.pl.heatmap(adata, top_spatial_genes, groupby=cluster_keys[0], show=False)
        else:
            sc.pl.heatmap(adata, top_spatial_genes, show=False)
        paths = []
        for fmt in formats:
            fig_path = output_dir / f"heatmap_spatial_genes.{fmt}"
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            paths.append(fig_path)
        
        plt.close(fig)
        heatmap_figs['spatial_genes'] = paths
        logger.info("Saved spatially variable genes heatmap")
    
    figure_paths['heatmaps'] = heatmap_figs
    
    logger.info(f"Exported {sum(len(paths) for category in figure_paths.values() for paths in category.values())} figures to {output_dir}")
    
    return figure_paths

def backup_data(adata, backup_dir, backup_name=None, include_raw=True, compress=True):
    """
    Create a backup of the AnnData object
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to back up
    backup_dir : str or Path
        Directory to save the backup
    backup_name : str, optional
        Name for the backup. If None, a timestamped name is generated
    include_raw : bool, optional
        Whether to include raw data in the backup
    compress : bool, optional
        Whether to compress the backup
        
    Returns
    -------
    Path
        Path to the backup file
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if backup_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"spatialflow_backup_{timestamp}.h5ad"
    
    backup_path = backup_dir / backup_name
    
    logger.info(f"Creating backup of AnnData object to {backup_path}")
    adata_copy = adata.copy()
    if not include_raw and adata_copy.raw is not None:
        adata_copy.raw = None
    try:
        adata_copy.write(backup_path, compression='gzip' if compress else None)
        logger.info(f"Backup created successfully")
        return backup_path
    
    except Exception as e:
        logger.warning(f"Could not create full backup: {str(e)}")
        logger.warning("Creating a minimal backup with essential data only")
        minimal_adata = sc.AnnData(
            X=adata.X,
            obs=adata.obs,
            var=adata.var,
            obsm={'spatial': adata.obsm['spatial']} if 'spatial' in adata.obsm else {}
        )
        if 'spatial' in adata.uns:
            minimal_adata.uns['spatial'] = adata.uns['spatial']
        if 'X_pca' in adata.obsm:
            minimal_adata.obsm['X_pca'] = adata.obsm['X_pca']
        if 'spatial_neighbors' in adata.uns and 'spatial_connectivities' in adata.obsp:
            minimal_adata.uns['spatial_neighbors'] = adata.uns['spatial_neighbors']
            minimal_adata.obsp['spatial_connectivities'] = adata.obsp['spatial_connectivities']
        minimal_adata.write(backup_path, compression='gzip' if compress else None)
        logger.info(f"Minimal backup created successfully")
        
        return backup_path