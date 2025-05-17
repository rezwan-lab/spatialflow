import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from anndata import AnnData
from pathlib import Path

logger = logging.getLogger('spatialflow.visualization.static')

def generate_static_figures(adata, output_dir, figsize=(10, 8), dpi=300, 
                          color_by=None, library_id=None, **kwargs):
    """
    Generate static figures for spatial data visualization
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    output_dir : str or Path
        Directory to save figures
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        DPI for saved figures
    color_by : list, optional
        List of variables to color by
    library_id : str, optional
        Library ID for Visium data
    **kwargs
        Additional parameters for visualization functions
    
    Returns
    -------
    dict
        Dictionary with paths to generated figures
    """
    plt.ioff()
    matplotlib.rcParams['figure.figsize'] = figsize
    matplotlib.rcParams['figure.dpi'] = dpi
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures to absolute path: {output_dir.absolute()}")
    figure_paths = {}
    if color_by is None:
        color_by = _get_default_color_variables(adata)
    
    logger.info(f"Generating static figures, coloring by: {color_by}")
    if 'spatial' in adata.obsm:
        logger.info("Generating spatial plots")
        for var in color_by:
            if var in adata.obs or var in adata.var_names:
                try:
                    fig, ax = plt.subplots(figsize=figsize)
                    sq.pl.spatial_scatter(adata, color=var, library_id=library_id, 
                                         ax=ax, **kwargs)
                    fig_path = output_dir / f"spatial_{var}.png"
                    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    figure_paths[f"spatial_{var}"] = fig_path
                    logger.info(f"Saved spatial plot for {var} to {fig_path}")
                except Exception as e:
                    logger.error(f"Error generating spatial plot for {var}: {str(e)}")
    try:
        logger.info("Generating QC plots")
        if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:
            fig, ax = plt.subplots(figsize=figsize)
            sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=ax)
            fig_path = output_dir / "qc_scatter.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            figure_paths["qc_scatter"] = fig_path
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axes[0])
        sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axes[1])
        fig_path = output_dir / "qc_violin.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        figure_paths["qc_violin"] = fig_path
        if 'spatial' in adata.obsm:
            for metric in ['total_counts', 'n_genes_by_counts']:
                if metric in adata.obs:
                    fig, ax = plt.subplots(figsize=figsize)
                    sq.pl.spatial_scatter(adata, color=metric, library_id=library_id, 
                                       ax=ax, **kwargs)
                    fig_path = output_dir / f"spatial_{metric}.png"
                    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    figure_paths[f"spatial_{metric}"] = fig_path
    
    except Exception as e:
        logger.error(f"Error generating QC plots: {str(e)}")
    if 'X_pca' in adata.obsm:
        try:
            logger.info("Generating PCA plots")
            for i, color in enumerate(color_by[:min(len(color_by), 4)]):
                fig, ax = plt.subplots(figsize=figsize)
                sc.pl.pca(adata, color=color, ax=ax)
                fig_path = output_dir / f"pca_scatter_{i}_{color}.png"
                fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                figure_paths[f"pca_scatter_{i}"] = fig_path
            fig, ax = plt.subplots(figsize=figsize)
            sc.pl.pca_variance_ratio(adata, log=True, ax=ax)
            fig_path = output_dir / "pca_variance.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            figure_paths["pca_variance"] = fig_path
        
        except Exception as e:
            logger.error(f"Error generating PCA plots: {str(e)}")
    if 'X_umap' in adata.obsm:
        try:
            logger.info("Generating UMAP plots")
            for i, color in enumerate(color_by[:min(len(color_by), 4)]):
                fig, ax = plt.subplots(figsize=figsize)
                sc.pl.umap(adata, color=color, ax=ax)
                fig_path = output_dir / f"umap_scatter_{i}_{color}.png"
                fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                figure_paths[f"umap_scatter_{i}"] = fig_path
        
        except Exception as e:
            logger.error(f"Error generating UMAP plots: {str(e)}")
    cluster_keys = [key for key in adata.obs.columns if 
                   any(key.startswith(prefix) for prefix in 
                      ['leiden', 'louvain', 'kmeans', 'spectral', 'spatial_domains'])]
    
    if cluster_keys:
        try:
            logger.info(f"Generating clustering plots for {cluster_keys}")
            
            for key in cluster_keys:
                if 'spatial' in adata.obsm:
                    fig, ax = plt.subplots(figsize=figsize)
                    sq.pl.spatial_scatter(adata, color=key, library_id=library_id, 
                                       ax=ax, **kwargs)
                    fig_path = output_dir / f"spatial_{key}.png"
                    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    figure_paths[f"spatial_{key}"] = fig_path
                if 'X_umap' in adata.obsm:
                    fig, ax = plt.subplots(figsize=figsize)
                    sc.pl.umap(adata, color=key, ax=ax)
                    fig_path = output_dir / f"umap_{key}.png"
                    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    figure_paths[f"umap_{key}"] = fig_path
        
        except Exception as e:
            logger.error(f"Error generating clustering plots: {str(e)}")
    if 'rank_genes_groups' in adata.uns:
        try:
            logger.info("Generating marker gene plots")
            try:
                fig, ax = plt.subplots(figsize=figsize)
                sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, show=False)
                heatmap_fig = plt.gcf()
                fig_path = output_dir / "marker_genes_heatmap.png"
                heatmap_fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                plt.close(heatmap_fig)
                plt.close(fig)  # Close the original figure too
                figure_paths["marker_genes_heatmap"] = fig_path
            except Exception as e:
                logger.error(f"Error generating heatmap: {str(e)}")
            try:
                fig, ax = plt.subplots(figsize=figsize)
                sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False)
                dotplot_fig = plt.gcf()
                fig_path = output_dir / "marker_genes_dotplot.png"
                dotplot_fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                plt.close(dotplot_fig)
                plt.close(fig)  # Close the original figure too
                figure_paths["marker_genes_dotplot"] = fig_path
            except Exception as e:
                logger.error(f"Error generating dotplot: {str(e)}")
            if 'spatial' in adata.obsm:
                try:
                    top_markers = []
                    for group in adata.uns['rank_genes_groups']['names'].dtype.names:
                        markers = adata.uns['rank_genes_groups']['names'][group][:2]
                        top_markers.extend(markers)
                    
                    top_markers = list(dict.fromkeys(top_markers))  # Remove duplicates
                    for gene in top_markers[:6]:  # Limit to top 6 genes
                        if gene in adata.var_names:
                            fig, ax = plt.subplots(figsize=figsize)
                            sq.pl.spatial_scatter(adata, color=gene, library_id=library_id, 
                                               ax=ax, **kwargs)
                            fig_path = output_dir / f"spatial_marker_{gene}.png"
                            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                            plt.close(fig)
                            figure_paths[f"spatial_marker_{gene}"] = fig_path
                except Exception as e:
                    logger.error(f"Error generating spatial marker gene plots: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating marker gene plots: {str(e)}")
    try:
        logger.info("Generating summary panel")
        
        fig = plt.figure(figsize=(figsize[0]*2, figsize[1]*2))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=ax1)
        ax1.set_title('QC Metrics')
        if 'X_pca' in adata.obsm:
            ax2 = fig.add_subplot(gs[0, 1])
            sc.pl.pca(adata, color=color_by[0] if color_by else None, ax=ax2)
            ax2.set_title('PCA')
        if 'X_umap' in adata.obsm:
            ax3 = fig.add_subplot(gs[0, 2])
            sc.pl.umap(adata, color=color_by[0] if color_by else None, ax=ax3)
            ax3.set_title('UMAP')
        if 'spatial' in adata.obsm:
            ax4 = fig.add_subplot(gs[1, 0])
            if cluster_keys:
                sq.pl.spatial_scatter(adata, color=cluster_keys[0], 
                                  ax=ax4)
                ax4.set_title(f'Spatial ({cluster_keys[0]})')
            else:
                sq.pl.spatial_scatter(adata, color='total_counts', 
                                  ax=ax4)
                ax4.set_title('Spatial (total counts)')
        if 'rank_genes_groups' in adata.uns and 'spatial' in adata.obsm:
            ax5 = fig.add_subplot(gs[1, 1])
            try:
                top_gene = adata.uns['rank_genes_groups']['names'][0][0]
                sq.pl.spatial_scatter(adata, color=top_gene, 
                                  ax=ax5)
                ax5.set_title(f'Top Marker: {top_gene}')
            except:
                pass
        if 'spatial' in adata.obsm and len(color_by) >= 2:
            ax6 = fig.add_subplot(gs[1, 2])
            var = color_by[1]
            sq.pl.spatial_scatter(adata, color=var, 
                              ax=ax6)
            ax6.set_title(f'Spatial ({var})')
        
        plt.tight_layout()
        fig_path = output_dir / "summary_panel.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        figure_paths["summary_panel"] = fig_path
    
    except Exception as e:
        logger.error(f"Error generating summary panel: {str(e)}")
    
    logger.info(f"Generated {len(figure_paths)} figures and saved to {output_dir}")
    
    return figure_paths

def _get_default_color_variables(adata):
    """
    Get default variables to color by based on AnnData content
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    
    Returns
    -------
    list
        List of variables to color by
    """
    color_vars = []
    for col in ['total_counts', 'n_genes_by_counts']:
        if col in adata.obs:
            color_vars.append(col)
    cluster_keys = [key for key in adata.obs.columns if 
                   any(key.startswith(prefix) for prefix in 
                      ['leiden', 'louvain', 'kmeans', 'spectral', 'spatial_domains'])]
    
    color_vars.extend(cluster_keys)
    cell_type_keys = [key for key in adata.obs.columns if 
                     any(key.startswith(prefix) for prefix in
                        ['cell_type', 'celltype', 'deconv', 'prop_'])]
    
    color_vars.extend(cell_type_keys[:5])  # Limit to 5 cell type variables
    if 'moranI' in adata.uns:
        try:
            top_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:5].tolist()
            color_vars.extend(top_genes)
        except:
            pass
    if 'highly_variable' in adata.var.columns:
        top_hvgs = adata.var_names[adata.var['highly_variable']].tolist()[:5]
        color_vars.extend(top_hvgs)
    return color_vars[:20]

def plot_cell_type_proportions(adata, deconv_key=None, cell_type_prefix='prop_', 
                              n_top=None, output_dir=None, figsize=(12, 8), dpi=300):
    """
    Plot cell type proportions from deconvolution
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    deconv_key : str, optional
        Key in adata.obsm for deconvolution results
    cell_type_prefix : str, optional
        Prefix for cell type proportions in adata.obs
    n_top : int, optional
        Number of top cell types to show
    output_dir : str or Path, optional
        Directory to save figure
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        DPI for saved figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    plt.ioff()
    
    logger.info("Plotting cell type proportions")
    if deconv_key is not None and deconv_key in adata.obsm:
        logger.info(f"Using deconvolution results from adata.obsm['{deconv_key}']")
        cell_props = adata.obsm[deconv_key].copy()
    else:
        prop_cols = [col for col in adata.obs.columns if col.startswith(cell_type_prefix)]
        
        if not prop_cols:
            logger.error(f"No columns with prefix {cell_type_prefix} found in adata.obs")
            raise ValueError(f"No columns with prefix {cell_type_prefix} found in adata.obs")
        
        logger.info(f"Using cell type proportions from adata.obs with prefix '{cell_type_prefix}'")
        cell_props = adata.obs[prop_cols].copy()
        cell_props.columns = [col[len(cell_type_prefix):] for col in prop_cols]
    mean_props = cell_props.mean().sort_values(ascending=False)
    
    if n_top is not None and n_top < len(mean_props):
        top_cell_types = mean_props.index[:n_top]
        cell_props = cell_props[top_cell_types]
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=cell_props, orient='h', ax=ax, palette='Set2')
    sns.swarmplot(data=cell_props, orient='h', ax=ax, color='0.3', size=3, alpha=0.7)
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Cell Type')
    ax.set_title('Cell Type Proportions')
    for i, (cell_type, mean_prop) in enumerate(mean_props.items()):
        if cell_type in cell_props.columns:
            ax.text(mean_prop + 0.02, i, f'μ = {mean_prop:.3f}', 
                   va='center', fontsize=9, color='#555555')
    plt.tight_layout()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / "cell_type_proportions.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved cell type proportions figure to {fig_path}")
    plt.close(fig)
    
    return fig

def plot_spatial_domains(adata, domain_key='spatial_domains', color_key=None, library_id=None, 
                       output_dir=None, figsize=(10, 8), dpi=300, **kwargs):
    """
    Plot spatial domains with additional metrics
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    domain_key : str, optional
        Key in adata.obs for spatial domain assignments
    color_key : str, optional
        Key in adata.obs for additional coloring
    library_id : str, optional
        Library ID for Visium data
    output_dir : str or Path, optional
        Directory to save figure
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        DPI for saved figure
    **kwargs
        Additional parameters for sq.pl.spatial_scatter
    
    Returns
    -------
    list
        List of matplotlib figures
    """
    plt.ioff()
    
    logger.info(f"Plotting spatial domains from {domain_key}")
    if domain_key not in adata.obs:
        logger.error(f"Domain key {domain_key} not found in adata.obs")
        raise ValueError(f"Domain key {domain_key} not found in adata.obs")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    
    figures = []
    fig1, ax1 = plt.subplots(figsize=figsize)
    sq.pl.spatial_scatter(adata, color=domain_key, library_id=library_id, 
                       ax=ax1, **kwargs)
    ax1.set_title(f'Spatial Domains ({domain_key})')
    figures.append(fig1)
    if color_key is not None and color_key in adata.obs:
        fig2, ax2 = plt.subplots(figsize=figsize)
        sq.pl.spatial_scatter(adata, color=color_key, library_id=library_id, 
                           ax=ax2, **kwargs)
        ax2.set_title(f'Spatial Map ({color_key})')
        figures.append(fig2)
    domain_stats = adata.obs.groupby(domain_key).size().reset_index()
    domain_stats.columns = [domain_key, 'Count']
    domain_stats['Percentage'] = domain_stats['Count'] / domain_stats['Count'].sum() * 100
    
    fig3, ax3 = plt.subplots(figsize=figsize)
    sns.barplot(x=domain_key, y='Percentage', data=domain_stats, ax=ax3)
    ax3.set_xlabel('Spatial Domain')
    ax3.set_ylabel('Percentage of Spots')
    ax3.set_title('Spatial Domain Distribution')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    for i, (_, row) in enumerate(domain_stats.iterrows()):
        ax3.text(i, row['Percentage'] + 1, f"n = {row['Count']}", 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    figures.append(fig3)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig1_path = output_dir / f"spatial_domains_{domain_key}.png"
        fig1.savefig(fig1_path, dpi=dpi, bbox_inches='tight')
        
        if color_key is not None and color_key in adata.obs:
            fig2_path = output_dir / f"spatial_map_{color_key}.png"
            fig2.savefig(fig2_path, dpi=dpi, bbox_inches='tight')
        
        fig3_path = output_dir / f"domain_stats_{domain_key}.png"
        fig3.savefig(fig3_path, dpi=dpi, bbox_inches='tight')
        
        logger.info(f"Saved spatial domain figures to {output_dir}")
    for fig in figures:
        plt.close(fig)
    
    return figures

def plot_spatial_heterogeneity(adata, gene_list=None, library_id=None, n_genes=4,
                             output_dir=None, figsize=(10, 8), dpi=300, **kwargs):
    """
    Plot spatial heterogeneity of gene expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    gene_list : list, optional
        List of genes to plot
    library_id : str, optional
        Library ID for Visium data
    n_genes : int, optional
        Number of genes to plot if gene_list is not provided
    output_dir : str or Path, optional
        Directory to save figure
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        DPI for saved figure
    **kwargs
        Additional parameters for sq.pl.spatial_scatter
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    plt.ioff()
    
    logger.info("Plotting spatial heterogeneity of gene expression")
    if 'spatial' not in adata.obsm:
        logger.error("No spatial coordinates found in AnnData object")
        raise ValueError("No spatial coordinates found in AnnData object")
    if gene_list is None:
        if 'moranI' in adata.uns:
            try:
                gene_list = adata.uns['moranI'].sort_values('I', ascending=False).index[:n_genes].tolist()
                logger.info(f"Using top {n_genes} spatially variable genes")
            except:
                logger.warning("Could not get top spatially variable genes, using random highly variable genes")
                if 'highly_variable' in adata.var.columns:
                    gene_list = adata.var_names[adata.var['highly_variable']].tolist()[:n_genes]
                else:
                    gene_list = adata.var_names[:n_genes].tolist()
        elif 'highly_variable' in adata.var.columns:
            gene_list = adata.var_names[adata.var['highly_variable']].tolist()[:n_genes]
            logger.info(f"Using top {n_genes} highly variable genes")
        else:
            gene_list = adata.var_names[:n_genes].tolist()
            logger.info(f"Using first {n_genes} genes in adata.var_names")
    gene_list = [gene for gene in gene_list if gene in adata.var_names]
    
    if not gene_list:
        logger.error("No valid genes to plot")
        raise ValueError("No valid genes to plot")
    n_cols = min(2, len(gene_list))
    n_rows = (len(gene_list) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(figsize[0] * n_cols / 2, figsize[1] * n_rows / 2))
    for i, gene in enumerate(gene_list):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        sq.pl.spatial_scatter(adata, color=gene, library_id=library_id, 
                           ax=ax, **kwargs)
        ax.set_title(gene)
    
    plt.tight_layout()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / "spatial_heterogeneity.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved spatial heterogeneity figure to {fig_path}")
    plt.close(fig)
    
    return fig

def plot_spatial_correlation(adata, genes=None, n_top_genes=10, include_pvalues=True,
                           output_dir=None, figsize=(12, 10), dpi=300):
    """
    Plot correlation between spatial location and gene expression
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    genes : list, optional
        List of genes to include
    n_top_genes : int, optional
        Number of top spatially variable genes to include if genes is None
    include_pvalues : bool, optional
        Whether to include p-values in the plot
    output_dir : str or Path, optional
        Directory to save figure
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        DPI for saved figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    plt.ioff()
    
    logger.info("Plotting spatial correlation")
    if 'moranI' not in adata.uns:
        logger.error("No spatial autocorrelation results found in adata.uns['moranI']")
        raise ValueError("No spatial autocorrelation results found in adata.uns['moranI']")
    moran_df = adata.uns['moranI'].copy()
    pval_col = None
    if include_pvalues:
        for col in moran_df.columns:
            if col.lower().startswith('pval'):
                pval_col = col
                break
    if genes is not None:
        valid_genes = [gene for gene in genes if gene in moran_df.index]
        if not valid_genes:
            logger.error("None of the specified genes found in Moran's I results")
            raise ValueError("None of the specified genes found in Moran's I results")
        moran_df = moran_df.loc[valid_genes]
    else:
        moran_df = moran_df.sort_values('I', ascending=False).head(n_top_genes)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y=moran_df.index, width=moran_df['I'], color='skyblue')
    if pval_col is not None and pval_col in moran_df.columns:
        for i, (gene, row) in enumerate(moran_df.iterrows()):
            pval = row[pval_col]
            stars = ''
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            
            if stars:
                ax.text(row['I'] + 0.01, i, stars, va='center', fontsize=10)
    ax.set_xlabel("Moran's I")
    ax.set_ylabel("Gene")
    ax.set_title("Spatial Autocorrelation (Moran's I)")
    if pval_col is not None:
        plt.text(0.95, 0.05, '* p < 0.05\n** p < 0.01\n*** p < 0.001',
                transform=ax.transAxes, horizontalalignment='right',
                verticalalignment='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / "spatial_correlation.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved spatial correlation figure to {fig_path}")
    plt.close(fig)
    
    return fig