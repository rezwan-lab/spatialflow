import os
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json
from datetime import datetime
import base64

logger = logging.getLogger('spatialflow.visualization.reporting')

def generate_report(adata, output_dir, title="Spatial Transcriptomics Analysis Report",
                   figsize=(12, 10), dpi=120, **kwargs):
    """
    Generate a comprehensive HTML report of the analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    output_dir : str or Path
        Directory to save the report
    title : str, optional
        Title of the report
    figsize : tuple, optional
        Figure size for plots
    dpi : int, optional
        DPI for plots
    **kwargs
        Additional arguments for report generation
        
    Returns
    -------
    Path
        Path to the generated report
    """
    logger.info(f"Generating analysis report in {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    report_data = _collect_report_data(adata, figures_dir, figsize, dpi)
    html_report = _create_html_report(adata, report_data, title)
    report_path = output_dir / "spatialflow_report.html"
    with open(report_path, 'w') as f:
        f.write(html_report)
    
    logger.info(f"Report generated successfully: {report_path}")
    
    return report_path

def _collect_report_data(adata, figures_dir, figsize, dpi):
    """
    Collect data and generate figures for the report
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    figures_dir : Path
        Directory to save the figures
    figsize : tuple
        Figure size
    dpi : int
        DPI for figures
        
    Returns
    -------
    dict
        Dictionary with report data
    """
    report_data = {
        'dataset': {},
        'preprocessing': {},
        'spatial': {},
        'clustering': {},
        'deconvolution': {},
        'ligrec': {},
        'pathways': {},
        'figures': {}
    }
    report_data['dataset'] = {
        'n_spots': adata.n_obs,
        'n_genes': adata.n_vars,
        'dataset_shape': str(adata.shape),
        'obs_columns': adata.obs.columns.tolist(),
        'var_columns': adata.var.columns.tolist(),
        'obsm_keys': list(adata.obsm.keys()),
        'uns_keys': list(adata.uns.keys()),
    }
    if 'highly_variable' in adata.var:
        report_data['preprocessing']['n_hvg'] = adata.var.highly_variable.sum()
    
    if 'qc_metrics' in adata.uns:
        report_data['preprocessing']['qc_metrics'] = True
        
    if 'pca' in adata.uns and 'variance_ratio' in adata.uns['pca']:
        var_ratio = adata.uns['pca']['variance_ratio']
        report_data['preprocessing']['pca_var_explained'] = np.cumsum(var_ratio).tolist()
    if 'spatial' in adata.obsm:
        report_data['spatial']['spatial_coords_available'] = True
        
    if 'spatial_neighbors' in adata.uns:
        report_data['spatial']['spatial_neighbors_available'] = True
        
    if 'moranI' in adata.uns:
        report_data['spatial']['moranI_available'] = True
        report_data['spatial']['n_spatial_genes'] = len(adata.uns['moranI'])
        top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).head(10)
        report_data['spatial']['top_spatial_genes'] = top_spatial_genes.to_dict('index')
    cluster_keys = []
    for col in adata.obs.columns:
        if 'cluster' in col.lower() or any(x in col.lower() for x in ['leiden', 'louvain', 'kmeans']):
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                cluster_keys.append(col)
    
    report_data['clustering']['cluster_keys'] = cluster_keys
    if cluster_keys:
        report_data['clustering']['n_clusters'] = {key: adata.obs[key].nunique() for key in cluster_keys}
    deconv_keys = [col for col in adata.obs.columns if any(x in col for x in ['prop_', 'cell_type_', 'deconv_'])]
    report_data['deconvolution']['deconv_keys'] = deconv_keys
    ligrec_keys = [key for key in adata.uns.keys() if 'ligrec' in key]
    report_data['ligrec']['ligrec_keys'] = ligrec_keys
    
    if ligrec_keys:
        for key in ligrec_keys[:1]:  # Just use the first key for the report
            if isinstance(adata.uns[key], dict) and 'means' in adata.uns[key]:
                means = adata.uns[key]['means']
                if isinstance(means, pd.DataFrame):
                    report_data['ligrec']['sample_interactions'] = means.head(10).to_dict('list')
                elif isinstance(means, dict):
                    sample_keys = list(means.keys())[:5]
                    report_data['ligrec']['sample_interactions'] = {str(k): str(means[k]) for k in sample_keys}
    pathway_keys = [col for col in adata.obs.columns if 'pathway_' in col]
    report_data['pathways']['pathway_keys'] = pathway_keys
    report_data['figures'] = _generate_report_figures(adata, figures_dir, figsize, dpi)
    
    return report_data

def _get_pathway_data(adata):
    """Get pathway analysis data for the report"""
    pathway_data = {
        'has_results': False,
        'top_pathways': []
    }
    pathway_keys = [col for col in adata.obs.columns if 'pathway_' in col]
    if pathway_keys:
        pathway_data['pathway_keys'] = pathway_keys
        pathway_data['has_results'] = True
    if 'gsea' in adata.uns and adata.uns['gsea']:
        pathway_data['has_results'] = True
        pathway_data['gsea_results'] = []
        for comparison, results in adata.uns['gsea'].items():
            if isinstance(results, dict):
                comparison_data = {'name': comparison, 'pathways': []}
                if 'fdr' in results and isinstance(results['fdr'], dict):
                    for pathway, fdr in results['fdr'].items():
                        if fdr < 0.25:  # Standard GSEA significance threshold
                            pathway_info = {
                                'name': pathway,
                                'fdr': fdr,
                                'nes': results.get('nes', {}).get(pathway, 'N/A'),
                                'pval': results.get('pval', {}).get(pathway, 'N/A'),
                                'size': results.get('geneset_size', {}).get(pathway, 'N/A')
                            }
                            comparison_data['pathways'].append(pathway_info)
                if comparison_data['pathways']:
                    pathway_data['gsea_results'].append(comparison_data)
    
    return pathway_data

def _generate_report_figures(adata, figures_dir, figsize, dpi):
    """
    Generate figures for the report
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    figures_dir : Path
        Directory to save the figures
    figsize : tuple
        Figure size
    dpi : int
        DPI for figures
        
    Returns
    -------
    dict
        Dictionary with figure paths
    """
    figures = {}
    import matplotlib
    matplotlib.use('Agg')
    def save_fig(fig, name):
        path = figures_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return os.path.join("figures", f"{name}.png")
    try:
        fig = plt.figure(figsize=figsize)
        if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:
            ax1 = fig.add_subplot(121)
            sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=ax1, show=False)
            
            ax2 = fig.add_subplot(122)
            sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=ax2, show=False)
            
            fig.tight_layout()
            figures['qc_metrics'] = save_fig(fig, "qc_metrics")
    except Exception as e:
        logger.warning(f"Could not generate QC metrics figure: {str(e)}")
    try:
        if 'spatial' in adata.obsm:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
            coords = adata.obsm['spatial']
            ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.7)
            ax.set_title('Spot Coordinates')
            ax.set_xlabel('Spatial X')
            ax.set_ylabel('Spatial Y')
            ax.set_aspect('equal')
            
            figures['spatial_coords'] = save_fig(fig, "spatial_coords")
    except Exception as e:
        logger.warning(f"Could not generate spatial coordinates figure: {str(e)}")
    try:
        if 'X_pca' in adata.obsm:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            cluster_keys = []
            for col in adata.obs.columns:
                if 'cluster' in col.lower() or any(x in col.lower() for x in ['leiden', 'louvain', 'kmeans']):
                    if pd.api.types.is_categorical_dtype(adata.obs[col]):
                        cluster_keys.append(col)
            
            if cluster_keys:
                color = cluster_keys[0]
            else:
                if 'total_counts' in adata.obs:
                    color = 'total_counts'
                else:
                    color = None
            
            sc.pl.pca(adata, color=color, ax=ax, show=False)
            
            figures['pca'] = save_fig(fig, "pca")
    except Exception as e:
        logger.warning(f"Could not generate PCA figure: {str(e)}")
    try:
        cluster_keys = []
        for col in adata.obs.columns:
            if 'cluster' in col.lower() or any(x in col.lower() for x in ['leiden', 'louvain', 'kmeans']):
                if pd.api.types.is_categorical_dtype(adata.obs[col]):
                    cluster_keys.append(col)
        
        if cluster_keys and 'spatial' in adata.obsm:
            fig = plt.figure(figsize=figsize)
            
            for i, key in enumerate(cluster_keys[:4]):  # Show up to 4 clustering results
                ax = fig.add_subplot(2, 2, i+1)
                coords = adata.obsm['spatial']
                cat_codes = adata.obs[key].cat.codes.values
                
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cat_codes, s=5, cmap='tab20')
                ax.set_title(f'Clustering: {key}')
                ax.set_xlabel('Spatial X')
                ax.set_ylabel('Spatial Y')
                ax.set_aspect('equal')
                from matplotlib.lines import Line2D
                categories = adata.obs[key].cat.categories
                custom_lines = [Line2D([0], [0], color=plt.cm.tab20(i/len(categories)),
                              marker='o', linestyle='None', markersize=8)
                           for i in range(min(len(categories), 10))]
                
                ax.legend(custom_lines, categories[:10], title='Clusters', loc='upper right', 
                        bbox_to_anchor=(1.1, 1), fontsize='small')
            
            fig.tight_layout()
            figures['clustering'] = save_fig(fig, "clustering")
    except Exception as e:
        logger.warning(f"Could not generate clustering figure: {str(e)}")
    try:
        if 'moranI' in adata.uns and 'spatial' in adata.obsm:
            top_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:4].tolist()
            
            fig = plt.figure(figsize=figsize)
            
            for i, gene in enumerate(top_genes):
                if gene in adata.var_names:
                    ax = fig.add_subplot(2, 2, i+1)
                    coords = adata.obsm['spatial']
                    gene_idx = adata.var_names.get_loc(gene)
                    if hasattr(adata.X, 'toarray'):
                        expression = adata.X[:, gene_idx].toarray().flatten()
                    else:
                        expression = adata.X[:, gene_idx].flatten()
                    
                    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=expression, s=5, cmap='viridis')
                    ax.set_title(f'Gene Expression: {gene}')
                    ax.set_xlabel('Spatial X')
                    ax.set_ylabel('Spatial Y')
                    ax.set_aspect('equal')
                    
                    plt.colorbar(scatter, ax=ax)
            
            fig.tight_layout()
            figures['spatial_genes'] = save_fig(fig, "spatial_genes")
    except Exception as e:
        logger.warning(f"Could not generate spatial gene expression figure: {str(e)}")
    try:
        deconv_keys = [col for col in adata.obs.columns if any(x in col for x in ['prop_', 'cell_type_', 'deconv_'])]
        
        if deconv_keys and 'spatial' in adata.obsm:
            fig = plt.figure(figsize=figsize)
            
            for i, key in enumerate(deconv_keys[:4]):  # Show up to 4 cell types
                ax = fig.add_subplot(2, 2, i+1)
                coords = adata.obsm['spatial']
                proportions = adata.obs[key].values
                
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=proportions, s=5, cmap='Reds')
                ax.set_title(f'Cell Type: {key}')
                ax.set_xlabel('Spatial X')
                ax.set_ylabel('Spatial Y')
                ax.set_aspect('equal')
                
                plt.colorbar(scatter, ax=ax)
            
            fig.tight_layout()
            figures['cell_types'] = save_fig(fig, "cell_types")
    except Exception as e:
        logger.warning(f"Could not generate cell type composition figure: {str(e)}")
    try:
        ligrec_keys = [key for key in adata.uns.keys() if 'ligrec' in key]
        
        if ligrec_keys:
            key = ligrec_keys[0]  # Use the first ligand-receptor analysis
            
            if isinstance(adata.uns[key], dict) and 'means' in adata.uns[key]:
                means = adata.uns[key]['means']
                
                if isinstance(means, pd.DataFrame):
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                    if means.size > 0:
                        values = means.values.flatten()
                        values = values[~np.isnan(values)]
                        if len(values) > 0:
                            threshold = np.percentile(values, 95)
                            means_filtered = means.copy()
                            means_filtered[means_filtered < threshold] = np.nan
                            import seaborn as sns
                            sns.heatmap(means_filtered, cmap='viridis', ax=ax)
                            ax.set_title('Top Ligand-Receptor Interactions')
                            
                            figures['ligrec'] = save_fig(fig, "ligrec")
                
                elif isinstance(means, dict):
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)
                    interactions = []
                    for k, v in means.items():
                        if isinstance(v, dict):
                            for k2, score in v.items():
                                interactions.append((str(k), k2, score))
                        if len(interactions) >= 20:
                            break
                    
                    if interactions:
                        source, target, score = zip(*interactions)
                        df = pd.DataFrame({
                            'Source': source,
                            'Target': target,
                            'Score': score
                        })
                        df = df.sort_values('Score', ascending=False).head(10)
                        ax.barh(df['Target'], df['Score'], color='skyblue')
                        ax.set_title('Top Ligand-Receptor Interactions')
                        ax.set_xlabel('Interaction Score')
                        
                        figures['ligrec'] = save_fig(fig, "ligrec")
    except Exception as e:
        logger.warning(f"Could not generate ligand-receptor figure: {str(e)}")
    
    return figures





def _ensure_json_serializable(obj):
    """
    Recursively convert data structure to be JSON serializable.
    Converts tuples to strings as keys and in lists.
    """
    if isinstance(obj, dict):
        return {str(k): _ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return str(obj)  # Convert tuples to strings
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def _create_html_report(adata, report_data, title):
    """
    Create HTML report
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    report_data : dict
        Dictionary with report data
    title : str
        Title of the report
        
    Returns
    -------
    str
        HTML report
    """
    try:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 40px; }}
                h1, h2, h3, h4 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .figure {{ margin-bottom: 20px; text-align: center; }}
                .figure img {{ max-width: 100%; height: auto; }}
                .code {{ font-family: monospace; background-color: #f0f0f0; padding: 10px; border-radius: 3px; overflow-x: auto; }}
                .flex-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .flex-item {{ flex: 1; min-width: 300px; }}
                pre {{ white-space: pre-wrap; }}
                .collapsible {{ background-color: #f2f2f2; color: #333; cursor: pointer; padding: 10px; width: 100%; border: none; text-align: left; outline: none; font-size: 16px; }}
                .active, .collapsible:hover {{ background-color: #e0e0e0; }}
                .content {{ padding: 0 18px; display: none; overflow: hidden; background-color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        html += """
        <div class="section">
            <h2>1. Dataset Overview</h2>
            <div class="flex-container">
                <div class="flex-item">
        """
        
        dataset = report_data['dataset']
        html += f"""
                    <table>
                        <tr><th>Spots</th><td>{dataset['n_spots']}</td></tr>
                        <tr><th>Genes</th><td>{dataset['n_genes']}</td></tr>
                        <tr><th>Shape</th><td>{dataset['dataset_shape']}</td></tr>
                    </table>
                </div>
                <div class="flex-item">
                    <button type="button" class="collapsible">Available Metadata</button>
                    <div class="content">
                        <h4>Observation Columns</h4>
                        <pre>{json.dumps(dataset['obs_columns'], indent=2)}</pre>
                        <h4>Variable Columns</h4>
                        <pre>{json.dumps(dataset['var_columns'], indent=2)}</pre>
                        <h4>obsm Keys</h4>
                        <pre>{json.dumps(dataset['obsm_keys'], indent=2)}</pre>
                        <h4>uns Keys</h4>
                        <pre>{json.dumps(dataset['uns_keys'], indent=2)}</pre>
                    </div>
                </div>
            </div>
        </div>
        """
        html += """
        <div class="section">
            <h2>2. Preprocessing and Feature Selection</h2>
        """
        
        if 'qc_metrics' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['qc_metrics']}" alt="QC Metrics">
                <p>Quality control metrics: Distribution of total counts and genes per spot</p>
            </div>
            """
        
        preprocessing = report_data['preprocessing']
        html += f"""
            <div class="flex-container">
                <div class="flex-item">
                    <table>
                        <tr><th>Highly Variable Genes</th><td>{preprocessing.get('n_hvg', 'N/A')}</td></tr>
                        <tr><th>QC Metrics Calculated</th><td>{'Yes' if preprocessing.get('qc_metrics', False) else 'No'}</td></tr>
                    </table>
                </div>
        """
        
        if 'pca_var_explained' in preprocessing:
            var_explained = preprocessing['pca_var_explained']
            html += f"""
                <div class="flex-item">
                    <h4>PCA Variance Explained</h4>
                    <div class="figure">
                        <img src="data:image/png;base64,{_plot_variance_explained_to_base64(var_explained)}" alt="PCA Variance Explained">
                    </div>
                </div>
            """
        
        html += """
            </div>
        """
        
        if 'pca' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['pca']}" alt="PCA Plot">
                <p>PCA visualization of the data</p>
            </div>
            """
        
        html += """
        </div>
        """
        html += """
        <div class="section">
            <h2>3. Spatial Analysis</h2>
        """
        
        if 'spatial_coords' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['spatial_coords']}" alt="Spatial Coordinates">
                <p>Spatial coordinates of spots</p>
            </div>
            """
        
        if 'spatial_genes' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['spatial_genes']}" alt="Spatial Gene Expression">
                <p>Spatial expression patterns of top spatially variable genes</p>
            </div>
            """
        
        spatial = report_data['spatial']
        if 'top_spatial_genes' in spatial:
            html += """
            <button type="button" class="collapsible">Top Spatially Variable Genes</button>
            <div class="content">
                <table>
                    <tr><th>Gene</th><th>Moran's I</th></tr>
            """
            
            for gene, values in spatial['top_spatial_genes'].items():
                i_value = values.get('I', 'N/A')
                html += f"<tr><td>{gene}</td><td>{i_value}</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </div>
        """
        html += """
        <div class="section">
            <h2>4. Clustering and Domain Identification</h2>
        """
        
        if 'clustering' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['clustering']}" alt="Clustering Results">
                <p>Visualization of clustering results</p>
            </div>
            """
        
        clustering = report_data['clustering']
        if 'cluster_keys' in clustering and clustering['cluster_keys']:
            html += """
            <div class="flex-container">
                <div class="flex-item">
                    <h4>Available Clusterings</h4>
                    <table>
                        <tr><th>Clustering Method</th><th>Number of Clusters</th></tr>
            """
            
            for key in clustering['cluster_keys']:
                n_clusters = clustering['n_clusters'].get(key, 'N/A')
                html += f"<tr><td>{key}</td><td>{n_clusters}</td></tr>"
            
            html += """
                    </table>
                </div>
            </div>
            """
        
        html += """
        </div>
        """
        html += """
        <div class="section">
            <h2>5. Cell Type Analysis and Deconvolution</h2>
        """
        
        if 'cell_types' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['cell_types']}" alt="Cell Type Composition">
                <p>Spatial distribution of cell types</p>
            </div>
            """
        
        deconvolution = report_data['deconvolution']
        if 'deconv_keys' in deconvolution and deconvolution['deconv_keys']:
            html += """
            <div class="flex-container">
                <div class="flex-item">
                    <h4>Detected Cell Types</h4>
                    <ul>
            """
            
            for key in deconvolution['deconv_keys']:
                html += f"<li>{key}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
            """
        else:
            html += "<p>No cell type deconvolution results available.</p>"
        
        html += """
        </div>
        """
        html += """
        <div class="section">
            <h2>6. Ligand-Receptor Interaction Analysis</h2>
        """
        
        if 'ligrec' in report_data['figures']:
            html += f"""
            <div class="figure">
                <img src="{report_data['figures']['ligrec']}" alt="Ligand-Receptor Interactions">
                <p>Visualization of ligand-receptor interactions</p>
            </div>
            """
        
        ligrec = report_data['ligrec']
        if 'ligrec_keys' in ligrec and ligrec['ligrec_keys']:
            html += f"""
            <p>Ligand-receptor analysis results available: {', '.join(ligrec['ligrec_keys'])}</p>
            """
            
            if 'sample_interactions' in ligrec:
                html += """
                <button type="button" class="collapsible">Sample Interactions</button>
                <div class="content">
                    <pre>{}</pre>
                </div>
                """.format(json.dumps(_ensure_json_serializable(ligrec['sample_interactions']), indent=2))
        else:
            html += "<p>No ligand-receptor analysis results available.</p>"
        
        html += """
        </div>
        """
        html += """
        <div class="section">
            <h2>7. Pathway Analysis</h2>
        """
        try:
            pathway_data = _get_pathway_data(adata)
            
            if pathway_data and pathway_data['has_results']:
                if 'pathway_keys' in pathway_data:
                    html += """
                    <div class="flex-container">
                        <div class="flex-item">
                            <h4>Pathway Scores</h4>
                            <ul>
                    """
                    
                    for key in pathway_data['pathway_keys']:
                        html += f"<li>{key.replace('pathway_', '')}</li>"
                    
                    html += """
                            </ul>
                        </div>
                    </div>
                    """
                if 'gsea_results' in pathway_data and pathway_data['gsea_results']:
                    html += """
                    <h4>GSEA Results</h4>
                    """
                    
                    for comparison in pathway_data['gsea_results']:
                        html += f"""
                        <button type="button" class="collapsible">{comparison['name']}</button>
                        <div class="content">
                            <table>
                                <tr><th>Pathway</th><th>NES</th><th>FDR</th></tr>
                        """
                        
                        for pathway in sorted(comparison['pathways'], key=lambda x: x.get('fdr', 1)):
                            html += f"""
                            <tr>
                                <td>{pathway['name']}</td>
                                <td>{pathway['nes']}</td>
                                <td>{pathway['fdr']}</td>
                            </tr>
                            """
                        
                        html += """
                            </table>
                        </div>
                        """
            else:
                html += "<p>No pathway analysis results available.</p>"
                
        except Exception as e:
            html += f"<p>Error processing pathway analysis results: {str(e)}</p>"
        
        html += """
        </div>
        """
        html += """
            </div>
            <script>
                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {
                    coll[i].addEventListener("click", function() {
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.display === "block") {
                            content.style.display = "none";
                        } else {
                            content.style.display = "block";
                        }
                    });
                }
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Generating Report</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Error Generating Analysis Report</h1>
            <p>An error occurred while generating the report: {str(e)}</p>
            <p>The analysis completed successfully and results were saved, but the HTML report could not be generated.</p>
        </body>
        </html>
        """

def _plot_variance_explained_to_base64(var_explained):
    """
    Create a base64 encoded plot of variance explained
    
    Parameters
    ----------
    var_explained : list
        List of cumulative explained variance values
        
    Returns
    -------
    str
        Base64 encoded PNG image
    """
    import io
    import base64
    
    fig, ax = plt.figure(figsize=(6, 4)), plt.gca()
    ax.plot(range(1, len(var_explained) + 1), var_explained, 'o-')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Explained Variance')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.8, color='r', linestyle='--')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def generate_pdf_report(adata, output_dir, title="Spatial Transcriptomics Analysis Report", **kwargs):
    """
    Generate a PDF report of the analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    output_dir : str or Path
        Directory to save the report
    title : str, optional
        Title of the report
    **kwargs
        Additional arguments for report generation
        
    Returns
    -------
    Path
        Path to the generated report
    """
    try:
        from weasyprint import HTML
        import tempfile
        
        logger.info("Generating PDF report")
        html_path = generate_report(adata, output_dir, title, **kwargs)
        output_dir = Path(output_dir)
        pdf_path = output_dir / "spatialflow_report.pdf"
        html = HTML(filename=html_path)
        html.write_pdf(pdf_path)
        
        logger.info(f"PDF report generated successfully: {pdf_path}")
        
        return pdf_path
    
    except ImportError:
        logger.error("WeasyPrint not installed. Please install with 'pip install weasyprint' for PDF generation")
        raise ImportError("WeasyPrint required for PDF report generation")

def generate_notebook_report(adata, output_path, title="Spatial Transcriptomics Analysis", **kwargs):
    """
    Generate a Jupyter notebook with the analysis
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with analysis results
    output_path : str or Path
        Path to save the notebook
    title : str, optional
        Title of the notebook
    **kwargs
        Additional arguments for notebook generation
        
    Returns
    -------
    Path
        Path to the generated notebook
    """
    try:
        import nbformat as nbf
        
        logger.info("Generating Jupyter notebook")
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        nb.cells.append(nbf.v4.new_code_cell(
            "import scanpy as sc\n"
            "import squidpy as sq\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "# Set plotting parameters\n"
            "sc.settings.set_figure_params(dpi=100, frameon=False)\n"
            "plt.rcParams['figure.figsize'] = [8, 6]"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Loading Data"))
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        if output_path.is_dir():
            output_path = output_path / "spatialflow_analysis.ipynb"
        adata_path = output_path.with_name("adata.h5ad")
        
        nb.cells.append(nbf.v4.new_code_cell(
            f"# Load the AnnData object\n"
            f"adata = sc.read_h5ad('{adata_path.name}')"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Dataset Overview"))
        nb.cells.append(nbf.v4.new_code_cell(
            "print(f'Dataset shape: {adata.shape}')\n"
            "print(f'Number of spots: {adata.n_obs}')\n"
            "print(f'Number of genes: {adata.n_vars}')\n\n"
            "print('\\nAvailable observation annotations:')\n"
            "print(adata.obs.columns.tolist())\n\n"
            "print('\\nAvailable obsm keys:')\n"
            "print(list(adata.obsm.keys()))\n\n"
            "print('\\nAvailable uns keys:')\n"
            "print(list(adata.uns.keys()))"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Quality Control and Preprocessing"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# QC metrics\n"
            "if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:\n"
            "    fig, axs = plt.subplots(1, 2, figsize=(15, 5))\n"
            "    sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axs[0], show=False)\n"
            "    sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axs[1], show=False)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "# Gene-spot relationship\n"
            "if 'total_counts' in adata.obs and 'n_genes_by_counts' in adata.obs:\n"
            "    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='total_counts')"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Spatial Analysis"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# Spatial coordinates\n"
            "if 'spatial' in adata.obsm:\n"
            "    sq.pl.spatial_scatter(adata, color=['total_counts', 'n_genes_by_counts'], ncols=2)\n\n"
            "# Spatial neighbors\n"
            "if 'spatial_neighbors' not in adata.uns:\n"
            "    print('Building spatial neighbors graph...')\n"
            "    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)\n\n"
            "# Spatial autocorrelation\n"
            "if 'moranI' in adata.uns:\n"
            "    print('Top genes by spatial autocorrelation (Moran\\'s I):')\n"
            "    display(adata.uns['moranI'].sort_values('I', ascending=False).head(10))\n"
            "    \n"
            "    # Plot top spatially variable genes\n"
            "    top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:6].tolist()\n"
            "    sq.pl.spatial_scatter(adata, color=top_spatial_genes, ncols=3)"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Clustering and Domain Identification"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# Check available clustering results\n"
            "cluster_keys = [col for col in adata.obs.columns \n"
            "               if 'cluster' in col.lower() or \n"
            "               any(x in col.lower() for x in ['leiden', 'louvain', 'kmeans'])]\n\n"
            "print(f'Available clustering results: {cluster_keys}')\n\n"
            "# Plot clustering results\n"
            "if cluster_keys:\n"
            "    sq.pl.spatial_scatter(adata, color=cluster_keys[:4], ncols=2)"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Cell Type Analysis"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# Check available cell type information\n"
            "cell_type_keys = [col for col in adata.obs.columns \n"
            "                 if any(x in col for x in ['prop_', 'cell_type_', 'deconv_'])]\n\n"
            "print(f'Available cell type information: {cell_type_keys}')\n\n"
            "# Plot cell type distribution\n"
            "if cell_type_keys:\n"
            "    sq.pl.spatial_scatter(adata, color=cell_type_keys[:6], ncols=3)"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Ligand-Receptor Analysis"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# Check available ligand-receptor analysis results\n"
            "ligrec_keys = [key for key in adata.uns.keys() if 'ligrec' in key]\n\n"
            "print(f'Available ligand-receptor analysis results: {ligrec_keys}')\n\n"
            "# Visualize ligand-receptor interactions\n"
            "if ligrec_keys and 'means' in adata.uns[ligrec_keys[0]]:\n"
            "    means = adata.uns[ligrec_keys[0]]['means']\n"
            "    \n"
            "    if isinstance(means, pd.DataFrame):\n"
            "        # Plot heatmap of interactions\n"
            "        plt.figure(figsize=(10, 8))\n"
            "        # Filter to show only strong interactions\n"
            "        values = means.values.flatten()\n"
            "        values = values[~np.isnan(values)]\n"
            "        if len(values) > 0:\n"
            "            threshold = np.percentile(values, 95)\n"
            "            means_filtered = means.copy()\n"
            "            means_filtered[means_filtered < threshold] = np.nan\n"
            "            \n"
            "            # Plot heatmap\n"
            "            sns.heatmap(means_filtered, cmap='viridis')\n"
            "            plt.title('Top Ligand-Receptor Interactions')\n"
            "            plt.show()"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Pathway Analysis"))
        nb.cells.append(nbf.v4.new_code_cell(
            "# Check available pathway scores\n"
            "pathway_keys = [col for col in adata.obs.columns if 'pathway_' in col]\n\n"
            "print(f'Available pathway scores: {pathway_keys}')\n\n"
            "# Plot pathway scores\n"
            "if pathway_keys:\n"
            "    sq.pl.spatial_scatter(adata, color=pathway_keys[:6], ncols=3)"
        ))
        nb.cells.append(nbf.v4.new_markdown_cell("## Custom Analysis\n\nAdd your custom analysis code below:"))
        nb.cells.append(nbf.v4.new_code_cell("# Your custom analysis code here"))
        with open(output_path, 'w') as f:
            nbf.write(nb, f)
        
        logger.info(f"Jupyter notebook generated successfully: {output_path}")
        adata.write(adata_path)
        logger.info(f"AnnData object saved to: {adata_path}")
        
        return output_path
    
    except ImportError:
        logger.error("nbformat not installed. Please install with 'pip install nbformat' for notebook generation")
        raise ImportError("nbformat required for notebook generation")