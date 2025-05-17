import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger('spatialflow.visualization.interactive')

def generate_interactive_figures(adata, output_dir, color_by=None, library_id=None, **kwargs):
    """
    Generate interactive figures for spatial data visualization
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    output_dir : str or Path
        Directory to save figures
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
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Plotly is required for interactive visualization. Install it with 'pip install plotly'")
        raise ImportError("Plotly is required for interactive visualization")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = {}
    if color_by is None:
        color_by = _get_default_color_variables(adata)
    
    logger.info(f"Generating interactive figures, coloring by: {color_by}")
    if 'spatial' in adata.obsm:
        logger.info("Generating spatial plots")
        plot_df = pd.DataFrame({
            'x': adata.obsm['spatial'][:, 0],
            'y': adata.obsm['spatial'][:, 1],
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                plot_df[col] = adata.obs[col].values
        gene_list = [gene for gene in color_by if gene in adata.var_names]
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            plot_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        if color_by:
            fig = _create_interactive_spatial_plot(plot_df, color_by)
            fig_path = output_dir / "spatial_interactive.html"
            fig.write_html(fig_path)
            figure_paths["spatial_interactive"] = fig_path
            logger.info(f"Saved interactive spatial plot to {fig_path}")
        dashboard_fig = _create_spatial_dashboard(plot_df, color_by)
        dash_path = output_dir / "spatial_dashboard.html"
        dashboard_fig.write_html(dash_path)
        figure_paths["spatial_dashboard"] = dash_path
        logger.info(f"Saved spatial dashboard to {dash_path}")
    if 'X_umap' in adata.obsm:
        logger.info("Generating UMAP plots")
        umap_df = pd.DataFrame({
            'UMAP1': adata.obsm['X_umap'][:, 0],
            'UMAP2': adata.obsm['X_umap'][:, 1],
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                umap_df[col] = adata.obs[col].values
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            umap_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        if color_by:
            umap_fig = _create_interactive_dimension_plot(umap_df, color_by, dimension_reduction='UMAP')
            umap_path = output_dir / "umap_interactive.html"
            umap_fig.write_html(umap_path)
            figure_paths["umap_interactive"] = umap_path
            logger.info(f"Saved interactive UMAP plot to {umap_path}")
    if 'X_pca' in adata.obsm:
        logger.info("Generating PCA plots")
        pca_df = pd.DataFrame({
            'PC1': adata.obsm['X_pca'][:, 0],
            'PC2': adata.obsm['X_pca'][:, 1],
            'PC3': adata.obsm['X_pca'][:, 2] if adata.obsm['X_pca'].shape[1] > 2 else np.zeros(adata.n_obs),
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                pca_df[col] = adata.obs[col].values
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            pca_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        if color_by:
            pca_fig = _create_interactive_dimension_plot(pca_df, color_by, dimension_reduction='PCA')
            pca_path = output_dir / "pca_interactive.html"
            pca_fig.write_html(pca_path)
            figure_paths["pca_interactive"] = pca_path
            logger.info(f"Saved interactive PCA plot to {pca_path}")
            pca3d_fig = _create_interactive_3d_plot(pca_df, color_by[0] if color_by else None, 
                                                   x='PC1', y='PC2', z='PC3', title='3D PCA Plot')
            pca3d_path = output_dir / "pca_3d_interactive.html"
            pca3d_fig.write_html(pca3d_path)
            figure_paths["pca_3d_interactive"] = pca3d_path
            logger.info(f"Saved interactive 3D PCA plot to {pca3d_path}")
    if gene_list and len(gene_list) >= 2:
        logger.info("Generating gene correlation heatmap")
        gene_expr = np.zeros((adata.n_obs, len(gene_list)))
        for i, gene in enumerate(gene_list):
            gene_idx = adata.var_names.get_loc(gene)
            gene_expr[:, i] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        
        corr_matrix = np.corrcoef(gene_expr.T)
        heatmap_fig = px.imshow(
            corr_matrix,
            x=gene_list,
            y=gene_list,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Gene-Gene Correlation Heatmap"
        )
        
        heatmap_fig.update_layout(
            width=900,
            height=900,
            title_font_size=20,
            title_x=0.5
        )
        heatmap_path = output_dir / "gene_correlation_heatmap.html"
        heatmap_fig.write_html(heatmap_path)
        figure_paths["gene_correlation_heatmap"] = heatmap_path
        logger.info(f"Saved gene correlation heatmap to {heatmap_path}")
    cell_type_cols = [col for col in adata.obs.columns if col.startswith('prop_')]
    if cell_type_cols:
        logger.info("Generating cell type proportion plot")
        cell_props = adata.obs[cell_type_cols].copy()
        cell_props.columns = [col[5:] for col in cell_type_cols]  # Remove 'prop_' prefix
        mean_props = cell_props.mean().sort_values(ascending=False)
        top_cell_types = mean_props.index.tolist()
        melted_df = cell_props[top_cell_types].reset_index().melt(
            id_vars=['index'],
            value_vars=top_cell_types,
            var_name='Cell Type',
            value_name='Proportion'
        )
        cell_type_fig = px.violin(
            melted_df,
            x='Cell Type',
            y='Proportion',
            color='Cell Type',
            box=True,
            points='all',
            title="Cell Type Proportions"
        )
        
        cell_type_fig.update_layout(
            width=1000,
            height=700,
            title_font_size=20,
            title_x=0.5
        )
        cell_type_path = output_dir / "cell_type_proportions.html"
        cell_type_fig.write_html(cell_type_path)
        figure_paths["cell_type_proportions"] = cell_type_path
        logger.info(f"Saved cell type proportion plot to {cell_type_path}")
    
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

def _create_interactive_spatial_plot(df, color_variables, spot_size=15):
    """
    Create an interactive spatial plot with dropdown for color variable
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spatial coordinates and color variables
    color_variables : list
        List of variables to color by
    spot_size : int, optional
        Size of spots in the plot
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure
    """
    initial_color = color_variables[0] if color_variables else None
    dropdown_buttons = []
    
    categorical_variables = []
    for var in color_variables:
        if var in df.columns and pd.api.types.is_categorical_dtype(df[var]):
            categorical_variables.append(var)
    if initial_color in categorical_variables:
        fig = px.scatter(
            df, x='x', y='y',
            color=initial_color,
            hover_name='spot_id',
            hover_data={var: True for var in color_variables if var in df},
            title=f"Spatial Gene Expression - {initial_color}",
            width=1000, height=800,
            category_orders={initial_color: sorted(df[initial_color].unique())}
        )
    else:
        fig = px.scatter(
            df, x='x', y='y',
            color=initial_color,
            color_continuous_scale='Viridis',
            hover_name='spot_id',
            hover_data={var: True for var in color_variables if var in df},
            title=f"Spatial Gene Expression - {initial_color}",
            width=1000, height=800
        )
    fig.update_traces(marker=dict(size=spot_size, opacity=0.85))
    for var in color_variables:
        if var in df.columns:
            if var in categorical_variables:
                dropdown_buttons.append(
                    dict(
                        args=[{"marker.color": df[var], 
                               "marker.colorscale": None,
                               "coloraxis.colorscale": None,
                               "title": f"Spatial Gene Expression - {var}"}],
                        label=var,
                        method="update"
                    )
                )
            else:
                dropdown_buttons.append(
                    dict(
                        args=[{"marker.color": df[var], 
                               "marker.colorscale": "Viridis",
                               "coloraxis.colorscale": "Viridis",
                               "title": f"Spatial Gene Expression - {var}"}],
                        label=var,
                        method="update"
                    )
                )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        title_text=f"Spatial Gene Expression - {initial_color}",
        title_x=0.5,
        title_font_size=20,
        template="plotly_white"
    )
    fig.add_annotation(
        dict(
            text="Color by:",
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
    )
    
    return fig

def _create_interactive_dimension_plot(df, color_variables, dimension_reduction='UMAP'):
    """
    Create an interactive dimension reduction plot with dropdown for color variable
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dimension reduction coordinates and color variables
    color_variables : list
        List of variables to color by
    dimension_reduction : str, optional
        Type of dimension reduction ('UMAP' or 'PCA')
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure
    """
    initial_color = color_variables[0] if color_variables else None
    if dimension_reduction == 'UMAP':
        x_col, y_col = 'UMAP1', 'UMAP2'
    elif dimension_reduction == 'PCA':
        x_col, y_col = 'PC1', 'PC2'
    else:
        raise ValueError(f"Unknown dimension reduction: {dimension_reduction}")
    dropdown_buttons = []
    
    categorical_variables = []
    for var in color_variables:
        if var in df.columns and pd.api.types.is_categorical_dtype(df[var]):
            categorical_variables.append(var)
    if initial_color in categorical_variables:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=initial_color,
            hover_name='spot_id',
            hover_data={var: True for var in color_variables if var in df},
            title=f"{dimension_reduction} - {initial_color}",
            width=1000, height=800,
            category_orders={initial_color: sorted(df[initial_color].unique())}
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=initial_color,
            color_continuous_scale='Viridis',
            hover_name='spot_id',
            hover_data={var: True for var in color_variables if var in df},
            title=f"{dimension_reduction} - {initial_color}",
            width=1000, height=800
        )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    for var in color_variables:
        if var in df.columns:
            if var in categorical_variables:
                dropdown_buttons.append(
                    dict(
                        args=[{"marker.color": df[var], 
                               "marker.colorscale": None,
                               "coloraxis.colorscale": None,
                               "title": f"{dimension_reduction} - {var}"}],
                        label=var,
                        method="update"
                    )
                )
            else:
                dropdown_buttons.append(
                    dict(
                        args=[{"marker.color": df[var], 
                               "marker.colorscale": "Viridis",
                               "coloraxis.colorscale": "Viridis",
                               "title": f"{dimension_reduction} - {var}"}],
                        label=var,
                        method="update"
                    )
                )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        title_text=f"{dimension_reduction} - {initial_color}",
        title_x=0.5,
        title_font_size=20,
        template="plotly_white"
    )
    fig.add_annotation(
        dict(
            text="Color by:",
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
    )
    
    return fig

def _create_interactive_3d_plot(df, color_var, x='PC1', y='PC2', z='PC3', title='3D Plot'):
    """
    Create an interactive 3D plot
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 3D coordinates and color variable
    color_var : str
        Variable to color by
    x, y, z : str, optional
        Column names for x, y, z coordinates
    title : str, optional
        Plot title
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure
    """
    if color_var in df.columns and pd.api.types.is_categorical_dtype(df[color_var]):
        fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color=color_var,
            hover_name='spot_id',
            title=title,
            width=1000, height=800,
            category_orders={color_var: sorted(df[color_var].unique())}
        )
    else:
        fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color=color_var,
            color_continuous_scale='Viridis',
            hover_name='spot_id',
            title=title,
            width=1000, height=800
        )
    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font_size=20,
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z
        ),
        template="plotly_white"
    )
    
    return fig

def _create_spatial_dashboard(df, color_variables):
    """
    Create a multi-view dashboard for spatial data
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spatial coordinates and color variables
    color_variables : list
        List of variables to color by
    
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure
    """
    cluster_vars = [var for var in color_variables if 
                   any(var.startswith(prefix) for prefix in 
                      ['leiden', 'louvain', 'kmeans', 'spectral', 'spatial_domains'])
                   and var in df.columns]
    
    qc_vars = [var for var in ['total_counts', 'n_genes_by_counts'] 
              if var in df.columns]
    
    cell_type_vars = [var for var in color_variables if 
                     any(var.startswith(prefix) for prefix in
                        ['cell_type', 'celltype', 'deconv', 'prop_'])
                     and var in df.columns]
    
    gene_vars = [var for var in color_variables if 
                var not in cluster_vars + qc_vars + cell_type_vars
                and var in df.columns]
    cluster_var = cluster_vars[0] if cluster_vars else None
    qc_var = qc_vars[0] if qc_vars else None
    cell_type_var = cell_type_vars[0] if cell_type_vars else None
    gene_var1 = gene_vars[0] if len(gene_vars) > 0 else None
    gene_var2 = gene_vars[1] if len(gene_vars) > 1 else None
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Spatial Domains ({cluster_var})" if cluster_var else "Spatial Clusters",
            f"Quality Control ({qc_var})" if qc_var else "Quality Control",
            f"Cell Types ({cell_type_var})" if cell_type_var else "Cell Type Distribution",
            f"Gene Expression ({gene_var1})" if gene_var1 else "Gene Expression"
        )
    )
    if cluster_var:
        if pd.api.types.is_categorical_dtype(df[cluster_var]):
            colors = df[cluster_var].astype('category').cat.codes
        else:
            colors = df[cluster_var]
        
        fig.add_trace(
            go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale='Viridis' if not pd.api.types.is_categorical_dtype(df[cluster_var]) else None,
                    showscale=True,
                    colorbar=dict(title=cluster_var, x=0.45)
                ),
                text=df['spot_id'],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
    if qc_var:
        fig.add_trace(
            go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df[qc_var],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=qc_var, x=1.0)
                ),
                text=df['spot_id'],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=2
        )
    if cell_type_var:
        fig.add_trace(
            go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df[cell_type_var],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=cell_type_var, x=0.45, y=0.25)
                ),
                text=df['spot_id'],
                hoverinfo='text',
                showlegend=False
            ),
            row=2, col=1
        )
    if gene_var1:
        fig.add_trace(
            go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df[gene_var1],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=gene_var1, x=1.0, y=0.25)
                ),
                text=df['spot_id'],
                hoverinfo='text',
                showlegend=False
            ),
            row=2, col=2
        )
    fig.update_layout(
        title_text="Spatial Transcriptomics Interactive Dashboard",
        height=1000,
        width=1200,
        template="plotly_white"
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    
    return fig

def create_interactive_report(adata, output_path, color_by=None, include_dashboard=True, include_gene_expr=True):
    """
    Create an interactive HTML report with multiple visualizations
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object
    output_path : str or Path
        Path to save the HTML report
    color_by : list, optional
        List of variables to color by
    include_dashboard : bool, optional
        Whether to include the dashboard view
    include_gene_expr : bool, optional
        Whether to include gene expression visualization
    
    Returns
    -------
    str
        Path to the saved HTML report
    """
    from pathlib import Path
    output_path = Path(output_path)
    if color_by is None:
        color_by = _get_default_color_variables(adata)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spatial Transcriptomics Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .plot-container {{
                margin-bottom: 30px;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
            }}
            .tab {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
            }}
            .tab button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 12px 16px;
                transition: 0.3s;
                font-size: 16px;
            }}
            .tab button:hover {{
                background-color: #ddd;
            }}
            .tab button.active {{
                background-color: #ccc;
            }}
            .tabcontent {{
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Spatial Transcriptomics Analysis Report</h1>
            <p>Dataset shape: {adata.shape[0]} spots, {adata.shape[1]} genes</p>
            
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview</button>
                <button class="tablinks" onclick="openTab(event, 'SpatialAnalysis')">Spatial Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'CellTypes')">Cell Types</button>
                <button class="tablinks" onclick="openTab(event, 'GeneExpression')">Gene Expression</button>
            </div>
            
            <div id="Overview" class="tabcontent">
                <h2>Dataset Overview</h2>
                <div id="overview_plot" class="plot-container"></div>
                <p>This section shows basic properties of the dataset, including quality control metrics,
                clustering results, and dimension reduction visualizations.</p>
            </div>
            
            <div id="SpatialAnalysis" class="tabcontent">
                <h2>Spatial Analysis</h2>
                <div id="spatial_plot" class="plot-container"></div>
                <p>This section displays spatial patterns and organization in the tissue.</p>
            </div>
            
            <div id="CellTypes" class="tabcontent">
                <h2>Cell Type Analysis</h2>
                <div id="celltype_plot" class="plot-container"></div>
                <p>This section shows cell type distributions and spatial organization.</p>
            </div>
            
            <div id="GeneExpression" class="tabcontent">
                <h2>Gene Expression Analysis</h2>
                <div id="gene_plot" class="plot-container"></div>
                <p>This section visualizes expression patterns of genes of interest.</p>
            </div>
            
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                // Get the element with id="defaultOpen" and click on it
                document.getElementById("defaultOpen").click();
            </script>
    """
    plot_data = {}
    if 'spatial' in adata.obsm:
        spatial_df = pd.DataFrame({
            'x': adata.obsm['spatial'][:, 0],
            'y': adata.obsm['spatial'][:, 1],
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                spatial_df[col] = adata.obs[col].values
        gene_list = [gene for gene in color_by if gene in adata.var_names]
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            spatial_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        
        plot_data['spatial'] = spatial_df
    if 'X_umap' in adata.obsm:
        umap_df = pd.DataFrame({
            'UMAP1': adata.obsm['X_umap'][:, 0],
            'UMAP2': adata.obsm['X_umap'][:, 1],
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                umap_df[col] = adata.obs[col].values
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            umap_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        
        plot_data['umap'] = umap_df
    if 'X_pca' in adata.obsm:
        pca_df = pd.DataFrame({
            'PC1': adata.obsm['X_pca'][:, 0],
            'PC2': adata.obsm['X_pca'][:, 1],
            'spot_id': adata.obs.index,
        })
        for col in color_by:
            if col in adata.obs:
                pca_df[col] = adata.obs[col].values
        for gene in gene_list:
            gene_idx = adata.var_names.get_loc(gene)
            pca_df[gene] = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx]
        
        plot_data['pca'] = pca_df
    html_content += """
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Saved interactive report to {output_path}")
    
    return str(output_path)
