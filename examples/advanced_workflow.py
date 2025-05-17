"""
Advanced Spatial Transcriptomics Analysis Workflow

This is an advanced workflow for analyzing spatial transcriptomics
data using the SpatialFlow package, including advanced spatial statistics,
topological data analysis, and Bayesian modeling.
"""

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path

from spatialflow.utils.logging import setup_logging, log_execution_time, log_system_info
from spatialflow.utils.parallel import parallelize
from spatialflow.core.data_loader import load_data
from spatialflow.core.preprocessing import run_preprocessing
from spatialflow.core.feature_selection import run_feature_selection, run_dimension_reduction
from spatialflow.spatial.neighbors import build_spatial_graph
from spatialflow.spatial.statistics import calculate_spatial_stats
from spatialflow.spatial.clustering import run_spatial_clustering
from spatialflow.spatial.trajectories import run_spatial_trajectory_analysis
from spatialflow.analysis.deconvolution import run_deconvolution
from spatialflow.analysis.ligrec import run_ligrec_analysis
from spatialflow.analysis.pathways import run_pathway_analysis
from spatialflow.analysis.integration import run_integration
from spatialflow.advanced.bayesian import run_bayesian_analysis
from spatialflow.advanced.gwr import run_gwr_analysis
from spatialflow.advanced.topology import run_tda_analysis
from spatialflow.visualization.static import generate_static_figures
from spatialflow.visualization.interactive import generate_interactive_figures
from spatialflow.visualization.reporting import generate_report
from spatialflow.utils.io import save_results, backup_data

def main():
    # Set up logging
    logger = setup_logging(level="INFO", log_file="spatialflow_advanced_workflow.log")
    logger.info("Starting advanced spatial transcriptomics analysis workflow")
    
    # Log system information
    log_system_info(logger)
    
    # Create output directories
    output_dir = Path("spatialflow_advanced_output")
    output_dir.mkdir(exist_ok=True)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    advanced_dir = output_dir / "advanced_analysis"
    advanced_dir.mkdir(exist_ok=True)

    end_timer = log_execution_time(logger)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    # You can use a local file or a built-in dataset
    adata = sq.datasets.visium("V1_Human_Lymph_Node")
    # load from a file:
    # adata = load_data("path/<path>/data.h5ad", format='h5ad')
    
    logger.info(f"Loaded dataset with {adata.n_obs} spots and {adata.n_vars} genes")
    
    # backup of the raw data
    backup_data(adata, data_dir, backup_name="raw_data_backup.h5ad")
    
    # Step 2: Preprocessing
    logger.info("Step 2: Preprocessing")
    preprocessing_timer = log_execution_time(logger)
    
    adata = run_preprocessing(
        adata,
        min_genes=200,
        min_cells=3,
        target_sum=1e4,
        log_transform=True,
        calculate_qc=True,
        highly_variable_genes=True
    )
    
    preprocessing_timer("Preprocessing completed")
    
    # Step 3: Feature selection
    logger.info("Step 3: Feature selection and dimensionality reduction")
    feature_timer = log_execution_time(logger)
    
    adata = run_feature_selection(
        adata,
        n_top_genes=2000,
        flavor='seurat',
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        span=0.3
    )
    
    # Advanced feature selection with spatial awareness
    adata = run_feature_selection(
        adata,
        n_top_genes=2000,
        use_moran=True,
        spatial_key='spatial_neighbors'
    )
    
    # multiple dimensionality reduction
    adata = run_dimension_reduction(adata, method='pca', n_comps=30, use_highly_variable=True)
    adata = run_dimension_reduction(adata, method='nmf', n_comps=20, use_highly_variable=True)
    
    feature_timer("Feature selection and dimensionality reduction completed")
    
    # Step 4: Spatial analysis
    logger.info("Step 4: Spatial analysis")
    spatial_timer = log_execution_time(logger)
    
    # spatial graph
    adata = build_spatial_graph(
        adata,
        n_neighs=6,
        coord_type="generic"
    )
    
    # calculation of multiple spatial statistics
    adata = calculate_spatial_stats(
        adata,
        methods=['moran', 'geary', 'ripley'],
        n_perms=100
    )
    
    spatial_timer("Spatial analysis completed")
    
    # Step 5: Clustering
    logger.info("Step 5: Clustering")
    clustering_timer = log_execution_time(logger)
    
    # Multiple clustering methods
    adata = run_spatial_clustering(
        adata,
        method='leiden',
        resolution=0.5,
        neighbors_key='spatial_connectivities',
        use_pca=True,
        use_spatial=True,
        key_added='leiden_spatial'
    )

    adata = run_spatial_clustering(
        adata,
        method='leiden',
        resolution=0.8,
        neighbors_key='spatial_connectivities',
        use_pca=True,
        use_spatial=False,
        key_added='leiden_expression'
    )

    adata = run_spatial_clustering(
        adata,
        method='kmeans',
        n_clusters=8,
        use_pca=True,
        key_added='kmeans_clusters'
    )
    
    clustering_timer("Clustering completed")
    
    # Step 6: Trajectory inference
    logger.info("Step 6: Trajectory inference")
    trajectory_timer = log_execution_time(logger)
    
    adata = run_spatial_trajectory_analysis(
        adata,
        method="shortest_path",
        cluster_key='leiden_spatial'
    )

    logger.info("Trajectory inference implemented")
    trajectory_timer("Trajectory inference completed")
    
    # Step 7: Cell type deconvolution
    logger.info("Step 7: Cell type deconvolution")
    deconv_timer = log_execution_time(logger)
    
    # define marker genes for cell types
    markers = {
        'B_cells': ['MS4A1', 'CD79A', 'CD79B', 'BANK1'],
        'T_cells': ['CD3D', 'CD3E', 'CD8A', 'CD4', 'IL7R'],
        'Plasma_cells': ['MZB1', 'JCHAIN', 'IGHA1', 'IGHG1'],
        'Macrophages': ['CD68', 'CD163', 'MARCO'],
        'Dendritic_cells': ['FCER1A', 'CD1C', 'CLEC10A'],
        'Endothelial_cells': ['PECAM1', 'VWF', 'CDH5'],
        'Fibroblasts': ['DCN', 'COL1A1', 'COL3A1']
    }
    
    adata = run_deconvolution(
        adata,
        method='manual_markers',
        marker_genes=markers
    )
    
    # NMF-based deconvolution
    adata = run_deconvolution(
        adata,
        method='simple_nmf',
        n_cell_types=7,
        key_added='nmf_deconv'
    )
    
    deconv_timer("Cell type deconvolution completed")
    
    # Step 8: Ligand-receptor analysis
    logger.info("Step 8: Ligand-receptor analysis")
    ligrec_timer = log_execution_time(logger)
    

    # check available cluster keys
    print("Available cluster keys:", [col for col in adata.obs.columns if 'cluster' in col or 'leiden' in col or 'louvain' in col])


    #adata = run_ligrec_analysis(
    #    adata,
    #    cluster_key='spatial_leiden',
    #    n_perms=100
    #)
    
    ligrec_timer("Ligand-receptor analysis skipped")
    
    # Step 9: Pathway analysis
    logger.info("Step 9: Pathway analysis")
    pathway_timer = log_execution_time(logger)
    
    pathways = {
        'Inflammation': ['IL6', 'CXCL8', 'TNF', 'IL1B', 'CXCL10', 'CCL2'],
        'T_cell_activation': ['CD3D', 'CD3E', 'CD28', 'ICOS', 'IL2RA', 'LCK'],
        'B_cell_function': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'IGKC', 'IGHM'],
        'Antigen_presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'CD74', 'CIITA'],
        'Cytotoxicity': ['GZMA', 'GZMB', 'PRF1', 'GNLY', 'NKG7'],
        'Interferon_response': ['STAT1', 'IRF1', 'IRF7', 'ISG15', 'MX1'],
        'Cell_cycle': ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'CDK1']
    }
    
    #adata = run_pathway_analysis(
    #    adata,
    #    method='average_expression',
    #    pathways=pathways
    #)
    
    # Run enrichment analysis
    #adata = run_pathway_analysis(
    #    adata,
    #    method='enrichment',
    #    cluster_key='leiden_spatial'
    #)
    
    pathway_timer("Pathway analysis skipped")
    
    # Step 10: Advanced spatial statistics
    logger.info("Step 10: Advanced spatial statistics")
    advanced_timer = log_execution_time(logger)
    
    # # Run Geographically Weighted Regression
    # if 'moranI' in adata.uns:
    #     # Use top spatially variable gene as response
    #     top_gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
        
    #     adata = run_gwr_analysis(
    #         adata,
    #         response_var=top_gene,
    #         explanatory_vars=None,  # Will use top spatially variable genes
    #         adaptive=True,
    #         key_added='gwr_results'
    #     )
    
    # # Run local bivariate analysis
    # if 'prop_B_cells' in adata.obs and 'prop_T_cells' in adata.obs:
    #     adata = run_gwr_analysis(
    #         adata,
    #         method='correlation',
    #         var1='prop_B_cells',
    #         var2='prop_T_cells',
    #         key_added='local_correlation'
    #     )
    
    # # Run Bayesian analysis
    # adata = run_bayesian_analysis(
    #     adata,
    #     method='gp',
    #     genes=None,  # Will use highly variable genes
    #     key_added='bayesian_gp',
    #     n_samples=50
    # )
    
    # # Run topological data analysis
    # adata = run_tda_analysis(
    #     adata,
    #     method='persistent_homology',
    #     key_added='tda_ph',
    #     max_dim=1
    # )
    
    # # Extract topological features
    # adata = run_tda_analysis(
    #     adata,
    #     method='persistent_homology',
    #     key_added='topo_features',
    #     max_dim=1
    # )
    
    advanced_timer("Advanced spatial statistics skipped")
    
    # Step 11: Visualization
    logger.info("Step 11: Visualization")
    viz_timer = log_execution_time(logger)
    
    # static figures
    generate_static_figures(
        adata,
        output_dir=figures_dir,
        figsize=(10, 8),
        dpi=300
    )
    
    # interactive figures
    generate_interactive_figures(
        adata,
        output_dir=figures_dir
    )
    
    viz_timer("Visualization completed")
    
    # Step 12: Generate report
    logger.info("Step 12: Generating report")
    report_timer = log_execution_time(logger)
    
    report_path = generate_report(
        adata,
        output_dir=reports_dir,
        title="Advanced Spatial Transcriptomics Analysis Report"
    )
    
    report_timer("Report generation completed")
    
    # Step 13: Save results
    logger.info("Step 13: Saving results")
    save_timer = log_execution_time(logger)
    
    save_results(
        adata,
        output_dir=data_dir,
        save_formats=['h5ad', 'csv', 'zarr'],
        compress=True
    )
    
    save_timer("Results saved")
    
    # Log the end of the workflow
    end_timer("Entire workflow completed")
    
    logger.info(f"Analysis completed successfully. Results saved to {output_dir}")
    logger.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()