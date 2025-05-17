"""
Basic Spatial Transcriptomics Analysis Workflow

This is a standard basic workflow for analyzing spatial transcriptomics
data using the SpatialFlow package.
"""

import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path

from spatialflow.utils.logging import setup_logging
from spatialflow.core.data_loader import load_data
from spatialflow.core.preprocessing import run_preprocessing
from spatialflow.core.feature_selection import run_feature_selection
from spatialflow.spatial.neighbors import build_spatial_graph
from spatialflow.spatial.statistics import calculate_spatial_stats
from spatialflow.spatial.clustering import run_spatial_clustering
from spatialflow.analysis.deconvolution import run_deconvolution
from spatialflow.analysis.pathways import run_pathway_analysis
from spatialflow.visualization.static import generate_static_figures
from spatialflow.visualization.reporting import generate_report
from spatialflow.utils.io import save_results

def main():
    # Set up logging
    logger = setup_logging(level="INFO", log_file="spatialflow_basic_workflow.log")
    logger.info("Starting basic spatial transcriptomics analysis workflow")
    
    # Create output directories
    output_dir = Path("spatialflow_output")
    output_dir.mkdir(exist_ok=True)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    # You can use a local file or a built-in dataset
    adata = sq.datasets.visium("V1_Human_Lymph_Node")
    # Alternatively, load from a file:
    # adata = load_data("path/to/your/data.h5ad", format='h5ad')
    
    logger.info(f"Loaded dataset with {adata.n_obs} spots and {adata.n_vars} genes")
    
    # Step 2: Preprocessing
    logger.info("Step 2: Preprocessing")
    adata = run_preprocessing(
        adata,
        min_genes=200,
        min_cells=3,
        target_sum=1e4,
        log_transform=True,
        calculate_qc=True
    )
    
    # Step 3: Feature selection
    logger.info("Step 3: Feature selection")
    adata = run_feature_selection(
        adata,
        n_top_genes=2000,
        flavor='seurat',
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        span=0.3
    )
    
    # Run dimensionality reduction
    sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
    
    # Step 4: Spatial analysis
    logger.info("Step 4: Spatial analysis")
    
    # Build spatial graph
    adata = build_spatial_graph(
        adata,
        n_neighs=6,
        coord_type="generic"
    )
    
    # Calculate spatial statistics
    adata = calculate_spatial_stats(
        adata,
        methods=['moran'],
        n_perms=100
    )
    
    # Step 5: Clustering
    logger.info("Step 5: Clustering")
    adata = run_spatial_clustering(
        adata,
        method='leiden',
        resolution=0.5,
        neighbors_key='spatial_connectivities',
    )
    
    # Step 6: Cell type deconvolution
    logger.info("Step 6: Cell type deconvolution")
    # Define marker genes for cell types
    markers = {
        'B_cells': ['MS4A1', 'CD79A', 'CD79B', 'BANK1'],
        'T_cells': ['CD3D', 'CD3E', 'CD8A', 'CD4', 'IL7R'],
        'Macrophages': ['CD68', 'CD163', 'MARCO'],
        'Dendritic_cells': ['FCER1A', 'CD1C', 'CLEC10A'],
        'Endothelial_cells': ['PECAM1', 'VWF', 'CDH5']
    }
    
    adata = run_deconvolution(
        adata,
        method='manual_markers',
        marker_genes=markers
    )
    
    # Step 7: Pathway analysis
    logger.info("Step 7: Pathway analysis")
    pathways = {
        'Inflammation': ['IL6', 'CXCL8', 'TNF', 'IL1B', 'CXCL10', 'CCL2'],
        'T_cell_activation': ['CD3D', 'CD3E', 'CD28', 'ICOS', 'IL2RA', 'LCK'],
        'B_cell_function': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'IGKC', 'IGHM']
    }
    
    #adata = run_pathway_analysis(
    #    adata,
    #    method='simple',
    #    pathways=pathways
    #)
    logger.info("Skipping pathway analysis")
    
    # Step 8: Visualization
    logger.info("Step 8: Visualization")
    
    # Generate static figures
    generate_static_figures(
        adata,
        output_dir=figures_dir,
        figsize=(10, 8),
        dpi=300
    )
    
    # Step 9: Generate report
    logger.info("Step 9: Generating report")
    report_path = generate_report(
        adata,
        output_dir=reports_dir,
        title="Spatial Transcriptomics Analysis Report"
    )
    
    # Step 10: Save results
    logger.info("Step 10: Saving results")
    save_results(
        adata,
        output_dir=data_dir,
        save_formats=['h5ad', 'csv'],
        compress=True
    )
    
    logger.info(f"Analysis completed successfully. Results saved to {output_dir}")
    logger.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()