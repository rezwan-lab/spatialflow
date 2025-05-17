import click
import yaml
import logging
import os
from pathlib import Path
from datetime import datetime

from spatialflow.config import read_config, validate_config
from spatialflow.core.data_loader import load_data
from spatialflow.core.preprocessing import run_preprocessing
from spatialflow.core.feature_selection import run_feature_selection
from spatialflow.spatial.neighbors import build_spatial_graph
from spatialflow.spatial.statistics import calculate_spatial_stats
from spatialflow.spatial.clustering import run_spatial_clustering
from spatialflow.analysis.deconvolution import run_deconvolution
from spatialflow.analysis.ligrec import run_ligrec_analysis
from spatialflow.analysis.pathways import run_pathway_analysis
from spatialflow.utils.io import save_results
from spatialflow.visualization.reporting import generate_report
from spatialflow.utils.logging import setup_logging

def setup_output_dir(output_dir):
    """Create output directory structure"""
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "figures").mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)
    
    return output_path

def create_default_config(config_path):
    """Create a default configuration file"""
    default_config = {
        "data": {
            "input_path": "",
            "format": "10x_visium",
            "library_id": None
        },
        "preprocessing": {
            "min_genes": 200,
            "min_cells": 3,
            "target_sum": 10000,
            "n_top_genes": 2000
        },
        "spatial": {
            "n_neighs": 6,
            "coord_type": "generic",
            "n_perms": 100,
            "cluster_resolution": 0.5
        },
        "analysis": {
            "run_deconvolution": True,
            "run_ligrec": True,
            "run_pathways": True,
            "custom_gene_lists": {}
        },
        "visualization": {
            "create_interactive": True,
            "figsize": [10, 8],
            "dpi": 300
        },
        "output": {
            "save_adata": True,
            "generate_report": True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    return default_config

@click.group()
def cli():
    """SpatialFlow: A comprehensive tool for spatial transcriptomics analysis"""
    pass

@cli.command()
@click.argument('output_path', type=click.Path())
def init_config(output_path):
    """Initialize a default configuration file"""
    config_path = Path(output_path)
    if config_path.exists() and not click.confirm(f"The file {output_path} already exists. Overwrite?"):
        click.echo("Aborted.")
        return
    
    create_default_config(config_path)
    click.echo(f"Default configuration created at {output_path}")

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--output-dir', '-o', type=click.Path(), default='./spatialflow_results', help='Output directory')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO', help='Logging level')
def run(input_path, config, output_dir, log_level):
    """Run the SpatialFlow pipeline on a dataset"""
    output_path = setup_output_dir(output_dir)
    log_file = output_path / f"spatialflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_level, log_file)
    logger = logging.getLogger('spatialflow')
    logger.info(f"SpatialFlow analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if config:
        cfg = read_config(config)
        logger.info(f"Using configuration from {config}")
    else:
        logger.info("Using default configuration")
        default_config_path = output_path / "default_config.yaml"
        cfg = create_default_config(default_config_path)
    cfg['data']['input_path'] = input_path
    validate_config(cfg)
    try:
        logger.info("Loading data...")
        adata = load_data(**cfg['data'])
        logger.info("Running preprocessing...")
        adata = run_preprocessing(adata, **cfg['preprocessing'])
        logger.info("Running feature selection...")
        adata = run_feature_selection(adata, **cfg['preprocessing'])
        logger.info("Building spatial graph...")
        adata = build_spatial_graph(adata, n_neighs=cfg['spatial']['n_neighs'], 
                            coord_type=cfg['spatial']['coord_type'])
        
        logger.info("Calculating spatial statistics...")
        adata = calculate_spatial_stats(adata, **cfg['spatial'])
        

        logger.info("Running spatial clustering...")
        adata = run_spatial_clustering(adata, resolution=cfg['spatial']['cluster_resolution'], 
                               neighbors_key="spatial_connectivities")
        if cfg['analysis']['run_deconvolution']:
            logger.info("Running cell type deconvolution...")
            adata = run_deconvolution(adata)
        
        if cfg['analysis']['run_ligrec']:
            logger.info("Running ligand-receptor analysis...")
            adata = run_ligrec_analysis(adata, cluster_key="spatial_clusters", 
                                 n_perms=cfg['spatial'].get('n_perms', 1000))


        if cfg['analysis']['run_pathways']:
            logger.info("Running pathway analysis...")
            try:
                adata = run_pathway_analysis(adata, 
                                           group_key="spatial_clusters",  # Add this parameter
                                           custom_gene_lists=cfg['analysis'].get('custom_gene_lists', {}))
            except Exception as e:
                logger.warning(f"Pathway analysis failed: {str(e)}")
        if cfg['output']['save_adata']:
            logger.info("Saving analysis results...")
            save_results(adata, output_path / "data")
        if cfg['output']['generate_report']:
            logger.info("Generating analysis report...")
            generate_report(adata, output_path / "reports")
        
        logger.info(f"Analysis completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"Analysis completed successfully. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        click.echo(f"Analysis failed: {str(e)}")
        raise

@cli.command()
@click.argument('adata_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='./spatialflow_results', help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file for visualization')
def visualize(adata_path, output_dir, config):
    """Generate visualizations from a saved AnnData object"""
    from spatialflow.visualization.static import generate_static_figures
    from spatialflow.visualization.interactive import generate_interactive_figures
    output_path = setup_output_dir(output_dir)
    if config:
        cfg = read_config(config)
    else:
        cfg = {
            'visualization': {
                'create_interactive': True,
                'figsize': [10, 8],
                'dpi': 300
            }
        }
    import scanpy as sc
    adata = sc.read(adata_path)
    generate_static_figures(adata, output_path / "figures", 
                          figsize=cfg['visualization'].get('figsize', [10, 8]),
                          dpi=cfg['visualization'].get('dpi', 300))
    if cfg['visualization'].get('create_interactive', True):
        generate_interactive_figures(adata, output_path / "figures")
    
    click.echo(f"Visualization completed. Results saved to {output_dir}/figures")

def main():
    cli()

if __name__ == "__main__":
    main()
