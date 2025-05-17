import yaml
import logging
from pathlib import Path
import json

logger = logging.getLogger('spatialflow.config')

def read_config(config_path):
    """
    Read a configuration file in YAML format
    
    Parameters
    ----------
    config_path : str or Path
        Path to the configuration file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    if suffix == '.yaml' or suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")
    
    return config

def validate_config(config):
    """
    Validate a configuration dictionary
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    
    Returns
    -------
    bool
        True if configuration is valid
        
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    required_sections = ['data', 'preprocessing', 'spatial']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    if 'input_path' not in config['data']:
        raise ValueError("Missing required parameter 'input_path' in data section")
    
    return True

def update_config(config, overrides):
    """
    Update a configuration dictionary with override values
    
    Parameters
    ----------
    config : dict
        Original configuration dictionary
    overrides : dict
        Dictionary with override values
    
    Returns
    -------
    dict
        Updated configuration dictionary
    """
    import copy
    updated_config = copy.deepcopy(config)
    def _update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_config = _update_dict(updated_config, overrides)
    
    return updated_config

def write_config(config, output_path):
    """
    Write a configuration dictionary to a file
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : str or Path
        Path to write the configuration file
    
    Returns
    -------
    Path
        Path to the written configuration file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = output_path.suffix.lower()
    
    if suffix == '.yaml' or suffix == '.yml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {suffix}")
    
    return output_path

def get_parameter_defaults():
    """
    Get default parameter values for all pipeline components
    
    Returns
    -------
    dict
        Dictionary with default parameter values
    """
    defaults = {
        'data': {
            'format': '10x_visium',
            'library_id': None
        },
        'preprocessing': {
            'min_genes': 200,
            'min_cells': 3,
            'target_sum': 10000,
            'n_top_genes': 2000
        },
        'spatial': {
            'n_neighs': 6,
            'coord_type': 'generic',
            'n_perms': 100,
            'cluster_resolution': 0.5
        },
        'analysis': {
            'run_deconvolution': True,
            'run_ligrec': True,
            'run_pathways': True,
            'custom_gene_lists': {}
        },
        'visualization': {
            'create_interactive': True,
            'figsize': [10, 8],
            'dpi': 300
        },
        'output': {
            'save_adata': True,
            'generate_report': True
        }
    }
    
    return defaults

def get_parameter_descriptions():
    """
    Get descriptions of all configurable parameters
    
    Returns
    -------
    dict
        Dictionary with parameter descriptions
    """
    descriptions = {
        'data': {
            'input_path': 'Path to the input data file',
            'format': 'Format of the input data (10x_visium, anndata, etc.)',
            'library_id': 'Library ID for Visium data'
        },
        'preprocessing': {
            'min_genes': 'Minimum number of genes expressed for a spot to be kept',
            'min_cells': 'Minimum number of spots a gene must be expressed in to be kept',
            'target_sum': 'Target sum for normalization',
            'n_top_genes': 'Number of highly variable genes to select'
        },
        'spatial': {
            'n_neighs': 'Number of neighbors for spatial graph construction',
            'coord_type': 'Type of coordinates (generic, visium)',
            'n_perms': 'Number of permutations for spatial statistics',
            'cluster_resolution': 'Resolution parameter for Leiden clustering'
        },
        'analysis': {
            'run_deconvolution': 'Whether to run cell type deconvolution',
            'run_ligrec': 'Whether to run ligand-receptor analysis',
            'run_pathways': 'Whether to run pathway analysis',
            'custom_gene_lists': 'Custom gene lists for pathway analysis'
        },
        'visualization': {
            'create_interactive': 'Whether to create interactive visualizations',
            'figsize': 'Figure size for static visualizations [width, height]',
            'dpi': 'DPI for static visualizations'
        },
        'output': {
            'save_adata': 'Whether to save the AnnData object',
            'generate_report': 'Whether to generate an analysis report'
        }
    }
    
    return descriptions
