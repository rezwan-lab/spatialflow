# Default configuration for SpatialFlow
data:
  input_path: ""           # Path to input data (will be set by CLI)
  format: "10x_visium"     # Format of input data: '10x_visium', 'h5ad', 'csv'
  library_id: null         # Library ID for Visium data (optional)

preprocessing:
  min_genes: 200           # Minimum number of genes for a spot to be kept
  min_cells: 3             # Minimum number of spots for a gene to be kept
  target_sum: 10000        # Target sum for normalization
  log_transform: true      # Whether to log-transform the data
  calculate_qc: true       # Whether to calculate QC metrics
  n_top_genes: 2000        # Number of highly variable genes to select
  flavor: "seurat"         # Method for identifying highly variable genes

spatial:
  n_neighs: 6              # Number of neighbors for spatial graph
  coord_type: "generic"    # Type of coordinates: 'generic', 'visium'
  n_perms: 100             # Number of permutations for spatial statistics
  spatial_methods:         # Spatial methods to run
    - "moran"
    - "ripley"
  cluster_resolution: 0.5  # Resolution parameter for Leiden clustering

clustering:
  method: "leiden"         # Clustering method: 'leiden', 'louvain', 'kmeans'
  resolution: 0.5          # Resolution parameter for community detection
  n_clusters: 8            # Number of clusters for k-means
  use_pca: true            # Whether to use PCA for clustering
  use_spatial: true        # Whether to use spatial information for clustering

deconvolution:
  run: true                # Whether to run deconvolution
  method: "marker_gene"    # Method: 'marker_gene', 'nmf', 'cell2location'
  n_cell_types: 7          # Number of cell types for NMF

ligrec:
  run: true                # Whether to run ligand-receptor analysis
  cluster_key: "leiden"    # Key for cluster assignments
  n_perms: 100             # Number of permutations

pathways:
  run: true                # Whether to run pathway analysis
  method: "average_expression"  # Method: 'average_expression', 'enrichment'
  custom_gene_lists: {}    # Custom gene lists for pathway analysis

advanced:
  run_gwr: true            # Whether to run GWR analysis
  run_bayesian: true       # Whether to run Bayesian analysis
  run_topology: true       # Whether to run topological analysis
  n_samples: 50            # Number of samples for Bayesian methods

visualization:
  create_static: true      # Whether to create static visualizations
  create_interactive: true # Whether to create interactive visualizations
  figsize: [10, 8]         # Figure size for static visualizations
  dpi: 300                 # DPI for static visualizations

output:
  output_dir: "spatialflow_results"  # Output directory
  save_adata: true                   # Whether to save the AnnData object
  save_formats:                      # Formats to save results in
    - "h5ad"
    - "csv"
  compress: true                     # Whether to compress saved files
  generate_report: true              # Whether to generate a report