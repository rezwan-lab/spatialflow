# Custom configuration for SpatialFlow - Focusing on cell-cell interactions
data:
  input_path: "data/my_spatial_dataset.h5ad"
  format: "h5ad"
  library_id: null

preprocessing:
  min_genes: 150
  min_cells: 5
  target_sum: 10000
  log_transform: true
  calculate_qc: true
  n_top_genes: 3000
  flavor: "seurat_v3"  # Seurat v3 method for HVG selection

spatial:
  n_neighs: 8              # neighbors
  coord_type: "generic"
  n_perms: 200             # permutations for robust statistics
  spatial_methods:
    - "moran"
    - "ripley"
    - "geary"              # Geary's C statistic
  cluster_resolution: 0.8  # finer clusters

clustering:
  method: "leiden"
  resolution: 0.8
  n_clusters: 12           # num of clusters for finer resolution
  use_pca: true
  use_spatial: true

deconvolution:
  run: true
  method: "marker_gene"
  markers:                  # predefined marker genes for major immune cell types
    B_cells:
      - "MS4A1"
      - "CD79A"
      - "CD79B"
      - "CD19"
      - "BANK1"
    T_cells_CD4:
      - "CD3D"
      - "CD3E"
      - "CD4"
      - "IL7R"
    T_cells_CD8:
      - "CD3D"
      - "CD3E"
      - "CD8A"
      - "CD8B"
    NK_cells:
      - "NCAM1"
      - "NKG7"
      - "KLRD1"
      - "KLRB1"
    Macrophages:
      - "CD68"
      - "CD163"
      - "MARCO"
      - "CSF1R"
    Dendritic_cells:
      - "FCER1A"
      - "CD1C"
      - "CLEC10A"
    Plasma_cells:
      - "MZB1"
      - "JCHAIN"
      - "IGHA1"
      - "IGHG1"

ligrec:
  run: true
  cluster_key: "leiden"
  n_perms: 200
  min_expr: 0.1          # Minimum expression threshold

pathways:
  run: true
  method: "average_expression"
  custom_gene_lists:
    Inflammation:
      - "IL6"
      - "IL1B"
      - "TNF"
      - "CXCL8"
      - "CXCL10"
      - "CCL2"
      - "PTGS2"
      - "IL1A"
    T_cell_activation:
      - "CD3D"
      - "CD3E"
      - "CD28"
      - "ICOS"
      - "IL2RA"
      - "LCK"
      - "ZAP70"
      - "ITK"
    B_cell_function:
      - "CD19"
      - "CD79A"
      - "CD79B"
      - "MS4A1"
      - "IGKC"
      - "IGHM"
      - "CD22"
      - "BANK1"
    Cytotoxicity:
      - "GZMA"
      - "GZMB"
      - "GZMH"
      - "PRF1"
      - "GNLY"
      - "NKG7"
    Cell_cell_communication:
      - "ICAM1"
      - "VCAM1"
      - "ITGAL"
      - "ITGB2"
      - "CD86"
      - "CD80"
      - "CD28"
      - "CTLA4"
      - "PDCD1"
      - "CD274"

advanced:
  run_gwr: true
  gwr_params:
    adaptive: true
    bw_min: 0.1
    bw_max: 0.5
  
  run_bayesian: true
  bayesian_params:
    method: "gp"
    n_samples: 100
    length_scale: 0.2
  
  run_topology: true
  topology_params:
    method: "persistent_homology"
    max_dim: 2
    use_expression: true
    n_components: 15

visualization:
  create_static: true
  create_interactive: true
  figsize: [12, 10]
  dpi: 300
  color_maps:
    expression: "viridis"
    clusters: "tab20"
    cell_types: "Set3"
    continuous: "plasma"
  spot_size: 2.0
  include_legends: true

output:
  output_dir: "cell_interaction_analysis"
  save_adata: true
  save_formats:
    - "h5ad"
    - "csv"
    - "zarr"
  compress: true
  generate_report: true
  report_title: "Cell-Cell Interaction Analysis in Spatial Context"