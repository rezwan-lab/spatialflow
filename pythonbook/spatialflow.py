"""spatialflow.ipynb

spatial_visium_squidpy_lymphnode
# Author: Dr Rezwanuzzaman Laskar

## Comprehensive Spatial Transcriptomics Analysis with Squidpy

Advanced spatial transcriptomics analysis workflow for Visium datasets

Comprehensive notebook for analyzing spatial transcriptomics data (specifically 10x Visium data) using the Squidpy package.

Contents

0. Environment Setup
1. Data Loading & Exploration
2. Quality Control & Preprocessing
3. Feature Selection & Dimensionality Reduction
4. Spatial Analysis Fundamentals
5. Clustering & Spatial Domains
6. Advanced Spatial Statistics
7. Marker Gene Identification
8. Cell Type Annotation & Deconvolution
9. Ligand-Receptor & Cell-Cell Communication
10. Pathway Analysis
11. Integration with External Datasets
12. Tissue-Specific Analysis
13. Validation Planning
14. Visualization & Publication
15. Data Export & Documentation

## 1. Environment Setup

1.1 Installing Dependencies
"""

pip install numpy==1.24.4
pip install scanpy==1.9.5 anndata==0.9.1
pip install xarray==2023.1.0 "dask[complete]==2023.3.2" pandas==2.0.3
pip install squidpy==1.3.1 docrep spatialdata validators matplotlib-scalebar
pip install matplotlib seaborn scikit-learn statsmodels
pip install leidenalg python-igraph numba networkx

pip install mgwr --quiet
pip install pymc3
pip install gudhi
pip install ripser
pip install mgwr

"""**1.2 Loading Libraries**"""

import numpy as np
import scanpy as sc
import squidpy as sq
import anndata as ad
import xarray as xr
import dask
import traceback
import mgwr
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100


sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, frameon=False)

print(f"numpy: {np.__version__}")
print(f"scanpy: {sc.__version__}")
print(f"squidpy: {sq.__version__}")
print(f"anndata: {ad.__version__}")
print(f"xarray: {xr.__version__}")
print(f"dask: {dask.__version__}")

"""## 2. Data Loading & Exploration

**2.1 Loading a Visium Dataset**
"""

mkdir -p data
#!wget -O data/heart_dataset.h5ad https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Heart_Sagittal_Anterior/V1_Heart_Sagittal_Anterior_filtered_feature_bc_matrix.h5
import urllib.request

url = "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Heart_Sagittal_Anterior/V1_Heart_Sagittal_Anterior_filtered_feature_bc_matrix.h5"
output_path = "data/heart_dataset.h5ad"

urllib.request.urlretrieve(url, output_path)


adata = sc.read_10x_h5("data/heart_dataset.h5ad")
adata.uns['spatial'] = {}

adata = sq.datasets.visium("V1_Human_Lymph_Node")
print(f"Dataset shape: {adata.shape}")  # (spots, genes)
print(f"Available data layers: {list(adata.layers.keys())}")
print(f"Spatial coordinates stored in: {adata.obsm.keys()}")

"""**2.2 Exploring the Data Structure**"""

adata.var_names_make_unique()
print(f"Observation (spot) metadata columns: {adata.obs.columns.tolist()}")
print(f"Variable (gene) metadata columns: {adata.var.columns.tolist()}")
print(f"Available metadata: {list(adata.uns.keys())}")
print(f"Spatial coordinates shape: {adata.obsm['spatial'].shape}")

"""# 2.3 Exploring Image Data"""

if 'spatial' in adata.uns:
    library_ids = list(adata.uns['spatial'].keys())
    print(f"Available library IDs: {library_ids}")
    first_library = library_ids[0]
    available_images = list(adata.uns['spatial'][first_library]['images'].keys())
    print(f"Available images for {first_library}: {available_images}")
    if 'hires' in available_images:
        img_shape = adata.uns['spatial'][first_library]['images']['hires'].shape
        print(f"High-resolution image shape: {img_shape}")
    if 'scalefactors' in adata.uns['spatial'][first_library]:
        print(f"Scale factors: {adata.uns['spatial'][first_library]['scalefactors']}")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sq.pl.spatial_scatter(adata, library_id="V1_Human_Lymph_Node", color="total_counts",
                     title="Human Lymph Node - Total UMI Counts")
sq.pl.spatial_scatter(adata, library_id="V1_Human_Lymph_Node", color="n_genes_by_counts",
                     title="Human Lymph Node - Genes Detected per Spot")

"""## 3. Quality Control & Preprocessing

**3.1 Quality Control**
"""

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
print("Quality metrics summary:")
print(adata.obs[['total_counts', 'n_genes_by_counts']].describe())
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
sc.pl.violin(adata, 'n_genes_by_counts', jitter=0.4, ax=axs[0], show=False)
sc.pl.violin(adata, 'total_counts', jitter=0.4, ax=axs[1], show=False)
plt.tight_layout()
plt.show()

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='total_counts')
sq.pl.spatial_scatter(adata, library_id="V1_Human_Lymph_Node",
                     color=["total_counts", "n_genes_by_counts"], ncols=2)

"""**3.2 Filtering Low-Quality Spots and Genes**"""

original_shape = adata.shape
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"Original matrix shape: {original_shape}")
print(f"After filtering: {adata.shape}")
print(f"Removed {original_shape[0] - adata.shape[0]} spots and {original_shape[1] - adata.shape[1]} genes")
if adata.shape[0] < original_shape[0]:
    orig = sq.datasets.visium("V1_Human_Lymph_Node")
    orig.obs['kept'] = orig.obs.index.isin(adata.obs.index).astype(str)
    sq.pl.spatial_scatter(orig, color='kept', library_id="V1_Human_Lymph_Node",
                         title="Spots retained after filtering")

"""**3.3 Normalization and Log-transformation**"""

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.obs['total_counts_normalized'] = adata.X.sum(axis=1)
print("Total counts after normalization:")
print(adata.obs['total_counts_normalized'].describe())

plt.figure(figsize=(10, 6))
plt.scatter(adata.obs['total_counts'], adata.obs['total_counts_normalized'], alpha=0.5)
plt.xscale('log')
plt.xlabel('Original total counts')
plt.ylabel('Normalized total counts')
plt.title('Effect of normalization')
plt.axhline(y=10000, color='red', linestyle='--')
plt.show()

"""# 4. Feature Selection & Dimensionality Reduction

**4.1 Identifying Highly Variable Genes**
"""

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)
print(f"Number of highly variable genes: {adata.var.highly_variable.sum()}")

sc.pl.highly_variable_genes(adata)
hvg_list = adata.var.index[adata.var.highly_variable].tolist()

"""**4.2 Manual Gene Variability Analysis**"""

gene_var = np.var(adata.X.toarray(), axis=0)
gene_mean = np.mean(adata.X.toarray(), axis=0)

top_var_idx = np.argsort(gene_var)[-50:]  # Top 50 variable genes
top_var_genes = adata.var_names[top_var_idx].tolist()

# Calculate coefficient of variation (CV = std/mean)
gene_cv = np.divide(np.sqrt(gene_var), gene_mean, out=np.zeros_like(gene_var), where=gene_mean!=0)
top_cv_idx = np.argsort(gene_cv)[-50:]  # Top 50 genes by CV
top_cv_genes = adata.var_names[top_cv_idx].tolist()

plt.figure(figsize=(10, 6))
plt.scatter(gene_mean, gene_var, alpha=0.3, s=3)
plt.scatter(gene_mean[top_var_idx], gene_var[top_var_idx], c='red', s=30)
for i, gene_name in enumerate(top_var_genes[:10]):
    plt.annotate(gene_name, (gene_mean[top_var_idx][-i-1], gene_var[top_var_idx][-i-1]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mean Expression')
plt.ylabel('Variance')
plt.title('Mean-Variance Relationship')
plt.tight_layout()
plt.show()

sq.pl.spatial_scatter(adata, color=top_var_genes[:5], ncols=5)

"""**4.3 Principal Component Analysis (PCA)**"""

sc.pp.pca(adata, n_comps=30, use_highly_variable=True)

plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(adata.uns['pca']['variance_ratio']), 'o-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.axhline(y=0.8, color='r', linestyle='--')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

sc.pl.pca(adata, color=['total_counts', 'n_genes_by_counts'])
sc.pl.pca_loadings(adata, components=[1, 2, 3, 4])

plt.figure(figsize=(10, 8))
plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1],
            c=adata.obs['total_counts'], cmap='viridis', s=50, alpha=0.7)
plt.colorbar(label='Total Counts')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Spots in PCA Space')
plt.grid(True, alpha=0.3)
plt.show()

"""# 5. Spatial Analysis Fundamentals

**5.1 Spatial Neighbors Graph**
"""

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

print("Spatial neighbors graph info:")
print(f"Number of connections: {adata.obsp['spatial_connectivities'].getnnz()}")
print(f"Average number of neighbors: {adata.obsp['spatial_connectivities'].getnnz() / adata.n_obs}")

sq.pl.spatial_scatter(adata, color="total_counts",
                     title="Spot UMI counts")
try:
    adata.obs['total_count_category'] = pd.qcut(adata.obs['total_counts'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    sq.pl.spatial_scatter(adata, color="total_count_category",
                         title="UMI Count Categories")
except Exception as e:
    print(f"Couldn't calculate enrichment: {e}")

"""**5.2 Spatial Autocorrelation**"""

sq.gr.spatial_autocorr(
    adata,
    genes=hvg_list[:100],  # top 100 highly variable genes
    mode="moran",
    n_perms=100,        # Number of permutations for significance testing
    n_jobs=4
)

if "moranI" in adata.uns:
    print("Available columns in Moran's I results:")
    print(adata.uns["moranI"].columns.tolist())

    top_spatial_genes = adata.uns["moranI"].sort_values("I", ascending=False).head(10)
    print("\nTop genes by spatial autocorrelation:")
    print(top_spatial_genes)

    pval_columns = [col for col in adata.uns["moranI"].columns if 'p' in col.lower()]
    print(f"\nP-value columns found: {pval_columns}")

    if pval_columns:
        pval_col = pval_columns[0]
        sig_genes = adata.uns["moranI"][adata.uns["moranI"][pval_col] < 0.05]
        print(f"\nGenes with significant spatial autocorrelation (p < 0.05): {len(sig_genes)}")

    plt.figure(figsize=(10, 5))
    adata.uns["moranI"].sort_values("I", ascending=False).head(20).plot.bar(y="I")
    plt.title("Top 20 Genes by Spatial Autocorrelation (Moran's I)")
    plt.tight_layout()
    plt.show()

    sq.pl.spatial_scatter(adata, color=top_spatial_genes.index[:5], ncols=5)

"""**5.3 Ripley's Statistics**"""

if 'leiden' not in adata.obs.columns:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['spatial_clusters'] = pd.Categorical(kmeans.labels_.astype(str))
    cluster_key = 'spatial_clusters'
else:
    cluster_key = 'leiden'
try:
    print(f"Data type of {cluster_key}: {type(adata.obs[cluster_key])}")

    sq.gr.ripley(adata, cluster_key=cluster_key, mode="L")
    sq.pl.ripley(adata, cluster_key=cluster_key, mode="L")
except Exception as e:
    print(f"Ripley's statistics calculation failed: {e}")
    print("This can happen when the spatial structure doesn't meet certain requirements.")

print("Recreating leiden clusters...")

sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.leiden(adata, resolution=0.5)

sq.pl.spatial_scatter(adata, color="leiden", size=30, shape=None, library_id="V1_Human_Lymph_Node")
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
sc.tl.leiden(adata, resolution=0.8, neighbors_key="spatial_neighbors", key_added="spatial_domains")
sq.pl.spatial_scatter(adata, color="spatial_domains", size=20, shape=None, library_id="V1_Human_Lymph_Node")

print("Done! both clustering columns recreated.")

# Check available library IDs
print("Available library IDs:", list(adata.uns['spatial'].keys()) if 'spatial' in adata.uns else "None")

"""# 6. Clustering & Spatial Domains

**6.1 Graph-Based Clustering (Leiden)**
"""

try:
    sc.tl.leiden(adata, resolution=0.5, neighbors_key="spatial_neighbors")
    adata.uns.pop('leiden_colors', None)
    sq.pl.spatial_scatter(adata, color="leiden", library_id="V1_Human_Lymph_Node")
    print(f"Identified {len(adata.obs['leiden'].unique())} distinct clusters")
except Exception as e:
    print(f"Leiden clustering failed with error: {e}")

"""**6.2 K-Means Clustering**"""

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
sq.pl.spatial_scatter(adata, color="kmeans", library_id="V1_Human_Lymph_Node",
                     title="KMeans Clusters")

plt.figure(figsize=(8, 6))
plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1],
           c=[int(x) for x in adata.obs['kmeans']], cmap='tab20', s=10)
plt.colorbar(label='KMeans Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA + KMeans clusters')
plt.show()

"""**6.3 Advanced Spatial Domain Identification**"""

spatial_coords = adata.obsm['spatial']

from sklearn.neighbors import NearestNeighbors

n_neighbors = 6
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(spatial_coords)
distances, indices = nbrs.kneighbors(spatial_coords)
n_spots = adata.n_obs
n_genes = adata.n_vars
neighborhood_expr = np.zeros((n_spots, n_genes))

print("Creating spatially-smoothed expression matrix...")
for i in range(n_spots):
    neighborhood = indices[i]
    neighborhood_expr[i] = adata.X[neighborhood].mean(axis=0)
adata_spatial = ad.AnnData(X=neighborhood_expr, obs=adata.obs.copy())
adata_spatial.obsm['spatial'] = spatial_coords

sc.pp.pca(adata_spatial, n_comps=20)
kmeans = KMeans(n_clusters=8).fit(adata_spatial.obsm['X_pca'])
adata.obs['spatial_domains'] = kmeans.labels_.astype(str)
sq.pl.spatial_scatter(adata, color='spatial_domains', palette='tab10',
                     title="Spatial Domains from Neighborhood Expression")

"""## **Spatial Trajectory Analysis**"""

print("\n=== Spatial Trajectory Analysis ===")

min_counts_spot = adata.obs['total_counts'].idxmin()
root_idx = adata.obs.index.get_indexer([min_counts_spot])[0]

from scipy.spatial.distance import pdist, squareform
coords = adata.obsm['spatial']
dist_matrix = squareform(pdist(coords))
distances_from_root = dist_matrix[root_idx]

# pseudo-pseudotime
adata.obs['distance_pseudotime'] = distances_from_root
sq.pl.spatial_scatter(adata, color='distance_pseudotime',
                     title="Distance-based Pseudotime")

from scipy.stats import spearmanr

gene_correlations = []
for gene in adata.uns["moranI"].index[:20]:
    if gene in adata.var_names:
        gene_expr = adata[:, gene].X.toarray().flatten()
        corr, pval = spearmanr(gene_expr, adata.obs['distance_pseudotime'])
        gene_correlations.append({
            'gene': gene,
            'correlation': corr,
            'pvalue': pval
        })

gene_corr_df = pd.DataFrame(gene_correlations)
gene_corr_df = gene_corr_df.sort_values('correlation', key=abs, ascending=False)
print("\nTop genes correlated with spatial pseudotime:")
print(gene_corr_df.head(10))
top_corr_genes = gene_corr_df.head(5)['gene'].tolist()
sq.pl.spatial_scatter(adata, color=top_corr_genes, ncols=5)

"""# 7. Advanced Spatial Statistics

**7.1 Co-occurrence Analysis**
"""

try:
    sq.gr.co_occurrence(adata, cluster_key="leiden")
    sq.pl.co_occurrence(adata, cluster_key="leiden")
    print("Co-occurrence analysis completed successfully")
except Exception as e:
    print(f"Co-occurrence analysis error: {e}")

try:
    if 'spatial_neighbors' not in adata.uns:
        sq.gr.spatial_neighbors(adata, coord_type="generic")
        print("Computed spatial neighbors")

    sq.gr.centrality_scores(adata, cluster_key="leiden")
    sq.pl.centrality_scores(adata, cluster_key="leiden")
    print("Centrality analysis completed")
except Exception as e:
    print(f"Centrality analysis error: {e}")

from scipy import ndimage

coords = adata.obsm['spatial']
labels = adata.obs['spatial_domains'].cat.codes.values

grid_size = 100
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
x_bins = np.linspace(x_min, x_max, grid_size)
y_bins = np.linspace(y_min, y_max, grid_size)

n_clusters = len(adata.obs['spatial_domains'].cat.categories)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for cluster_id in range(min(n_clusters, 6)):
    cluster_mask = (labels == cluster_id)

    hist, _, _ = np.histogram2d(
        coords[cluster_mask, 0], coords[cluster_mask, 1],
        bins=[x_bins, y_bins]
    )

    smoothed = ndimage.gaussian_filter(hist, sigma=1.5)
    ax = axs[cluster_id]
    im = ax.imshow(smoothed.T, origin='lower', cmap='viridis',
                 extent=[x_min, x_max, y_min, y_max])
    ax.set_title(f"Cluster {cluster_id} Density")

plt.tight_layout()
plt.show()

"""**7.2 Centrality Scores**"""

if 'spatial_domains_colors' in adata.uns:
    adata.uns.pop('spatial_domains_colors')
    print("Removed existing color palette")
n_categories = len(adata.obs[cluster_key].cat.categories)
print(f"Number of categories in {cluster_key}: {n_categories}")

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

if not all(adata.obs[cluster_key].cat.categories == [str(i) for i in range(n_categories)]):
    old_categories = adata.obs[cluster_key].cat.categories
    new_categories = [str(i) for i in range(len(old_categories))]
    adata.obs[cluster_key] = adata.obs[cluster_key].cat.rename_categories(dict(zip(old_categories, new_categories)))
    print(f"Renamed categories to: {new_categories}")

n_categories = len(adata.obs[cluster_key].cat.categories)
colors = plt.cm.tab10(range(n_categories))
palette = [to_hex(c) for c in colors]
adata.uns[f'{cluster_key}_colors'] = palette
print(f"Created new palette with {len(palette)} colors")

try:
    sq.gr.centrality_scores(adata, cluster_key=cluster_key)
    sq.pl.centrality_scores(adata, cluster_key=cluster_key)
    print("Centrality analysis completed")
except Exception as e:
    print(f"Centrality analysis error: {e}")

"""**7.3 Custom Spatial Analysis**"""

cluster_assignments = adata.obs[cluster_key].values

from scipy.stats import entropy

local_heterogeneity = np.zeros(adata.n_obs)
neighbors_graph = adata.obsp['spatial_connectivities']
for i in range(adata.n_obs):
    neighbor_indices = neighbors_graph[i].nonzero()[1]
    neighbor_clusters = cluster_assignments[neighbor_indices]
    unique_clusters, counts = np.unique(neighbor_clusters, return_counts=True)
    if len(counts) > 1:
        probs = counts / counts.sum()
        local_heterogeneity[i] = entropy(probs)

adata.obs['local_heterogeneity'] = local_heterogeneity
sq.pl.spatial_scatter(adata, color='local_heterogeneity',
                     title="Local Cluster Heterogeneity")

"""## **Advanced Spatial Statistics**"""

# Geographically Weighted Regression (GWR)
t_marker = 'CD3E'  # T cell
b_marker = 'MS4A1'  # B cell
mac_marker = 'CD68'  # Macrophage
endo_marker = 'PECAM1'  # Endothelial


available_markers = []
for marker in [t_marker, b_marker, mac_marker, endo_marker]:
    if marker in adata.var_names:
        available_markers.append(marker)
        print(f"Found marker gene: {marker}")
    else:
        print(f"Marker gene not found: {marker}")

if len(available_markers) >= 2:
    response_marker = available_markers[0]
    y = adata[:, response_marker].X.toarray().flatten()

    explanatory_markers = available_markers[1:]
    X = adata[:, explanatory_markers].X.toarray()
    X = np.column_stack([np.ones(X.shape[0]), X])
    coords = adata.obsm['spatial']

    print(f"\nRunning GWR with:")
    print(f"  Response variable: {response_marker}")
    print(f"  Explanatory variables: {explanatory_markers}")

    try:
        bw = Sel_BW(coords, y, X).search(criterion='AICc')
        print(f"Optimal bandwidth: {bw}")
        gwr_model = GWR(coords, y, X, bw).fit()
        column_names = ['intercept'] + explanatory_markers
        for i, col in enumerate(column_names):
            adata.obs[f'gwr_{col}'] = gwr_model.params[:, i]
        coef_cols = [f'gwr_{col}' for col in explanatory_markers]
        sq.pl.spatial_scatter(adata, color=coef_cols, ncols=len(coef_cols))
        print(f"GWR model R²: {gwr_model.R2}")
    except Exception as e:
        print(f"GWR analysis failed: {e}")
        print("This may be due to insufficient data points or multicollinearity.")
else:
    print("Not enough marker genes found for GWR analysis.")
    print("\nPerforming simple spatial correlation analysis instead:")

    top_genes = adata.uns["moranI"].sort_values("I", ascending=False).head(3).index.tolist()
    print(f"Using top spatially variable genes: {top_genes}")

    gene_expr = adata[:, top_genes].X.toarray()
    for i, gene1 in enumerate(top_genes):
        for gene2 in top_genes[i+1:]:
            corr = np.corrcoef(gene_expr[:, i], gene_expr[:, top_genes.index(gene2)])[0, 1]
            print(f"Correlation between {gene1} and {gene2}: {corr:.3f}")

# Geographically Weighted Regression (GWR) 2
from scipy import stats

t_marker = 'CD3E'
b_marker = 'MS4A1'
mac_marker = 'CD68'
endo_marker = 'PECAM1'

available_markers = [m for m in [t_marker, b_marker, mac_marker, endo_marker] if m in adata.var_names]
print(f"Available markers: {available_markers}")

if len(available_markers) >= 2:
    marker_expr = adata[:, available_markers].X.toarray()
    coords = adata.obsm['spatial']
    n_spots = adata.n_obs
    n_markers = len(available_markers)
    k_neighbors = 30

    for i in range(n_markers):
        for j in range(i+1, n_markers):
            marker1 = available_markers[i]
            marker2 = available_markers[j]
            local_corr = np.zeros(n_spots)
            local_pval = np.zeros(n_spots)

            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
            distances, indices = nbrs.kneighbors(coords)

            for spot_idx in range(n_spots):
                neighborhood = indices[spot_idx]
                expr1 = marker_expr[neighborhood, i]
                expr2 = marker_expr[neighborhood, j]
                corr, pval = stats.pearsonr(expr1, expr2)
                local_corr[spot_idx] = corr
                local_pval[spot_idx] = pval

            adata.obs[f'local_corr_{marker1}_{marker2}'] = local_corr
            adata.obs[f'local_pval_{marker1}_{marker2}'] = local_pval

            plt.figure(figsize=(10, 8))
            sq.pl.spatial_scatter(adata, color=f'local_corr_{marker1}_{marker2}',
                                 title=f'Local Correlation: {marker1} vs {marker2}',
                                 cmap='coolwarm', vmin=-1, vmax=1)

            global_corr, global_pval = stats.pearsonr(marker_expr[:, i], marker_expr[:, j])
            print(f"Global correlation between {marker1} and {marker2}: {global_corr:.3f} (p={global_pval:.3e})")
            print(f"Mean local correlation: {local_corr.mean():.3f}")
            print(f"Range of local correlations: [{local_corr.min():.3f}, {local_corr.max():.3f}]")
    print("\nAnalysis complete. Local correlation maps show how the relationship between markers varies spatially.")
else:
    print("Not enough marker genes available for analysis.")

"""## **Bayesian Spatial Analysis**"""

try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Verdana']

    if 'moranI' in adata.uns:
        gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
    else:
        gene_var = np.var(adata.X.toarray(), axis=0)
        top_var_idx = np.argsort(gene_var)[-1]
        gene = adata.var_names[top_var_idx]
    print(f"Running spatial analysis for gene: {gene}")

    gene_idx = adata.var_names.get_loc(gene)
    expression = adata.X[:, gene_idx].toarray().flatten()
    coords = adata.obsm['spatial']
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords_norm[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

    n_samples = len(expression)
    if n_samples > 500:
        print(f"Using a subset of 500 spots out of {n_samples} for computational efficiency")
        np.random.seed(42)
        subset_idx = np.random.choice(n_samples, size=500, replace=False)
        expression_subset = expression[subset_idx]
        coords_subset = coords_norm[subset_idx]
        D_subset = squareform(pdist(coords_subset))
        model_expression = expression_subset
        model_coords = coords_subset
        model_D = D_subset
    else:
        model_expression = expression
        model_coords = coords_norm
        model_D = squareform(pdist(coords_norm))


    def exponential_cov(D, theta):
        """
        Exponential covariance function.

        Parameters:
        -----------
        D : array
            Distance matrix.
        theta : array
            Parameters [amplitude, length_scale, noise]

        Returns:
        --------
        Covariance matrix
        """
        amplitude, length_scale, noise = theta
        return amplitude**2 * np.exp(-D / length_scale) + noise**2 * np.eye(D.shape[0])


    def neg_log_likelihood(theta, D, y):
        """
        Negative log marginal likelihood for Gaussian process.

        Parameters:
        -----------
        theta : array
            Parameters [amplitude, length_scale, noise]
        D : array
            Distance matrix
        y : array
            Observed values

        Returns:
        --------
        Negative log marginal likelihood
        """
        K = exponential_cov(D, theta)

        try:
            L = cholesky(K, lower=True)
            alpha = cho_solve((L, True), y)

            log_det_K = 2 * np.sum(np.log(np.diag(L)))

            return 0.5 * (np.dot(y, alpha) + log_det_K + D.shape[0] * np.log(2 * np.pi))
        except np.linalg.LinAlgError:

            return 1e10

    initial_theta = [1.0, 0.5, 0.1]
    bounds = [(1e-5, 10.0), (1e-5, 2.0), (1e-5, 2.0)]

    print("Optimizing spatial parameters...")
    result = minimize(
        neg_log_likelihood,
        initial_theta,
        args=(model_D, model_expression - np.mean(model_expression)),
        method='L-BFGS-B',
        bounds=bounds
    )
    opt_amplitude, opt_length_scale, opt_noise = result.x
    print(f"Optimized parameters: amplitude={opt_amplitude:.3f}, length_scale={opt_length_scale:.3f}, noise={opt_noise:.3f}")

    def predict(X_new, X_train, y_train, theta):
        """
        Make predictions at new points using the GP posterior.

        Parameters:
        -----------
        X_new : array
            New coordinates to predict at
        X_train : array
            Training coordinates
        y_train : array
            Training values
        theta : array
            Parameters [amplitude, length_scale, noise]

        Returns:
        --------
        mean : array
            Posterior mean
        var : array
            Posterior variance
        """


        D_train = squareform(pdist(X_train))
        K = exponential_cov(D_train, theta)
        L = cholesky(K, lower=True)
        D_new_train = np.zeros((X_new.shape[0], X_train.shape[0]))
        for i in range(X_new.shape[0]):
            for j in range(X_train.shape[0]):
                D_new_train[i, j] = np.sqrt(np.sum((X_new[i] - X_train[j])**2))
        Ks = theta[0]**2 * np.exp(-D_new_train / theta[1])
        alpha = cho_solve((L, True), y_train)
        mu = np.dot(Ks, alpha)
        v = cho_solve((L, True), Ks.T)
        var = theta[0]**2 - np.sum(Ks * v.T, axis=1)
        return mu, var


    n_grid = 50
    x_grid = np.linspace(0, 1, n_grid)
    y_grid = np.linspace(0, 1, n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([X_grid.flatten(), Y_grid.flatten()])


    print("Making predictions on a grid...")
    pred_mean, pred_var = predict(
        grid_coords,
        model_coords,
        model_expression - np.mean(model_expression),
        [opt_amplitude, opt_length_scale, opt_noise]
    )


    pred_mean += np.mean(model_expression)
    pred_std = np.sqrt(pred_var)
    pred_mean_grid = pred_mean.reshape(n_grid, n_grid)
    pred_std_grid = pred_std.reshape(n_grid, n_grid)
    X_grid_orig = X_grid * (coords[:, 0].max() - coords[:, 0].min()) + coords[:, 0].min()
    Y_grid_orig = Y_grid * (coords[:, 1].max() - coords[:, 1].min()) + coords[:, 1].min()
    fig = plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(sc, label='Expression')
    plt.title(f'Original {gene} Expression Data', fontsize=14)
    plt.xlabel('Spatial X', fontsize=12)
    plt.ylabel('Spatial Y', fontsize=12)



    plt.subplot(2, 2, 2)
    im = plt.pcolormesh(X_grid_orig, Y_grid_orig, pred_mean_grid, cmap='viridis', shading='auto')
    plt.colorbar(im, label='Predicted Expression')
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=10, alpha=0.1)
    plt.title(f'Spatial Estimate of {gene}', fontsize=14)
    plt.xlabel('Spatial X', fontsize=12)
    plt.ylabel('Spatial Y', fontsize=12)


    plt.subplot(2, 2, 3)
    im = plt.pcolormesh(X_grid_orig, Y_grid_orig, pred_std_grid, cmap='magma', shading='auto')
    plt.colorbar(im, label='Standard Deviation')
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=10, alpha=0.1)
    plt.title('Uncertainty (Standard Deviation)', fontsize=14)
    plt.xlabel('Spatial X', fontsize=12)
    plt.ylabel('Spatial Y', fontsize=12)


    plt.subplot(2, 2, 4)
    cv_grid = pred_std_grid / (pred_mean_grid + 1e-10)
    im = plt.pcolormesh(X_grid_orig, Y_grid_orig, cv_grid, cmap='magma', shading='auto')
    plt.colorbar(im, label='Coefficient of Variation')
    plt.title('Relative Uncertainty (CV = σ/μ)', fontsize=14)
    plt.xlabel('Spatial X', fontsize=12)
    plt.ylabel('Spatial Y', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{gene}_bayesian_spatial_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


    if cluster_key in adata.obs.columns:
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(coords[:, 0], coords[:, 1],
                         c=adata.obs[cluster_key].astype('category').cat.codes,
                         cmap='tab20', s=30, alpha=0.8)
        plt.colorbar(sc, label='Cluster')
        plt.title(f'Spatial Clusters ({cluster_key}) vs {gene} Expression Pattern', fontsize=14)

        contour = plt.contour(X_grid_orig, Y_grid_orig, pred_mean_grid,
                             colors='black', alpha=0.7, levels=5)
        plt.clabel(contour, inline=True, fontsize=10)
        plt.xlabel('Spatial X', fontsize=12)
        plt.ylabel('Spatial Y', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{gene}_clusters_with_expression_contours.png", dpi=300, bbox_inches='tight')
        plt.show()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    high_expr_low_unc = (pred_mean_grid > np.median(pred_mean_grid)) & (pred_std_grid < np.median(pred_std_grid))
    plt.pcolormesh(X_grid_orig, Y_grid_orig, high_expr_low_unc, cmap='Blues', shading='auto')
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=10, alpha=0.1)
    plt.title(f'Regions of High {gene} Expression\nwith High Certainty', fontsize=12)
    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')


    plt.subplot(1, 2, 2)
    high_uncertainty = pred_std_grid > np.percentile(pred_std_grid, 75)
    plt.pcolormesh(X_grid_orig, Y_grid_orig, high_uncertainty, cmap='Reds', shading='auto')
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=10, alpha=0.1)
    plt.title('Regions Requiring Further\nInvestigation (High Uncertainty)', fontsize=12)
    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')

    plt.tight_layout()
    plt.savefig(f"{gene}_regions_of_interest.png", dpi=300, bbox_inches='tight')
    plt.show()
    ls_physical = opt_length_scale * (coords[:, 0].max() - coords[:, 0].min())

    print(f"\nKey findings from Bayesian-inspired Spatial Analysis of {gene}:")
    print(f"1. Spatial correlation length scale: {ls_physical:.2f} units")
    print(f"   (This means expression values are correlated up to ~{ls_physical:.2f} distance units apart)")
    print(f"2. Signal-to-noise ratio: {opt_amplitude / opt_noise:.2f}")
    print(f"   (Higher values indicate stronger spatial pattern relative to noise)")
    print(f"3. We identified {np.sum(high_expr_low_unc)}/{high_expr_low_unc.size} regions ({np.sum(high_expr_low_unc)*100/high_expr_low_unc.size:.1f}%) with high expression and high certainty")
    print(f"4. We identified {np.sum(high_uncertainty)}/{high_uncertainty.size} regions ({np.sum(high_uncertainty)*100/high_uncertainty.size:.1f}%) with high uncertainty that require further investigation")

    print("\nSpatial analysis completed successfully!")

except Exception as e:
    print(f"Spatial analysis failed: {e}")
    import traceback
    traceback.print_exc()

    try:
        print("\nTrying a very simplified spatial correlation analysis...")

        if 'moranI' in adata.uns:
            gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
        else:
            gene_var = np.var(adata.X.toarray(), axis=0)
            top_var_idx = np.argsort(gene_var)[-1]
            gene = adata.var_names[top_var_idx]

        gene_idx = adata.var_names.get_loc(gene)
        expression = adata.X[:, gene_idx].toarray().flatten()
        coords = adata.obsm['spatial']
        distances = squareform(pdist(coords))
        max_dist = np.percentile(distances, 95)  # 95th percentile for avoid outliers
        bins = np.linspace(0, max_dist, 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        corr_by_distance = []

        for i in range(len(bins) - 1):
            mask = (distances > bins[i]) & (distances <= bins[i+1])
            if np.sum(mask) < 10:
                corr_by_distance.append(np.nan)
                continue
            corrs = []
            for j in range(len(expression)):
                for k in range(j+1, len(expression)):
                    if mask[j, k]:
                        similarity = 1 - abs(expression[j] - expression[k]) / (max(expression) - min(expression))
                        corrs.append(similarity)
            corr_by_distance.append(np.mean(corrs) if corrs else np.nan)

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, corr_by_distance, 'o-', markersize=8)
        plt.xlabel('Distance', fontsize=12)
        plt.ylabel('Spatial Correlation', fontsize=12)
        plt.title(f'Spatial Correlation of {gene} by Distance', fontsize=14)
        plt.grid(True, alpha=0.3)

        valid_idx = ~np.isnan(corr_by_distance)
        if sum(valid_idx) >= 3:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            try:
                popt, _ = curve_fit(
                    exp_decay,
                    bin_centers[valid_idx],
                    np.array(corr_by_distance)[valid_idx],
                    bounds=([0, 0, -1], [2, 10, 1])
                )
                x_fit = np.linspace(0, max_dist, 100)
                y_fit = exp_decay(x_fit, *popt)
                plt.plot(x_fit, y_fit, 'r--', label=f'Exp. decay (scale={1/popt[1]:.2f})')
                plt.legend()

                corr_length = 1/popt[1]
                plt.axvline(x=corr_length, color='gray', linestyle='--', alpha=0.5)
                plt.text(corr_length, 0.5, f'Correlation length: {corr_length:.2f}',
                        rotation=90, verticalalignment='center')
            except:
                print("Could not fit exponential decay curve")

        plt.tight_layout()
        plt.savefig(f"{gene}_spatial_correlation.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=expression, cmap='viridis',
                       s=50, edgecolor='black', linewidth=0.5)
        plt.colorbar(sc, label=f'{gene} Expression')
        plt.title(f'Spatial Pattern of {gene} Expression', fontsize=14)
        plt.xlabel('Spatial X', fontsize=12)
        plt.ylabel('Spatial Y', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{gene}_expression_pattern.png", dpi=300)
        plt.show()

        print("\nSimplified spatial analysis completed successfully!")
    except Exception as e2:
        print(f"Even simplified analysis failed: {e2}")
        print("Please try running basic squidpy spatial analysis instead.")

"""# **Spatial Smoothing Analysis**"""

# Spatial Smoothing Analysis
import matplotlib as mpl

try:
    if 'moranI' in adata.uns:
        gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
    else:
        gene_var = np.var(adata.X.toarray(), axis=0)
        top_var_idx = np.argsort(gene_var)[-1]
        gene = adata.var_names[top_var_idx]
    print(f"Analyzing spatial pattern for gene: {gene}")

    gene_idx = adata.var_names.get_loc(gene)
    expression = adata.X[:, gene_idx].toarray().flatten()
    coords = adata.obsm['spatial']

    from scipy.spatial.distance import pdist, squareform

    def smooth_spatial(coords, values, sigma=1.5):
        """Apply Gaussian smoothing to spatial data."""
        # Calculate pairwise distances
        dists = squareform(pdist(coords))
        # Calculate Gaussian weights
        weights = np.exp(-(dists**2) / (2 * sigma**2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        # Apply smoothing
        smoothed = weights @ values
        return smoothed

    sigmas = [0.5, 1.5, 3.0]
    smoothed_expressions = []

    for sigma in sigmas:
        smoothed = smooth_spatial(coords, expression, sigma=sigma)
        smoothed_expressions.append(smoothed)
    print("Visualizing results...")

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Verdana']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=expression,
                     cmap='viridis', s=30, alpha=0.8)
    axes[0].set_title(f"Original Expression: {gene}", fontsize=14)
    axes[0].set_xlabel("Spatial X", fontsize=12)
    axes[0].set_ylabel("Spatial Y", fontsize=12)
    fig.colorbar(sc, ax=axes[0])

    for i, (sigma, smoothed) in enumerate(zip(sigmas, smoothed_expressions)):
        sc = axes[i+1].scatter(coords[:, 0], coords[:, 1], c=smoothed,
                           cmap='viridis', s=30, alpha=0.8)
        axes[i+1].set_title(f"Smoothed (sigma={sigma})", fontsize=14)
        axes[i+1].set_xlabel("Spatial X", fontsize=12)
        axes[i+1].set_ylabel("Spatial Y", fontsize=12)
        fig.colorbar(sc, ax=axes[i+1])
    plt.tight_layout()
    plt.show()

    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import griddata

    grid_x = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    grid_y = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata(coords, smoothed_expressions[1],
                     (grid_x, grid_y), method='linear')

    from scipy.ndimage import gaussian_filter, sobel

    grid_z_smooth = np.copy(grid_z)
    if np.any(np.isnan(grid_z_smooth)):
        mask = np.isnan(grid_z_smooth)
        grid_z_smooth[mask] = griddata(
            (grid_x[~mask].ravel(), grid_y[~mask].ravel()),
            grid_z_smooth[~mask].ravel(),
            (grid_x[mask].ravel(), grid_y[mask].ravel()),
            method='nearest'
        )

    grid_z_smooth = gaussian_filter(grid_z_smooth, sigma=1.0)
    grad_x = sobel(grid_z_smooth, axis=1)
    grad_y = sobel(grid_z_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(grid_x, grid_y, grid_z_smooth, cmap='viridis', shading='auto')
    plt.colorbar(label='Smoothed Expression')
    plt.title(f'Smoothed {gene} Expression')

    plt.subplot(2, 2, 2)
    plt.pcolormesh(grid_x, grid_y, grad_mag, cmap='magma', shading='auto')
    plt.colorbar(label='Gradient Magnitude')
    plt.title(f'Expression Gradient Magnitude')

    plt.subplot(2, 2, 3)
    if cluster_key in adata.obs.columns:
        sc = plt.scatter(coords[:, 0], coords[:, 1],
                       c=adata.obs[cluster_key].astype('category').cat.codes,
                       cmap='tab20', s=30, alpha=0.8)
        plt.colorbar(sc, label='Cluster')
        plt.title(f'Spatial Clusters ({cluster_key})')
    else:
        sc = plt.scatter(coords[:, 0], coords[:, 1],
                       c=expression, cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(sc, label='Expression')
        plt.title(f'Original {gene} Expression')

    plt.subplot(2, 2, 4)
    plt.hist(expression, bins=30, alpha=0.8)
    plt.axvline(x=np.mean(expression), color='red', linestyle='--',
               label=f'Mean: {np.mean(expression):.2f}')
    plt.title(f'Distribution of {gene} Expression')
    plt.xlabel('Expression Level')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    try:
        fig.savefig(f"{gene}_smoothing_analysis.png", dpi=300, bbox_inches='tight')
        plt.gcf().savefig(f"{gene}_gradient_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved figures to {gene}_smoothing_analysis.png and {gene}_gradient_analysis.png")
    except Exception as save_error:
        print(f"Could not save figures: {save_error}")
    print("Spatial smoothing analysis completed successfully")
except Exception as e:
    print(f"Spatial smoothing analysis failed: {e}")
    import traceback
    traceback.print_exc()

    try:

        if 'moranI' in adata.uns:
            top_genes = adata.uns['moranI'].sort_values('I', ascending=False).index[:3].tolist()
        else:
            gene_var = np.var(adata.X.toarray(), axis=0)
            top_var_idx = np.argsort(gene_var)[-3:]
            top_genes = adata.var_names[top_var_idx].tolist()

        sq.pl.spatial_scatter(adata, color=top_genes)

        print("Simple visualization completed")
    except Exception as e2:
        print(f"Even simple visualization failed: {e2}")

"""## **Topological Data Analysis**"""

# TDA using ripser
print("\n=== Topological Data Analysis (TDA) with Ripser ===")

try:
    from ripser import ripser
    from persim import plot_diagrams
    import matplotlib.pyplot as plt

    X = adata.obsm['spatial']

    print("Computing persistent homology...")
    results = ripser(X, maxdim=1)
    plot_diagrams(results['dgms'], title="Persistence Diagram of Spatial Structure")
    plt.show()

    if cluster_key in adata.obs.columns:
        for cluster in list(adata.obs[cluster_key].cat.categories)[:3]:
            mask = adata.obs[cluster_key] == cluster
            if sum(mask) >= 10:
                X_cluster = adata.obsm['spatial'][mask]

                print(f"Computing persistent homology for cluster {cluster}...")
                results_cluster = ripser(X_cluster, maxdim=1)

                plot_diagrams(results_cluster['dgms'],
                             title=f"Persistence Diagram for Cluster {cluster}")
                plt.show()
    print("Topological data analysis completed successfully")
except ImportError:
    print("Could not install ripser. For TDA, manually install ripser or gudhi.")
except Exception as e:
    print(f"Error in topological analysis: {e}")

print("\n=== Topological Features Analysis ===")

try:
    from scipy.spatial import distance
    from scipy.cluster.hierarchy import linkage, fcluster

    X = adata.obsm['spatial']
    dist_matrix = distance.pdist(X)
    Z = linkage(dist_matrix, method='single')
    thresholds = np.linspace(0, np.max(dist_matrix) * 0.5, 50)
    n_components = []

    for threshold in thresholds:
        clusters = fcluster(Z, threshold, criterion='distance')
        n_components.append(len(np.unique(clusters)))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, n_components)
    plt.xlabel('Distance Threshold')
    plt.ylabel('Number of Connected Components')
    plt.title('Topological "Barcode" - Connected Components')
    plt.grid(True)
    plt.show()

    interesting_thresholds = [
        thresholds[len(thresholds)//5],
        thresholds[len(thresholds)//2],
        thresholds[len(thresholds)*3//4],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, threshold in enumerate(interesting_thresholds):
        clusters = fcluster(Z, threshold, criterion='distance')

        axes[i].scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab20', s=20)
        axes[i].set_title(f'Clusters at threshold = {threshold:.2f}\n({len(np.unique(clusters))} components)')
        axes[i].set_xlabel('Spatial X')
        axes[i].set_ylabel('Spatial Y')
    plt.tight_layout()
    plt.show()
    print("Simple topological analysis completed!")
except Exception as e:
    print(f"Error in simple topological analysis: {e}")

"""# 8. Marker Gene Identification

**8.1 Finding Cluster Marker Genes**
"""

#reimport
import scanpy as sc
import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt

cluster_key = 'spatial_domains' if 'spatial_domains' in adata.obs.columns else 'leiden'
if cluster_key not in adata.obs.columns:
    cluster_key = 'kmeans'
try:
    sc.tl.rank_genes_groups(adata, cluster_key, method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby=cluster_key,
                                    dendrogram=True, swap_axes=True)
    marker_genes = {}
    for i in adata.obs[cluster_key].cat.categories:
        markers = sc.get.rank_genes_groups_df(adata, group=i, key='rank_genes_groups')
        marker_genes[f"Cluster {i}"] = markers['names'][:5].tolist()
    print("Top marker genes by cluster:")
    for cluster, genes in marker_genes.items():
        print(f"{cluster}: {', '.join(genes)}")

    top_markers = []
    for cluster_markers in marker_genes.values():
        top_markers.extend(cluster_markers[:2])  # Top 2 from each cluster
    top_markers = list(dict.fromkeys(top_markers))
    sq.pl.spatial_scatter(adata, color=top_markers[:9], ncols=3)

except Exception as e:
    print(f"Error finding marker genes: {e}")

"""**8.2 Advanced Marker Gene Analysis**"""

try:
    if cluster_key in adata.obs.columns and 'marker_genes' in locals():
        top_markers_flat = []
        if isinstance(marker_genes, dict):
            for genes_list in marker_genes.values():
                if isinstance(genes_list, list):
                    top_markers_flat.extend(genes_list[:5])
        elif isinstance(marker_genes, list):
            top_markers_flat = marker_genes[:30]  # up to 30 genes
        else:
            print(f"Unexpected marker_genes type: {type(marker_genes)}")
        if not top_markers_flat and 'rank_genes_groups' in adata.uns:
            try:
                for group in adata.obs[cluster_key].cat.categories:
                    markers = sc.get.rank_genes_groups_df(adata, group=group)
                    if 'names' in markers.columns:
                        top_markers_flat.extend(markers['names'][:3].tolist())
            except Exception as err:
                print(f"Could not extract markers from rank_genes_groups: {err}")
        if not top_markers_flat:
            if 'highly_variable' in adata.var.columns:
                top_markers_flat = adata.var_names[adata.var.highly_variable][:20].tolist()
            else:
                mean_expr = adata.X.mean(axis=0)
                top_expr_idx = np.argsort(mean_expr)[-20:]
                top_markers_flat = adata.var_names[top_expr_idx].tolist()

            print(f"Using {len(top_markers_flat)} genes based on {'highly variable genes' if 'highly_variable' in adata.var.columns else 'mean expression'}")
        top_markers_flat = list(dict.fromkeys(top_markers_flat))
        top_markers_flat = top_markers_flat[:30]
        valid_markers = [gene for gene in top_markers_flat if gene in adata.var_names]

        if valid_markers:
            print(f"Using {len(valid_markers)} marker genes for visualization")
            sc.pl.dotplot(adata, valid_markers, groupby=cluster_key)
            if len(valid_markers) >= 5:
                sc.pl.violin(adata, valid_markers[:5], groupby=cluster_key, rotation=90)
            valid_markers_for_heatmap = valid_markers[:10]
            avg_exp = pd.DataFrame(index=adata.obs[cluster_key].cat.categories,
                                 columns=valid_markers_for_heatmap)
            for cluster in avg_exp.index:
                cluster_spots = adata[adata.obs[cluster_key] == cluster]
                for gene in valid_markers_for_heatmap:
                    try:
                        gene_idx = adata.var_names.get_loc(gene)
                        values = cluster_spots.X[:, gene_idx].toarray().flatten()
                        avg_exp.loc[cluster, gene] = float(np.mean(values))
                    except Exception as err:
                        print(f"Couldn't calculate mean for {gene} in cluster {cluster}: {err}")
                        avg_exp.loc[cluster, gene] = np.nan
            avg_exp = avg_exp.astype(float)
            if not avg_exp.isnull().all().all():
                # Visualize as a heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(avg_exp, cmap='viridis', annot=True, fmt=".2f")
                plt.title(f"Average Expression of Top Markers Across {cluster_key} Clusters")
                plt.tight_layout()
                plt.show()
            else:
                print("No valid data for heatmap visualization")
        else:
            print("No valid marker genes found in the dataset")
    else:
        print(f"Missing requirements: cluster_key in obs columns: {cluster_key in adata.obs.columns}, marker_genes exists: {'marker_genes' in locals()}")
except Exception as e:
    print(f"Error in advanced marker analysis: {e}")
    traceback.print_exc()

"""# 9. Cell Type Annotation & Deconvolution

**9.1 Manual Cell Type Annotation Based on Markers**
"""

cell_type_markers = {
    'B_cells': ['MS4A1', 'CD19', 'CD79A', 'CD79B', 'BANK1'],
    'T_cells': ['CD3D', 'CD3E', 'CD8A', 'CD4', 'IL7R'],
    'Plasma_cells': ['MZB1', 'JCHAIN', 'IGHA1', 'IGHA2', 'IGHG1'],
    'Macrophages': ['CD68', 'CD163', 'MARCO', 'SIGLEC1'],
    'Dendritic_cells': ['FCER1A', 'CD1C', 'CD1A', 'CLEC10A'],
    'Endothelial_cells': ['PECAM1', 'VWF', 'CDH5', 'CLDN5'],
    'Fibroblasts': ['DCN', 'COL1A1', 'COL3A1', 'COL6A1']
}

available_markers = {}
for cell_type, markers in cell_type_markers.items():
    available = [m for m in markers if m in adata.var_names]
    if available:
        available_markers[cell_type] = available

print("Available cell type markers:")
for cell_type, markers in available_markers.items():
    print(f"{cell_type}: {', '.join(markers)}")
for cell_type, markers in available_markers.items():
    if markers:
        print(f"\nVisualizing {cell_type} markers:")
        sq.pl.spatial_scatter(adata, color=markers[:3])

"""**9.2 Cell Type Deconvolution**"""

from sklearn.decomposition import NMF

def normalize_for_deconv(X):
    """Min-max normalize data for deconvolution."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    return (X - X_min) / X_range

all_markers = []
cell_types = []
for cell_type, markers in available_markers.items():
    if len(markers) >= 3:  # cell types with at least 3 markers
        all_markers.extend(markers)
        cell_types.append(cell_type)

if len(all_markers) > 10:
    marker_expr = adata[:, all_markers].X.toarray()
    marker_expr_norm = normalize_for_deconv(marker_expr)
    n_cell_types = len(cell_types)
    nmf = NMF(n_components=n_cell_types, random_state=0)
    W = nmf.fit_transform(marker_expr_norm)  # Spot x CellType
    proportions = W / W.sum(axis=1, keepdims=True)

    for i, cell_type in enumerate(cell_types):
        adata.obs[f'prop_{cell_type}'] = proportions[:, i]

    proportion_columns = [col for col in adata.obs.columns if col.startswith('prop_')]
    sq.pl.spatial_scatter(adata, library_id="V1_Human_Lymph_Node",
                         color=proportion_columns, ncols=3)
    if cluster_key in adata.obs.columns:
        cluster_props = adata.obs.groupby(cluster_key)[proportion_columns].mean()

        plt.figure(figsize=(12, 6))
        cluster_props.plot(kind='bar', stacked=True, colormap='tab10')
        plt.title(f'Average Cell Type Proportions by {cluster_key}')
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.legend(title='Cell Type')
        plt.tight_layout()
        plt.show()

"""## **Advanced Deconvolution with Cell2location**"""

print("\n=== Deconvolution with Cell2location ===")

'''
import cell2location

# Load reference scRNA-seq data with cell type annotations
ref_adata = sc.read_h5ad("reference_lymph_node_data.h5ad")

# Prepare reference dataset
# This learns molecular signatures of cell types
mod = cell2location.models.RegressionModel(ref_adata)
mod.train(max_epochs=250)

# Prepare spatial data
spatial_data = adata.copy()

# Fit the spatial model to decompose expression into cell types
cell2loc_mod = cell2location.models.Cell2location(
    spatial_data, mod.summary_stats,
    N_cells_per_location=30
)
cell2loc_mod.train(max_epochs=30000)

# Add cell abundance estimates to the spatial data
cell_abund = cell2loc_mod.export_posterior()
adata.obsm['cell2location'] = cell_abund

# Visualize cell type abundance
for cell_type in adata.obsm['cell2location'].columns:
    adata.obs[f'c2l_{cell_type}'] = adata.obsm['cell2location'][cell_type]

sq.pl.spatial_scatter(adata, color=[col for col in adata.obs.columns if col.startswith('c2l_')])
'''

print("Cell2location requires a reference scRNA-seq dataset and specific installation.")
print("Source: https://cell2location.readthedocs.io/")

"""# 10. Ligand-Receptor & Cell-Cell Communication

**10.1 Basic Ligand-Receptor Analysis**
"""

import difflib
print(difflib.get_close_matches('spatial_domains_ligrec', adata.uns.keys(), n=5))

import squidpy as sq

# Optional: subset genes to reduce memory/time
sq.gr.ligrec(adata, cluster_key='spatial_domains', n_perms=100, use_raw=False)

print('spatial_domains_ligrec' in adata.uns)

try:
    expected_key = f'{cluster_key}_ligrec'
    if expected_key in adata.uns:
        print(f"Using existing ligand-receptor results in adata.uns['{expected_key}']")
        categories = adata.obs[cluster_key].cat.categories
        source_groups = categories[:2] if len(categories) >= 2 else [categories[0]]
        target_groups = categories[2:4] if len(categories) >= 4 else categories[:2]
        try:
            print("\nVisualizing interaction scores between clusters:")
            sq.pl.ligrec(adata, cluster_key=cluster_key,
                       source_groups=source_groups,
                       target_groups=target_groups)
        except Exception as viz_error:
            print(f"Standard visualization failed: {viz_error}")
            print("Creating custom visualization instead...")
            try:
                interactions = adata.uns[expected_key]['means']
                pvals = adata.uns[expected_key]['pvals']
                interaction_data = []
                for source in source_groups:
                    for target in target_groups:
                        if (source, target) in interactions:
                            for ligand_receptor, score in interactions[(source, target)].items():
                                ligand, receptor = ligand_receptor.split('_')
                                pval = pvals[(source, target)].get(ligand_receptor, 1.0)

                                interaction_data.append({
                                    'source': source,
                                    'target': target,
                                    'ligand': ligand,
                                    'receptor': receptor,
                                    'pair': f"{ligand}-{receptor}",
                                    'score': score,
                                    'p_value': pval,
                                    'significance': -np.log10(pval + 1e-10),  # -log10(p) for plotting
                                    'interaction': f"{source} → {target}"
                                })

                if interaction_data:
                    interactions_df = pd.DataFrame(interaction_data)
                    sig_interactions = interactions_df[interactions_df['p_value'] < 0.05].sort_values('score', ascending=False)
                    if len(sig_interactions) > 0:
                        top_n = min(15, len(sig_interactions))
                        top_interactions = sig_interactions.head(top_n)

                        plt.figure(figsize=(12, 8))
                        ax = sns.barplot(data=top_interactions, y='pair', x='score', hue='interaction')
                        plt.title(f'Top {top_n} Significant Ligand-Receptor Interactions', fontsize=14)
                        plt.ylabel('Ligand-Receptor Pair', fontsize=12)
                        plt.xlabel('Interaction Score', fontsize=12)
                        plt.tight_layout()
                        plt.show()

                        if len(source_groups) > 1 or len(target_groups) > 1:
                            top_pairs = top_interactions['pair'].unique()[:10]
                            pairs_data = interactions_df[interactions_df['pair'].isin(top_pairs)]

                            pivot_data = pairs_data.pivot_table(
                                index='pair',
                                columns='interaction',
                                values='score',
                                aggfunc='mean'
                            ).fillna(0)

                            plt.figure(figsize=(12, 8))
                            sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.2f')
                            plt.title('Ligand-Receptor Interaction Scores by Cluster Pairs', fontsize=14)
                            plt.tight_layout()
                            plt.show()

                    print("\nTop ligand-receptor interactions:")
                    display_cols = ['source', 'target', 'ligand', 'receptor', 'score', 'p_value']
                    print(sig_interactions[display_cols].head(15).to_string(index=False))

                    top_genes = []
                    for _, row in top_interactions.head(5).iterrows():
                        if row['ligand'] in adata.var_names and row['ligand'] not in top_genes:
                            top_genes.append(row['ligand'])
                        if row['receptor'] in adata.var_names and row['receptor'] not in top_genes:
                            top_genes.append(row['receptor'])
                    if top_genes:
                        print("\nVisualizing expression of top interaction genes:")
                        sq.pl.spatial_scatter(adata, color=top_genes[:6], ncols=3)
                else:
                    print("No significant interactions found in the data.")
            except Exception as custom_viz_error:
                print(f"Custom visualization failed: {custom_viz_error}")
                print("\nRaw ligand-receptor data structure:")
                for key, value in adata.uns[expected_key].items():
                    print(f"- {key}: {type(value)}")
                    if isinstance(value, dict) and len(value) > 0:
                        first_key = next(iter(value))
                        print(f"  Example key: {first_key}")
                        print(f"  Example value type: {type(value[first_key])}")
                        if hasattr(value[first_key], 'keys'):
                            print(f"  Example subkeys: {list(value[first_key].keys())[:5]}")
        print("\nLigand-receptor analysis completed")
    else:
        print(f"Expected output key '{expected_key}' not found in adata.uns.")
        print(f"Available keys in adata.uns: {list(adata.uns.keys())}")
except Exception as e:
    print(f"Visualization failed: {e}")
    import traceback
    traceback.print_exc()

try:
    expected_key = f'{cluster_key}_ligrec'
    if expected_key in adata.uns:
        print(f"\nAnalyzing ligand-receptor results from adata.uns['{expected_key}']")
        means_df = adata.uns[expected_key]['means']
        pvalues_df = adata.uns[expected_key]['pvalues']
        metadata_df = adata.uns[expected_key]['metadata']

        print(f"Found data with shape: means {means_df.shape}, pvalues {pvalues_df.shape}")
        print("\nPreview of interaction means:")
        print(means_df.head())
        print("\nPreview of p-values:")
        print(pvalues_df.head())

        if means_df.notna().any().any():
            plt.figure(figsize=(14, 10))
            heatmap_data = means_df.fillna(0)

            if heatmap_data.shape[0] > 30:
                row_means = heatmap_data.mean(axis=1)
                top_indices = row_means.nlargest(30).index
                heatmap_data = heatmap_data.loc[top_indices]
            sns.heatmap(heatmap_data, cmap='viridis', linewidths=.5)
            plt.title('Ligand-Receptor Interaction Scores (All Interactions)', fontsize=14)
            plt.tight_layout()
            plt.show()

            stacked = means_df.stack().reset_index()
            stacked.columns = ['ligand_receptor', 'cluster_pair', 'score']
            stacked = stacked.dropna()

            if not stacked.empty:
                top_interactions = stacked.sort_values('score', ascending=False).head(20)
                if pvalues_df is not None:
                    pvalues_stacked = pvalues_df.stack().reset_index()
                    pvalues_stacked.columns = ['ligand_receptor', 'cluster_pair', 'p_value']
                    top_interactions = top_interactions.merge(
                        pvalues_stacked,
                        on=['ligand_receptor', 'cluster_pair'],
                        how='left'
                    )
                top_interactions[['ligand', 'receptor']] = top_interactions['ligand_receptor'].str.split('_', expand=True)
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(data=top_interactions.head(15), y='ligand_receptor', x='score',
                                hue='cluster_pair' if len(top_interactions['cluster_pair'].unique()) < 8 else None)
                plt.title('Top 15 Ligand-Receptor Interactions by Score', fontsize=14)
                plt.ylabel('Ligand-Receptor Pair', fontsize=12)
                plt.xlabel('Interaction Score', fontsize=12)
                plt.tight_layout()
                plt.show()
                print("\nTop ligand-receptor interactions:")
                display_cols = ['ligand', 'receptor', 'cluster_pair', 'score']
                if 'p_value' in top_interactions.columns:
                    display_cols.append('p_value')
                print(top_interactions[display_cols].head(15).to_string(index=False))

                top_genes = set()
                for _, row in top_interactions.head(6).iterrows():
                    if row['ligand'] in adata.var_names:
                        top_genes.add(row['ligand'])
                    if row['receptor'] in adata.var_names:
                        top_genes.add(row['receptor'])

                if top_genes:
                    print(f"\nVisualizing expression of top {len(top_genes)} interaction genes:")
                    sq.pl.spatial_scatter(adata, color=list(top_genes), ncols=3)
            else:
                print("No non-null interactions found in the data.")
        else:
            print("No non-null values found in the interaction data.")


            lr_pairs = [
                ('CCL19', 'CCR7'),    # T cell chemotaxis
                ('CXCL13', 'CXCR5'),  # B cell organization
                ('CD40LG', 'CD40'),   # T-B cell interaction
                ('IL7', 'IL7R'),      # T cell survival
                ('CD274', 'PDCD1'),   # PD-L1/PD-1 checkpoint
                ('ICAM1', 'ITGAL'),   # Adhesion
                ('HLA-DRA', 'CD4')    # MHC II interaction
            ]

            available_genes = []
            for ligand, receptor in lr_pairs:
                if ligand in adata.var_names:
                    available_genes.append(ligand)
                if receptor in adata.var_names:
                    available_genes.append(receptor)

            if available_genes:
                print(f"\nVisualizing expression of {len(available_genes)} known LR genes:")
                sq.pl.spatial_scatter(adata, color=available_genes[:6], ncols=3)

        print("\nLigand-receptor analysis visualization completed")
    else:
        print(f"Expected output key '{expected_key}' not found in adata.uns.")
        print(f"Available keys in adata.uns: {list(adata.uns.keys())}")
except Exception as e:
    print(f"Visualization failed: {e}")
    import traceback
    traceback.print_exc()

try:
    expected_key = f'{cluster_key}_ligrec'
    if expected_key in adata.uns:
        print(f"\nAnalyzing ligand-receptor results from adata.uns['{expected_key}']")

        means_df = adata.uns[expected_key]['means']
        pvalues_df = adata.uns[expected_key].get('pvalues')
        metadata_df = adata.uns[expected_key].get('metadata')

        print(f"Data shapes: means {means_df.shape}")
        if pvalues_df is not None:
            print(f"p-values {pvalues_df.shape}")
        print("\nMeans DataFrame structure:")
        print(f"Columns: {means_df.columns.tolist()[:5]}...")
        print(f"Index: {means_df.index.tolist()[:5]}...")

        plt.figure(figsize=(14, 10))
        heatmap_data = means_df.fillna(0)
        if heatmap_data.shape[0] > 30:
            row_means = heatmap_data.mean(axis=1)
            top_indices = row_means.nlargest(30).index
            heatmap_data = heatmap_data.loc[top_indices]

        sns.heatmap(heatmap_data, cmap='viridis', linewidths=.5)
        plt.title('Ligand-Receptor Interaction Scores (All Interactions)', fontsize=14)
        plt.tight_layout()
        plt.show()

        if isinstance(means_df.index, pd.MultiIndex):
            means_reset = means_df.reset_index()
            print("\nReset DataFrame has columns:", means_reset.columns.tolist())

            if len(means_reset.columns) >= 3:
                id_vars = means_reset.columns[:2].tolist()
                value_vars = means_reset.columns[2:].tolist()

                melted = means_reset.melt(id_vars=id_vars, value_vars=value_vars,
                                         var_name='cluster_pair', value_name='score')
                melted = melted.dropna(subset=['score'])

                if not melted.empty:
                    top_interactions = melted.sort_values('score', ascending=False).head(20)
                    plt.figure(figsize=(12, 8))
                    top_interactions['interaction_label'] = top_interactions.apply(
                        lambda row: f"{row[id_vars[0]]}_{row[id_vars[1]]}", axis=1)

                    ax = sns.barplot(data=top_interactions.head(15), y='interaction_label', x='score')
                    plt.title('Top 15 Ligand-Receptor Interactions by Score', fontsize=14)
                    plt.ylabel('Interaction', fontsize=12)
                    plt.xlabel('Interaction Score', fontsize=12)
                    plt.tight_layout()
                    plt.show()

                    print("\nTop ligand-receptor interactions:")
                    display_cols = id_vars + ['cluster_pair', 'score']
                    print(top_interactions[display_cols].head(15).to_string(index=False))

                    potential_genes = set()

                    for col in id_vars:
                        potential_genes.update(top_interactions[col].unique())

                    valid_genes = [gene for gene in potential_genes if gene in adata.var_names]

                    if valid_genes:
                        print(f"\nVisualizing expression of {len(valid_genes)} interaction genes:")
                        sq.pl.spatial_scatter(adata, color=valid_genes[:6], ncols=3)
                    else:
                        print("No valid genes found for visualization.")
                else:
                    print("No non-null interactions found after melting.")
            else:
                print("DataFrame doesn't have enough columns for melting.")
        else:
            print("\nSimplified analysis for non-MultiIndex DataFrame")

            flat_values = means_df.values.flatten()
            non_nan = flat_values[~np.isnan(flat_values)]

            if len(non_nan) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(non_nan, bins=30)
                plt.title('Distribution of Interaction Scores')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                plt.show()

                known_genes = ['CD3E', 'CD19', 'CD4', 'CD8A', 'HLA-DRA', 'PTPRC']
                available_genes = [gene for gene in known_genes if gene in adata.var_names]

                if available_genes:
                    print(f"\nVisualizing expression of {len(available_genes)} common immune genes:")
                    sq.pl.spatial_scatter(adata, color=available_genes, ncols=3)

        print("\nLigand-receptor analysis visualization completed")
    else:
        print(f"Expected output key '{expected_key}' not found in adata.uns.")
        print(f"Available keys in adata.uns: {list(adata.uns.keys())}")
except Exception as e:
    print(f"Visualization failed: {e}")
    import traceback
    traceback.print_exc()

    try:
        known_genes = ['CD3E', 'CD19', 'CD4', 'CD8A', 'HLA-DRA', 'PTPRC',
                      'IL7R', 'CCR7', 'MS4A1', 'CD79A', 'CD68', 'CD14']
        available_genes = [gene for gene in known_genes if gene in adata.var_names]

        if available_genes:
            print(f"\nFallback: Visualizing expression of {len(available_genes)} common immune genes:")
            sq.pl.spatial_scatter(adata, color=available_genes[:6], ncols=3)
    except Exception as fallback_error:
        print(f"Even fallback visualization failed: {fallback_error}")

"""## **Advanced Cell-Cell Communication**"""

# CellPhoneDB or NicheNet-inspired spatial cell-cell communication
print("\n=== Cell-Cell Communication Analysis ===")

# Define receiver and sender clusters
receivers = adata.obs[cluster_key].cat.categories[:2]
senders = adata.obs[cluster_key].cat.categories[2:4]

lr_pairs = [
    ('CCL19', 'CCR7'),  # T cell chemotaxis
    ('CXCL13', 'CXCR5'), # B cell organization
    ('CD40LG', 'CD40'),  # T-B interaction
    ('ICAM1', 'ITGAL'),  # Adhesion
    ('PDCD1', 'CD274')   # Checkpoint
]

comm_scores = []

for sender in senders:
    for receiver in receivers:
        for ligand, receptor in lr_pairs:
            if ligand in adata.var_names and receptor in adata.var_names:
                sender_spots = adata[adata.obs[cluster_key] == sender]
                receiver_spots = adata[adata.obs[cluster_key] == receiver]
                ligand_exp = sender_spots[:, ligand].X.mean()
                receptor_exp = receiver_spots[:, receptor].X.mean()
                sender_coords = sender_spots.obsm['spatial']
                receiver_coords = receiver_spots.obsm['spatial']

                from scipy.spatial.distance import cdist
                mean_dist = cdist(sender_coords, receiver_coords).mean()
                proximity_factor = 1 / (1 + mean_dist/100)
                comm_score = ligand_exp * receptor_exp * proximity_factor

                comm_scores.append({
                    'sender': sender,
                    'receiver': receiver,
                    'ligand': ligand,
                    'receptor': receptor,
                    'ligand_exp': ligand_exp,
                    'receptor_exp': receptor_exp,
                    'mean_distance': mean_dist,
                    'comm_score': comm_score
                })

if comm_scores:
    comm_df = pd.DataFrame(comm_scores)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=comm_df.sort_values('comm_score', ascending=False).head(10),
               x='comm_score', y='ligand', hue='sender')
    plt.title('Top Ligand-Receptor Interactions by Communication Score')
    plt.tight_layout()
    plt.show()

"""# 11. Pathway Analysis

**11.1 Simple Pathway Enrichment**
"""

example_pathways = {
    'Inflammation': ['IL6', 'CXCL8', 'TNF', 'IL1B', 'CXCL10', 'IL1A', 'CCL2', 'PTGS2'],
    'T_cell_activation': ['CD3D', 'CD3E', 'CD3G', 'CD28', 'ICOS', 'IL2RA', 'LCK', 'ZAP70', 'ITK'],
    'B_cell_function': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'IGKC', 'IGHM', 'CD22', 'BANK1', 'BTK'],
    'Antigen_presentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'CD74', 'CIITA', 'HLA-A', 'HLA-B', 'HLA-C'],
    'Cytotoxicity': ['GZMA', 'GZMB', 'GZMH', 'PRF1', 'GNLY', 'NKG7', 'KLRD1', 'KLRB1'],
    'Interferon_response': ['STAT1', 'IRF1', 'IRF7', 'ISG15', 'MX1', 'OAS1', 'IFIT1', 'IFITM1'],
    'Cell_cycle': ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'CDK1', 'CCNB1', 'CCNA2', 'CCNE1']
}

def calculate_pathway_scores(adata, pathways):
    """Calculate average expression of pathway genes for each spot."""
    scores = {}
    for pathway, genes in pathways.items():
        available_genes = [gene for gene in genes if gene in adata.var_names]
        if len(available_genes) >= 3:
            adata.obs[f'pathway_{pathway}'] = adata[:, available_genes].X.mean(axis=1)
            scores[pathway] = available_genes

    return scores

pathway_scores = calculate_pathway_scores(adata, example_pathways)
print("Pathway coverage:")
for pathway, genes in pathway_scores.items():
    print(f"{pathway}: {len(genes)} genes - {', '.join(genes[:5])}...")

pathway_columns = [col for col in adata.obs.columns if col.startswith('pathway_')]
if pathway_columns:
    sq.pl.spatial_scatter(adata, color=pathway_columns, ncols=3)

    pathway_corr = adata.obs[pathway_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pathway_corr, annot=True, cmap='coolwarm')
    plt.title('Pathway Score Correlations')
    plt.tight_layout()
    plt.show()

"""**11.2 Advanced Gene Set Enrichment**"""

if cluster_key in adata.obs.columns and pathway_columns:
    enrichment_results = []

    for cluster in adata.obs[cluster_key].cat.categories:
        in_cluster = adata.obs[cluster_key] == cluster
        for pathway in pathway_columns:
            in_scores = adata.obs.loc[in_cluster, pathway]
            out_scores = adata.obs.loc[~in_cluster, pathway]
            mean_in = in_scores.mean()
            mean_out = out_scores.mean()
            fold_change = mean_in / mean_out if mean_out > 0 else float('inf')

            from scipy import stats
            t_stat, p_val = stats.ttest_ind(in_scores, out_scores)

            enrichment_results.append({
                'cluster': cluster,
                'pathway': pathway.replace('pathway_', ''),
                'mean_in_cluster': mean_in,
                'mean_out_cluster': mean_out,
                'fold_change': fold_change,
                't_statistic': t_stat,
                'p_value': p_val
            })

    enrichment_df = pd.DataFrame(enrichment_results)

    from statsmodels.stats.multitest import multipletests
    enrichment_df['p_adj'] = multipletests(enrichment_df['p_value'], method='fdr_bh')[1]
    sig_enrichment = enrichment_df[enrichment_df['p_adj'] < 0.05].sort_values('p_adj')

    print("\nSignificantly enriched pathways by cluster:")
    print(sig_enrichment[['cluster', 'pathway', 'fold_change', 'p_adj']].head(10))

    if not sig_enrichment.empty:
        pivot_df = sig_enrichment.pivot(index='pathway', columns='cluster', values='fold_change')
        pivot_df = pivot_df.fillna(1)
        pivot_df = np.log2(pivot_df)

        plt.figure(figsize=(12, len(pivot_df) * 0.5))
        sns.heatmap(pivot_df, cmap='coolwarm', center=0,
                    robust=True, annot=True, fmt='.2f')
        plt.title('Log2 Fold Enrichment of Pathways by Cluster')
        plt.tight_layout()
        plt.show()

"""# 12. Integration with External Datasets

## **Multi-Modal Integration**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color, exposure

print("\n=== Multi-modal Integration: Transcriptome + Histology ===")

if 'spatial' in adata.uns:
    print("'spatial' key found in adata.uns")
    if 'V1_Human_Lymph_Node' in adata.uns['spatial']:
        print("'V1_Human_Lymph_Node' found in adata.uns['spatial']")
        if 'images' in adata.uns['spatial']['V1_Human_Lymph_Node']:
            print("'images' key found under 'V1_Human_Lymph_Node'")
            img = adata.uns['spatial']['V1_Human_Lymph_Node']['images'].get('hires', None)

            if img is not None:
                print("Found high-resolution image")

                plt.imshow(img)
                plt.title("Original Histology Image")
                plt.axis('off')
                plt.show()
                img_gray = color.rgb2gray(img)
                img_eq = exposure.equalize_hist(img_gray)

                plt.imshow(img_eq, cmap='gray')
                plt.title("Equalized Grayscale Image")
                plt.axis('off')
                plt.show()

                spot_features = []
                scale = adata.uns['spatial']['V1_Human_Lymph_Node']['scalefactors']['spot_diameter_fullres']
                print(f"Using scale factor: {scale}")

                for spot_idx in range(adata.n_obs):
                    spot_coords = adata.obsm['spatial'][spot_idx]
                    x_pixel = int(spot_coords[0])
                    y_pixel = int(spot_coords[1])


                    if x_pixel < img.shape[1] and y_pixel < img.shape[0]:
                        patch_size = int(scale)
                        x_min = max(0, x_pixel - patch_size//2)
                        x_max = min(img.shape[1], x_pixel + patch_size//2)
                        y_min = max(0, y_pixel - patch_size//2)
                        y_max = min(img.shape[0], y_pixel + patch_size//2)
                        patch = img_gray[y_min:y_max, x_min:x_max]


                        print(f" Spot {spot_idx} | Patch shape: {patch.shape} | x=({x_min}, {x_max}), y=({y_min}, {y_max})")

                        if patch.size > 0:
                            mean_intensity = np.mean(patch)
                            variance = np.var(patch)
                            try:
                                from skimage.feature import graycomatrix, graycoprops
                                glcm = graycomatrix(np.uint8(patch * 255), distances=[1], angles=[0],
                                                    levels=256, symmetric=True, normed=True)
                                contrast = graycoprops(glcm, 'contrast')[0, 0]
                                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                                spot_features.append({
                                    'spot_idx': spot_idx,
                                    'mean_intensity': mean_intensity,
                                    'variance': variance,
                                    'contrast': contrast,
                                    'homogeneity': homogeneity
                                })
                            except Exception as e:
                                print(f"Error at spot {spot_idx}: {e}")
                                spot_features.append({
                                    'spot_idx': spot_idx,
                                    'mean_intensity': mean_intensity,
                                    'variance': variance
                                })
                        else:
                            print(f"Empty patch at spot {spot_idx} ({x_pixel}, {y_pixel})")

                if spot_features:
                    features_df = pd.DataFrame(spot_features).set_index('spot_idx')
                    print(f"Extracted features for {len(features_df)} spots")

                    for col in features_df.columns:
                        adata.obs[f'img_{col}'] = np.nan
                        for idx in features_df.index:
                            adata.obs.loc[adata.obs.index[idx], f'img_{col}'] = features_df.loc[idx, col]

                    img_feature_cols = [col for col in adata.obs.columns if col.startswith('img_')]
                    if img_feature_cols:
                        print(f" Visualizing spatial features: {img_feature_cols}")
                        import squidpy as sq
                        sq.pl.spatial_scatter(adata, color=img_feature_cols, ncols=2)

                    if 'moranI' in adata.uns:
                        top_spatial_gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
                        print(f" Top spatial gene: {top_spatial_gene}")
                        if top_spatial_gene in adata.var_names:
                            gene_exp = adata[:, top_spatial_gene].X.toarray().flatten()
                            corr_results = []
                            for feature in img_feature_cols:
                                mask = ~np.isnan(adata.obs[feature])
                                if np.sum(mask) > 10:
                                    from scipy.stats import pearsonr
                                    corr, pval = pearsonr(gene_exp[mask], adata.obs[feature][mask])
                                    corr_results.append({
                                        'feature': feature,
                                        'gene': top_spatial_gene,
                                        'correlation': corr,
                                        'p_value': pval
                                    })
                            if corr_results:
                                corr_df = pd.DataFrame(corr_results)
                                plt.figure(figsize=(10, 6))
                                sns.barplot(data=corr_df, x='correlation', y='feature')
                                plt.title(f'Correlation of Image Features with {top_spatial_gene} Expression')
                                plt.axvline(x=0, color='red', linestyle='--')
                                plt.tight_layout()
                                plt.show()
                else:
                    print("No features extracted from image patches.")
            else:
                print("High-resolution image not found")
        else:
            print("'images' key not found")
    else:
        print("'V1_Human_Lymph_Node' not found in adata.uns['spatial']")
else:
    print("'spatial' key not found in adata.uns")


plt.imshow(img_eq, cmap='gray')
plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], s=5, c='red')
plt.title("Spot positions over histology image")
plt.axis('off')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import color, exposure, transform

print("\n=== Corrected Multi-modal Integration: Transcriptome + Histology ===")

if 'spatial' in adata.uns and 'V1_Human_Lymph_Node' in adata.uns['spatial']:
    library_id = 'V1_Human_Lymph_Node'

    img = adata.uns['spatial'][library_id]['images'].get('hires', None)

    if img is None:
        img = adata.uns['spatial'][library_id]['images'].get('lowres', None)

    if img is not None:
        print(f"Found image with shape: {img.shape}")
        scale_factors = adata.uns['spatial'][library_id]['scalefactors']
        print(f"Scale factors: {scale_factors}")
        spot_diameter_fullres = scale_factors['spot_diameter_fullres']
        tissue_hires_scalef = scale_factors.get('tissue_hires_scalef', 1.0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        image_coords = adata.obsm['spatial'].copy()
        for i in range(image_coords.shape[0]):
            image_coords[i, 0] = image_coords[i, 0] * scale_factors.get('tissue_hires_scalef', 1.0)
            image_coords[i, 1] = image_coords[i, 1] * scale_factors.get('tissue_hires_scalef', 1.0)

        spot_size = spot_diameter_fullres * 0.5
        if 'leiden' in adata.obs:
            cluster_colors = adata.obs['leiden'].cat.codes
            scatter = ax.scatter(
                image_coords[:, 0],
                image_coords[:, 1],
                s=spot_size,
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5,
                c=cluster_colors,
                cmap='tab20'
            )
            from matplotlib.lines import Line2D
            categories = adata.obs['leiden'].cat.categories
            custom_lines = [Line2D([0], [0], color=plt.cm.tab20(i/len(categories)),
                                  marker='o', linestyle='None', markersize=8)
                           for i in range(len(categories))]
            ax.legend(custom_lines, categories, title='Clusters',
                     loc='upper right', bbox_to_anchor=(1.1, 1))
        else:
            scatter = ax.scatter(
                image_coords[:, 0],
                image_coords[:, 1],
                s=spot_size,
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5,
                c='red'
            )

        plt.title("Correctly Aligned Spots on Histology Image")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('aligned_spots_histology.png', dpi=300)
        plt.show()

        print("\nExtracting image features from spots...")
        spot_features = []

        for spot_idx in range(adata.n_obs):
            x_pixel = int(image_coords[spot_idx, 0])
            y_pixel = int(image_coords[spot_idx, 1])

            if 0 <= x_pixel < img.shape[1] and 0 <= y_pixel < img.shape[0]:
                patch_size = int(spot_diameter_fullres)
                x_min = max(0, x_pixel - patch_size//2)
                x_max = min(img.shape[1], x_pixel + patch_size//2)
                y_min = max(0, y_pixel - patch_size//2)
                y_max = min(img.shape[0], y_pixel + patch_size//2)

                if spot_idx % 500 == 0:
                    print(f"Processing spot {spot_idx}/{adata.n_obs}...")

                patch = img[y_min:y_max, x_min:x_max]

                if patch.size > 0:
                    if len(patch.shape) == 3:
                        patch_gray = color.rgb2gray(patch)
                    else:
                        patch_gray = patch
                    mean_intensity = np.mean(patch_gray)
                    variance = np.var(patch_gray)
                    spot_features.append({
                        'spot_idx': spot_idx,
                        'mean_intensity': mean_intensity,
                        'variance': variance
                    })

                    if min(patch.shape[:2]) >= 3:
                        try:
                            from skimage.feature import graycomatrix, graycoprops
                            patch_norm = (patch_gray * 255).astype(np.uint8)
                            glcm = graycomatrix(patch_norm, distances=[1], angles=[0],
                                             levels=256, symmetric=True, normed=True)
                            contrast = graycoprops(glcm, 'contrast')[0, 0]
                            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

                            spot_features[-1].update({
                                'contrast': contrast,
                                'homogeneity': homogeneity
                            })
                        except Exception as e:
                            if spot_idx % 500 == 0:
                                print(f"⚠️ Could not extract texture features for spot {spot_idx}: {e}")

        if spot_features:
            features_df = pd.DataFrame(spot_features)
            features_df = features_df.set_index('spot_idx')

            print(f"Extracted features for {len(features_df)} spots")

            for col in features_df.columns:
                feature_name = f'img_{col}'
                adata.obs[feature_name] = np.nan

                for idx in features_df.index:
                    if idx < len(adata.obs_names):
                        adata.obs.iloc[idx, adata.obs.columns.get_loc(feature_name)] = features_df.loc[idx, col]

            img_feature_cols = [col for col in adata.obs.columns if col.startswith('img_')]
            print(f"Image features added: {img_feature_cols}")

            import squidpy as sq
            sq.pl.spatial_scatter(adata, color=img_feature_cols, ncols=2, library_id=library_id)

            if 'moranI' in adata.uns:
                try:
                    top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).head(3).index.tolist()

                    for gene in top_spatial_genes:
                        if gene in adata.var_names:
                            gene_exp = adata[:, gene].X.toarray().flatten()
                            correlations = []
                            for feature in img_feature_cols:
                                mask = ~np.isnan(adata.obs[feature])
                                if sum(mask) > 10:
                                    from scipy.stats import spearmanr
                                    rho, pval = spearmanr(gene_exp[mask], adata.obs[feature][mask])
                                    correlations.append({
                                        'feature': feature,
                                        'gene': gene,
                                        'rho': rho,
                                        'p_value': pval
                                    })

                            if correlations:
                                corr_df = pd.DataFrame(correlations)
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(data=corr_df, x='rho', y='feature', ax=ax)
                                ax.set_title(f'Correlation between image features and {gene} expression')
                                ax.axvline(0, color='red', linestyle='--')
                                plt.tight_layout()
                                plt.show()
                except Exception as e:
                    print(f"Error in correlation analysis: {e}")
        else:
            print("No features extracted from image patches")
    else:
        print("Image not found in AnnData object")
else:
    print("Spatial information not found in AnnData object")

"""# 13. Tissue-Specific Analysis

**13.1 Lymph Node Region Identification**
"""

lymph_node_regions = {
    'B_cell_zone': ['MS4A1', 'CD19', 'CD79A', 'CD79B', 'CR2', 'FCER2'],  # Follicles
    'T_cell_zone': ['CD3D', 'CD3E', 'CD3G', 'CCR7', 'IL7R', 'LTB'],      # Paracortex
    'Medullary_cords': ['JCHAIN', 'IGHA1', 'IGHG1', 'MZB1', 'SDC1'],     # Plasma cells
    'Subcapsular_sinus': ['CD169', 'SIGLEC1', 'LYVE1', 'PDPN', 'STAB2'], # Macrophages
    'HEVs': ['PECAM1', 'VWF', 'CDH5', 'SELP', 'SELE', 'ICAM1']           # High endothelial venules
}

region_scores = {}
for region, markers in lymph_node_regions.items():
    available = [m for m in markers if m in adata.var_names]
    if len(available) >= 2:
        adata.obs[f'region_{region}'] = adata[:, available].X.mean(axis=1)
        region_scores[region] = available

print("Lymph node region marker coverage:")
for region, markers in region_scores.items():
    print(f"{region}: {len(markers)} markers - {', '.join(markers)}")

region_columns = [col for col in adata.obs.columns if col.startswith('region_')]
if region_columns:
    sq.pl.spatial_scatter(adata, color=region_columns, ncols=3)
    adata.obs['dominant_region'] = adata.obs[region_columns].idxmax(axis=1).str.replace('region_', '')
    sq.pl.spatial_scatter(adata, color='dominant_region', title="Dominant Lymph Node Regions")

"""**13.2 Custom Lymph Node Analysis**"""

if 'region_B_cell_zone' in adata.obs.columns:
    follicle_threshold = adata.obs['region_B_cell_zone'].quantile(0.75)
    follicle_spots = adata.obs['region_B_cell_zone'] > follicle_threshold
    follicle_coords = adata.obsm['spatial'][follicle_spots]

    from scipy.spatial.distance import cdist

    distances = cdist(adata.obsm['spatial'], follicle_coords)
    min_distances = distances.min(axis=1)

    adata.obs['distance_to_follicle'] = min_distances
    sq.pl.spatial_scatter(adata, color='distance_to_follicle',
                         title="Distance to B Cell Follicles")

    if 'CD3E' in adata.var_names:
        cd3e_idx = adata.var_names.get_loc('CD3E')
        cd3e_expr = adata.X[:, cd3e_idx].toarray().flatten()

        from scipy.stats import spearmanr

        corr, pval = spearmanr(min_distances, cd3e_expr)

        plt.figure(figsize=(8, 6))
        plt.scatter(min_distances, cd3e_expr, alpha=0.5)
        plt.xlabel('Distance to Follicle')
        plt.ylabel('CD3E Expression')
        plt.title(f'CD3E vs Distance to Follicle (Spearman corr: {corr:.3f}, p={pval:.3e})')
        plt.tight_layout()
        plt.show()

"""# 14. Validation Planning

14.1 Identifying Targets for Experimental Validation
"""

validation_targets = pd.DataFrame(columns=[
    'Target', 'Type', 'Finding', 'Validation_Method', 'Priority'
])

if 'moranI' in adata.uns:
    top_spatial_genes = adata.uns['moranI'].sort_values('I', ascending=False).head(5).index.tolist()
    new_rows = []
    for i, gene in enumerate(top_spatial_genes):
        moran_value = adata.uns['moranI'].loc[gene, 'I']
        p_value = None
        for col in adata.uns['moranI'].columns:
            if 'pval' in col.lower():
                p_value = adata.uns['moranI'].loc[gene, col]
                break
        if p_value is not None:
            finding = f'Strong spatial pattern (Moran I={moran_value:.3f}, p={p_value:.3e})'
        else:
            finding = f'Strong spatial pattern (Moran I={moran_value:.3f})'

        new_rows.append({
            'Target': gene,
            'Type': 'Gene',
            'Finding': finding,
            'Validation_Method': 'RNA-FISH or IHC',
            'Priority': i+1
        })
    validation_targets = pd.concat([validation_targets, pd.DataFrame(new_rows)], ignore_index=True)

if cluster_key in adata.obs.columns:
    domain_row = pd.DataFrame([{
        'Target': 'Spatial_domains',
        'Type': 'Pattern',
        'Finding': f'Identified {len(adata.obs[cluster_key].cat.categories)} distinct spatial domains',
        'Validation_Method': 'Multiplex immunofluorescence',
        'Priority': 1
    }])
    validation_targets = pd.concat([validation_targets, domain_row], ignore_index=True)

if 'ligrec' in adata.uns:
    interaction_row = pd.DataFrame([{
        'Target': 'Cell_interactions',
        'Type': 'Interaction',
        'Finding': 'Potential ligand-receptor interactions between domains',
        'Validation_Method': 'Multiplex RNA-FISH for ligand-receptor pairs',
        'Priority': 2
    }])
    validation_targets = pd.concat([validation_targets, interaction_row], ignore_index=True)

if 'dominant_region' in adata.obs.columns:
    region_row = pd.DataFrame([{
        'Target': 'Lymph_node_architecture',
        'Type': 'Structure',
        'Finding': 'Mapped B/T cell zones and other lymph node regions',
        'Validation_Method': 'IHC for region-specific markers',
        'Priority': 3
    }])
    validation_targets = pd.concat([validation_targets, region_row], ignore_index=True)

print("\nValidation targets:")
print(validation_targets)
validation_targets.to_csv('validation_targets.csv', index=False)

"""# 15. Visualization & Publication

15.1 Creating Publication-Quality Figures
"""

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.5

fig = plt.figure(figsize=(15, 12))
fig.suptitle('Spatial Transcriptomic Analysis of Human Lymph Node', fontsize=16, y=0.95)

gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0, 0])
if cluster_key in adata.obs.columns:
    sq.pl.spatial_scatter(adata, color=cluster_key, ax=ax1, title="A) Spatial Domains")
else:
    sq.pl.spatial_scatter(adata, ax=ax1, title="A) Tissue Structure")

ax2 = fig.add_subplot(gs[0, 1])
b_marker = None
for marker in ['MS4A1', 'CD19', 'CD79A']:
    if marker in adata.var_names:
        b_marker = marker
        break
if b_marker:
    sq.pl.spatial_scatter(adata, color=b_marker, ax=ax2, title=f"B) {b_marker} Expression (B cells)")
else:
    hvg = adata.var_names[adata.var.highly_variable][0] if 'highly_variable' in adata.var.columns else adata.var_names[0]
    sq.pl.spatial_scatter(adata, color=hvg, ax=ax2, title=f"B) {hvg} Expression")

ax3 = fig.add_subplot(gs[0, 2])
t_marker = None
for marker in ['CD3E', 'CD3D', 'CD8A']:
    if marker in adata.var_names:
        t_marker = marker
        break
if t_marker:
    sq.pl.spatial_scatter(adata, color=t_marker, ax=ax3, title=f"C) {t_marker} Expression (T cells)")
else:
    hvgs = adata.var_names[adata.var.highly_variable] if 'highly_variable' in adata.var.columns else adata.var_names
    hvg = hvgs[1] if len(hvgs) > 1 else hvgs[0]
    sq.pl.spatial_scatter(adata, color=hvg, ax=ax3, title=f"C) {hvg} Expression")

ax4 = fig.add_subplot(gs[1, 0])
if 'dominant_region' in adata.obs.columns:
    sq.pl.spatial_scatter(adata, color='dominant_region', ax=ax4, title="D) Lymph Node Regions")
elif pathway_columns:
    sq.pl.spatial_scatter(adata, color=pathway_columns[0].replace('pathway_', ''), ax=ax4,
                         title=f"D) {pathway_columns[0].replace('pathway_', '')} Pathway")
else:
    sq.pl.spatial_scatter(adata, color='total_counts', ax=ax4, title="D) Total UMI Counts")

ax5 = fig.add_subplot(gs[1, 1])
if 'moranI' in adata.uns:
    top_spatial_gene = adata.uns['moranI'].sort_values('I', ascending=False).index[0]
    sq.pl.spatial_scatter(adata, color=top_spatial_gene, ax=ax5,
                         title=f"E) {top_spatial_gene} (Top Spatial Gene)")
else:
    hvgs = adata.var_names[adata.var.highly_variable] if 'highly_variable' in adata.var.columns else adata.var_names
    hvg = hvgs[2] if len(hvgs) > 2 else hvgs[0]
    sq.pl.spatial_scatter(adata, color=hvg, ax=ax5, title=f"E) {hvg} Expression")

ax6 = fig.add_subplot(gs[1, 2])
if 'prop_B_cells' in adata.obs.columns:
    sq.pl.spatial_scatter(adata, color='prop_B_cells', ax=ax6, title="F) B Cell Proportion")
elif 'distance_to_follicle' in adata.obs.columns:
    sq.pl.spatial_scatter(adata, color='distance_to_follicle', ax=ax6, title="F) Distance to Follicle")
else:
    if 'local_heterogeneity' in adata.obs.columns:
        sq.pl.spatial_scatter(adata, color='local_heterogeneity', ax=ax6, title="F) Local Heterogeneity")
    else:
        hvgs = adata.var_names[adata.var.highly_variable] if 'highly_variable' in adata.var.columns else adata.var_names
        hvg = hvgs[3] if len(hvgs) > 3 else hvgs[0]
        sq.pl.spatial_scatter(adata, color=hvg, ax=ax6, title=f"F) {hvg} Expression")

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('lymph_node_spatial_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lymph_node_spatial_figure.png', dpi=300, bbox_inches='tight')
print("Saved publication-quality figure in PDF and PNG formats")

"""## 16. Interactive Visualization

**16.1 Interactive Spatial Scatter Plots**
"""

print("\n=== Interactive Visualization ===")
try:
    import plotly.express as px

    plot_df = pd.DataFrame({
        'x': adata.obsm['spatial'][:, 0],
        'y': adata.obsm['spatial'][:, 1],
        'spot_id': adata.obs.index,
    })

    for col in ['leiden', 'kmeans', 'spatial_domains', 'dominant_region']:
        if col in adata.obs.columns:
            plot_df[col] = adata.obs[col].values
            print(f"Added clustering column: {col}")

    gene_var = np.var(adata.X.toarray(), axis=0)
    top_var_idx = np.argsort(gene_var)[-10:]
    top_var_genes = adata.var_names[top_var_idx].tolist()

    print(f"Adding expression data for top {len(top_var_genes)} variable genes")
    for gene in top_var_genes:
        gene_idx = adata.var_names.get_loc(gene)
        plot_df[gene] = adata.X[:, gene_idx].toarray().flatten()

    key_markers = ['CD3E', 'CD19', 'CD4', 'CD8A', 'MS4A1', 'CD68']
    for marker in key_markers:
        if marker in adata.var_names and marker not in plot_df.columns:
            gene_idx = adata.var_names.get_loc(marker)
            plot_df[marker] = adata.X[:, gene_idx].toarray().flatten()
            print(f"Added marker gene: {marker}")

    cell_type_props = [col for col in adata.obs.columns if col.startswith('prop_')]
    for col in cell_type_props[:5]:
        plot_df[col] = adata.obs[col].values
        print(f"Added cell proportion column: {col}")

    pathway_cols = [col for col in adata.obs.columns if col.startswith('pathway_')]
    for col in pathway_cols[:3]:
        plot_df[col] = adata.obs[col].values
        print(f"Added pathway column: {col}")

    print("\nAvailable columns in plot_df:", plot_df.columns.tolist())

    if 'spatial_domains' in plot_df.columns:
        color_col = 'spatial_domains'
    elif 'leiden' in plot_df.columns:
        color_col = 'leiden'
    elif 'kmeans' in plot_df.columns:
        color_col = 'kmeans'
    elif 'dominant_region' in plot_df.columns:
        color_col = 'dominant_region'
    else:
        color_col = top_var_genes[0]

    print(f"Using {color_col} for initial coloring")

    hover_data = {
        'spot_id': True,
        'x': False,
        'y': False
    }

    for gene in top_var_genes[:3]:
        hover_data[gene] = True

    for marker in key_markers[:3]:
        if marker in plot_df.columns:
            hover_data[marker] = True

    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color=color_col,
        title=f"Spatial Transcriptomics Data - Colored by {color_col}",
        hover_data=hover_data,
        width=1000,
        height=800
    )

    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        title_font_size=18,
        template="plotly_white"
    )

    fig.show()
    print("\nCreating multi-view dashboard...")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    multi_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Spatial Domains",
            f"Cell Type Distribution",
            f"Variable Gene: {top_var_genes[0]}",
            f"Marker Expression: CD3E"
        )
    )

    # Plot 1: Spatial domains
    domains_col = color_col
    scatter1 = go.Scatter(
        x=plot_df['x'],
        y=plot_df['y'],
        mode='markers',
        marker=dict(
            size=5,
            color=plot_df[domains_col].astype('category').cat.codes,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=domains_col, x=0.45)
        ),
        text=plot_df['spot_id'],
        showlegend=False
    )
    multi_fig.add_trace(scatter1, row=1, col=1)

    # Plot 2: Cell type proportions
    if 'prop_B_cells' in plot_df.columns and 'prop_T_cells' in plot_df.columns:
        scatter2 = go.Scatter(
            x=plot_df['x'],
            y=plot_df['y'],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_df['prop_B_cells'] / (plot_df['prop_B_cells'] + plot_df['prop_T_cells'] + 0.001),
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title='B/T Cell Ratio', x=1.0)
            ),
            text=plot_df['spot_id'],
            showlegend=False
        )
    else:
        scatter2 = go.Scatter(
            x=plot_df['x'],
            y=plot_df['y'],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_df[top_var_genes[1]],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=top_var_genes[1], x=1.0)
            ),
            text=plot_df['spot_id'],
            showlegend=False
        )
    multi_fig.add_trace(scatter2, row=1, col=2)

    # Plot 3: Variable gene
    scatter3 = go.Scatter(
        x=plot_df['x'],
        y=plot_df['y'],
        mode='markers',
        marker=dict(
            size=5,
            color=plot_df[top_var_genes[0]],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title=top_var_genes[0], x=0.45, y=0.25)
        ),
        text=plot_df['spot_id'],
        showlegend=False
    )
    multi_fig.add_trace(scatter3, row=2, col=1)

    # Plot 4: T cell marker
    if 'CD3E' in plot_df.columns:
        marker_gene = 'CD3E'
    else:
        marker_gene = top_var_genes[2]

    scatter4 = go.Scatter(
        x=plot_df['x'],
        y=plot_df['y'],
        mode='markers',
        marker=dict(
            size=5,
            color=plot_df[marker_gene],
            colorscale='Magma',
            showscale=True,
            colorbar=dict(title=marker_gene, x=1.0, y=0.25)
        ),
        text=plot_df['spot_id'],
        showlegend=False
    )
    multi_fig.add_trace(scatter4, row=2, col=2)

    multi_fig.update_layout(
        title_text="Multi-view Spatial Transcriptomics Dashboard",
        height=1000,
        width=1200
    )

    multi_fig.update_xaxes(showticklabels=False, showgrid=False)
    multi_fig.update_yaxes(showticklabels=False, showgrid=False)

    multi_fig.show()

    fig.write_html("spatial_transcriptomics_single_view.html")
    multi_fig.write_html("spatial_transcriptomics_dashboard.html")
    print("Interactive visualizations saved as HTML files:")
    print("  - spatial_transcriptomics_single_view.html")
    print("  - spatial_transcriptomics_dashboard.html")

except ImportError:
    print("For interactive visualization, install plotly with: pip install plotly")
except Exception as e:
    print(f"Interactive visualization error: {e}")
    import traceback
    traceback.print_exc()

"""**16.2 3D Visualization with PCA and Gene Expression**"""

try:
    import plotly.express as px
    import plotly.graph_objects as go

    pca_df = pd.DataFrame({
        'PC1': adata.obsm['X_pca'][:, 0],
        'PC2': adata.obsm['X_pca'][:, 1],
        'PC3': adata.obsm['X_pca'][:, 2] if adata.obsm['X_pca'].shape[1] > 2 else np.zeros(adata.n_obs),
        'spot_id': adata.obs.index,
    })

    if color_col in adata.obs.columns:
        pca_df[color_col] = adata.obs[color_col]

    if genes_to_include:
        gene = genes_to_include[0]
        gene_idx = adata.var_names.get_loc(gene)
        pca_df[gene] = adata.X[:, gene_idx].toarray().flatten()

    fig3 = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color=gene if genes_to_include else color_col,
        color_continuous_scale='Viridis' if genes_to_include else None,
        hover_name='spot_id',
        title="3D PCA with Gene Expression",
        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'}
    )

    fig3.update_traces(marker=dict(size=5, opacity=0.7))
    fig3.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
        ),
        title_font_size=18,
        width=900,
        height=700
    )

    fig3.show()
    fig3.write_html("pca_3d_interactive.html")

except Exception as e:
    print(f"3D visualization error: {e}")

"""**16.3 Interactive Heatmap of Gene-Gene Correlations**"""

try:
    import plotly.express as px

    corr_genes = genes_to_include[:15]

    if corr_genes:
        expr_matrix = adata[:, corr_genes].X.toarray()
        corr_matrix = np.corrcoef(expr_matrix.T)

        fig4 = px.imshow(
            corr_matrix,
            x=corr_genes,
            y=corr_genes,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Gene-Gene Correlation Heatmap"
        )

        hover_text = [[f"{corr_genes[i]}-{corr_genes[j]}: {corr_matrix[i, j]:.2f}"
                      for j in range(len(corr_genes))]
                     for i in range(len(corr_genes))]

        fig4.update_traces(text=hover_text, hoverinfo='text')
        fig4.update_layout(
            title_font_size=18,
            width=800,
            height=800
        )
        fig4.show()
        fig4.write_html("gene_correlation_heatmap.html")

except Exception as e:
    print(f"Correlation heatmap error: {e}")

"""# 16. Data Export & Documentation

**16.1 Saving Processed Data**
"""

print("Starting comprehensive data export...")
print("Backing up essential data to CSV...")
adata.obs.to_csv('spatial_metadata.csv')
print("- Saved observation metadata")

adata.var.to_csv('gene_metadata.csv')
print("- Saved gene metadata")
coordinates_df = pd.DataFrame(
    adata.obsm['spatial'],
    index=adata.obs.index,
    columns=['x_coord', 'y_coord']
)
coordinates_df.to_csv('spatial_coordinates.csv')
print("- Saved spatial coordinates")

if 'moranI' in adata.uns:
    try:
        adata.uns['moranI'].to_csv('spatial_genes.csv')
        print("- Saved spatially variable genes")
    except Exception as e:
        print(f"  Could not save spatial genes directly: {e}")
        try:
            adata.uns['moranI'].reset_index().to_csv('spatial_genes_fixed.csv', index=False)
            print("- Saved spatially variable genes with workaround")
        except Exception as e2:
            print(f"  Could not save spatial genes with workaround: {e2}")

print("CSV backups complete!")
print("\nCreating HDF5-compatible AnnData object...")

minimal_adata = sc.AnnData(
    X=adata.X,
    obs=adata.obs,
    var=adata.var,
    obsm={'spatial': adata.obsm['spatial']}
)
def make_hdf5_compatible(obj, max_depth=10, current_depth=0):
    """
    Recursively make all data HDF5-compatible by:
    1. Converting all keys to strings
    2. Converting complex data structures to simpler ones
    3. Removing unsupported data types
    """
    if current_depth > max_depth:
        return "ERROR: Max recursion depth exceeded"
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                str_key = str(k)
                print(f"  Converting {type(k).__name__} key to string: {k} → {str_key}")
            else:
                str_key = k
            processed_v = make_hdf5_compatible(v, max_depth, current_depth + 1)
            if processed_v is not None:
                result[str_key] = processed_v
        return result

    elif isinstance(obj, list):
        result = []
        for item in obj:
            processed_item = make_hdf5_compatible(item, max_depth, current_depth + 1)
            if processed_item is not None:
                result.append(processed_item)
        return result

    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        try:
            return make_hdf5_compatible(obj.to_dict(), max_depth, current_depth + 1)
        except Exception as e:
            print(f"  Could not convert {type(obj).__name__} to dict: {e}")
            return str(obj)
    elif isinstance(obj, (int, float, str, bool, np.integer, np.floating, np.bool_)):
        return obj

    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'biufcSU':
            return obj
        else:
            return np.array([str(x) for x in obj.flatten()]).reshape(obj.shape)

    else:
        print(f"  Converting {type(obj).__name__} to string representation")
        return str(obj)

if 'spatial' in adata.uns:
    spatial_uns = {'spatial': {}}

    for lib_id in adata.uns['spatial']:
        spatial_uns['spatial'][lib_id] = {}
        if 'images' in adata.uns['spatial'][lib_id]:
            spatial_uns['spatial'][lib_id]['images'] = adata.uns['spatial'][lib_id]['images']
        if 'scalefactors' in adata.uns['spatial'][lib_id]:
            spatial_uns['spatial'][lib_id]['scalefactors'] = adata.uns['spatial'][lib_id]['scalefactors']
    minimal_adata.uns = spatial_uns
    print("Added essential spatial information")
if 'categorical' in adata.uns:
    try:
        minimal_adata.uns['categorical'] = make_hdf5_compatible(adata.uns['categorical'])
        print("Added categorical information")
    except Exception as e:
        print(f"Could not add categorical information: {e}")
for color_key in [k for k in adata.uns.keys() if k.endswith('_colors')]:
    try:
        minimal_adata.uns[color_key] = adata.uns[color_key]
        print(f"Added {color_key}")
    except Exception as e:
        print(f"Could not add {color_key}: {e}")
if 'moranI' in adata.uns:
    try:
        moranI_dict = adata.uns['moranI'].to_dict('index')
        minimal_adata.uns['moranI_compatible'] = make_hdf5_compatible(moranI_dict)
        print("Added Moran's I results in compatible format")
    except Exception as e:
        print(f"Could not add Moran's I results: {e}")

print("\nSaving HDF5-compatible AnnData...")
try:
    minimal_adata.write('lymph_node_spatial_minimal.h5ad')
    print("Successfully saved HDF5-compatible AnnData to lymph_node_spatial_minimal.h5ad")
except Exception as e:
    print(f"Error saving minimal AnnData: {e}")
    try:
        bare_adata = sc.AnnData(
            X=adata.X,
            obs=adata.obs,
            var=adata.var,
            obsm={'spatial': adata.obsm['spatial']}
        )
        bare_adata.uns = {}

        bare_adata.write('lymph_node_spatial_bare.h5ad')
        print("Saved bare-minimum AnnData without any uns data")
    except Exception as e2:
        print(f"Even bare-minimum save failed: {e2}")

print("\nData export process complete!")
print("All essential data has been backed up to CSV files")
print("You can reload the data for visualization using:")
print("  adata = sc.read('lymph_node_spatial_minimal.h5ad')")

"""16.2 Creating a Summary Report"""

from datetime import datetime

if 'moranI' in adata.uns:
    print("Available columns in moranI:", adata.uns['moranI'].columns.tolist())
    pval_col = None
    for col in adata.uns['moranI'].columns:
        if 'pval' in col.lower() or 'p_val' in col.lower() or 'p-val' in col.lower():
            pval_col = col
            print(f"Found p-value column: '{pval_col}'")
            break
    i_col = 'I' if 'I' in adata.uns['moranI'].columns else None
    if i_col is None:
        for col in adata.uns['moranI'].columns:
            if col.lower() == 'i' or 'moran' in col.lower():
                i_col = col
                print(f"Found Moran's I column: '{i_col}'")
                break
    if pval_col:
        sig_genes_count = sum(adata.uns['moranI'][pval_col] < 0.05)
        print(f"Found {sig_genes_count} genes with significant spatial variation")
    else:
        sig_genes_count = 'Unknown (p-value column not found)'
else:
    print("No moranI results found")
    pval_col = None
    i_col = None
    sig_genes_count = 0

today_date = datetime.now().strftime('%Y-%m-%d')

html_header = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Spatial Transcriptomics Analysis Report - Human Lymph Node</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .section {{ margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Spatial Transcriptomics Analysis Report</h1>

    <div class="section">
        <h2>1. Dataset Summary</h2>
        <p>Tissue: Human Lymph Node</p>
        <p>Spots analyzed: {adata.n_obs}</p>
        <p>Genes detected: {adata.n_vars}</p>
        <p>Highly variable genes: {adata.var.highly_variable.sum() if 'highly_variable' in adata.var.columns else 'Not calculated'}</p>
    </div>

    <div class="section">
        <h2>2. Key Findings</h2>
        <ul>
            <li>Identified {len(adata.obs[cluster_key].cat.categories) if cluster_key in adata.obs.columns else 0} spatial domains</li>
            <li>Found {sig_genes_count} genes with significant spatial variation</li>
            <li>Mapped potential tissue regions including {', '.join(['B cell zones', 'T cell zones', 'Medullary regions'])}</li>
        </ul>
    </div>

    <div class="section">
        <h2>3. Top Spatially Variable Genes</h2>
        <table>
            <tr>
                <th>Gene</th>
                <th>Moran's I</th>
                {f"<th>P-value</th>" if pval_col else ""}
            </tr>
"""

gene_rows = ""
if 'moranI' in adata.uns and i_col:
    if i_col:
        top_genes = adata.uns['moranI'].sort_values(i_col, ascending=False).head(10).index.tolist()
    else:
        top_genes = adata.uns['moranI'].index[:10].tolist()
    for gene in top_genes:
        gene_rows += f"<tr><td>{gene}</td>"
        if i_col:
            gene_rows += f"<td>{adata.uns['moranI'].loc[gene, i_col]:.3f}</td>"
        else:
            gene_rows += "<td>N/A</td>"
        if pval_col:
            gene_rows += f"<td>{adata.uns['moranI'].loc[gene, pval_col]:.2e}</td>"

        gene_rows += "</tr>\n"
else:
    gene_rows = "<tr><td colspan='3'>Spatial statistics not calculated</td></tr>"

html_footer = f"""
        </table>
    </div>

    <div class="section">
        <h2>4. Methods</h2>
        <p>Analysis performed using:</p>
        <ul>
            <li>scanpy {sc.__version__}</li>
            <li>squidpy {sq.__version__}</li>
            <li>numpy {np.__version__}</li>
        </ul>
        <p>Key processing steps:</p>
        <ol>
            <li>Quality control and preprocessing</li>
            <li>Normalization and log-transformation</li>
            <li>Identification of highly variable genes</li>
            <li>Dimensional reduction with PCA</li>
            <li>Spatial analysis including neighbor graph, Moran's I, and clustering</li>
            <li>Cell type and region annotation</li>
            <li>Pathway and functional analysis</li>
        </ol>
    </div>

    <div class="section">
        <h2>5. Next Steps</h2>
        <p>Recommended follow-up analyses:</p>
        <ul>
            <li>Validation of key spatial genes with RNAscope or immunohistochemistry</li>
            <li>Integration with single-cell RNA-seq data for better cell type resolution</li>
            <li>Comparison with additional lymph node samples to assess reproducibility</li>
            <li>Focused analysis of B cell follicles and T cell zones interaction</li>
        </ul>
    </div>

    <div class="section">
        <h2>6. Session Information</h2>
        <p>Analysis completed on {today_date}</p>
    </div>
</body>
</html>
"""

complete_html = html_header + gene_rows + html_footer

try:
    with open('spatial_analysis_report.html', 'w') as f:
        f.write(complete_html)
    print("Generated HTML report: spatial_analysis_report.html")
except Exception as e:
    print(f"Error generating HTML report: {e}")

    text_report = f"""
Spatial Transcriptomics Analysis Report
======================================

1. Dataset Summary
-----------------
Tissue: Human Lymph Node
Spots analyzed: {adata.n_obs}
Genes detected: {adata.n_vars}
Highly variable genes: {adata.var.highly_variable.sum() if 'highly_variable' in adata.var.columns else 'Not calculated'}

2. Key Findings
--------------
- Identified {len(adata.obs[cluster_key].cat.categories) if cluster_key in adata.obs.columns else 0} spatial domains
- Found {sig_genes_count} genes with significant spatial variation
- Mapped B cell and T cell zones in the lymph node tissue

3. Top Spatially Variable Genes
-----------------------------
"""
    if 'moranI' in adata.uns and i_col:
        if i_col:
            top_genes = adata.uns['moranI'].sort_values(i_col, ascending=False).head(10).index.tolist()
        else:
            top_genes = adata.uns['moranI'].index[:10].tolist()

        for i, gene in enumerate(top_genes, 1):
            text_report += f"{i}. {gene}"
            if i_col:
                text_report += f" (Moran's I: {adata.uns['moranI'].loc[gene, i_col]:.3f})"
            if pval_col:
                text_report += f" (p-value: {adata.uns['moranI'].loc[gene, pval_col]:.2e})"
            text_report += "\n"
    else:
        text_report += "Spatial statistics not calculated\n"

    text_report += f"""
4. Analysis Completed
-------------------
Analysis performed using scanpy {sc.__version__}, squidpy {sq.__version__}, and numpy {np.__version__}.
Date: {today_date}
"""
    with open('spatial_analysis_report.txt', 'w') as f:
        f.write(text_report)
    print("Generated simple text report as fallback: spatial_analysis_report.txt")