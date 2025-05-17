# SpatialFlow

SpatialFlow is an open-source workflow/pipeline for analyzing spatial transcriptomics data.  
It explore gene expression in tissue with spatial information.

Spatial transcriptomics is a growing field in biology.
This project includes all basic and advanced steps — from loading data to generating results that can be used in research papers.

---

## This Pipeline includes

* Load and explore spatial transcriptomics data  
* Do quality control (QC) and filtering  
* Normalize and reduce dimensions (PCA, UMAP)  
* Cluster spatial spots or cells  
* Detect marker genes  
* Map spatial domains  
* Estimate cell types in mixed spots (deconvolution)  
* Study cell-cell communication (ligand-receptor analysis)  
* Run pathway analysis  
* Visualize and save results

---
## 1. Clone the repository:

```bash
   git clone https://github.com/rezwan-lab/spatialflow.git
   cd spatialflow
```

## 2. Create a conda environment
```bash
conda create -n spatialflow python=3.10
```

## 3. Activate SpatialFlow
```bash
conda activate spatialflow
```

## 4.1 Install dependencies
```bash
pip install -r requirements-spatialflow.txt

or use mamba
requirements-mamba.txt

```
## 4.2 Install the package in development mode
```bash
pip install -e .
```


## 5.1 Load test data
```python
import squidpy as sq

# Load the example Visium dataset
adata = sq.datasets.visium("V1_Human_Lymph_Node")

# Print basic information about the loaded dataset
print(f"AnnData object: {adata}")
print(f"Shape of data matrix: {adata.shape}")
print(f"Available layers: {list(adata.layers.keys())}")
print(f"Spatial coordinates available: {'spatial' in adata.obsm}")
```

## 5.2 Convert data
```python
import squidpy as sq

# Load the example Visium dataset (human lymph node)
adata = sq.datasets.visium("V1_Human_Lymph_Node")

# Save the data for SpatialFlow
adata.write_h5ad("./V1_Human_Lymph_Node.h5ad")
```

## 6. Generate config file
```bash
python -m spatialflow.cli init-config --output config.yaml
python -m spatialflow.cli init-config config.yaml
```
## 7. Usage
```bash
python -m spatialflow.cli --help
```

## 8.1 Run SpatialFlow workflow mode
```bash
python examples/basic_workflow.py

#or

python examples/advanced_workflow.py
```
## 8.2 Run SpatialFlow Analysis
```python
python -m spatialflow.cli run <.h5ad file path> --config config.yaml --output-dir spatialflow_output
```

## 8.3 Run SpatialFlow visualization

```python
python -m spatialflow.cli visualize spatialflow_output/data/*.h5ad -o spatialflow_output/figures
```

## Author

Dr. Rezwanuzzaman Laskar
