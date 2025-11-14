# ORBIT: Oncogenic Representation Learning via Bi-Prototype Contrastive Learning in Hyperbolic Space for Cancer Driver Gene Identification

## Summary

This repository contains the official implementation of **ORBIT**, a novel deep learning framework for Cancer Driver Gene Identification.

The identification of cancer driver genes is hindered by challenges in integrating heterogeneous data and by methodological limitations that overlook cancer-specific network dynamics. **ORBIT (Oncogenic Representation Learning via Bi-Prototype Contrastive Learning in Hyperbolic Space)** is proposed to overcome these limitations.

**Key Features of ORBIT:**

* **Synergistic Data Integration:** ORBIT synergistically integrates multi-omics profiles, protein sequence features, and functional gene sets through a data-aware attention mechanism.
* **Dynamic Graph Learning:** It introduces a **Context-Adaptive Graph Rewiring** mechanism to learn cancer-specific network dynamics.
* **Hyperbolic Geometry:** It employs a novel **Bi-Prototype Contrastive Learning** designed for hyperbolic space. This approach preserves the network's natural hierarchy while structuring the gene representation space using driver and non-driver prototypes as distinct semantic anchors.

---

## Requirements

This model was developed and tested in the following environment, based on the provided `conda list`. We recommend using Anaconda to create a dedicated environment.

* **Python:** `3.8.20`
* **PyTorch:** `2.4.1+cu121`
* **PyTorch Geometric (PyG):** `2.6.1`
    * `torch-cluster`: `1.6.3`
    * `torch-scatter`: `2.1.2`
    * `torch-sparse`: `0.6.18`
    * `torch-spline-conv`: `1.2.2`
* **Geoopt (for Hyperbolic ops):** `0.5.1`
* **Core Libraries:**
    * `numpy`: `1.24.1`
    * `pandas`: `2.0.3`
    * `scikit-learn`: `1.3.2`
    * `scipy`: `1.10.1`
    * `networkx`: `3.0`
    * `pyyaml`: `6.0.2`


## Implementation

To run the ORBIT model, you must specify the following two arguments:

1.  **PPI Network Source:** `STRING`, `CPDB`, or `BioGRID`
2.  **Cancer Type:** `pan-cancer` or a specific cancer type code (e.g., `BRCA`, `KIRC`, `LUAD`, `UCEC`)

### Example Usage

(Assuming the main execution script is named `run_model.py`)

```bash
# Run ORBIT using the STRING network for pan-cancer prediction
python run_model.py STRING pan-cancer

# Run ORBIT using the CPDB network for BRCA
python run_model.py CPDB BRCA

# Run ORBIT using the BioGRID network for KIRC
python run_model.py BioGRID KIRC

