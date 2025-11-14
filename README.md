# ORBIT: Oncogenic Representation Learning via Bi-Prototype Contrastive Learning in Hyperbolic Space for Cancer Driver Gene Identification

[cite_start]**Authors:** Sang-Pil Cho, Young-Rae Cho [cite: 2]
[cite_start]**Laboratory:** ADS Lab, Yonsei University [cite: 3, 12]
**Paper:** (Link to your published paper when available)
[cite_start]**Source Code & Datasets:** [https://ads.yonsei.ac.kr/ORBIT](https://ads.yonsei.ac.kr/ORBIT) [cite: 12]

---

## 📖 Summary

This repository contains the official implementation of **ORBIT**, a novel deep learning framework for Cancer Driver Gene Identification.

[cite_start]The identification of cancer driver genes is hindered by challenges in integrating heterogeneous data and by methodological limitations that overlook cancer-specific network dynamics[cite: 4]. [cite_start]**ORBIT (Oncogenic Representation Learning via Bi-Prototype Contrastive Learning in Hyperbolic Space)** is proposed to overcome these limitations[cite: 5].

**Key Features of ORBIT:**

* [cite_start]**Synergistic Data Integration:** ORBIT synergistically integrates multi-omics profiles, protein sequence features (from ESM-2), and functional gene sets (from MSigDB) through a data-aware attention mechanism[cite: 6, 35, 37, 40].
* [cite_start]**Dynamic Graph Learning:** It introduces a **Context-Adaptive Graph Rewiring** mechanism to learn cancer-specific network dynamics[cite: 7, 85].
* **Hyperbolic Geometry:** It employs a novel **Bi-Prototype Contrastive Learning** designed for hyperbolic space. [cite_start]This approach preserves the network's natural hierarchy while structuring the gene representation space using driver and non-driver prototypes as distinct semantic anchors[cite: 7, 8, 84].

[cite_start]Comprehensive experiments show that ORBIT demonstrates superior performance compared to state-of-the-art models on pan-cancer and multiple cancer-specific datasets[cite: 9, 31].

---

## ⚙️ Requirements

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


## 🚀 Implementation

To run the ORBIT model, you must specify the following two arguments:

1.  **PPI Network Source:** `STRING`, `CPDB`, or `BioGRID`
2.  **Cancer Type:** `pan-cancer` or a specific cancer type code (e.g., `BRCA`, `KIRC`, `LUAD`, `UCEC`)

### 💡 Example Usage

(Assuming the main execution script is named `run_orbit.py`)

```bash
# Run ORBIT using the STRING network for pan-cancer prediction
python run_orbit.py STRING pan-cancer

# Run ORBIT using the CPDB network for BRCA (Breast Cancer)
python run_orbit.py CPDB BRCA

# Run ORBIT using the BioGRID network for KIRC (Kidney Renal Clear Cell Carcinoma)
python run_orbit.py BioGRID KIRC

