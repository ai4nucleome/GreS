# GreS: Graph-Regulated Semantic Learning for Spatial Domain Identification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

**GreS** is a novel graph-based deep learning framework that leverages **semantic embeddings** to modulate the learning of spatial domains. By integrating gene regulatory networks (GRNs) and large language model (LLM)-derived semantic knowledge, GreS enhances the representation of spatial spots, leading to more accurate clustering and domain identification.

Key features include:
*   🧠 **Semantic Knowledge Integration**: Utilizes semantic embeddings derived from LLMs and GRNs to guide representation learning.
*   🧬 **Dual Graph Encoding**: Captures both spatial dependencies (Spatial GCN) and functional gene relationships (Feature GCN).
*   🎯 **Adaptive Fusion**: Employs a gated fusion mechanism with FiLM (Feature-wise Linear Modulation) to dynamically weigh spatial vs. semantic information.
*   📉 **Robust Reconstruction**: Uses a ZINB decoder to handle sparsity and noise inherent in spatial transcriptomics data.

## Table of Contents

* [Installation](#installation)
* [Repository Structure](#repository-structure)
* [Data Preparation](#data-preparation)
* [Usage](#usage)
    *   [Preprocessing](#preprocessing)
    *   [Training](#training)
* [Model Architecture](#model-architecture)
* [Citation](#citation)
* [License](#license)

## Installation

### Option 1: Conda (Recommended)

```bash
# Create and activate environment
conda create -n gres python=3.9
conda activate gres

# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install scanpy pandas numpy scipy scikit-learn matplotlib tqdm
```

## Repository Structure

```
GreS/
├── config/                 # Configuration files (e.g., DLPFC.ini)
├── preprocess/             # Preprocessing pipeline
│   ├── config.py           # Preprocessing configuration
│   ├── construction.py     # Graph construction utilities
│   └── preprocess_data.py  # Data cleaning and normalization
├── fig/                    # Figure assets
├── models.py               # GreS model architecture
├── train.py                # Main training script
├── run_preprocess.sh       # Automated preprocessing script
├── build_fadj_from_geneemb.py # Feature graph construction
├── build_programST_assets.py  # Semantic embedding generation
├── grn_generate_spot_embedding.py # Spot embedding generation
├── utils.py                # Utility functions
└── README.md
```

## Data Preparation

### 1. Prepare H5AD Files
Your spatial transcriptomics data should be in `.h5ad` format with:
*   `adata.X`: Raw integer counts of gene expression.
*   `adata.obsm['spatial']`: Spatial coordinates.
*   `adata.var_names`: Gene symbols.

### 2. Directory Structure
Place your raw `.h5ad` files in the `data/raw_h5ad/` directory before running the preprocessing pipeline.

## Usage

### Preprocessing

We provide a comprehensive shell script `run_preprocess.sh` that automates the entire preprocessing workflow, including data cleaning, semantic embedding generation, spot embedding generation, and graph construction.

```bash
# Syntax: ./run_preprocess.sh <dataset_id> <config_name>

# Example for DLPFC dataset
./run_preprocess.sh 151507 DLPFC

# Example for Embryo dataset
./run_preprocess.sh E1S1 Embryo
```

**Pipeline Steps:**
1.  **Data Preprocessing**: Filters genes/cells and normalizes data.
2.  **Semantic Embedding**: Generates semantic embeddings using GRN diffusion.
3.  **Spot Embedding**: Aggregates gene embeddings to the spot level.
4.  **Graph Construction**: Builds the feature adjacency graph.

### Training

Train the GreS model using `train.py`. The script supports both supervised (with ground truth for metrics) and unsupervised modes.

```bash
# Basic usage
python train.py \
    --dataset_id 151672 \
    --config_name DLPFC \
    --llm_emb_dir data/npys_grn/ \
    --run_name test_run
```

**Key Arguments:**
*   `--dataset_id`: Identifier for the dataset (must match the ID used in preprocessing).
*   `--config_name`: Configuration file to use (e.g., `DLPFC`).
*   `--use_llm`: Whether to use semantic embedding modulation (`true` or `false`).
*   `--n_clusters`: (Optional) Force unsupervised mode by specifying the number of clusters manually.
*   `--save_best_ckpt`: Save the model checkpoint with the best performance.

**Output:**
Results are saved in `data/result/<config>/<dataset_id>/<run_name>/`:
*   `best_cluster_outputs.npz`: Contains embeddings, cluster labels, and metrics.
*   `metrics_best.json`: Summary of best performance metrics.
*   `GreS.png`: Visualization of the spatial domains.
*   `checkpoints/`: Model checkpoints.

## Model Architecture

GreS employs a dual-encoder architecture with a gated fusion mechanism and FiLM conditioning:

*   **Dual GCN Encoders**:
    *   **Spatial GCN (SGCN)**: Captures spatial dependencies using a spatial adjacency graph.
    *   **Feature GCN (FGCN)**: Captures functional gene relationships using a feature adjacency graph derived from GRN-based embeddings.
*   **Semantic Modulation (FiLM)**:
    *   Utilizes semantic embeddings (derived from LLMs and GRNs) to modulate both the **gating mechanism** and the **fused representation**.
    *   This allows the model to dynamically weigh spatial vs. feature information based on semantic context.
*   **ZINB Decoder**:
    *   Reconstructs gene expression data using a Zero-Inflated Negative Binomial (ZINB) distribution to handle sparsity and noise in ST data.

## Citation

If you use GreS in your research, please cite:

```
(Coming Soon)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
