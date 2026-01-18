# GreS: Graph-Regulated Semantic Learning for Spatial Domain Identification

The repo contains the official implementation of **GreS: Graph-Regulated Semantic Learning for Spatial Domain Identification**.

## Contents

* [1. Introduction](#1-introduction)
* [2. Model Architecture](#2-model-architecture)
* [3. Setup Environment](#3-setup-environment)
* [4. Data Preparation](#4-data-preparation)
* [5. Preprocessing Pipeline](#5-preprocessing-pipeline)
* [6. Training](#6-training)
* [7. Citation](#7-citation)

## 1. Introduction

Spatial transcriptomics (ST) technologies have revolutionized our understanding of tissue organization by capturing gene expression with spatial context. However, effectively integrating spatial information, gene expression data, and semantic knowledge remains a challenge for accurate spatial domain identification.

**GreS** is a novel graph-based deep learning framework that leverages **semantic embeddings** to modulate the learning of spatial domains. By integrating gene regulatory networks (GRNs) and large language model (LLM)-derived semantic knowledge, GreS enhances the representation of spatial spots, leading to more accurate clustering and domain identification.

## 2. Model Architecture

GreS employs a dual-encoder architecture with a gated fusion mechanism and FiLM (Feature-wise Linear Modulation) conditioning:

*   **Dual GCN Encoders**: 
    *   **Spatial GCN (SGCN)**: Captures spatial dependencies using a spatial adjacency graph.
    *   **Feature GCN (FGCN)**: Captures functional gene relationships using a feature adjacency graph derived from GRN-based embeddings.
*   **Semantic Modulation (FiLM)**: 
    *   Utilizes semantic embeddings (derived from LLMs and GRNs) to modulate both the **gating mechanism** and the **fused representation**.
    *   This allows the model to dynamically weigh spatial vs. feature information based on semantic context.
*   **ZINB Decoder**: Reconstructs gene expression data using a Zero-Inflated Negative Binomial (ZINB) distribution to handle sparsity and noise in ST data.

## 3. Setup Environment

Create a virtual environment and install the required dependencies:

```bash
# Create and activate environment
conda create -n gres python=3.9
conda activate gres

# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install scanpy pandas numpy scipy scikit-learn matplotlib tqdm
```

## 4. Data Preparation

Standardize your input data before running the pipeline:

1.  **Format**: Store your spatial transcriptomics data in `.h5ad` format.
2.  **Raw Counts**: Ensure `adata.X` contains **raw integer counts**.
3.  **Spatial Coordinates**: Store spatial coordinates (and annotations if available) in `adata.obsm['spatial']`.
4.  **Directory**: Place your raw `.h5ad` files in `data/raw_h5ad/`.

## 5. Preprocessing Pipeline

We provide a comprehensive shell script `run_preprocess.sh` that automates the entire preprocessing workflow, including data cleaning, semantic embedding generation, spot embedding generation, and graph construction.

### Usage

```bash
./run_preprocess.sh <dataset_id> <config_name>
```

### Examples

**For DLPFC dataset:**
```bash
./run_preprocess.sh 151507 DLPFC
```

**For Embryo dataset:**
```bash
./run_preprocess.sh E1S1 Embryo
```

### Pipeline Steps (Automated by `run_preprocess.sh`)

1.  **Data Preprocessing**: Filters genes/cells and normalizes data (`preprocess/preprocess_data.py`).
2.  **Semantic Embedding**: Generates semantic embeddings using GRN diffusion (`build_programST_assets.py`).
3.  **Spot Embedding**: Aggregates gene embeddings to spot level (`grn_generate_spot_embedding.py`).
4.  **Graph Construction**: Builds the feature adjacency graph (`build_fadj_from_geneemb.py`).

## 6. Training

Train the GreS model using `train.py`. The script supports both supervised (with ground truth for metrics) and unsupervised modes.

### Basic Usage

```bash
python train.py \
    --dataset_id 151672 \
    --config_name DLPFC \
    --llm_emb_dir data/npys_grn/ \
    --run_name test_run
```

### Key Arguments

*   `--dataset_id`: Identifier for the dataset (must match preprocessing).
*   `--config_name`: Configuration file to use (e.g., `DLPFC`, `Embryo`).
*   `--use_llm`: Whether to use semantic embedding modulation (`true` or `false`).
*   `--n_clusters`: (Optional) Force unsupervised mode by specifying the number of clusters manually.
*   `--save_best_ckpt`: Save the model checkpoint with the best performance.

### Output

Results are saved in `data/result/<config>/<dataset_id>/<run_name>/`:
*   `best_cluster_outputs.npz`: Contains embeddings, cluster labels, and metrics.
*   `metrics_best.json`: Summary of best performance metrics.
*   `GreS.png`: Visualization of the spatial domains.
*   `checkpoints/`: Model checkpoints.

## 7. Citation

If you use GreS in your work, please kindly cite our paper:

```bibtex
@article{YourCitationHere,
  title={GreS: Graph-Regulated Semantic Learning for Spatial Domain Identification},
  author={Your Name and Co-authors},
  journal={Bioinformatics / ISMB},
  year={2024}
}
```
