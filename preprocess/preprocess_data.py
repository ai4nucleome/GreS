#!/usr/bin/env python
"""
Preprocessing Script for Standardized h5ad Data

This script takes standardized h5ad files (from data_converter.py) and performs:
1. HVG (Highly Variable Genes) selection
2. Raw counts preservation (layers['counts'])
3. Library size calculation
4. Normalization and scaling
5. Feature and spatial graph construction

Input: Standardized h5ad file with:
    - adata.X: Raw counts
    - adata.obs['ground_truth']: Cell type annotations
    - adata.obsm['spatial']: 2D spatial coordinates

Output: Preprocessed h5ad file with additional fields:
    - adata.X: Normalized/scaled features
    - adata.layers['counts']: Raw counts (post-HVG)
    - adata.obs['library_size']: Library size per spot
    - adata.obs['ground']: Integer-encoded labels
    - adata.obsm['fadj']: Feature adjacency graph
    - adata.obsm['sadj']: Spatial adjacency graph
    - adata.obsm['graph_nei']: Neighbor graph
    - adata.obsm['graph_neg']: Negative samples graph

Usage:
    python DLPFC_generate_data.py --input_h5ad /path/to/raw.h5ad --output_dir /path/to/output
    python DLPFC_generate_data.py --dataset_id 151507  # Uses default paths
"""

from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from config import Config
from construction import features_construct_graph, spatial_construct_graph1


def normalize(adata, highly_genes=3000):
    """
    Perform HVG selection and normalization/scaling.
    
    Steps:
    1. Filter genes expressed in < 100 cells
    2. Select top highly variable genes (HVGs)
    3. Save raw counts to adata.layers['counts']
    4. Compute and save library size
    5. Normalize (CPM-like) and scale
    
    Args:
        adata: AnnData object with raw counts in X
        highly_genes: Number of HVGs to select
    
    Returns:
        adata: Processed AnnData with normalized X and raw counts in layers
    """
    print("Starting HVG selection...")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    
    # Save raw counts BEFORE normalization
    if hasattr(adata.X, 'toarray'):
        counts = adata.X.toarray().astype(np.float32)
    else:
        counts = np.array(adata.X, dtype=np.float32)
    
    adata.layers['counts'] = counts
    
    # Compute and save library size
    library_size = counts.sum(axis=1)
    adata.obs['library_size'] = library_size
    adata.obs['log_library_size'] = np.log1p(library_size)
    
    print(f"  [INFO] Saved raw counts to layers['counts'], shape: {counts.shape}")
    print(f"  [INFO] library_size: min={library_size.min():.1f}, max={library_size.max():.1f}, mean={library_size.mean():.1f}")
    
    # Normalization and scaling for encoder input
    # Ensure X is in correct format (dense or sparse csr) before scaling
    if sp.issparse(adata.X):
        # Convert to CSR if it's sparse but not CSR (e.g. COO)
        if not sp.isspmatrix_csr(adata.X):
             adata.X = adata.X.tocsr()
        # Perform normalization on sparse matrix
        # Note: np.sum on sparse matrix returns numpy matrix, reshape works
        # But division of sparse matrix by dense vector might be tricky, let's use scanpy's normalize_total for safety and efficiency if possible
        # However, to keep exact logic: "X / sum * 10000"
        
        # Safe way to normalize sparse matrix:
        sc.pp.normalize_total(adata, target_sum=10000)
    else:
        # Dense matrix normalization
        adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000

    # Ensure CSR again before scale (scanpy's scale requires CSR/CSC for sparse inputs)
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
        
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
    return adata


def load_standardized_h5ad(input_path: str):
    """
    Load a standardized h5ad file and validate required fields.
    
    Args:
        input_path: Path to the standardized h5ad file
    
    Returns:
        adata: Validated AnnData object
    """
    print(f"Loading standardized h5ad: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    adata = sc.read_h5ad(input_path)
    
    # Validate required fields
    if 'ground_truth' not in adata.obs.columns:
        raise ValueError("Missing 'ground_truth' column in adata.obs")
    
    if 'spatial' not in adata.obsm:
        raise ValueError("Missing 'spatial' in adata.obsm")
    
    print(f"  [INFO] Loaded {adata.n_obs} spots, {adata.n_vars} genes")
    print(f"  [INFO] Ground truth labels: {adata.obs['ground_truth'].nunique()} unique values")
    
    # Create integer-encoded ground labels
    labels = adata.obs['ground_truth'].values
    _, ground = np.unique(labels, return_inverse=True)
    adata.obs['ground'] = ground
    
    return adata


def preprocess_adata(adata, highly_genes: int, fadj_k: int, sadj_radius: int):
    """
    Full preprocessing pipeline: HVG selection, normalization, graph construction.
    
    Args:
        adata: AnnData object with raw counts
        highly_genes: Number of HVGs to select
        fadj_k: Number of neighbors for feature graph
        sadj_radius: Radius for spatial graph
    
    Returns:
        adata: Fully preprocessed AnnData
    """
    # Step 1: HVG selection and normalization
    adata = normalize(adata, highly_genes=highly_genes)
    
    # Step 2: Feature adjacency graph (based on expression similarity)
    print("Building feature adjacency graph...")
    fadj = features_construct_graph(adata.X, k=fadj_k)
    adata.obsm["fadj"] = fadj
    
    # Step 3: Spatial adjacency graph (based on physical distance)
    print("Building spatial adjacency graph...")
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=sadj_radius)
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    
    adata.var_names_make_unique()
    
    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess standardized h5ad data for GreS model."
    )
    parser.add_argument(
        "--input_h5ad", type=str, default=None,
        help="Path to input standardized h5ad file. If not provided, uses --dataset_id with default paths."
    )
    parser.add_argument(
        "--dataset_id", type=str, default=None,
        help="Dataset ID (e.g., 151507, Human_Breast_Cancer). Used to construct default input/output paths."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for preprocessed data. Defaults to data/generated/{dataset_id}/"
    )
    parser.add_argument(
        "--config_name", type=str, default="DLPFC",
        help="Name of the config file (without .ini). E.g., 'DLPFC', 'Embryo'. Default: DLPFC"
    )
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="Path to config file. Overrides config_name if provided."
    )
    
    args = parser.parse_args()
    
    # Determine project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Resolve input path logic for various datasets
    if args.input_h5ad is not None:
        input_path = args.input_h5ad
        if args.dataset_id is None:
            args.dataset_id = os.path.splitext(os.path.basename(input_path))[0]
    elif args.dataset_id is not None:
        # Default path structure based on dataset ID patterns
        if args.dataset_id.isdigit():
            # DLPFC (numeric IDs)
            input_path = os.path.join(root_dir, "data", "raw_h5ad", "DLPFC", f"{args.dataset_id}.h5ad")
        elif "E" in args.dataset_id and "S" in args.dataset_id and any(c.isdigit() for c in args.dataset_id):
            # Embryo (e.g., E1S1)
            input_path = os.path.join(root_dir, "data", "raw_h5ad", "Embryo", f"{args.dataset_id}.h5ad")
        else:
            # Fallback (e.g., Human_Breast_Cancer)
            input_path = os.path.join(root_dir, "data", "raw_h5ad", args.dataset_id, f"{args.dataset_id}.h5ad")
            # If not found there, try generic dataset folder
            if not os.path.exists(input_path):
                 input_path = os.path.join(root_dir, "data", "raw_h5ad", f"{args.dataset_id}.h5ad")

    else:
        parser.error("Either --input_h5ad or --dataset_id must be provided.")
    
    # Resolve output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(root_dir, "data", "generated", args.dataset_id)
    
    # Resolve config file from config_name
    if args.config_file is not None:
        config_file = args.config_file
    else:
        config_file = os.path.join(root_dir, "config", f"{args.config_name}.ini")
    
    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input h5ad: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Config file: {config_file}")
    print(f"Dataset ID: {args.dataset_id}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = Config(config_file)
    print(f"Config: fdim={config.fdim}, fadj_k={config.fadj_k}, sadj_k={config.sadj_k}")
    
    # Load standardized h5ad
    adata = load_standardized_h5ad(input_path)
    
    # Preprocess
    adata = preprocess_adata(
        adata,
        highly_genes=config.fdim,
        fadj_k=config.fadj_k,
        sadj_radius=config.sadj_k
    )
    
    # Save
    output_path = os.path.join(output_dir, "data.h5ad")
    print(f"\nSaving preprocessed data to: {output_path}")
    adata.write_h5ad(output_path)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output: {output_path}")
    print(f"Final shape: {adata.n_obs} spots x {adata.n_vars} genes")
    print("=" * 60)
