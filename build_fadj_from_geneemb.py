#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build fadj (feature adjacency) from GRN-diffused gene embeddings + expression.

This script computes spot embeddings by weighted averaging gene embeddings,
then builds a cosine-kNN graph to replace the original expression-based fadj.

Output:
  - fadj_spotemb_grn_k{k}.npz: sparse adjacency matrix (scipy coo_matrix)

Usage:
  python build_fadj_from_geneemb.py --dataset_id 151672 --k 14
"""

import os
import argparse
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph


def get_script_dir() -> str:
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def load_expression_matrix(h5ad_path: str) -> np.ndarray:
    """Load expression matrix from h5ad, return as dense float32."""
    print(f"[INFO] Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    print(f"[INFO] Expression matrix shape: {X.shape}")
    return X


def load_gene_embeddings_grn(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GRN-diffused gene embeddings from grn_gene_embeddings.pt.
    
    Returns:
        E: gene embedding matrix (n_genes, emb_dim)
        emb_mask: boolean mask (n_genes,)
    """
    pt_path = os.path.join(data_dir, "grn_gene_embeddings.pt")
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"grn_gene_embeddings.pt not found: {pt_path}")
    
    print(f"[INFO] Loading GRN gene embeddings: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    E = obj["E_grn"].cpu().numpy().astype(np.float32)
    emb_mask = obj["emb_mask"].cpu().numpy().astype(bool)
    
    print(f"[INFO] E_grn shape: {E.shape}, emb_mask True count: {emb_mask.sum()}")
    return E, emb_mask


def compute_spot_embeddings(
    X: np.ndarray,
    E: np.ndarray,
    emb_mask: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute spot embeddings by weighted average of gene embeddings.
    
    Formula:
        w_{ig} = log(1 + x_{ig})
        s_i = sum_{g in M} w_{ig} * E_g / (sum_{g in M} w_{ig} + eps)
    
    where M = {g : emb_mask[g] == True}
    
    Args:
        X: expression matrix (n_spots, n_genes)
        E: gene embedding matrix (n_genes, emb_dim)
        emb_mask: boolean mask (n_genes,)
        eps: small constant to avoid division by zero
    
    Returns:
        S: spot embeddings (n_spots, emb_dim)
    """
    n_spots, n_genes = X.shape
    emb_dim = E.shape[1]
    
    # Validate shapes
    if n_genes != E.shape[0]:
        raise ValueError(f"Gene count mismatch: X has {n_genes}, E has {E.shape[0]}")
    if n_genes != len(emb_mask):
        raise ValueError(f"Gene count mismatch: X has {n_genes}, emb_mask has {len(emb_mask)}")
    
    # Apply emb_mask: zero out genes without embedding
    X_masked = X.copy()
    X_masked[:, ~emb_mask] = 0.0
    
    # Weight transform: log1p
    W = np.log1p(X_masked)  # (n_spots, n_genes)
    
    # Compute weighted sum: W @ E
    weighted_sum = W @ E  # (n_spots, emb_dim)
    
    # Compute weight sums for normalization
    weight_sums = W.sum(axis=1, keepdims=True)  # (n_spots, 1)
    
    # Normalize
    S = weighted_sum / (weight_sums + eps)
    
    # Stats
    zero_weight_spots = int((weight_sums.flatten() < eps).sum())
    print(f"[INFO] Spot embeddings shape: {S.shape}")
    print(f"[INFO] Zero-weight spots (will have near-zero embedding): {zero_weight_spots}/{n_spots}")
    print(f"[INFO] Spot embedding stats: min={S.min():.6f}, max={S.max():.6f}, mean={S.mean():.6f}")
    
    return S.astype(np.float32)


def build_cosine_knn_fadj(S: np.ndarray, k: int) -> sp.coo_matrix:
    """
    Build cosine-kNN adjacency matrix from spot embeddings.
    
    Args:
        S: spot embeddings (n_spots, emb_dim)
        k: number of neighbors
    
    Returns:
        fadj: sparse adjacency (n_spots, n_spots), symmetric, 0/1 connectivity
    """
    n_spots = S.shape[0]
    print(f"[INFO] Building cosine-kNN graph with k={k}")
    
    # Build kNN graph (k+1 to include self, then remove diagonal)
    A = kneighbors_graph(S, k + 1, mode="connectivity", metric="cosine", include_self=True)
    A = A.toarray()
    
    # Remove self-loops
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    
    # Symmetrize: A = A + A^T, then clip to 0/1
    A = A + A.T
    A = np.where(A > 1, 1, A)
    
    # Convert to sparse
    fadj = sp.coo_matrix(A, dtype=np.float32)
    
    # Stats
    n_edges = fadj.nnz
    avg_degree = n_edges / n_spots
    print(f"[INFO] fadj: {n_spots} nodes, {n_edges} edges (after symmetrization)")
    print(f"[INFO] Average degree: {avg_degree:.2f}")
    
    return fadj


def main():
    parser = argparse.ArgumentParser(
        description="Build fadj from GRN gene embeddings + expression (cosine-kNN on spot embeddings)."
    )
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset ID (e.g., 151672)")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of neighbors for kNN (should match config.k)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing data.h5ad and grn_gene_embeddings.pt. "
                             "Default: data/generated/{dataset_id}/")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing fadj file if it exists")
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = get_script_dir()
    
    if args.data_dir is None:
        data_dir = os.path.join(script_dir, "data", "generated", args.dataset_id)
    else:
        data_dir = args.data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(script_dir, data_dir)
    
    h5ad_path = os.path.join(data_dir, "data.h5ad")
    out_path = os.path.join(data_dir, f"fadj_spotemb_grn_k{args.k}.npz")
    
    # Validate
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"data.h5ad not found: {h5ad_path}")
    
    print("=" * 60)
    print(f"Building fadj from GRN gene embeddings for dataset: {args.dataset_id}")
    print(f"k: {args.k}")
    print(f"Data dir: {data_dir}")
    print(f"Output: {out_path}")
    print("=" * 60)
    
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[SKIP] Output already exists: {out_path}")
        print(f"       Use --overwrite to regenerate.")
        return
    
    # Load expression matrix
    X = load_expression_matrix(h5ad_path)
    
    # Load GRN gene embeddings
    E, emb_mask = load_gene_embeddings_grn(data_dir)
    
    # Compute spot embeddings
    S = compute_spot_embeddings(X, E, emb_mask)
    
    # Build fadj
    fadj = build_cosine_knn_fadj(S, args.k)
    
    # Save
    sp.save_npz(out_path, fadj)
    print(f"[INFO] Saved: {out_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
