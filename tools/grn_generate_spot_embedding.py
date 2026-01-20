#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate spot embeddings from GRN-diffused gene embeddings.

For each spot, it selects topk genes (only from emb_mask=True) based on expression,
then computes a weighted average of gene embeddings using expression as weights.

Output:
  - embeddings_{dataset}.npy: spot embeddings (n_spots x emb_dim)

Usage:
  python grn_generate_spot_embedding.py --dataset_id 151672
  python grn_generate_spot_embedding.py --dataset_id 151672 --topk 50
"""

import os
import argparse
import numpy as np
import scanpy as sc
import torch


def load_gene_embeddings_grn(pt_path: str):
    """
    Load gene embeddings from grn_gene_embeddings.pt (GRN-diffused).
    
    Returns:
        genes: list of gene names
        E: np.ndarray of shape (n_genes, emb_dim)
        emb_mask: np.ndarray of shape (n_genes,), bool
    """
    print(f"[INFO] Loading GRN gene embeddings: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    genes = obj["genes"]
    E = obj["E_grn"].cpu().numpy().astype(np.float32)
    emb_mask = obj["emb_mask"].cpu().numpy().astype(bool)
    return genes, E, emb_mask


def generate_spot_embeddings(
    h5ad_path: str,
    genes: list,
    E: np.ndarray,
    emb_mask: np.ndarray,
    topk: int = 30,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Generate spot embeddings from gene embeddings.
    
    Args:
        h5ad_path: Path to data.h5ad (expression matrix).
        genes: List of gene names (must match adata.var_names order).
        E: Gene embedding matrix, shape (n_genes, emb_dim).
        emb_mask: Boolean mask, shape (n_genes,). True = has embedding.
        topk: Number of top genes to select per spot.
        eps: Small constant to avoid division by zero.
    
    Returns:
        spot_embs: np.ndarray of shape (n_spots, emb_dim).
    """
    # 1. Load h5ad
    print(f"[INFO] Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    X = X.astype(np.float32)
    n_spots, n_genes = X.shape
    print(f"[INFO] X shape: {X.shape}")
    
    emb_dim = E.shape[1]
    print(f"[INFO] E shape: {E.shape}")
    print(f"[INFO] emb_mask True count: {int(emb_mask.sum())} / {len(emb_mask)}")
    
    # 2. Validate gene order consistency
    genes_h5ad = adata.var_names.astype(str).tolist()
    if genes_h5ad != genes:
        raise ValueError(
            f"Gene order mismatch! adata.var_names != genes from embedding source.\n"
            f"First 5 h5ad: {genes_h5ad[:5]}\n"
            f"First 5 source: {genes[:5]}"
        )
    print("[INFO] Gene order validated: h5ad matches embedding source")
    
    # 3. Shape validation
    if n_genes != len(genes):
        raise ValueError(f"Gene count mismatch: h5ad has {n_genes}, source has {len(genes)}")
    if n_genes != E.shape[0]:
        raise ValueError(f"Gene count mismatch: h5ad has {n_genes}, E has {E.shape[0]}")
    
    # 4. Select topk in emb_mask=True and compute weighted average
    # If topk <= 0: use ALL emb_mask=True genes (no top-k selection)
    if topk is None or topk <= 0:
        X_mask = X.copy()
        X_mask[:, ~emb_mask] = 0.0
        weight_sums = X_mask.sum(axis=1)  # (n_spots,)
        zero_weight_spots = int((weight_sums == 0).sum())

        print(f"[INFO] topk: ALL (use all emb_mask=True genes)")
        print(f"[INFO] zero_weight_sum_spots: {zero_weight_spots} / {n_spots}")

        # Vectorized weighted average: (X_mask @ E) / sum
        spot_embs = (X_mask @ E).astype(np.float32)
        spot_embs = spot_embs / (weight_sums[:, None].astype(np.float32) + eps)
    else:
        # Mask out emb_mask=False genes by setting them to -inf for topk selection
        X_masked = X.copy()
        X_masked[:, ~emb_mask] = -np.inf

        # If topk larger than available masked genes, just take all masked genes
        masked_gene_count = int(emb_mask.sum())
        topk_eff = min(int(topk), masked_gene_count)

        # Get topk indices per spot
        topk_idx = np.argsort(X_masked, axis=1)[:, -topk_eff:]  # (n_spots, topk_eff)

        # Get weights from original X (not masked)
        row_idx = np.arange(n_spots)[:, None]  # (n_spots, 1)
        topk_weights = X[row_idx, topk_idx]     # (n_spots, topk_eff)

        # Compute weight sums
        weight_sums = topk_weights.sum(axis=1)  # (n_spots,)
        zero_weight_spots = int((weight_sums == 0).sum())

        print(f"[INFO] topk: {topk_eff}")
        print(f"[INFO] zero_weight_sum_spots: {zero_weight_spots} / {n_spots}")

        # 5. Compute weighted average (keep loop for memory safety)
        spot_embs = np.zeros((n_spots, emb_dim), dtype=np.float32)
        for i in range(n_spots):
            idx = topk_idx[i]  # (topk_eff,)
            w = topk_weights[i]  # (topk_eff,)
            denom = float(w.sum()) + eps
            spot_embs[i] = (w[:, None] * E[idx]).sum(axis=0) / denom
    
    # 6. Print statistics
    print(f"[INFO] spot_embs shape: {spot_embs.shape}")
    print(f"[INFO] spot_embs min: {spot_embs.min():.6f}, max: {spot_embs.max():.6f}, mean: {spot_embs.mean():.6f}")
    
    return spot_embs


def main():
    parser = argparse.ArgumentParser(
        description="Generate spot embeddings from GRN-diffused gene embeddings."
    )
    parser.add_argument("--dataset_id", type=str, required=True, 
                        help="Dataset ID (e.g., 151672)")
    parser.add_argument("--topk", type=int, default=30, 
                        help="Number of top genes per spot (default: 30). Use <=0 to use all emb_mask=True genes.")
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Directory containing data.h5ad and grn_gene_embeddings.pt. "
                             "Default: data/generated/{dataset_id}/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: data/npys_grn/")
    
    args = parser.parse_args()
    
    dataset_id = args.dataset_id
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.data_dir is None:
        data_dir = os.path.join(script_dir, "../data", "generated", dataset_id)
    else:
        data_dir = args.data_dir
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(script_dir, data_dir)
    
    # Output directory
    if args.out_dir is None:
        out_dir = os.path.join(script_dir, "../data", "npys_grn")
    else:
        out_dir = args.out_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(script_dir, out_dir)
    
    h5ad_path = os.path.join(data_dir, "data.h5ad")
    out_path = os.path.join(out_dir, f"embeddings_{dataset_id}.npy")
    
    # Validate h5ad exists
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"h5ad not found: {h5ad_path}")
    
    # Load GRN gene embeddings
    pt_path = os.path.join(data_dir, "grn_gene_embeddings.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"grn_gene_embeddings.pt not found: {pt_path}")
    genes, E, emb_mask = load_gene_embeddings_grn(pt_path)
    
    # Create output directory if needed (always overwrite)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Generating spot embeddings for dataset: {dataset_id}")
    print("=" * 60)
    print(f"h5ad: {h5ad_path}")
    print(f"embedding file: {pt_path}")
    print(f"output: {out_path}")
    print(f"topk: {args.topk}")
    print("=" * 60)
    
    # Generate embeddings
    spot_embs = generate_spot_embeddings(
        h5ad_path=h5ad_path,
        genes=genes,
        E=E,
        emb_mask=emb_mask,
        topk=args.topk,
    )
    
    # Save (always overwrite)
    np.save(out_path, spot_embs)
    print(f"[INFO] Saved: {out_path}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
