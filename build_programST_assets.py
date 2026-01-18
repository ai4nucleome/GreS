#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build programST assets from:
  - h5ad (3000 genes, ordered)
  - vocab.json (gene -> vocab_idx)
  - embedding.pt (Tensor [V,d] or dict containing such Tensor)
  - weighted_networks_nsga2r_final.rds (GR network for subgraph)

Outputs:
  1) programST_index.json
  2) programST_tensors.pt
  3) gr_subgraph.pt (optional, if --gr_rds is provided)

Notes:
  - Input files are treated as read-only. This script only writes to --out_dir.

Examples:
  # Basic usage (2 files)
  python build_programST_assets.py --h5ad /path/to/data.h5ad --vocab /path/to/vocab.json \\
    --embedding /path/to/embedding.pt --out_dir /path/to/out --overwrite

  # With GR subgraph (3 files)
  python build_programST_assets.py --h5ad /path/to/data.h5ad --vocab /path/to/vocab.json \\
    --embedding /path/to/embedding.pt --out_dir /path/to/out \\
    --gr_rds /path/to/weighted_networks_nsga2r_final.rds --overwrite
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
import anndata as ad
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


def setup_logging(out_dir: str, log_name: str = "build_assets.log") -> logging.Logger:
    """Setup logging to both console and file."""
    log_path = os.path.join(out_dir, log_name)
    
    # Create logger
    logger = logging.getLogger("build_programST_assets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    
    # Format
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def _norm_gene(g: str) -> str:
    """Normalize gene symbol for matching across h5ad/vocab."""
    return str(g).strip().lower()


def load_vocab_lower(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    vocab_lower_to_idx: Dict[str, int] = {}
    for tok, idx in vocab_json.items():
        low = _norm_gene(tok)
        # keep first occurrence if duplicates exist
        if low not in vocab_lower_to_idx:
            vocab_lower_to_idx[low] = int(idx)
    return vocab_lower_to_idx


def load_embedding_matrix(embedding_pt: str) -> torch.Tensor:
    obj = torch.load(embedding_pt, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        E_full = obj
    elif isinstance(obj, dict):
        # Try common keys
        for k in ["weight", "weights", "embeddings", "embedding", "gene_embeddings", "pretrained_weights"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                E_full = obj[k]
                break
        else:
            # If it's a state_dict-like dict with "gene_embedding_layer.weight"
            for k in ["gene_embedding_layer.weight", "embedding.weight", "emb.weight"]:
                if k in obj and isinstance(obj[k], torch.Tensor):
                    E_full = obj[k]
                    break
            else:
                raise ValueError(
                    f"embedding.pt is a dict but no known tensor key found. "
                    f"Available keys: {list(obj.keys())[:30]}"
                )
    else:
        raise ValueError(f"Unsupported embedding.pt type: {type(obj)}")

    if E_full.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D [V,d], got shape {tuple(E_full.shape)}")
    return E_full.contiguous()


def build_local_maps(genes_ordered: list, vocab_lower_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns:
      local2vocab_idx: int64 array len=3000, -1 for missing
      emb_mask: bool array len=3000
      gene2local_lower: normalized gene -> local idx
    """
    n = len(genes_ordered)
    local2vocab = np.full((n,), -1, dtype=np.int64)
    emb_mask = np.zeros((n,), dtype=bool)
    gene2local_lower: Dict[str, int] = {}

    for i, g in enumerate(genes_ordered):
        gl = _norm_gene(g)
        gene2local_lower[gl] = i
        vid = vocab_lower_to_idx.get(gl, -1)
        local2vocab[i] = int(vid)
        emb_mask[i] = (vid >= 0)

    return local2vocab, emb_mask, gene2local_lower


def extract_E_use(E_full: torch.Tensor, local2vocab: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """
    Build E_use[3000,d] aligned to local order.
    Missing indices (-1) => zero row.
    """
    V, d = E_full.shape
    n = local2vocab.shape[0]
    E_use = torch.zeros((n, d), dtype=dtype)

    valid = local2vocab >= 0
    if valid.any():
        idx = torch.from_numpy(local2vocab[valid].astype(np.int64))
        if idx.max().item() >= V:
            raise ValueError(f"vocab idx out of range: max idx {idx.max().item()} >= V {V}")
        E_use[torch.from_numpy(np.where(valid)[0].astype(np.int64))] = E_full[idx].to(dtype)
    return E_use


def load_gr_edges_from_rds(rds_path: str) -> pd.DataFrame:
    """
    Load 'gr' DataFrame from RDS list file.
    Returns DataFrame with columns: from, to, weight
    """
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    r_list = readRDS(rds_path)
    df_gr = pandas2ri.rpy2py(r_list.rx2('gr'))
    if not isinstance(df_gr, pd.DataFrame):
        df_gr = pd.DataFrame(df_gr)
    # 标准化列名为小写
    df_gr.columns = [c.lower() for c in df_gr.columns]
    return df_gr


def build_gr_subgraph(
    df_gr: pd.DataFrame,
    gene2local_lower: Dict[str, int],
    emb_mask: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, set]:
    """
    Filter GR edges and build bidirectional edge_index/edge_weight.
    
    Filter rules:
      - from/to both in gene2local_lower
      - from/to both have emb_mask=1
      - weight >= 0 and not NaN
    
    Returns:
      edge_index: [2, E*2] after bidirectional
      edge_weight: [E*2]
      unique_nodes: set of local indices of genes in subgraph
    """
    u_list, v_list, w_list = [], [], []
    
    for _, row in df_gr.iterrows():
        src = _norm_gene(str(row['from']))
        dst = _norm_gene(str(row['to']))
        w = row['weight']
        
        # 检查 from/to 是否在 genes 里
        if src not in gene2local_lower or dst not in gene2local_lower:
            continue
        
        u = gene2local_lower[src]
        v = gene2local_lower[dst]
        
        # 检查 embedding 是否可用
        if not emb_mask[u] or not emb_mask[v]:
            continue
        
        # 检查 weight 非负、非 NaN
        if pd.isna(w) or w < 0:
            continue
        
        u_list.append(u)
        v_list.append(v)
        w_list.append(float(w))
    
    # 双向化: 添加反向边
    u_bi = u_list + v_list
    v_bi = v_list + u_list
    w_bi = w_list + w_list
    
    edge_index = torch.tensor([u_bi, v_bi], dtype=torch.long)
    edge_weight = torch.tensor(w_bi, dtype=torch.float32)
    
    unique_nodes = set(u_bi + v_bi)
    
    return edge_index, edge_weight, unique_nodes


def log_stats(
    logger: logging.Logger,
    genes: List[str],
    emb_mask: np.ndarray,
    edge_count_before_bidir: int,
    edge_count_after_bidir: int,
    nodes_in_subgraph: int
) -> None:
    """Log detailed statistics."""
    n_total = len(genes)
    n_with_emb = int(emb_mask.sum())
    n_without_emb = n_total - n_with_emb
    
    logger.info(f"[STATS] h5ad 总基因数: {n_total}")
    logger.info(f"[STATS] 有 embedding 的基因数: {n_with_emb}")
    logger.info(f"[STATS] 无 embedding 的基因数: {n_without_emb}")
    
    # 列出无 embedding 的基因
    missing_genes = [genes[i] for i in range(n_total) if not emb_mask[i]]
    logger.info(f"[STATS] 无 embedding 的基因列表: {missing_genes}")
    
    logger.info(f"[STATS] GR 子图过滤后边数 (单向): {edge_count_before_bidir}")
    logger.info(f"[STATS] GR 子图双向化后边数: {edge_count_after_bidir}")
    logger.info(f"[STATS] GR 子图中参与的基因数: {nodes_in_subgraph}")


def _safe_tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Basic stats for a 1D tensor x (float). Returns python floats.
    """
    if x.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    x = x.detach()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _quantiles(x: torch.Tensor, qs: List[float]) -> Dict[float, float]:
    """
    Quantiles for a 1D tensor x. Returns {q: value}.
    Uses torch.quantile when available; otherwise falls back to numpy.
    """
    x = x.detach().flatten()
    if x.numel() == 0:
        return {q: float("nan") for q in qs}
    try:
        qv = torch.quantile(x, torch.tensor(qs, device=x.device, dtype=torch.float32)).cpu().numpy().tolist()
        return {q: float(v) for q, v in zip(qs, qv)}
    except Exception:
        arr = x.cpu().numpy()
        qv = np.quantile(arr, qs).tolist()
        return {q: float(v) for q, v in zip(qs, qv)}


def preprocess_edge_weight(
    w: torch.Tensor,
    winsor_q: float = 0.99,
    transform: str = "log1p",
    scale: str = "minmax",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Preprocess GR edge weights for stable diffusion.
    - winsorize at quantile winsor_q (set winsor_q<=0 or >=1 to disable)
    - transform: none|log1p|sqrt
    - scale: none|minmax  (minmax -> [0,1])
    """
    w = w.clone()
    # Ensure finite, non-negative
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.clamp(w, min=0.0)

    # 1) winsorize
    if 0.0 < winsor_q < 1.0 and w.numel() > 0:
        qv = float(_quantiles(w, [winsor_q])[winsor_q])
        w = torch.clamp(w, max=qv)

    # 2) monotonic transform
    transform = str(transform).lower()
    if transform == "log1p":
        w = torch.log1p(w)
    elif transform == "sqrt":
        w = torch.sqrt(w)
    elif transform == "none":
        pass
    else:
        raise ValueError(f"Unknown transform: {transform}. Choose from: none|log1p|sqrt")

    # 3) scale
    scale = str(scale).lower()
    if scale == "minmax":
        wmin = w.min() if w.numel() > 0 else torch.tensor(0.0, device=w.device)
        wmax = w.max() if w.numel() > 0 else torch.tensor(1.0, device=w.device)
        w = (w - wmin) / (wmax - wmin + eps)
    elif scale == "none":
        pass
    else:
        raise ValueError(f"Unknown scale: {scale}. Choose from: none|minmax")

    return w


def build_anorm_sparse(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    self_loop_lambda: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Build A_norm = D^{-1/2} (A + lambda I) D^{-1/2} as torch.sparse COO.
    Returns (A_norm, stats) where stats contains degree/value summaries.
    """
    device = edge_index.device
    # A (coalesce sums duplicates)
    A = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()

    # Add self-loops
    if self_loop_lambda != 0.0:
        idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        I = torch.sparse_coo_tensor(
            torch.stack([idx, idx], dim=0),
            torch.full((num_nodes,), float(self_loop_lambda), device=device, dtype=torch.float32),
            size=(num_nodes, num_nodes),
            device=device,
        ).coalesce()
        A_tilde = (A + I).coalesce()
    else:
        A_tilde = A

    # Degree
    deg = torch.sparse.sum(A_tilde, dim=1).to_dense().to(torch.float32)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    deg_inv_sqrt = torch.nan_to_num(deg_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize values: w' = w * d^{-1/2}[u] * d^{-1/2}[v]
    idx = A_tilde.indices()
    val = A_tilde.values()
    u = idx[0]
    v = idx[1]
    val_norm = val * deg_inv_sqrt[u] * deg_inv_sqrt[v]
    A_norm = torch.sparse_coo_tensor(idx, val_norm, size=A_tilde.shape, device=device).coalesce()

    stats = {
        "deg_min": float(deg.min().item()) if deg.numel() > 0 else float("nan"),
        "deg_mean": float(deg.mean().item()) if deg.numel() > 0 else float("nan"),
        "deg_max": float(deg.max().item()) if deg.numel() > 0 else float("nan"),
        "anorm_val_min": float(A_norm.values().min().item()) if A_norm._nnz() > 0 else float("nan"),
        "anorm_val_mean": float(A_norm.values().mean().item()) if A_norm._nnz() > 0 else float("nan"),
        "anorm_val_max": float(A_norm.values().max().item()) if A_norm._nnz() > 0 else float("nan"),
    }
    return A_norm, stats


def grn_one_step_diffusion(
    E: torch.Tensor,
    emb_mask: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    alpha: float = 0.2,
    self_loop_lambda: float = 1.0,
    do_weight_preprocess: bool = True,
    winsor_q: float = 0.99,
    transform: str = "log1p",
    scale: str = "minmax",
    logger: logging.Logger = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Deterministic 1-step GRN diffusion with residual:
      E_grn = (1-alpha) E + alpha (A_norm E)

    Notes:
      - edge_weight is only used to build A_norm (not used as any loss)
      - rows with emb_mask==0 will be forced to 0 in the output (as user requested)
    """
    assert E.ndim == 2, "E must be [G,d]"
    G = E.shape[0]
    device = E.device

    w_raw = edge_weight.to(torch.float32)
    w_proc = w_raw
    if do_weight_preprocess:
        w_proc = preprocess_edge_weight(w_proc, winsor_q=winsor_q, transform=transform, scale=scale)

    # Stats (weights)
    qs = [0.5, 0.9, 0.95, 0.99, 0.995]
    w_raw_q = _quantiles(w_raw.detach().cpu(), qs)
    w_proc_q = _quantiles(w_proc.detach().cpu(), qs)

    # Build A_norm
    A_norm, anorm_stats = build_anorm_sparse(
        num_nodes=G,
        edge_index=edge_index.to(device),
        edge_weight=w_proc.to(device),
        self_loop_lambda=self_loop_lambda,
    )

    # Diffuse: M = A_norm @ E
    M = torch.sparse.mm(A_norm, E)
    E_grn = (1.0 - float(alpha)) * E + float(alpha) * M

    # Force missing-embedding genes to 0
    emb_mask_t = torch.from_numpy(emb_mask.astype(np.int64)).to(device)
    missing_idx = torch.where(emb_mask_t == 0)[0]
    if missing_idx.numel() > 0:
        E_grn[missing_idx] = 0.0

    # Delta stats (on emb_mask==1 only)
    keep_idx = torch.where(emb_mask_t == 1)[0]
    delta = (E_grn - E)
    if keep_idx.numel() > 0:
        delta_norm = torch.norm(delta[keep_idx], dim=1)
        # cosine similarity between E and E_grn
        e1 = E[keep_idx]
        e2 = E_grn[keep_idx]
        denom = (torch.norm(e1, dim=1) * torch.norm(e2, dim=1)).clamp_min(1e-12)
        cos = (e1 * e2).sum(dim=1) / denom
        delta_q = _quantiles(delta_norm.detach().cpu(), [0.5, 0.9, 0.99])
        cos_q = _quantiles(cos.detach().cpu(), [0.1, 0.5])
        delta_mean = float(delta_norm.mean().item())
        cos_mean = float(cos.mean().item())
    else:
        delta_mean, cos_mean = float("nan"), float("nan")
        delta_q, cos_q = {}, {}

    # missing rows max abs
    if missing_idx.numel() > 0:
        missing_max_abs = float(E_grn[missing_idx].abs().max().item())
    else:
        missing_max_abs = 0.0

    stats = {
        "alpha": float(alpha),
        "self_loop_lambda": float(self_loop_lambda),
        "w_raw_min": float(w_raw.min().item()) if w_raw.numel() > 0 else float("nan"),
        "w_raw_max": float(w_raw.max().item()) if w_raw.numel() > 0 else float("nan"),
        "w_proc_min": float(w_proc.min().item()) if w_proc.numel() > 0 else float("nan"),
        "w_proc_max": float(w_proc.max().item()) if w_proc.numel() > 0 else float("nan"),
        "w_raw_quantiles": {str(k): float(v) for k, v in w_raw_q.items()},
        "w_proc_quantiles": {str(k): float(v) for k, v in w_proc_q.items()},
        "missing_rows_max_abs": float(missing_max_abs),
        "delta_norm_mean(emb_mask==1)": float(delta_mean),
        "delta_norm_quantiles(emb_mask==1)": {str(k): float(v) for k, v in delta_q.items()},
        "cosine_mean(emb_mask==1)": float(cos_mean),
        "cosine_quantiles(emb_mask==1)": {str(k): float(v) for k, v in cos_q.items()},
    }
    stats.update(anorm_stats)

    if logger is not None:
        logger.info("[GRN_DIFFUSION] enabled=True")
        logger.info(f"[GRN_DIFFUSION] alpha={alpha} | self_loop_lambda={self_loop_lambda}")
        logger.info(
            f"[GRN_DIFFUSION] weight_preprocess={do_weight_preprocess} "
            f"| winsor_q={winsor_q} | transform={transform} | scale={scale}"
        )
        logger.info(
            f"[GRN_DIFFUSION] edge_weight raw quantiles "
            f"p50={w_raw_q.get(0.5):.6g}, p90={w_raw_q.get(0.9):.6g}, p95={w_raw_q.get(0.95):.6g}, "
            f"p99={w_raw_q.get(0.99):.6g}, p99.5={w_raw_q.get(0.995):.6g}, max={stats['w_raw_max']:.6g}"
        )
        if do_weight_preprocess:
            logger.info(
                f"[GRN_DIFFUSION] edge_weight processed quantiles "
                f"p50={w_proc_q.get(0.5):.6g}, p90={w_proc_q.get(0.9):.6g}, p95={w_proc_q.get(0.95):.6g}, "
                f"p99={w_proc_q.get(0.99):.6g}, p99.5={w_proc_q.get(0.995):.6g}, max={stats['w_proc_max']:.6g}"
            )
        logger.info(
            f"[GRN_DIFFUSION] degree(after self-loop): min={anorm_stats['deg_min']:.6g}, "
            f"mean={anorm_stats['deg_mean']:.6g}, max={anorm_stats['deg_max']:.6g}"
        )
        logger.info(
            f"[GRN_DIFFUSION] A_norm values: min={anorm_stats['anorm_val_min']:.6g}, "
            f"mean={anorm_stats['anorm_val_mean']:.6g}, max={anorm_stats['anorm_val_max']:.6g}"
        )
        logger.info(f"[GRN_DIFFUSION] E_grn shape={tuple(E_grn.shape)}")
        logger.info(f"[GRN_DIFFUSION] missing rows max_abs={missing_max_abs:.6g}")
        if keep_idx.numel() > 0:
            logger.info(
                f"[GRN_DIFFUSION] delta_norm(emb_mask==1): mean={delta_mean:.6g}, "
                f"p50={delta_q.get(0.5, float('nan')):.6g}, p90={delta_q.get(0.9, float('nan')):.6g}, "
                f"p99={delta_q.get(0.99, float('nan')):.6g}"
            )
            logger.info(
                f"[GRN_DIFFUSION] cosine(E,E_grn) (emb_mask==1): mean={cos_mean:.6g}, "
                f"p10={cos_q.get(0.1, float('nan')):.6g}, p50={cos_q.get(0.5, float('nan')):.6g}"
            )

    return E_grn, stats


def save_index_json(
    out_path: str, 
    genes: list, 
    local2vocab: np.ndarray, 
    emb_mask: np.ndarray,
    grn_sub_genes: List[str] = None
) -> None:
    """
    Save index JSON with gene mappings and optional GRN_sub field.
    
    Args:
        grn_sub_genes: List of gene names that have embedding AND appear in GRN subgraph.
    """
    payload = {
        "genes": genes,
        "local2vocab_idx": local2vocab.tolist(),
        "emb_mask": emb_mask.astype(int).tolist(),  # 0/1 to be compact
    }
    
    # Add GRN_sub if provided
    if grn_sub_genes is not None:
        payload["GRN_sub"] = grn_sub_genes
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Build programST assets (2 files) from h5ad+vocab+embedding.")
    parser.add_argument("--h5ad", required=True, help="Path to processed h5ad (3000 genes).")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json (gene->idx).")
    parser.add_argument("--embedding", required=True, help="Path to embedding.pt (Tensor or dict containing Tensor).")
    
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--index_name", default="programST_index.json", help="Index json filename.")
    parser.add_argument("--tensor_name", default="programST_tensors.pt", help="Tensor pt filename.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist.")

    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"], help="dtype for E_use.")
    
    # GR subgraph options
    parser.add_argument("--gr_rds", default=None,
        help="Path to weighted_networks_nsga2r_final.rds (optional). If provided, builds gr_subgraph.pt")
    parser.add_argument("--gr_subgraph_name", default="gr_subgraph.pt",
        help="Output filename for GR subgraph.")

    # Optional: deterministic GRN diffusion (no training)
    parser.add_argument("--grn_diffuse", action="store_true",
        help="If set, run 1-step GRN diffusion with residual to produce grn_gene_embeddings.pt (no training).")
    parser.add_argument("--grn_out_name", default="grn_gene_embeddings.pt",
        help="Output filename for GRN-diffused gene embeddings (saved in out_dir).")
    parser.add_argument("--grn_alpha", type=float, default=0.2,
        help="Residual diffusion strength alpha in E_grn=(1-alpha)E + alpha(A_norm E). Default 0.2.")
    parser.add_argument("--grn_self_loop_lambda", type=float, default=1.0,
        help="Self-loop weight lambda in A_tilde=A+lambda*I. Default 1.0.")
    parser.add_argument("--grn_weight_preprocess", action="store_true", default=True,
        help="Whether to preprocess GRN edge weights before building A_norm. Default True.")
    parser.add_argument("--grn_winsor_q", type=float, default=0.99,
        help="Winsorize quantile for edge weights (e.g., 0.99). Set <=0 or >=1 to disable.")
    parser.add_argument("--grn_weight_transform", type=str, default="log1p", choices=["none", "log1p", "sqrt"],
        help="Monotonic transform for weights. Default log1p.")
    parser.add_argument("--grn_weight_scale", type=str, default="minmax", choices=["none", "minmax"],
        help="Scale method for weights after transform. Default minmax to [0,1].")
    
    # Logging options
    parser.add_argument("--log_name", default="build_assets.log",
        help="Log filename (saved in out_dir).")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.out_dir, args.log_name)
    logger.info(f"===== Build programST assets started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    logger.info(f"Arguments: {vars(args)}")

    # A) load h5ad, genes
    adata = ad.read_h5ad(args.h5ad)
    if adata.n_vars != 3000:
        logger.warning(f"adata.n_vars={adata.n_vars} (expected 3000). Continue anyway.")
    if not adata.var_names.is_unique:
        raise ValueError("adata.var_names is not unique; cannot build stable gene index mapping.")
    genes = [str(g) for g in list(adata.var_names)]

    # B) vocab mapping
    vocab_lower_to_idx = load_vocab_lower(args.vocab)
    local2vocab, emb_mask, gene2local_lower = build_local_maps(genes, vocab_lower_to_idx)
    missing = np.where(local2vocab < 0)[0]
    logger.info(f"genes={len(genes)} | missing_embedding={len(missing)}")

    # C) embedding -> E_use
    E_full = load_embedding_matrix(args.embedding)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    E_use = extract_E_use(E_full, local2vocab, dtype=dtype)
    logger.info(f"E_full shape={tuple(E_full.shape)} | E_use shape={tuple(E_use.shape)}")

    # F) prepare output paths
    index_path = os.path.join(args.out_dir, args.index_name)
    tensor_path = os.path.join(args.out_dir, args.tensor_name)

    if (not args.overwrite) and (os.path.exists(index_path) or os.path.exists(tensor_path)):
        raise FileExistsError(
            f"Output already exists. Use --overwrite to overwrite:\n  - {index_path}\n  - {tensor_path}"
        )

    # ===== GR Subgraph (optional) =====
    grn_sub_genes = None  # Will be set if gr_rds is provided
    gr_edge_index = None
    gr_edge_weight = None
    gr_nodes_in_subgraph = 0
    
    if args.gr_rds:
        logger.info(f"Loading GR edges from: {args.gr_rds}")
        df_gr = load_gr_edges_from_rds(args.gr_rds)
        logger.info(f"GR 原始边数: {len(df_gr)}")
        
        edge_index, edge_weight, unique_nodes = build_gr_subgraph(
            df_gr, gene2local_lower, emb_mask
        )
        
        nodes_in_subgraph = len(unique_nodes)
        gr_nodes_in_subgraph = nodes_in_subgraph
        edge_count_before = edge_index.shape[1] // 2
        edge_count_after = edge_index.shape[1]
        gr_edge_index = edge_index
        gr_edge_weight = edge_weight
        
        # 打印统计日志
        log_stats(logger, genes, emb_mask, edge_count_before, edge_count_after, nodes_in_subgraph)
        
        # 获取 GRN_sub: 有 embedding 且在 GRN 子图中出现的基因名列表
        grn_sub_genes = sorted([genes[idx] for idx in unique_nodes])
        logger.info(f"[STATS] GRN_sub 基因数: {len(grn_sub_genes)}")
        
        # 保存 gr_subgraph.pt
        subgraph_path = os.path.join(args.out_dir, args.gr_subgraph_name)
        torch.save({
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': len(genes),
            'num_nodes_in_subgraph': nodes_in_subgraph,
        }, subgraph_path)
        logger.info(f"Saved GR subgraph: {subgraph_path}")

    # G) save index json (with optional GRN_sub)
    save_index_json(index_path, genes, local2vocab, emb_mask, grn_sub_genes)
    logger.info(f"Saved: {index_path}")

    # H) save tensors
    tensors = {
        "E_use": E_use.cpu(),
    }
    torch.save(tensors, tensor_path)
    logger.info(f"Saved: {tensor_path}")

    # ===== Optional: GRN diffusion (no training) =====
    if args.grn_diffuse:
        if gr_edge_index is None or gr_edge_weight is None:
            raise ValueError("--grn_diffuse requires --gr_rds to be provided (to build GR subgraph edges).")

        isolated_in_hvg = len(genes) - int(gr_nodes_in_subgraph)
        logger.info(
            f"[GRN_DIFFUSION] HVG genes={len(genes)} | nodes_in_subgraph={gr_nodes_in_subgraph} "
            f"| isolated_in_hvg={isolated_in_hvg}"
        )

        E_grn, grn_stats = grn_one_step_diffusion(
            E=E_use.to(torch.float32),
            emb_mask=emb_mask,
            edge_index=gr_edge_index,
            edge_weight=gr_edge_weight,
            alpha=float(args.grn_alpha),
            self_loop_lambda=float(args.grn_self_loop_lambda),
            do_weight_preprocess=bool(args.grn_weight_preprocess),
            winsor_q=float(args.grn_winsor_q),
            transform=str(args.grn_weight_transform),
            scale=str(args.grn_weight_scale),
            logger=logger,
        )

        grn_out_path = os.path.join(args.out_dir, args.grn_out_name)
        torch.save(
            {
                "genes": genes,
                "E_grn": E_grn.cpu(),
                "emb_mask": torch.from_numpy(emb_mask.astype(np.int64)).cpu(),
                "meta": {
                    "alpha": float(args.grn_alpha),
                    "self_loop_lambda": float(args.grn_self_loop_lambda),
                    "weight_preprocess": bool(args.grn_weight_preprocess),
                    "winsor_q": float(args.grn_winsor_q),
                    "weight_transform": str(args.grn_weight_transform),
                    "weight_scale": str(args.grn_weight_scale),
                    "num_nodes": int(len(genes)),
                    "num_nodes_in_subgraph": int(gr_nodes_in_subgraph),
                    "isolated_in_hvg": int(isolated_in_hvg),
                    "num_edges_bidir": int(gr_edge_index.shape[1]),
                },
                "stats": grn_stats,
            },
            grn_out_path,
        )
        logger.info(f"[GRN_DIFFUSION] Saved GRN-diffused embedding: {grn_out_path}")
    
    logger.info(f"===== Build completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")


if __name__ == "__main__":
    main()
