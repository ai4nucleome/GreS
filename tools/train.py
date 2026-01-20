from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import warnings
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
import scanpy as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from models import GreS
from preprocess.config import Config
from utils import (
    regularization_loss,
    regularization_loss_sparse,
    dicr_loss,
    sparse_mx_to_torch_sparse_tensor,
    normalize_sparse_matrix,
    run_precomputation_llm_base,
    hungarian_match,
    ZINB,
)

warnings.filterwarnings("ignore")


# SCRIPT_DIR and PROJECT_ROOT setup moved to top


def evaluate_kmeans_repeated(emb, labels, n_clusters, repeats=1, seed_base=0):
    
    Args:
        emb: embedding matrix (n_samples, n_features)
        labels: ground truth labels
        n_clusters: number of clusters
        repeats: number of KMeans runs
        seed_base: base seed (run i uses seed_base + i)
    
    Returns:
        dict with best metrics and corresponding idx
    """
    best_ari = -1
    best_result = None
    
    for i in range(repeats):
        seed_i = seed_base + i
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed_i).fit(emb)
        idx = kmeans.labels_
        
        ari = metrics.adjusted_rand_score(labels, idx) * 100
        nmi = metrics.normalized_mutual_info_score(labels, idx) * 100
        ami = metrics.adjusted_mutual_info_score(labels, idx) * 100
        
        if ari > best_ari:
            best_ari = ari
            best_result = {
                'idx': idx,
                'ari': ari,
                'nmi': nmi,
                'ami': ami,
                'kmeans_seed': seed_i,
            }
    
    return best_result


def _resolve_under_project_root(p: str) -> str:
    """
    Resolve a user-provided path. If it's relative, make it relative to the project root.
    """
    if p is None:
        return p
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def _assert_path_inside_project(p: str, description: str = "path") -> None:
    """
    Safety guard: raise an error if the resolved path would write outside the project.
    """
    if p is None:
        return
    real_p = os.path.realpath(p)
    real_root = os.path.realpath(PROJECT_ROOT)
    if not (real_p == real_root or real_p.startswith(real_root + os.sep)):
        raise RuntimeError(
            f"[PATH GUARD] {description} would write outside project!\n"
            f"  path: {real_p}\n"
            f"  allowed root: {real_root}\n"
            f"Refusing to proceed. Please use a path inside the project."
        )


def load_cached_fadj(dataset_id: str, source: str, k: int, data_dir: str = None) -> sp.spmatrix:
    """
    Load pre-computed fadj from gene-embedding-based spot embeddings.
    
    Args:
        dataset_id: Dataset ID (e.g., "151672")
        source: "grn" (which gene embedding source was used)
        k: Number of neighbors (should match config.k)
    
    Returns:
        fadj: scipy sparse matrix (coo or csr)
    """
    if data_dir is None:
        fadj_dir = os.path.join(PROJECT_ROOT, "data", "generated", dataset_id)
    else:
        fadj_dir = data_dir
    fadj_path = os.path.join(fadj_dir, f"fadj_spotemb_{source}_k{k}.npz")
    if not os.path.exists(fadj_path):
        raise FileNotFoundError(
            f"Cached fadj not found: {fadj_path}\n"
            f"Please run: python build_fadj_from_geneemb.py --dataset_id {dataset_id} --k {k}"
        )
    print(f"[INFO] Loading cached fadj: {fadj_path}")
    fadj = sp.load_npz(fadj_path)
    print(f"[INFO] fadj shape: {fadj.shape}, nnz: {fadj.nnz}")
    return fadj


class Tee:
    """同时输出到终端和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_data(dataset, data_path_prefix="", data_dir: str = None): 
    """
    Load preprocessed h5ad data.
    
    Returns:
        adata: AnnData object
        features: torch.FloatTensor of normalized/scaled expression (for encoder)
        labels: ground truth labels
        nsadj: normalized spatial adjacency (torch sparse)
        graph_nei: neighbor graph for regularization
        graph_neg: negative samples for regularization
    
    Note: fadj is loaded separately from GRN-based spot embeddings (load_cached_fadj).
    """
    print("load data:")

    if data_dir is not None:
        path = os.path.join(data_dir, "data.h5ad")
    else:
        base_path = os.path.join(PROJECT_ROOT, "data", "generated")
        if data_path_prefix:
            path = os.path.join(base_path, data_path_prefix, dataset, "data.h5ad")
        else:
            path = os.path.join(base_path, dataset, "data.h5ad")

    adata = sc.read_h5ad(path)
    
    # Encoder input: normalized/scaled features
    if sp.issparse(adata.X):
        features = torch.FloatTensor(adata.X.toarray())
    else:
        features = torch.FloatTensor(adata.X)
    
    labels = adata.obs['ground']
    sadj = adata.obsm['sadj']
    
    # Normalize sadj (add self-loops, row-normalize) and convert to torch sparse
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    
    # graph_nei / graph_neg can be HUGE if stored as dense NxN matrices.
    # For large N we avoid materializing them as torch tensors and instead use a sparse regularization loss.
    graph_nei = None
    graph_neg = None
    if 'graph_nei' in adata.obsm and 'graph_neg' in adata.obsm:
        graph_nei_raw = adata.obsm['graph_nei']
        graph_neg_raw = adata.obsm['graph_neg']
        try:
            n0, n1 = int(graph_nei_raw.shape[0]), int(graph_nei_raw.shape[1])
            is_dense_square = (graph_nei_raw.ndim == 2 and n0 == n1)
            # Heuristic: dense NxN beyond this size is not safe on GPU
            if is_dense_square and n0 >= 10000:
                approx_gb = (n0 * n0 * 4) / (1024**3)
                print(
                    f"[WARNING] Detected dense graph_nei/graph_neg with shape=({n0},{n0}) "
                    f"(~{approx_gb:.1f}GB if float32). "
                    f"Will NOT load them; will use sparse adjacency for regularization loss."
                )
                graph_nei = None
                graph_neg = None
            else:
                graph_nei = torch.FloatTensor(graph_nei_raw)
                graph_neg = torch.FloatTensor(graph_neg_raw)
        except Exception as e:
            print(f"[WARNING] Failed to load graph_nei/graph_neg from adata.obsm due to: {e}. "
                  f"Will use sparse adjacency for regularization loss.")
            graph_nei = None
            graph_neg = None
    print("done")
    return adata, features, labels, nsadj, graph_nei, graph_neg


def train(model, features, sadj, fadj, precomputed_llm_base_embeddings,
          graph_nei, graph_neg, config, optimizer):
    """
    Single training step with ZINB reconstruction loss.
    
    Args:
        features: normalized/scaled input for encoder
        sadj: spatial adjacency (torch sparse)
        fadj: feature adjacency (torch sparse)
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    emb, pi, disp, mean, emb1, emb2 = model(features, sadj, fadj, precomputed_llm_base_embeddings)
    
    # ZINB reconstruction loss
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0.05).loss(features, mean, mean=True)
    
    # Regularization loss
    if graph_nei is not None and graph_neg is not None:
        reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    else:
        reg_loss = regularization_loss_sparse(emb, sadj, n_neg=getattr(config, "reg_n_neg", 5))
    
    # DICR loss
    dcir_loss_val = dicr_loss(emb1, emb2)

    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss + config.beta * dcir_loss_val
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        emb, _, _, mean, _, _ = model(features, sadj, fadj, precomputed_llm_base_embeddings)

    return emb, mean, zinb_loss, reg_loss, dcir_loss_val, total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=str, default='Mouse_Brain_Anterior')
    parser.add_argument('--use_llm', type=str, default='true', choices=['true', 'false'],
                        help='Whether to use LLM/semantic embeddings. Use "true" or "false". Default: true.')
    parser.add_argument('--llm_emb_dir', type=str, default='data/npys/', help='Directory for LLM embeddings.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override the default generated data directory.')
    parser.add_argument('--config_name', type=str, default=None,
                        help='Override which config ini to use (without .ini).')
    parser.add_argument('--result_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'result'),
                        help='Root directory to save results.')
    parser.add_argument('--run_name', type=str, default='default',
                        help='Sub-directory name under {result_dir}/{config}/{dataset_id}/ for this run.')
    parser.add_argument('--save_best_ckpt', action='store_true', default=True,
                        help='Whether to save the best checkpoint (selected by ARI).')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory to save checkpoints.')
    parser.add_argument('--llm_modulation_ratio', type=float, default=None,
                        help='Override llm_modulation_ratio from config ini.')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Override alpha (ZINB reconstruction loss weight) from config ini.')
    parser.add_argument('--beta', type=float, default=None,
                        help='Override beta (dicr_loss weight) from config ini.')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override gamma (regularization_loss weight) from config ini.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config ini.')
    # Model capacity / regularization overrides
    parser.add_argument('--nhid1', type=int, default=None,
                        help='Override nhid1 (first hidden layer size) from config ini.')
    parser.add_argument('--nhid2', type=int, default=None,
                        help='Override nhid2 (second hidden layer size) from config ini.')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout from config ini.')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override weight_decay from config ini.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed from config ini.')
    # KMeans evaluation settings
    parser.add_argument('--kmeans_repeats', type=int, default=1,
                        help='Number of KMeans runs per epoch evaluation (take max ARI). Default: 1.')
    parser.add_argument('--kmeans_seed_base', type=int, default=0,
                        help='Base seed for KMeans repeats.')
    # Unsupervised clustering mode
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Override number of clusters for KMeans. When set, enables unsupervised mode.')
    # Early stopping
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='Enable early stopping.')
    parser.add_argument('--early_stop_warmup', type=int, default=20,
                        help='Warmup epochs before early stopping can trigger. Default: 20.')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Patience epochs before stopping. Default: 30.')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-3,
                        help='Minimum absolute loss improvement to reset patience. Default: 1e-3.')

    args = parser.parse_args()

    dataset_id = args.dataset_id

    def _is_embryo_dataset_id(x: str) -> bool:
        s = str(x)
        return bool(re.match(r"^E\d+S\d+$", s)) or s == "Embryo_E1S1" or s.startswith("Embryo_") or s.startswith("Embryo/")

    if args.config_name is not None and str(args.config_name).strip():
        config_name = str(args.config_name).strip()
    else:
        if dataset_id in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673','151674','151675','151676']:
            config_name = 'DLPFC'
        elif _is_embryo_dataset_id(dataset_id):
            config_name = 'Embryo'
        else:
            config_name = dataset_id

    # Parse use_llm from string to bool
    args.use_llm = (args.use_llm.lower() == 'true')

    # Resolve all user paths relative to project root
    args.llm_emb_dir = _resolve_under_project_root(args.llm_emb_dir)
    args.result_dir = _resolve_under_project_root(args.result_dir)
    if args.ckpt_dir is not None:
        args.ckpt_dir = _resolve_under_project_root(args.ckpt_dir)
    if args.data_dir is not None:
        args.data_dir = _resolve_under_project_root(args.data_dir)

    # Path guard: ensure outputs stay inside project
    _assert_path_inside_project(args.result_dir, "result_dir")
    if args.ckpt_dir is not None:
        _assert_path_inside_project(args.ckpt_dir, "ckpt_dir")

    # Determine path prefixes based on config_name
    data_path_prefix_str = ""
    result_subdir = config_name if config_name in ["DLPFC", "Embryo"] else ""

    print(f"Processing dataset: {dataset_id}")

    config_file_path = os.path.join(PROJECT_ROOT, 'config', f'{config_name}.ini')
    config = Config(config_file_path)
    
    # Apply CLI overrides
    if args.llm_modulation_ratio is not None:
        config.llm_modulation_ratio = float(args.llm_modulation_ratio)
        print(f"[INFO] Override config.llm_modulation_ratio -> {config.llm_modulation_ratio}")
    if args.alpha is not None:
        config.alpha = float(args.alpha)
        print(f"[INFO] Override config.alpha -> {config.alpha}")
    if args.beta is not None:
        config.beta = float(args.beta)
        print(f"[INFO] Override config.beta -> {config.beta}")
    if args.gamma is not None:
        config.gamma = float(args.gamma)
        print(f"[INFO] Override config.gamma -> {config.gamma}")
    if args.lr is not None:
        config.lr = float(args.lr)
        print(f"[INFO] Override config.lr -> {config.lr}")
    if args.nhid1 is not None:
        config.nhid1 = int(args.nhid1)
        print(f"[INFO] Override config.nhid1 -> {config.nhid1}")
    if args.nhid2 is not None:
        config.nhid2 = int(args.nhid2)
        print(f"[INFO] Override config.nhid2 -> {config.nhid2}")
    if args.dropout is not None:
        config.dropout = float(args.dropout)
        print(f"[INFO] Override config.dropout -> {config.dropout}")
    if args.weight_decay is not None:
        config.weight_decay = float(args.weight_decay)
        print(f"[INFO] Override config.weight_decay -> {config.weight_decay}")
    if args.seed is not None:
        config.seed = int(args.seed)
        print(f"[INFO] Override config.seed -> {config.seed}")

    llm_emb_file_path = os.path.join(args.llm_emb_dir, f'embeddings_{dataset_id}.npy')

    # Determine where to read generated data
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif config_name == "Embryo":
        embryo_dir = os.path.join(PROJECT_ROOT, "data", "generated", "Embryo", dataset_id)
        flat_dir = os.path.join(PROJECT_ROOT, "data", "generated", dataset_id)
        if os.path.exists(os.path.join(embryo_dir, "data.h5ad")):
            data_dir = embryo_dir
        elif os.path.exists(os.path.join(flat_dir, "data.h5ad")):
            print(
                f"[INFO] Embryo dataset detected but {embryo_dir} not found; "
                f"falling back to flat generated dir: {flat_dir}"
            )
            data_dir = flat_dir
        else:
            data_dir = embryo_dir
    else:
        data_dir = os.path.join(PROJECT_ROOT, "data", "generated", dataset_id)

    adata, features, labels, sadj, graph_nei, graph_neg = load_data(
        dataset_id, data_path_prefix=data_path_prefix_str, data_dir=data_dir
    )

    # Load GRN-based spot embedding fadj (always use spotemb_grn)
    fadj_sparse = load_cached_fadj(dataset_id, "grn", config.k, data_dir=data_dir)
    # Normalize: (fadj + I) then row-normalize
    fadj = normalize_sparse_matrix(fadj_sparse + sp.eye(fadj_sparse.shape[0]))
    fadj = sparse_mx_to_torch_sparse_tensor(fadj)

    plt.rcParams["figure.figsize"] = (3, 4)
    current_savepath = os.path.join(args.result_dir, result_subdir, dataset_id, args.run_name)
    os.makedirs(current_savepath, exist_ok=True)

    # 设置日志：同时输出到终端和文件
    log_path = os.path.join(current_savepath, "train.log")
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"\n{'='*60}")
    print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}\n")

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(current_savepath, "checkpoints")
    if args.save_best_ckpt and (not os.path.exists(ckpt_dir)):
        os.makedirs(ckpt_dir)

    cuda_enabled = not config.no_cuda and torch.cuda.is_available()

    # Determine number of clusters and whether we're in unsupervised mode
    unsupervised_mode = args.n_clusters is not None
    
    if unsupervised_mode:
        config.class_num = args.n_clusters
        config.n = adata.n_obs
        ground_truth_labels = torch.zeros(config.n, dtype=torch.long)
        print(f"[INFO] Unsupervised mode: n_clusters={config.class_num}, no ground truth evaluation")
    else:
        _, ground_truth_labels = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground_truth_labels = torch.LongTensor(ground_truth_labels)
        config.n = len(ground_truth_labels) 
        config.class_num = len(ground_truth_labels.unique())
    
    print('seed:', config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    if cuda_enabled:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    print(f"{dataset_id} LR: {config.lr}, Alpha: {config.alpha}, Beta: {config.beta}, Gamma: {config.gamma}")

    if cuda_enabled:
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        if graph_nei is not None and graph_neg is not None:
            try:
                n0 = int(graph_nei.shape[0])
                dense_square = (graph_nei.dim() == 2 and graph_nei.shape[0] == graph_nei.shape[1])
                if dense_square and n0 >= 10000:
                    print(f"[WARNING] Skip moving dense graph masks to CUDA (shape={tuple(graph_nei.shape)}). "
                          f"Falling back to sparse sampled regularization loss.")
                    graph_nei = None
                    graph_neg = None
                else:
                    graph_nei = graph_nei.cuda()
                    graph_neg = graph_neg.cuda()
            except Exception as e:
                print(f"[WARNING] Failed to move graph masks to CUDA due to: {e}. "
                      f"Falling back to sparse sampled regularization loss.")
                graph_nei = None
                graph_neg = None

    # Load LLM embeddings only if use_llm is True
    if args.use_llm:
        precomputed_llm_base_embeddings, llm_base_emb_dim = run_precomputation_llm_base(llm_emb_file_path)
        if cuda_enabled:
            precomputed_llm_base_embeddings = precomputed_llm_base_embeddings.cuda()
        print(f"[INFO] Loaded LLM embeddings: shape={precomputed_llm_base_embeddings.shape}, dim={llm_base_emb_dim}")
    else:
        precomputed_llm_base_embeddings = None
        llm_base_emb_dim = 1
        print("[INFO] use_llm=False: skipping LLM embedding loading, model will not use semantic branch.")

    model_instance = GreS(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout,
        llm_dim=llm_base_emb_dim,
        llm_modulation_ratio=config.llm_modulation_ratio,
        use_llm=args.use_llm
    )
    if cuda_enabled:
        model_instance.cuda()

    optimizer_instance = optim.Adam(model_instance.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    current_dataset_best_ari = 0.0
    current_dataset_best_nmi = 0.0
    current_dataset_best_acc = 0.0
    current_dataset_best_f1_w = 0.0
    current_dataset_best_f1_m = 0.0
    best_idx_for_dataset = []
    best_mean_for_dataset = []
    best_emb_for_dataset = []
    best_epoch_for_dataset = -1
    best_kmeans_seed_for_dataset = args.kmeans_seed_base
    best_total_loss_for_dataset = float('inf')
    stopped_early = False
    stop_epoch = None
    no_improve_count = 0
    best_ckpt_path = os.path.join(
        ckpt_dir,
        f'best_{dataset_id}{"_llm" if args.use_llm else "_wo_llm"}.pt'
    )

    for epoch in tqdm(range(config.epochs)):
        emb, mean, zinb_loss, reg_loss, dcir_loss_val, total_loss = train(
            model_instance, features, sadj, fadj, 
            precomputed_llm_base_embeddings, graph_nei, graph_neg, 
            config, optimizer_instance
        )
        print(f"{dataset_id} epoch: {epoch}, zinb={zinb_loss:.2f}"
              f", reg={reg_loss:.2f}, dcir={dcir_loss_val:.2f}, total={total_loss:.2f}")

        emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
        mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=config.class_num, n_init='auto', random_state=args.kmeans_seed_base).fit(emb)
        idx = kmeans.labels_
        best_kmeans_seed = args.kmeans_seed_base

        if unsupervised_mode:
            ari_res = 0.0
            nmi_res = 0.0
            ami_res = 0.0
            acc_res = 0.0
            f1_res_w = 0.0
            f1_res_m = 0.0
            
            current_loss = total_loss.item()
            is_best = current_loss < best_total_loss_for_dataset
            if is_best:
                best_total_loss_for_dataset = current_loss
                no_improve_count = 0
            else:
                if args.early_stop and epoch >= int(args.early_stop_warmup):
                    if (best_total_loss_for_dataset - current_loss) <= float(args.early_stop_min_delta):
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
        else:
            kmeans_result = evaluate_kmeans_repeated(
                emb, labels, config.class_num,
                repeats=args.kmeans_repeats,
                seed_base=args.kmeans_seed_base
            )
            idx = kmeans_result['idx']
            ari_res = kmeans_result['ari']
            nmi_res = kmeans_result['nmi']
            ami_res = kmeans_result['ami']
            best_kmeans_seed = kmeans_result['kmeans_seed']

            if config_name == 'DLPFC':
                _, labels = np.unique(labels, return_inverse=True)
            idx_aligned = hungarian_match(labels, idx)
            acc_res = metrics.accuracy_score(labels, idx_aligned) * 100
            f1_res_w = metrics.f1_score(labels, idx_aligned, average='weighted') * 100
            f1_res_m = metrics.f1_score(labels, idx_aligned, average='macro') * 100
            
            is_best = ari_res > current_dataset_best_ari

        if is_best:
            current_dataset_best_ari = ari_res
            current_dataset_best_nmi = nmi_res
            current_dataset_best_acc = acc_res
            current_dataset_best_f1_w = f1_res_w
            current_dataset_best_f1_m = f1_res_m

            best_idx_for_dataset = idx
            best_mean_for_dataset = mean
            best_emb_for_dataset = emb
            best_epoch_for_dataset = epoch
            best_kmeans_seed_for_dataset = best_kmeans_seed

            if args.save_best_ckpt:
                ckpt_obj = {
                    "dataset_id": dataset_id,
                    "config_name": config_name,
                    "epoch": epoch,
                    "unsupervised_mode": unsupervised_mode,
                    "n_clusters": config.class_num,
                    "metrics": {
                        "ari": float(ari_res),
                        "nmi": float(nmi_res),
                        "acc": float(acc_res),
                        "f1_weighted": float(f1_res_w),
                        "f1_macro": float(f1_res_m),
                        "total_loss": float(total_loss.item()) if unsupervised_mode else None,
                    },
                    "model_state_dict": model_instance.state_dict(),
                    "optimizer_state_dict": optimizer_instance.state_dict(),
                }
                tmp_path = best_ckpt_path + ".tmp"
                torch.save(ckpt_obj, tmp_path)
                os.replace(tmp_path, best_ckpt_path)

        # Early stopping trigger
        if unsupervised_mode and args.early_stop and epoch >= int(args.early_stop_warmup):
            if no_improve_count >= int(args.early_stop_patience):
                stopped_early = True
                stop_epoch = epoch
                print(
                    f"[EARLY_STOP] Stop at epoch={epoch} "
                    f"(warmup={args.early_stop_warmup}, patience={args.early_stop_patience}, "
                    f"min_delta={args.early_stop_min_delta}). "
                    f"Best loss={best_total_loss_for_dataset:.4f} (best_epoch={best_epoch_for_dataset})."
                )
                break

    if unsupervised_mode:
        print(f"Best loss for {dataset_id}: {best_total_loss_for_dataset:.4f} (epoch={best_epoch_for_dataset})")
    else:
        print(f"Best ARI for {dataset_id}: {current_dataset_best_ari}")
    if args.save_best_ckpt:
        print(f"Best checkpoint saved to: {best_ckpt_path} (epoch={best_epoch_for_dataset})")

    if unsupervised_mode:
        title_str = (f'GreS (unsupervised, K={config.class_num})\n'
                     f'Best loss={best_total_loss_for_dataset:.2f}, epoch={best_epoch_for_dataset}')
    else:
        title_str = (f'GreS: ARI={current_dataset_best_ari:.2f}\n'
                     f'NMI={current_dataset_best_nmi:.2f}, ACC={current_dataset_best_acc:.2f}\n'
                     f'F1_m={current_dataset_best_f1_m:.2f}, F1_w={current_dataset_best_f1_w:.2f}')

    adata.obs['idx'] = best_idx_for_dataset.astype(str)
    adata.obsm['emb'] = best_emb_for_dataset
    adata.obsm['mean'] = best_mean_for_dataset

    # Save best clustering outputs
    npz_data = {
        "idx": best_idx_for_dataset,
        "emb": best_emb_for_dataset,
        "mean": best_mean_for_dataset,
        "labels": np.array(labels, dtype=str) if not unsupervised_mode else np.array(['NA'] * len(best_idx_for_dataset)),
        "dataset_id": dataset_id,
        "config_name": config_name,
        "use_llm": bool(args.use_llm),
        "fadj_mode": "spotemb_grn",
        "best_epoch": int(best_epoch_for_dataset),
        "n_clusters": int(config.class_num),
        "unsupervised_mode": bool(unsupervised_mode),
    }
    if unsupervised_mode:
        npz_data["best_total_loss"] = float(best_total_loss_for_dataset)
    else:
        npz_data.update({
            "ari": float(current_dataset_best_ari),
            "nmi": float(current_dataset_best_nmi),
            "acc": float(current_dataset_best_acc),
            "f1_weighted": float(current_dataset_best_f1_w),
            "f1_macro": float(current_dataset_best_f1_m),
        })
    np.savez(os.path.join(current_savepath, "best_cluster_outputs.npz"), **npz_data)
    
    # Save KMeans labels
    np.save(os.path.join(current_savepath, "kmeans_labels.npy"), best_idx_for_dataset)

    # Save best metrics to JSON only
    metrics_json_path = os.path.join(current_savepath, "metrics_best.json")
    metrics_row = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "use_llm": bool(args.use_llm),
        "llm_emb_dir": args.llm_emb_dir,
        "llm_emb_file": llm_emb_file_path,
        "fadj_mode": "spotemb_grn",
        "seed": int(config.seed),
        "best_epoch": int(best_epoch_for_dataset),
        "n_clusters": int(config.class_num),
        "unsupervised_mode": bool(unsupervised_mode),
        "early_stop": bool(args.early_stop),
        "early_stop_warmup": int(args.early_stop_warmup),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "stopped_early": bool(stopped_early),
        "stop_epoch": int(stop_epoch) if stop_epoch is not None else None,
        "ARI": float(current_dataset_best_ari) if not unsupervised_mode else None,
        "NMI": float(current_dataset_best_nmi) if not unsupervised_mode else None,
        "ACC": float(current_dataset_best_acc) if not unsupervised_mode else None,
        "F1_weighted": float(current_dataset_best_f1_w) if not unsupervised_mode else None,
        "F1_macro": float(current_dataset_best_f1_m) if not unsupervised_mode else None,
        "best_total_loss": float(best_total_loss_for_dataset) if unsupervised_mode else None,
        "ckpt_path": best_ckpt_path if args.save_best_ckpt else "",
        "result_dir": current_savepath,
        # Hyperparameters
        "lr": float(config.lr),
        "llm_modulation_ratio": float(config.llm_modulation_ratio),
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "gamma": float(config.gamma),
        "nhid1": int(config.nhid1),
        "nhid2": int(config.nhid2),
        "dropout": float(config.dropout),
        "weight_decay": float(config.weight_decay),
        "kmeans_repeats": int(args.kmeans_repeats),
        "kmeans_seed_base": int(args.kmeans_seed_base),
        "best_kmeans_seed": int(best_kmeans_seed_for_dataset),
        "k": int(config.k),
        "radius": int(config.radius),
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_row, f, ensure_ascii=False, indent=2)
    print(f"Best metrics saved to: {metrics_json_path}")

    # Save best output to h5ad
    best_h5ad_path = os.path.join(current_savepath, "best_output.h5ad")
    adata.write_h5ad(best_h5ad_path)
    print(f"Best output h5ad saved to: {best_h5ad_path}")

    if config_name in ['DLPFC', 'Human_Breast_Cancer', 'Mouse_Brain_Anterior']:
        sc.pl.spatial(adata,
                      img_key='hires',
                      color=['idx'],
                      title=title_str,
                      show=False
                      )
    else:
        sc.pl.embedding(adata,
                        basis="spatial",
                        color="idx",
                        s=25,
                        show=False,
                        title=title_str
                        )
    plot_filename = f'GreS' if args.use_llm else 'GreS_wo_llm'
    plt.savefig(os.path.join(current_savepath, f'{plot_filename}.png'),
                bbox_inches='tight',
                dpi=300)

    plt.show()
