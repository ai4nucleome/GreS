#!/bin/bash
# ============================================================
# Complete Preprocessing Pipeline
# 
# This script runs all 4 preprocessing steps in order:
#   1. Data preprocessing (preprocess_data.py)
#   2. Semantic embedding generation (build_programST_assets.py)
#   3. Spot embedding generation (grn_generate_spot_embedding.py)
#   4. Feature adjacency graph construction (build_fadj_from_geneemb.py)
#
# Usage:
#   ./run_preprocess.sh <dataset_id> <config_name>
#   ./run_preprocess.sh 151507 DLPFC
#   ./run_preprocess.sh E1S1 Embryo
# ============================================================

set -e  # Exit on error

# ============================================================
# Check arguments
# ============================================================
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <dataset_id> <config_name>"
    echo "Example: $0 151507 DLPFC"
    exit 1
fi

DATASET="$1"
CONFIG_NAME="$2"
TOTAL_STEPS=4
CURRENT_STEP=0

# ============================================================
# Progress bar function
# ============================================================
print_progress() {
    local step=$1
    local total=$2
    local desc=$3
    local width=40
    local percent=$((step * 100 / total))
    local filled=$((step * width / total))
    local empty=$((width - filled))
    
    # Build progress bar
    local bar=""
    for ((i=0; i<filled; i++)); do bar+="█"; done
    for ((i=0; i<empty; i++)); do bar+="░"; done
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Progress: [${bar}] ${percent}%"
    echo "║  Step ${step}/${total}: ${desc}"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================
# Path Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data directories
# Note: Raw data path logic is now handled inside python scripts based on standardized h5ad
GENERATED_DIR="${SCRIPT_DIR}/data/generated/${DATASET}"

# Embedding resources
if [ -d "${SCRIPT_DIR}/embeddings" ]; then
    EMBED_DIR="${SCRIPT_DIR}/embeddings"
else
    EMBED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/embeddings"
fi

# Input files for semantic embedding generation
VOCAB="${EMBED_DIR}/vocab.json"
EMBEDDING="${EMBED_DIR}/pretrained_gene_embeddings.pt"
GR_RDS="${EMBED_DIR}/weighted_networks_nsga2r_final.rds"

# Parameters (can be modified as needed)
TOPK=30          # Top-k genes for spot embedding
FADJ_K=14        # k-NN neighbors for feature adjacency graph

# ============================================================
# Banner
# ============================================================
clear 2>/dev/null || true
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             GreS Preprocessing Pipeline                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Dataset ID : ${DATASET}"
echo "  Config Name: ${CONFIG_NAME}"
echo "  Script dir : ${SCRIPT_DIR}"
echo "  Output dir : ${GENERATED_DIR}"
echo ""
echo "  Parameters:"
echo "    • topk   : ${TOPK}"
echo "    • fadj_k : ${FADJ_K}"
echo ""

# ============================================================
# Validation
# ============================================================
echo "┌────────────────────────────────────────────────────────────┐"
echo "│  Validating input files...                                 │"
echo "└────────────────────────────────────────────────────────────┘"

# Check embedding files
for f in "${VOCAB}" "${EMBEDDING}" "${GR_RDS}"; do
    if [ ! -f "$f" ]; then
        echo "  ✗ Required file not found: $f"
        exit 1
    fi
done
echo "  ✓ Embedding files exist"
echo ""

# ============================================================
# Step 1: Data Preprocessing
# ============================================================
CURRENT_STEP=1
print_progress ${CURRENT_STEP} ${TOTAL_STEPS} "Data Preprocessing"

echo "Running: python preprocess/preprocess_data.py --dataset_id ${DATASET} --config_name ${CONFIG_NAME}"
echo "─────────────────────────────────────────────────────────────"

cd "${SCRIPT_DIR}"
python preprocess/preprocess_data.py --dataset_id "${DATASET}" --config_name "${CONFIG_NAME}"

echo ""
echo "  ✓ Output: ${GENERATED_DIR}/data.h5ad"

# ============================================================
# Step 2: Semantic Embedding Generation
# ============================================================
CURRENT_STEP=2
print_progress ${CURRENT_STEP} ${TOTAL_STEPS} "Semantic Embedding Generation (GRN Diffusion)"

echo "Running: python build_programST_assets.py ..."
echo "─────────────────────────────────────────────────────────────"

python build_programST_assets.py \
    --h5ad "${GENERATED_DIR}/data.h5ad" \
    --vocab "${VOCAB}" \
    --embedding "${EMBEDDING}" \
    --out_dir "${GENERATED_DIR}" \
    --gr_rds "${GR_RDS}" \
    --grn_diffuse \
    --log_name "build_assets_${DATASET}.log" \
    --overwrite

echo ""
echo "  ✓ Outputs:"
echo "    • ${GENERATED_DIR}/programST_index.json"
echo "    • ${GENERATED_DIR}/programST_tensors.pt"
echo "    • ${GENERATED_DIR}/gr_subgraph.pt"
echo "    • ${GENERATED_DIR}/grn_gene_embeddings.pt"

# ============================================================
# Step 3: Spot Embedding Generation
# ============================================================
CURRENT_STEP=3
print_progress ${CURRENT_STEP} ${TOTAL_STEPS} "Spot Embedding Generation (topk=${TOPK})"

echo "Running: python grn_generate_spot_embedding.py --dataset_id ${DATASET} --topk ${TOPK}"
echo "─────────────────────────────────────────────────────────────"

python grn_generate_spot_embedding.py \
    --dataset_id "${DATASET}" \
    --topk "${TOPK}"

echo ""
echo "  ✓ Output: data/npys_grn/embeddings_${DATASET}.npy"

# ============================================================
# Step 4: Feature Adjacency Graph Construction
# ============================================================
CURRENT_STEP=4
print_progress ${CURRENT_STEP} ${TOTAL_STEPS} "Feature Adjacency Graph (k=${FADJ_K})"

echo "Running: python build_fadj_from_geneemb.py --dataset_id ${DATASET} --k ${FADJ_K}"
echo "─────────────────────────────────────────────────────────────"

python build_fadj_from_geneemb.py \
    --dataset_id "${DATASET}" \
    --k "${FADJ_K}" \
    --overwrite

echo ""
echo "  ✓ Output: ${GENERATED_DIR}/fadj_spotemb_grn_k${FADJ_K}.npz"

# ============================================================
# Summary
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ✓ Preprocessing Complete!                          ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Progress: [████████████████████████████████████████] 100% ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Dataset: ${DATASET}"
echo ""
echo "  Generated files:"
echo "    1. ${GENERATED_DIR}/data.h5ad"
echo "    2. ${GENERATED_DIR}/programST_index.json"
echo "    3. ${GENERATED_DIR}/programST_tensors.pt"
echo "    4. ${GENERATED_DIR}/gr_subgraph.pt"
echo "    5. ${GENERATED_DIR}/grn_gene_embeddings.pt"
echo "    6. data/npys_grn/embeddings_${DATASET}.npy"
echo "    7. ${GENERATED_DIR}/fadj_spotemb_grn_k${FADJ_K}.npz"
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│  Next step: Run training                                   │"
echo "│                                                            │"
echo "│  python train.py --dataset_id ${DATASET} \\                 "
echo "│      --config_name ${CONFIG_NAME} \\                        "
echo "│      --llm_emb_dir data/npys_grn/ \\                        "
echo "│      --fadj_mode spotemb_grn \\                             "
echo "│      --run_name <your_run_name>                            │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""
