#!/usr/bin/env bash
set -euo pipefail

HF_REPO_ID="${1:-}"
TARGET_DIR="${2:-}"

if [[ -z "${HF_REPO_ID}" ]]; then
  echo "[ERROR] 用法: bash scripts/hf_download_embeddings.sh <HF_REPO_ID> [target_dir]"
  echo "        例如: bash scripts/hf_download_embeddings.sh ai4nucleome/GreS-embeddings"
  exit 1
fi

REPO_TYPE="${HF_REPO_TYPE:-dataset}"  # dataset | model
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${TARGET_DIR}" ]]; then
  TARGET_DIR="${ROOT_DIR}/embeddings"
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 未找到 git，请先安装 git。"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "[ERROR] 未找到 git-lfs。请先安装 git-lfs（Hugging Face 大文件依赖 LFS）。"
  exit 1
fi

URL=""
if [[ "${REPO_TYPE}" == "dataset" ]]; then
  URL="https://huggingface.co/datasets/${HF_REPO_ID}"
else
  URL="https://huggingface.co/${HF_REPO_ID}"
fi

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "[INFO] Repo: ${HF_REPO_ID} (type=${REPO_TYPE})"
echo "[INFO] Download to: ${TARGET_DIR}"

git lfs install --skip-repo >/dev/null 2>&1 || true
git clone --depth 1 "${URL}" "${TMP_DIR}/hf_repo" >/dev/null

if [[ ! -d "${TMP_DIR}/hf_repo/embeddings" ]]; then
  echo "[ERROR] Hugging Face 仓库里没有找到 embeddings/ 目录：${URL}"
  echo "        期望结构: <repo>/embeddings/<files>"
  exit 1
fi

mkdir -p "${TARGET_DIR}"
cp -a "${TMP_DIR}/hf_repo/embeddings/." "${TARGET_DIR}/"

REQ_FILES=("vocab.json" "pretrained_gene_embeddings.pt" "weighted_networks_nsga2r_final.rds")
for f in "${REQ_FILES[@]}"; do
  if [[ ! -f "${TARGET_DIR}/${f}" ]]; then
    echo "[WARN] 缺少文件: ${TARGET_DIR}/${f}"
  fi
done

echo "[OK] embeddings 已下载完成。"

