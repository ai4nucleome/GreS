#!/usr/bin/env bash
set -euo pipefail

HF_REPO_ID="${1:-}"
EMB_DIR="${2:-}"

if [[ -z "${HF_REPO_ID}" ]]; then
  echo "[ERROR] 用法: bash scripts/hf_upload_embeddings.sh <HF_REPO_ID> [embeddings_dir]"
  echo "        例如: bash scripts/hf_upload_embeddings.sh ai4nucleome/GreS-embeddings embeddings"
  exit 1
fi

REPO_TYPE="${HF_REPO_TYPE:-dataset}"  # dataset | model
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${EMB_DIR}" ]]; then
  EMB_DIR="${ROOT_DIR}/embeddings"
fi

if [[ ! -d "${EMB_DIR}" ]]; then
  echo "[ERROR] embeddings_dir 不存在: ${EMB_DIR}"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 未找到 git，请先安装 git。"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  echo "[ERROR] 未找到 git-lfs。请先安装 git-lfs（Hugging Face 大文件依赖 LFS）。"
  exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] 未找到 huggingface-cli。请先安装 huggingface_hub 并登录："
  echo "        pip install -U huggingface_hub"
  echo "        huggingface-cli login"
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
echo "[INFO] Upload from: ${EMB_DIR}"
echo "[INFO] Clone: ${URL}"

git lfs install --skip-repo >/dev/null 2>&1 || true
git clone "${URL}" "${TMP_DIR}/hf_repo" >/dev/null

mkdir -p "${TMP_DIR}/hf_repo/embeddings"
cp -a "${EMB_DIR}/." "${TMP_DIR}/hf_repo/embeddings/"

pushd "${TMP_DIR}/hf_repo" >/dev/null

# Track common large file types with LFS
git lfs track "*.pt" "*.pth" "*.bin" "*.rds" "*.npy" "*.npz" "*.h5ad" >/dev/null || true

git add .gitattributes embeddings
if git diff --cached --quiet; then
  echo "[INFO] 没有检测到需要提交的变更（可能已上传过相同内容）。"
  exit 0
fi

git commit -m "Add GreS embeddings" >/dev/null
git push >/dev/null

popd >/dev/null

echo "[OK] embeddings 已上传完成。"

