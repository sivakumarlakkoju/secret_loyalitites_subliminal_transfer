set -a
source /home/claudeuser/.secrets/subliminal-loyalty.env
set +a

export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf/transformers
export HF_DATASETS_CACHE=/workspace/hf/datasets
export PIP_CACHE_DIR=/workspace/.pipcache
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export TMPDIR=/workspace/tmp
export TRITON_CACHE_DIR=/workspace/hf/triton
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"

source /workspace/venv/bin/activate
