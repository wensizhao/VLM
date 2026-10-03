#!/usr/bin/env zsh
# >>> vlm-ws managed env_setup >>>

export VLM_WS_PATH="${WORKSPACE_DIR:-/workspace/vlm_ws}"
export PYTHONPATH="${VLM_WS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.config/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"

if [ -d "${VLM_WS_PATH}/scripts" ]; then
    export PATH="${PATH}:${VLM_WS_PATH}/scripts"
fi

if [ -f "${VLM_WS_PATH}/.venv/bin/activate" ]; then
    source "${VLM_WS_PATH}/.venv/bin/activate"
fi

# <<< vlm-ws managed env_setup <<<
