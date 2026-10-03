#!/usr/bin/env bash
set -euo pipefail

HOME_DIR="${HOME:-/tmp/devhome}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/vlm_ws}"
SHARE_DIR="${VLM_WS_SHARE_DIR:-/usr/local/share/vlm_ws}"
OPENCODE_AUTH_STAGE="${OPENCODE_AUTH_STAGE:-/opt/opencode-host/auth.json}"
OPENCODE_AUTH_DIR="${HOME_DIR}/.local/share/opencode"
ZSHRC_TEMPLATE="${SHARE_DIR}/zshrc.template"
ENV_SETUP_TEMPLATE="${SHARE_DIR}/env_setup.zsh"
ZSHRC_PATH="${HOME_DIR}/.zshrc"
ENV_SETUP_PATH="${HOME_DIR}/env_setup.zsh"
ZSHRC_MARKER="# >>> vlm-ws managed zshrc >>>"
ZSHRC_LEGACY_MARKER="# >>> vlm-ws shell config >>>"
ENV_SETUP_MARKER="# >>> vlm-ws managed env_setup >>>"

mkdir -p \
  "${HOME_DIR}" \
  "${WORKSPACE_DIR}/configs" \
  "${WORKSPACE_DIR}/results" \
  "${WORKSPACE_DIR}/logs" \
  "${WORKSPACE_DIR}/checkpoints" \
  "${MPLCONFIGDIR:-${HOME_DIR}/.config/matplotlib}" \
  "${PIP_CACHE_DIR:-${HOME_DIR}/.cache/pip}" \
  "${HF_HOME:-${HOME_DIR}/.cache/huggingface}" \
  "${TORCH_HOME:-${HOME_DIR}/.cache/torch}" \
  "${HOME_DIR}/.config" \
  "${HOME_DIR}/.npm-global/bin" \
  "${HOME_DIR}/.codex/tmp/arg0" \
  "${HOME_DIR}/.local/share/opencode"

if [ -f "${OPENCODE_AUTH_STAGE}" ]; then
  if [ -w "${OPENCODE_AUTH_DIR}" ]; then
    install -m 0600 "${OPENCODE_AUTH_STAGE}" "${OPENCODE_AUTH_DIR}/auth.json"
  else
    echo "warning: ${OPENCODE_AUTH_DIR} is not writable; run 'docker compose down -v' to recreate the dev-home volume." >&2
  fi
fi

if [ ! -e "${HOME_DIR}/.oh-my-zsh" ]; then
  if [ -d /opt/oh-my-zsh ]; then
    ln -s /opt/oh-my-zsh "${HOME_DIR}/.oh-my-zsh"
  fi
fi

chmod 700 "${HOME_DIR}/.codex/tmp/arg0"

sync_template() {
  local source_path="$1"
  local target_path="$2"
  local marker="$3"
  local legacy_marker="${4:-}"

  if [ ! -f "${source_path}" ]; then
    return
  fi

  if [ ! -f "${target_path}" ]; then
    install -m 0644 "${source_path}" "${target_path}"
    return
  fi

  if grep -qF "${marker}" "${target_path}" || { [ -n "${legacy_marker}" ] && grep -qF "${legacy_marker}" "${target_path}"; }; then
    install -m 0644 "${source_path}" "${target_path}"
  fi
}

sync_template "${ZSHRC_TEMPLATE}" "${ZSHRC_PATH}" "${ZSHRC_MARKER}" "${ZSHRC_LEGACY_MARKER}"
sync_template "${ENV_SETUP_TEMPLATE}" "${ENV_SETUP_PATH}" "${ENV_SETUP_MARKER}"

if [ ! -f "${ZSHRC_PATH}" ]; then
  install -m 0644 "${ZSHRC_TEMPLATE}" "${ZSHRC_PATH}"
fi

if [ ! -f "${ENV_SETUP_PATH}" ]; then
  install -m 0644 "${ENV_SETUP_TEMPLATE}" "${ENV_SETUP_PATH}"
fi

if [ "$#" -eq 0 ]; then
  set -- zsh
fi

exec "$@"
