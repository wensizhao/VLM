#!/usr/bin/env bash
# One entry point for Linux CLI and Dev Containers initialization.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/dev.sh cpu|gpu COMMAND [ARGS...]

  prepare     Create .env if needed and synchronize host UID/GID (no container start)
  ensure-image  Reuse a local image, otherwise pull it and build if pulling fails
  up -d       Start the development container
  web up -d   Start Web Studio with the selected CPU or CUDA image
  build       Build the selected CPU or CUDA image locally
  shell       Open zsh in the running development container
  check       Run a one-off environment/permissions/device check
  test        Run the project pytest suite in a one-off container
  config      Show the merged Docker Compose configuration
  down        Stop this project's container, preserving home/cache volumes

Other commands/arguments are forwarded to Docker Compose.
Optional personal mounts: VLM_HOST_TOOLS=1 ./scripts/dev.sh cpu|gpu COMMAND
EOF
}

mode="${1:-help}"
action="${2:-help}"
case "${mode}" in
  -h|--help|help) usage; exit 0 ;;
  cpu|gpu) ;;
  *) usage >&2; exit 2 ;;
esac
case "${action}" in
  -h|--help|help) usage; exit 0 ;;
esac
shift 2

web_mode=0
if [[ "${action}" == web ]]; then
  web_mode=1
  action="${1:-up}"
  if [[ $# -gt 0 ]]; then
    shift
  else
    set -- -d
  fi
  case "${action}" in
    -h|--help|help) usage; exit 0 ;;
  esac
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
if [[ ! "${LOCAL_UID}" =~ ^[0-9]+$ || ! "${LOCAL_GID}" =~ ^[0-9]+$ ]]; then
  echo 'LOCAL_UID and LOCAL_GID must be numeric.' >&2
  exit 2
fi

if [[ "${mode}" == gpu ]]; then
  case "${action}" in
    up|build|run|check|test|prepare|ensure-image)
      case "$(uname -m)" in
        x86_64|amd64) ;;
        *) echo 'The pinned CUDA image requires x86-64; use cpu on ARM64 (Jetson needs a separate L4T image).' >&2; exit 2 ;;
      esac
      ;;
  esac
fi

# Never source .env as shell code. Preserve all user settings except these IDs.
umask 077
if [[ ! -e .env ]]; then
  cp .env.example .env
fi
env_tmp="$(mktemp "${project_dir}/.env.tmp.XXXXXX")"
trap 'rm -f -- "${env_tmp}"' EXIT
awk -v uid="${LOCAL_UID}" -v gid="${LOCAL_GID}" '
  /^[[:space:]]*(export[[:space:]]+)?LOCAL_UID[[:space:]]*=/ { print "LOCAL_UID=" uid; u=1; next }
  /^[[:space:]]*(export[[:space:]]+)?LOCAL_GID[[:space:]]*=/ { print "LOCAL_GID=" gid; g=1; next }
  { print }
  END { if (!u) print "LOCAL_UID=" uid; if (!g) print "LOCAL_GID=" gid }
' .env > "${env_tmp}"
# Keep the existing file's permissions and any symlink target intact.
cat "${env_tmp}" > .env

if [[ "${action}" == prepare ]]; then
  echo "Prepared .env for UID:GID ${LOCAL_UID}:${LOCAL_GID} (${mode})."
  exit 0
fi

compose=(docker compose --project-directory "${project_dir}" -f "${project_dir}/docker-compose.yml")
expect=cpu
if [[ "${mode}" == gpu ]]; then
  compose+=(-f "${project_dir}/docker-compose.gpu.yml")
  expect=cuda
fi
if [[ "${VLM_HOST_TOOLS:-0}" == 1 ]]; then
  compose+=(-f "${project_dir}/docker-compose.host-tools.yml")
fi
if [[ "${web_mode}" == 1 ]]; then
  compose+=(-f "${project_dir}/docker-compose.web.yml")
fi

# Let Compose parse dotenv interpolation instead of executing the file as shell.
compose_setting() {
  "${compose[@]}" config --environment | awk -v key="$1" '
    index($0, key "=") == 1 { value=substr($0, length(key)+2) }
    END { print value }
  '
}

build_image() {
  local base_image torch_index image shell_tools
  base_image="$(compose_setting CPU_BASE_IMAGE)"
  base_image="${base_image:-python:3.10-slim-bookworm}"
  torch_index=https://download.pytorch.org/whl/cpu
  if [[ "${mode}" == gpu ]]; then
    base_image="$(compose_setting CUDA_BASE_IMAGE)"
    base_image="${base_image:-pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime}"
    torch_index=https://download.pytorch.org/whl/cu121
  fi
  image="$("${compose[@]}" config --images vlm-dev)"
  if [[ "${image}" == *@* ]]; then
    echo 'A digest-pinned image cannot be replaced by a local build. Use an image tag in VLM_CPU_IMAGE or VLM_CUDA_IMAGE to build locally.' >&2
    return 2
  fi
  shell_tools="$(compose_setting INSTALL_OPTIONAL_SHELL_TOOLS)"
  # Both explicit builds and first-use fallback use the same Dockerfile/arguments.
  docker build --tag "${image}" \
    --build-arg "BASE_IMAGE=${base_image}" \
    --build-arg "TORCH_INDEX_URL=${torch_index}" \
    --build-arg "INSTALL_OPTIONAL_SHELL_TOOLS=${shell_tools:-0}" \
    "$@" "${project_dir}"
}

ensure_image() {
  local image required_platform cached_platform
  # Fail directly if the daemon is unavailable, before trying a pull/build.
  required_platform="$(docker info --format '{{.OSType}}/{{.Architecture}}')"
  if [[ "${mode}" == gpu ]]; then
    required_platform=linux/amd64
  else
    required_platform="${DOCKER_DEFAULT_PLATFORM:-${required_platform}}"
  fi
  case "${required_platform}" in
    linux/x86_64|linux/amd64) required_platform=linux/amd64 ;;
    linux/aarch64|linux/arm64|linux/arm64/v8) required_platform=linux/arm64 ;;
    *) echo "Unsupported image platform: ${required_platform}; use Linux amd64 or arm64 (CPU)." >&2; return 2 ;;
  esac
  image="$("${compose[@]}" config --images vlm-dev)"
  cached_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${image}" 2>/dev/null || true)"
  if [[ "${cached_platform}" == "${required_platform}" ]]; then
    echo "Using local image ${image} (${required_platform})."
    return 0
  fi
  echo "Pulling ${image} (${required_platform})..."
  if docker pull --platform "${required_platform}" "${image}"; then
    return 0
  fi
  echo "Pull failed for ${image}; building the ${mode} image locally..." >&2
  build_image --platform "${required_platform}"
}

case "${action}" in
  build) build_image "$@" ;;
  ensure-image) ensure_image ;;
  shell) "${compose[@]}" exec vlm-dev zsh "$@" ;;
  check) "${compose[@]}" run --rm --no-deps vlm-dev python scripts/docker/check_environment.py --expect "${expect}" "$@" ;;
  test) "${compose[@]}" run --rm --no-deps vlm-dev python -m pytest -q "$@" ;;
  *) "${compose[@]}" "${action}" "$@" ;;
esac
