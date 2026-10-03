ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG NODE_VERSION=22.15.1
ARG OPENCODE_VERSION=latest
ARG INSTALL_OPTIONAL_SHELL_TOOLS=0

ENV TZ=Asia/Shanghai \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  SHELL=/usr/bin/zsh \
  WORKSPACE_DIR=/workspace/vlm_ws \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  PYTHONPATH=/workspace/vlm_ws

# 替换 apt 源为阿里云（适用于 Ubuntu 22.04 Jammy）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
  sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  vim \
  wget \
  xz-utils \
  zsh \
  && rm -rf /var/lib/apt/lists/*

RUN if [ "${INSTALL_OPTIONAL_SHELL_TOOLS}" = "1" ]; then \
  arch="$(dpkg --print-architecture)" && \
  case "${arch}" in \
  amd64) node_arch="x64" ;; \
  arm64) node_arch="arm64" ;; \
  *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
  esac && \
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" -o /tmp/node.tar.xz && \
  tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 --no-same-owner && \
  rm /tmp/node.tar.xz && \
  npm install -g @openai/codex "opencode-ai@${OPENCODE_VERSION}" && \
  git clone --depth=1 https://gitee.com/mirrors/oh-my-zsh.git /opt/oh-my-zsh && \
  codex --version && \
  opencode --version; \
  else \
  echo "Skipping optional shell tooling install (Node.js, Codex CLI, opencode, oh-my-zsh)."; \
  fi

WORKDIR /workspace/vlm_ws

COPY requirements /tmp/requirements
RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install -r /tmp/requirements/dev.txt \
  && rm -rf /tmp/requirements

COPY scripts/docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/docker/env_setup.zsh /usr/local/share/vlm_ws/env_setup.zsh
COPY scripts/docker/zshrc.template /usr/local/share/vlm_ws/zshrc.template
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
  && mkdir -p \
  /workspace/vlm_ws/configs \
  /workspace/vlm_ws/results \
  /workspace/vlm_ws/logs \
  /workspace/vlm_ws/checkpoints \
  /usr/local/share/vlm_ws \
  /tmp/devhome/.cache/pip \
  /tmp/devhome/.cache/huggingface \
  /tmp/devhome/.cache/torch \
  /tmp/devhome/.config/matplotlib \
  && chmod 1777 /tmp/devhome \
  && chmod -R 1777 /tmp/devhome/.cache /tmp/devhome/.config

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["zsh"]
