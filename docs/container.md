# Container Setup

This project uses Docker Compose for a reproducible research environment.

## Files

- `Dockerfile`: builds the base image with PyTorch, video libraries, and common VLM research dependencies.
- `docker-compose.yml`: starts the default development container, mounts the current repo, and exposes common ports.
- `docker-compose.gpu.yml`: optional GPU override for NVIDIA hosts.
- `.env.example`: editable environment variables for user ID, ports, and shared memory.
- `requirements/base.txt`: core runtime dependencies for training and evaluation.
- `requirements/dev.txt`: development extras layered on top of the base dependencies.
- `scripts/docker/entrypoint.sh`: creates runtime directories before the container command starts.
- `.devcontainer/devcontainer.json`: VS Code Dev Containers entrypoint for `Reopen in Container`.

The project source code is bind-mounted at `/workspace/vlm_ws`, but the container home directory and caches are stored in the Docker named volume `dev-home`. They are isolated from the host filesystem and persist across container restarts.

The image also ships the `opencode` CLI (installed when `INSTALL_OPTIONAL_SHELL_TOOLS=1`). The host
configuration is bind-mounted into the container so `opencode` reuses it directly:

- `$HOST_OPENCODE_CONFIG` (`~/.config/opencode`) -> `/tmp/devhome/.config/opencode`
- `$HOST_OPENCODE_AUTH` (`~/.local/share/opencode/auth.json`) -> `/opt/opencode-host/auth.json` (read-only; the entrypoint copies it to `/tmp/devhome/.local/share/opencode/auth.json`)
- `$HOST_AGENTS_DIR` (`~/.agents`) -> `/tmp/devhome/.agents`

Sessions and the SQLite database stay inside the `dev-home` volume, so the host database is never
written to by the container. Override the three variables in `.env` when the host paths differ, or
set them to a directory that exists locally (a missing `auth.json` becomes an empty directory and is
ignored by the CLI).

## Quick Start

1. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Optional: switch to a smaller CPU base image if you do not need CUDA:

   ```bash
   sed -i 's|^BASE_IMAGE=.*|BASE_IMAGE=pytorch/pytorch:2.3.1-cpu|' .env
   ```

3. Build the image:

   ```bash
   docker compose build
   ```

4. Start the container:

   ```bash
   docker compose up -d
   ```

5. Open a shell in the container:

   ```bash
   docker compose exec vlm-dev zsh
   ```

The host terminal can be `zsh` or `bash`. The container stays alive with `sleep infinity`, and you open an interactive shell with `docker compose exec vlm-dev zsh`.

## GPU Start

If your host has NVIDIA Container Toolkit installed and `BASE_IMAGE` is a CUDA image, use:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Troubleshooting

If the build fails with errors such as `failed to fetch anonymous token` or `connection reset by peer`, the problem is usually network or proxy access to Docker Hub rather than the project files.

Common fixes:

```bash
# Retry after network/proxy recovers
docker compose build

# If you use a local proxy, ensure it is running and stable first

# If you do not need GPU, switch to the smaller CPU image in .env
sed -i 's|^BASE_IMAGE=.*|BASE_IMAGE=pytorch/pytorch:2.3.1-cpu|' .env
docker compose build
```

If some layers finished downloading before the failure, Docker will usually reuse the completed layers on the next build attempt.

## Common Commands

Start Jupyter inside the container:

```bash
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

Start TensorBoard inside the container:

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

Stop the container:

```bash
docker compose down
```

## VS Code Dev Container

Open the project in VS Code, then run `Dev Containers: Reopen in Container`. VS Code will use `.devcontainer/devcontainer.json` and the `vlm-dev` service from `docker-compose.yml`.
