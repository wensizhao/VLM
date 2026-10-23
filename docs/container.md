# Linux 开发容器

团队共用一个 Dockerfile、一份研究依赖和一个 Compose 服务 `vlm-dev`，按计算后端选择镜像。
宿主发行版与镜像内部系统是两件事：Ubuntu、Debian、Fedora、Arch 等 Linux 宿主安装
可用的 Docker 后，运行相同镜像，不必分别维护 apt、dnf、pacman 版本的开发容器。
CPU 架构仍须匹配；参见 [Docker 多平台说明](https://docs.docker.com/build/building/multi-platform/)。

## 镜像与硬件

| 宿主硬件 | 模式 | 镜像标签 | 范围 |
|---|---|---|---|
| Intel / AMD x86-64 | `cpu` | `zgzdocker/vlm-ws:cpu` | Linux amd64，真正的 CPU-only PyTorch |
| ARM64 CPU | `cpu` | 同一个 `:cpu` 标签 | Linux arm64，Docker 自动选择对应架构 |
| NVIDIA GPU + x86-64 CPU | `gpu` | `zgzdocker/vlm-ws:cuda12.1` | PyTorch 2.3.1 / CUDA 12.1 |
| Jetson / ARM64 + NVIDIA GPU | `cpu` 可用；GPU 另行适配 | 当前 CUDA 标签不适用 | GPU 需要匹配 JetPack/L4T 的基础镜像 |

Intel 与 AMD 不需要分成两套镜像。CPU 与 CUDA 镜像共用 Python 3.10、
`requirements/torch.txt` 中的 Torch 2.3.1 / torchvision 0.18.1 和其他项目依赖。
默认镜像的 `video_backend: auto` 在两个 CPU 架构均使用 OpenCV。
Decord 0.6.0 的 PyPI wheel 内部仍标记为 `cp36-cp36m`，无法通过 Python 3.10 的
`pip check`，也没有 Linux ARM64 wheel，因此不再默认安装；其代码入口保留给自行
提供兼容安装包的环境。显式指定 `decord` 且未安装时会报错。
AMD/Intel 显卡目前按 CPU 使用；CUDA 镜像的 GPU 兼容性取决于固定的 CUDA/PyTorch 版本。

`:cpu` 是包含 amd64/arm64 的 manifest；`:cpu-amd64`、`:cpu-arm64` 是发布流水线的架构标签。
`:dev` 暂时保留为旧 CUDA 镜像别名，新配置不再使用这个含糊的标签。
预构建镜像已发布在 [Docker Hub](https://hub.docker.com/r/zgzdocker/vlm-ws)，可按需直接拉取：

```bash
docker pull zgzdocker/vlm-ws:cpu
docker pull zgzdocker/vlm-ws:cuda12.1
```

## 首次启动

宿主准备 Docker Engine 24+、Docker Compose 插件 2.24+，并确保当前用户能执行
`docker info`。启动脚本只依赖 Bash、awk 和 Docker，不要求宿主先安装 Python、Conda 或 CUDA Toolkit。

```bash
# CPU 开发；首次自动创建 .env 并填写当前用户 UID/GID
./scripts/dev.sh cpu up -d
./scripts/dev.sh cpu check
./scripts/dev.sh cpu shell
```

```bash
# NVIDIA 开发：宿主先安装兼容 CUDA 12.1 的驱动和 NVIDIA Container Toolkit
./scripts/dev.sh gpu up -d
./scripts/dev.sh gpu check
./scripts/dev.sh gpu shell
```

GPU 模式遵循 [Docker GPU 设备预留](https://docs.docker.com/compose/how-tos/gpu-support/)。
驱动、Toolkit 的安装及 Docker runtime 配置按
[NVIDIA 官方指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
执行；宿主不需要安装与镜像相同的 Linux 发行版或 CUDA Toolkit。
此处以普通 Docker Engine 为支持基线；rootless Docker、Podman 需要单独验证用户映射与 GPU runtime。

`check` 启动一次性容器，检查核心导入、实际 CPU/CUDA 张量计算、HOME 和实验输出目录写权限。
GPU 检查在 CUDA 不可用时返回非零，不会静默按 CPU 通过。

## Web Studio

Web UI 直接使用所选的 CPU/CUDA 开发镜像，在同一服务上增加 Web 启动配置：

```bash
./scripts/dev.sh cpu web up -d
# NVIDIA 使用 ./scripts/dev.sh gpu web up -d
# 浏览器打开 http://127.0.0.1:8000
./scripts/dev.sh cpu web logs -f
./scripts/dev.sh cpu web down
```

首次启动自动拉取镜像。端口由 `WEB_PORT` 控制，
数据与命令行共用；详见 [Web Studio](webapp.md)。

## 构建、测试和停止

```bash
# 修改 Dockerfile / requirements 后，选择同一后端构建并启动
./scripts/dev.sh cpu build
./scripts/dev.sh cpu up -d
./scripts/dev.sh cpu test
./scripts/dev.sh cpu down

# GPU 对应命令
./scripts/dev.sh gpu build
./scripts/dev.sh gpu test
```

`up` 使用已有/拉取的镜像，日常使用无需构建。修改 Dockerfile 或依赖时，使用对应的 `build`。
`docker-compose.build.yml` 已移除，本地构建由启动脚本调用同一个 Dockerfile，按所选
后端传入基础镜像和 PyTorch 安装源；镜像标签通过 Compose 解析，与 `up` 完全一致。
需要更新远端镜像时，先执行 `./scripts/dev.sh cpu pull` 或 `gpu pull`，再执行对应的 `up -d`。

原生 Compose 命令仍然可用；先运行一次 `./scripts/dev.sh cpu prepare` 同步 UID/GID：

```bash
docker compose up -d
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

CPU/GPU 共用同一个服务，切换后端会重建该服务容器。容器名由 Compose 项目决定，
使用 `compose exec vlm-dev` 或 `scripts/dev.sh ... shell`，不要硬编码 `vlm-ws-dev`。
需要同时启动多个副本时，为每个副本设置不同的 `COMPOSE_PROJECT_NAME` 和端口。

## 文件、缓存和不同发行版

- 仓库绑定挂载到 `/workspace/vlm_ws`，数据、配置和实验输出保留在仓库目录。
- HOME 和模型缓存位于项目命名卷 `dev-home`，停止/重建容器后保留。
- 启动脚本仅同步 `.env` 的 `LOCAL_UID`、`LOCAL_GID`，其他设置保留；同名 shell 环境变量可显式覆盖。
- 容器 shell 使用镜像内的 Python，不自动激活绑定挂载进来的宿主 `.venv`。
- 仓库挂载使用 SELinux `z` 共享标签，供启用 SELinux 的发行版使用；该设置会重新标记仓库目录。
- 缺失的宿主绑定挂载路径直接报错，不会自动创建由 root 拥有的目录。
- 使用独立的共享内存，默认 `2gb`；可在 `.env` 中增大 `SHM_SIZE`。
- Jupyter/TensorBoard 默认绑定宿主 `127.0.0.1`，通过 `.env` 调整地址和端口。

如果旧 home 卷归属其他 UID，先检查卷所有权或使用新的 Compose 项目名分配独立 home。
`down` 保留卷；`down -v` 会删除缓存和容器内保存的会话，不用于日常切换。

## VS Code

CPU 与 CUDA 的 Dev Container 配置分别位于 `.devcontainer/cpu/devcontainer.json` 和
`.devcontainer/cuda/devcontainer.json`，使用与 CLI 相同的 Compose 服务。
初始化命令在宿主同步 UID/GID，并执行 `scripts/dev.sh cpu|gpu ensure-image`：

1. 本地已有对应架构的镜像时直接复用。
2. 本地没有镜像时，先从 Docker Hub 拉取 CPU 或 CUDA 镜像。
3. 拉取失败时，用同一个 Dockerfile 和所选后端的构建参数在本地构建，成功后继续打开容器。

Dev Containers 使用 `.devcontainer/docker-compose.yml` 禁止启动时再次拉取，
因此回退构建的镜像也能直接使用。初始化失败时停止打开，VS Code 日志保留拉取和构建错误。
如果基础镜像或依赖也没有缓存，本地构建仍需要网络；固定到 `@sha256:...` 的镜像
拉取失败时会直接报错，避免用本地构建替代指定摘要。
VS Code 不再另外修改镜像的用户 ID。
使用 `Dev Containers: Reopen in Container`；需要 CUDA 时选择 NVIDIA CUDA 配置。
切换 CPU/CUDA 配置前，先在宿主执行当前模式的 `scripts/dev.sh ... down`，再重新选择
配置打开，避免编辑器复用仍在运行的另一模式容器。
之后重新打开会复用本地镜像；需要更新预构建镜像时，先运行 `scripts/dev.sh cpu|gpu pull`。
修改依赖后，先在宿主运行
对应的 `scripts/dev.sh ... build`，再重建开发容器。这也避免了不同 Compose/Bake
版本对 Dev Containers 临时构建文件的权限处理差异。

终端采用 `af-magic` 风格：灰色分隔线、彩色路径和 Git 分支，
例如 `/workspace/vlm_ws (main*) »`；有未提交改动时显示 `*`，离开 Git 仓库后隐藏分支。
未安装 Oh My Zsh 的默认 CPU/CUDA 镜像也支持该样式。
Compose 从绑定仓库的 `scripts/docker` 读取终端模板，更新模板无需重新构建镜像；
容器启动时同步受项目管理的配置，并保留自定义配置。已打开的终端执行
`source ~/.zshrc` 可重新加载同步后的提示符。

## 可选个人开发工具

默认环境不安装 Node/Codex/opencode，也不挂载个人配置或凭据。
如需这些工具，构建个人镜像，再显式启用 `docker-compose.host-tools.yml`：

```bash
export VLM_CPU_IMAGE=vlm-ws:cpu-tools
export INSTALL_OPTIONAL_SHELL_TOOLS=1
./scripts/dev.sh cpu build
VLM_HOST_TOOLS=1 ./scripts/dev.sh cpu up -d
VLM_HOST_TOOLS=1 ./scripts/dev.sh cpu shell
```

宿主的 `~/.config/opencode`、`~/.local/share/opencode/auth.json` 和 `~/.agents` 必须存在；
可通过 `.env` 中的 `HOST_OPENCODE_CONFIG`、`HOST_OPENCODE_AUTH`、`HOST_AGENTS_DIR` 改路径。
三者只读挂载，认证文件在启动时复制到容器 HOME；会话数据库留在命名卷中。
这些工具独立于研究依赖；默认发布镜像不包含滚动更新的 CLI 安装。

## 发布与复现

自动发布需要在仓库 Actions Secrets 中配置 `DOCKERHUB_USERNAME=zgzdocker` 和具有
Read & Write 权限的 `DOCKERHUB_TOKEN`；本机手动发布使用 `docker login` 的认证。

`.github/workflows/docker-publish.yml` 在原生 amd64、arm64 runner 分别构建并运行
环境检查和项目测试；CUDA 镜像在 amd64 runner 上验证依赖及 CPU 路径。
GitHub runner 没有 GPU，实际 GPU 计算由 NVIDIA 宿主的 `gpu check` 验证。
两种 CPU 架构都通过后才合并 `:cpu` 标签。

旧 `.env` 中的 `VLM_IMAGE`、`BASE_IMAGE` 已不再读取；自定义镜像分别迁移到
`VLM_CPU_IMAGE`、`VLM_CUDA_IMAGE`，基础镜像分别使用 `CPU_BASE_IMAGE`、`CUDA_BASE_IMAGE`。
无需删除现有数据、实验输出或旧镜像。

正式实验记录镜像摘要、配置、随机种子和模型/数据来源。可将 `.env` 的镜像变量设为
`zgzdocker/vlm-ws:cpu-sha-<完整提交号>`、`zgzdocker/vlm-ws:cuda12.1-sha-<完整提交号>`，
或 `zgzdocker/vlm-ws@sha256:<digest>`；日常 `:cpu`、`:cuda12.1` 标签会随发布更新。
直接依赖虽已固定，基础镜像标签和传递依赖仍可能变化；精确复现使用已发布镜像摘要。

## 容器内服务

```bash
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```
