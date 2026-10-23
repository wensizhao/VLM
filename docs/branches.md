# 分支与上游同步

`upstream` 是 `wensizhao/VLM`，`origin` 是自己的 `ZGZ713912/VLM`。

`merge/upstream-main-webui-docker` 整合了最新上游主线、CPU/CUDA 容器、
Web Studio、官方标注评估、normality baseline 和可复现的数据/特征缓存流程。
它通过合并保留原提交记录，并推送到自己的 `origin` 同名分支。

继续同步上游时，在此分支执行：

```bash
git switch merge/upstream-main-webui-docker
git fetch upstream
git merge upstream/main
./scripts/dev.sh cpu test
git push origin HEAD
```

上游有新提交时，按功能解决冲突后提交。特别检查显式 `data.train/test` 路径、
官方标签来源、真实 `frame_indices` 聚合，以及 CPU/CUDA 和 Web Compose 组合。

本次保留本地 `main`、`dev/v1`、`feat/web-studio` 及其已有工作树。
数据、权重、运行结果和 `.env` 留在本机；新分支不包含这些运行产物。
