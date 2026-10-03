# 人工标注工具

## 概述

这是一个用于VLM-VAD项目的人工标注Web界面，支持视频异常检测的prompt标注任务。

## 功能特点

- 🎥 **视频播放**：支持多种视频格式，逐帧播放
- 📝 **Prompt标注**：三种类型的prompt选择（Label/Scene/Contrast）
- 🎯 **时间轴标注**：可视化标注异常发生的时间段
- 🔍 **质量检查**：自动检查标注质量和一致性
- 📊 **进度跟踪**：实时显示标注进度和统计信息
- 👥 **多用户支持**：支持多用户协作标注

## 安装和运行

### 1. 安装依赖

Flask 已包含在项目依赖中，在**仓库根目录**安装即可：

```bash
# 仓库根目录（vlm_ws/）
pip install -r requirements/base.txt   # 完整环境
# 或只装标注工具所需依赖
pip install flask==3.0.3
```

Docker 环境（`docker compose up -d` 构建的镜像）已预装，无需额外安装。

### 2. 启动服务器
```bash
cd tools/annotation_tool
python app.py
```

### 3. 访问界面
打开浏览器访问：http://localhost:5000

## 使用指南

### 1. 登录系统
- 使用用户名和密码登录
- 管理员可以查看所有标注数据
- 普通用户只能查看自己的标注

### 2. 选择视频
- 从视频列表中选择要标注的视频
- 支持按视频ID、类型、状态筛选

### 3. 观看视频
- 使用视频播放器控制视频播放
- 支持逐帧查看、快进、快退
- 可以调节播放速度

### 4. 标注异常
1. 播放视频，找到异常发生的时间段
2. 在时间轴上标记异常范围
3. 选择合适的prompt类型和具体描述
4. 填写异常严重程度
5. 保存标注结果

### 5. 质量检查
- 系统自动检查标注的合理性
- 提供标注质量评分
- 标注不一致时会提示警告

## 数据结构

### 标注数据格式
```json
{
  "video_id": "video_001",
  "annotator_id": "user_001",
  "annotation_timestamp": "2024-01-01 10:00:00",
  "anomaly_segments": [
    {
      "start_time": 10.5,
      "end_time": 15.2,
      "prompt_type": "label",
      "prompt": "a fighting behavior",
      "severity": "high",
      "confidence": 0.9,
      "notes": "多人斗殴事件"
    }
  ]
}
```

### Prompt类型定义
- **label**: 直接描述异常行为
- **scene**: 描述场景和异常活动
- **contrast**: 对比正常和异常行为

## 配置说明

### 视频配置
```python
VIDEO_CONFIG = {
    "upload_dir": "data/videos",
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv"],
    "max_file_size": 500 * 1024 * 1024,  # 500MB
    "thumbnail_interval": 1.0  # 缩略图间隔（秒）
}
```

### 标注配置
```python
ANNOTATION_CONFIG = {
    "prompt_types": {
        "label": {
            "template": "a {behavior} behavior",
            "positive_labels": ["fighting", "theft", "vandalism"],
            "negative_labels": ["normal", "walking", "standing"]
        },
        "scene": {
            "template": "a {scene} with abnormal activity",
            "positive_scenes": ["street", "mall", "bank"],
            "negative_scenes": ["empty street", "quiet mall"]
        },
        "contrast": {
            "template": "normal vs abnormal {activity}",
            "positive_activities": ["behavior", "movement", "action"],
            "negative_activities": ["normal behavior", "regular movement"]
        }
    },
    "severity_levels": ["low", "medium", "high"],
    "confidence_threshold": 0.7
}
```

## API接口

### 视频相关
- `GET /api/videos` - 获取视频列表
- `POST /api/videos` - 上传新视频
- `GET /api/videos/{video_id}` - 获取视频详情
- `DELETE /api/videos/{video_id}` - 删除视频

### 标注相关
- `GET /api/annotations` - 获取标注列表
- `POST /api/annotations` - 创建标注
- `GET /api/annotations/{annotation_id}` - 获取标注详情
- `PUT /api/annotations/{annotation_id}` - 更新标注
- `DELETE /api/annotations/{annotation_id}` - 删除标注

### 统计相关
- `GET /api/stats/progress` - 获取标注进度
- `GET /api/stats/quality` - 获取标注质量统计
- `GET /api/stats/user/{user_id}` - 获取用户标注统计

## 扩展功能

### 1. 自动预标注
- 基于运动检测自动预标注异常区域
- 提供预标注结果供用户确认和修改

### 2. 相似度推荐
- 基于历史标注推荐相似的prompt
- 减少标注者的思考时间

### 3. 批量标注
- 支持批量处理多个视频
- 提高标注效率

### 4. 数据导出
- 支持导出多种格式的标注数据
- 兼容主流的机器学习框架

## 技术架构

### 前端技术
- **HTML5**: 视频播放和界面渲染
- **CSS3**: 界面样式和动画
- **JavaScript**: 交互逻辑和数据处理
- **Vue.js**: 前端框架（可选）

### 后端技术
- **Python**: 主要开发语言
- **Flask**: Web框架
- **SQLite**: 数据库（可扩展为PostgreSQL）
- **OpenCV**: 视频处理
- **FFmpeg**: 视频编解码

### 部署选项
- **本地部署**: 开发和测试环境
- **Docker容器**: 标准化部署
- **云服务**: AWS/Azure/GCP

## 故障排除

### 常见问题
1. **视频无法播放**
   - 检查视频格式是否支持
   - 确认视频文件完整性
   - 检查网络连接

2. **标注无法保存**
   - 检查数据库连接
   - 确认用户权限
   - 检查数据格式

3. **界面卡顿**
   - 减少同时打开的视频数量
   - 清理浏览器缓存
   - 检查网络带宽

### 日志查看
```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱：annotation@example.com
- GitHub Issues：[项目Issues页面](https://github.com/example/vlm-vad/issues)