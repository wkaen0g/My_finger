# MicroGesture

MicroGesture 是一个基于摄像头手势识别的桌面控制项目。它通过 MediaPipe 检测手部关键点，并将手势映射为鼠标移动、点击、拖拽、滚动等系统输入操作。

## 功能特点

- 识别手掌、拳头、两指和捏合手势
- 支持空中点按与右键触发
- 支持光标移动、拖拽和滚动
- 提供系统托盘菜单，便于切换跟踪、调节灵敏度和退出程序
- 支持摄像头自动重连与日志输出

## 项目结构

- `microgesture/main.py`：程序主入口，负责串联整个手势处理流水线
- `microgesture/pipeline/`：包含摄像头采集、手部检测、手势分类、点按/捏合/滚动检测与光标控制
- `microgesture/system/`：负责系统输入模拟和托盘菜单
- `microgesture/config.json`：默认配置文件
- `start_microgesture.bat`：Windows 下的快速启动脚本

## 环境要求

- Python 3.9+
- 可用摄像头
- Windows / Linux / macOS（本项目代码中已包含跨平台兼容逻辑，但 Windows 下托盘支持最完整）

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 运行方式

### Windows

直接运行：

```bash
start_microgesture.bat
```

或者在终端中执行：

```bash
python -m microgesture.main
```

## 配置说明

默认配置文件位于：

- `microgesture/config.json`

你可以通过修改该文件调整摄像头参数、手势阈值、光标灵敏度和日志设置。

## 常见问题

### 1. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 确认系统已授予摄像头权限
- 尝试修改 `microgesture/config.json` 中的 `camera.device_id`

### 2. 缺少 MediaPipe 模型
- 程序首次启动时会自动下载模型文件到 `microgesture/models/`
- 如果下载失败，请检查网络连接或手动下载后放置到对应目录

### 3. 鼠标控制异常
- 运行时请避免将鼠标移动到屏幕角落，避免触发 PyAutoGUI 的安全保护
- 可以通过托盘菜单关闭跟踪或调整灵敏度

## 说明

这个项目适合用于学习：
- 计算机视觉手势识别
- MediaPipe 的实际应用
- Python 下的多线程/状态机设计
- 基于摄像头的交互控制系统
