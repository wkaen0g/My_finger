# 微手势识别系统 — 开发文档

最后更新: 2026-07-08

## 概述

基于笔记本/PC 自带摄像头的消费级手势交互系统。通过摄像头追踪手部关键点，将手势映射为系统鼠标事件（光标移动、点击、右键、拖拽、滚动），支持用户自定义手势注册。

**当前状态**: Phase 1-4 全部完成，52 单元测试全过，6 类 ONNX 模型 100% val_acc。

## 核心设计决策汇总

| 决策点 | 选择 | 现状 |
|--------|------|------|
| 手势集合 | 6类静态手势 + 捏合/点按 + DTW 自定义 | ✅ 全部实现 |
| 底层追踪 | MediaPipe Hands (21点手部关键点), Tasks API | ✅ |
| 手势识别 | RuleEngine (几何归一化) + ONNX MLP 影子模式 (conf≥90% 接管) | ✅ |
| 光标控制 | 混合模式：PALM_OPEN/PINCH=光标, SINGLE_FINGER/FIST/TWO_FINGER=冻结 | ✅ |
| 点击 (左键) | SINGLE_FINGER + AirTapDetector 积分检测, 武装锁存 15 帧 | ✅ |
| 点击 (右键) | FIST + AirTapDetector (fist_tap 模式) | ✅ |
| 拖拽 | PinchDetector 滞后状态机 (start=0.35, release=0.55) | ✅ |
| 滚动 | TWO_FINGER 双指 y 位移 | ✅ |
| DTW | 运动量自动分割 (静止→运动→静止) + 每模板自适应阈值 | ✅ |
| 训练数据 | 自采 20,500 样本 (6 类) | ✅ |
| ONNX 模型 | 70→128→128→64→6, val_acc 100%, 测试集 99.98% | ✅ |
| 线程模型 | 5 线程: tk-ui + camera + pipeline + tray + main | ✅ |
| 配置系统 | 控制面板 (4 标签页) + config.json 热重载 | ✅ |

---

## 项目路径

所有代码在 `D:\Myfinger\`，包名为 `microgesture`（即 `D:\Myfinger\microgesture\`）。

---

## Phase 1: 规则引擎 MVP（目标：可用的手势控制系统）✅ 已完成 2026-05-12

### 1.1 项目骨架搭建
- [x] 在 `D:\Myfinger\` 创建 `microgesture/` 包结构
- [x] `main.py` 入口：启动托盘、初始化管道、启动事件循环
- [x] `config.py` + `config.json`：配置加载与热重载（watchdog 监听文件变化）
- [x] 日志系统：`logging` 模块，分级(DEBUG/INFO/WARNING/ERROR)，输出到 `logs/microgesture.log`
- [x] 错误降级：摄像头丢失自动重连、无手检测超时休眠、模型加载失败告警

### 1.2 摄像头采集 (`pipeline/capture.py`)
- [x] OpenCV 摄像头采集，30fps 目标
- [x] 采集线程 + `collections.deque(maxlen=2)` 帧缓冲
- [x] 摄像头丢失自动重连（指数退避，最多5次）

### 1.3 MediaPipe 推理 (`pipeline/detector.py`)
- [x] MediaPipe Hands 封装：输入帧 → 输出 21点归一化坐标（使用 Tasks API，自动下载模型）
- [x] 置信度过滤（<0.5 丢弃帧）
- [x] 手选择：自动选最高置信度手
- [x] 处理线程主循环：取帧→推理→手势引擎→系统事件

### 1.3.1 MediaPipe 手部关键点索引（21点）

```
食指 ──┐
        │  8 ─ 指尖 (TIP)
        │  7 ─ 远指间关节 (DIP)
        │  6 ─ 近指间关节 (PIP)
        │  5 ─ 掌指关节 (MCP)
        │
拇指 ──┤  4 ─ 指尖 (TIP)
        │  3 ─ 指间关节 (IP)
        │  2 ─ 掌指关节 (MCP)
        │  1 ─ 腕掌关节 (CMC)
        │
        │  0 ─ 手腕 (WRIST)
        │
中指 ──┤ 12 ─ 指尖 (TIP)
        │ 11 ─ DIP
        │ 10 ─ PIP
        │  9 ─ MCP
        │
无名指─┤ 16 ─ 指尖 (TIP)
        │ 15 ─ DIP
        │ 14 ─ PIP
        │ 13 ─ MCP
        │
小指 ──┤ 20 ─ 指尖 (TIP)
        │ 19 ─ DIP
        │ 18 ─ PIP
        │ 17 ─ MCP
```

**各模块使用的关键点（实际实现）：**

| 模块 | 使用关键点 | 用途 |
|------|-----------|------|
| `cursor.py` | 食指 TIP(8) x,y | 光标位移（1€滤波 + 死区） |
| `air_tap.py` | 食指 MCP(5)+PIP(6)+TIP(8) | 弯曲比积分式点按 + 光标抑制 |
| `gesture_engine.py` | TIP(4,8,12,16,20) + MCP(2,5,9,13,17) | 指尖-MCP距离 → 伸/屈判定 |
| `pinch.py` | 拇指 TIP(4) + 食指 TIP(8) + WRIST(0) + 中指 MCP(9) | 捏合距离归一化 + 滞后状态机 |
| `scroll.py` | 食指 TIP(8) + 中指 TIP(12) | 双指中点 y 位移 |

### 1.4 规则引擎 (`pipeline/gesture_engine.py`)

6 类几何判定。所有指尖-MCP 距离除以 `hand_scale` (wrist→middle_MCP) 做摄像头距离归一化。

**判定优先级**: SINGLE_FINGER → FIST → PINCH → TWO_FINGER → PALM_OPEN → fallback

| 手势 | 判定规则 | 阈值 (归一化比例) |
|------|---------|-------------------|
| SINGLE_FINGER | 食指 > open_ratio, 食指>中指×1.3, 中指<0.60, 无名指<0.55, 小指<0.45, 拇指不检查 | open_ratio=0.70 |
| FIST | 4+ 指 < fist_ratio, 且食指不是优势指 (防止点击弯曲误判) | fist_ratio=0.45 |
| PINCH | 拇指-食指归一化距离 < pinch_ratio, 且食指非全伸 | pinch_ratio=0.35 |
| TWO_FINGER | 食指+中指 > fist_ratio, 无名指+小指 < fist_ratio | — |
| PALM_OPEN | 3+ 指 > open_ratio, 或 fallback | — |
| NO_HAND | MediaPipe 无检测 | — |

### 1.5 空中点按检测 (`pipeline/air_tap.py`) — 积分相位 + 手势门控

**原理**: 食指弯曲比 `ratio = |TIP(8)-MCP(5)| / |PIP(6)-MCP(5)|`
- 直指: ratio ≈ 2.5~3.0, 弯曲: ratio 下降

**双相积分状态机**:
- BEND 相: 累计 |Δratio| > `min_bend(0.10)` → 有效的预备动作
- REBOUND 相: 累计 Δratio > `tap_threshold(0.20)` → 触发 TapEvent
- 超时: BEND 12帧, REBOUND 12帧, 冷却 8帧

**手势门控**: 仅 SINGLE_FINGER/FIST 时运行 AirTapDetector，PALM_OPEN 下跳过（消除光标移动中的误触）

**阈值历史**: tap_threshold 0.3→0.20, min_bend 0.15→0.10 (减小所需手指动作幅度)

### 1.6 捏合检测 (`pipeline/pinch.py`) — 已加滞后
- [x] 拇指尖-食指尖归一化距离 / 手腕-中指MCP 距离
- [x] 滞后状态机：start_threshold(0.35) → 捏合开始，release_threshold(0.55) → 释放
- [x] 3帧去抖确认状态切换，防止振荡
- [x] 捏合状态跟踪：用于拖拽（捏合+移动）

### 1.7 光标控制 (`pipeline/cursor.py`)
- [x] 1€滤波器平滑食指 TIP(8) 坐标
- [x] 手掌张开/捏合：指尖帧间位移 → 光标移动增量
- [x] 握拳/双指：冻结光标，不发送移动事件
- [x] 死区过滤：位移 < deadzone × 屏幕宽度 → 忽略
- [x] 灵敏度使用实际屏幕分辨率（非硬编码 1920×1080）

### 1.8 双指滚动 (`pipeline/scroll.py`)
- [x] 检测双指伸出状态
- [x] 两指尖(食指TIP+中指TIP)中点 y 轴位移 → 垂直滚动量
- [x] 1€滤波平滑，使用实际屏幕高度

### 1.9 系统事件模拟 (`system/input.py`) — 已改为 Win32 原生 API
- [x] 光标移动：Win32 `SetCursorPos`（替换 pyautogui `moveRel`）
- [x] 点击/右键/拖拽/滚动：Win32 `mouse_event`
- [x] 软钳制：光标触及屏幕边缘时钳住，不再抛异常

### 1.10 系统托盘 (`system/tray.py`) — 已重构
- [x] pystray 托盘图标与菜单
- [x] 动态勾选标记：灵敏度(低/中/高)、右键模式、Tracking 状态
- [x] 状态指示：图标颜色（绿=正常，黄=无摄像头，灰=休眠）
- [x] 摄像头状态实时回调更新图标

---

## Phase 2: 分类器替换规则引擎 ✅ 已完成

### 2.1 数据采集工具
- [x] 引导式采集：`guided_collector.py` — PIL 中文指令 + 倒计时 + `--gestures` 筛选手势
- [x] `--frames` 参数可调每类帧数 (默认 500)
- [x] `start_collector_single.bat` — 单指专用采集脚本

### 2.2 训练管道
- [x] MLP 分类器：70→128→128→64→**6**，ReLU+Dropout(0.3)
- [x] 输入：70 维特征，输出：**6 类** (FIST, NO_HAND, PALM_OPEN, PINCH, TWO_FINGER, SINGLE_FINGER)
- [x] 训练数据：**20,500 样本** (FIST/NO_HAND/PALM_OPEN/PINCH/TWO_FINGER 各 3500, SINGLE_FINGER 3000)
- [x] 验证准确率：**100.0%**，测试集：**99.98% (8498/8500)**
- [x] 导出 ONNX：`microgesture/models/classifier.onnx`
- [x] `start_train.bat` — 训练+导出脚本
- [x] `start_test.bat` — ONNX+PyTorch 双模型测试
- [x] `training/model_test.py` — 准确率/混淆矩阵/Precision/Recall/F1

### 2.3 影子模式
- [x] ONNX Runtime 封装 (每帧推理), 置信度 ≥ 90% 自动覆盖规则引擎
- [x] `GestureRecognizer` 抽象基类 + `StaticClassifier` (RuleEngine 适配器)

---

## Phase 3: 自定义手势 ✅ 已完成

### 3.1 DTW 匹配器 (`recognition/dtw_matcher.py`)
- [x] 运动量自动分割: 静止→运动→静止（食指指尖速度 > motion_threshold）
- [x] `motion_threshold=0.012` (~8px/帧), `still_frames=15` (~0.5s)
- [x] `max_record_frames=180` (~6s), `min_record_frames=15`
- [x] fastdtw 匹配 (radius=10), 路径长度归一化
- [x] **每模板自适应阈值**: `self_distance × 2.0` (注册时自动计算 3 次录制的平均 DTW 距离)
- [x] 全局 `match_threshold=50.0` 作为天花板
- [x] 平移不变性：每帧减去手腕坐标
- [x] 原子写入 templates.json (mkstemp + os.replace)
- [x] 冷却 90 帧 (~3s) 防止重复触发

### 3.2 DTW 训练器 (`recognition/dtw_trainer.py`)
- [x] 3 次录制 → DBA 平均 (5 轮迭代)
- [x] 倒计时: 首次 8s, 后续 5s
- [x] 录制引导: "开始! 做手势 (做完保持静止)" → "录制中... N 帧 (完成后静止)" → "静止确认... N"
- [x] 超时/太短 → 自动清除事件，不会卡死轮询
- [x] `self_distance` 存入模板 → 自动计算匹配阈值

### 3.3 注册与管理
- [x] 控制面板 DTW 标签页: Name 输入框 + 快捷键选择器 + Register/Delete 按钮
- [x] 托盘菜单 "Register Gesture" 快捷入口
- [x] 注册后自动刷新模板列表 + 按钮恢复
- [x] 训练中状态轮询 (300ms) + 异常兜底

---

## Phase 4: 精修与增强 ✅ 已完成

### 4.1 点击系统重构 (2026-07-07)
- [x] AirTapDetector 手势门控: 仅 SINGLE_FINGER/FIST 时运行 (消除 PALM_OPEN 误触)
- [x] 武装锁存状态机: 手势丢失后 15 帧内仍可消费 tap (防点击弯曲掉手势)
- [x] Tap 消费后立即 disarm (防双击)
- [x] 右键修复: FIST 手势也跑 tap 检测器
- [x] 阈值下调: tap_threshold 0.3→0.20, min_bend 0.15→0.10
- [x] 实测转化率: 100% (8/8), bend 值从 0.234~0.510 降至 0.111~0.265

### 4.2 手势引擎归一化 (2026-07-07)
- [x] 指尖-MCP 距离归一化: 除以 hand_scale (wrist→middle_MCP), 摄像头距离不变
- [x] open_ratio: 0.25(绝对)→0.70(比例), fist_ratio: 0.12→0.45
- [x] SINGLE_FINGER 判定优化: 移除拇指检查, 中/无名指宽松阈值(0.60/0.55), 食指指数优势(>1.3x中指)
- [x] FIST 护盾: 食指优势时不抢 SINGLE_FINGER (防点击弯曲误判)
- [x] 捏合归一化复用 hand_scale 避免重复计算

### 4.3 基础设施修复 (2026-07-07)
- [x] tkinter 跨线程: Queue + after 轮询 (修复 "main thread not in main loop")
- [x] Config JSON 竞态: reload 重试 + watchdog 0.5s 防抖
- [x] fastdtw 安装 + 一次性告警 (修复 DTW 静默降级为欧氏距离)
- [x] 日志 TRACE 级别: `--trace` CLI, 逐帧数据从 DEBUG 迁出, 默认 INFO
- [x] 系统托盘退出: quit_app→tray.stop(), shutdown() 仅在主线程 finally 执行一次
- [x] Tk 线程优雅退出: `_ui_queue.put("stop")` → root.quit()

### 4.4 模型重训 (2026-07-07)
- [x] 6 类 ONNX: 20,500 样本, val_acc 100%, 测试集 99.98%
- [x] SINGLE_FINGER 增量 2000 帧

### 4.5 工具脚本 (2026-07-07)
- [x] `start_collector_single.bat` — 单指专用采集
- [x] `start_train.bat` — 训练+导出 ONNX
- [x] `start_test.bat` — ONNX+PyTorch 双模型测试
- [x] guided_collector `--gestures` 参数

### 4.6 代码质量 (2026-07-07)
- [x] 52 单元测试全过 (air_tap/gesture_engine/pinch/cursor/dtw/main/config)
- [x] 线程安全: Lock + Event + try/finally
- [x] 原子写入: mkstemp + os.replace
- [x] 异常处理: 训练完成/超时/失败 均有 event 清除路径

---

## 代码质量 ✅

全项目累计 52 单元测试。关键修复：

| # | 类别 | 修复 |
|---|------|------|
| 1 | 测试 | 52 个单元测试 (air_tap 6, gesture_engine 14, pinch 6, cursor 8, dtw 12, config 1, main 3, 其他) |
| 2 | 异常处理 | _run_loop 计数+节流, shutdown() try/except, key_combo try/finally |
| 3 | 线程安全 | Lock + Event, Tk Queue 轮询, 训练完成/超时 event 清除 |
| 4 | 资源管理 | try/finally + context manager (capture/detector) |
| 5 | 原子写入 | templates.json (mkstemp + os.replace), Config.save() |
| 6 | 配置一致性 | 统一 config.json 与代码默认值 |
| 7 | 日志分级 | TRACE→DEBUG→INFO→WARNING→ERROR 五级 |

---

## 依赖清单

```
# 核心
opencv-python>=4.8
mediapipe>=0.10
numpy>=1.24
pystray>=0.19
pillow>=10.0
watchdog>=3.0

# Phase 2 训练与推理
torch>=2.0
onnxruntime>=1.15

# Phase 3 DTW
fastdtw>=0.3

# 开发
pytest>=7.0
```

---

## 启动命令

```bash
# 日常使用 (INFO 级别)
python -m microgesture.main

# 调试 (Inference 摘要 + DTW 详情)
python -m microgesture.main --debug

# 逐帧跟踪 (每帧手势数据)
python -m microgesture.main --trace

# 采集单指数据
start_collector_single.bat [帧数]

# 训练模型
start_train.bat

# 测试模型
start_test.bat
```

---

## 手势→事件映射 (当前)

| 手势 | 光标 | 点击 | 其他 |
|------|------|------|------|
| PALM_OPEN | 移动 | — | 捏合拖拽 |
| SINGLE_FINGER | 冻结 | 左键 (AirTapDetector + 武装锁存) | — |
| FIST | 冻结 | 右键 (fist_tap 模式) | DTW 触发 |
| TWO_FINGER | 冻结 | — | 垂直滚动 |
| PINCH | 移动 | — | — |
