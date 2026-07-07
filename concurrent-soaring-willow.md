# 微手势识别系统 — 实施计划

## 概述

基于笔记本/PC 自带摄像头的消费级手势交互系统。通过摄像头追踪手部关键点，将手势映射为系统鼠标事件（光标移动、点击、右键、拖拽、滚动），支持用户自定义手势注册。

## 核心设计决策汇总

| 决策点 | 选择 |
|--------|------|
| 手势集合 | 静态手势(5类) + 精细手指运动(捏合/点按) + 自定义手势(DTW) |
| 底层追踪 | MediaPipe Hands (21点手部关键点) |
| 光标控制 | 混合模式：手掌张开=光标模式，握拳=冻结光标/手势模式 |
| 点击 | 捏合(拖拽) + 空中点按(单击)，分工不重叠 |
| 右键 | 握拳冻结态下点按 或 双指点按，配置可选 |
| 滚动 | 双指伸出(食指+中指)，仅垂直滚动 |
| 分类器 | 静态5类(手掌张开/握拳/双指/捏合/无手)，Geometry规则先上线 |
| 训练数据 | HaGRID预训练 + 合成数据扩增 + 自采微调 |
| 线程模型 | 双线程：采集线程 + 处理线程，deque缓冲(maxlen=2) |
| 配置系统 | 系统托盘 + JSON热重载，后续加GUI面板 |
| 项目结构 | `microgesture/` 包，`models/` 目录放ONNX |
| 交付节奏 | Phase 1(规则引擎可用) → Phase 2(分类器替换) → Phase 3(自定义手势) → Phase 4(精修) |

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
纯几何规则判断5个手势状态（每5帧输出 DEBUG 日志）：

| 手势 | 判定规则 |
|------|---------|
| 手掌张开 | 所有5指指尖-MCP距离 > open_threshold(0.25) |
| 握拳 | 所有5指指尖-MCP距离 < fist_threshold(0.16) |
| 双指伸出 | 食指+中指指尖-MCP距离 > open_threshold，其余 < fist_threshold |
| 捏合 | 拇指尖-食指尖归一化距离 < pinch_ratio(0.35) |
| 无手 | MediaPipe 无检测或置信度<0.5 |

### 1.5 空中点按检测 (`pipeline/air_tap.py`) — 已改为积分相位检测
- [x] 信号源从 z 轴加速度改为**弯曲比**：`ratio = |TIP(8)-MCP(5)| / |PIP(6)-MCP(5)|`
- [x] 双相积分状态机：BEND 相（累计\|Δratio\|）→ REBOUND 相（累计 Δratio）→ 达标触发
- [x] BEND 超时 12 帧，REBOUND 超时 12 帧，冷却 8 帧
- [x] 输出：`TapResult(event, suppress_cursor, ratio, dratio)`
- [x] 光标抑制：\|Δratio\| > 0.1 时跳过光标移动

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

## Phase 2: 分类器替换规则引擎（目标：提升识别精度和鲁棒性）🔧 搭建中

### 2.1 数据采集工具 (`training/data_collector.py`, `training/guided_collector.py`)
- [x] 自由模式：指定手势标签 → 逐帧采集 70 维特征 + 标签 → .npz
- [x] 引导式采集：全屏 PIL 中文指令覆盖，10 秒准备倒计时，自动录 500 帧/手势
- [x] 进度显示：实时显示 `已录/目标` 帧数，超时保护 5000 迭代
- [ ] 伪标签纠错（后续）
- [x] 输出标注数据格式：`(70维特征向量, 标签)`

### 2.2 训练管道 (`training/train_classifier.py`, `training/export_onnx.py`)
- [x] HaGRID 预训练数据加载（`hagrid_loader.py` 本地图像 + `hagrid_download.py` HuggingFace 下载）
- [x] 20BN-Jester V1 伪标签预训练数据加载（`jester_loader.py`，规则引擎自动标注）
- [x] 共享模块 `_hagrid_common.py`：统一 CLASS_MAP / GESTURE_LABELS / 路径解析 / 特征保存
- [x] 统一 CLI `hagrid.py`：`download` / `process` / `jester` 三个子命令
- [x] MLP 分类器：70→128→128→64→5，ReLU+Dropout(0.3)，CosineAnnealing 调度
- [x] 输入：70 维特征（63 坐标 + 5 指尖-MCP距离 + 1 捏合距离 + 1 弯曲比）
- [x] 输出：5类（手掌张开/握拳/双指/捏合/无手）
- [x] 自采数据训练：5 类 × 500 帧 = 2500 样本，验证准确率 **100%**
- [x] 导出 ONNX 到 `microgesture/models/classifier.onnx`（验证通过）
- [x] `train_with_pretrain()` 两阶段训练：预训练数据集 → 自采数据微调

### 2.3 分类器推理 (`pipeline/classifier.py`)
- [x] ONNX Runtime 推理封装
- [x] 输出类别+置信度（Softmax）
- [x] 影子模式：规则引擎和分类器并行跑，置信度 ≥ 90% 自动切换
- [x] ONNX 每 5 帧推理一次（降低负载，避免卡顿）
- [x] `GestureRecognizer` 抽象基类 (`recognition/base.py`)：`predict(landmarks) → RecognitionResult(label, confidence, features)`

### 2.4 规则引擎实现基类接口 (`recognition/static_classifier.py`)
- [x] StaticClassifier 包装 RuleEngine，实现 `GestureRecognizer`
- [x] `gesture_engine.py` 通过影子模式工厂方法获取当前活跃的识别器

### 2.5 新增依赖
```
torch>=2.0
onnxruntime>=1.15
onnxscript          (torch.onnx 导出依赖)
```

### 2.6 新增文件
```
microgesture/
├── recognition/
│   ├── base.py                    # GestureRecognizer 抽象基类 + extract_features(70维)
│   └── static_classifier.py       # RuleEngine 适配器
├── pipeline/
│   └── classifier.py              # ONNX Runtime 推理 (ONNXClassifier)
├── training/
│   ├── __init__.py
│   ├── _hagrid_common.py           # 共享常量/标签/工具（单一事实来源）
│   ├── data_collector.py          # 核心采集器
│   ├── guided_collector.py        # 引导模式采集 (PIL中文 + 10s倒计时 + 500帧/手势)
│   ├── train_classifier.py         # MLP 训练管道 (含两阶段训练)
│   ├── export_onnx.py             # 导出 ONNX
│   ├── hagrid_loader.py            # HaGRID 本地图像特征提取
│   ├── hagrid_download.py          # HaGRID HuggingFace 下载
│   ├── jester_loader.py            # 20BN-Jester V1 伪标签加载
│   └── hagrid.py                   # 统一 CLI (download/process/jester)
├── models/
│   ├── hand_landmarker.task        # MediaPipe 模型
│   ├── best_model.pt               # PyTorch 最佳模型
│   └── classifier.onnx             # ONNX 分类器
└── training/data/
    ├── features_PALM_OPEN.npz      # 500 样本
    ├── features_FIST.npz            # 500 样本
    ├── features_TWO_FINGER.npz      # 500 样本
    ├── features_PINCH.npz           # 500 样本
    ├── features_NO_HAND.npz         # 500 样本
    └── metadata.json
```

---

## Phase 3: 自定义手势（目标：用户可注册个性化手势）

### 3.1 DTW 匹配器 (`recognition/dtw_matcher.py`) ✅ 2026-07-06
- [x] 状态机: IDLE → ARMING → RECORDING → MATCHING
- [x] 手势分段：握拳(开始) → 动作序列 → 握拳(结束)
- [x] 多模板存储：DBA 平均（3次 → 1个模板）
- [x] DBA (DTW Barycenter Averaging) 合成平均模板
- [x] 分段式 DTW 匹配：序列与各模板计算 fastdtw 距离，最小距离<阈值 → 匹配
- [x] 使用 `fastdtw` 库 (radius=10, 路径长度归一化)
- [x] 平移不变性：每帧减去手腕坐标

### 3.2 注册流程 ✅ 2026-07-06
- [x] 托盘菜单"Register Gesture..."入口
- [ ] 首次启动引导向导（后续）
- [x] 注册步骤：握拳→做动作→握拳结束，重复3次
- [x] 手势模板持久化到 `templates.json`

### 3.3 自定义手势触发 ✅ 2026-07-06
- [x] FIST 分隔 → 动作序列匹配 → 触发系统事件
- [x] 事件可配置：映射到键盘快捷键 (Win32 keybd_event)

---

## Phase 4: 精修与增强 🔧 部分完成

### 4.0 Phase 2/3 精修 ✅ 2026-07-06
- [x] FIST/PINCH 判定优先级修复（FIST → PINCH → TWO_FINGER → PALM）
- [x] ONNX 每帧推理（去掉 5 帧跳帧）
- [x] 双指滚动灵敏度 40.0→2.0 + 死区 0.03
- [x] 日志噪音优化（DEBUG 间隔 5→300 帧）
- [x] 训练数据量可调（--frames 参数）
- [x] 独立测试集支持（evaluate + test_dir）
- [x] config set/save 运行时持久化
- [x] InputController 键盘快捷键模拟

### 4.1 诊断面板 ✅ 2026-07-07
- [x] 预览画面上叠加实时诊断：FPS、死区模式(NORM/TAP)、推理源(ONNX/RULE) + 置信度
- [x] 实时叠加层：关键点、手势标签、DTW状态、诊断信息

### 4.2 自动恢复
- [ ] 光线恢复自动重新检测手
- [ ] 摄像头热插拔支持

### 4.3 设置 GUI 面板
- [ ] 灵敏度滑块
- [ ] 右键模式下拉选择
- [ ] 手势灵敏度实时预览

### 4.4 多显示器支持
- [ ] 检测活动显示器
- [ ] 光标跨屏移动

---

## 依赖清单

```
# 核心
opencv-python>=4.8
mediapipe>=0.10
numpy>=1.24
pyautogui>=0.9.54
pystray>=0.19
pillow>=10.0

# 配置与日志
watchdog>=3.0

# Phase 2 训练与推理
torch>=2.0
onnxruntime>=1.15
onnx>=1.14

# Phase 3 DTW
fastdtw>=0.3

# 开发
pytest>=7.0
```

---

## 验证方案

### Phase 1 验证
1. 运行 `python -m microgesture.main`
2. 系统托盘出现图标，绿色表示摄像头正常
3. 手掌张开，移动食指 → 光标跟随移动
4. 拇食指捏合 → 进入拖拽模式，可拖拽窗口
5. 食指向前快速点按 → 触发单击
6. 握拳冻结 → 光标停止移动
7. 握拳冻结态下点按 → 右键菜单出现
8. 双指伸出上下移动 → 页面滚动
9. 托盘切换灵敏度 → 光标响应速度变化

### Phase 2 验证
1. 运行采集工具录50帧/手势
2. 训练分类器，准确率 > 95%（测试集）
3. 影子模式下分类器置信度持续高于规则引擎
4. 自动切换后交互行为不变或改善

### Phase 3 验证
1. 注册一个自定义手势（如画圈→截图）
2. 在握拳冻结态下执行该手势 → 触发截图
3. 未注册手势不误触发
