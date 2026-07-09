#  — 项目答辩文稿

**MicroGesture — 基于摄像头的实时手势交互系统**

---

## 一、项目概述

本项目构建了一套基于笔记本/PC 内置摄像头的消费级手势交互系统。用户在摄像头前做出手部动作，系统实时将手势映射为系统鼠标事件——光标移动、点击、右键、拖拽、滚动——并可注册自定义手部轨迹绑定快捷键。全程无需额外硬件。

**核心指标**：

| 指标 | 数值 |
|------|------|
| 实时帧率 | 30fps |
| 手势类别 | 6 类静态 + 动态自定义 |
| ONNX 分类准确率 | 100% (验证), 99.98% (测试) |
| 点击转化率 | 100% (实测) |
| 代码规模 | 34 源文件, 5652 行 Python |
| 测试覆盖 | 52 单元测试, 零回归 |
| 端到端延迟 | < 100ms |


## 二、项目结构

```
microgesture/                    # 主包 (34 源文件, 5652 行)
├── main.py                      # 入口: 管线编排 + 手势派发 + 主循环
├── config.py / config.json      # 配置管理: JSON 热重载 + 线程安全读写
│
├── pipeline/                    # 实时处理管线
│   ├── capture.py               # 摄像头采集线程 (deque 帧缓冲)
│   ├── detector.py              # MediaPipe Hands 21点检测
│   ├── gesture_engine.py        # 规则引擎: 6 类归一化几何判定
│   ├── air_tap.py               # 空中点按: 积分相位检测 + 手势门控
│   ├── pinch.py                 # 捏合检测: 滞后状态机
│   ├── cursor.py                # 光标控制: 1€ 滤波 + 死区
│   ├── scroll.py                # 双指滚动: y 位移映射
│   └── classifier.py            # ONNX Runtime 推理
│
├── recognition/                 # 识别层
│   ├── base.py                  # GestureRecognizer 抽象基类
│   ├── static_classifier.py     # RuleEngine 适配器
│   ├── dtw_matcher.py           # DTW 序列匹配: 运动量分割 + 自适应阈值
│   └── dtw_trainer.py           # DTW 模板训练: 3次录制 + DBA 平均
│
├── system/                      # 系统层
│   ├── tray.py                  # 系统托盘 (pystray)
│   ├── input.py                 # Win32 系统事件模拟
│   └── control_panel.py         # 统一控制面板 (4 标签页)
│
├── training/                    # 训练子系统
│   ├── train_classifier.py      # MLP 训练管道
│   ├── export_onnx.py           # ONNX 导出
│   ├── model_test.py            # 模型测试 (混淆矩阵 + F1)
│   ├── guided_collector.py      # 引导式数据采集
│   ├── data_collector.py        # 核心采集器
│   └── _hagrid_common.py        # 标签定义 (单一事实来源)
│
├── models/                      # 模型文件
│   └── classifier.onnx          # 6 类 ONNX 分类器
│
└── logs/                        # 运行日志 (RotatingFileHandler)
```

**项目包含的独立脚本**：

| 脚本 | 用途 |
|------|------|
| `start_collector_single.bat` | 单指手势数据采集 |
| `start_train.bat` | 训练 MLP → 导出 ONNX |
| `start_test.bat` | ONNX + PyTorch 双模型测试 |


## 三、模块分工与技术选择

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                       摄像头 (640×480, 30fps)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  CameraCapture (线程)    │ deque(maxlen=2)
              │  OpenCV 采集 + 水平镜像   │ 帧缓冲
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  HandDetector (同线程)   │
              │  MediaPipe Hands 21点    │
              │  置信度过滤 + 最优手选择  │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                  ▼                  ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ RuleEngine  │  │ ONNX MLP     │  │ DtwMatcher   │
│ 几何归一化   │  │ 影子模式      │  │ 运动量分割    │
│ 6 类判定    │  │ conf≥90%接管  │  │ 每模板阈值    │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘
       │                │                  │
       └────────────────┼──────────────────┘
                        │
              ┌─────────▼──────────┐
              │  手势派发 (主循环)   │
              │  PALM→光标移动      │
              │  SINGLE→左键点击    │
              │  FIST→右键/DTW     │
              │  TWO→滚动          │
              │  PINCH→拖拽        │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  系统事件模拟       │
              │  Win32 API 原生调用 │
              │  光标/点击/滚动     │
              └────────────────────┘
```

### 3.2 各模块技术选型与理由

#### 3.2.1 手部追踪: MediaPipe Hands

| 对比项 | MediaPipe Hands | 传统方案 (OpenCV Haar/Depth) |
|--------|-----------------|------------------------------|
| 检测精度 | 21 点 3D 关键点 | 仅手部区域/少量特征点 |
| 光照鲁棒性 | 深度学习，适应性强 | 环境光依赖大 |
| 设备需求 | 普通 RGB 摄像头 | 需深度摄像头或严格光照 |
| 推理速度 | CPU 5-15ms | 不等 |
| 模型获取 | 自动下载 | 需要自行训练 |

**选择理由**: MediaPipe 提供 21 点手部关键点 (含手指关节)，精度足以支撑指尖距离归一化判定和 DTW 轨迹匹配。Tasks API 模型自动下载，部署零配置。

#### 3.2.2 手势分类: RuleEngine + ONNX 影子模式

| 层级 | 技术 | 特点 |
|------|------|------|
| 规则引擎 (主) | 几何判定, tip-MCP 距离归一化 | 零延迟, 可解释, 即时可用 |
| ONNX MLP (影子) | 70→128→128→64→6, Softmax | 机器学习, 泛化强, conf≥90% 接管 |
| 决策逻辑 | 规则先行, ONNX 高置信度覆盖 | 兼顾可靠性与智能化 |

**ONNX MLP 是什么**：

- **MLP (Multi-Layer Perceptron)**：多层感知器，4 层全连接网络 (70→128→128→64→6)，Softmax 输出 6 类概率。手部特征已是 70 维向量而非图像，不需要 CNN。6 类分类问题 MLP 足够，推理 < 1ms/帧。
- **ONNX (Open Neural Network Exchange)**：开放模型交换格式。PyTorch 训练出的 `.pt` 模型通过 `torch.onnx.export()` 导出为 `.onnx` 文件 (136KB)，`onnxruntime` 加载推理，无需 PyTorch 运行环境。安装体积从 2.5GB (PyTorch) 降到 ~30MB (onnxruntime)。

**选择理由**:
- **规则引擎**: 几何特征 (指尖距离/手部尺度) 天然具有旋转、平移不变性。归一化后 camera-distance invariant。零推理延迟，适合实时 30fps。
- **影子模式**: ONNX 和规则引擎并行运行，互不阻塞。ONNX 高置信度 (>90%) 时自动覆盖规则结果。低置信度时规则引擎兜底。既享受 ML 泛化能力，又防止模型误判。
- **归一化**: 所有指尖距离除以 `hand_scale` (wrist→middle_MCP)，消除手离摄像头远近的影响。

#### 3.2.3 空中点击: 积分相位检测

| 方案对比 | 加速度检测 (旧) | 积分相位检测 (当前) |
|----------|----------------|---------------------|
| 信号源 | z 轴加速度 | 食指弯曲比 ratio |
| 噪声 | z 轴抖动误触发 | 手指弯曲意图明确 |
| 手势门控 | 无 (每帧跑) | 仅 SINGLE_FINGER/FIST |
| 锁存防丢 | 无 | 武装 15 帧 |

**选择理由**: 弯曲比 `|tip-MCP| / |pip-MCP|` 是相对量，不受手距摄像头远近影响。双相积分 (BEND→REBOUND) 要求完整的手指弯曲-回弹周期，过滤了随机抖动。手势门控消除了 PALM_OPEN 光标移动中的误触。武装锁存解决了点击瞬间手势掉落的问题。

#### 3.2.4 自定义手势: DTW + 自适应阈值

| 对比项 | 传统 DTW | 本项目改进 |
|--------|---------|-----------|
| 分割方式 | 握拳协议 (FIST 起止) | 运动量自动分割 (静止→运动→静止) |
| 匹配阈值 | 单一全局阈值 | 每模板自适应阈值 (self_distance × 2.0) |
| 距离算法 | 欧氏距离 (需同长度序列) | fastdtw (支持不同速度/长度) |
| 注册流程 | 无引导 | 8/5s 倒计时 + 中文引导 + 静止确认 |

**选择理由**:
- **fastdtw**: 动态时间规整处理同一手势快慢不同版本，路径长度归一化消除长度偏差。
- **自适应阈值**: 注册时自动计算 3 次录制的"自然波动"(self_distance)，阈值 = self_distance × 2.0。每个手势有自己独立的匹配标准。
- **运动量分割**: 无需记忆"先握拳再松开"的协议。食指指尖速度 > 0.012 即运动中，连续 0.5s 静止确认为手势结束。

#### 3.2.5 光标控制: 1€ 滤波器

| 对比项 | 移动平均 | 1€ Filter |
|--------|---------|-----------|
| 低速抖动 | 需大窗口 | 自适应截止频率，近静态时强滤波 |
| 高速延迟 | 窗口越大越滞后 | 截止频率随速度上升，延迟恒定 |
| 参数 | 窗口大小 | beta (速度敏感度) + fcmin (最小截止频率) |

**选择理由**: 1€ Filter 在低速时有强滤波消除手颤，高速时自适应提高截止频率减少延迟。一阶低通实现简单、计算量低。

#### 3.2.6 线程模型

```
5 线程协作:

Main Thread    — 启动各线程 + 阻塞等待托盘退出 + 清理资源
Tk-UI Thread   — Tk.mainloop() (daemon), 所有 UI 操作通过 Queue+after 调度
Camera Thread  — 采集循环 (daemon), deque(maxlen=2) 帧缓冲
Pipeline Thread— 处理主循环 (daemon): 检测→分类→派发→事件
Tray Thread    — pystray 事件循环 (daemon), 菜单回调

跨线程通信:
  帧传递: deque(maxlen=2) — GIL 下 append/index 原子
  配置: threading.Lock 保护读写
  训练状态: threading.Event
  UI 调度: queue.Queue + after() 轮询 (解决 Tcl apartment 限制)
```

#### 3.2.7 系统事件: Win32 原生 API

弃用 pyautogui，改用 Win32 `SetCursorPos` + `mouse_event` + `keybd_event`。自行实现软钳制光标边界检测，避免 pyautogui FAILSAFE 异常。

#### 3.2.8 日志系统

五级日志层次：

| 级别 | 内容 | 触发方式 |
|------|------|---------|
| TRACE (5) | 逐帧: 指尖距离/捏合参数/光标感知/DTW震荡 | `--trace` |
| DEBUG (10) | Inference摘要/DTW匹配/点按状态 | `--debug` |
| INFO (20) | 启动/手势触发/模板注册 | 默认 |
| WARNING | 异常可恢复 | — |
| ERROR | 崩溃/致命错误 | — |

`RotatingFileHandler` (10MB × 5) + `StreamHandler` (WARNING+)


## 四、手势引擎详解

### 4.1 6 类静态手势

所有判定使用归一化后的指尖-MCP 距离 (除以 `hand_scale`):

```
归一化距离 = |tip - mcp| / |wrist - middle_mcp|
```

| 手势 | 判定规则 | 置信度 |
|------|---------|--------|
| SINGLE_FINGER | 食指>0.70, 食指>中指×1.3, 中指<0.60, 无名指<0.55, 小指<0.45, 拇指忽略 | 0.9 |
| FIST | 4+指<0.45, 且食指非优势指 (防点击弯曲误判) | 0.9 |
| PINCH | 拇指-食指归一化距离<0.35, 且食指非全伸 | 1.0 |
| TWO_FINGER | 食指+中指>0.45, 无名指+小指<0.45 | 0.9 |
| PALM_OPEN | 3+指>0.70, 或 fallback | 0.7/0.3 |

**判定优先级**: SINGLE_FINGER → FIST → PINCH → TWO_FINGER → PALM_OPEN → fallback

### 4.2 手势→事件映射

| 手势 | 光标 | 点击 | 功能 |
|------|------|------|------|
| PALM_OPEN | ✅ 移动 | — | 捏合=拖拽 |
| SINGLE_FINGER | ❌ 冻结 | 左键 (AirTap) | 空中点击 |
| FIST | ❌ 冻结 | 右键 (fist_tap) | DTW 触发 |
| TWO_FINGER | ❌ 冻结 | — | 垂直滚动 |
| PINCH | ✅ 移动 | — | 拖拽进行中 |


## 五、DTW 自定义手势

### 5.1 注册流程

```
1. 控制面板输入名称 + 选快捷键 → Register New
2. 第 1 次: 8s 倒计时 → "开始! 做手势 (做完保持静止)" → 录制
3. 第 2 次: 5s 倒计时 → 录制
4. 第 3 次: 5s 倒计时 → 录制
5. DBA 平均 (5 轮 DTW 对齐迭代) → 模板
6. 自动计算 self_distance → 匹配阈值 = self_distance × 2.0
7. 保存到 templates.json, UI 自动刷新
```

### 5.2 匹配流程

```
每帧: 食指指尖速度 > 0.012 → MOVING (缓冲)
      静止连续 15 帧 → STILL → 匹配所有模板
      DTW 距离 < 模板阈值 → 触发快捷键 → 冷却 90 帧
```

### 5.3 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| motion_threshold | 0.012 | 指尖 ~8px/帧 以上 = 运动中 |
| still_frames | 15 | 连续 0.5s 静止 = 手势结束 |
| max_record_frames | 180 | 最长 6s 录制 |
| match_threshold | 50.0 (全局) / 自适应 (每模板) | 匹配通过标准 |


## 六、模型训练

### 6.0 PyTorch — 只用于训练，不用于部署

PyTorch 是 Meta 开源的深度学习框架。本项目采用标准**训推分离**模式：

```
开发阶段 (PyTorch):
  定义网络 → 加载数据 → Adam 优化 → 60 epochs 训练
  → torch.save("best_model.pt")
  → torch.onnx.export("classifier.onnx")

运行阶段 (onnxruntime):
  加载 classifier.onnx → 推理 (无需 torch)
```

| | PyTorch 直接部署 | ONNX 部署 (本项目) |
|---|---|---|
| 安装体积 | ~2.5GB | ~30MB |
| 启动时间 | 3-5s | 即时 |
| 内存占用 | 500MB+ | ~50MB |
| 推理速度 | 快 (GPU) | 快 (CPU 优化) |
| 依赖 | torch + CUDA | 仅 onnxruntime |

### 6.1 网络架构

```
输入 (70维) → Linear(128) → ReLU → Dropout(0.3)
           → Linear(128) → ReLU → Dropout(0.3)
           → Linear(64) → ReLU → Dropout(0.3)
           → Linear(6) → Softmax
```

### 6.2 训练配置

| 项目 | 配置 |
|------|------|
| 训练样本 | 20,500 (FIST/NO_HAND/PALM_OPEN/PINCH/TWO_FINGER 各 3500, SINGLE_FINGER 3000) |
| 训练/验证 | 80/20 切分 |
| 优化器 | Adam (lr=1e-3) |
| 调度 | CosineAnnealing (60 epochs) |
| 验证准确率 | **100.0%** |
| 测试集准确率 | **99.98%** (8498/8500, 仅 2 个 PINCH→PALM 误判) |


## 七、系统可靠性设计

| 设计要素 | 实现 |
|----------|------|
| 摄像头断连恢复 | 无限重连 (移除 max_attempts), 指数退避 |
| 手势判定容错 | 武装锁存 15 帧 + FIST 护盾 (食指优势抑制) |
| DTW 训练异常 | try/finally 保证 event 清除, 超时/失败不卡死 |
| 配置并发安全 | threading.Lock + 原子写入 (mkstemp + os.replace) |
| UI 跨线程安全 | Queue + after() 轮询 (Tcl apartment 兼容) |
| 退出机制 | quit_app→tray.stop(), shutdown() 仅在 main finally 执行一次 |
| 日志持久化 | RotatingFileHandler (10MB × 5) |


## 八、未来展望

| 方向 | 说明 |
|------|------|
| 多摄像头/深度摄像头 | 利用深度信息提升 z 轴手势识别 |
| Transformer 分类器 | 替代 MLP，提升时序手势识别 |
| 连续手势识别 | 无需停顿的自然手势流识别 |
| Web 部署 | ONNX Web Runtime，浏览器内手势交互 |
| 移动端移植 | MediaPipe 原生支持 Android/iOS |

---

**项目仓库**: `https://github.com/wkaen0g/My_finger`
