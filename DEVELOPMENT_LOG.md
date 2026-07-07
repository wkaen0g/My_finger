# 微手势识别系统 — 开发日志

## 2026-05-12 — Phase 1 骨架搭建完成

---

## 1. 项目结构

```
microgesture/
├── __init__.py
├── main.py                  # 入口 + GesturePipeline 编排
├── config.py                # Config 类：JSON 加载 + watchdog 热重载
├── config.json              # 默认配置
├── pipeline/
│   ├── __init__.py
│   ├── capture.py           # 摄像头采集线程
│   ├── detector.py          # MediaPipe Hands 封装
│   ├── gesture_engine.py    # 规则引擎：5 类几何判定
│   ├── air_tap.py           # 食指 z 轴加速度点按检测
│   ├── pinch.py             # 捏合状态机
│   ├── cursor.py            # 1€ 滤波器 + 光标位移映射
│   └── scroll.py            # 双指 y 位移 → 垂直滚动
├── system/
│   ├── __init__.py
│   ├── input.py             # PyAutoGUI 事件模拟
│   └── tray.py              # pystray 托盘
├── models/                  # 预留给 ONNX
└── logs/
```

---

## 2. 线程模型

系统共运行 **3 个线程**：

```
┌─────────────────────────────────────────────────────────┐
│  Main Thread (main.py:main)                            │
│  - 启动 pipeline                                       │
│  - 启动托盘线程                                         │
│  - 阻塞等待托盘线程退出                                  │
│  - 处理 KeyboardInterrupt → shutdown                    │
└────────┬──────────────────────────────────┬─────────────┘
         │ start()                          │ start()
         ▼                                  ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│ Thread: camera-     │      │ Thread: tray                │
│ capture (daemon)    │      │ (daemon)                    │
│                     │      │                             │
│ 循环:               │      │ pystray.Icon.run()          │
│  read() → flip()   │      │ 事件循环阻塞                 │
│  → buffer.append() │      │ 菜单回调触发                  │
│                     │      │ pipeline.toggle_tracking()  │
│ deques(maxlen=2)    │      │ pipeline.set_sensitivity()  │
│                     │      │ pipeline.stop()             │
└────────┬────────────┘      └─────────────────────────────┘
         │ latest_frame()
         ▼
┌─────────────────────────────────────────────────────────┐
│ Thread: pipeline (daemon)                               │
│                                                         │
│ while _running:                                         │
│   _process_frame()                                      │
│   sleep(0.001s)  ← 非阻塞让出 CPU                       │
│                                                         │
│ _process_frame:                                         │
│   frame = capture.latest_frame()                        │
│   hand = detector.detect(frame)                         │
│   gesture = engine.classify(hand.landmarks)              │
│   dispatch(gesture) → cursor/tap/pinch/scroll/input     │
└─────────────────────────────────────────────────────────┘
```

### 线程间通信

| 方向 | 机制 | 说明 |
|------|------|------|
| camera-capture → pipeline | `deque(maxlen=2)` | 生产者-消费者，自动丢弃旧帧 |
| tray → pipeline | 直接方法调用 | `toggle_tracking()` / `set_sensitivity()` / `set_right_click()` |
| pipeline → tray | `tray.set_state()` | 更新托盘图标颜色 |
| 停止信号 | `threading.Event` | `capture._stop_event` / `pipeline._running` bool |

### 线程安全分析

- **`deque(maxlen=2)`**：Python `collections.deque` 的 `append` 和 `[-1]` 索引在 GIL 下是原子的，不需要额外加锁。`maxlen=2` 确保缓冲区不会无限增长。
- **`config.get()`**：使用 `threading.Lock` 保护读写，watchdog 回调在同一线程触发 reload。
- **`CursorController._frozen`**：仅在 pipeline 线程读写，无竞态。
- **`InputController._dragging`**：仅在 pipeline 线程读写。

---

## 3. 帧处理时序（单帧全生命周期）

```
时间轴 ──────────────────────────────────────────────────────────────►

CameraCapture._run()                   GesturePipeline._process_frame()
│                                      │
├─ cap.read() → BGR frame             │
├─ cv2.flip(frame, 1) 水平镜像        │
├─ buffer.append(frame)               │
│  (deque maxlen=2, 旧帧自动丢弃)      │
│                                      ├─ frame = capture.latest_frame()
│                                      │  (取 buffer[-1], 无锁)
│                                      │
│                                      ├─ detector.detect(frame)
│                                      │  BGR→RGB, flags.writeable=False
│                                      │  MediaPipe Hands.process()
│                                      │  → HandLandmarks(landmarks(21,3),
│                                      │       handedness, confidence)
│                                      │  若 confidence<0.5 → 返回 None
│                                      │  若多手 → 选最高置信度
│                                      │
│                                      ├─ hand is None?
│                                      │  YES → _handle_no_hand()
│                                      │    记录无手起始时间
│                                      │    超时(5s) → cursor.freeze(),
│                                      │    scroll.stop()
│                                      │    return  ← 跳过后续
│                                      │
│                                      ├─ engine.classify(landmarks)
│                                      │  计算5指尖-MCP距离
│                                      │  判定: PALM_OPEN / FIST /
│                                      │        TWO_FINGER / PINCH
│                                      │  → GestureResult(gesture, landmarks,
│                                      │       confidence)
│                                      │
│                                      ├─ 手势分发 ─────────────────┐
│                                      │                            │
│                                      │  PALM_OPEN:                │
│                                      │    cursor.unfreeze()        │
│                                      │    → cursor.update(x,y)     │
│                                      │      1€ filter滤波         │
│                                      │      位移×灵敏度×1920      │
│                                      │      → input.move(dx,dy)   │
│                                      │    → tap.update(landmarks)  │
│                                      │      z加速度脉冲检测       │
│                                      │      → input.click()       │
│                                      │    → pinch.update(landmarks)│
│                                      │      状态机 OPEN↔PINCHING  │
│                                      │      → input.drag_start/end│
│                                      │                            │
│                                      │  FIST:                     │
│                                      │    cursor.freeze()          │
│                                      │    → tap.update(landmarks)  │
│                                      │      (fist_tap模式→右键)   │
│                                      │                            │
│                                      │  TWO_FINGER:               │
│                                      │    cursor.freeze()          │
│                                      │    scroll.start()           │
│                                      │    → scroll.update()        │
│                                      │      双指中点y位移→滚动量  │
│                                      │      → input.scroll(delta)  │
│                                      │    (two_finger模式→右键)   │
│                                      │                            │
│                                      │  PINCH:                    │
│                                      │    cursor.unfreeze()        │
│                                      │    → cursor.update(x,y)     │
│                                      │    (捏合时不触发tap,        │
│                                      │     因为捏合=拖拽态)       │
│                                      │                            │
│                                      │  _prev_gesture ← gesture    │
│                                      └────────────────────────────┘
│
├─ 下一帧...
```

### 关键时序约束

| 约束 | 值 | 说明 |
|------|-----|------|
| 采集帧率 | 30fps | `_frame_interval = 1/30 ≈ 33.3ms` |
| frame_interval 遵守方式 | `sleep_time = interval - elapsed` | 仅在采集耗时 < 33ms 时 sleep |
| 缓冲区深度 | 2 帧 | 管道始终消费最新帧，最大延迟 1 帧 (~33ms) |
| 管道轮询间隔 | 1ms | `sleep(0.001)` 非阻塞让出 CPU |
| 推理耗时 (MediaPipe) | ~5-15ms (CPU) | model_complexity=1 下的典型值 |
| 空中点按冷却 | 8 帧 (~267ms) | 防止连续误触发 |
| 捏合去抖 | 3 帧 (~100ms) | 状态切换需连续3帧确认 |
| 无手休眠 | 5s | 超时后冻结光标 |
| 摄像头重连 | 指数退避 0.5→8s, 最多5次 | 总等待 ~15.5s |

---

## 4. 模块 API 文档

### 4.1 `config.py` — 配置管理

```
Config(path: Path | None = None)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `reload()` | `→ None` | 从 JSON 文件重新加载，异常时保留旧数据 |
| `get(*keys, default=None)` | `→ Any` | 线程安全嵌套取值：`config.get("cursor", "sensitivity")` |
| `__getitem__(key)` | `→ Any` | 字典式访问 |
| `_start_watch()` | `→ None` | 启动 watchdog 文件监听（内部调用） |
| `_stop_watch()` | `→ None` | 停止文件监听 |

**工厂函数**：`get_config(path=None) → Config` — 全局单例

---

### 4.2 `pipeline/capture.py` — 摄像头采集

```
CameraCapture(device_id=0, width=640, height=480, target_fps=30, buffer_size=2)
```

| 成员 | 类型 | 说明 |
|------|------|------|
| `is_connected` | `bool` (property) | 摄像头是否已打开 |
| `latest_frame()` | `→ np.ndarray \| None` | 取缓冲中最新的 BGR 帧 |
| `start()` | `→ None` | 启动采集线程（幂等） |
| `stop()` | `→ None` | 设置停止标志，join(3s)，释放摄像头 |

**内部行为**：
- 帧读取后自动水平翻转 (`cv2.flip(frame, 1)`)
- 摄像头断开时指数退避重连：0.5s → 1s → 2s → 4s → 8s，最多 5 次
- 5 次失败后采集线程退出，不再自动恢复

---

### 4.3 `pipeline/detector.py` — 手部检测

```
HandLandmarks  (dataclass)
  .landmarks: np.ndarray    # (21, 3) 归一化坐标 [0,1]
  .handedness: str           # "Left" | "Right"
  .confidence: float         # [0, 1]
  .valid: bool (property)    # confidence >= 0.5

HandDetector(model_complexity=1, min_detection_confidence=0.5,
             min_tracking_confidence=0.5, max_num_hands=1)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `detect(frame)` | `→ HandLandmarks \| None` | BGR 帧输入，返回最佳手或 None |
| `close()` | `→ None` | 释放 MediaPipe 资源 |

**MediaPipe 关键点索引**（21 点）：

```
 0: 手腕        1-4: 拇指(CMC→TIP)
 5-8: 食指(MCP→TIP)    9-12: 中指(MCP→TIP)
13-16: 无名指(MCP→TIP)  17-20: 小指(MCP→TIP)
```

`landmarks[i] = (x, y, z)` — x/y 归一化到 [0,1]，z 以手腕深度为原点。

---

### 4.4 `pipeline/gesture_engine.py` — 规则引擎

```python
Gesture      (Enum)
  NO_HAND      # MediaPipe 无检测
  PALM_OPEN    # 五指全伸展
  FIST         # 五指全屈曲
  TWO_FINGER   # 食指+中指伸展，其余屈曲
  PINCH        # 拇指尖与食指尖接近

GestureResult  (dataclass)
  .gesture: Gesture
  .landmarks: np.ndarray  # (21, 3)
  .confidence: float      # 规则引擎固定 1.0/0.5

RuleEngine(tip_mcp_open_threshold=0.25, tip_mcp_fist_threshold=0.12,
           pinch_threshold_ratio=0.35)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `classify(landmarks)` | `→ GestureResult` | 21点 → 手势分类 |

**判定优先级**（从高到低）：

```
1. open_count==5                    → PALM_OPEN  (conf=1.0)
2. closed_count==5                  → FIST       (conf=1.0)
3. idx_open & mid_open &            → TWO_FINGER (conf=1.0)
   ring_closed & pinky_closed
4. pinch_dist/hand_scale < ratio    → PINCH      (conf=1.0)
5. fallback                         → PALM_OPEN  (conf=0.5)
```

**已知局限**：捏合判定优先级最低，如果同时满足 TWO_FINGER 和 PINCH（例如双指伸出时拇指靠近食指），会先命中 TWO_FINGER。这是有意为之——双指伸出是一个更强的用户意图信号。

---

### 4.5 `pipeline/air_tap.py` — 空中点按

```
TapEvent  (dataclass)
  .timestamp: float

AirTapDetector(z_sensitivity=0.015, rebound_frames=5)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `update(landmarks)` | `→ TapEvent \| None` | 处理新关键点，检测到点按时返回事件 |

**检测原理**：

```
z_history: [z(t-3), z(t-2), z(t-1), z(t)]
              │        │        │        │
              ▼        ▼        ▼        ▼
velocity:   [v(t-2),  v(t-1),  v(t)  ]
              │        │        │
              ▼        ▼        ▼
accel:      [a(t-1),  a(t)  ]

触发条件: a(t-1) > +threshold  AND  a(t) < -threshold
          (正加速脉冲)              (紧接着负加速脉冲)
```

**参数**：
- `z_sensitivity=0.015`：加速度检测阈值，值越小越灵敏
- `rebound_frames=5`：保留参数，当前版本使用固定的 8 帧冷却 (`_tap_cooldown_frames=8`)
- 冷却期间不产生新的 TapEvent，防止单次点按被多次检测

---

### 4.6 `pipeline/pinch.py` — 捏合检测

```
PinchState  (Enum)
  OPEN        # 拇指和食指分离
  PINCHING    # 拇指和食指捏合

PinchEvent  (dataclass)
  .state: PinchState

PinchDetector(pinch_threshold_ratio=0.35)
```

| 成员 | 类型 | 说明 |
|------|------|------|
| `is_pinching` | `bool` (property) | 当前是否处于捏合状态 |
| `update(landmarks)` | `→ PinchEvent \| None` | 状态变化时返回事件，否则 None |

**状态机**：

```
         pinch_dist/hand_scale < 0.35 × 3帧
  OPEN ────────────────────────────────────► PINCHING
   ▲                                          │
   └──────────────────────────────────────────┘
         pinch_dist/hand_scale ≥ 0.35 × 3帧
```

**归一化**：`pinch_dist = ||thumb_tip - index_tip|| / ||wrist - middle_mcp||`，使阈值与手的大小、距离摄像头远近无关。

**去抖**：需要连续 3 帧确认状态切换，避免单帧噪声导致误触发。在 30fps 下折合约 100ms 延迟。

---

### 4.7 `pipeline/cursor.py` — 光标控制

#### OneEuroFilter

```
OneEuroFilter(beta=0.007, fcmin=1.0, min_cutoff=1.0, fps=30.0)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `filter(value)` | `→ float` | 对输入值进行自适应低通滤波 |
| `reset()` | `→ None` | 重置滤波器内部状态 |

**算法**（1€ Filter, Casiez et al. 2012）：

```
dx = (x - x_prev) × fps              # 原始导数
edx = α(min_cutoff)·dx + (1-α)·dx_prev    # 平滑导数
cutoff = fcmin + β × |edx|           # 自适应截止频率
x_hat = α(cutoff)·x + (1-α)·x_prev   # 平滑值

其中 α(cutoff) = 1 / (1 + τ/T_e)
     τ = 1/(2π·cutoff)
     T_e = 1/fps
```

- **低速时**（手静止）：`cutoff ≈ fcmin`，强滤波消除抖动
- **高速时**（快速移动）：`cutoff` 随速度增大，减少延迟

#### CursorController

```
CursorController(sensitivity=1.5, beta=0.007, fcmin=1.0,
                  min_cutoff=1.0, fps=30.0)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `update(x, y)` | `→ (dx, dy)` | 输入归一化坐标，输出像素级光标位移 |
| `freeze()` | `→ None` | 冻结光标，重置滤波器，后续 update 返回 (0,0) |
| `unfreeze()` | `→ None` | 解冻光标 |
| `is_frozen` | `bool` (property) | 当前是否冻结 |

**灵敏度计算**：`dx = Δsmooth_x × sensitivity × 1920`（y 同 ×1080）。即 sensitivity=1.5 时，手横跨整个画面宽度（归一化位移 1.0）对应光标横跨 2880 像素。

---

### 4.8 `pipeline/scroll.py` — 双指滚动

```
ScrollDetector(sensitivity=40.0, beta=0.007, fcmin=1.0,
               min_cutoff=1.0, fps=30.0)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `start()` | `→ None` | 激活滚动，重置滤波器 |
| `stop()` | `→ None` | 停用滚动 |
| `update(index_tip, middle_tip)` | `→ float` | 返回滚动增量（正=上滚） |
| `is_active` | `bool` (property) | 是否处于滚动激活态 |

**计算**：`delta = (prev_mid_y - current_mid_y) × sensitivity × 1080`

y 在图像坐标中向下增长，所以 `prev - current` 得到正滚（手指上移=页面上滚）。

---

### 4.9 `system/input.py` — 系统事件模拟

```
InputController()
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `move(dx, dy)` | `→ None` | 相对位移移动光标 |
| `click()` | `→ None` | 左键单击（拖拽中忽略） |
| `double_click()` | `→ None` | 左键双击 |
| `right_click()` | `→ None` | 右键单击 |
| `scroll(amount)` | `→ None` | 垂直滚动，正=上 |
| `drag_start()` | `→ None` | 按下鼠标左键 |
| `drag_end()` | `→ None` | 释放鼠标左键 |
| `is_dragging` | `bool` (property) | 是否处于拖拽状态 |

**安全特性**：`pyautogui.FAILSAFE = True` — 光标移动到屏幕四角 (0,0) 时抛出异常。

---

### 4.10 `system/tray.py` — 系统托盘

```
TrayState  (Enum)
  NORMAL      # 绿色图标：摄像头正常
  NO_CAMERA   # 橙色图标：摄像头不可用
  SLEEP       # 灰色图标：休眠

SystemTray(callbacks: dict)
```

| 方法 | 签名 | 说明 |
|------|------|------|
| `run()` | `→ None` | 启动 pystray 事件循环（阻塞） |
| `stop()` | `→ None` | 停止托盘 |
| `set_state(state)` | `→ None` | 切换图标颜色 |

**callbacks 键**：`toggle_tracking`, `set_sensitivity(value)`, `set_right_click(mode)`, `quit_app`

**菜单结构**：

```
Toggle Tracking ──────────────── 启停追踪
──────────
Sensitivity ─┬─ Low (0.5x)
             ├─ Medium (1.5x) ✓
             └─ High (3.0x)
Right Click Mode ─┬─ Fist + Tap ✓
                  └─ Two Finger Tap
──────────
Quit ────────────────────────── 退出
```

---

### 4.11 `main.py` — 入口与编排

```
setup_logging(config) → None
  初始化日志：RotatingFileHandler(5MB×3) + StreamHandler(WARNING+)

GesturePipeline(config)
  .start()              → None   启动采集线程 + 管道处理线程
  .stop()               → None   停止采集 + join 管道线程 + 释放 detector
  .toggle_tracking()    → None   启停追踪（管道循环仍运行，但不处理帧）
  .set_sensitivity(v)   → None   运行时调整光标灵敏度
  .set_right_click(m)   → None   切换右键触发模式

main() → None
  1. get_config() → 加载配置 + 启动 watchdog
  2. setup_logging() → 日志就绪
  3. GesturePipeline.start() → 采集 + 处理启动
  4. SystemTray.run() → 托盘启动（独立线程）
  5. 阻塞等待托盘退出
  6. shutdown() → 清理资源

shutdown(pipeline, tray) → None
  1. pipeline.stop()
  2. tray.stop()
```

---

## 5. 手势→事件映射（完整）

| 当前手势 | 上一帧手势 | 光标 | 点按 | 捏合 | 滚动 | 说明 |
|----------|-----------|------|------|------|------|------|
| PALM_OPEN | * | 移动 | 左键单击 | 拖拽启停 | — | 默认操作模式 |
| FIST | * | 冻结 | 右键(fist_tap) | — | — | 冻结+右键模式 |
| TWO_FINGER | 非TWO_FINGER | 冻结 | 右键(two_finger) | — | 启动滚动 | 切换进滚动模式 |
| TWO_FINGER | TWO_FINGER | 冻结 | 右键(two_finger) | — | 持续滚动 | 滚动保持 |
| PINCH | * | 移动 | — | — | — | 拖拽中手势(捏合=放开前按住) |
| NO_HAND | * | 超时冻结 | — | — | 停止 | 5s后休眠 |

注：PINCH 手势本身不触发拖拽——拖拽由 `PinchDetector` 状态机在 PALM_OPEN 模式下独立检测。

---

## 6. 已知问题与风险清单

### 6.1 阻塞性风险

| ID | 风险 | 严重度 | 可能性 | 缓解措施 |
|----|------|--------|--------|---------|
| R01 | MediaPipe 首次加载模型耗时 3-10s，期间应用无响应 | 中 | 确定 | `main.py` 可加启动画面；模型在 `HandDetector.__init__` 中懒加载 |
| R02 | pystray 在 Windows 上需要 `.ico` 格式图标，Pillow RGBA Image 可能不被接受 | 高 | 中 | 需在 Windows 上实测；必要时预生成 `.ico` 文件 |
| R03 | PyAutoGUI `moveRel` 在高 DPI 显示器上可能位移量不准 | 中 | 中 | 需在 150%/200% 缩放下测试 |
| R04 | 无虚拟环境，全局 `pip install` 可能污染系统 Python | 低 | 低 | 建议创建 venv |
| R05 | OpenCV `cv2.VideoCapture` 在部分笔记本上 device_id=0 可能不是内置摄像头 | 中 | 低 | 配置中可改 `device_id`，未来可加自动检测 |

### 6.2 功能缺陷

| ID | 缺陷 | 影响 | 修复方向 |
|----|------|------|---------|
| B01 | 规则引擎捏合判定优先级最低，TWO_FINGER + 拇指靠近食指时会误判为 TWO_FINGER | 用户做捏合手势但伸出两指时行为不符预期 | 调整判定优先级，或在 PINCH 判定中加入对食指中指状态的额外检查 |
| B02 | 空中点按使用固定 8 帧冷却，与配置的 `rebound_frames` 不一致 | `rebound_frames` 配置项不生效 | 将 `_tap_cooldown_frames` 改为使用 `rebound_frames` 参数 |
| B03 | `CursorController.freeze()` 重置滤波器历史，解冻后第一帧会产生跳变 | 握拳后张开手时光标可能瞬跳 | freeze 时记录最后位置，unfreeze 时用该位置初始化，而非重置 filter |
| B04 | 滚动仅支持垂直方向 | 无法水平滚动 | Phase 4 扩展 |
| B05 | 仅支持单手，多人场景选最高置信度手 | 多人场景行为不确定 | 配置中可指定左右手偏好，但未暴露到托盘菜单 |
| B06 | 摄像头重连成功后采集线程重置 `attempts=0`，但管道线程可能在重连期间积压无帧或旧帧 | 重连后短暂表现异常 | 重连成功时清空 buffer |
| B07 | `setup_logging` 配置了 `RotatingFileHandler` 但 `StreamHandler` 只输出 WARNING+，INFO/DEBUG 仅在文件中可见 | 终端调试不便 | 可通过 `config.json` 中 `logging.level` 调整 |

### 6.3 性能边界

| ID | 边界 | 预期行为 | 未知点 |
|----|------|---------|--------|
| P01 | MediaPipe 推理耗时 > 33ms (30fps budget) | 管道自然降帧率，`latest_frame()` 跳到最新帧 | 具体降帧程度取决于硬件 |
| P02 | PyAutoGUI 事件模拟耗时（Windows API 调用） | 如果单次 >1ms，累积影响帧率 | 未测量 |
| P03 | 1€ 滤波器参数 (beta=0.007) 针对 30fps 设计 | 降帧时滤波质量下降 | 可加实时 fps 测量动态调整 Te |

### 6.4 跨平台兼容性

| 平台 | 问题 | 状态 |
|------|------|------|
| Windows 11 | pystray 托盘图标格式 | 待测 |
| macOS | PyAutoGUI 需要辅助功能权限 | 待测 |
| Linux (X11) | PyAutoGUI 依赖 `python3-xlib` | 待测 |
| Linux (Wayland) | PyAutoGUI 不支持 Wayland | 需改用 `ydotool` 或 `uinput` |

### 6.5 配置阈值未校准

所有默认阈值（`tip_mcp_open_threshold=0.25`, `pinch_threshold_ratio=0.35` 等）基于 MediaPipe 归一化坐标的理论估计，尚未经过实际摄像头测试校准。不同摄像头、不同用户手型的表现可能有显著差异。

---

## 7. 下一步

### 立即
1. ~~创建虚拟环境，安装依赖清单~~ ✅ 2026-05-12
2. `python -m microgesture.main` 启动验证（需桌面环境 + 摄像头）
3. 按计划 Phase 1 验证清单逐项测试 9 条

---

## 2026-05-12 — 依赖安装与导入验证

### 环境
- Python 3.13.5 (`C:\Program Files\Python313\`)
- venv: `d:\Myfinger\.venv\`
- 安装方式: `pip install` (PyPI)

### 已安装包（30个）

| 包 | 版本 | 用途 |
|----|------|------|
| opencv-python | 4.13.0.92 | 摄像头采集 |
| opencv-contrib-python | 4.13.0.92 | mediapipe 依赖 |
| mediapipe | 0.10.35 | 手部关键点检测 |
| numpy | 2.4.4 | 数值计算 |
| pyautogui | 0.9.54 | 系统事件模拟 |
| pystray | 0.19.5 | 系统托盘 |
| pillow | 12.2.0 | 托盘图标绘制 |
| watchdog | 6.0.0 | 配置文件热重载 |
| matplotlib | 3.10.9 | mediapipe 依赖 |
| sounddevice | 0.5.5 | mediapipe 依赖 |

### 导入验证

全部 10 个模块导入成功 (`All imports successful!`)：
`config`, `capture`, `detector`, `gesture_engine`, `air_tap`, `pinch`, `cursor`, `scroll`, `input`, `tray`

### 已修复

| 日期 | 问题 | 修复 |
|------|------|------|
| 05-12 | `mediapipe` 无 `solutions` 属性 | 重写 `detector.py` 使用 Tasks API + 自动下载模型 |
| 05-12 | watchdog `_ReloadHandler` 闭包 bug | `config_self = self` 捕获外层引用 |
| 05-12 | 启动时托盘黄灯不恢复 | 加 `_on_status_change` 回调，启动后等最多 3s 检测摄像头 |

### 待桌面环境验证
- [x] `python -m microgesture.main` 启动
- [x] 模型自动下载
- [x] 摄像头自动打开
- [x] 系统托盘绿色图标
- [x] 托盘启停追踪
- [ ] 手势识别→系统事件全链路（待手势参数调优）
- [ ] 9 条 Phase 1 验证清单逐项通过

### Phase 1 验证清单

- [ ] 系统托盘出现，绿色图标
- [ ] 手掌张开 + 移动食指 → 光标跟随移动
- [ ] 拇食指捏合 → 进入拖拽模式
- [ ] 食指空中点按 → 触发单击
- [ ] 握拳 → 光标停止移动
- [ ] 握拳态下点按 → 右键菜单 (fist_tap 模式)
- [ ] 双指伸出上下移动 → 页面滚动
- [ ] 托盘切换灵敏度 → 光标响应变化
- [ ] 托盘切换右键模式 → 行为变更

### 短期 (Phase 2 预备)
- ~~采集训练数据工具~~ ✅ 2026-05-12
- ~~MLP 分类器训练管道~~ ✅ 2026-05-12
- ~~ONNX 导出与推理封装~~ ✅ 2026-05-12
- [x] 引导模式数据采集 ✅ 2026-07-04
- [x] HaGRID 预训练数据加载 ✅ 2026-07-05
- [x] 20BN-Jester V1 伪标签预训练 ✅ 2026-07-06
- [ ] 合成数据扩增

---

## 2026-05-12 — Phase 2 分类器骨架搭建完成

### 新增文件

```
microgesture/
├── recognition/
│   ├── __init__.py
│   ├── base.py                    # GestureRecognizer 抽象基类
│   └── static_classifier.py       # RuleEngine 适配器
├── pipeline/
│   └── classifier.py              # ONNX Runtime 推理
├── training/
│   ├── __init__.py
│   ├── data_collector.py          # 自由模式数据采集
│   ├── train_classifier.py         # MLP 训练管道 (PyTorch)
│   └── export_onnx.py             # ONNX 导出
```

### 架构设计

**GestureRecognizer 抽象基类** (`recognition/base.py`)
- `predict(landmarks: 21×3) → RecognitionResult(label, confidence, features)`
- `extract_features(landmarks) → 70-dim vector`
  - 63 dim: raw landmarks 展平
  - 7 dim: 5 指尖-MCP距离 + 1 捏合距离 + 1 食指弯曲比

**StaticClassifier** (`recognition/static_classifier.py`)
- 包装 Phase 1 RuleEngine，输出统一 `RecognitionResult`

**ONNXClassifier** (`pipeline/classifier.py`)
- ONNX Runtime 推理，输入 70-dim，输出 5 类 softmax
- 标签顺序: FIST, NO_HAND, PALM_OPEN, PINCH, TWO_FINGER

**训练管道** (`training/train_classifier.py`)
- 3 层 MLP: 70→128→128→64→5, ReLU + Dropout(0.3)
- CosineAnnealing 学习率调度
- 8:2 训练/验证切分，取最佳 acc 保存

**数据采集** (`training/data_collector.py`)
- 自由模式: `start(label)` → 逐帧 `record(landmarks)` → `stop()`
- 按标签分组保存为 `features_<label>.npz` + `metadata.json`

### 影子模式 (main.py)

```
每帧:
  rule_result = StaticClassifier.predict(landmarks)
  gesture = Gesture[rule_result.label]

  若 ONNX 模型存在:
    onnx_result = ONNXClassifier.predict(landmarks)
    若 onnx_result.confidence >= 0.90:
      gesture = Gesture[onnx_result.label]   # 自动切换
    否则:
      gesture = Gesture[rule_result.label]   # 沿用规则
```

- 分类器置信度 ≥ 90% 自动覆盖规则引擎
- 低于阈值时记录 `Shadow defer` 日志，用于分析差异
- ONNX 模型不存在时降级为纯规则模式，行为与 Phase 1 一致

### 新增依赖

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.11.0 | MLP 训练 |
| onnxruntime | 1.26.0 | ONNX 推理 |

### 新增配置项

| 键 | 默认值 | 说明 |
|----|--------|------|
| `system.shadow_confidence_threshold` | 0.90 | 分类器接管阈值 |

### 下一步

1. 运行数据采集工具录制各手势样本
2. 训练 MLP 分类器 → 导出 ONNX
3. 影子模式自动化切换验证
4. (后续) HaGRID 预训练数据加载 + 合成扩增

---

## 2026-07-05 — Phase 2 训练数据管道完善

### HaGRID 预训练数据加载

修复了已有的 HaGRID loader 雏形，统一了架构：

**新建 `_hagrid_common.py` — 单一事实来源**
- `CLASS_MAP` — HaGRID 类名 → 规范手势标签
- `GESTURE_LABELS` — 有序元组: FIST=0, NO_HAND=1, PALM_OPEN=2, PINCH=3, TWO_FINGER=4
  - `train_classifier.py`, `classifier.py`, `export_onnx.py` 全部导入此元组
  - 消除了 3 处硬编码重复
- `get_output_dir()` / `get_raw_dir()` — 基于 `__file__` 的绝对路径，不再依赖 CWD
- `save_features_by_label()` — 统一 .npz 保存逻辑

**修复 `hagrid_download.py`**
- 🔴 `import cv2` 从 `__main__` 守卫移到模块顶部（修复模块导入崩溃）
- 标签映射改用 `_hagrid_common` 导入

**修复 `hagrid_loader.py`**
- 路径改为包相对路径 + tqdm 进度条 + argparse CLI

**删除 `download.py`** — 损坏文件 (内容为 "404: Not Found")

**新建 `hagrid.py` — 统一 CLI**
```bash
python -m microgesture.training.hagrid download   # HuggingFace 下载
python -m microgesture.training.hagrid process    # 本地图像处理
python -m microgesture.training.hagrid jester     # Jester 伪标签
```

### 标签一致性修复

| 文件 | 修改 |
|------|------|
| `train_classifier.py` | `_GESTURES` → `from ._hagrid_common import GESTURE_LABELS` |
| `classifier.py` | `_LABELS` → `from ..training._hagrid_common import GESTURE_LABELS` |

---

## 2026-07-06 — 20BN-Jester V1 伪标签数据加载

### 背景

改用 20BN-Jester V1 (替代 HaGRID) 作为预训练数据集。Jester 有 148K 视频 (每个 ~36 帧 JPG)，标签为 27 类动态手势，无法直接映射到我们的 5 类静态手形分类。

### 策略

**伪标签法**: 规则引擎 (RuleEngine) 对每帧自动标注：

```
Jester 帧 → MediaPipe Hands → RuleEngine.classify() → 伪标签
  → 滤除低置信度 (conf < 0.5)
  → extract_features() → 70 维特征
  → 按标签分组 → data_jester/features_{LABEL}.npz
```

### 新建 `jester_loader.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `frames_per_video` | 3 | 每视频采样帧数 |
| `max_per_class` | 5000 | 每类上限，满后提前终止 |
| `pseudo_confidence_min` | 0.5 | 伪标签最低置信度 |

### 实测 (356 视频, 1068 帧)

| 手势 | 样本数 |
|------|--------|
| FIST | 50 |
| PALM_OPEN | 50 |
| PINCH | 50 |
| TWO_FINGER | 50 |

- 处理速度: ~17.5 vid/s; 预计 34 min 可收集 20K 样本 (5000×4)
- 输出与 `train_classifier.load_data()` 完全兼容

### 下一步

~~1. Jester 预训练数据~~ ✅ 2026-07-06
~~2. 两阶段训练~~ ✅ 2026-07-06
~~3. Phase 3: DTW 匹配器~~ ✅ 2026-07-06
4. 真实摄像头下对比验证
5. Phase 4: 精修增强

---

## 2026-07-06 — Phase 3 DTW 自定义手势匹配器完成

### 新建文件

```
microgesture/recognition/
├── dtw_matcher.py        # DTW 序列匹配器
└── dtw_trainer.py        # 手势模板训练器
```

### DTW 匹配器 (`dtw_matcher.py`)

**状态机**: `IDLE → ARMING → RECORDING → MATCHING`
- IDLE: 检测到 FIST → ARMING
- ARMING: 连续握拳 arm_frames 帧 → RECORDING
- RECORDING: 缓冲关键点序列 → 再次握拳 → 计算 DTW 匹配
- arm_frames=15 (~500ms)，防止短暂握拳误触发

**核心功能**:
- FIST 分隔协议: 握拳(开始) → 动作序列 → 握拳(结束)
- fastdtw 近似匹配 (radius=10)，路径长度归一化
- 平移不变: 每帧减去手腕坐标
- 模板加载/持久化到 `templates.json`

### DTW 训练器 (`dtw_trainer.py`)

- 3 次录制 → DBA (DTW Barycenter Averaging) 平均
- DBA 算法: 5 轮迭代，DTW 对齐 → 逐帧平均
- 返回 `TrainerResult` 供持久化

### 关联修改

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 `set()`, `save()`, `as_dict()` — 运行时持久化 |
| `config.json` | 新增 `dtw` 配置段 |
| `system/input.py` | 新增 `key_combo()` — Win32 键盘快捷键模拟 (50+ 键码) |
| `system/tray.py` | 新增 "Register Gesture..." 菜单项 |
| `main.py` | DTW 并行路径 + 训练模式 + 预览叠加 |

### 测试结果

- 状态机转换: PASS (IDLE↔ARMING↔RECORDING)
- 圆形手势模板匹配: dist=0.221, conf=100%
- 3 次录制 + DBA 平均: PASS

---

## 2026-07-06 — Phase 2/3 精修 (手势优先级 & 日志 & 训练增强)

### 规则引擎: FIST vs PINCH 优先级修复

**问题**: 握拳时拇指卷曲贴向食指，pinch_norm < 0.35，FIST 被误判为 PINCH

**修复** (`gesture_engine.py`):
1. FIST 提到最先检查（5指全屈曲是最强信号）
2. PINCH 加额外条件: index 不能是伸展状态（防 TWO_FINGER 误判）

新判定顺序: `FIST → PINCH → TWO_FINGER → PALM_OPEN → fallback`

### 日志系统优化

**问题**: DEBUG 级别每5帧输出6行 → ~30行/秒 → 5MB 日志一天就满

**修复**:
| 模块 | 之前 | 之后 |
|------|------|------|
| gesture_engine 指尖判定 | 每 5 帧 | 每 300 帧 |
| air_tap phase 转换 | 每次 | 每 300 帧 |
| pinch 捏合参数 | 每 5 帧 | 每 300 帧 |
| cursor 光标位移 | 每 5 帧 | 每 300 帧 |
| ONNX shadow 推理 | 逐帧 log | 统一为 150帧周期摘要 |

新增 `--debug` CLI 参数 + 模型加载日志。

### 推理日志修复

**Bug**: `source=onnx` 仅在 ONNX 与规则引擎不一致时才记录 → ONNX 覆盖率误报为 14%

**修复**: ONNX 置信度 ≥ 阈值时始终标记 source=onnx，不一致时额外标记 rule_said=XXX。实际 ONNX 覆盖率 ~73-91%。

### 训练增强

- `guided_collector.py`: 新增 `--frames` 参数，可调每类帧数 (默认 500)
- `train_classifier.py`: 新增 `evaluate()` 独立测试集评估 + `test_dir` 参数
- 新增 `start_collector_test.bat` — 测试集采集脚本

### 两阶段训练结果 (最新)

| 数据集 | 准确率 |
|--------|--------|
| 测试集 (data_test) | **100.0%** |
| 自采训练集 (data) | 97.9% |
| Jester 预训练 (data_jester) | 78.1% |

### 双指滚动灵敏度

`sensitivity: 40.0 → 8.0 → 3.0 → 2.0` + 新增 deadzone=0.03

### 下一步

1. 真实摄像头下对比 ONNX vs 规则引擎
2. Phase 4: 诊断面板 + 设置 GUI
3. 多显示器支持
