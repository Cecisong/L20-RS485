# SOP 工站视频步骤分割设计说明

**日期**: 2026-11-22  
**状态**: 修订稿 v4（X-AnyLabeling 桥接 + sop.json 范例；fewshot 标准一轮 / 长视频可漏步）  
**阶段目标**: 步骤时间轴分割准确度（循环统计 / P50-P95 / NVA 报表接口预留，本期不优化）

---

## 1. 背景与目标

制造业量产前爬坡阶段，精益工程师需要对工站 SOP 逐步掐表，用于线平衡。工站有固定俯视 AI Camera，画面聚焦桌面手部操作。

**本期要解决的问题**: 给定工站视频 + SOP 文本 + 该工站 1 轮已标注 SOP 区间，自动将视频时间轴切成有序步骤与 idle，并尽量准确。

**非目标（本期）**

- 手/工具检测等需要大批量图片标注的结构化方案
- 针对单条测试视频的手工规则、ROI、颜色阈值
- 以报表美观度为优化目标
- 公司视觉大模型密集调用

---

## 2. 输入 / 输出契约

### 2.1 输入角色（三件套）

| 角色 | 文件 | 谁维护 | 说明 |
|------|------|--------|------|
| 工序定义 | **`sop.json`** | 精益/工艺 | 步骤顺序、短 label、**长文本描述**；与具体视频无关 |
| 时间轴标注 | **X-AnyLabeling sidecar**（`视频同名.json`） | 标注员 | 只标 `label` + 起止时间；**不在此写每步长描述** |
| 算法配置 | **`station_config.yaml`** | 算法 | fps、chunk、Viterbi、VLM 等超参 |

标注工具：**不自研**，使用 X-AnyLabeling Video Classifier。流水线用 importer 桥接到内部 `Interval`。

### 2.2 业务约束（fewshot vs 长视频）

| | Fewshot（建原型用的标准一轮） | 长视频（推理 / 评测） |
|--|------------------------------|----------------------|
| 步骤是否齐全 | **必须**覆盖 `sop.json` 中全部 `step_id` 各至少一次 | 允许漏步 |
| 步骤顺序 | **必须**按 `1→2→…→N` 递增出现（可中间插 idle） | 默认仍倾向有序；**允许向前跳步**（`allow_skip_forward: true`） |
| 中间 `empty`/idle | **允许**，例如 `empty→step1→step2→empty→step3→empty` | 允许 |
| 回退/返工 `N→…→更早步` | fewshot **不允许** | 仅当 `allow_rework: true`（默认 false） |
| segment `description` | **忽略**；文案以 `sop.json` 的 `text` 为准 | 同左 |

### 2.3 `sop.json` 范例（工序定义，权威文本来源）

路径建议：`stations/<station_id>/sop.json`。

```json
{
  "station_id": "ws_demo",
  "sop_name": "组装工位-示例SOP",
  "version": "1.0",
  "steps": [
    {
      "step_id": 1,
      "label": "step1",
      "text": "从料盘取零件A，放入治具并压实"
    },
    {
      "step_id": 2,
      "label": "step2",
      "text": "锁付两颗 M3 螺丝，力矩达标"
    },
    {
      "step_id": 3,
      "label": "step3",
      "text": "扫码绑定，将成品放入流出料道"
    }
  ],
  "idle_labels": ["empty"],
  "notes": "label 必须与 X-AnyLabeling 中创建的类别名一致；text 供人读、可选图文相似与 VLM，不写入标注片段 description"
}
```

字段约定：

- `step_id`：从 1 连续到 N，决定 Viterbi 合法顺序  
- `label`：标注工具里的短名（`step1` / `step_1` 均可，importer 按 `sop.json` 映射，不靠猜）  
- `text`：**每步唯一权威描述**；不要依赖 X-AnyLabeling 的 `segments[].description`  
- `idle_labels`：映射为内部 `idle`（常见 `empty`）

### 2.4 X-AnyLabeling sidecar 范例（时间轴标注）

与视频同目录、同主文件名，例如 `demo.mp4` → `demo.json`。格式遵循 X-AnyLabeling Video Classifier；下列为 **合格 fewshot** 示意（步骤齐全、顺序正确，中间可有 empty）：

```json
{
  "version": "1.0.0",
  "type": "video_classification",
  "video": "demo.mp4",
  "fps": 30.0,
  "duration_ms": 16233,
  "width": 720,
  "height": 1280,
  "labels": ["empty", "step1", "step2", "step3"],
  "label_colors": {
    "empty": "#aaffff",
    "step1": "#ffaaaa",
    "step2": "#aaffaa",
    "step3": "#ffffaa"
  },
  "segments": [
    {
      "id": "s_idle_0",
      "label": "empty",
      "start_ms": 0,
      "end_ms": 1100,
      "start_frame": 0,
      "end_frame": 33,
      "description": ""
    },
    {
      "id": "s_step1",
      "label": "step1",
      "start_ms": 1100,
      "end_ms": 2367,
      "start_frame": 33,
      "end_frame": 71,
      "description": ""
    },
    {
      "id": "s_step2",
      "label": "step2",
      "start_ms": 2367,
      "end_ms": 5467,
      "start_frame": 71,
      "end_frame": 164,
      "description": ""
    },
    {
      "id": "s_idle_1",
      "label": "empty",
      "start_ms": 5467,
      "end_ms": 6200,
      "start_frame": 164,
      "end_frame": 186,
      "description": ""
    },
    {
      "id": "s_step3",
      "label": "step3",
      "start_ms": 6200,
      "end_ms": 11367,
      "start_frame": 186,
      "end_frame": 341,
      "description": ""
    },
    {
      "id": "s_idle_2",
      "label": "empty",
      "start_ms": 11367,
      "end_ms": 16233,
      "start_frame": 341,
      "end_frame": 487,
      "description": ""
    }
  ]
}
```

`description` 可空；流水线 **不读取** 它作为 SOP 文本。

### 2.5 桥接规则（X-AnyLabeling → 内部）

```text
sop.json.steps[].label  +  idle_labels
        │
        ▼
sidecar.segments[].label  ──map──►  step_id | idle
start_ms/end_ms           ──/1000──►  t_start_sec / t_end_sec
segments[].description    ──ignore──►  (丢弃)
segments[].id             ──ignore──►  (仅标注工具)
```

内部结构（内存 / 可选缓存，**不是**标注员手写主格式）：

```json
{
  "source": "xanylabeling",
  "video": "demo.mp4",
  "segments": [
    {"label": "idle", "step_id": null, "t_start_sec": 0.0, "t_end_sec": 1.1},
    {"label": "step_1", "step_id": 1, "t_start_sec": 1.1, "t_end_sec": 2.367},
    {"label": "step_2", "step_id": 2, "t_start_sec": 2.367, "t_end_sec": 5.467},
    {"label": "idle", "step_id": null, "t_start_sec": 5.467, "t_end_sec": 6.2},
    {"label": "step_3", "step_id": 3, "t_start_sec": 6.2, "t_end_sec": 11.367},
    {"label": "idle", "step_id": null, "t_start_sec": 11.367, "t_end_sec": 16.233}
  ]
}
```

**Fewshot 校验**（导入后强制）：

1. 每个 `step_id ∈ 1..N` 至少出现一次  
2. 仅保留 step 段时，按时间排序的 `step_id` 序列必须是严格递增的 `1,2,…,N`（允许中间有 idle 段）  
3. 缺步、跳步、回退 → `ValueError`，拒绝建原型  

**长视频 GT / 推理**：不做「必须齐全」校验；解码侧用 `allow_skip_forward`。

### 2.6 其它输入

| 字段 | 格式 | 说明 |
|------|------|------|
| `video` | mp4 等 | 固定机位；分钟～小时 |
| `station_config.yaml` | YAML | 见 §6 |

### 2.7 主输出（本期验收）

```json
{
  "segments": [
    {"t_start": 12.0, "t_end": 18.5, "label": "step_3", "confidence": 0.82},
    {"t_start": 18.5, "t_end": 21.0, "label": "idle", "confidence": 0.71}
  ],
  "sample_fps": 2.0,
  "encoder": "dinov2_vitb14+hoof_farneback",
  "aligner": "viterbi_ordered_hmm"
}
```

`label ∈ {"step_1".. "step_N", "idle"}`。

### 2.8 衍生输出（接口预留，本期不算分）

由同一套 `segments` 聚合即可，不另建模型：

- SOP 循环次数
- 每步工时均值 / P50 / P95
- NVA：步骤间等待、长静止、未匹配动作占比

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Offline Pipeline                          │
│                                                                  │
│  Video chunk ──► FG/motion gate ──┬──► Appearance (DINOv2)       │
│                                   └──► Optical-flow descriptor   │
│                                              │                   │
│  Fewshot 1-cycle ──► dual-stream prototypes (app + flow)         │
│                                              │                   │
│                         Similarity curves (per step)             │
│                                              │                   │
│              Ordered HMM + Viterbi  ←── 唯一决定「第几步」的模块   │
│                         (强制 1→2→…→N，可插 idle)                 │
│                                              │                   │
│                         segments (+ optional sparse VLM)         │
└──────────────────────────────────────────────────────────────────┘
```

**原则**

1. **外观不够用时必须有运动流**：固定机位、帧间外观接近时，DINOv2 单流不是完整方案（见 4.1 / 4.4）。
2. **步骤顺序不靠相似度猜**：相似度只提供「像哪一步」的发射分；**第几步由有序 Viterbi 转移图强制**（见 4.5，这是分割主引擎，不是后处理）。
3. 主路径全本地；公司 VLM 仅稀疏复核。
4. 长视频流式；工站切换只换 fewshot sidecar + `sop.json` + config。  
5. Fewshot 必须是标准一轮（可插 idle、不漏步）；长视频推理允许漏步（`allow_skip_forward`）。

### 3.1 特征是「每帧独立」还是「带序列」？（易混点）

**答案：两层分开——局部观测近似独立，全局顺序在序列模型里。**

| 层 | 是否看时间序列 | 具体做法 |
|----|----------------|----------|
| **外观编码（DINOv2）** | **基本每采样帧独立** | 每个 `t` 单独前向；不把一整段视频送进 Video Transformer |
| **短时成对** | 只看 `(t-1, t)` | 光流 / HOOF、`Δa_t=a_t-a_{t-1}`；不是长窗口序列特征 |
| **步骤顺序 / 分割** | **整段序列** | 所有采样点的发射分排成时间序列，用 **有序 HMM+Viterbi** 一次解码 |

因此：

- 不是「纯每帧 argmax、完全无序」；
- 也不是本期就上的「clip 级序列图像特征」（InternVideo2 等仍是预留升级）；
- **顺序性来自对齐层，不是来自 DINO 内部。**

若评测显示「局部很像但步态/手法要看更长上下文」再开短窗 clip 编码；在那之前不要默认上重视频模型。

---

## 4. 具体模型与算法选型

### 4.1 双流特征：外观（DINOv2）+ 光流 —— 不只用 DINO

**结论先说：只用 DINOv2 整帧 CLS，在本场景不够稳。**  
固定俯视下桌面/夹具占画面大部分，相邻步骤手部姿态往往相近，CLS 会被背景主导，帧间余弦相似度曲线会「糊成一片」。因此本期主特征是 **双流拼接**，不是单 DINO。

#### 4.1.1 外观流（Appearance）

| 项 | 选型 | 说明 |
|----|------|------|
| 模型 | **DINOv2 ViT-B/14** | 本地、无配额；做外观原型 |
| 输入 | 见 **4.1.3 背景/前景处理** 后的帧（不是生吞原图均值） | |
| 特征 | **不用裸 CLS 当唯一表示**。取最后一层 **patch tokens**（14×14=196），用运动/前景权重做加权池化，再 L2 归一化 → `a_t ∈ R^{768}` | 降低背景占比 |
| 时序差分（默认开） | 额外算 `Δa_t = normalize(a_t - a_{t-1})`，与 `a_t` 拼接或作为第二外观通道进原型 | 放大「变了什么」 |

#### 4.1.2 光流流（Optical Flow）—— 一等公民，不是边角料

上一版只用帧差标量做 idle/变点，**丢掉了方向与空间分布**，在外观近似时这是错误的偷懒。修订如下：

| 项 | 选型 |
|----|------|
| 算法 | OpenCV **Farneback** 稠密光流（`calcOpticalFlowFarneback`） |
| 计算时机 | 仅在 **采样帧对** `(I_{t-1}, I_t)` 上算（默认 2 fps），**不对原始 25/30fps 逐帧算**（小时级视频否则算力/IO 浪费；挑战点：更重的 RAFT 本期不上，除非评测证明 Farneback 不够） |
| 标量能量 | `m_t = mean(sqrt(u^2+v^2))`，供 idle / PELT |
| **运动描述子（进原型）** | 对光流做 **HOOF**：把画面划 **3×3 空间格**，每格对方向角量化 **8 bin**，幅度加权投票 → `o_t ∈ R^{72}`，再 L2 归一化。这是步骤级「手怎么动」的签名 |
| 可选增强 | 对 `(u,v)` 可视化成 flow-RGB 图，再过一遍轻量 CNN/同一 DINOv2 得 `o_t_deep`；**默认关闭**，先上 HOOF（快、稳、无额外大模型） |

#### 4.1.3 背景处理（固定机位 + 已验证的 MOG2）

**同意需要压背景。** 业务侧已在同类固定机位上验证 **MOG2 可行**，因此将 MOG2 从「辅」提升为 **默认前景门控主来源**（仍零图片标注）。

| 方法 | 算法 | 用法 |
|------|------|------|
| **主：MOG2** | OpenCV `createBackgroundSubtractorMOG2`（`detectShadows=True` 建议开，阴影区降权） | 输出前景概率/掩膜 → resize 到 patch 网格，作 DINOv2 **软权重**；跨 chunk **持久化 subtractor 状态**，避免每块冷启动 |
| **辅：光流幅度** | Farneback `mag` | 与 MOG2 门控取 **max**（或 `0.6*mog + 0.4*mag`），弥补「刚放下的零件被吸进背景、下一步再拿起」时 MOG 滞后 |
| **明确不做（本期）** | 人工 ROI、语义分割训练集、按测试视频写死裁剪框；默认 **硬二值抠空后再送 DINO** | 防过拟合 / 防特征塌缩 |

**仍保留的挑战**：MOG2「可行」≠「硬 mask 最好」。默认 `hard_mask: false`：门控是 0～1 权重。若你后续坚持硬抠，必须在评测里 A/B（soft vs hard），过了再改默认。

**MOG 更新频率可以 ≠ 特征频率**：见 §4.2，`mog_update_fps` 可高于 `sample_fps`（只更新背景模型、不跑 DINO），让模型更稳且特征仍降采样。

#### 4.1.4 帧级融合向量

```
x_t = normalize( concat( a_t ,  γ · o_t ) )     # 默认 γ=1.0，可配
```

- 维度约 `768+72=840`（若开启 `Δa_t` 再加 768，或改为 `a_t` 与 `Δa_t` 的加权和以控制维度）。  
- **默认开启 `Δa_t` 时**：`a'_t = normalize(0.7 a_t + 0.3 Δa_t)`，再与 `o_t` concat。  
- 原型与推理必须用 **同一套** `x_t`。

**预留升级（评测不够再开）**: InternVideo2 / VideoMAE 短窗 clip；Chinese-CLIP 文本分支。对齐层接口不变。

### 4.2 视频解码与降采样（不逐帧提特征）

**明确约定：不对原始视频每一帧跑 DINO / HOOF。** 特征只在降采样时间栅格上提取；这是默认行为，不是优化可选项。

| 项 | 选型 |
|----|------|
| 解码库 | **Decord**（按需 seek/batch）或 PyAV；禁止整片解码进 RAM |
| Chunk | 默认 **60s**，**overlap 3s** |
| **特征采样率 `sample_fps`** | 默认 **2.0**（每 0.5s 一个特征点）。DINO、光流对、HOOF、相似度、Viterbi 观测都只在这些点上计算 |
| **MOG 更新率 `mog_update_fps`** | 默认 **5.0**（仅 `apply` 背景模型，不提外观/HOOF）。可与特征栅格解耦 |
| 帧处理 | 仅对采样帧：BGR→RGB，224×224，DINOv2 预处理 |
| 内存 | chunk 内临时帧；全局只存采样栅格上的 `x_t`、`m_t` |

**推荐工作区间（配置，不写死单视频）**

| `sample_fps` | 时间粒度 | 适用 |
|--------------|----------|------|
| 1.0 | 1.0s | 步骤很长、只关心粗分割；**Boundary F1@0.5s 会吃亏** |
| **2.0（默认）** | 0.5s | 与本期 Boundary@0.5s 指标匹配的最低合理档 |
| 4.0～5.0 | 0.2～0.25s | 短步骤 / 掐表要更尖时再升 |

**挑战**：再往下降（如 0.5 fps）算力更省，但和「分割准确度 / 边界误差」目标冲突——除非你们接受 ±1～2s 的边界。demo 可先 2 fps；换工站若一步只有 1～2 秒，应升到 4 fps，而不是加视频特化规则。

**小时级内存（示意，`sample_fps=2`）**：`T≈7200` 时特征约数十 MB 量级，与像素全缓存无关。

### 4.3 原型构建算法（双流）

**输入**: fewshot 区间 + SOP 文本（文本可选进特征；顺序约束不靠文本，靠 4.5）。

对每个 `step_k`：

1. 区间内每个采样时刻算双流向量 `{x_i}`（§4.1.4）。  
2. 主原型：`p_k = normalize(mean_i x_i)`。  
3. **多原型**：`M >= 12` 时对 `{x_i}` 做 K-Means，`K=min(3, M//4)`；匹配  
   `sim(t,k) = max_j cos(x_t, p_{k,j})`。  
4. **分项诊断（导出，不进主路径也可）**：可另存纯外观原型 / 纯 HOOF 原型，用于分析「到底是外观糊还是运动糊」。  
5. 落盘 `prototypes.pt`。

**idle 原型**：fewshot 中步骤外且 `m_t` 低于该片第 20 百分位的 `x_i` 均值；若无足够帧则不建，idle 仅靠运动分（§4.5.2）。

**文本（可选）**：Chinese-CLIP 等可用时  
`sim(t,k) ← α_appflow * sim_dual(t,k) + (1-α_appflow) * sim_text(t,k)`，默认无文本编码器则 `α_appflow=1`。

### 4.4 运动能量、光流与变点

| 项 | 选型 |
|----|------|
| 主运动 | **Farneback 光流幅度** `m_t`（与 HOOF 同源，算一次多用） |
| 回退 | 若某环境 Farneback 过慢，可降为帧差能量，但 **HOOF 原型关闭时必须在评测里对比掉点**，不得默认永久关掉光流 |
| 平滑 | `m_t` 窗口 5 中值滤波 |
| 变点 | **`ruptures.Pelt(model="rbf")`**，输入 `m_t` 或 `1-max_k sim(t,k)`；`pen` 默认 3.0，只许在评测集上调并写入 config |
| 用途 | idle 倾向；Viterbi 变点偏置；**不单独当步骤分类器** |

`τ_motion = percentile(m, 25)` 按 chunk/视频自适应；`idle_min_sec` 默认 1.0s。

### 4.5 有序对齐：HMM + Viterbi —— 「第几步」在这里决定

> **常见误解**：以为「和哪一步原型最像」就是分割结果。  
> **正确关系**：原型相似度只是发射分（emission）；**允许的步骤顺序写在转移图里**，Viterbi 在整段时间上求满足顺序的最优路径。  
> 没有本模块，系统确实「没有顺序性」。Fewshot 建库假设标准顺序；**长视频**可配置向前漏步，但默认仍禁止回退返工。

#### 4.5.1 状态与允许边（顺序的精确定义）

```
S = { idle, step_1, step_2, ..., step_N }
```

**基线边（始终开启）**：

```
                  ┌──── self-loop ────┐
                  ▼                   │
idle ──────────► step_1 ──► step_2 ──► ... ──► step_N ──┐
  ▲               │          │                    │      │
  └───────────────┴──────────┴────────────────────┘      │
        (step_k → idle → step_{k+1}；k=N 则 idle→step_1)   │
                    step_N 也可直接 ──► step_1 ──────────┘
```

| 转移 | 默认 log-潜势 | 顺序含义 |
|------|----------------|----------|
| `step_k → step_k` | `0.0` | 仍停在第 k 步 |
| `step_k → step_{k+1}` | `-0.2` | 正常前进一步 |
| `step_k → idle` | `-0.8` | 步间等待 / 作业中停顿（对应标注 `empty`） |
| `idle → idle` | `0.0` | 连续等待 |
| `idle → step_{k+1}` | `-0.3` | 等待后进入下一步（开局为 step_1） |
| `step_N → step_1` | `-0.5` | 新一轮 SOP |

**长视频漏步（`allow_skip_forward: true`，推理默认开）**：

| 转移 | 默认 log-潜势 | 含义 |
|------|----------------|------|
| `step_k → step_j`（`j > k+1`） | `-1.5` | 向前跳过中间步（低于正常 +1 步，避免动辄跳步） |
| `idle → step_j`（`j` 大于期望 next） | `-1.2` | 停顿后直接进入更后面的步 |

**返工（`allow_rework: false` 默认）**：`j < k` 的回退边为 `-inf`；仅当显式打开时再加低概率回退。

因此：相似度不会单独决定步序；漏步必须走转移图上的 skip 边，并可在结果里标成低置信 / 异常供抽检。

**SOP 文本**：来自 `sop.json` 的 `text`；不替代转移图。

#### 4.5.2 发射概率（Emission）—— 「像不像」

```
s_k(t)    = sim(t, k) = max_j cos(x_t, p_{k,j})
s_idle(t) = β * cos(x_t, p_idle) + (1-β) * (1 - normalize(m_t))
P(obs_t | state) = softmax(s / τ),  τ=0.07
```

无 `p_idle` 时：`s_idle(t)=(1-normalize(m_t))*(1-max_k s_k(t))`。

#### 4.5.3 Viterbi 解码

手写 **log-Viterbi**：`O(T·(N+1)²)`。  
可选：落在 PELT 变点 ±1 帧时，给非自环转移 `+0.3` log 潜势（`use_changepoint_bias`）。

输出状态序列后合并为区间；`< min_segment_sec`（默认 0.5s）的短 step 段并入邻段。

#### 4.5.4 用合成例子说明顺序（实现测试必覆盖）

设 N=3。若某段时间内发射分始终是「最像 step_3」，但路径从 step_1 起步，则最优合法路径仍是 `1→2→3` 的推进（在 1、2 上可能很短），**而不是**整段直接标 3。单测应用人造 `E[t,k]` 矩阵断言此行为。

### 4.6 稀疏 VLM 复核（可选）

| 项 | 选型 |
|----|------|
| 模型 | **公司内部视觉多模态大模型**（通过现有 API；具体模型名以公司平台为准） |
| 默认 | **`vlm.enabled = false`**；主分割不依赖它 |
| 触发 | segment 级 `confidence < θ`（默认 0.45），或 `max_k s_k(t)` 在该段均值低于 θ |
| 输入 | 将该段均匀抽 **6 或 9 帧**拼成 **2×3 / 3×3 多宫格 JPEG** + SOP 全文 + 当前粗标签；一次 API 一张拼图 |
| 输出 | JSON：`{"label": "step_k"|"idle", "confidence": 0~1}`；写回覆盖该段 |
| 配额 | `vlm.max_calls_per_hour_video` 默认 **20**；超出则跳过剩余低置信段，仅保留本地结果 |
| 禁止 | 滑窗每秒调用；整视频逐块无差别调用 |

多宫格构造：PIL/OpenCV 按时间从左到右、从上到下贴帧，带可选时间码文字（通用绘制，非业务规则）。

### 4.7 置信度定义（供抽检与 VLM）

对某一合并段 `[t_a, t_b]` 预测为 `k`：

```
confidence = mean_{t in [a,b]} ( P(state=k|obs_t) )
```

即该段内 Viterbi 所用发射分布上该状态的平均概率（或 softmax 后均值）。导出到结果 JSON。

---

## 5. 分割评测（现阶段唯一优化目标）

### 5.1 评测数据

- 与 fewshot **不同** 的 1～2 轮人工区间标注（同工站同 SOP）。  
- 至少再准备 **另一工站或另一时段** 各一段，用于防过拟合验收。

### 5.2 指标（必须算）

| 指标 | 定义 | 实现参考 |
|------|------|----------|
| **帧级 Accuracy / Macro-F1** | 采样时刻预测标签 vs GT（含 idle） | sklearn |
| **分段 IoU** | 对每个 step_k，预测区间并集与 GT 的 IoU，再对 k 平均 | 动作分割常用 |
| **Boundary F1 @ δ** | 预测边界与 GT 边界在 δ 秒内算命中；δ∈{0.5, 1.0}s | MS-TCN / action segmentation 设定 |
| **段编辑得分** | 对步骤序列（忽略 idle）算归一化编辑距离 `1 - Levenshtein/ max_len` | 检测并步/裂步/跳步 |

本期 **不以** 循环次数、P50、NVA 准确率为发布门槛。

### 5.3 过拟合防护约定

- 禁止在代码中写死某测试视频路径下的特殊分支。  
- 所有阈值进 `station_config`；调参只允许在 **评测集** 上做，并记录配置哈希。  
- Demo 汇报必须附带：同配置下 ≥2 段视频的指标表。

---

## 6. 配置项清单（`station_config.yaml`）

```yaml
sample_fps: 2.0                   # 特征降采样；绝不对原生每帧跑 DINO/HOOF
mog_update_fps: 5.0               # 仅更新 MOG2，可高于 sample_fps
chunk_sec: 60.0
chunk_overlap_sec: 3.0
encoder: dinov2_vitb14
use_delta_appearance: true
appearance_delta_mix: 0.3          # a' = (1-mix)*a + mix*Δa
flow:
  enabled: true
  algo: farneback
  hoof_grid: [3, 3]
  hoof_bins: 8
  gamma: 1.0                       # concat 时 flow 侧权重
background:
  mode: soft_mog2_union            # MOG2 主门控 ∪ 光流幅度（业务已验证 MOG 可行）
  hard_mask: false                 # 默认软权重；硬抠需评测 A/B 后才可改
  persist_mog_across_chunks: true
kmeans_max_prototypes: 3
text_encoder: null
alpha_appflow: 1.0                 # 有文本编码器时可 <1
softmax_temperature: 0.07
idle_min_sec: 1.0
min_segment_sec: 0.5
allow_skip_forward: true           # 长视频推理允许向前漏步；fewshot 校验仍禁止漏步
allow_rework: false                # 默认禁止回退返工
use_changepoint_bias: true
ruptures_pen: 3.0
vlm:
  enabled: false
  max_calls_per_hour_video: 20
  confidence_threshold: 0.45
  grid: "3x3"
```

---

## 7. 模块与文件边界（实现时）

```
sop_step_segmentation/
  configs/station_example.yaml
  src/
    io/video_stream.py             # Decord chunk 读取
    io/sop_schema.py
    features/dinov2_encoder.py     # patch tokens + 加权池化
    features/optical_flow.py       # Farneback + HOOF + m_t
    features/background_gate.py    # soft motion / MOG2 门控
    features/changepoints.py       # ruptures PELT
    features/fuse.py               # a_t、Δa_t、o_t → x_t
    prototypes/build.py
    prototypes/similarity.py
    align/hmm_viterbi.py           # 有序转移图 + Viterbi（顺序核心）
    vlm/multigrid.py
    vlm/review.py
    eval/segmentation_metrics.py
    pipeline/run_offline.py
  tests/
    test_viterbi_order_constraints.py  # 禁止跳步/强制 1→2→3
    test_stream_memory.py
    test_hoof_shape.py
```

每个文件单一职责；对齐逻辑不依赖具体视频。

---

## 8. 端到端数据流（算法级）

1. **Build**: fewshot → 背景软门控 → DINOv2 加权外观 + Farneback HOOF → 双流原型。  
2. **Extract**: 目标视频 chunk → 同套双流 `x_t`、`m_t`。  
3. **Score**: `sim(t,k)`、`s_idle` → 发射矩阵 `E`。  
4. **Changepoint**: PELT(`m_t`)。  
5. **Align（顺序）**: 合法转移图上 log-Viterbi → segments。  
6. **Optional VLM** / **Eval**。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 背景主导 CLS | patch 运动加权池化 + soft FG；禁止只靠整帧均值 |
| 帧间外观差很小 | 双流 HOOF + `Δa_t`；不单押 DINO |
| 两步外观极似 | **有序 Viterbi 禁止跳步**；运动描述子区分手法 |
| 硬背景减除误伤静止工件 | 默认 soft gate，`hard_mask: false` |
| 光流算力 | 只在 sample_fps 帧对上 Farneback；不上默认 RAFT |
| 换工站 | 每站重建 fewshot 原型；算法与阈值走 config |
| VLM 配额 | 默认关 + 硬上限 |

---

## 10. 决议摘要

| 决策 | 选择 |
|------|------|
| 业务形态 | 离线分析 |
| 监督 | 每工站 1 轮区间 + 文本 SOP |
| 外观 | DINOv2 ViT-B/14 **patch 运动加权**（非裸 CLS） |
| 运动 | **Farneback + HOOF 进原型**；幅度供 idle/PELT |
| 背景 | **MOG2 主门控**（已验证）+ 光流辅助；默认 soft，不硬抠 |
| 采样 | **特征降采样**默认 2 fps；MOG 更新可 5 fps；非逐帧提特征 |
| 标注输入 | X-AnyLabeling sidecar；SOP 文本在独立 `sop.json` |
| Fewshot | 标准一轮：可插 empty，不漏步、不回退 |
| 长视频 | `allow_skip_forward: true`；`allow_rework: false` |
| 顺序 | **有序 HMM 转移图 + Viterbi**（相似度不决定步序） |
| VLM | 默认关；多宫格稀疏复核 |
| 本期优化目标 | 仅分割准确度 |

---

## 11. 后续可改项（不阻塞初稿）

- 编码器升级为 InternVideo2 clip 特征  
- 打开中文 CLIP 文本分支  
- `allow_rework` 与更细转移势  
- 报表层（循环 / P50 / NVA）  
- 公司 VLM 的具体 API 字段映射
