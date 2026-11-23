# SOP Step Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现离线流水线：用工站 1 轮 fewshot 区间 + SOP 文本建双流原型，对流式降采样视频做有序 Viterbi 步骤分割，并以分割指标验收。

**Architecture:** 局部观测（MOG2 软门控 + DINOv2 patch 加权 + Farneback HOOF）在 `sample_fps` 栅格上独立/成对提取；全局用有序 HMM+Viterbi。Fewshot 为标准一轮（可插 idle、不漏步），经 X-AnyLabeling sidecar + `sop.json` 桥接；长视频推理默认 `allow_skip_forward`。公司 VLM 默认关闭。

**Tech Stack:** Python 3.11+、PyTorch、timm/transformers(DINOv2)、OpenCV、Decord 或 PyAV、NumPy、SciPy/scikit-learn、ruptures、PyYAML、pytest

**Spec:** [docs/superpowers/specs/2026-11-22-sop-step-segmentation-design.md](../specs/2026-11-22-sop-step-segmentation-design.md)

## Global Constraints

- 不对原生视频逐帧跑 DINO/HOOF；特征仅在 `sample_fps`（默认 2.0）上提取
- 禁止针对单条测试视频写死 ROI/颜色/分支；阈值只进 `station_config`
- 主路径不依赖公司 VLM（`vlm.enabled` 默认 false）
- 长视频按 chunk 流式处理，禁止整片帧列表常驻内存
- 本期优化目标仅分割准确度；循环/P50/NVA 报表不做
- Fewshot 校验：步骤 `1..N` 齐全且时间序严格递增；中间 idle/`empty` 允许；缺步拒绝建原型
- 长视频推理：默认 `allow_skip_forward: true`；默认 `allow_rework: false`
- 标注主格式为 X-AnyLabeling sidecar；SOP 长文本只在 `sop.json`；忽略 segment `description`
- `background.hard_mask: false`
- 所有公共 API 与测试使用英文标识符；用户可见文档可中英

---

## File Structure (lock-in)

```
sop-step-segmentation/
  pyproject.toml
  configs/station_example.yaml
  src/sop_seg/
    __init__.py
    types.py
    io/
      sop_schema.py
      video_stream.py
      config.py
    features/
      background_gate.py
      optical_flow.py
      dinov2_encoder.py
      fuse.py
      changepoints.py
    prototypes/
      build.py
      similarity.py
    align/
      hmm_viterbi.py
    eval/
      segmentation_metrics.py
    vlm/
      multigrid.py
      review.py
    pipeline/
      extract.py
      run_offline.py
    cli.py
  tests/
    test_sop_schema.py
    test_video_stream.py
    test_optical_flow.py
    test_background_gate.py
    test_fuse.py
    test_changepoints.py
    test_prototypes.py
    test_viterbi_order_constraints.py
    test_segmentation_metrics.py
    test_pipeline_synthetic.py
    conftest.py
```

---

### Task 1: Project scaffold + config loader

**Files:**
- Create: `pyproject.toml`
- Create: `configs/station_example.yaml`
- Create: `src/sop_seg/__init__.py`
- Create: `src/sop_seg/types.py`
- Create: `src/sop_seg/io/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `load_station_config(path: str | Path) -> StationConfig` dataclass mirroring spec §6
- Produces: `Segment`, `StepDef`, `FewshotAnnotation` dataclasses in `types.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from pathlib import Path
from sop_seg.io.config import load_station_config

def test_load_station_example_defaults():
    cfg = load_station_config(Path("configs/station_example.yaml"))
    assert cfg.sample_fps == 2.0
    assert cfg.mog_update_fps == 5.0
    assert cfg.flow.enabled is True
    assert cfg.background.mode == "soft_mog2_union"
    assert cfg.background.hard_mask is False
    assert cfg.vlm.enabled is False
    assert cfg.allow_skip_forward is True
    assert cfg.allow_rework is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_station_example_defaults -v`  
Expected: FAIL (import / file missing)

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml` package `sop_seg` under `src/`, deps: `numpy`, `pyyaml`, `opencv-python-headless`, `torch`, `timm`, `scikit-learn`, `ruptures`, `av`, `pytest`.

Copy defaults from spec §6 into `configs/station_example.yaml`.

```python
# src/sop_seg/io/config.py (sketch — implement full nested dataclasses)
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class FlowConfig:
    enabled: bool = True
    algo: str = "farneback"
    hoof_grid: tuple[int, int] = (3, 3)
    hoof_bins: int = 8
    gamma: float = 1.0

@dataclass
class BackgroundConfig:
    mode: str = "soft_mog2_union"
    hard_mask: bool = False
    persist_mog_across_chunks: bool = True

@dataclass
class VlmConfig:
    enabled: bool = False
    max_calls_per_hour_video: int = 20
    confidence_threshold: float = 0.45
    grid: str = "3x3"

@dataclass
class StationConfig:
    sample_fps: float = 2.0
    mog_update_fps: float = 5.0
    chunk_sec: float = 60.0
    chunk_overlap_sec: float = 3.0
    encoder: str = "dinov2_vitb14"
    use_delta_appearance: bool = True
    appearance_delta_mix: float = 0.3
    flow: FlowConfig = None  # type: ignore
    background: BackgroundConfig = None  # type: ignore
    kmeans_max_prototypes: int = 3
    text_encoder: str | None = None
    alpha_appflow: float = 1.0
    softmax_temperature: float = 0.07
    idle_min_sec: float = 1.0
    min_segment_sec: float = 0.5
    allow_skip_forward: bool = True
    allow_rework: bool = False
    use_changepoint_bias: bool = True
    ruptures_pen: float = 3.0
    vlm: VlmConfig = None  # type: ignore

def load_station_config(path: Path) -> StationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # nest-construct FlowConfig/BackgroundConfig/VlmConfig from raw dicts; fill defaults for missing keys
    ...
```

```python
# src/sop_seg/types.py
from dataclasses import dataclass

@dataclass(frozen=True)
class StepDef:
    step_id: int
    label: str
    text: str

@dataclass(frozen=True)
class Interval:
    t_start_sec: float
    t_end_sec: float
    label: str  # "step_1" / "idle"
    step_id: int | None  # None for idle

@dataclass
class OutputSegment:
    t_start: float
    t_end: float
    label: str  # "step_k" or "idle"
    confidence: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/station_example.yaml src/sop_seg tests/test_config.py tests/conftest.py
git commit -m "chore: scaffold package and station config loader"
```

---

### Task 2: sop.json + X-AnyLabeling bridge + output JSON

**Files:**
- Create: `src/sop_seg/io/sop_schema.py`
- Create: `src/sop_seg/io/xanylabeling.py`
- Create: `tests/test_sop_schema.py`
- Create: `tests/test_xanylabeling_bridge.py`
- Reference examples: `examples/sop.json`, `examples/xanylabeling_fewshot_sidecar.json`

**Interfaces:**
- Consumes: spec §2.3–2.5
- Produces:
  - `load_sop(path) -> SopDocument` (`station_id`, `steps: list[StepDef]`, `idle_labels: list[str]`)
  - `StepDef(step_id: int, label: str, text: str)`
  - `load_xanylabeling_sidecar(path) -> XanyVideoAnn`
  - `bridge_intervals(sop: SopDocument, ann: XanyVideoAnn) -> list[Interval]`
    - map `label` via `sop.steps[].label` / `idle_labels`; `ms/1000`; **ignore `description`**
  - `validate_fewshot_one_cycle(sop, intervals) -> None`
    - every `step_id` once in order `1..N`; idle segments allowed between; raise on skip/missing/reorder
  - `dump_result(path, segments, sample_fps, encoder, aligner) -> None`（spec §2.7）

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sop_schema.py
from pathlib import Path
from sop_seg.io.sop_schema import load_sop

def test_load_example_sop():
    sop = load_sop(Path("examples/sop.json"))
    assert sop.station_id == "ws_demo"
    assert [s.step_id for s in sop.steps] == [1, 2, 3]
    assert sop.steps[0].text.startswith("从料盘")
    assert "empty" in sop.idle_labels

# tests/test_xanylabeling_bridge.py
from pathlib import Path
from sop_seg.io.sop_schema import load_sop, validate_fewshot_one_cycle
from sop_seg.io.xanylabeling import load_xanylabeling_sidecar, bridge_intervals

def test_bridge_example_fewshot_allows_mid_idle():
    sop = load_sop(Path("examples/sop.json"))
    ann = load_xanylabeling_sidecar(Path("examples/xanylabeling_fewshot_sidecar.json"))
    intervals = bridge_intervals(sop, ann)
    validate_fewshot_one_cycle(sop, intervals)  # empty between step2 and step3 OK
    step_ids = [i.step_id for i in intervals if i.step_id is not None]
    assert step_ids == [1, 2, 3]

def test_fewshot_rejects_missing_step(tmp_path: Path):
    # sidecar missing step2 -> validate_fewshot_one_cycle raises ValueError
    ...
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_sop_schema.py tests/test_xanylabeling_bridge.py -v`

- [ ] **Step 3: Implement loaders + bridge + fewshot validator**

- Unknown labels → `ValueError` with label name  
- Overlapping time ranges → `ValueError`  
- `description` never copied into `Interval`

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/sop_seg/io/sop_schema.py src/sop_seg/io/xanylabeling.py tests/examples
git commit -m "feat: load sop.json and bridge X-AnyLabeling fewshot sidecars"
```

---

### Task 3: Chunked video sampling (no full-frame cache)

**Files:**
- Create: `src/sop_seg/io/video_stream.py`
- Create: `tests/test_video_stream.py`

**Interfaces:**
- Produces:
  - `iter_sample_frames(video_path, sample_fps, chunk_sec, chunk_overlap_sec) -> Iterator[SampleFrame]`
  - `SampleFrame(t_sec: float, frame_bgr: np.ndarray, chunk_index: int)`
  - `iter_mog_update_frames(...)` at `mog_update_fps` (frames for MOG only)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_video_stream.py
import numpy as np
import av
from pathlib import Path
from sop_seg.io.video_stream import iter_sample_frames

def _write_dummy_video(path: Path, n_frames=30, fps=10, size=64):
    # write solid-color frames with PyAV
    ...

def test_sample_fps_downsamples(tmp_path: Path):
    path = tmp_path / "v.mp4"
    _write_dummy_video(path, n_frames=30, fps=10)  # 3.0s
    ts = [sf.t_sec for sf in iter_sample_frames(path, sample_fps=2.0, chunk_sec=60.0, chunk_overlap_sec=0.0)]
    assert len(ts) == 6  # 0.0,0.5,...,2.5 approximately
    assert max(ts) < 3.0

def test_iterator_does_not_return_all_native_frames(tmp_path: Path):
    path = tmp_path / "v.mp4"
    _write_dummy_video(path, n_frames=30, fps=10)
    n = sum(1 for _ in iter_sample_frames(path, sample_fps=2.0, chunk_sec=60.0, chunk_overlap_sec=0.0))
    assert n < 30
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement with PyAV (or Decord if installable on the machine)**

Prefer **PyAV** for Windows reliability. Seek/decode only timestamps on the `sample_fps` grid. Never accumulate all frames in a list inside the iterator (yield one-by-one).

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add chunked downsampled video frame iterator"
```

---

### Task 4: Optical flow + HOOF + motion energy

**Files:**
- Create: `src/sop_seg/features/optical_flow.py`
- Create: `tests/test_optical_flow.py`

**Interfaces:**
- Produces:
  - `farneback_flow(prev_gray, cur_gray) -> tuple[np.ndarray, np.ndarray]`  # u,v
  - `flow_magnitude(u, v) -> float`
  - `hoof_descriptor(u, v, grid=(3,3), bins=8) -> np.ndarray`  # shape (72,), L2-normalized

- [ ] **Step 1: Write failing tests**

```python
def test_hoof_shape_and_norm():
    h, w = 60, 60
    # synthetic rightward flow
    u = np.ones((h, w), np.float32)
    v = np.zeros((h, w), np.float32)
    desc = hoof_descriptor(u, v, grid=(3, 3), bins=8)
    assert desc.shape == (72,)
    assert abs(np.linalg.norm(desc) - 1.0) < 1e-5

def test_static_flow_near_zero_magnitude():
    g = np.zeros((32, 32), np.uint8)
    u, v = farneback_flow(g, g)
    assert flow_magnitude(u, v) < 1e-3
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement Farneback + HOOF**

HOOF: split into 3×3 cells; each cell histogram of `atan2(v,u)` into 8 bins weighted by magnitude; concat → L2 normalize. If norm≈0, return zeros.

- [ ] **Step 4: PASS + Commit**

```bash
git commit -m "feat: add Farneback flow magnitude and HOOF descriptor"
```

---

### Task 5: MOG2 soft background gate

**Files:**
- Create: `src/sop_seg/features/background_gate.py`
- Create: `tests/test_background_gate.py`

**Interfaces:**
- Produces: class `BackgroundGate`
  - `__init__(self, mode: str, hard_mask: bool, persist: bool)`
  - `update(self, frame_bgr: np.ndarray) -> None`  # MOG apply
  - `gate_map(self, frame_bgr: np.ndarray, flow_mag_map: np.ndarray | None) -> np.ndarray`
    - returns float32 HxW in `[0,1]`
  - `patch_weights(self, gate_map: np.ndarray, grid_h: int = 14, grid_w: int = 14) -> np.ndarray`
    - shape `(196,)` for ViT-B/14

- [ ] **Step 1: Write failing tests**

```python
def test_gate_map_in_unit_interval():
    gate = BackgroundGate(mode="soft_mog2_union", hard_mask=False, persist=True)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    for _ in range(10):
        gate.update(frame)
    g = gate.gate_map(frame, flow_mag_map=np.zeros((64, 64), np.float32))
    assert g.shape == (64, 64)
    assert g.min() >= 0.0 and g.max() <= 1.0

def test_hard_mask_false_keeps_soft_values():
    # after updates, gate should have values strictly between 0 and 1 on mixed content
    ...
```

- [ ] **Step 2–4: Implement**

- Use `cv2.createBackgroundSubtractorMOG2(detectShadows=True)`.
- Soft mask: `fg.astype(f32)/255`, shadow pixels (value 127) → weight ~0.3.
- `soft_mog2_union`: `gate = max(mog_soft, normalize(flow_mag_map))` elementwise.
- Never binarize when `hard_mask=False`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add MOG2 soft background gate with flow union"
```

---

### Task 6: DINOv2 encoder with patch weighted pooling (mockable)

**Files:**
- Create: `src/sop_seg/features/dinov2_encoder.py`
- Create: `tests/test_dinov2_encoder.py`

**Interfaces:**
- Produces: protocol / class `AppearanceEncoder`
  - `embed(self, frame_rgb: np.ndarray, patch_weights: np.ndarray | None) -> np.ndarray` → `(768,)` L2-normalized
- Produces: `DinoV2Encoder(device: str = "cpu")` loading `facebook/dinov2-base` OR `timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)`
- Produces: `FakeAppearanceEncoder(dim=768)` for unit tests (hash/random projection of resized pixels — deterministic seed)

**Important:** Unit tests **must not** require downloading DINOv2 weights. Use `FakeAppearanceEncoder` in tests; optional `@pytest.mark.gpu` / `integration` for real model.

- [ ] **Step 1: Write failing tests with Fake encoder**

```python
def test_weighted_pool_changes_embedding():
    enc = FakeAppearanceEncoder(dim=32)
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :112] = 255
    w_left = np.ones(196); w_left[98:] = 0.01
    w_right = np.ones(196); w_right[:98] = 0.01
    e1 = enc.embed(img, w_left)
    e2 = enc.embed(img, w_right)
    assert e1.shape == (32,)
    assert np.linalg.norm(e1 - e2) > 0.1
```

- [ ] **Step 2–4: Implement Fake + real DinoV2Encoder**

Real encoder: preprocess ImageNet norm; forward hooks/get intermediate tokens; weighted sum `sum(w_i * token_i)/sum(w)`; L2 normalize. If `patch_weights is None`, uniform weights.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add appearance encoder with weighted patch pooling"
```

---

### Task 7: Feature fusion (`a`, `Δa`, `o` → `x`)

**Files:**
- Create: `src/sop_seg/features/fuse.py`
- Create: `tests/test_fuse.py`

**Interfaces:**
- Produces: `fuse_features(a_t, a_tm1, o_t, use_delta: bool, delta_mix: float, gamma: float) -> np.ndarray`

```python
def fuse_features(a_t, a_tm1, o_t, use_delta=True, delta_mix=0.3, gamma=1.0):
    a = a_t
    if use_delta and a_tm1 is not None:
        da = a_t - a_tm1
        da = da / (np.linalg.norm(da) + 1e-8)
        a = (1 - delta_mix) * a_t + delta_mix * da
        a = a / (np.linalg.norm(a) + 1e-8)
    o = gamma * o_t
    x = np.concatenate([a, o], axis=0)
    return x / (np.linalg.norm(x) + 1e-8)
```

- [ ] TDD + Commit: `feat: add dual-stream feature fusion`

---

### Task 8: Changepoints (ruptures PELT)

**Files:**
- Create: `src/sop_seg/features/changepoints.py`
- Create: `tests/test_changepoints.py`

**Interfaces:**
- Produces: `detect_changepoints(m: np.ndarray, pen: float) -> list[int]`  # indices into sample grid

- [ ] Test: synthetic step signal with two regimes → at least one breakpoint near true index (±2).
- [ ] Implement with `ruptures.Pelt(model="rbf").fit(m.reshape(-1,1)).predict(pen=pen)`
- [ ] Commit: `feat: add PELT changepoint detection on motion energy`

---

### Task 9: Prototypes build + similarity curves

**Files:**
- Create: `src/sop_seg/prototypes/build.py`
- Create: `src/sop_seg/prototypes/similarity.py`
- Create: `tests/test_prototypes.py`

**Interfaces:**
- Produces:
  - `StepPrototypes` dataclass: `step_id -> list[np.ndarray]` (K sub-prototypes), optional `idle: np.ndarray | None`
  - `build_prototypes(features: np.ndarray, times: np.ndarray, intervals: list[Interval], k_max: int) -> StepPrototypes`
  - `save_prototypes(path, prototypes)`, `load_prototypes(path)`
  - `similarity_matrix(features: np.ndarray, prototypes: StepPrototypes) -> np.ndarray`  # shape `(T, N)` max-cosine per step

- [ ] **Step 1: Tests**

```python
def test_build_and_similarity_prefers_matching_step():
    # handcraft 2 steps with orthogonal features
    T, D = 20, 8
    feats = np.zeros((T, D), np.float32)
    feats[:10, 0] = 1
    feats[10:, 1] = 1
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    times = np.arange(T) * 0.5
    intervals = [Interval(1, 0, 5), Interval(2, 5, 10)]
    proto = build_prototypes(feats, times, intervals, k_max=3)
    S = similarity_matrix(feats, proto)
    assert S[:10, 0].mean() > S[:10, 1].mean()
    assert S[10:, 1].mean() > S[10:, 0].mean()
```

K-Means via `sklearn.cluster.KMeans` when `M >= 12` and `k=min(k_max, M//4)`, else single mean.

- [ ] Commit: `feat: build dual-stream step prototypes and similarity matrix`

---

### Task 10: Ordered HMM + Viterbi (core)

**Files:**
- Create: `src/sop_seg/align/hmm_viterbi.py`
- Create: `tests/test_viterbi_order_constraints.py`

**Interfaces:**
- Produces:
  - `build_transition_log_matrix(n_steps: int, allow_skip_forward: bool, allow_rework: bool, potentials: dict | None) -> np.ndarray`
    - states order: `idle, step_1, ..., step_N` → size `(N+1, N+1)`
  - `emissions_from_similarity(S: np.ndarray, m: np.ndarray, temperature: float, idle_proto_sim: np.ndarray | None) -> np.ndarray`
    - returns log emission `(T, N+1)`
  - `viterbi(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray`  # state indices length T
  - `states_to_segments(states, times, min_segment_sec, confidences) -> list[OutputSegment]`
  - `apply_changepoint_bias(log_trans_per_t, changepoints, bias=0.3)` optional per-t transition boost for non-self loops

**Default log potentials** (from spec): self 0.0; `k→k+1` -0.2; `k→idle` -0.8; idle→idle 0.0; idle→next -0.3; `N→1` -0.5; illegal -inf.

- [ ] **Step 1: Critical failing tests**

```python
def test_forbids_skip_even_if_emission_prefers_later_step():
    # N=3, T=30; emission always highest for step_3; start forces step_1 path
    n, T = 3, 30
    log_emit = np.full((T, n + 1), -10.0)
    # idle col 0, steps 1..3 -> cols 1..3
    log_emit[:, 3] = 0.0  # always "looks like" step 3
    log_trans = build_transition_log_matrix(n, allow_skip_forward=False, allow_rework=False)
    path = viterbi(log_emit, log_trans)
    # path may stay on early steps then advance; must never visit step_3 before step_1 and step_2 appeared
    first = {s: int(np.argmax(path == s)) for s in (1, 2, 3) if np.any(path == s)}
    assert first[1] < first[2] < first[3]

def test_skip_forward_edge_exists_when_enabled():
    log_trans = build_transition_log_matrix(3, allow_skip_forward=True, allow_rework=False)
    # step_1 -> step_3 must be finite when skip enabled (exact state index per augmented scheme)
    ...

def test_rework_disabled_blocks_backward():
    log_trans = build_transition_log_matrix(3, allow_skip_forward=True, allow_rework=False)
    # step_2 -> step_1 must be -inf
    ...
```


Implement idle bookkeeping: track `next_step_after_idle` via expanding state or by encoding next-step in Viterbi with augmented states.

**Augmented-state recommendation (implement this):** states = `idle_before_k` for k=1..N and `step_k` for k=1..N, OR simpler: keep `(N+1)` states but make `idle→step_j` legal only for j in `{1..N}` with equal potential **only if** `allow_open_idle_restart`; **preferred simpler v1:** disallow bare idle ambiguity by using states:

`S = [step_1..step_N, idle_1..idle_N]` where `idle_k` means “idle after finishing step k (or before step_1 if k=0)”.

Minimal v1 that matches tests:

- States: `0=idle, 1..N=steps`
- From idle, allowed emissions go to `step_1` only at start; after leaving `step_k` into idle, next forced step is `k+1` (or 1 if k=N). Implement with **2D Viterbi over (label, phase)** or store predecessor that carries `expected_next`.

**Concrete approach to implement:** use states indexed as:

- `0 .. N-1` → `step_1 .. step_N`
- `N .. 2N-1` → `idle_after_step_1 .. idle_after_step_N` where `idle_after_step_N` can go to `step_1`

Map back to labels `step_*` / `idle` in `states_to_segments`.

- [ ] **Step 2–4: Implement until order tests pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: ordered HMM Viterbi with no-skip transitions"
```

---

### Task 11: Per-video feature extraction loop

**Files:**
- Create: `src/sop_seg/pipeline/extract.py`
- Create: `tests/test_pipeline_synthetic.py` (partial: extraction on dummy video with Fake encoder)

**Interfaces:**
- Produces: `extract_feature_timeline(video_path, cfg: StationConfig, encoder: AppearanceEncoder) -> FeatureTimeline`
  - `FeatureTimeline(times: np.ndarray, features: np.ndarray, motion: np.ndarray)`

Algorithm:

1. Init `BackgroundGate` from cfg.
2. Iterate native timeline for MOG updates at `mog_update_fps` interleaved OR second pass; keep last gray for flow.
3. On each `sample_fps` timestamp: compute flow vs previous sample → `o_t`, `m_t`, `gate_map`, `patch_weights`, `a_t`, fuse → `x_t`.
4. Append; do not keep pixel buffers.

- [ ] Test with Fake encoder + dummy video: `features.shape[0] == len(times)`, `features.shape[1] == dim_a + 72`.
- [ ] Commit: `feat: stream feature extraction timeline`

---

### Task 12: Offline pipeline (build prototypes → align → dump)

**Files:**
- Create: `src/sop_seg/pipeline/run_offline.py`
- Create: `src/sop_seg/cli.py`
- Modify: `tests/test_pipeline_synthetic.py`

**Interfaces:**
- Produces:
  - `build_station_prototypes(fewshot_video, fewshot_intervals, cfg, encoder) -> StepPrototypes`
  - `segment_video(video, prototypes, cfg, encoder) -> list[OutputSegment]`
  - CLI:
    - `python -m sop_seg.cli build-prototypes --video ... --fewshot ... --sop ... --config ... --out prototypes.npz`
    - `python -m sop_seg.cli segment --video ... --prototypes ... --sop ... --config ... --out result.json`

- [ ] **Synthetic E2E test:** construct features/prototypes in-memory (bypass DINO) where true path is 1→2→3 with idle; assert decoded labels order and segment count ≥ 3 step segments.

- [ ] Wire optional changepoint bias + `min_segment_sec` merge.

- [ ] Commit: `feat: offline build-prototypes and segment CLI`

---

### Task 13: Segmentation metrics

**Files:**
- Create: `src/sop_seg/eval/segmentation_metrics.py`
- Create: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Produces: `evaluate_segmentation(pred: list[OutputSegment], gt: list[Interval], sample_times: np.ndarray, n_steps: int, boundary_deltas=(0.5, 1.0)) -> dict`

Metrics:

- frame accuracy / macro-F1 (map intervals onto `sample_times`)
- mean per-step IoU
- boundary F1 at 0.5s and 1.0s
- sequence edit score ignoring idle

```python
def test_perfect_match_scores_high():
    # identical pred/gt -> accuracy 1.0, mean_iou 1.0, edit 1.0
    ...
```

- [ ] CLI subcommand: `python -m sop_seg.cli eval --pred result.json --gt fewshot_or_gt.json --config ...`
- [ ] Commit: `feat: add segmentation evaluation metrics and eval CLI`

---

### Task 14: Sparse VLM stubs (disabled by default)

**Files:**
- Create: `src/sop_seg/vlm/multigrid.py`
- Create: `src/sop_seg/vlm/review.py`
- Create: `tests/test_multigrid.py`

**Interfaces:**
- Produces: `make_multigrid(frames_rgb: list[np.ndarray], grid: str) -> np.ndarray` (HxWx3 uint8)
- Produces: `VlmReviewer` protocol with `review(grid_image, sop_text, coarse_label) -> label`
- Produces: `NullVlmReviewer` (no-op) and `BudgetedVlmReviewer` wrapping a callable with `max_calls` counter

- [ ] Do **not** call real company API in tests.
- [ ] Hook in `segment_video`: if `cfg.vlm.enabled`, refine low-confidence segments only.
- [ ] Commit: `feat: add multigrid helper and budgeted VLM review stub`

---

### Task 15: README for running demo (minimal)

**Files:**
- Create: `README.md`

Content: install (`pip install -e .`); prepare `examples/sop.json` + X-AnyLabeling sidecar；`build-prototypes` / `segment` / `eval`；GPU optional；point to spec §2 examples.

- [ ] Commit: `docs: add README with offline segmentation workflow`

---

## Spec coverage checklist (self-review)

| Spec item | Task |
|-----------|------|
| Downsampled features / no full-frame cache | 3, 11 |
| MOG2 soft gate + flow union | 5 |
| DINOv2 weighted patches + Δa | 6, 7 |
| Farneback + HOOF prototypes | 4, 9 |
| Ordered Viterbi no-skip | 10 |
| Changepoints PELT | 8 |
| Offline CLI build/segment | 12 |
| Segmentation metrics only | 13 |
| VLM optional budgeted | 14 |
| Config from YAML | 1 |
| sop.json + X-AnyLabeling bridge | 2 |
| Fewshot one-cycle validation (idle OK, no skip) | 2 |
| allow_skip_forward for long video | 10 |
| Reports P50/NVA | Explicitly out of scope |
| InternVideo2 / Chinese-CLIP | Explicitly out of scope |

## Placeholder / consistency notes

- Encoder dim: Fake tests may use `dim=32`; real DINO uses 768; `fuse`/`prototypes` must take `D` from data, not hardcode 768 in aligner.
- Viterbi state indexing must stay consistent between `build_transition_log_matrix`, tests, and `states_to_segments` label mapping — implement augmented idle states as specified in Task 10.
- Video backend: PyAV first; if Decord is added later, keep behind same `iter_sample_frames` interface.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-22-sop-step-segmentation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — 每个 Task 开新 subagent，Task 间做审查，迭代快  
2. **Inline Execution** — 本会话按 executing-plans 连续执行，设检查点  

你要哪一种？
