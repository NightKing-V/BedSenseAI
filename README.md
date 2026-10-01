# BedSense AI — Elderly Bed-Monitoring & Agentic Kinematics System

BedSense AI is an intelligent computer vision and agentic reasoning system designed for continuous monitoring of elderly patients around hospital and home-care beds. It determines posture activity states, detects bed-exit and bed-return events, tracks durations, and assigns clinical monitoring decisions (`NORMAL` / `MONITOR` / `ALERT`).

---

## 1. Architecture Overview

```mermaid
flowchart TD
    V[VIDEO STREAM] --> PV

    subgraph PV["Primary Vision Layer"]
        direction TB
        PV1[Single YOLO-Pose Model]
        PV2[ByteTrack Multi-Object Tracking]
        PV3[17 COCO Keypoints Extraction]
        PV4[Dynamic Bed Detection & Overlap Calculation]
    end

    PV --> TSE

    subgraph TSE["Temporal State Engine"]
        direction TB
        TSE1[Posture Geometry Classification]
        TSE2[Transition Smoothing >= 2.0s]
        TSE3[Duration & History Buffering]
        TSE4["Bed-Exit / Return Pattern Matcher"]
    end

    TSE --> ROUTE{Confidence Router}
    ROUTE -->|"High Confidence (Clear pattern match)"| POLICY
    ROUTE -->|"Low Confidence / Ambiguous (§4.1 Triggers)"| AGENT

    subgraph AGENT["Agent LLM (Qwen via Ollama)"]
        A1["Inspect Ambiguous Segment + Context [t-30s, t+30s]"]
        A2["Decide: Need more evidence?"]
    end

    AGENT -->|Decision Resolved| POLICY
    AGENT -->|Call Tool (Cap = 4)| TOOLS

    subgraph TOOLS["Agent Tool Suite"]
        direction TB
        T1["get_segment(t0, t1)"]
        T2["check_bed_overlap(t)"]
        T3["vlm_describe(t0, t1, question) - Qwen-VL"]
    end

    TOOLS --> AGENT

    subgraph POLICY["Clinical Policy Engine"]
        direction TB
        P1[NORMAL]
        P2[MONITOR]
        P3[ALERT]
        P4[Caregiver Escalation Suppression]
    end

    POLICY --> OUT["Outputs (timeline.json / events.json / summary.json)"]
```

---

## 2. Why the Confidence Router is Explicit

Not every frame or transition requires an expensive LLM call:
- **Unambiguous cases** (e.g. `LYING_IN_BED` $\to$ `SITTING_ON_BED` $\to$ `STANDING` $\to$ `WALKING` $\to$ `OUT_OF_BED`, with high keypoint confidence and sustained durations) resolve straight from the **Temporal State Engine's pattern matcher** with **no LLM call**.
- **Ambiguous cases** (low keypoint confidence $<0.4$, bed boundary hovering $0.4-0.6$, horizontal posture detected off-bed, or temporary track ID loss) route to the **Agent LLM** with surrounding context $[t - 30\text{s}, t + 30\text{s}]$.

This architecture bounds latency, reduces computational cost, and provides a clear, explainable audit trail (`tool_trace`) for clinical reviews.

---

## 3. Dynamic Bed Detection & Overlap Calculation

Rather than assuming rigid, fixed bed coordinates:
1. **Dynamic Scene Detection**: The `BedRelationEngine` runs object detection (`bed` and `couch` classes via YOLO) on initial video frames to calculate the median bounding box and boundary polygon.
2. **Pose Kinematic Calibration**: For scenes without clear object detections, the engine calculates the spatial bounding polygon from horizontal resting keypoint distributions.
3. **Continuous Overlap Ratio**: Computes continuous geometric intersection area:
   $$\text{overlap} = 0.55 \times \frac{\text{Area}(\text{Person} \cap \text{Bed})}{\text{Area}(\text{Person})} + 0.45 \times \text{Keypoint\_Containment\_Ratio}$$

---

## 4. The 7 Canonical Activity States

| State | Posture × Bed Relation Criteria |
| --- | --- |
| `LYING_IN_BED` | Torso horizontal ($>50^\circ$ from vertical) and bed overlap $\ge 0.35$ |
| `SITTING_ON_BED` | Torso upright/intermediate, bent knees, and bed overlap $\ge 0.35$ |
| `SITTING_OUTSIDE_BED` | Torso upright, seated in chair/furniture, bed overlap $< 0.10$ |
| `STANDING` | Upright vertical posture, low displacement velocity ($<25$ px/s) |
| `WALKING` | Upright vertical posture, high displacement velocity ($\ge 25$ px/s) |
| `OUT_OF_BED` | Patient detected outside bed zone or horizontal on floor |
| `UNKNOWN` | Low keypoint confidence ($<0.25$) or contradictory geometric signals |

**Smoothing Rule**: A state only forms a segment if held for $\ge 2.0\text{s}$ (`MIN_SEGMENT_SEC`). Shorter blips are collapsed into neighboring states to eliminate visual noise.

---

## 5. Clinical Alert Rules Justification

| Condition | Decision | Clinical Justification |
| --- | --- | --- |
| Normal lying/sitting/standing/walking | `NORMAL` | Expected baseline activity in patient room. |
| Sitting on bed edge $> 300\text{s}$ (5 min) | `MONITOR` | Prolonged edge-sitting indicates hesitation, fatigue, or imminent unassisted exit risk. |
| `UNKNOWN` sustained $> 60\text{s}$ (1 min) | `MONITOR` | Prolonged sensor occlusion requires staff check-in. |
| Confirmed bed exit, out of bed $> 600\text{s}$ (10 min) | `ALERT` | Elderly patient unassisted out of bed for extended period presents high fall/wandering risk. |
| Horizontal posture detected off-bed | `ALERT` | High-priority fall detection emergency. |
| Bed exit with no return by observation end | `ALERT` | Patient has not returned to bed safely. |
| Caregiver present in room | **Suppress 1 Level** | Assisted patient movement: `ALERT` $\to$ `MONITOR`, `MONITOR` $\to$ `NORMAL`. |

---

## 6. Model Choices & Local Docker Setup

- **Vision**: `yolo11n-pose.pt` (17 COCO keypoints + ByteTrack) provides single-pass detection, tracking, and pose at $>60$ FPS.
- **Agent LLM**: Qwen (e.g. `qwen2.5:3b` / `qwen3:4b` via Ollama) handles narrow, structured decision-making over pre-computed signals.
- **VLM Tool**: `qwen2.5-vl:3b` (via Ollama) is called on-demand when visual ambiguity cannot be resolved by geometry alone (e.g., blanket occlusion).

---

## 7. Quickstart & How to Run

### Option A: Using Docker Compose (Recommended)

```bash
# 1. Start Ollama and BedSense Application
docker compose up -d

# 2. Pull Qwen models into Ollama container
docker exec bedsense_ollama ollama pull qwen2.5:3b
docker exec bedsense_ollama ollama pull qwen2.5-vl:3b
```

**Run CLI in PowerShell (Windows):**
```powershell
docker exec elderly_vision_app python -m src.cli `
  --video old/data/sample1.mp4 `
  --output-dir outputs/sample1 `
  --annotated-video outputs/sample1/annotated.mp4
```

*Or as a single line in PowerShell / CMD:*
```powershell
docker exec elderly_vision_app python -m src.cli --video old/data/sample1.mp4 --output-dir outputs/sample1 --annotated-video outputs/sample1/annotated.mp4
```

**Run CLI in Bash (Linux/macOS):**
```bash
docker exec elderly_vision_app python -m src.cli \
  --video old/data/sample1.mp4 \
  --output-dir outputs/sample1 \
  --annotated-video outputs/sample1/annotated.mp4
```

### Option B: Interactive Jupyter Notebook

Launch JupyterLab at `http://localhost:8888` and open [`run.ipynb`](run.ipynb) to execute each stage interactively with visual HUD previews.

### Option C: Run Unit Tests

```bash
docker exec elderly_vision_app pytest tests/ -v
```

---

## 8. Output Formats

The pipeline generates outputs matching §6.2:

- **`events.json`**:
```json
[
  {
    "event": "bed_exit",
    "start_time": "00:05:08",
    "confirmed_time": "00:05:20",
    "previous_state": "sitting_on_bed",
    "current_state": "walking",
    "confidence": 0.92,
    "decision": "MONITOR"
  }
]
```

- **`summary.json`**:
```json
{
  "observation_duration_sec": 1200,
  "activity_duration_sec": {
    "lying_in_bed": 702,
    "sitting_on_bed": 128,
    "sitting_outside_bed": 95,
    "standing": 63,
    "walking": 167,
    "unknown": 45
  },
  "bed_exit_count": 2,
  "bed_return_count": 2,
  "total_in_bed_sec": 830,
  "total_out_of_bed_sec": 370,
  "longest_out_of_bed_period_sec": 241,
  "final_state": "lying_in_bed"
}
```

- **`timeline.json`**:
```json
[
  {"start": "00:00:00", "end": "00:04:32", "state": "LYING_IN_BED"}
]
```

*Invariant Guarantee*: $\sum \text{activity\_duration\_sec} \equiv \text{observation\_duration\_sec}$.

---

## 9. Failure Case Analysis (§11)

1. **Heavy Blanket / Quilt Occlusion**:
   - *Problem*: Blankets cover lower limbs, reducing knee/ankle keypoint confidences.
   - *Mitigation*: Trigger 1 flags segment as ambiguous (`mean_kpt_conf < 0.40`). The Agent LLM inspects surrounding context and invokes `vlm_describe` to confirm patient presence.
2. **Caregiver Occlusion / ID Switching**:
   - *Problem*: Caregiver assisting bed exit occludes patient, leading to potential track ID switching.
   - *Mitigation*: Trigger 5 detects multi-track presence in segment; Policy engine suppresses alert escalation by one level.
3. **Bed Edge Hovering**:
   - *Problem*: Borderline overlap ($0.40 - 0.60$) during edge sitting.
   - *Mitigation*: Trigger 3 flags boundary hovering and applies `EDGE_SIT_MONITOR_SEC` timer for proactive fall prevention.

---

## 10. Future Improvements with More Time

- **3D Pose & Ground Plane Estimation**: Integrate monocular depth / 3D pose to compute metric vertical velocity for instantaneous fall impact detection.
- **Edge TensorRT Compilation**: Export YOLO-Pose to TensorRT FP16 / INT8 engines to run real-time inference on edge devices (Jetson / Orin).
- **Audio Multimodal Fusion**: Integrate acoustic event detection (cries for help, bed alarm beeps) into the Agent decision loop.
