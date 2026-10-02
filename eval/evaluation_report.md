# BedSense AI — Clinical Evaluation & Benchmark Report

> [!IMPORTANT]
> Comprehensive benchmark evaluating **Activity Recognition**, **Bed Events (Precision & Recall)**, and **Duration Estimation** across ground-truth annotated clinical sequences.

## 1. Executive Benchmark Summary

| Metric | Benchmark Result | Clinical Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **State Classification Accuracy** | **81.4%** | ≥ 90.0% | ⚠️ MONITOR |
| **Macro Average F1-Score** | **62.1%** | ≥ 85.0% | ⚠️ MONITOR |
| **Bed-Exit Precision** | **100.0%** | ≥ 90.0% | ✅ PASS |
| **Bed-Exit Recall** | **100.0%** | ≥ 90.0% | ✅ PASS |
| **Mean Duration Absolute Error** | **3.9 sec** | ≤ 5.0 sec | ✅ PASS |

---

## 2. Sample Benchmark: `sample1`

- **Ground Truth Source**: `/workspace/data/sample1.json`
- **Predictions Directory**: `/workspace/outputs/sample1`
- **Observation Duration**: `00:04:50` (290s)

### 2.1 Activity Recognition Performance

| State | Precision | Recall | F1-Score | GT Support (Duration) |
| :--- | :--- | :--- | :--- | :--- |
| **Lying In Bed** | 82.0% | 90.9% | 86.2% | 145.3s |
| **Sitting On Bed** | 11.4% | 6.4% | 8.2% | 26.7s |
| **Walking** | 92.7% | 59.8% | 72.7% | 12.7s |
| **Unknown** | 99.1% | 99.4% | 99.2% | 105.8s |

**Overall Classification Accuracy**: `84.89%`  
**Macro F1-Score**: `66.59%` | **Weighted F1-Score**: `83.19%`

#### Confusion Matrix (Time in Seconds)

| True \ Pred | **Lying** | **Sitting** | **Walking** | **Unknown** |
| :--- |  :---: | :---: | :---: | :---: |
| **Lying** | 132.1s | 13.2s | 0.0s | 0.0s |
| **Sitting** | 25.0s | 1.7s | 0.0s | 0.0s |
| **Walking** | 4.1s | 0.0s | 7.6s | 1.0s |
| **Unknown** | 0.0s | 0.0s | 0.6s | 105.2s |
### 2.2 Clinical Bed Events

| Event Type | Ground Truth Count | Predicted Count | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bed Exit** | 2 | 2 | 2 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Bed Return** | 2 | 2 | 2 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Missing** | 2 | 2 | 2 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |

### 2.3 Duration Estimation Comparison

| Activity State | Ground Truth (HH:MM:SS) | Predicted (HH:MM:SS) | Duration Error (sec) | Relative Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Lying In Bed** | `00:02:25` (145s) | `00:02:43` (163s) | **18 sec** | 12.2% |
| **Sitting On Bed** | `00:00:27` (27s) | `00:00:13` (13s) | **14 sec** | 51.4% |
| **Walking** | `00:00:13` (13s) | `00:00:09` (9s) | **4 sec** | 29.0% |
| **Unknown** | `00:01:46` (106s) | `00:01:46` (106s) | **0 sec** | 0.2% |
| **Total Time In Bed** | `00:02:52` (172s) | `00:02:56` (176s) | **4 sec** | 2.3% |
| **Total Time Out Of Bed** | `00:01:58` (118s) | `00:01:55` (115s) | **3 sec** | 2.9% |

**Mean Absolute Duration Error across classes**: `5.90 seconds`

---

## 3. Sample Benchmark: `sample2`

- **Ground Truth Source**: `/workspace/data/sample2.json`
- **Predictions Directory**: `/workspace/outputs/sample2`
- **Observation Duration**: `00:00:39` (39s)

### 2.1 Activity Recognition Performance

| State | Precision | Recall | F1-Score | GT Support (Duration) |
| :--- | :--- | :--- | :--- | :--- |
| **Lying In Bed** | 79.4% | 99.1% | 88.2% | 23.3s |
| **Sitting On Bed** | 96.6% | 67.9% | 79.7% | 8.4s |
| **Walking** | 14.3% | 10.8% | 12.3% | 3.7s |
| **Unknown** | 100.0% | 33.3% | 50.0% | 3.6s |

**Overall Classification Accuracy**: `77.95%`  
**Macro F1-Score**: `57.55%` | **Weighted F1-Score**: `75.63%`

#### Confusion Matrix (Time in Seconds)

| True \ Pred | **Lying** | **Sitting** | **Walking** | **Unknown** |
| :--- |  :---: | :---: | :---: | :---: |
| **Lying** | 23.1s | 0.2s | 0.0s | 0.0s |
| **Sitting** | 2.7s | 5.7s | 0.0s | 0.0s |
| **Walking** | 3.3s | 0.0s | 0.4s | 0.0s |
| **Unknown** | 0.0s | 0.0s | 2.4s | 1.2s |
### 2.2 Clinical Bed Events

| Event Type | Ground Truth Count | Predicted Count | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bed Exit** | 1 | 1 | 1 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Bed Return** | 1 | 1 | 1 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |
| **Missing** | 1 | 1 | 1 | 0 | 0 | **100.0%** | **100.0%** | **100.0%** |

### 2.3 Duration Estimation Comparison

| Activity State | Ground Truth (HH:MM:SS) | Predicted (HH:MM:SS) | Duration Error (sec) | Relative Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Lying In Bed** | `00:00:23` (23s) | `00:00:29` (29s) | **6 sec** | 24.8% |
| **Sitting On Bed** | `00:00:08` (8s) | `00:00:06` (6s) | **2 sec** | 28.6% |
| **Walking** | `00:00:04` (4s) | `00:00:03` (3s) | **1 sec** | 19.1% |
| **Unknown** | `00:00:04` (4s) | `00:00:01` (1s) | **3 sec** | 72.5% |
| **Total Time In Bed** | `00:00:32` (32s) | `00:00:35` (35s) | **3 sec** | 10.7% |
| **Total Time Out Of Bed** | `00:00:07` (7s) | `00:00:04` (4s) | **3 sec** | 45.5% |

**Mean Absolute Duration Error across classes**: `1.92 seconds`

---
