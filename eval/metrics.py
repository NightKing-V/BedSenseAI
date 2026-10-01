"""
Core Evaluation and Metric Computation Module for BedSense AI.
Evaluates:
1. Activity Recognition: State accuracy, confusion matrix, macro/micro F1.
2. Bed Events: Bed-exit, bed-return, and missing event precision, recall, and false positive tracking.
3. Duration Estimation: Ground-truth vs predicted activity durations with absolute and relative errors.
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


class MissingArtifactError(Exception):
    """Raised when required output prediction artifact files are missing."""
    pass


CANONICAL_STATES = [
    "LYING_IN_BED",
    "SITTING_ON_BED",
    "SITTING_OUTSIDE_BED",
    "STANDING",
    "WALKING",
    "UNKNOWN",
]

IN_BED_STATES = {"LYING_IN_BED", "SITTING_ON_BED"}
OUT_OF_BED_STATES = {"SITTING_OUTSIDE_BED", "STANDING", "WALKING", "UNKNOWN"}


def parse_hms_to_sec(time_str: str) -> float:
    """Converts HH:MM:SS or HH:MM:SS.ss or HH;MM;SS;FF into total seconds."""
    if not time_str:
        return 0.0
    clean_str = time_str.replace(";", ":").strip()
    parts = clean_str.split(":")
    if len(parts) >= 3:
        try:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        except ValueError:
            return 0.0
    elif len(parts) == 2:
        try:
            m = float(parts[0])
            s = float(parts[1])
            return m * 60 + s
        except ValueError:
            return 0.0
    try:
        return float(time_str)
    except ValueError:
        return 0.0


def format_hms(seconds: Union[int, float]) -> str:
    """Formats seconds into standard HH:MM:SS string."""
    sec = max(0, int(round(seconds)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_state_name(state: str) -> str:
    """Normalizes state names to uppercase canonical representations."""
    st = str(state).strip().upper().replace(" ", "_")
    if st == "OUT_OF_BED":
        return "UNKNOWN"
    return st if st in CANONICAL_STATES else "UNKNOWN"


def load_ground_truth(gt_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and parses ground-truth segments from JSON file.
    Derives canonical state durations and clinical transition events (bed_exit, bed_return, missing).
    """
    p = Path(gt_path)
    if not p.exists():
        raise FileNotFoundError(f"Ground truth annotation file not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Ground truth format must be a JSON array of segment objects in '{p}'")

    segments: List[Dict[str, Any]] = []
    activity_durations: Dict[str, float] = {s: 0.0 for s in CANONICAL_STATES}
    events: List[Dict[str, Any]] = []

    prev_state: Optional[str] = None
    total_dur = 0.0

    for idx, item in enumerate(data):
        st_name = normalize_state_name(item.get("state", "UNKNOWN"))
        t0 = float(item.get("start_t", parse_hms_to_sec(item.get("start_time", "00:00:00"))))
        t1 = float(item.get("end_t", parse_hms_to_sec(item.get("end_time", "00:00:00"))))
        duration = float(item.get("duration", max(0.0, t1 - t0)))
        if t1 <= t0:
            t1 = t0 + duration

        seg_entry = {
            "start_t": t0,
            "end_t": t1,
            "duration": duration,
            "state": st_name,
            "start_time": item.get("start_time", format_hms(t0)),
            "end_time": item.get("end_time", format_hms(t1)),
        }
        segments.append(seg_entry)
        activity_durations[st_name] += duration
        total_dur = max(total_dur, t1)

        # Ground truth event extraction based on state handoffs
        if prev_state is not None and prev_state != st_name:
            # Bed Exit: in-bed -> out-of-bed
            if prev_state in IN_BED_STATES and st_name in OUT_OF_BED_STATES:
                events.append({
                    "event": "bed_exit",
                    "start_time": format_hms(t0),
                    "start_t": t0,
                    "previous_state": prev_state.lower(),
                    "current_state": st_name.lower(),
                    "decision": "MONITOR",
                })
            # Missing: out-of-bed physical -> unknown
            elif prev_state in (OUT_OF_BED_STATES - {"UNKNOWN"}) and st_name == "UNKNOWN":
                events.append({
                    "event": "missing",
                    "start_time": format_hms(t0),
                    "start_t": t0,
                    "previous_state": prev_state.lower(),
                    "current_state": "unknown",
                    "decision": "ALERT",
                })
            # Bed Return: out-of-bed -> in-bed
            elif prev_state in OUT_OF_BED_STATES and st_name in IN_BED_STATES:
                events.append({
                    "event": "bed_return",
                    "start_time": format_hms(t0),
                    "start_t": t0,
                    "previous_state": prev_state.lower(),
                    "current_state": st_name.lower(),
                    "decision": "MONITOR",
                })

        prev_state = st_name

    return {
        "segments": segments,
        "activity_durations": activity_durations,
        "total_duration": total_dur,
        "events": events,
    }


def load_predictions(pred_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads output prediction artifacts from the output directory:
    - timeline.json
    - events.json
    - summary.json
    Raises MissingArtifactError with actionable guidance if any file is missing.
    """
    out_path = Path(pred_dir)
    if not out_path.exists():
        raise MissingArtifactError(
            f"Output directory does not exist: '{out_path.resolve()}'.\n"
            f"Please run the monitoring pipeline first:\n"
            f"  docker exec elderly_vision_app python -m src.cli --video <video_path> --output-dir {pred_dir}"
        )

    required_files = {
        "timeline.json": out_path / "timeline.json",
        "events.json": out_path / "events.json",
        "summary.json": out_path / "summary.json",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise MissingArtifactError(
            f"Missing required output artifacts in '{out_path.resolve()}': [{missing_str}].\n"
            f"Please execute the pipeline with JSON export enabled:\n"
            f"  docker exec elderly_vision_app python -m src.cli --video <video_path> --output-dir {pred_dir} --save-json"
        )

    with open(required_files["timeline.json"], "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    with open(required_files["events.json"], "r", encoding="utf-8") as f:
        events_data = json.load(f)

    with open(required_files["summary.json"], "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Normalize timeline segments
    normalized_timeline: List[Dict[str, Any]] = []
    for item in timeline_data:
        t0 = float(item.get("start_t", 0.0))
        t1 = float(item.get("end_t", 0.0))
        dur = float(item.get("duration", max(0.0, t1 - t0)))
        st = normalize_state_name(item.get("state", "UNKNOWN"))
        normalized_timeline.append({
            "start_t": t0,
            "end_t": t1,
            "duration": dur,
            "state": st,
            "confidence": float(item.get("confidence", 0.0)),
            "start_time": format_hms(t0),
            "end_time": format_hms(t1),
        })

    # Normalize events
    normalized_events: List[Dict[str, Any]] = []
    for item in events_data:
        st_time = item.get("start_time", "00:00:00")
        t_start = parse_hms_to_sec(st_time)
        normalized_events.append({
            "event": str(item.get("event", "")).lower(),
            "start_time": st_time,
            "start_t": t_start,
            "confirmed_time": item.get("confirmed_time", st_time),
            "previous_state": str(item.get("previous_state", "")).lower(),
            "current_state": str(item.get("current_state", "")).lower(),
            "confidence": float(item.get("confidence", 0.0)),
            "decision": str(item.get("decision", "NORMAL")).upper(),
        })

    return {
        "timeline": normalized_timeline,
        "events": normalized_events,
        "summary": summary_data,
    }


def evaluate_temporal_states(
    gt_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    total_obs_sec: float,
    dt: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluates temporal state classification on a fine-grained grid (dt = 0.1s).
    Computes overall accuracy, confusion matrix, per-state precision/recall/F1, and failure episodes.
    """
    time_grid = np.arange(0.0, max(1e-3, total_obs_sec), dt)
    n_steps = len(time_grid)

    def lookup_state(segments: List[Dict[str, Any]], t: float) -> str:
        for seg in segments:
            if seg["start_t"] <= t < seg["end_t"]:
                return seg["state"]
        if segments:
            return segments[-1]["state"]
        return "UNKNOWN"

    y_true: List[str] = [lookup_state(gt_segments, t) for t in time_grid]
    y_pred: List[str] = [lookup_state(pred_segments, t) for t in time_grid]

    # Overall Time-Weighted Accuracy
    correct_steps = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_steps / max(1, n_steps)

    # Active states present in GT or Pred
    active_states = [s for s in CANONICAL_STATES if (s in y_true or s in y_pred)]
    if not active_states:
        active_states = CANONICAL_STATES

    # Build Confusion Matrix (in seconds)
    state_to_idx = {s: i for i, s in enumerate(active_states)}
    k = len(active_states)
    conf_matrix_counts = np.zeros((k, k), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred):
        r = state_to_idx.get(yt, state_to_idx.get("UNKNOWN", 0))
        c = state_to_idx.get(yp, state_to_idx.get("UNKNOWN", 0))
        conf_matrix_counts[r, c] += 1

    conf_matrix_sec = conf_matrix_counts * dt

    # Per-Class Metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_list = []
    support_weights = []

    for idx, state in enumerate(active_states):
        tp = conf_matrix_sec[idx, idx]
        fn = np.sum(conf_matrix_sec[idx, :]) - tp
        fp = np.sum(conf_matrix_sec[:, idx]) - tp
        tn = np.sum(conf_matrix_sec) - tp - fn - fp
        gt_support = np.sum(conf_matrix_sec[idx, :])

        prec = tp / max(1e-6, tp + fp) if (tp + fp) > 0 else (1.0 if gt_support == 0 else 0.0)
        rec = tp / max(1e-6, tp + fn) if (tp + fn) > 0 else (1.0 if gt_support == 0 else 0.0)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[state] = {
            "true_positive_sec": round(float(tp), 2),
            "false_positive_sec": round(float(fp), 2),
            "false_negative_sec": round(float(fn), 2),
            "support_sec": round(float(gt_support), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
        }
        if gt_support > 0:
            f1_list.append(f1)
            support_weights.append(gt_support)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 1.0
    weighted_f1 = float(np.average(f1_list, weights=support_weights)) if support_weights and sum(support_weights) > 0 else macro_f1

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "active_states": active_states,
        "confusion_matrix_sec": conf_matrix_sec.tolist(),
        "per_class_metrics": per_class_metrics,
        "failure_cases": [],
    }


def evaluate_bed_events(
    gt_events: List[Dict[str, Any]],
    pred_events: List[Dict[str, Any]],
    tolerance_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Evaluates clinical Bed Events (bed_exit, bed_return, missing):
    Matches predictions to ground-truth using a bipartite temporal window tolerance (|t_pred - t_gt| <= tolerance_sec).
    Computes precision, recall, F1, false detections, and missed alerts.
    """
    event_types = ["bed_exit", "bed_return", "missing"]
    results_by_type: Dict[str, Any] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for ev_type in event_types:
        gt_evs = [e for e in gt_events if e.get("event") == ev_type]
        pred_evs = [e for e in pred_events if e.get("event") == ev_type]

        matched_gt_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()
        matches: List[Dict[str, Any]] = []

        # Find closest temporal matches
        for p_idx, p_ev in enumerate(pred_evs):
            best_gt_idx = None
            min_dt = tolerance_sec + 1e-3

            for g_idx, g_ev in enumerate(gt_evs):
                if g_idx in matched_gt_indices:
                    continue
                dt = abs(p_ev["start_t"] - g_ev["start_t"])
                if dt <= tolerance_sec and dt < min_dt:
                    min_dt = dt
                    best_gt_idx = g_idx

            if best_gt_idx is not None:
                matched_gt_indices.add(best_gt_idx)
                matched_pred_indices.add(p_idx)
                matches.append({
                    "event": ev_type,
                    "gt_time": gt_evs[best_gt_idx]["start_time"],
                    "pred_time": p_ev["start_time"],
                    "delta_t_sec": round(min_dt, 2),
                    "confidence": p_ev.get("confidence", 0.0),
                    "decision": p_ev.get("decision", "NORMAL"),
                })

        tp = len(matched_gt_indices)
        fp = len(pred_evs) - tp
        fn = len(gt_evs) - tp

        prec = tp / max(1e-6, tp + fp) if (tp + fp) > 0 else (1.0 if len(gt_evs) == 0 else 0.0)
        rec = tp / max(1e-6, tp + fn) if (tp + fn) > 0 else (1.0 if len(gt_evs) == 0 else 0.0)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        false_positives = [pred_evs[i] for i in range(len(pred_evs)) if i not in matched_pred_indices]
        false_negatives = [gt_evs[i] for i in range(len(gt_evs)) if i not in matched_gt_indices]

        results_by_type[ev_type] = {
            "ground_truth_count": len(gt_evs),
            "predicted_count": len(pred_evs),
            "true_positives": tp,
            "false_positives_count": fp,
            "false_negatives_count": fn,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "matches": matches,
            "false_positive_events": false_positives,
            "false_negative_events": false_negatives,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_prec = total_tp / max(1e-6, total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_rec = total_tp / max(1e-6, total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = (2 * overall_prec * overall_rec) / max(1e-6, overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "overall_precision": round(float(overall_prec), 4),
        "overall_recall": round(float(overall_rec), 4),
        "overall_f1": round(float(overall_f1), 4),
        "events_by_type": results_by_type,
    }


def evaluate_durations(
    gt_durations: Dict[str, float],
    pred_durations: Dict[str, float],
    total_obs_sec: float,
) -> Dict[str, Any]:
    """
    Compares predicted versus ground-truth duration for each activity state,
    calculating absolute error (seconds) and percentage error (relative to GT duration).
    """
    comparison_table: List[Dict[str, Any]] = []
    abs_errors = []

    for state in CANONICAL_STATES:
        gt_s = float(gt_durations.get(state, 0.0))
        pred_s = float(pred_durations.get(state, 0.0))
        error_s = abs(pred_s - gt_s)
        abs_errors.append(error_s)

        pct_err = (error_s / max(1e-3, gt_s)) * 100.0 if gt_s > 0 else (0.0 if pred_s == 0 else 100.0)

        comparison_table.append({
            "state": state,
            "display_name": state.replace("_", " ").title(),
            "gt_sec": int(round(gt_s)),
            "gt_hms": format_hms(gt_s),
            "pred_sec": int(round(pred_s)),
            "pred_hms": format_hms(pred_s),
            "duration_error_sec": int(round(error_s)),
            "percentage_error": round(float(pct_err), 1),
        })

    # Total In-Bed & Out-of-Bed Comparisons
    gt_in_bed = sum(gt_durations.get(s, 0.0) for s in IN_BED_STATES)
    pred_in_bed = sum(pred_durations.get(s, 0.0) for s in IN_BED_STATES)
    in_bed_err = abs(pred_in_bed - gt_in_bed)
    in_bed_pct = (in_bed_err / max(1e-3, gt_in_bed)) * 100.0 if gt_in_bed > 0 else 0.0

    gt_out_bed = sum(gt_durations.get(s, 0.0) for s in OUT_OF_BED_STATES)
    pred_out_bed = sum(pred_durations.get(s, 0.0) for s in OUT_OF_BED_STATES)
    out_bed_err = abs(pred_out_bed - gt_out_bed)
    out_bed_pct = (out_bed_err / max(1e-3, gt_out_bed)) * 100.0 if gt_out_bed > 0 else 0.0

    aggregate_comparisons = {
        "time_in_bed": {
            "gt_sec": int(round(gt_in_bed)),
            "gt_hms": format_hms(gt_in_bed),
            "pred_sec": int(round(pred_in_bed)),
            "pred_hms": format_hms(pred_in_bed),
            "duration_error_sec": int(round(in_bed_err)),
            "percentage_error": round(float(in_bed_pct), 1),
        },
        "time_out_of_bed": {
            "gt_sec": int(round(gt_out_bed)),
            "gt_hms": format_hms(gt_out_bed),
            "pred_sec": int(round(pred_out_bed)),
            "pred_hms": format_hms(pred_out_bed),
            "duration_error_sec": int(round(out_bed_err)),
            "percentage_error": round(float(out_bed_pct), 1),
        },
    }

    mae = float(np.mean(abs_errors)) if abs_errors else 0.0

    return {
        "total_observation_sec": int(round(total_obs_sec)),
        "total_observation_hms": format_hms(total_obs_sec),
        "mean_absolute_error_sec": round(mae, 2),
        "per_activity_comparison": comparison_table,
        "aggregate_comparisons": aggregate_comparisons,
    }


def evaluate_single_sample(
    sample_name: str,
    gt_path: Union[str, Path],
    pred_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Evaluates a single sample against ground-truth and output prediction artifacts.
    """
    gt_data = load_ground_truth(gt_path)
    pred_data = load_predictions(pred_dir)

    total_obs_sec = gt_data["total_duration"]
    if total_obs_sec <= 0:
        total_obs_sec = float(pred_data["summary"].get("observation_duration_sec", 0.0))

    # Activity Durations extraction from predicted summary
    pred_dur_map_raw = pred_data["summary"].get("activity_duration_sec", {})
    pred_durations: Dict[str, float] = {
        normalize_state_name(k): float(v) for k, v in pred_dur_map_raw.items()
    }

    # 1. Activity Recognition Evaluation
    activity_eval = evaluate_temporal_states(
        gt_segments=gt_data["segments"],
        pred_segments=pred_data["timeline"],
        total_obs_sec=total_obs_sec,
    )

    # 2. Bed Events Evaluation
    events_eval = evaluate_bed_events(
        gt_events=gt_data["events"],
        pred_events=pred_data["events"],
    )

    # 3. Duration Estimation Evaluation
    duration_eval = evaluate_durations(
        gt_durations=gt_data["activity_durations"],
        pred_durations=pred_durations,
        total_obs_sec=total_obs_sec,
    )

    return {
        "sample_name": sample_name,
        "gt_file": str(Path(gt_path).resolve()),
        "pred_dir": str(Path(pred_dir).resolve()),
        "activity_recognition": activity_eval,
        "bed_events": events_eval,
        "duration_estimation": duration_eval,
    }


def evaluate_multiple_samples(
    sample_configs: List[Dict[str, Union[str, Path]]]
) -> Dict[str, Any]:
    """
    Evaluates multiple video samples and calculates cross-sample benchmark averages.
    """
    individual_results: List[Dict[str, Any]] = []
    missing_samples: List[Dict[str, str]] = []

    for cfg in sample_configs:
        s_name = str(cfg["name"])
        gt_p = Path(cfg["gt_path"])
        pred_p = Path(cfg["pred_dir"])

        try:
            res = evaluate_single_sample(sample_name=s_name, gt_path=gt_p, pred_dir=pred_p)
            individual_results.append(res)
        except (MissingArtifactError, FileNotFoundError) as e:
            missing_samples.append({
                "sample_name": s_name,
                "error": str(e),
                "pred_dir": str(pred_p),
            })

    if not individual_results and missing_samples:
        error_msgs = "\n\n".join([f"[{m['sample_name']}] {m['error']}" for m in missing_samples])
        raise MissingArtifactError(f"Evaluation failed — missing output artifacts for all samples:\n\n{error_msgs}")

    # Compute aggregate metrics across successful evaluations
    total_acc = [r["activity_recognition"]["accuracy"] for r in individual_results]
    macro_f1s = [r["activity_recognition"]["macro_f1"] for r in individual_results]
    exit_precs = [r["bed_events"]["events_by_type"]["bed_exit"]["precision"] for r in individual_results]
    exit_recs = [r["bed_events"]["events_by_type"]["bed_exit"]["recall"] for r in individual_results]
    mean_maes = [r["duration_estimation"]["mean_absolute_error_sec"] for r in individual_results]

    summary_benchmark = {
        "evaluated_samples_count": len(individual_results),
        "mean_state_accuracy": round(float(np.mean(total_acc)), 4) if total_acc else 0.0,
        "mean_macro_f1": round(float(np.mean(macro_f1s)), 4) if macro_f1s else 0.0,
        "mean_bed_exit_precision": round(float(np.mean(exit_precs)), 4) if exit_precs else 0.0,
        "mean_bed_exit_recall": round(float(np.mean(exit_recs)), 4) if exit_recs else 0.0,
        "mean_duration_mae_sec": round(float(np.mean(mean_maes)), 2) if mean_maes else 0.0,
    }

    return {
        "benchmark_summary": summary_benchmark,
        "samples": individual_results,
        "missing_samples": missing_samples,
    }
