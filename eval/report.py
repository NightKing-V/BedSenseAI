"""
Evaluation Report Formatter and Markdown Generator for BedSense AI.
Renders rich terminal evaluation summaries and comprehensive GitHub Markdown reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Union


CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_terminal_evaluation(eval_data: Dict[str, Any]):
    """Prints a beautiful, comprehensive terminal report adhering to the evaluation specification."""
    if "samples" in eval_data:
        samples = eval_data["samples"]
        summary = eval_data.get("benchmark_summary", {})
        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║         BEDSENSE AI — MULTI-DATASET CLINICAL BENCHMARK EVALUATION                    ║{RESET}")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")
        print(f"Evaluated Samples:         {summary.get('evaluated_samples_count', len(samples))}")
        print(f"Mean State Accuracy:       {GREEN}{summary.get('mean_state_accuracy', 0.0)*100:.1f}%{RESET}")
        print(f"Mean Macro F1-Score:       {GREEN}{summary.get('mean_macro_f1', 0.0)*100:.1f}%{RESET}")
        print(f"Mean Bed-Exit Precision:   {GREEN}{summary.get('mean_bed_exit_precision', 0.0)*100:.1f}%{RESET}")
        print(f"Mean Bed-Exit Recall:      {GREEN}{summary.get('mean_bed_exit_recall', 0.0)*100:.1f}%{RESET}")
        print(f"Mean Duration MAE:         {GREEN}{summary.get('mean_duration_mae_sec', 0.0):.1f} sec{RESET}")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}\n")

        for sample_eval in samples:
            _print_single_sample_terminal(sample_eval)
    else:
        _print_single_sample_terminal(eval_data)


def _print_single_sample_terminal(sample_eval: Dict[str, Any]):
    s_name = sample_eval["sample_name"]
    act = sample_eval["activity_recognition"]
    evs = sample_eval["bed_events"]
    dur = sample_eval["duration_estimation"]

    print(f"\n{CYAN}{BOLD}================================================================================{RESET}")
    print(f"{CYAN}{BOLD}EVALUATION REPORT: {s_name.upper()} (Duration: {dur['total_observation_hms']}){RESET}")
    print(f"{CYAN}{BOLD}================================================================================{RESET}")

    # 1. Activity Recognition
    print(f"\n{BOLD}1. ACTIVITY RECOGNITION{RESET}")
    print(f"  • State Classification Accuracy: {GREEN}{BOLD}{act['accuracy']*100:.1f}%{RESET}")
    print(f"  • Macro F1-Score:                {GREEN}{act['macro_f1']*100:.1f}%{RESET}")
    print(f"  • Weighted F1-Score:             {GREEN}{act['weighted_f1']*100:.1f}%{RESET}")

    print(f"\n  {BOLD}Per-State Classification Breakdown:{RESET}")
    print(f"  {'State':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("  " + "-" * 70)
    for st, m in act["per_class_metrics"].items():
        if m["support_sec"] > 0 or m["false_positive_sec"] > 0:
            print(f"  {st.replace('_', ' ').title():<22} | {m['precision']*100:6.1f}%   | {m['recall']*100:6.1f}%   | {m['f1_score']*100:6.1f}%   | {m['support_sec']:>6.1f}s")

    # Confusion Matrix
    print(f"\n  {BOLD}Confusion Matrix Between States (in Seconds):{RESET}")
    active_states = act["active_states"]
    short_names = [s.replace("_IN_BED", "").replace("_ON_BED", "").title()[:8] for s in active_states]
    title_lbl = "True \\ Pred"
    header = "  " + f"{title_lbl:<14} | " + " | ".join([f"{n:>8}" for n in short_names])
    print(header)
    print("  " + "-" * len(header))
    for r_idx, r_state in enumerate(active_states):
        row_vals = act["confusion_matrix_sec"][r_idx]
        row_str = " | ".join([f"{v:8.1f}" for v in row_vals])
        print(f"  {short_names[r_idx]:<14} | {row_str}")

    # 2. Bed Events
    print(f"\n{BOLD}2. BED EVENTS EVALUATION{RESET}")
    exit_m = evs["events_by_type"]["bed_exit"]
    return_m = evs["events_by_type"]["bed_return"]
    missing_m = evs["events_by_type"]["missing"]

    print(f"  • Bed-Exit Precision:            {GREEN}{BOLD}{exit_m['precision']*100:.1f}%{RESET} ({exit_m['true_positives']}/{exit_m['predicted_count']} detected)")
    print(f"  • Bed-Exit Recall:               {GREEN}{BOLD}{exit_m['recall']*100:.1f}%{RESET} ({exit_m['true_positives']}/{exit_m['ground_truth_count']} ground truth)")
    print(f"  • False Bed-Exit Detections:     {GREEN if exit_m['false_positives_count'] == 0 else RED}{exit_m['false_positives_count']}{RESET}")
    print(f"  • Bed-Return Precision / Recall: {return_m['precision']*100:.1f}% / {return_m['recall']*100:.1f}%")
    print(f"  • Missing Alert Precision/Recall:{missing_m['precision']*100:.1f}% / {missing_m['recall']*100:.1f}%")

    if exit_m["false_positive_events"]:
        print(f"    {YELLOW}False Bed-Exit Events Details:{RESET}")
        for fp_ev in exit_m["false_positive_events"]:
            print(f"      - Time: {fp_ev['start_time']} | Conf: {fp_ev['confidence']*100:.0f}% | Decision: {fp_ev['decision']}")

    # 3. Duration Estimation
    print(f"\n{BOLD}3. DURATION ESTIMATION (Ground Truth vs. Predicted){RESET}")
    print(f"  {'Activity State':<24} | {'Ground Truth':<16} | {'Predicted':<16} | {'Duration Error':<18}")
    print("  " + "-" * 82)
    for row in dur["per_activity_comparison"]:
        if row["gt_sec"] > 0 or row["pred_sec"] > 0:
            err_str = f"{row['duration_error_sec']:>3} sec ({row['percentage_error']:4.1f}%)"
            print(f"  {row['display_name']:<24} | {row['gt_hms']} ({row['gt_sec']:>3}s)   | {row['pred_hms']} ({row['pred_sec']:>3}s)   | {err_str:<18}")

    print("  " + "-" * 82)
    in_bed = dur["aggregate_comparisons"]["time_in_bed"]
    out_bed = dur["aggregate_comparisons"]["time_out_of_bed"]
    print(f"  {'Time In Bed':<24} | {in_bed['gt_hms']} ({in_bed['gt_sec']:>3}s)   | {in_bed['pred_hms']} ({in_bed['pred_sec']:>3}s)   | {in_bed['duration_error_sec']:>3} sec ({in_bed['percentage_error']:4.1f}%)")
    print(f"  {'Time Out Of Bed':<24} | {out_bed['gt_hms']} ({out_bed['gt_sec']:>3}s)   | {out_bed['pred_hms']} ({out_bed['pred_sec']:>3}s)   | {out_bed['duration_error_sec']:>3} sec ({out_bed['percentage_error']:4.1f}%)")
    print(f"  • Mean Absolute Duration Error: {GREEN}{dur['mean_absolute_error_sec']:.1f} sec{RESET}\n")


def generate_markdown_report(eval_data: Dict[str, Any], output_file: Union[str, Path]) -> str:
    """Generates a complete, publication-grade GitHub Markdown evaluation report."""
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    samples = eval_data.get("samples", [eval_data]) if "samples" in eval_data else [eval_data]
    benchmark_summary = eval_data.get("benchmark_summary", {})

    lines: List[str] = [
        "# BedSense AI — Clinical Evaluation & Benchmark Report",
        "",
        "> [!IMPORTANT]",
        "> Comprehensive benchmark evaluating **Activity Recognition**, **Bed Events (Precision & Recall)**, and **Duration Estimation** across ground-truth annotated clinical sequences.",
        "",
        "## 1. Executive Benchmark Summary",
        "",
        "| Metric | Benchmark Result | Clinical Benchmark Target | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    mean_acc = benchmark_summary.get("mean_state_accuracy", samples[0]["activity_recognition"]["accuracy"]) if samples else 0.0
    mean_f1 = benchmark_summary.get("mean_macro_f1", samples[0]["activity_recognition"]["macro_f1"]) if samples else 0.0
    mean_exit_p = benchmark_summary.get("mean_bed_exit_precision", samples[0]["bed_events"]["events_by_type"]["bed_exit"]["precision"]) if samples else 0.0
    mean_exit_r = benchmark_summary.get("mean_bed_exit_recall", samples[0]["bed_events"]["events_by_type"]["bed_exit"]["recall"]) if samples else 0.0
    mean_mae = benchmark_summary.get("mean_duration_mae_sec", samples[0]["duration_estimation"]["mean_absolute_error_sec"]) if samples else 0.0

    lines.extend([
        f"| **State Classification Accuracy** | **{mean_acc*100:.1f}%** | ≥ 90.0% | {'✅ PASS' if mean_acc >= 0.90 else '⚠️ MONITOR'} |",
        f"| **Macro Average F1-Score** | **{mean_f1*100:.1f}%** | ≥ 85.0% | {'✅ PASS' if mean_f1 >= 0.85 else '⚠️ MONITOR'} |",
        f"| **Bed-Exit Precision** | **{mean_exit_p*100:.1f}%** | ≥ 90.0% | {'✅ PASS' if mean_exit_p >= 0.90 else '⚠️ MONITOR'} |",
        f"| **Bed-Exit Recall** | **{mean_exit_r*100:.1f}%** | ≥ 90.0% | {'✅ PASS' if mean_exit_r >= 0.90 else '⚠️ MONITOR'} |",
        f"| **Mean Duration Absolute Error** | **{mean_mae:.1f} sec** | ≤ 5.0 sec | {'✅ PASS' if mean_mae <= 5.0 else '⚠️ MONITOR'} |",
        "",
        "---",
        "",
    ])

    for s_idx, sample in enumerate(samples, 1):
        s_name = sample["sample_name"]
        act = sample["activity_recognition"]
        evs = sample["bed_events"]
        dur = sample["duration_estimation"]

        lines.extend([
            f"## {s_idx + 1}. Sample Benchmark: `{s_name}`",
            "",
            f"- **Ground Truth Source**: `{sample['gt_file']}`",
            f"- **Predictions Directory**: `{sample['pred_dir']}`",
            f"- **Observation Duration**: `{dur['total_observation_hms']}` ({dur['total_observation_sec']}s)",
            "",
            "### 2.1 Activity Recognition Performance",
            "",
            "| State | Precision | Recall | F1-Score | GT Support (Duration) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for st, m in act["per_class_metrics"].items():
            if m["support_sec"] > 0 or m["false_positive_sec"] > 0:
                lines.append(
                    f"| **{st.replace('_', ' ').title()}** | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1_score']*100:.1f}% | {m['support_sec']:.1f}s |"
                )

        lines.extend([
            "",
            f"**Overall Classification Accuracy**: `{act['accuracy']*100:.2f}%`  ",
            f"**Macro F1-Score**: `{act['macro_f1']*100:.2f}%` | **Weighted F1-Score**: `{act['weighted_f1']*100:.2f}%`",
            "",
            "#### Confusion Matrix (Time in Seconds)",
            "",
        ])

        active_states = act["active_states"]
        col_headers = " | ".join([f"**{s.replace('_IN_BED', '').replace('_ON_BED', '').title()[:8]}**" for s in active_states])
        lines.append(f"| True \\ Pred | {col_headers} |")
        lines.append(f"| :--- | {' :---: |' * len(active_states)}")

        for r_idx, r_state in enumerate(active_states):
            r_name = r_state.replace('_IN_BED', '').replace('_ON_BED', '').title()[:8]
            row_vals = " | ".join([f"{v:.1f}s" for v in act["confusion_matrix_sec"][r_idx]])
            lines.append(f"| **{r_name}** | {row_vals} |")

        # Bed Events
        lines.extend([
            "### 2.2 Clinical Bed Events",
            "",
            "| Event Type | Ground Truth Count | Predicted Count | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for ev_type, m in evs["events_by_type"].items():
            lines.append(
                f"| **{ev_type.replace('_', ' ').title()}** | {m['ground_truth_count']} | {m['predicted_count']} | {m['true_positives']} | {m['false_positives_count']} | {m['false_negatives_count']} | **{m['precision']*100:.1f}%** | **{m['recall']*100:.1f}%** | **{m['f1_score']*100:.1f}%** |"
            )

        lines.extend([
            "",
            "### 2.3 Duration Estimation Comparison",
            "",
            "| Activity State | Ground Truth (HH:MM:SS) | Predicted (HH:MM:SS) | Duration Error (sec) | Relative Error (%) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])

        for row in dur["per_activity_comparison"]:
            if row["gt_sec"] > 0 or row["pred_sec"] > 0:
                lines.append(
                    f"| **{row['display_name']}** | `{row['gt_hms']}` ({row['gt_sec']}s) | `{row['pred_hms']}` ({row['pred_sec']}s) | **{row['duration_error_sec']} sec** | {row['percentage_error']:.1f}% |"
                )

        in_b = dur["aggregate_comparisons"]["time_in_bed"]
        out_b = dur["aggregate_comparisons"]["time_out_of_bed"]
        lines.extend([
            f"| **Total Time In Bed** | `{in_b['gt_hms']}` ({in_b['gt_sec']}s) | `{in_b['pred_hms']}` ({in_b['pred_sec']}s) | **{in_b['duration_error_sec']} sec** | {in_b['percentage_error']:.1f}% |",
            f"| **Total Time Out Of Bed** | `{out_b['gt_hms']}` ({out_b['gt_sec']}s) | `{out_b['pred_hms']}` ({out_b['pred_sec']}s) | **{out_b['duration_error_sec']} sec** | {out_b['percentage_error']:.1f}% |",
            "",
            f"**Mean Absolute Duration Error across classes**: `{dur['mean_absolute_error_sec']:.2f} seconds`",
            "",
            "---",
            "",
        ])

    content = "\n".join(lines)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(content)

    return content
