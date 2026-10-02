"""
BedSense AI — Benchmark Evaluation CLI.
Executes quantitative evaluation against ground-truth annotations:
  - Activity Recognition: State accuracy, confusion matrix, per-class F1
  - Bed Events: Bed-exit precision, recall, false positive detections
  - Duration Estimation: Predicted vs ground-truth duration and absolute duration error

Usage:
  python -m eval.evaluate
  python eval/evaluate.py --sample sample2
  python eval/evaluate.py --gt-dir data --pred-dir outputs
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from .metrics import (
    MissingArtifactError,
    evaluate_multiple_samples,
    evaluate_single_sample,
)
from .report import (
    generate_markdown_report,
    print_terminal_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BedSense AI — Clinical Evaluation & Benchmark Engine"
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="data",
        help="Directory containing ground-truth JSON files (default: data)",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs",
        help="Directory containing pipeline output folders (default: outputs)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Specific sample name to evaluate (e.g. sample1, sample2). If omitted, evaluates all discovered samples.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="eval/results.json",
        help="Path to save evaluation metrics in JSON format (default: eval/results.json)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="eval/evaluation_report.md",
        help="Path to save evaluation report in Markdown format (default: eval/evaluation_report.md)",
    )
    return parser


def run_evaluation(
    gt_dir: str = "data",
    pred_dir: str = "outputs",
    sample: Optional[str] = None,
    output_json: str = "eval/results.json",
    output_report: str = "eval/evaluation_report.md",
) -> Dict[str, Any]:
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: '{gt_path.resolve()}'")

    sample_configs: List[Dict[str, Path]] = []

    if sample:
        clean_sample = sample.replace(".json", "").replace(".mp4", "")
        gt_file = gt_path / f"{clean_sample}.json"
        sample_pred_dir = pred_path / clean_sample
        sample_configs.append({
            "name": clean_sample,
            "gt_path": gt_file,
            "pred_dir": sample_pred_dir,
        })
    else:
        # Discover all ground truth JSON files in gt_dir (sample1.json, sample2.json, etc.)
        for gt_file in sorted(gt_path.glob("*.json")):
            s_name = gt_file.stem
            sample_pred_dir = pred_path / s_name
            sample_configs.append({
                "name": s_name,
                "gt_path": gt_file,
                "pred_dir": sample_pred_dir,
            })

    if not sample_configs:
        raise FileNotFoundError(f"No ground-truth .json annotation files discovered in '{gt_path.resolve()}'.")

    # Run evaluation across sample configs
    eval_results = evaluate_multiple_samples(sample_configs)

    # Print rich terminal report
    print_terminal_evaluation(eval_results)

    # If any sample was missing artifacts, print a warning notice
    missing = eval_results.get("missing_samples", [])
    if missing:
        print("\n\033[1;93m⚠️ [MISSING SAMPLES NOTICE]\033[0m")
        for m in missing:
            print(f"  • {m['sample_name']}: {m['error']}")

    # Save JSON results
    out_json_path = Path(output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\033[1;92m✓ Machine-readable metrics saved to:\033[0m {out_json_path.resolve()}")

    # Save Markdown report
    out_md_path = Path(output_report)
    generate_markdown_report(eval_results, out_md_path)
    print(f"\033[1;92m✓ Publication-grade Markdown report saved to:\033[0m {out_md_path.resolve()}\n")

    return eval_results


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_evaluation(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            sample=args.sample,
            output_json=args.output_json,
            output_report=args.output_report,
        )
    except MissingArtifactError as e:
        print(f"\n\033[1;91m[EVALUATION ERROR — MISSING OUTPUT ARTIFACTS]\033[0m\n{e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[1;91m[EVALUATION ERROR]\033[0m {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
