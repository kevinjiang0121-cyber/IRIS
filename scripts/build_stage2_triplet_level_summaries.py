import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_completions import _build_v13_summary, _load_behaviors_csv


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_results_by_level(results, logs):
    by_level = {"T1": {}, "T2": {}, "T3": {}}

    for behavior_id, samples in results.items():
        log_entries = logs.get(behavior_id, [])
        if len(samples) != len(log_entries):
            raise ValueError(
                f"Sample/log length mismatch for {behavior_id}: "
                f"{len(samples)} samples vs {len(log_entries)} logs"
            )

        for sample, log_entry in zip(samples, log_entries):
            level = str((log_entry or {}).get("source_field", "")).strip()
            if level not in by_level:
                continue
            by_level[level].setdefault(behavior_id, []).append(sample)

    return by_level


def _build_level_summaries(
    results_path: Path, logs_path: Path, behaviors_path: Path, behavior_lang: str
):
    results = _load_json(results_path)
    logs = _load_json(logs_path)
    behaviors = _load_behaviors_csv(str(behaviors_path), behavior_lang=behavior_lang)
    behaviors_by_id = {b["BehaviorID"]: b for b in behaviors}

    split = _split_results_by_level(results, logs)

    summaries = {}
    for level, level_results in split.items():
        summary = _build_v13_summary(
            level_results,
            behaviors_by_id=behaviors_by_id,
            prompt_family="direct_request",
            judge_model_name=None,
        )
        summary["metadata"]["stage2_triplet_level"] = level
        summary["metadata"]["source_results_file"] = str(results_path)
        summary["metadata"]["source_logs_file"] = str(logs_path)
        summaries[level] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Build per-level T1/T2/T3 summaries from Stage2Triplet judge results."
    )
    parser.add_argument(
        "--results_root",
        default="results/Stage2Triplet/default/results",
        help="Root directory containing per-model Stage2Triplet judge result files.",
    )
    parser.add_argument(
        "--logs_path",
        default="results/Stage2Triplet/default/test_cases/logs.json",
        help="Stage2Triplet logs.json path containing T1/T2/T3 source_field mapping.",
    )
    parser.add_argument(
        "--behaviors_path",
        default="tmp/stage2_exports/behavior_stage2_t1.csv",
        help="Behavior CSV used for semantic-category metadata.",
    )
    parser.add_argument(
        "--behavior_lang",
        default="en",
        choices=["en", "eo", "es", "tk"],
        help="Language selector for the behaviors CSV.",
    )
    parser.add_argument(
        "--output_root",
        default="results/Stage2Triplet/default/results_by_level",
        help="Output directory for per-level summaries.",
    )
    parser.add_argument(
        "--judge_version",
        default="v2",
        help="Judge result version suffix to match (e.g. 'v2' matches *.judge_v2.json; 'v13' matches *.judge_v13.json).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logs_path = Path(args.logs_path)
    behaviors_path = Path(args.behaviors_path)

    rows = []
    glob_pattern = f"*/*.judge_{args.judge_version}.json"
    for result_path in sorted(results_root.glob(glob_pattern)):
        model_name = result_path.parent.name
        summaries = _build_level_summaries(
            results_path=result_path,
            logs_path=logs_path,
            behaviors_path=behaviors_path,
            behavior_lang=args.behavior_lang,
        )

        model_out_dir = output_root / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for level, summary in summaries.items():
            out_path = model_out_dir / f"{result_path.stem}.{level}.summary.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            overall = summary["overall"]
            rows.append(
                {
                    "model": model_name,
                    "level": level,
                    "summary_path": str(out_path),
                    "num_samples": overall["num_samples"],
                    "valid_judgment_rate": overall["valid_judgment_rate"],
                    "ur": overall["ur"],
                    "shift_rate": overall["shift_rate"],
                    "partial_harm_rate": overall["partial_harm_rate"],
                    "full_success_rate": overall["full_success_rate"],
                    "combined_asr": overall["combined_asr"],
                    "blocked_empty_ratio": overall["blocked_empty_ratio"],
                    "conditional_refusal_given_correct_recovery": overall[
                        "conditional_refusal_given_correct_recovery"
                    ],
                    "conditional_harm_given_correct_recovery": overall[
                        "conditional_harm_given_correct_recovery"
                    ],
                }
            )

    csv_path = output_root / "stage2_triplet_levels_overall.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "level",
                "summary_path",
                "num_samples",
                "valid_judgment_rate",
                "ur",
                "shift_rate",
                "partial_harm_rate",
                "full_success_rate",
                "combined_asr",
                "blocked_empty_ratio",
                "conditional_refusal_given_correct_recovery",
                "conditional_harm_given_correct_recovery",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} level summaries to {output_root}")
    print(f"Wrote overall CSV to {csv_path}")


if __name__ == "__main__":
    main()
