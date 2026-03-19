from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from eval.result_io import extract_prediction_text, iter_jsonl, write_jsonl
    from eval.scoring import compute_acc_from_raw_answer, get_scorer_config
else:
    from .result_io import extract_prediction_text, iter_jsonl, write_jsonl
    from .scoring import compute_acc_from_raw_answer, get_scorer_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score new_eval result files.")
    parser.add_argument("paths", nargs="+", help="Result JSONL files to score.")
    parser.add_argument("--score-key", default="score")
    parser.add_argument("--write-scored", action="store_true")
    parser.add_argument("--local", action="store_true", help="Default to localhost scorer if no base URL is set.")
    return parser.parse_args()


def score_file(path: str, *, score_key: str, write_scored: bool, local: bool) -> None:
    entries = list(iter_jsonl(path))
    if not entries:
        print(f"{path}: empty file")
        return

    scorer_config = get_scorer_config(local=local)
    print(f"SCORING WITH: {scorer_config['model']} @ {scorer_config['base_url'] or 'OpenAI default'}")

    total = 0
    scored_entries = []
    for index, entry in enumerate(entries, start=1):
        prediction = extract_prediction_text(entry)
        score, normalized_prediction = compute_acc_from_raw_answer(
            entry.get("query", ""),
            entry.get("ground_truth", ""),
            prediction,
            local=local,
        )
        entry[score_key] = score
        entry["scored_prediction"] = normalized_prediction
        scored_entries.append(entry)
        total += score

        if index % 25 == 0 or index == len(entries):
            print(f"[score] {Path(path).name}: {index}/{len(entries)}")

    average = total / len(entries)
    print(f"{path}: {total}/{len(entries)} = {average:.4f}")

    if write_scored:
        output_path = str(Path(path).with_suffix("")) + "_scored.jsonl"
        write_jsonl(output_path, scored_entries)
        print(f"WROTE: {output_path}")


def main() -> None:
    args = parse_args()
    for path in args.paths:
        score_file(path, score_key=args.score_key, write_scored=args.write_scored, local=args.local)


if __name__ == "__main__":
    main()
