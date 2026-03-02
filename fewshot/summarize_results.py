"""
Summarize all experiment results into a comparison table.

Usage:
    python fewshot/summarize_results.py --results_dir fewshot/results
"""

import os
import json
import argparse
from collections import defaultdict


def main(results_dir):
    results = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    # Group by (method, dataset, k_shot)
    grouped = defaultdict(list)
    for r in results:
        key = (r["method"], r["dataset"], r["k_shot"])
        grouped[key].append(r["test_accuracy"])

    # Print table
    print(f"\n{'Method':<22} {'Dataset':<22} {'k_shot':>6} {'Accuracy':>12} {'Std':>8} {'N':>4}")
    print("-" * 78)

    for (method, dataset, k_shot) in sorted(grouped.keys(), key=lambda x: (x[1], x[0], x[2])):
        accs = grouped[(method, dataset, k_shot)]
        mean_acc = sum(accs) / len(accs)
        if len(accs) > 1:
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
        else:
            std_acc = 0.0

        ds_short = os.path.basename(dataset).replace("img_dataset_", "")
        print(f"{method:<22} {ds_short:<22} {k_shot:>6} {mean_acc:>11.4f} {std_acc:>8.4f} {len(accs):>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="fewshot/results")
    args = parser.parse_args()
    main(args.results_dir)
