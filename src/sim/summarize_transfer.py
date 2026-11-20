#!/usr/bin/env python3
"""Summarize cross-simulator transfer results into the paper's Table.

Reads the per-agent score CSVs written by extract_features.py for each
formation and reports, for the known critical agents, their mean GNN score and
Precision@k (k = number of critical agents). Critical agents are assigned by
formation design (see src/sim/run_scenario.py), not discovered.

Usage (after running the scenarios):
    python src/sim/summarize_transfer.py
"""
import csv
import os
from collections import defaultdict

# scenario -> critical agent ids (by formation design)
CRITICALS = {
    "moving_escort": {0},
    "single_relay_mesh": {8},
    "escort_many": {0},
    "two_cores": {0, 1},
    "relay_bridge": {18, 19},
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def mean_scores(path):
    sc = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            sc[int(r["agent_id"])].append(float(r["score"]))
    return {a: sum(v) / len(v) for a, v in sc.items()}


def main():
    print(f"{'Formation':18s} {'agents':>6s} {'crit. score':>22s} {'others(max)':>12s} {'P@k':>6s}")
    print("-" * 70)
    for name, crits in CRITICALS.items():
        path = os.path.join(OUT, f"{name}_scores.csv")
        if not os.path.exists(path):
            print(f"{name:18s}  MISSING {path}")
            continue
        means = mean_scores(path)
        n = len(means)
        order = sorted(means, key=lambda a: -means[a])
        k = len(crits)
        top = set(order[:k])
        p_at_k = len(top & crits) / k
        crit_scores = [f"{means[c]:.4f}" for c in sorted(crits)]
        others_max = max((v for a, v in means.items() if a not in crits), default=0.0)
        print(f"{name:18s} {n:6d} {', '.join(crit_scores):>22s} {others_max:12.4f} {p_at_k:6.2f}")
    print()
    print("'crit. score' = mean GNN score of the critical agent(s); 'others(max)'")
    print("= highest mean score among non-critical agents; P@k = share of the")
    print("top-k ranked agents that are critical.")


if __name__ == "__main__":
    main()
