#!/usr/bin/env python3
"""
Rank-displacement evaluation.

Compares the model's predicted critical ranking against the ground-truth
*criticality ranking* (the continuous `criticality_score` in the dataset).

For each snapshot, agents are ordered once by ground-truth criticality_score
(descending) and once by the model's predicted probability (descending). Two
metrics are reported, averaged over test snapshots:

  * rank_disp  - normalized mean absolute rank displacement of the CRITICAL
                 agents (is_critical == 1). 0 = every critical agent keeps its
                 exact rank; ~1/3 is the random-baseline expectation; 1 is the
                 worst case. This is the "agent is #1 by criticality but #3 by
                 prediction" metric: displacement 2, divided by (n-1).
  * spearman   - Spearman rank correlation between predicted probability and
                 criticality_score over ALL agents in the snapshot (+1 = perfect
                 order agreement).

Reads the per-agent predictions CSV written by train_gnn.py (--save-preds) and
joins with the dataset for criticality_score. No retraining.

Usage:
  python src/gnn/rank_displacement.py \
      --preds results/no_target/gnn_knn_preds.csv \
      --data data/swarm_criticality_dataset_100_scenarios.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def rank_disp(scores, cs, labels):
    """Return (rank_disp, spearman) for one snapshot."""
    n = len(scores)
    if n < 2:
        return 0.0, 0.0
    rt = np.empty(n, dtype=np.float64)
    rp = np.empty(n, dtype=np.float64)
    rt[np.argsort(-cs, kind="stable")] = np.arange(n)
    rp[np.argsort(-scores, kind="stable")] = np.arange(n)
    crit = labels.astype(bool)
    if crit.any():
        disp = float(np.mean(np.abs(rp[crit] - rt[crit]))) / (n - 1)
    else:
        disp = 0.0
    if np.std(rt) == 0.0 or np.std(rp) == 0.0:
        sp = 0.0
    else:
        sp = float(np.corrcoef(rp, rt)[0, 1])
    return disp, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 1. Read predictions, grouped per snapshot.
    snap = defaultdict(lambda: {"agent": [], "prob": [], "crit": []})
    with open(args.preds, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            key = (r["case_id"], int(r["step"]))
            snap[key]["agent"].append(int(r["agent_id"]))
            snap[key]["prob"].append(float(r["prob"]))
            snap[key]["crit"].append(int(r["is_critical"]))
    needed = set(snap.keys())
    print(f"[preds] {len(needed)} snapshots, {sum(len(v['agent']) for v in snap.values()):,} agent rows")

    # 2. Join criticality_score from the dataset for the needed snapshots only.
    cs_map = defaultdict(dict)
    with open(args.data, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            key = (r["case_id"], int(r["step"]))
            if key in needed:
                cs_map[key][int(r["agent_id"])] = float(r["criticality_score"])
    print(f"[data] criticality_score joined for {len(cs_map)}/{len(needed)} snapshots")

    # 3. Per-snapshot metrics + random baseline.
    model_disp, model_sp, rand_disp = [], [], []
    rng = np.random.default_rng(args.seed)
    matched = 0
    for key, d in snap.items():
        if key not in cs_map:
            continue
        agents = d["agent"]
        cs_lookup = cs_map[key]
        cs = np.array([cs_lookup[a] for a in agents], dtype=np.float64)
        scores = np.array(d["prob"], dtype=np.float64)
        labels = np.array(d["crit"], dtype=np.float64)
        disp, sp = rank_disp(scores, cs, labels)
        model_disp.append(disp)
        model_sp.append(sp)
        # random baseline: shuffle the predicted scores once per snapshot
        rdisp, _ = rank_disp(rng.permutation(scores), cs, labels)
        rand_disp.append(rdisp)
        matched += 1

    print(f"[eval] {matched} snapshots scored\n")
    print(f"{'metric':28s} {'mean':>10s}")
    print("-" * 40)
    print(f"{'rank_disp (critical agents)':28s} {np.mean(model_disp):10.4f}")
    print(f"{'rank_disp (random baseline)':28s} {np.mean(rand_disp):10.4f}")
    print(f"{'spearman (all agents)':28s} {np.mean(model_sp):10.4f}")

    result = {
        "n_snapshots": matched,
        "rank_disp_criticals": float(np.mean(model_disp)),
        "rank_disp_random_baseline": float(np.mean(rand_disp)),
        "spearman_all_agents": float(np.mean(model_sp)),
    }
    if args.out:
        import json, os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
