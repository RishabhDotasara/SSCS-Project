#!/usr/bin/env python3
"""
Random Forest baseline for agent criticality prediction in the swarm dataset.

Goal: predict `is_critical` (binary, 1 = critical/high-value agent) from
NODE-LEVEL features only, i.e., WITHOUT the graph-centrality columns
(degree/closeness/betweenness/eigenvector) and WITHOUT `criticality_score`.

Why these two are excluded:
  * graph-centrality columns encode swarm-relational (global graph) structure;
    leaving them out is the point of this experiment -- we want per-agent data.
  * criticality_score is a precomputed estimate of the very target we predict
    (a severe leak: with it, models saturate at ~0.9998 PR-AUC regardless of
    anything else). It must not appear in features.

TIME DIMENSION (read this)
--------------------------
The dataset is time-marched telemetry (rows = agent x timestep, sampled every
2 sim steps). This baseline treats each row independently and does not create
temporal or sequence features:

  * `step` is excluded from features (it is a phase index, not agent data),
  * every row is treated as an independent sample,
  * the train/test split is grouped by scenario (case_id), so no scenario
    (and therefore no future/past information of a test swarm) leaks into
    training. Within a scenario, consecutive timesteps of the same agent are
    near-duplicates, which inflates the effective test sample size -- use
    --eval-step to measure on a single-timestep snapshot instead.


Feature variants -- ablation ladder, each rung adds one information level:
  motion_only      pure kinematics (velocity, heading, accel) -- NO geometry:
                   shows how weak the signal is without any position info
  node_only        motion + target-independent position (5 params: rel_x/y/z,
                   dist/rank to centroid)
  local_features   node data + local neighbor geometry
                   (num_neighbors, distances, density, angular coverage)
                   [scenario-level metadata -- swarm size, obstacles,
                   attrition -- is excluded: it describes the mission, not
                   the node, and measured feature importance is <1.5%]

Validation: scenario-grouped split (train/test by case_id, stratified over
swarm size x attrition level) to avoid temporally-correlated leakage.

Examples (run from this directory):
  python train_rf.py --variant motion_only     --n-trees 300
  python train_rf.py --variant node_only       --n-trees 300
  python train_rf.py --variant local_features  --n-trees 300


  # honest evaluation on a single timestep snapshot (e.g. step 400)
  python train_rf.py --variant local_features --eval-step 400 --n-trees 300
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------
LOCAL_GEOMETRY = [
    "num_neighbors",
    "mean_neighbor_dist",
    "min_neighbor_dist",
    "weighted_density",
    "angular_coverage",
]
ENVIRONMENT = [
    "num_agents",
    "n_obstacles",
    "obstacle_density",
    "mean_obstacle_size",
    "clutter_index",
    "obstacle_clustering",
    "attrition_rate",
]
MOTION = [
    "vx", "vy", "vz",
    "speed", "heading", "flight_path_angle",
    "accel", "heading_rate",
]
POSITION = [
    "rel_x", "rel_y", "rel_z",
    "dist_to_centroid", "rank_dist_to_centroid",
]
KINEMATICS = MOTION + POSITION
ALWAYS_DROP = {"case_id", "step", "agent_id", "is_critical", "was_promoted"}

# Ablation ladder (each rung adds one information level):
#   motion_only     -> pure kinematics (NO geometry at all): what does motion alone give?
#   node_only       -> + position/geometry (5 params):       how much does position-in-formation add?
#   local_features  -> + local neighbor geometry:            how much does the local escort signature add?
#
# Note: scenario-level metadata (num_agents, obstacles, attrition_rate, ...)
# is deliberately NOT a feature in any variant -- it describes the mission
# setup, not the node. A target-blind with/without ablation is available via
# --include-environment; metadata can still be used by --by-groups for analysis.
VARIANT_FEATURES = {
    "motion_only": MOTION,                                        #  8 features
    "node_only": KINEMATICS,                                      # 13 features
    "local_features": KINEMATICS + LOCAL_GEOMETRY,                # 18 features
}

ATTRITION_MAP = {"none": 0, "light": 1, "medium": 2, "heavy": 3}


def load_data(path):
    """Stream the CSV into per-column numpy arrays (pandas-free).

    The file is ordered by (case_id, step, agent_id). Rows are loaded as
    independent observations; no temporal features are constructed.
    """
    t0 = time.time()
    cols = defaultdict(list)
    cases = []
    case_meta = {}  # case_id -> (attrition_code, num_agents) for stratified split

    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ix = {c: i for i, c in enumerate(header)}
        for r in rd:
            case_id = r[ix["case_id"]]

            cases.append(case_id)
            if case_id not in case_meta:
                case_meta[case_id] = (ATTRITION_MAP[r[ix["attrition_rate"]]], int(r[ix["num_agents"]]))
            cols["attrition_rate"].append(ATTRITION_MAP[r[ix["attrition_rate"]]])
            cols["is_critical"].append(1.0 if r[ix["is_critical"]] == "True" else 0.0)
            cols["num_agents"].append(float(r[ix["num_agents"]]))
            for c in header:
                if c == "case_id" or c in ("attrition_rate", "num_agents", "is_critical"):
                    continue
                if c == "was_promoted":
                    cols[c].append(1.0 if r[ix[c]] == "True" else 0.0)
                    continue
                cols[c].append(float(r[ix[c]]))

    arrays = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    cases = np.asarray(cases)
    print(f"[load] {len(cases):,} rows, {len(header)} cols in {time.time()-t0:.1f}s")
    return arrays, cases, case_meta


def grouped_split(cases, case_meta, train_frac=0.8, seed=42, min_test=1):
    """Split *cases* into train/test, stratified over (attrition, swarm size)."""
    rng = np.random.default_rng(seed)
    by_combo = defaultdict(list)
    for case in sorted(set(cases.tolist())):
        att, n = case_meta[case]
        by_combo[(att, n)].append(case)
    train_cases, test_cases = set(), set()
    for combo, cset in by_combo.items():
        rng.shuffle(cset)
        n_test = max(min_test, int(round(len(cset) * (1 - train_frac))))
        test_cases.update(cset[:n_test])
        train_cases.update(cset[n_test:])
    mask_tr = np.isin(cases, list(train_cases))
    mask_te = np.isin(cases, list(test_cases))
    return mask_tr, mask_te, train_cases, test_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    ap.add_argument("--variant", choices=VARIANT_FEATURES, default="local_features")
    ap.add_argument("--include-environment", action="store_true",
                    help="add scenario metadata for a feature ablation")
    ap.add_argument("--eval-step", type=int, default=None,
                    help="evaluate only rows at this sim step (single-timestep snapshot, "
                         "avoids temporally-correlated test rows)")
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="cap training rows (0 = use all); rows are randomly sampled")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default=None, help="JSON output path for metrics")
    ap.add_argument("--by-groups", action="store_true",
                    help="also report test metrics broken down by attrition level and swarm size")
    ap.add_argument("--save-preds", default=None, metavar="FILE",
                    help="save test rows (case_id, step, agent_id, is_critical, prob) as CSV "
                         "for PR curves / Precision@k / Damage-at-k analysis")
    args = ap.parse_args()

    X_all, cases, case_meta = load_data(args.data)
    y_all = X_all.pop("is_critical")

    features = VARIANT_FEATURES[args.variant]
    if args.include_environment:
        features = features + [f for f in ENVIRONMENT if f not in features]
    print(f"[variant] {args.variant}: {len(features)} features -> {features}")

    mask_tr, mask_te, tr_cases, te_cases = grouped_split(
        cases, case_meta, seed=args.seed)
    print(f"[split] train: {mask_tr.sum():,} rows / {len(tr_cases)} cases | "
          f"test: {mask_te.sum():,} rows / {len(te_cases)} cases")

    if args.max_train_rows:
        n = min(args.max_train_rows, int(mask_tr.sum()))
        rng = np.random.default_rng(args.seed)
        idx = np.where(mask_tr)[0]
        pick = rng.choice(idx, size=n, replace=False)
        mask_tr = np.zeros_like(mask_tr)
        mask_tr[pick] = True
        print(f"[split] capped training rows at {n:,}")

    if args.eval_step is not None:
        n_before = int(mask_te.sum())
        mask_te &= X_all["step"] == args.eval_step
        print(f"[split] eval step {args.eval_step}: test rows {n_before:,} -> {int(mask_te.sum()):,}")
        if mask_te.sum() == 0:
            print("[warn] eval-step produced no test rows (scenarios too short); using all test rows")
            mask_te = np.zeros_like(mask_te)
            mask_te[np.isin(cases, list(te_cases))] = True

    X_tr = np.column_stack([X_all[f][mask_tr] for f in features])
    y_tr = y_all[mask_tr]
    X_te = np.column_stack([X_all[f][mask_te] for f in features])
    y_te = y_all[mask_te]
    print(f"[data] train X {X_tr.shape}, test X {X_te.shape} | "
          f"prevalence train {y_tr.mean():.3f}, test {y_te.mean():.3f}")

    model = RandomForestClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=args.jobs,
        random_state=args.seed,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    print(f"[fit] done in {time.time()-t0:.1f}s")

    p = model.predict_proba(X_te)[:, 1]
    yhat = model.predict(X_te)

    metrics = {
        "variant": args.variant,
        "target_known": False,
        "include_environment": args.include_environment,
        "eval_step": args.eval_step,
        "n_features": len(features),
        "features": features,
        "n_train_rows": int(y_tr.shape[0]),
        "n_test_rows": int(y_te.shape[0]),
        "prevalence_train": float(y_tr.mean()),
        "prevalence_test": float(y_te.mean()),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "pr_auc": float(average_precision_score(y_te, p)),
        "baseline_pr_auc = prevalence": float(y_te.mean()),
        "n_trees": args.n_trees,
    }
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, yhat, average="binary")
    metrics.update({"precision@0.5": float(prec), "recall@0.5": float(rec), "f1@0.5": float(f1)})
    metrics["confusion_matrix(tn,fp,fn,tp)"] = [int(v) for v in confusion_matrix(y_te, yhat).ravel()]

    print("\n===== RESULTS =====")
    print(json.dumps(metrics, indent=2))
    print("\nclassification report @0.5 threshold:")
    print(classification_report(y_te, yhat, digits=4))

    order = np.argsort(model.feature_importances_)[::-1]
    print("\nfeature importances:")
    for i in order:
        print(f"  {features[i]:24s} {model.feature_importances_[i]:.4f}")

    # ---- per-group breakdown (robustness section of the report) ----
    if args.by_groups:
        groups = {}
        for label, code in [("none", 0), ("light", 1), ("medium", 2), ("heavy", 3)]:
            m = (X_all["attrition_rate"][mask_te] == code) & (y_te >= 0)
            if m.sum() >= 30:
                groups[f"attrition_{label}"] = {
                    "n": int(m.sum()),
                    "prevalence": float(y_te[m].mean()),
                    "roc_auc": float(roc_auc_score(y_te[m], p[m])),
                    "pr_auc": float(average_precision_score(y_te[m], p[m])),
                }
        for n_ag in sorted({float(v) for v in set(X_all["num_agents"][mask_te].tolist())}):
            m = X_all["num_agents"][mask_te] == n_ag
            if m.sum() >= 30:
                groups[f"swarm_n{int(n_ag)}"] = {
                    "n": int(m.sum()),
                    "prevalence": float(y_te[m].mean()),
                    "roc_auc": float(roc_auc_score(y_te[m], p[m])),
                    "pr_auc": float(average_precision_score(y_te[m], p[m])),
                }
        metrics["by_groups"] = groups
        print("\nper-group test metrics (attrition / swarm size):")
        for k, v in groups.items():
            print(f"  {k:18s} n={v['n']:7d} prev={v['prevalence']:.3f} "
                  f"roc_auc={v['roc_auc']:.4f} pr_auc={v['pr_auc']:.4f}")

    # ---- save per-row predictions for offline analysis ----
    if args.save_preds:
        test_idx = np.where(mask_te)[0]
        with open(args.save_preds, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "step", "agent_id", "is_critical", "prob"])
            for i in test_idx:
                w.writerow([cases[i], int(X_all["step"][i]), int(X_all["agent_id"][i]),
                            int(y_all[i]), f"{p[i]:.6f}"])
        print(f"\n[saved] predictions for {len(test_idx):,} test rows -> {args.save_preds}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"metrics": metrics,
                       "feature_importances": {features[i]: float(model.feature_importances_[i])
                                               for i in order}},
                      fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
