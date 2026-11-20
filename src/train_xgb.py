#!/usr/bin/env python3
"""XGBoost baseline on the local_features node-level feature set.

Uses the same scenario-grouped split and seed as train_rf.py so the result can
be compared directly with the 20-feature Random Forest baseline.
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import xgboost
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

from train_rf import VARIANT_FEATURES, grouped_split, load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    parser.add_argument("--out", default="results/local_features_xgboost.json")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    arrays, cases, case_meta = load_data(args.data)
    y = arrays.pop("is_critical").astype(np.int8)
    features = VARIANT_FEATURES["local_features"]
    train_mask, test_mask, train_cases, test_cases = grouped_split(
        cases, case_meta, seed=args.seed
    )

    X_train = np.column_stack([arrays[name][train_mask] for name in features]).astype(np.float32)
    X_test = np.column_stack([arrays[name][test_mask] for name in features]).astype(np.float32)
    y_train, y_test = y[train_mask], y[test_mask]
    del arrays, cases, case_meta, y, train_mask, test_mask
    gc.collect()

    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / positive
    print(f"[split] train: {len(y_train):,} rows / {len(train_cases)} cases | "
          f"test: {len(y_test):,} rows / {len(test_cases)} cases")
    print(f"[features] local_features: {len(features)}")
    print(f"[class balance] scale_pos_weight={scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=args.jobs,
        random_state=args.seed,
    )

    started = time.time()
    model.fit(X_train, y_train)
    print(f"[fit] complete in {time.time() - started:.1f}s")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    metrics = {
        "model": "XGBoost",
        "variant": "local_features",
        "target_known": False,
        "n_features": len(features),
        "features": features,
        "xgboost_version": xgboost.__version__,
        "n_estimators": args.n_estimators,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "seed": args.seed,
        "n_train_rows": int(len(y_train)),
        "n_test_rows": int(len(y_test)),
        "n_train_cases": len(train_cases),
        "n_test_cases": len(test_cases),
        "prevalence_test": float(y_test.mean()),
        "baseline_pr_auc = prevalence": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy@0.5": float(accuracy_score(y_test, predictions)),
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "f1@0.5": float(f1),
        "confusion_matrix(tn,fp,fn,tp)": [int(tn), int(fp), int(fn), int(tp)],
    }

    result = {
        "metrics": metrics,
        "feature_importances": {
            name: float(value)
            for name, value in zip(features, model.feature_importances_)
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
