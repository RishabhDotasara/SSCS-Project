#!/usr/bin/env python3
"""Tune RF and XGBoost local-feature baselines without touching the test split.

The outer scenario-grouped split matches train_rf.py. A second scenario-grouped
split inside the outer training scenarios is used for hyperparameter selection.
PR-AUC is the tuning objective; final test metrics use the fixed 0.5 threshold.
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from train_rf import VARIANT_FEATURES, grouped_split, load_data


def sampled_indices(indices, limit, rng):
    if len(indices) <= limit:
        return indices
    return rng.choice(indices, size=limit, replace=False)


def score(model, x, y):
    probabilities = model.predict_proba(x)[:, 1]
    return float(average_precision_score(y, probabilities))


def final_metrics(model, x, y, model_name, params, features, prevalence_train,
                  n_train, n_test, n_train_cases, n_test_cases, validation_pr_auc):
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    metrics = {
        "model": model_name,
        "variant": "local_features",
        "target_known": False,
        "tuned_for": "validation PR-AUC",
        "validation_pr_auc": validation_pr_auc,
        "selected_params": params,
        "n_features": len(features),
        "features": features,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "n_train_cases": n_train_cases,
        "n_test_cases": n_test_cases,
        "prevalence_train": prevalence_train,
        "prevalence_test": float(y.mean()),
        "baseline_pr_auc = prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "accuracy@0.5": float(accuracy_score(y, predictions)),
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "f1@0.5": float(f1),
        "confusion_matrix(tn,fp,fn,tp)": [int(tn), int(fp), int(fn), int(tp)],
    }
    return {
        "metrics": metrics,
        "feature_importances": {
            name: float(value)
            for name, value in zip(features, model.feature_importances_)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    parser.add_argument("--out-dir", default="results/tuned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trials", type=int, default=10)
    parser.add_argument("--xgb-trials", type=int, default=12)
    parser.add_argument("--search-train-rows", type=int, default=150_000)
    parser.add_argument("--search-validation-rows", type=int, default=150_000)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    arrays, cases, case_meta = load_data(args.data)
    y = arrays.pop("is_critical").astype(np.int8)
    features = VARIANT_FEATURES["local_features"]

    # Preserve the existing held-out 36-scenario test set exactly.
    outer_train_mask, outer_test_mask, outer_train_cases, outer_test_cases = grouped_split(
        cases, case_meta, seed=args.seed
    )
    outer_train_idx = np.flatnonzero(outer_train_mask)
    outer_test_idx = np.flatnonzero(outer_test_mask)

    # Hyperparameters are selected only on scenarios within the outer training set.
    inner_train_mask, inner_val_mask, inner_train_cases, inner_val_cases = grouped_split(
        cases[outer_train_mask], case_meta, train_frac=0.8, seed=args.seed + 1
    )
    inner_train_idx = outer_train_idx[inner_train_mask]
    inner_val_idx = outer_train_idx[inner_val_mask]

    x_all = np.column_stack([arrays[name] for name in features]).astype(np.float32)
    del arrays, cases, case_meta, outer_train_mask, outer_test_mask
    gc.collect()

    rng = np.random.default_rng(args.seed)
    search_train_idx = sampled_indices(inner_train_idx, args.search_train_rows, rng)
    search_val_idx = sampled_indices(inner_val_idx, args.search_validation_rows, rng)
    x_search_train, y_search_train = x_all[search_train_idx], y[search_train_idx]
    x_search_val, y_search_val = x_all[search_val_idx], y[search_val_idx]
    x_train, y_train = x_all[outer_train_idx], y[outer_train_idx]
    x_test, y_test = x_all[outer_test_idx], y[outer_test_idx]
    del x_all, y, outer_train_idx, outer_test_idx, inner_train_idx, inner_val_idx
    gc.collect()

    print(f"[outer split] train={len(y_train):,}/{len(outer_train_cases)} scenarios; "
          f"test={len(y_test):,}/{len(outer_test_cases)} scenarios")
    print(f"[inner split] train={len(inner_train_cases)} scenarios; "
          f"validation={len(inner_val_cases)} scenarios")
    print(f"[search sample] train={len(y_search_train):,}; validation={len(y_search_val):,}")

    rf_space = {
        "n_estimators": [150, 250, 400],
        "max_depth": [None, 16, 24, 32],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, 0.8],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    rf_candidates = list(ParameterSampler(rf_space, n_iter=args.rf_trials,
                                          random_state=args.seed))
    best_rf_params, best_rf_score = None, -1.0
    rf_search_results = []
    for i, params in enumerate(rf_candidates, 1):
        model = RandomForestClassifier(
            **params, n_jobs=args.jobs, random_state=args.seed,
        )
        started = time.time()
        model.fit(x_search_train, y_search_train)
        val_score = score(model, x_search_val, y_search_val)
        rf_search_results.append({"params": params, "validation_pr_auc": val_score})
        print(f"[RF search {i}/{len(rf_candidates)}] validation PR-AUC={val_score:.5f} "
              f"params={params} ({time.time() - started:.1f}s)", flush=True)
        if val_score > best_rf_score:
            best_rf_params, best_rf_score = params, val_score
        del model
        gc.collect()

    neg, pos = int((y_search_train == 0).sum()), int((y_search_train == 1).sum())
    scale_pos_weight = neg / pos
    xgb_space = {
        "n_estimators": [200, 300, 450, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1, 3, 10],
    }
    xgb_candidates = list(ParameterSampler(xgb_space, n_iter=args.xgb_trials,
                                           random_state=args.seed))
    best_xgb_params, best_xgb_score = None, -1.0
    xgb_search_results = []
    for i, params in enumerate(xgb_candidates, 1):
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            n_jobs=args.jobs,
            random_state=args.seed,
        )
        started = time.time()
        model.fit(x_search_train, y_search_train)
        val_score = score(model, x_search_val, y_search_val)
        xgb_search_results.append({"params": params, "validation_pr_auc": val_score})
        print(f"[XGBoost search {i}/{len(xgb_candidates)}] validation PR-AUC={val_score:.5f} "
              f"params={params} ({time.time() - started:.1f}s)", flush=True)
        if val_score > best_xgb_score:
            best_xgb_params, best_xgb_score = params, val_score
        del model
        gc.collect()

    del x_search_train, y_search_train, x_search_val, y_search_val
    gc.collect()

    rf_model = RandomForestClassifier(
        **best_rf_params, n_jobs=args.jobs, random_state=args.seed,
    )
    started = time.time()
    rf_model.fit(x_train, y_train)
    print(f"[RF refit] complete in {time.time() - started:.1f}s", flush=True)
    rf_result = final_metrics(
        rf_model, x_test, y_test, "RandomForest_tuned", best_rf_params, features,
        float(y_train.mean()), len(y_train), len(y_test), len(outer_train_cases),
        len(outer_test_cases), best_rf_score,
    )

    full_scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_model = XGBClassifier(
        **best_xgb_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=full_scale_pos_weight,
        n_jobs=args.jobs,
        random_state=args.seed,
    )
    started = time.time()
    xgb_model.fit(x_train, y_train)
    print(f"[XGBoost refit] complete in {time.time() - started:.1f}s", flush=True)
    xgb_result = final_metrics(
        xgb_model, x_test, y_test, "XGBoost_tuned", best_xgb_params, features,
        float(y_train.mean()), len(y_train), len(y_test), len(outer_train_cases),
        len(outer_test_cases), best_xgb_score,
    )
    xgb_result["metrics"]["xgboost_version"] = xgboost.__version__
    xgb_result["metrics"]["scale_pos_weight"] = full_scale_pos_weight

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        "results/local_features_rf_tuned.json": rf_result,
        "results/local_features_xgboost_tuned.json": xgb_result,
    }
    for path, payload in outputs.items():
        with open(os.path.join(args.out_dir, os.path.basename(path)), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[saved] {os.path.join(args.out_dir, os.path.basename(path))}")

    summary = {
        "objective": "validation PR-AUC",
        "target_known": False,
        "outer_split_seed": args.seed,
        "inner_split_seed": args.seed + 1,
        "outer_train_scenarios": len(outer_train_cases),
        "inner_train_scenarios": len(inner_train_cases),
        "inner_validation_scenarios": len(inner_val_cases),
        "search_train_rows": len(search_train_idx),
        "search_validation_rows": len(search_val_idx),
        "rf_trials": rf_search_results,
        "xgboost_trials": xgb_search_results,
        "selected_rf": {"params": best_rf_params, "validation_pr_auc": best_rf_score},
        "selected_xgboost": {"params": best_xgb_params, "validation_pr_auc": best_xgb_score},
    }
    summary_path = os.path.join(args.out_dir, "tuning_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
