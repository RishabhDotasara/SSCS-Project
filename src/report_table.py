#!/usr/bin/env python3
"""
Aggregate train_rf.py result JSONs into a comparison table for the report.

Usage:
  python report_table.py results/*.json                # markdown table to stdout
  python report_table.py results/*.json --out table.csv  # CSV export
  python report_table.py results/*.json --groups          # include per-group breakdown
"""
import argparse
import csv
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="CSV export path")
    ap.add_argument("--groups", action="store_true",
                    help="print per-attrition / per-swarm-size breakdowns for runs that have them")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)))

    rows = []
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        m = data["metrics"]
        rows.append({
            "run": path,
            "variant": (f"{m['model']}/" if m.get("model") else "")
                       + m["variant"] + ("+temp" if m.get("temporal") else ""),
            "eval_step": m.get("eval_step", "all"),
            "n_feat": m.get("n_features"),
            "train_rows": m.get("n_train_rows"),
            "test_rows": m.get("n_test_rows"),
            "prev_test": round(m.get("prevalence_test", 0), 3),
            "roc_auc": round(m.get("roc_auc", float("nan")), 4),
            "pr_auc": round(m.get("pr_auc", float("nan")), 4),
            "prec@0.5": round(m.get("precision@0.5", float("nan")), 4),
            "rec@0.5": round(m.get("recall@0.5", float("nan")), 4),
            "f1@0.5": round(m.get("f1@0.5", float("nan")), 4),
        })

    cols = ["run", "variant", "eval_step", "n_feat", "train_rows", "test_rows",
            "prev_test", "roc_auc", "pr_auc", "prec@0.5", "rec@0.5", "f1@0.5"]

    # Markdown table
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[saved] {args.out}")

    if args.groups:
        for path in paths:
            with open(path) as fh:
                data = json.load(fh)
            bg = data["metrics"].get("by_groups")
            if not bg:
                continue
            print(f"\n== per-group breakdown: {path} ==")
            print(f"{'group':20s} {'n':>8s} {'prev':>6s} {'roc_auc':>8s} {'pr_auc':>8s}")
            for k in sorted(bg):
                v = bg[k]
                print(f"{k:20s} {v['n']:8d} {v['prevalence']:.3f} {v['roc_auc']:8.4f} {v['pr_auc']:8.4f}")


if __name__ == "__main__":
    main()
