#!/usr/bin/env python3
"""Evaluate a trained target-blind GNN under graph-construction uncertainty.

Reuses the saved checkpoint from train_gnn.py and evaluates the same test
snapshots with different k for the k-NN test graph, so the model itself is
never retrained.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, \
    precision_recall_fscore_support

from dataset import load_snapshots, SwarmGraphDataset, build_edges
from model import GNN


def evaluate(model, ds_test, batch, device, k_override=None, graph_type="knn", seed=42):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for start in range(0, len(ds_test), batch):
            subset = ds_test[start:start + batch]
            xs, ys, edges = [], [], []
            off = 0
            for idx, g in enumerate(subset):
                kk = k_override if k_override is not None else 12
                ei = build_edges(g["pos"], graph_type, kk, seed=seed + start + idx)
                xs.append(g["x"]); ys.append(g["y"]); edges.append(ei + off)
                off += g["n"]
            x = torch.cat(xs).to(device)
            ei = torch.cat(edges, dim=1).to(device)
            logits = model(x, ei)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(torch.cat(ys).numpy())
    s = np.concatenate(scores); y = np.concatenate(labels)
    roc = roc_auc_score(y, s); pr = average_precision_score(y, s)
    yhat = (s >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary")
    return {"roc_auc": float(roc), "pr_auc": float(pr),
            "prec@0.5": float(prec), "rec@0.5": float(rec), "f1@0.5": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    ap.add_argument("--graph", default="knn")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/no_target/controls")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots, case_meta, features = load_snapshots(args.data)
    ds = SwarmGraphDataset(snapshots, case_meta, graph_type=args.graph,
                           k=args.k, seed=args.seed)

    model = GNN(len(features), hidden=args.hidden, layers=args.layers,
                conv="sage", dropout=0.2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    os.makedirs(args.out_dir, exist_ok=True)
    report = {"checkpoint": args.checkpoint, "clean": None, "k_variants": {}}
    clean = evaluate(model, ds.test, args.batch, device, None, args.graph, args.seed)
    report["clean"] = clean
    print("clean:", clean)

    for kk in (4, 6, 8, 16, 20):
        m = evaluate(model, ds.test, args.batch, device, kk, args.graph, args.seed)
        report["k_variants"][str(kk)] = m
        print(f"k={kk}:", m)

    out = os.path.join(args.out_dir, "graph_uncertainty.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
