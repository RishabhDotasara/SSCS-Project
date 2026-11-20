"""
Train and evaluate the message-passing GNN for critical-node ranking.

Ranking, not just classification: the model outputs a per-node logit ->
sigmoid -> score in [0,1]; nodes are ranked by score within each
(scenario, timestep) snapshot. Metrics: ROC-AUC, PR-AUC, F1@0.5, and
per-snapshot Precision@k / Recall@k / nDCG@k averaged over test snapshots.

Run:
  python gnn/train_gnn.py --graph knn --k 12 --conv sage --epochs 30
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_fscore_support)

from dataset import build_edges, load_snapshots, SwarmGraphDataset
from model import GNN


def ranking_metrics(scores, labels, ks=(1, 3, 5, 10)):
    """Per-snapshot ranking metrics for one snapshot's score/label arrays."""
    order = np.argsort(-scores)          # descending score
    ranked = labels[order].astype(float)
    npos = int(labels.sum())
    n = len(ranked)
    res = {}
    for k in ks:
        kk = min(k, n)                    # clamp to snapshot size, key stays the requested k
        tp = float(ranked[:kk].sum())
        res[f"prec@{k}"] = tp / kk
        res[f"rec@{k}"] = (tp / npos) if npos > 0 else 0.0
        dcg = float(sum(rel / np.log2(i + 2) for i, rel in enumerate(ranked[:kk])))
        ideal = np.sort(labels)[::-1][:kk].astype(float)
        idcg = float(sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal)))
        res[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0
    return res


def rank_disp_metrics(scores, labels, cs):
    """Rank agreement vs the continuous criticality_score.

    Orders agents by ground-truth criticality_score (desc) and by predicted
    score (desc). `rank_disp` is the normalized mean absolute rank displacement
    of the critical agents (label==1): 0 = every critical keeps its rank,
    ~1/3 = random. `spearman` is the rank correlation over all agents.
    """
    n = len(scores)
    if n < 2:
        return {"rank_disp": 0.0, "spearman": 0.0}
    rt = np.empty(n, dtype=np.float64)
    rp = np.empty(n, dtype=np.float64)
    rt[np.argsort(-cs, kind="stable")] = np.arange(n)
    rp[np.argsort(-scores, kind="stable")] = np.arange(n)
    crit = labels.astype(bool)
    disp = (float(np.mean(np.abs(rp[crit] - rt[crit]))) / (n - 1)) if crit.any() else 0.0
    sp = (float(np.corrcoef(rp, rt)[0, 1])
          if (np.std(rt) > 0 and np.std(rp) > 0) else 0.0)
    return {"rank_disp": disp, "spearman": sp}


def perturb_snapshot_batch(batch, features, graph_type, k, noise_std, seed):
    """Add test-time position noise and recompute position-derived inputs/edges."""
    if noise_std <= 0:
        return batch
    index = {name: i for i, name in enumerate(features)}
    required = {"rel_x", "rel_y", "rel_z", "dist_to_centroid", "rank_dist_to_centroid"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"position-noise evaluation requires features: {sorted(missing)}")

    noisy_batch = []
    for graph_index, graph in enumerate(batch):
        rng = np.random.default_rng(seed + graph_index)
        pos = np.asarray(graph["pos"], dtype=np.float32).copy()
        pos += rng.normal(0.0, noise_std, size=pos.shape).astype(np.float32)
        # Estimate the centroid from the noisy tracks and re-center positions.
        pos -= pos.mean(axis=0, keepdims=True)

        x = graph["x"].clone()
        x[:, index["rel_x"]] = torch.from_numpy(pos[:, 0])
        x[:, index["rel_y"]] = torch.from_numpy(pos[:, 1])
        x[:, index["rel_z"]] = torch.from_numpy(pos[:, 2])
        dist = np.linalg.norm(pos, axis=1)
        x[:, index["dist_to_centroid"]] = torch.from_numpy(dist.astype(np.float32))
        n = len(dist)
        ranks = (dist[None, :] < dist[:, None]).sum(axis=1) / max(n - 1, 1)
        x[:, index["rank_dist_to_centroid"]] = torch.from_numpy(ranks.astype(np.float32))

        noisy_graph = dict(graph)
        noisy_graph["x"] = x
        noisy_graph["pos"] = pos
        noisy_graph["edge_index"] = build_edges(
            pos, graph_type, k, seed=seed + graph_index
        )
        noisy_batch.append(noisy_graph)
    return noisy_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    ap.add_argument("--graph", choices=["knn", "true", "full", "none", "rewired"], default="knn")
    ap.add_argument("--k", type=int, default=12, help="k for k-NN graph")
    ap.add_argument("--conv", choices=["sage", "attn"], default="sage")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--position-noise-std", type=float, default=0.0,
                    help="test-only iid Gaussian position noise, in dataset units; "
                         "recomputes centroid features and graph edges")
    ap.add_argument("--checkpoint-out", default=None,
                    help="optional path to save trained model weights")
    ap.add_argument("--checkpoint-in", default=None,
                    help="load trained weights and skip fitting (for robustness evaluation)")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--save-preds", default=None, metavar="FILE",
                    help="save test per-agent scores (case_id, step, agent_id, "
                         "is_critical, prob) as CSV for Damage-at-k / PR curves")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[load] building snapshots...")
    snapshots, case_meta, features = load_snapshots(args.data)
    ds = SwarmGraphDataset(snapshots, case_meta, graph_type=args.graph,
                           k=args.k, seed=args.seed)
    print(f"[data] {len(ds.train)} train snapshots / {len(ds.test)} test snapshots | "
          f"{ds.features_dim} node features | graph={args.graph} k={args.k}")
    print(f"[features] {features}")

    model = GNN(ds.features_dim, hidden=args.hidden, layers=args.layers,
                conv=args.conv, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # class-weighted BCE (prevalence ~0.39 -> weight decoys down / criticals up)
    all_y = torch.cat([b["y"] for b in ds.train])
    pos = all_y.sum(); neg = all_y.numel() - pos
    pos_weight = neg / pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    print(f"[loss] weighted BCE pos_weight={pos_weight:.3f}")

    if args.checkpoint_in:
        model.load_state_dict(torch.load(args.checkpoint_in, map_location=device, weights_only=True))
        print(f"[checkpoint] loaded {args.checkpoint_in}; fitting skipped")
    else:
        train_idx = list(range(len(ds.train)))
        for epoch in range(args.epochs):
            model.train()
            random.shuffle(train_idx)
            total_loss = 0.0
            nb = 0
            for start in range(0, len(train_idx), args.batch):
                batch = [ds.train[i] for i in train_idx[start:start + args.batch]]
                x, edge_index, y, _, _, _ = SwarmGraphDataset.collate(batch)
                x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
                logits = model(x, edge_index)
                loss = criterion(logits, y)
                if torch.isnan(loss):
                    print("[abort] NaN loss — lower --lr (try 3e-4) or reduce --layers")
                    return
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * y.numel()
                nb += y.numel()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[epoch {epoch+1:3d}] loss={total_loss / nb:.5f}")

        if args.checkpoint_out:
            os.makedirs(os.path.dirname(args.checkpoint_out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint_out)
            print(f"[checkpoint] saved {args.checkpoint_out}")

    # ---- evaluation ----
    model.eval()
    test_scores, test_labels = [], []
    snap_scores, snap_labels, snap_cs = [], [], []
    pred_rows = []
    with torch.no_grad():
        for start in range(0, len(ds.test), args.batch):
            batch = ds.test[start:start + args.batch]
            batch = perturb_snapshot_batch(
                batch, features, args.graph, args.k, args.position_noise_std,
                args.seed + start,
            )
            x, edge_index, y, aid, slices, cs_batch = SwarmGraphDataset.collate(batch)
            x, edge_index = x.to(device), edge_index.to(device)
            logits = model(x, edge_index)
            s = torch.sigmoid(logits).cpu().numpy()
            yn = y.cpu().numpy()
            csn = cs_batch.cpu().numpy()
            an = aid.cpu().numpy()
            test_scores.append(s); test_labels.append(yn)
            for bi, (a, b) in enumerate(zip(slices[:-1], slices[1:])):
                snap_scores.append(s[a:b]); snap_labels.append(yn[a:b])
                snap_cs.append(csn[a:b])
                if args.save_preds:
                    meta = batch[bi]
                    for i in range(a, b):
                        pred_rows.append((meta["case_id"], int(meta["step"]),
                                          int(an[i]), int(yn[i]), f"{s[i]:.9f}"))

    scores = np.concatenate(test_scores)
    labels = np.concatenate(test_labels)

    roc = roc_auc_score(labels, scores)
    pr = average_precision_score(labels, scores)
    yhat = (scores >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, yhat, average="binary")

    metrics = {
        "graph": args.graph, "k": args.k, "conv": args.conv,
        "hidden": args.hidden, "layers": args.layers,
        "target_known": False,
        "position_noise_std": args.position_noise_std,
        "n_node_features": ds.features_dim,
        "n_train_snapshots": len(ds.train), "n_test_snapshots": len(ds.test),
        "n_train_cases": ds.n_train_cases, "n_test_cases": ds.n_test_cases,
        "roc_auc": float(roc), "pr_auc": float(pr),
        "prec@0.5": float(prec), "rec@0.5": float(rec), "f1@0.5": float(f1),
        "prevalence_test": float(labels.mean()),
        "ranking": {},
    }

    agg = {}
    for ss, sl in zip(snap_scores, snap_labels):
        r = ranking_metrics(ss, sl)
        for kk, v in r.items():
            agg.setdefault(kk, []).append(v)
    for kk, v in agg.items():
        metrics["ranking"][kk] = float(np.mean(v))

    rd_agg = {}
    for ss, sl, sc in zip(snap_scores, snap_labels, snap_cs):
        r = rank_disp_metrics(ss, sl, sc)
        for kk, v in r.items():
            rd_agg.setdefault(kk, []).append(v)
    metrics["rank_disp"] = {kk: float(np.mean(v)) for kk, v in rd_agg.items()}

    print("\n===== RESULTS =====")
    print(json.dumps(metrics, indent=2))

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        with open(args.save_preds, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "step", "agent_id", "is_critical", "prob"])
            w.writerows(pred_rows)
        print(f"\n[saved] {len(pred_rows):,} per-agent predictions -> {args.save_preds}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"metrics": metrics, "features": features}, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
