"""
Damage-at-k evaluation: does removing the model's top-k targets actually
fragment the swarm?

For each test snapshot we rebuild the swarm's connectivity graph from agent
positions (true R<=55 ball graph by default, or the attacker's k-NN graph),
then remove the top-k agents chosen by different strategies and measure the
resulting damage:

  * model      - top-k by the GNN's predicted criticality score
  * random     - k random agents (averaged over repeats)
  * degree     - top-k by degree (local centrality heuristic)
  * critical   - true-label oracle: remove actual critical agents first
  * betweenness- top-k by betweenness (optional, --betweenness)

Damage metrics (relative to the intact snapshot):
  * lcc_drop        - drop in largest-connected-component fraction
  * mean_deg_drop   - drop in mean degree
  * ncomp_gain      - increase in number of connected components
  * crit_removed    - fraction of true critical agents removed (mission damage)

Usage:
  python gnn/damage_at_k.py --preds results/gnn_preds.csv \
      --k-list 1,3,5,10 --max-snapshots 500 --out results/damage_at_k.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from dataset import load_snapshots, grouped_split


def build_adj(pos, graph_type="true", k=12):
    n = len(pos)
    adj = [[] for _ in range(n)]
    if graph_type == "true":
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) <= 55.0:
                    adj[i].append(j); adj[j].append(i)
    elif graph_type == "knn":
        for i in range(n):
            d = np.linalg.norm(pos - pos[i], axis=1)
            d[i] = np.inf
            for j in np.argsort(d)[:min(k, n - 1)]:
                j = int(j)
                adj[i].append(j); adj[j].append(i)
        for i in range(n):  # dedupe
            adj[i] = sorted(set(adj[i]))
    else:
        raise ValueError(graph_type)
    return adj


def metrics(adj, keep_mask):
    """LCC fraction, #components, mean degree on the induced subgraph."""
    n = len(adj)
    kept = np.where(keep_mask)[0]
    nk = len(kept)
    if nk == 0:
        return {"lcc": 0.0, "ncomp": 0, "mean_deg": 0.0}
    visited = np.zeros(n, dtype=bool)
    comps = []
    for s in kept:
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            for w in adj[v]:
                if keep_mask[w] and not visited[w]:
                    visited[w] = True
                    stack.append(w)
        comps.append(size)
    deg_sum = sum(1 for v in kept for w in adj[v] if keep_mask[w])
    return {"lcc": max(comps) / nk,
            "ncomp": len(comps),
            "mean_deg": deg_sum / nk}


def fiedler_value(adj, keep_mask):
    """Algebraic connectivity (2nd-smallest Laplacian eigenvalue) of the
    induced subgraph. More sensitive than LCC for dense graphs."""
    kept = np.where(keep_mask)[0]
    nk = len(kept)
    if nk < 2:
        return 0.0
    idx = {v: i for i, v in enumerate(kept)}
    L = np.zeros((nk, nk))
    for v in kept:
        i = idx[v]
        for w in adj[v]:
            if keep_mask[w]:
                j = idx[w]
                L[i, j] -= 1.0
                L[i, i] += 1.0
    vals = np.linalg.eigvalsh(L)  # ascending
    return float(vals[1])


def betweenness(adj):
    """Brandes betweenness (undirected, unweighted), normalized."""
    n = len(adj)
    CB = np.zeros(n)
    for s in range(n):
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n); sigma[s] = 1.0
        d = np.full(n, -1); d[s] = 0
        Q = [s]
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in adj[v]:
                if d[w] < 0:
                    Q.append(w); d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = np.zeros(n)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                CB[w] += delta[w]
    CB /= 2.0
    if n > 2:
        CB /= ((n - 1) * (n - 2) / 2.0)
    return CB


def load_preds(path):
    """(case_id, step) -> {agent_id: prob}."""
    out = defaultdict(dict)
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            out[(r["case_id"], int(r["step"]))][int(r["agent_id"])] = float(r["prob"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    ap.add_argument("--preds", required=True, help="CSV from train_gnn.py --save-preds")
    ap.add_argument("--graph", choices=["true", "knn"], default="true",
                    help="connectivity graph used to measure damage")
    ap.add_argument("--knn-k", type=int, default=12)
    ap.add_argument("--k-list", default="1,3,5,10")
    ap.add_argument("--max-snapshots", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5, help="random-strategy repeats")
    ap.add_argument("--betweenness", action="store_true",
                    help="also evaluate the betweenness-centrality strategy (slower)")
    ap.add_argument("--fiedler", action="store_true",
                    help="also report algebraic-connectivity (Fiedler value) damage "
                         "(more sensitive than LCC for dense graphs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ks = [int(x) for x in args.k_list.split(",")]
    rng = np.random.default_rng(args.seed)

    print("[load] snapshots...")
    snapshots, case_meta, _ = load_snapshots(args.data)
    _, test_cases = grouped_split([s["case_id"] for s in snapshots], case_meta, seed=args.seed)
    test_snaps = [s for s in snapshots if s["case_id"] in test_cases]
    if len(test_snaps) > args.max_snapshots:
        idx = np.linspace(0, len(test_snaps) - 1, args.max_snapshots).astype(int)
        test_snaps = [test_snaps[i] for i in idx]
    print(f"[data] evaluating {len(test_snaps)} test snapshots")

    preds = load_preds(args.preds)

    strategies = ["model", "random", "degree", "critical"]
    if args.betweenness:
        strategies.append("betweenness")
    acc = {s: {k: defaultdict(list) for k in ks} for s in strategies}
    base_acc = defaultdict(list)
    n_used = 0

    for snap in test_snaps:
        key = (snap["case_id"], int(snap["step"]))
        pm = preds.get(key)
        if pm is None:
            continue
        aid = snap["aid"]
        if any(int(a) not in pm for a in aid):
            continue
        scores = np.array([pm[int(a)] for a in aid])
        y = np.asarray(snap["y"])
        pos = np.asarray(snap["pos"])
        n = len(aid)
        adj = build_adj(pos, args.graph, args.knn_k)

        keep_all = np.ones(n, dtype=bool)
        base = metrics(adj, keep_all)
        base_acc["lcc"].append(base["lcc"])
        base_acc["ncomp"].append(base["ncomp"])
        base_acc["mean_deg"].append(base["mean_deg"])
        base_fiedler = fiedler_value(adj, keep_all) if args.fiedler else None
        if args.fiedler:
            base_acc["fiedler"].append(base_fiedler)
        n_used += 1
        crit_total = max(int(y.sum()), 1)

        deg = np.array([len(a) for a in adj])
        order = {
            "model": np.argsort(-scores),
            "degree": np.argsort(-deg),
            "critical": np.concatenate([
                np.where(y > 0)[0][np.argsort(-scores[np.where(y > 0)[0]])],
                np.where(y <= 0)[0][np.argsort(-scores[np.where(y <= 0)[0]])],
            ]),
        }
        if args.betweenness:
            order["betweenness"] = np.argsort(-betweenness(adj))

        for strat in strategies:
            if strat == "random":
                continue
            ordv = order[strat]
            for k in ks:
                kk = min(k, n - 1)
                removed = ordv[:kk]
                keep = keep_all.copy()
                keep[removed] = False
                m = metrics(adj, keep)
                acc[strat][k]["lcc_drop"].append(1 - m["lcc"] / base["lcc"])
                acc[strat][k]["mean_deg_drop"].append(1 - m["mean_deg"] / base["mean_deg"])
                acc[strat][k]["ncomp_gain"].append(m["ncomp"] - base["ncomp"])
                acc[strat][k]["crit_removed"].append(float(y[removed].sum()) / crit_total)
                if args.fiedler and base_fiedler and base_fiedler > 0:
                    fv = fiedler_value(adj, keep)
                    acc[strat][k]["fiedler_drop"].append(1 - fv / base_fiedler)

        # random strategy: average over repeats
        for k in ks:
            kk = min(k, n - 1)
            lcc_d, deg_d, comp_g, crit_r, fied_d = [], [], [], [], []
            for _ in range(args.repeats):
                removed = rng.choice(n, size=kk, replace=False)
                keep = keep_all.copy()
                keep[removed] = False
                m = metrics(adj, keep)
                lcc_d.append(1 - m["lcc"] / base["lcc"])
                deg_d.append(1 - m["mean_deg"] / base["mean_deg"])
                comp_g.append(m["ncomp"] - base["ncomp"])
                crit_r.append(float(y[removed].sum()) / crit_total)
                if args.fiedler and base_fiedler and base_fiedler > 0:
                    fied_d.append(1 - fiedler_value(adj, keep) / base_fiedler)
            acc["random"][k]["lcc_drop"].append(float(np.mean(lcc_d)))
            acc["random"][k]["mean_deg_drop"].append(float(np.mean(deg_d)))
            acc["random"][k]["ncomp_gain"].append(float(np.mean(comp_g)))
            acc["random"][k]["crit_removed"].append(float(np.mean(crit_r)))
            if fied_d:
                acc["random"][k]["fiedler_drop"].append(float(np.mean(fied_d)))

    results = {}
    for strat in strategies:
        results[strat] = {}
        for k in ks:
            results[strat][str(k)] = {
                metric: float(np.mean(vals)) for metric, vals in acc[strat][k].items()
            }

    summary = {
        "settings": {
            "graph": args.graph, "knn_k": args.knn_k,
            "k_list": ks, "max_snapshots": args.max_snapshots,
            "repeats": args.repeats, "betweenness": args.betweenness,
            "fiedler": args.fiedler,
            "n_snapshots_used": n_used,
        },
        "baseline": {m: float(np.mean(v)) for m, v in base_acc.items()},
        "results": results,
    }

    print("\n===== DAMAGE-AT-K =====")
    print(f"baseline: LCC={summary['baseline']['lcc']:.3f} "
          f"components={summary['baseline']['ncomp']:.2f} "
          f"mean_deg={summary['baseline']['mean_deg']:.2f}"
          + (f" fiedler={summary['baseline']['fiedler']:.2f}" if args.fiedler else ""))
    header = f"{'strategy':12s} {'k':>3s} {'LCC drop':>9s} {'deg drop':>9s} {'+comps':>7s} {'crit removed':>13s}"
    if args.fiedler:
        header += f" {'fiedler drop':>13s}"
    print(header)
    print("-" * len(header))
    for strat in strategies:
        for k in ks:
            r = results[strat][str(k)]
            line = (f"{strat:12s} {k:3d} {r['lcc_drop']:9.3f} {r['mean_deg_drop']:9.3f} "
                    f"{r['ncomp_gain']:7.2f} {r['crit_removed']:13.3f}")
            if args.fiedler:
                line += f" {r.get('fiedler_drop', float('nan')):13.3f}"
            print(line)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
