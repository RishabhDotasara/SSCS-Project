#!/usr/bin/env python3
"""
Reusable target-blind feature extractor for cross-simulator transfer.

Takes a raw per-step state log and produces the same 13 node features the GNN
was trained on, standardizes them to the original training distribution, and
optionally applies a saved GNN checkpoint to score every agent.

Raw input CSV columns (one row per agent per step):
    step, agent_id, x, y, z, vx, vy, vz

Feature computation (per snapshot = all agents at one step):
    rel_x/y/z          = position minus swarm centroid
    dist_to_centroid   = ||rel||
    rank_dist_to_centroid = (# agents strictly closer) / (n_alive - 1)
    speed              = ||v||
    heading            = atan2(vy, vx)
    flight_path_angle  = asin(vz / speed)
    accel              = (speed_t - speed_{t-1}) / dt
    heading_rate       = wrapped(heading_t - heading_{t-1}) / dt

Standardization uses sim/feature_stats.json (computed from the original
training data). Rank and the two angles are passed through unchanged; all other
features are z-scored with the training mean/std.

Usage:
    python sim/extract_features.py --raw raw_states.csv \
        --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
        --k 12 --out features_and_scores.csv
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

FEATURES = ["rel_x", "rel_y", "rel_z", "dist_to_centroid", "rank_dist_to_centroid",
            "vx", "vy", "vz", "speed", "heading", "flight_path_angle",
            "accel", "heading_rate"]


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def read_raw(path):
    """Return dict: step -> list of (agent_id, x, y, z, vx, vy, vz)."""
    steps = defaultdict(list)
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            s = int(r["step"])
            steps[s].append((int(r["agent_id"]),
                             float(r["x"]), float(r["y"]), float(r["z"]),
                             float(r["vx"]), float(r["vy"]), float(r["vz"])))
    return steps


def build_features(steps, dt=1.0):
    """Return a list of (step, agent_id, feature_vector) in reading order."""
    out = []
    prev = {}  # agent_id -> (speed, heading)
    for s in sorted(steps):
        rows = steps[s]
        # centroid from all agents alive at this step
        pos = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float64)
        centroid = pos.mean(axis=0)
        n = len(rows)
        feat_by_agent = {}
        for aid, x, y, z, vx, vy, vz in rows:
            rel = np.array([x, y, z]) - centroid
            d = float(np.linalg.norm(rel))
            rank = float((np.linalg.norm(pos - centroid, axis=1) < d).sum()) / max(n - 1, 1)
            sp = math.sqrt(vx * vx + vy * vy + vz * vz)
            heading = math.atan2(vy, vx)
            fpa = math.asin(max(-1.0, min(1.0, vz / sp))) if sp > 0 else 0.0
            pv = prev.get(aid)
            accel = (sp - pv[0]) / dt if pv is not None else 0.0
            hrate = wrap_pi(heading - pv[1]) / dt if pv is not None else 0.0
            prev[aid] = (sp, heading)
            feat_by_agent[aid] = [rel[0], rel[1], rel[2], d, rank,
                                  vx, vy, vz, sp, heading, fpa, accel, hrate]
        # keep reading order for the output
        for aid, *_ in rows:
            out.append((s, aid, feat_by_agent[aid]))
    return out


def standardize(features_rows, stats_path):
    with open(stats_path) as fh:
        stats = json.load(fh)
    for name, st in stats.items():
        assert name in FEATURES
    idx = {f: i for i, f in enumerate(FEATURES)}
    out = []
    for s, aid, vec in features_rows:
        v = list(vec)
        for f in FEATURES:
            st = stats[f]
            if st.get("keep_raw"):
                continue
            v[idx[f]] = (v[idx[f]] - st["mean"]) / st["std"]
        out.append((s, aid, v))
    return out


def build_graphs(rows, k=12):
    """Return per-step k-NN edge lists using standardized relative positions."""
    steps = defaultdict(list)
    for s, aid, vec in rows:
        steps[s].append((aid, np.array([vec[0], vec[1], vec[2]], dtype=np.float64)))
    # relative coords are already centroid-centered, so distances are correct.
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw state CSV from the simulator")
    ap.add_argument("--stats", default="src/sim/feature_stats.json")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="time between logged steps (for accel/heading_rate)")
    ap.add_argument("--checkpoint", default=None, help="optional saved GNN weights")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--neutralize-motion", action="store_true",
                    help="zero out motion features (vx/vy/vz/speed/accel/heading_rate "
                         "and angles) so the model uses only position + rank + graph")
    ap.add_argument("--out", default="src/sim/output/extracted.csv")
    args = ap.parse_args()

    steps = read_raw(args.raw)
    rows = build_features(steps, dt=args.dt)
    std_rows = standardize(rows, args.stats)

    if args.neutralize_motion:
        motion_idx = [FEATURES.index(f) for f in
                      ("vx", "vy", "vz", "speed", "heading",
                       "flight_path_angle", "accel", "heading_rate")]
        std_rows = [(s, aid, [0.0 if i in motion_idx else v
                              for i, v in enumerate(vec)])
                    for s, aid, vec in std_rows]

    header = ["step", "agent_id"] + FEATURES
    scores = None
    if args.checkpoint:
        # Lazy import so this script works without torch when only extracting.
        _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        sys.path.insert(0, _src)
        import torch
        from gnn.model import GNN
        from gnn.dataset import build_edges as _build_edges
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNN(len(FEATURES), hidden=64, layers=2, conv="sage", dropout=0.2).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.eval()
        by_step = defaultdict(list)
        for s, aid, vec in std_rows:
            by_step[s].append((aid, vec))
        score_map = {}
        with torch.no_grad():
            for s in sorted(by_step):
                items = by_step[s]
                ids = [aid for aid, _ in items]
                x = torch.tensor([vec for _, vec in items], dtype=torch.float32).to(device)
                pos = np.array([[vec[0], vec[1], vec[2]] for _, vec in items], dtype=np.float32)
                ei = _build_edges(pos, "knn", args.k, seed=s).to(device)
                logits = model(x, ei)
                sc = torch.sigmoid(logits).cpu().numpy()
                for aid, val in zip(ids, sc):
                    score_map[(s, aid)] = float(val)
        header.append("score")
        scores = score_map

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for s, aid, vec in std_rows:
            row = [s, aid] + [f"{v:.6f}" for v in vec]
            if scores is not None:
                row.append(f"{scores[(s, aid)]:.6f}")
            w.writerow(row)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
