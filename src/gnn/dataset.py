"""
Graph dataset builder for the swarm criticality GNN (attacker POV).

Builds one graph per (scenario, timestep) snapshot:
  * nodes  = alive agents, features = physically-observable columns only
  * edges  = reconstructed proximity graph (k-NN or true 55-unit ball)
  * label  = is_critical (ground truth, for training/eval only)

Feature choice ("don't use anything we can't get physically"):
  positions/velocities come from tracking; centroid and ranks are derived from
  the tracked positions; accel/heading_rate are per-row motion/control inputs.
  The mission target is assumed unknown, so target-distance features are not
  included in PHYSICAL_FEATURES at all.

No criticality_score, no graph-centrality columns, no was_promoted.
"""
import csv
from collections import defaultdict

import numpy as np
import torch

PHYSICAL_FEATURES = [
    "rel_x", "rel_y", "rel_z",
    "dist_to_centroid", "rank_dist_to_centroid",
    "vx", "vy", "vz",
    "speed", "heading", "flight_path_angle",
    "accel", "heading_rate",
]

ATTRITION_MAP = {"none": 0, "light": 1, "medium": 2, "heavy": 3}


def load_snapshots(path):
    """Stream the CSV and return (snapshots, case_meta, features).

    Each snapshot is a dict: case_id, step, x (n x d float32), y (n,),
    pos (n x 3 float32). The file is already sorted by case -> step -> agent.
    """
    features = PHYSICAL_FEATURES[:]
    snapshots = []
    case_meta = {}  # case_id -> (attrition_code, num_agents)
    cur_key = None
    buf = []

    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        ix = {c: i for i, c in enumerate(header)}
        feat_ix = [ix[f] for f in features]
        for r in rd:
            case_id = r[ix["case_id"]]
            step = int(r[ix["step"]])
            key = (case_id, step)
            if key != cur_key:
                if cur_key is not None:
                    snapshots.append(_pack(cur_key, buf, ix, feat_ix))
                cur_key = key
                buf = []
            if case_id not in case_meta:
                case_meta[case_id] = (ATTRITION_MAP[r[ix["attrition_rate"]]],
                                      int(r[ix["num_agents"]]))
            buf.append(r)
        if cur_key is not None:
            snapshots.append(_pack(cur_key, buf, ix, feat_ix))

    return snapshots, case_meta, features


def _pack(key, rows, ix, feat_ix):
    n = len(rows)
    d = len(feat_ix)
    x = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    cs = np.empty(n, dtype=np.float64)
    aid = np.empty(n, dtype=np.int64)
    pos = np.empty((n, 3), dtype=np.float32)
    rx = ix["rel_x"]; ry = ix["rel_y"]; rz = ix["rel_z"]
    ic = ix["is_critical"]; ia = ix["agent_id"]; ics = ix["criticality_score"]
    for i, r in enumerate(rows):
        for j, ci in enumerate(feat_ix):
            x[i, j] = float(r[ci])
        pos[i, 0] = float(r[rx])
        pos[i, 1] = float(r[ry])
        pos[i, 2] = float(r[rz])
        y[i] = 1.0 if r[ic] == "True" else 0.0
        cs[i] = float(r[ics])
        aid[i] = int(r[ia])
    return {"case_id": key[0], "step": key[1], "x": x, "y": y, "cs": cs,
            "pos": pos, "aid": aid}


# ---- edge builders ----------------------------------------------------------

def _edge_index_from_pairs(pairs, n):
    src, dst = [], []
    for i, j in pairs:
        src.extend((i, j))
        dst.extend((j, i))
    for i in range(n):
        src.append(i)
        dst.append(i)
    return torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)


def _degree_preserving_rewire(pairs, seed, attempts_per_edge=3):
    """Randomize undirected edges using swaps that preserve every node degree."""
    edges = set((min(i, j), max(i, j)) for i, j in pairs if i != j)
    if len(edges) < 2:
        return edges
    rng = np.random.default_rng(seed)
    edge_list = list(edges)
    for _ in range(attempts_per_edge * len(edge_list)):
        ai, bi = rng.choice(len(edge_list), size=2, replace=False)
        a, b = edge_list[ai]
        c, d = edge_list[bi]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < 0.5:
            new1 = (min(a, d), max(a, d))
            new2 = (min(c, b), max(c, b))
        else:
            new1 = (min(a, c), max(a, c))
            new2 = (min(b, d), max(b, d))
        if new1 == new2 or new1 in edges or new2 in edges:
            continue
        edges.remove(edge_list[ai])
        edges.remove(edge_list[bi])
        edges.add(new1)
        edges.add(new2)
        edge_list[ai] = new1
        edge_list[bi] = new2
    return edges


def build_edges(pos, graph_type="knn", k=12, seed=42):
    """Build an undirected edge_index with self-loops.

    ``none`` is a self-loop-only, graphless control. ``rewired`` starts with
    k-NN edges and randomizes their endpoints while preserving every degree.
    """
    n = pos.shape[0]
    if graph_type == "none" or n <= 1:
        return torch.arange(n, dtype=torch.long).repeat(2, 1)

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    pairs = set()
    if graph_type in ("knn", "rewired"):
        kk = min(k, n - 1)
        order = np.argsort(d, axis=1)[:, :kk]
        for i in range(n):
            for j in order[i]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    elif graph_type == "true":
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= 55.0:
                    pairs.add((i, j))
    elif graph_type == "full":
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    else:
        raise ValueError(f"unknown graph_type: {graph_type}")

    if graph_type == "rewired":
        pairs = _degree_preserving_rewire(pairs, seed)
    return _edge_index_from_pairs(pairs, n)


# ---- split (identical to the RF baseline for fair comparison) ---------------

def grouped_split(case_ids, case_meta, train_frac=0.8, seed=42, min_test=1):
    rng = np.random.default_rng(seed)
    by_combo = defaultdict(list)
    for case in sorted(set(case_ids)):
        att, n = case_meta[case]
        by_combo[(att, n)].append(case)
    train_cases, test_cases = set(), set()
    for combo, cset in by_combo.items():
        rng.shuffle(cset)
        n_test = max(min_test, int(round(len(cset) * (1 - train_frac))))
        test_cases.update(cset[:n_test])
        train_cases.update(cset[n_test:])
    return train_cases, test_cases


class SwarmGraphDataset:
    """Holds snapshots as pre-built graph tensors, split into train/test."""

    def __init__(self, snapshots, case_meta, graph_type="knn", k=12, seed=42):
        self.features_dim = snapshots[0]["x"].shape[1] if snapshots else 0
        train_cases, test_cases = grouped_split(
            [s["case_id"] for s in snapshots], case_meta, seed=seed)
        self.train = self._build(snapshots, train_cases, graph_type, k)
        self.test = self._build(snapshots, test_cases, graph_type, k)
        self.n_train_cases = len(train_cases)
        self.n_test_cases = len(test_cases)

    def _build(self, snapshots, case_set, graph_type, k):
        out = []
        for s in snapshots:
            if s["case_id"] not in case_set:
                continue
            stable_seed = sum((i + 1) * ord(c) for i, c in enumerate(s["case_id"])) + int(s["step"])
            ei = build_edges(s["pos"], graph_type, k, seed=stable_seed)
            out.append({
                "case_id": s["case_id"],
                "step": s["step"],
                "x": torch.from_numpy(s["x"]),
                "y": torch.from_numpy(s["y"]),
                "cs": torch.from_numpy(s["cs"]),
                "aid": torch.from_numpy(s["aid"]),
                "pos": s["pos"],
                "edge_index": ei,
                "n": s["x"].shape[0],
            })
        return out

    @staticmethod
    def collate(batch):
        """Batch variable-size graphs into (x, edge_index, y, aid, slices, cs)."""
        x = torch.cat([b["x"] for b in batch], dim=0)
        y = torch.cat([b["y"] for b in batch], dim=0)
        aid = torch.cat([b["aid"] for b in batch], dim=0)
        cs = torch.cat([b["cs"] for b in batch], dim=0)
        edges = []
        off = 0
        slices = [0]
        for b in batch:
            edges.append(b["edge_index"] + off)
            off += b["n"]
            slices.append(off)
        edge_index = torch.cat(edges, dim=1)
        return x, edge_index, y, aid, slices, cs
