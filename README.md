# SSCS Project — Critical-Agent Ranking in Aerial Swarms

Identifies the critical (high-value) agents in a simulated aerial swarm from
target-blind telemetry, using tabular baselines (random forest, XGBoost) and a
message-passing graph neural network (GNN).

## Layout

```
data/         dataset documentation (the CSV itself is not tracked — see below)
src/          all code
  ├── train_rf.py, train_xgb.py, tune_baselines.py   tabular models
  ├── analyze_scenarios.py                           dataset vs. test coverage
  ├── gnn/                                           GNN models and evaluation
  │   ├── dataset.py, model.py, train_gnn.py
  │   ├── damage_at_k.py, eval_graph_uncertainty.py, rank_displacement.py
  └── sim/                                           cross-simulator (PyBullet)
      ├── run_scenario.py, extract_features.py, feature_stats.json
      ├── summarize_transfer.py, README.md
      └── output/                                    generated CSVs
reproduce.sh   one-command reproduction of every experiment
run.sh         interactive setup + model launcher
requirements.txt
```

## Data

The dataset `swarm_criticality_dataset_100_scenarios.csv` (~880 MB) is **not**
tracked by git (it exceeds GitHub's 100 MB per-file limit). Place it at
`data/swarm_criticality_dataset_100_scenarios.csv` before running anything.
Its columns are documented in `data/README.md`.

## Environments

Two Python environments, created once at the repo root:

```bash
# 1) dataset models (RF, XGBoost, GNN)
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 2) cross-simulator transfer (gym-pybullet-drones)
python3 -m venv .venv-drones
.venv-drones/bin/pip install numpy torch
.venv-drones/bin/pip install pybullet            # needs gcc/build-essential
git clone https://github.com/learnsyslab/gym-pybullet-drones.git
.venv-drones/bin/pip install -e gym-pybullet-drones
```

(Or run `bash run.sh`, which sets these up and offers an interactive menu.)

## Run everything

From the repository root:

```bash
bash reproduce.sh
```

This runs the full pipeline end-to-end and writes results to `results/` and
`src/sim/output/`:

1. **Tabular** — RF feature ladder (motion → +position → +local), XGBoost,
   hyperparameter tuning.
2. **GNN** — train the k-NN GNN (saves a checkpoint + per-agent predictions),
   the radius-graph comparison, and the no-message / rewired controls.
3. **Robustness** — position noise (1/3/5), graph-construction uncertainty,
   damage-at-k, and rank-displacement vs `criticality_score`.
4. **Cross-simulator** — five PyBullet formations, scored by the unmodified GNN.

Skip a stage with e.g. `SKIP_TABULAR=1 bash reproduce.sh`.

## Run individual pieces

### Tabular models (`venv`)
```bash
venv/bin/python src/train_rf.py --variant motion_only    --n-trees 300 --out results/no_target/rf_motion_only.json
venv/bin/python src/train_rf.py --variant node_only      --n-trees 300 --out results/no_target/rf_node_only.json
venv/bin/python src/train_rf.py --variant local_features --n-trees 300 --out results/no_target/rf_local_features.json
venv/bin/python src/train_xgb.py --out results/no_target/xgb_local_features.json
venv/bin/python src/tune_baselines.py --out-dir results/no_target/tuned
```

### GNN (`venv`)
```bash
# main target-blind k-NN model (checkpoint + per-agent predictions)
venv/bin/python src/gnn/train_gnn.py --graph knn --k 12 --conv sage --epochs 30 \
    --checkpoint-out results/no_target/controls/gnn_knn_targetblind.pt \
    --save-preds results/no_target/gnn_knn_preds.csv \
    --out results/no_target/gnn_knn_sage.json

# radius-graph comparison + controls
venv/bin/python src/gnn/train_gnn.py --graph true    --conv sage --epochs 30 --out results/no_target/gnn_true_sage.json
venv/bin/python src/gnn/train_gnn.py --graph none    --conv sage --epochs 30 --out results/no_target/controls/gnn_graphless.json
venv/bin/python src/gnn/train_gnn.py --graph rewired --k 12 --conv sage --epochs 30 --out results/no_target/controls/gnn_rewired.json

# robustness, all reusing the saved checkpoint
venv/bin/python src/gnn/train_gnn.py --graph knn --k 12 --conv sage \
    --checkpoint-in results/no_target/controls/gnn_knn_targetblind.pt \
    --position-noise-std 5 --out results/no_target/controls/gnn_knn_noise5.json
venv/bin/python src/gnn/eval_graph_uncertainty.py \
    --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
    --graph knn --k 12 --out-dir results/no_target/controls
venv/bin/python src/gnn/damage_at_k.py --preds results/no_target/gnn_knn_preds.csv \
    --k-list 1,3,5,10 --max-snapshots 500 --out results/no_target/damage_at_k.json
venv/bin/python src/gnn/rank_displacement.py --preds results/no_target/gnn_knn_preds.csv \
    --out results/no_target/rank_displacement.json
```

### Cross-simulator transfer (`.venv-drones`)
```bash
.venv-drones/bin/python src/sim/run_scenario.py --scenario moving_escort --steps 300 --speed 1.5 \
    --out src/sim/output/raw_moving_escort.csv
.venv-drones/bin/python src/sim/extract_features.py --raw src/sim/output/raw_moving_escort.csv \
    --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
    --out src/sim/output/moving_escort_scores.csv
.venv-drones/bin/python src/sim/summarize_transfer.py
```

## Key results

- RF ladder (PR-AUC): motion 0.772 → +position 0.900 → +local geometry 0.963.
- Tuned XGBoost (local features): 0.969.
- Target-blind GNN (k-NN): 0.964; graphless control 0.893; rewired 0.952.
- Damage-at-10: GNN removes 62.8% of criticals (random 29.1%, oracle 66.3%).
- Rank displacement of criticals: 0.119 (random baseline 0.369); Spearman 0.717.
- Cross-simulator: the unmodified GNN ranks every critical agent in the top-k
  across five PyBullet formations, with high confidence for central escorted
  agents and low confidence for bridge relays.
