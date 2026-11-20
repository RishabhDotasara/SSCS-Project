#!/usr/bin/env bash
# Reproduce every number in paper/paper.tex from scratch.
#
# Runs the full pipeline in order and writes all results to results/ and
# src/sim/output/. Assumes the two environments already exist (see README):
#   * venv/          -- numpy, scikit-learn, xgboost, torch (dataset models)
#   * .venv-drones/  -- gym-pybullet-drones + torch (cross-simulator transfer)
#
# Usage (from the repo root):
#   bash reproduce.sh
#
# Set SKIP_TABULAR=1 / SKIP_GNN=1 / SKIP_CROSSSIM=1 to skip stages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=venv/bin/python
PYD=.venv-drones/bin/python
DATA=data/swarm_criticality_dataset_100_scenarios.csv
R=results/no_target
CKPT=$R/controls/gnn_knn_targetblind.pt
PREDS=$R/gnn_knn_preds.csv

info()  { printf '\n\033[1;34m==== %s ====\033[0m\n' "$*"; }
step()  { printf '\033[1;32m  -- %s\033[0m\n' "$*"; }

mkdir -p "$R/controls" "$R/tuned" src/sim/output

# ---------------------------------------------------------------------------
# Stage 1: tabular baselines
# ---------------------------------------------------------------------------
if [ "${SKIP_TABULAR:-0}" != "1" ]; then
  info "Stage 1: tabular baselines"
  step "RF motion_only"
  $PY src/train_rf.py --variant motion_only --n-trees 300 --out "$R/rf_motion_only.json"
  step "RF node_only"
  $PY src/train_rf.py --variant node_only --n-trees 300 --out "$R/rf_node_only.json"
  step "RF local_features"
  $PY src/train_rf.py --variant local_features --n-trees 300 --out "$R/rf_local_features.json"
  step "RF local_features + environment metadata (ablation)"
  $PY src/train_rf.py --variant local_features --include-environment --n-trees 300 \
      --out "$R/rf_local_with_environment.json"
  step "XGBoost (300 trees)"
  $PY src/train_xgb.py --out "$R/xgb_local_features.json"
  step "Hyperparameter tuning (RF + XGBoost)"
  $PY src/tune_baselines.py --out-dir "$R/tuned"
fi

# ---------------------------------------------------------------------------
# Stage 2: GNN training (main k-NN checkpoint + radius + controls)
# ---------------------------------------------------------------------------
if [ "${SKIP_GNN:-0}" != "1" ]; then
  info "Stage 2: GNN training"
  step "k-NN GNN (saves checkpoint + per-agent predictions)"
  $PY src/gnn/train_gnn.py --graph knn --k 12 --conv sage --epochs 30 \
      --checkpoint-out "$CKPT" --save-preds "$PREDS" \
      --out "$R/gnn_knn_sage.json"
  step "Radius graph GNN (R=55)"
  $PY src/gnn/train_gnn.py --graph true --conv sage --epochs 30 \
      --out "$R/gnn_true_sage.json"
  step "Graph control: no messages"
  $PY src/gnn/train_gnn.py --graph none --conv sage --epochs 30 \
      --out "$R/controls/gnn_graphless.json"
  step "Graph control: degree-preserving rewired"
  $PY src/gnn/train_gnn.py --graph rewired --k 12 --conv sage --epochs 30 \
      --out "$R/controls/gnn_rewired.json"
fi

# ---------------------------------------------------------------------------
# Stage 3: robustness evals (all reuse the saved k-NN checkpoint)
# ---------------------------------------------------------------------------
if [ "${SKIP_GNN:-0}" != "1" ]; then
  info "Stage 3: robustness evaluations"
  step "Position noise (1 / 3 / 5 units)"
  for n in 1 3 5; do
    $PY src/gnn/train_gnn.py --graph knn --k 12 --conv sage \
        --checkpoint-in "$CKPT" --position-noise-std "$n" \
        --out "$R/controls/gnn_knn_noise${n}.json"
  done
  step "Graph-construction uncertainty (k variants)"
  $PY src/gnn/eval_graph_uncertainty.py --checkpoint "$CKPT" \
      --graph knn --k 12 --out-dir "$R/controls"
  step "Damage-at-k"
  $PY src/gnn/damage_at_k.py --preds "$PREDS" \
      --k-list 1,3,5,10 --max-snapshots 500 --out "$R/damage_at_k.json"
  step "Rank displacement"
  $PY src/gnn/rank_displacement.py --preds "$PREDS" --data "$DATA" \
      --out "$R/rank_displacement.json"
fi

# ---------------------------------------------------------------------------
# Stage 4: cross-simulator transfer (gym-pybullet-drones)
# ---------------------------------------------------------------------------
if [ "${SKIP_CROSSSIM:-0}" != "1" ]; then
  info "Stage 4: cross-simulator transfer"
  for sc in moving_escort single_relay_mesh escort_many two_cores relay_bridge; do
    step "scenario: $sc"
    $PYD src/sim/run_scenario.py --scenario "$sc" --steps 300 --speed 1.5 \
        --out "src/sim/output/raw_${sc}.csv"
    $PYD src/sim/extract_features.py --raw "src/sim/output/raw_${sc}.csv" \
        --checkpoint "$CKPT" --out "src/sim/output/${sc}_scores.csv"
  done
  step "Summarizing transfer results"
  $PYD src/sim/summarize_transfer.py
fi

info "Done. Results are in $R/ and src/sim/output/."
