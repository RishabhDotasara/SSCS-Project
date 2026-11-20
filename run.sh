#!/usr/bin/env bash
# Setup + run script for the critical-agent ranking project.
#
# Phase 1: create the two Python environments and install packages.
# Phase 2: ask before running.
# Phase 3: interactive menu to start the models.
#
# Run from the repository root:
#   bash run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

info() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Phase 1: environments and packages
# ---------------------------------------------------------------------------

# Main environment (dataset models: RF, XGBoost, GNN).
if [ ! -d "venv" ]; then
    info "Creating main environment 'venv' (this installs torch, ~hundreds of MB)..."
    python3 -m venv venv
    venv/bin/pip install --upgrade pip
    venv/bin/pip install numpy scikit-learn xgboost torch
else
    ok "Main environment 'venv' already exists."
fi

# Drones environment (cross-simulator transfer).
if [ ! -d ".venv-drones" ]; then
    info "Creating drones environment '.venv-drones'..."
    python3 -m venv .venv-drones
    .venv-drones/bin/pip install --upgrade pip
    .venv-drones/bin/pip install numpy torch
    warn "pybullet needs a C compiler (gcc/build-essential on Linux)."
    .venv-drones/bin/pip install pybullet
else
    ok "Drones environment '.venv-drones' already exists."
fi

if [ ! -d "gym-pybullet-drones" ]; then
    info "Cloning gym-pybullet-drones..."
    git clone https://github.com/learnsyslab/gym-pybullet-drones.git
fi
.venv-drones/bin/pip install -e gym-pybullet-drones

ok "Setup complete."

# ---------------------------------------------------------------------------
# Phase 2: confirmation
# ---------------------------------------------------------------------------
printf '\nStart the models now? [y/N] '
read -r answer
case "$answer" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Exiting."; exit 0 ;;
esac

# ---------------------------------------------------------------------------
# Phase 3: interactive menu
# ---------------------------------------------------------------------------
while true; do
    echo
    echo "Select a model to run:"
    echo "  1) Train target-blind RF (local_features)"
    echo "  2) Train target-blind XGBoost"
    echo "  3) Tune RF + XGBoost"
    echo "  4) Train target-blind GNN (k-NN, saves checkpoint)"
    echo "  5) Damage-at-k"
    echo "  6) Graph-uncertainty evaluation"
    echo "  7) Cross-simulator scenario + score"
    echo "  8) Exit"
    printf 'Choice [1-8]: '
    read -r choice
    case "$choice" in
        1)
            venv/bin/python src/train_rf.py --variant local_features --n-trees 300 \
                --out results/no_target/rf_local_features.json
            ;;
        2)
            venv/bin/python src/train_xgb.py --out results/no_target/xgb_local_features.json
            ;;
        3)
            venv/bin/python src/tune_baselines.py --out-dir results/no_target/tuned
            ;;
        4)
            mkdir -p results/no_target/controls
            venv/bin/python src/gnn/train_gnn.py --graph knn --k 12 --conv sage --epochs 30 \
                --save-preds results/no_target/gnn_knn_preds.csv \
                --checkpoint-out results/no_target/controls/gnn_knn_targetblind.pt \
                --out results/no_target/gnn_knn_sage.json
            ;;
        5)
            if [ ! -f results/no_target/gnn_knn_preds.csv ]; then
                warn "Run option 4 first to generate the predictions CSV."
                continue
            fi
            venv/bin/python src/gnn/damage_at_k.py \
                --preds results/no_target/gnn_knn_preds.csv \
                --k-list 1,3,5,10 --max-snapshots 500 \
                --out results/no_target/damage_at_k.json
            ;;
        6)
            if [ ! -f results/no_target/controls/gnn_knn_targetblind.pt ]; then
                warn "Run option 4 first to generate the checkpoint."
                continue
            fi
            venv/bin/python src/gnn/eval_graph_uncertainty.py \
                --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
                --graph knn --k 12 --out-dir results/no_target/controls
            ;;
        7)
            if [ ! -f results/no_target/controls/gnn_knn_targetblind.pt ]; then
                warn "Run option 4 first to generate the checkpoint."
                continue
            fi
            mkdir -p src/sim/output
            .venv-drones/bin/python src/sim/run_scenario.py \
                --scenario moving_escort --steps 300 --speed 1.5 \
                --out src/sim/output/raw.csv
            .venv-drones/bin/python src/sim/extract_features.py \
                --raw src/sim/output/raw.csv \
                --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
                --out src/sim/output/scores.csv
            ;;
        8)
            echo "Done."
            exit 0
            ;;
        *)
            echo "Invalid choice."
            ;;
    esac
done
