# Cross-simulator transfer pipeline

Generates swarm data with `gym-pybullet-drones` and scores it with the trained
target-blind GNN. Runs in the `.venv-drones` environment (has `gym-pybullet-drones`
and `torch`).

## Files
- `run_scenario.py` — drives a pybullet scenario and logs raw per-step state
  (`step, agent_id, x, y, z, vx, vy, vz`).
- `extract_features.py` — raw state → 13 target-blind features → z-score with
  `feature_stats.json` → build k-NN graph → optional GNN scoring.
- `feature_stats.json` — per-feature training mean/std used for standardization.
- `RESULTS.md` — the transfer results and interpretation.
- `output/` — generated raw-state and score CSVs.

## Scenarios
`single_relay_mesh`, `moving_escort`, `escort_many`, `two_cores`,
`relay_bridge`, `offcenter_core`, `orbiting_escort`.

## Pipeline
```bash
source .venv-drones/bin/activate
python src/sim/run_scenario.py --scenario moving_escort --steps 300 --speed 1.5 \
    --out src/sim/output/raw.csv
python src/sim/extract_features.py --raw src/sim/output/raw.csv \
    --checkpoint results/no_target/controls/gnn_knn_targetblind.pt \
    --out src/sim/output/scores.csv
```

## Notes
- The simulator provides no critical label. Criticality is assigned by formation
  design (inner/central = critical), matching the original dataset's concept.
- pybullet uses meters and m/s. Standardization is required; without it the
  model sees out-of-distribution scales.
- Drones must move (not be frozen) so motion features stay in a plausible range.
- The model identifies criticals by *global* centroid proximity and expects
  moving agents; see `RESULTS.md` for the scenarios where it fails.
