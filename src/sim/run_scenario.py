#!/usr/bin/env python3
"""
Scenario runner for cross-simulator data generation (gym-pybullet-drones).

Logs raw per-step drone state (step, agent_id, x, y, z, vx, vy, vz) that
`extract_features.py` turns into target-blind node features.

Scenarios:
  - single_relay_mesh: two clusters connected through one relay (connectivity).
  - moving_escort: one core agent at the centroid, escorts on a ring, all
    translating at a common velocity (matches the "inner = critical" concept).

Motion uses waypoint (PID) control: the whole formation's target positions are
advanced each step, so every drone translates at the requested speed while
keeping its relative position. This produces motion speeds comparable to the
original training data (0.5-8.5 m/s).
"""
import argparse
import csv

import numpy as np

from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType

COMMS_RADIUS = 100.0
CTRL_HZ = 30  # env.step() advances 1/CTRL_HZ seconds


def single_relay_mesh():
    positions = []
    for x in (-100.0, -80.0):
        for y in (-20.0, 20.0):
            positions.append([x, y, 3.0])
    for x in (80.0, 100.0):
        for y in (-20.0, 20.0):
            positions.append([x, y, 3.0])
    positions.append([0.0, 0.0, 3.0])  # single relay
    return np.array(positions)


def moving_escort(ring_radius=30.0, n_escorts=8):
    positions = [[0.0, 0.0, 3.0]]  # core agent at centroid
    for i in range(n_escorts):
        ang = 2 * np.pi * i / n_escorts
        positions.append([ring_radius * np.cos(ang), ring_radius * np.sin(ang), 3.0])
    return np.array(positions)


SCENARIOS = {
    "single_relay_mesh": single_relay_mesh,
    "moving_escort": moving_escort,
    "escort_many": lambda: _escort_many(),
    "two_cores": lambda: _two_cores(),
    "relay_bridge": lambda: _relay_bridge(),
    "offcenter_core": lambda: _offcenter_core(),
    "orbiting_escort": lambda: _orbiting_escort(),
}


def _offcenter_core():
    """A locally escorted core placed far from the global centroid (20 drones).

    The core sits at x=+40 with 5 escorts in a tight ring, while 14 drones are
    spread on the left, pulling the global centroid far from the core. The core
    is therefore "outer" by centroid distance but "inner" to its local escort.
    """
    positions = [[40.0, 0.0, 3.0]]  # core (agent 0)
    for i in range(5):
        ang = 2 * np.pi * i / 5
        positions.append([40.0 + 8.0 * np.cos(ang), 8.0 * np.sin(ang), 3.0])
    for x in (-120.0, -110.0, -100.0, -90.0, -80.0, -70.0, -60.0):
        for y in (-20.0, 20.0):
            positions.append([x, y, 3.0])
    return np.array(positions)


def _orbiting_escort(ring_radius=30.0, n_escorts=8):
    """1 core at centroid, escorts orbiting around it (9 drones)."""
    positions = [[0.0, 0.0, 3.0]]
    for i in range(n_escorts):
        ang = 2 * np.pi * i / n_escorts
        positions.append([ring_radius * np.cos(ang), ring_radius * np.sin(ang), 3.0])
    return np.array(positions)


def _escort_many(ring_radius=30.0, n_escorts=17):
    """1 core at centroid, 17 escorts on a ring (18 drones)."""
    positions = [[0.0, 0.0, 3.0]]
    for i in range(n_escorts):
        ang = 2 * np.pi * i / n_escorts
        positions.append([ring_radius * np.cos(ang), ring_radius * np.sin(ang), 3.0])
    return np.array(positions)


def _two_cores(ring_radius=35.0, n_escorts=16):
    """2 core agents near the centroid, 16 escorts on a ring (18 drones)."""
    positions = [[-8.0, 0.0, 3.0], [8.0, 0.0, 3.0]]
    for i in range(n_escorts):
        ang = 2 * np.pi * i / n_escorts
        positions.append([ring_radius * np.cos(ang), ring_radius * np.sin(ang), 3.0])
    return np.array(positions)


def _relay_bridge():
    """Two 9-drone clusters joined only by 2 relays (20 drones)."""
    positions = []
    for x in (-110.0, -100.0, -90.0):
        for y in (-10.0, 0.0, 10.0):
            positions.append([x, y, 3.0])
    for x in (90.0, 100.0, 110.0):
        for y in (-10.0, 0.0, 10.0):
            positions.append([x, y, 3.0])
    positions.append([0.0, -8.0, 3.0])  # relay 1 (agent 18)
    positions.append([0.0, 8.0, 3.0])   # relay 2 (agent 19)
    return np.array(positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="moving_escort")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--speed", type=float, default=1.5,
                    help="formation drift speed in m/s (waypoint control)")
    ap.add_argument("--out", default="src/sim/output/raw_states.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    positions = SCENARIOS[args.scenario]()
    num_drones = len(positions)

    env = MultiHoverAviary(
        drone_model=DroneModel.CF2X,
        num_drones=num_drones,
        initial_xyzs=positions,
        physics=Physics.PYB,
        act=ActionType.PID,
        gui=args.gui,
    )
    env.reset(seed=args.seed)

    targets = positions.copy()
    step_dx = args.speed / CTRL_HZ
    orbiting = (args.scenario == "orbiting_escort")
    omega = args.speed / 30.0 if orbiting else 0.0  # escorts orbit at `speed` m/s

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "agent_id", "x", "y", "z", "vx", "vy", "vz"])
        for s in range(args.steps):
            if orbiting:
                for i in range(num_drones):
                    if i == 0:
                        targets[0] = positions[0]  # core stays at centroid
                    else:
                        ang = omega * s / CTRL_HZ + 2 * np.pi * (i - 1) / (num_drones - 1)
                        targets[i] = [30.0 * np.cos(ang), 30.0 * np.sin(ang), 3.0]
            else:
                targets[:, 0] += step_dx
            env.step(targets.copy())
            for i in range(num_drones):
                st = env._getDroneStateVector(i)
                w.writerow([s, i, st[0], st[1], st[2], st[10], st[11], st[12]])

    env.close()
    motion = "orbiting" if orbiting else f"drift {args.speed} m/s"
    print(f"[saved] {args.out}  ({num_drones} drones x {args.steps} steps, {motion})")


if __name__ == "__main__":
    main()
