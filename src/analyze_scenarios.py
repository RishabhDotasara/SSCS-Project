#!/usr/bin/env python3
"""
Dataset vs. simulator scenario coverage analysis.

Answers three questions:
  1. What scenarios are actually in the training dataset?
  2. Which simulator test scenarios match a dataset concept?
  3. For the simulator scenarios the model scored LOW on (off-center core,
     orbiting escort, relay bridge), does the dataset contain that concept at
     all -- i.e. is the poor performance a *data coverage* problem?

Uses only the Python standard library, so it runs with plain `python3`:

    python3 src/analyze_scenarios.py [--data data/swarm_criticality_dataset_100_scenarios.csv]
"""
import argparse
import csv
from collections import defaultdict
import statistics as st


# ---------------------------------------------------------------------------
# Simulator test scenarios (from src/sim/run_scenario.py) and the concept they
# encode. "concept" is the *geometric* property that makes an agent critical.
# ---------------------------------------------------------------------------
SIM_SCENARIOS = [
    # name, agents, criticals, concept
    ("moving_escort",  9, "1 core", "central + surrounded + moving"),
    ("escort_many",   18, "1 core", "central + surrounded + moving"),
    ("two_cores",     18, "2 cores", "central + surrounded + moving (multiple)"),
    ("single_relay_mesh", 9,  "1 relay", "flanked bridge (NOT central)"),
    ("relay_bridge",  20, "2 relays", "flanked bridge (NOT central)"),
    ("offcenter_core", 20, "1 core", "persistently off-center (outer)"),
    ("orbiting_escort", 9, "1 core", "stationary core + orbiting escorts"),
]


def load_cases(path):
    """Group rows by case_id; return per-case summaries and crit/decoy feature pools."""
    cases = {}
    crit = defaultdict(list)
    decoy = defaultdict(list)
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            c = row["case_id"]
            if c not in cases:
                cases[c] = {
                    "n": int(row["num_agents"]),
                    "obst": int(row["n_obstacles"]),
                    "attr": c.split("_")[-1],
                    "steps": set(),
                    "crit_ids": set(),
                    "crit_rows": 0,
                    "tot_rows": 0,
                    # feature accumulators (step-level)
                    "crit_rank": [],      # rank_dist_to_centroid of criticals
                    "crit_aco": [],       # angular_coverage of criticals
                    "crit_offcenter": 0,  # critical rows with rank > 0.8
                    "min_speed": 1e9,
                }
            d = cases[c]
            sp = float(row["speed"])
            d["steps"].add(int(row["step"]))
            d["tot_rows"] += 1
            d["min_speed"] = min(d["min_speed"], sp)
            is_crit = (row["is_critical"] == "True")
            if is_crit:
                d["crit_ids"].add(int(row["agent_id"]))
                d["crit_rows"] += 1
                d["crit_rank"].append(float(row["rank_dist_to_centroid"]))
                d["crit_aco"].append(float(row["angular_coverage"]))
                if float(row["rank_dist_to_centroid"]) > 0.8:
                    d["crit_offcenter"] += 1
            # global concept pools
            g = crit if is_crit else decoy
            g["rank"].append(float(row["rank_dist_to_centroid"]))
            g["aco"].append(float(row["angular_coverage"]))
            g["neighbors"].append(float(row["num_neighbors"]))
            g["dist_target"].append(float(row["dist_to_target"]))
            g["betweenness"].append(float(row["betweenness_centrality"]))
    return cases, crit, decoy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/swarm_criticality_dataset_100_scenarios.csv")
    args = ap.parse_args()

    cases, crit, decoy = load_cases(args.data)

    print("=" * 78)
    print("PART 1 -- What scenarios exist in the dataset?")
    print("=" * 78)
    print(f"Total cases: {len(cases)}")
    print()
    print("The case_id only encodes: <case#>_<n agents>_<attrition level>.")
    print("There is NO 'formation' field (no escort/relay/bridge/off-center).")
    print("All cases are a *moving swarm* in which the inner/central/surrounded")
    print("agents are labeled critical (see Part 2).")
    print()
    print("  case            n   attrition  obstacles  criticals  prevalence  min_speed")
    print("  " + "-" * 74)
    for c in sorted(cases):
        d = cases[c]
        prev = d["crit_rows"] / d["tot_rows"] * 100
        print(f"  {c:14s} {d['n']:3d}  {d['attr']:8s} {d['obst']:5d}     "
              f"{len(d['crit_ids']):3d}      {prev:4.1f}%      {d['min_speed']:.1f} m/s")

    # Swarm-size and attrition distribution
    sizes = defaultdict(int)
    attr = defaultdict(int)
    for d in cases.values():
        sizes[d["n"]] += 1
        attr[d["attr"]] += 1
    print()
    print("Swarm sizes: ", ", ".join(f"{n}drones x{cnt}" for n, cnt in sorted(sizes.items())))
    print("Attrition:   ", ", ".join(f"{a} x{cnt}" for a, cnt in sorted(attr.items())))

    print()
    print("=" * 78)
    print("PART 2 -- The dataset's criticality *concept* (crit vs decoy geometry)")
    print("=" * 78)
    print("What makes an agent critical in the dataset? Averaged over ALL snapshots:")
    print()
    rows = [
        ("rank_dist_to_centroid (0=closest)", "rank"),
        ("angular_coverage (0..1, surrounded?)", "aco"),
        ("num_neighbors", "neighbors"),
        ("dist_to_target", "dist_target"),
        ("betweenness_centrality (bridge?)", "betweenness"),
    ]
    print(f"  {'feature':34s} {'critical mean':>14s} {'decoy mean':>12s}")
    print("  " + "-" * 62)
    for label, key in rows:
        print(f"  {label:34s} {st.mean(crit[key]):14.4f} {st.mean(decoy[key]):12.4f}")
    print()
    print("  => Critical agents are, on average, CLOSER to the centroid, MORE")
    print("     surrounded (higher angular coverage), have MORE neighbors, are")
    print("     CLOSER to the mission target, and are NOT more 'bridge-like'")
    print("     (betweenness is essentially equal). This is a graded")
    print("     'inner = critical' concept over a large fraction of the swarm")
    print("     (~30-45%).")
    all_min = min(d["min_speed"] for d in cases.values())
    print(f"  Min observed speed across all cases: {all_min:.1f} m/s")
    print("  => every agent moves; there are NO stationary/hovering agents.")

    print()
    print("=" * 78)
    print("PART 3 -- Do the simulator test scenarios match a dataset concept?")
    print("=" * 78)
    print()
    print("  The dataset concept is 'central + surrounded + moving = critical'.")
    print("  Each simulator scenario either matches it or encodes a concept the")
    print("  dataset never contains.")
    print()

    concept_match = {
        "central + surrounded + moving": True,
        "central + surrounded + moving (multiple)": True,
        "flanked bridge (NOT central)": False,
        "persistently off-center (outer)": False,
        "stationary core + orbiting escorts": False,
    }

    # Quantify the two "absent" concepts in the dataset for the summary.
    # (a) stationary agents
    n_stationary = sum(1 for d in cases.values() if d["min_speed"] < 0.5)
    # (b) persistently off-center criticals: critical whose mean rank > 0.8
    #     (transient off-center rows exist, but no critical stays outer)
    offcenter_crit_mean = []
    for c in sorted(cases):
        d = cases[c]
        if d["crit_rank"] and st.mean(d["crit_rank"]) > 0.8:
            offcenter_crit_mean.append(c)
    # (c) "flanked bridge" criticals: high betweenness AND outer at the same time
    n_bridge_rows = sum(
        1 for bc, rk in zip(crit["betweenness"], crit["rank"])
        if bc > 0.01 and rk > 0.6
    )
    n_crit_rows = len(crit["rank"])

    print(f"  {'simulator scenario':22s} {'agents':>6s} {'concept':38s} {'in dataset?':>12s}")
    print("  " + "-" * 78)
    for name, agents, crits, concept in SIM_SCENARIOS:
        present = concept_match[concept]
        print(f"  {name:22s} {agents:6d} {concept:38s} "
              f"{'YES (core concept)' if present else 'NO'}")
    print()
    print("  Match summary:")
    print("    - moving_escort / escort_many / two_cores :")
    print("        concept matches the dataset ('central + surrounded + moving').")
    print("        => the model transfers. (two_cores is fine because the dataset")
    print("           labels MANY central criticals per case, not just one.)")
    print("    - single_relay_mesh / relay_bridge :")
    print("        concept is a flanked BRIDGE; dataset criticals are central and")
    print("        surrounded, NOT bridges. => transfers with correct ranking but")
    print("        LOW confidence (0.003-0.3).")
    print("    - offcenter_core :")
    print("        concept is a critical that is PERSISTENTLY outer (rank~1.0).")
    print("        => FAILS (ranked 17th).")
    print("    - orbiting_escort :")
    print("        concept is a STATIONARY core; dataset has no stationary agents.")
    print("        => DEGRADES (core ranked 2nd).")

    print()
    print("=" * 78)
    print("PART 4 -- Is the poor performance due to LACK OF DATA?")
    print("=" * 78)
    print()
    print(f"  Cases with any stationary agent (speed < 0.5 m/s):  {n_stationary}/100")
    print(f"  Cases with a critical whose MEAN rank > 0.8 (i.e. a")
    print(f"  persistently off-center core):                        {len(offcenter_crit_mean)}/100")
    print(f"  Critical rows that are 'flanked bridges' (betweenness")
    print(f"  > 0.01 AND rank > 0.6):  {n_bridge_rows}/{n_crit_rows} "
          f"({100.0 * n_bridge_rows / n_crit_rows:.2f}%)")
    print()
    print("  Interpretation:")
    print("    The low-confidence / failing simulator scenarios correspond to")
    print("    formation concepts that are STRUCTURALLY ABSENT from the training")
    print("    dataset:")
    print("      * no stationary agents          -> orbiting escort (stationary core)")
    print("      * no persistently off-center    -> off-center core")
    print("      * criticals are never bridges   -> relay / relay-bridge (low conf.)")
    print("    So the drop in performance is a DATA COVERAGE / concept-mismatch")
    print("    issue, not necessarily a model-architecture failure: the model")
    print("    faithfully learned 'central + surrounded + moving = critical' and")
    print("    cannot generalise to formations that never appear in the data.")


if __name__ == "__main__":
    main()
