"""Unit-level v8 contract tests (no network or pretrained weights required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark_ears_v8 import _paired_statistics, topology_metrics
from src.agents.ears_v8 import EARSTopoSafePolicy, TopologyGateConfig
from src.environment.city_twin import CityTwinEnvironment
from src.training.swap_protocol import seeded_mission_view, validate_swap_view


def layers(*, wall: bool = False, n_targets: int = 3):
    return {
        "obstacles": {(5, y) for y in range(10)} if wall else set(),
        "restricted_zones": set(),
        "mission_zones": {(8, 8), (7, 6), (2, 8)} if n_targets == 3 else {(8, 8)},
        "base_stations": {(1, 1)},
        "priority_cells": {(8, 8): 1.0, (7, 6): .7, (2, 8): .5},
        "metadata": {"source": "synthetic", "place_name": "unit-city"},
    }


def env_from(layers_data):
    return CityTwinEnvironment(
        grid_size=10, n_agents=1, seed=9, layers=layers_data,
        max_steps=20, communication_dropout_prob=0.0, allow_network=False,
    )


def test_graph_route_rejects_disconnected_goal():
    env = env_from(layers(wall=True))
    policy = EARSTopoSafePolicy(seed=9)
    source = env.agents[0].position
    distances = policy._route_distances(env, source)
    assert (8, 8) not in distances
    assert (2, 8) in distances
    assert policy._route_to_base(env)[source] == 0


def test_benefit_gate_can_reject_without_creating_an_active_option():
    env = env_from(layers())
    policy = EARSTopoSafePolicy(seed=7, gate_config=TopologyGateConfig(min_net_gain=999.))
    policy._select_relocation_goals(env, [0], env.get_positions())
    assert policy.rejected_benefit == 1
    assert policy.accepted_relocations == 0
    assert 0 not in policy._relocation_goals


def test_route_only_ablation_accepts_feasible_goal():
    env = env_from(layers())
    policy = EARSTopoSafePolicy(
        seed=7, gate_config=TopologyGateConfig(enable_benefit_gate=False),
    )
    policy._select_relocation_goals(env, [0], env.get_positions())
    assert policy.accepted_relocations == 1
    assert 0 in policy._relocation_goals
    goal = policy._relocation_goals[0]
    assert goal in policy._route_distances(env, env.agents[0].position)
    assert goal in policy._route_to_base(env)
    assert np.isfinite(policy.diagnostics()["v8_mean_detour"])


def test_communication_drop_prevents_new_coordination():
    env = env_from(layers())
    env.agents[0].communication_status = False
    policy = EARSTopoSafePolicy(seed=7)
    policy._select_relocation_goals(env, [0], env.get_positions())
    assert policy.rejected_communication == 1
    assert not policy._relocation_goals


def test_no_actual_targets_needed_for_observable_route_decisions():
    env1 = env_from(layers(n_targets=3))
    env2 = env_from(layers(n_targets=1))
    assert np.array_equal(env1.observation_map, env2.observation_map)
    a = EARSTopoSafePolicy(seed=1, gate_config=TopologyGateConfig(enable_benefit_gate=False))
    b = EARSTopoSafePolicy(seed=1, gate_config=TopologyGateConfig(enable_benefit_gate=False))
    a._select_relocation_goals(env1, [0], env1.get_positions())
    b._select_relocation_goals(env2, [0], env2.get_positions())
    assert a._relocation_goals == b._relocation_goals


def test_topology_flags_unreachable_targets():
    metrics = topology_metrics(layers(wall=True), 10)
    assert metrics["components"] == 2
    assert metrics["reachable_mission_fraction"] < 1


def test_swap_quality_manifest_records_one_target_degeneracy():
    one = layers(n_targets=1)
    swap = seeded_mission_view(one, 2042)
    assert swap["metadata"]["swap_dataset_changed"] is False


def test_strict_swap_quality_rejects_degenerate_and_duplicate_views():
    swap = seeded_mission_view(layers(n_targets=1), 2042)
    metadata = swap["metadata"]
    problems = validate_swap_view(metadata, min_original_targets=6)
    assert "too few original mission targets" in problems
    assert "alternate target set unchanged" in problems
    duplicate = validate_swap_view(
        metadata, min_original_targets=1,
        seen_signatures={metadata["swap_signature"]},
    )
    assert "duplicate SWAP target signature" in duplicate



def test_city_block_permutation_distinguishes_episode_counts():
    records = []
    for city, shift in (("A", .03), ("B", -.01)):
        for ep in range(10):
            for method, score in (("AntSwarmSafe", .7), ("EARS-Safe-v7", .7 + shift),
                                  ("EARS-TopoSafe", .7 + shift),
                                  ("EARS-TopoSafe-Gated", .7 + shift)):
                records.append({"city": city, "start_zone": "north_east",
                                "episode": ep, "agents": 2, "dropout": .03,
                                "strategy": method, "operational_score": score})
    stats = _paired_statistics(pd.DataFrame(records), seed=2)
    assert len(stats) == 3
    assert all(stats.city_count == 2)
    assert all(stats.paired_episodes == 20)
    assert all((stats.city_signflip_p >= 0) & (stats.city_signflip_p <= 1))
