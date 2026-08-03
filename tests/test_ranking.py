import pandas as pd

from src.evaluation.metrics import rank_algorithms


def test_ranking_prefers_safe_high_discovery_strategy():
    common = {
        "mission_success_rate": 0.5,
        "coverage_ratio": 0.5,
        "number_of_safety_violations": 0,
        "safety_interventions": 0,
        "collision_count": 0,
        "restricted_zone_entries": 0,
        "battery_failures": 0,
        "energy_consumption": 40.0,
        "redundant_coverage": 0.1,
        "communication_efficiency": 1.0,
        "distance_travelled": 20,
        "runtime_seconds": 0.1,
        "runtime_overhead": 0.001,
        "blocked_moves": 0,
        "agents": 4,
        "steps": 40,
        "episode": 0,
        "data_source": "synthetic",
    }
    rows = [
        {**common, "city": "Vienna", "strategy": "Strong", "weighted_target_discovery": 0.9, "actual_safety_incidents": 0},
        {**common, "city": "Vienna", "strategy": "Weak", "weighted_target_discovery": 0.3, "actual_safety_incidents": 4},
    ]
    _, overall = rank_algorithms(pd.DataFrame(rows))
    assert overall.iloc[0]["strategy"] == "Strong"
    assert overall.iloc[0]["rank"] == 1
