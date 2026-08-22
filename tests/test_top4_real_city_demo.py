from pathlib import Path

from experiments.demo_top4_real_city import (
    PUBLICATION_REFERENCE,
    TOP_STRATEGIES,
    _checkpoint_paths,
    _resolve_cities,
    _slug,
)


def _protocol() -> dict:
    return {
        "cities": [
            {"name": "Train", "place": "Train", "split": "train"},
            {"name": "San Francisco", "place": "San Francisco, USA", "split": "test"},
            {"name": "Tokyo", "place": "Tokyo, Japan", "split": "test"},
        ],
        "start_zones": {"train": ["center"], "test": ["north_east", "south_west"]},
    }


def test_top4_strategy_order_matches_publication_demo() -> None:
    assert TOP_STRATEGIES == (
        "H-MAPPO-EARS-Safe",
        "EARS-Safe",
        "EARS-NP-Safe",
        "AntSwarmSafe",
    )
    assert all(name in PUBLICATION_REFERENCE for name in TOP_STRATEGIES)
    assert PUBLICATION_REFERENCE["H-MAPPO-EARS-Safe"]["operational_score"] == 0.7845
    assert PUBLICATION_REFERENCE["AntSwarmSafe"]["operational_score"] == 0.7754


def test_city_selection_accepts_comma_separated_names() -> None:
    cities = _resolve_cities(_protocol(), "Tokyo, San Francisco", False)
    assert [city["name"] for city in cities] == ["Tokyo", "San Francisco"]


def test_all_test_cities_preserves_protocol_order() -> None:
    cities = _resolve_cities(_protocol(), "Tokyo", True)
    assert [city["name"] for city in cities] == ["San Francisco", "Tokyo"]


def test_slug_is_stable_for_output_paths() -> None:
    assert _slug("H-MAPPO-EARS-Safe") == "h-mappo-ears-safe"
    assert _slug("San Francisco") == "san-francisco"


def test_checkpoint_contract_uses_trained_models_only_where_needed(tmp_path: Path) -> None:
    paths = _checkpoint_paths(tmp_path)
    assert paths["AntSwarmSafe"] == []
    assert paths["EARS-Safe"] == [tmp_path / "ears_safe.json"]
    assert paths["EARS-NP-Safe"] == [tmp_path / "ears_np_safe.json"]
    assert paths["H-MAPPO-EARS-Safe"] == [
        tmp_path / "h_mappo_ears_safe.json",
        tmp_path / "mappo_safe.json",
    ]
