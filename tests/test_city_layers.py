from pathlib import Path

from src.environment.obstacles import load_layer_cache, save_layer_cache, synthetic_city_layers


def test_city_layer_cache_roundtrip(tmp_path: Path):
    layers = synthetic_city_layers(20, seed=7, place_name="Vienna test")
    target = tmp_path / "city.json"
    save_layer_cache(target, layers)
    loaded = load_layer_cache(target)
    assert loaded["obstacles"] == layers["obstacles"]
    assert loaded["mission_zones"] == layers["mission_zones"]
    assert loaded["priority_cells"] == layers["priority_cells"]
    assert loaded["metadata"]["source"] == "synthetic"
