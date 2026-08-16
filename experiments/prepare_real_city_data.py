"""Download/cache every real-city snapshot used by the train/validation/test protocol."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.obstacles import OSM_ATTRIBUTION, load_real_city_layers  # noqa: E402
from src.training.geography import load_protocol, validate_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/real_city_protocol.json")
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-root", default="results/prepare-data")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    checks = validate_protocol(protocol)
    if not all(checks.values()):
        raise RuntimeError(f"Protocol integrity check failed: {checks}")
    records: list[dict] = []
    for city in protocol["cities"]:
        layers = load_real_city_layers(
            city["place"], args.grid_size, args.seed,
            radius_m=int(city.get("radius_m", 1400)), cache_dir=args.cache_dir,
            allow_network=not args.offline, force_refresh=args.force_refresh,
        )
        metadata = dict(layers.get("metadata", {}))
        metadata.update({"city": city["name"], "split": city["split"]})
        if args.require_real_data and metadata.get("source") != "openstreetmap":
            raise RuntimeError(f"{city['name']} fell back to {metadata.get('source')}: {metadata.get('fallback_reason', 'unknown reason')}")
        records.append(metadata)
        print(f"[{city['split']}] {city['name']}: source={metadata.get('source')} features={metadata.get('feature_count', 0)} cache={metadata.get('cache_path', '')}", flush=True)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "city_data_manifest.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "integrity": checks,
        "cities": records,
        "attribution": OSM_ATTRIBUTION,
        "arguments": vars(args),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
