"""Audited projection of isolated OSM mission cells for a NEW v9 study.

This is an explicit geographic-protocol amendment. It does not modify the raw
OSM cache, remove missions, edit barriers, or rewrite historical evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.benchmark_ears_v8 import _graph_distances, topology_metrics
from src.environment.obstacles import _cache_path, load_layer_cache, load_real_city_layers, save_layer_cache
from src.training.geography import apply_start_zone, load_protocol, select_cities

METHOD = "publication-v9-accessible-osm-poi-grid-projection-v1"


def reachable_from_bases(layers: dict, grid: int) -> set[tuple[int, int]]:
    blocked = set(layers["obstacles"]) | set(layers["restricted_zones"])
    reachable: set[tuple[int, int]] = set()
    for base in sorted(layers["base_stations"]):
        if base in blocked:
            raise ValueError(f"Start base is blocked: {base}")
        reachable.update(_graph_distances(grid, base, blocked))
    return reachable


def project_targets(layers: dict, grid: int, zones: list[str], max_shift: int) -> tuple[dict, dict]:
    """Snap stranded targets to unique nearby cells accessible from every test zone.

    Priority, target count and obstacle topology are invariant. All source
    positions and movement distances are retained in machine-readable metadata.
    """
    if max_shift < 0 or not zones:
        raise ValueError("Expected max_shift >= 0 and at least one start zone")
    originals = set(layers["mission_zones"])
    priorities = dict(layers["priority_cells"])
    blocked = set(layers["obstacles"]) | set(layers["restricted_zones"])
    if not originals or originals & blocked:
        raise ValueError("Missing mission targets or source targets on blocked cells")
    if not originals <= set(priorities):
        raise ValueError("Every OSM target must retain its original priority")

    reach = {
        zone: reachable_from_bases(apply_start_zone(layers, grid, zone), grid)
        for zone in zones
    }
    common = set.intersection(*reach.values())
    if not common:
        raise ValueError("Test start zones have no common reachable area")

    replacement: dict[tuple[int, int], tuple[int, int]] = {}
    used = set(originals)
    changes = []
    for source in sorted(originals - common):
        available = common - used
        if not available:
            raise ValueError("Insufficient distinct reachable cells to preserve all missions")
        distance, dest = min(
            ((abs(c[0] - source[0]) + abs(c[1] - source[1]), c) for c in available),
            key=lambda value: (
                value[0],
                (value[1][0] - source[0]) ** 2 + (value[1][1] - source[1]) ** 2,
                value[1],
            ),
        )
        if distance > max_shift:
            raise ValueError(
                f"OSM target {source} requires {distance} grid cells of projection "
                f"(max-shift={max_shift}). A larger shift requires an explicit "
                "prospective geographic-methodology amendment."
            )
        used.add(dest)
        replacement[source] = dest
        changes.append({
            "original_cell": list(source),
            "projected_cell": list(dest),
            "grid_manhattan_shift": distance,
            "priority": float(priorities[source]),
        })

    amended = copy.deepcopy(layers)
    amended["mission_zones"] = (originals - set(replacement)) | set(replacement.values())
    amended["priority_cells"] = {
        replacement.get(source, source): float(priorities[source])
        for source in sorted(originals)
    }
    if len(amended["mission_zones"]) != len(originals):
        raise AssertionError("A mission target was lost")
    if amended["obstacles"] != layers["obstacles"] or amended["restricted_zones"] != layers["restricted_zones"]:
        raise AssertionError("An obstacle/restricted zone was modified")

    before = {zone: topology_metrics(apply_start_zone(layers, grid, zone), grid) for zone in zones}
    after = {zone: topology_metrics(apply_start_zone(amended, grid, zone), grid) for zone in zones}
    if any(q["reachable_mission_fraction"] != 1.0 for q in after.values()):
        raise ValueError(f"Target reachability still fails: {after}")

    audit = {
        "method": METHOD,
        "source_osm_target_count": len(originals),
        "projected_osm_target_count": len(amended["mission_zones"]),
        "modified_targets": len(changes),
        "max_shift_grid_cells": max_shift,
        "modifications": changes,
        "before": before,
        "after": after,
    }
    amended["metadata"] = {
        **layers["metadata"],
        "target_projection_method": METHOD,
        "target_projection_audit": {
            "modified_targets": len(changes),
            "max_shift_grid_cells": max_shift,
            "modifications": changes,
            "preprocessing_is_protocol_amendment": True,
            "original_target_cells_are_osm_derived": True,
        },
    }
    return amended, audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/confirmation_protocol_v9.json")
    p.add_argument("--grid-size", type=int, default=40)
    p.add_argument("--source-cache-dir", default="data/cache")
    p.add_argument("--output-cache-dir", default="data/cache/v9-projected-v1")
    p.add_argument("--audit-path", default="results/publication-v9-map-amendment/audit.json")
    p.add_argument("--max-shift", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_cache_dir).resolve()
    dest = Path(args.output_cache_dir).resolve()
    if source == dest or source.is_relative_to(dest):
        raise ValueError("Input and output caches must be separate")
    audit_path = Path(args.audit_path)
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing amendment audit: {audit_path}")

    protocol = load_protocol(args.protocol)
    zones = list(protocol["start_zones"]["test"])
    cities = select_cities(protocol, "test")
    staged = []
    report = {
        "description": "Explicit new v9 geographic-methodology amendment, not original confirmation",
        "method": METHOD,
        "protocol": args.protocol,
        "grid_size": args.grid_size,
        "zones": zones,
        "max_shift_grid_cells": args.max_shift,
        "cities": [],
    }

    # Validate every source city and all proposed targets before any write.
    for city in cities:
        radius = int(city.get("radius_m", 1400))
        original_path = _cache_path(source, city["place"], args.grid_size, radius)
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing raw OSM city cache: {original_path}")
        layers = load_real_city_layers(
            city["place"], args.grid_size, radius_m=radius,
            cache_dir=source, allow_network=False,
        )
        metadata = layers["metadata"]
        if metadata.get("source") != "openstreetmap" or metadata.get("mission_source") != "openstreetmap":
            raise ValueError(f"{city['name']}: actual OSM geography and missions required")
        if metadata.get("target_projection_method"):
            raise ValueError(f"{city['name']}: source must be raw, not an already amended map")
        amended, audit = project_targets(layers, args.grid_size, zones, args.max_shift)
        source_sha = hashlib.sha256(original_path.read_bytes()).hexdigest()
        amended["metadata"]["source_osm_snapshot_sha256"] = source_sha
        output_path = _cache_path(dest, city["place"], args.grid_size, radius)
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing derived map: {output_path}; "
                "use a new versioned cache path."
            )
        audit.update({
            "city": city["name"],
            "source_snapshot_sha256": source_sha,
            "source_path": str(original_path),
            "output_path": str(output_path),
        })
        report["cities"].append(audit)
        staged.append((output_path, amended))
        print(f"{city['name']}: projected {audit['modified_targets']}/"
              f"{audit['source_osm_target_count']} missions", flush=True)

    for path, amended in staged:
        save_layer_cache(path, amended)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Preserved original OSM cache: {source}")
    print(f"New amended OSM cache: {dest}")
    print(f"Complete cell-level audit: {audit_path}")
    print("Run v9 with --offline, --cache-dir pointing to the new cache, and a NEW --output-root.")


if __name__ == "__main__":
    main()
