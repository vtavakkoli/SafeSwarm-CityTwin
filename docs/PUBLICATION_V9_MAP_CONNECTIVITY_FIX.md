# SafeSwarm v9 — audited OSM mission-cell connectivity amendment

## Why forcing OSM refresh cannot fix the quality gate

A city raster can contain a target cell isolated by blocked neighboring cells.
The current publication gate requires **all** mission cells to be reachable
from the selected start-zone bases in four directions; a sample failure was
543/550 reachable mission cells and 6 traversable components. Downloading a
fresh OSM snapshot does not itself repair the raster topology.

**Do not change the reachability gate to 0.98.**

## Versioned, auditable map preprocessing (new study only)

The explicit `prepare_publication_v9_maps.py` tool finds cells reachable
from each test start zone, intersects the reachable sets and projects
isolated OSM POI *grid representations* onto the nearest unique free cells in
the common reachable area. It preserves each original POI priority and target
count. It never modifies obstacles, restricted zones or original OSM caches.

The maximum allowed Manhattan grid shift is predeclared (default 3). An OSM POI
requiring a larger shift **causes the command to fail**. Review such cases
instead of expanding the threshold after seeing outcome data.

Every relocation records the original grid cell, replacement grid cell,
distance and priority, plus a hash of the original OSM cache. These records
appear in the amendment audit and in metadata of derived snapshots, which are
also frozen by the publication pipeline.

Run from the repository root after reviewing the methodology and before
looking at any confirmation performance:

```bash
# Download original, strictly OSM-derived geographical snapshots.
docker compose run --rm --build prepare-data python experiments/prepare_real_city_data.py --protocol configs/confirmation_protocol_v9.json --grid-size 40 --force-refresh --require-real-data

# Prepare separate derived maps. Original data/cache stays unchanged.
docker compose run --rm --build ears-v9-publication python experiments/prepare_publication_v9_maps.py --output-cache-dir data/cache/v9-projected-v1 --audit-path results/publication-v9-map-amendment/audit.json --max-shift 3

# Run a NEW immutable study with the existing historical EARS checkpoint.
docker compose run --rm --build ears-v9-publication python experiments/publication_ears_v9.py all --offline --cache-dir data/cache/v9-projected-v1 --output-root results/publication-v9-projected-v1
```

Result locations:

- `results/publication-v9-map-amendment/audit.json`: every OSM mission-cell shift and source hash.
- `data/cache/v9-projected-v1/`: distinct versioned preprocessed city snapshots.
- `results/publication-v9-projected-v1/frozen/`: code, checkpoint, protocol and final map hashes.
- `results/publication-v9-projected-v1/report/report.pdf`: publication PDF if evaluation completes.
- `results/publication-v9-projected-v1/report/index.html`: HTML report.
- `results/publication-v9-projected-v1/episodes.csv`: all raw, matched episodes.

Check tests:

```bash
docker compose run --rm --build unit-test
```

If the projection fails because it exceeds the allowed distance, **do not
silently drop a target**, remove a barrier, or retune its distance threshold
based on performance. Investigate the OSM tile and define a documented
geographic amendment or a genuinely untouched replacement test protocol.

## Publication limitations

This process **amends the evaluation geography** and cannot be presented as
the original v9 frozen geography. Its effective mission cells are accessible
surrogate locations projected from OSM POIs, not necessarily the exact physical
POI positions. Disclose the full projection audit, preprocessing tolerance,
checkpoint lineage, all safety/coverage results, and any negative findings.

Independence is not restored by preprocessing if the same six confirmation
cities were already used to select model hyperparameters or inspect test
performance. Preserve original caches, derived caches and all audit files.


## Diagnose all six confirmation cities before revising a displacement limit

When a target needs more than `--max-shift 3`, first run the **read-only
preflight**. It uses the same deterministic target-assignment algorithm with
the mathematical grid-distance maximum only to *measure* displacement; it
creates no derived OSM map and does not authorize a higher tolerance.

```bash
docker compose run --rm --build ears-v9-publication python -m experiments.prepare_publication_v9_maps --audit-only --max-shift 3 --audit-path results/publication-v9-map-amendment/preflight.json
```

Inspect `results/publication-v9-map-amendment/preflight.json`:

- `cities[].required_max_shift`: largest proposed Manhattan displacement
- `cities[].over_limit`: the original target coordinates, candidate substitute
  coordinates, priority and displacement for each target exceeding the limit
- `all_within_requested_limit`: whether all six pass the specified limit
- `source_snapshot_sha256`: original OSM-cache file fingerprint

If the preflight proposes large shifts, investigate the rasterization,
the restricted/obstacle masks and the original OSM POIs. Do **not** silently
adopt the measured maximum as the new default or call the resulting study
unmodified, independent confirmation evidence. Prespecify and justify a
new geographic protocol before creating any amended maps or collecting results.


## Missing OSM cache in read-only preflight (e.g. Copenhagen)

Before this fix, `--audit-only` aborted with `FileNotFoundError` when the
last city had no original 40x40 OSM cache; earlier city diagnostics were lost.
It now records a `missing_osm_cache`/ `invalid_osm_source` entry and writes
the whole audit. Missing cities are explicitly counted and the overall
`all_within_requested_limit` stays **false**. No synthetic fallback is
accepted as real OSM evidence.

To download **only absent** original OSM snapshots (internet access required)
and inspect every city's required displacement:

```bash
git pull --ff-only
docker compose run --rm --build ears-v9-publication python -m experiments.prepare_publication_v9_maps --audit-only --fetch-missing --max-shift 3 --audit-path results/publication-v9-map-amendment/preflight-with-copenhagen.json
```

Without `--fetch-missing`, `--audit-only` is purely read-only. With that
flag it may create a **raw** OSM city cache, but it never writes an amended
or derived map and never refreshes or overwrites an existing cache. Provenance
is checked; if fetching fails and OSM returns synthetic fallback, the audit
reports that city as `invalid_osm_source`. Existing source caches are left
unmodified in either case.

Do not treat a report with missing OSM caches as a six-city success.
In the observed run, Lisbon requires up to **seven grid cells** of
target relocation (seven candidate moves; four exceed the fixed three-cell
limit). That is a substantive geographic-methodology issue, not a cache
error. Complete the six-city audit and review those displacements before
running a *new, explicitly amended* publication study. This patch does not
raise the limit, remove targets or change the frozen quality gate.
