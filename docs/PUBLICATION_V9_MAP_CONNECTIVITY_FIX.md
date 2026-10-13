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


## Retry a missing original OSM snapshot against an alternate Overpass server

A corporate gateway or the default Overpass service may return HTTP 500 or a
network timeout. The publication CLI now supports an **explicit** OSMnx
`--overpass-url` override for `--fetch-missing`. OSMnx expects the base URL
ending in `/api`, **not** `/api/interpreter`; it appends that path itself.
The override only affects absent raw OSM snapshots. Cached city data remain
unchanged, and only verified OSM geography and missions are accepted.

Example, from the repository root (choose a **new** audit filename each time):

```bash
git switch main
git pull --ff-only
docker compose run --rm --build ears-v9-publication python -m experiments.prepare_publication_v9_maps --audit-only --fetch-missing --overpass-url https://overpass.private.coffee/api --max-shift 3 --audit-path results/publication-v9-map-amendment/preflight-overpass-alt.json
```

The public alternate instance comes from the OpenStreetMap Wiki's listed
global Overpass servers. Availability varies. This is not guaranteed to fix
a firewall/proxy timeout: network access from **inside the Docker container**
must permit the destination. If all backends fail, keep the original failure
visible in the audit; **never substitute a synthetic Copenhagen map**.

Note the separate geographic blocker: Lisbon required **up to 7 grid cells**
of target projection in the reported run. The default 3-cell safety gate is
unchanged. A complete six-city OSM audit is prerequisite to any justified
new methodology amendment, not automatic permission to run the benchmark.


## Explicit five-city exploratory study: omit unavailable Copenhagen

The default `ears-v9-publication` service **still requires the original six
cities**. When Copenhagen has no verified OSM source due to a server outage,
use an **explicit** `--exclude-cities Copenhagen` on BOTH the map-preparation
and evaluation commands. This is a **new exploratory, reduced-city protocol**
with a different study root, not the preregistered six-city confirmation.

In the audited real cache, Lisbon has seven stranded POIs. Four exceed the
original three-cell shift bound; the largest proposed projection is **7** grid
cells. The following example `--max-shift 7` is an **explicit, post-audit
geographic-methodology amendment for exploratory analysis only**, not a
scientific validation of those projections. Review all source/target cells,
their physical meaning and resulting metric sensitivity before publication.
The three-cell default and original source maps remain unchanged.

```powershell
git switch main
git pull origin main

# Explicit five-city OSM map amendment: EXCLUDES Copenhagen.
# Choose NEW, versioned output paths; all source caches stay immutable.
docker compose run --rm --build ears-v9-publication python -m experiments.prepare_publication_v9_maps --exclude-cities Copenhagen --max-shift 7 --output-cache-dir data/cache/v9-five-city-shift7-v1 --audit-path results/publication-v9-map-amendment/five-city-shift7-v1.json

# Run a SEPARATE exploratory study. It does NOT touch results/publication-v9.
docker compose run --rm --build ears-v9-publication python -m experiments.publication_ears_v9 all --exclude-cities Copenhagen --offline --cache-dir data/cache/v9-five-city-shift7-v1 --output-root results/publication-v9-five-city-shift7-v1
```

Expected paired matrix: five cities, 16 agent/dropout configurations, two
standard panels, 20 episodes, four methods, three SWAP target seeds: **14,000
episodes** (6,400 mission-stop + 6,400 fixed-budget + 1,200 SWAP).

The output is under `results/publication-v9-five-city-shift7-v1/`, including
`report/report.pdf`, `report/index.html`, `report/conclusion.json` and
`frozen/study.json`. The report discloses excluded cities and amended maps.
It must NOT present this run as the original six-city confirmation.

Statistical limitation: with five independent city units, the smallest
two-sided exact city sign-flip p-value is **2 / 32 = 0.0625**; superiority
cannot pass the original alpha=0.05 primary threshold regardless of the
observed positive difference. The pre-existing historical validation failures
also remain part of the paper. Report comparative outcomes descriptively.

Never reuse the same `--output-root` with different geography or controller
code; frozen source hashes explicitly reject this. Never silently omit
additional cities, lower the target-reachability gate, or replace absent OSM
data with synthetic fallback.


## Explicit opt-in: omit a sparse real-OSM city during reduced-city evaluation

Even after connectivity repair, the original quality gate requires at least
**six** mission targets per city. If an OSM city has only **four**, no
sensible real-OSM transformation may fabricate two additional targets. More
critically, SWAP's `min_targets=6` yields identical target subsets for a
city with at most six targets, making three independent SWAP dataset seeds
impossible for that city. Do **not** lower the minimum target threshold.

For the previously prepared five-city projected OSM cache, use this separate,
explicitly requested **exploratory** exclusion if a city lacks enough real
targets for the v9 matrix:

```powershell
git switch main
git pull origin main
docker compose run --rm --build unit-test

docker compose run --rm --build ears-v9-publication python -m experiments.publication_ears_v9 all --exclude-cities Copenhagen --skip-sparse-cities --offline --cache-dir data/cache/v9-five-city-shift7-v1 --output-root results/publication-v9-available-sparse-filter-v1
```

The command records the real city/cities rejected for insufficient mission
targets and their original counts in `frozen/study.json` under
`sparse_city_exclusions` and in `report/conclusion.json`. All remaining
cities still must pass their full geographic reachability gate, verified OSM
origin and distinct SWAP mission views. No city is dropped unless the user
provided both `--exclude-cities` and `--skip-sparse-cities`; this
is never enabled for the original six-city study.

If one of the five cities has four targets and the other four pass, the
effective study runs **11,200 paired episodes**: 5,120 mission-stop,
5,120 fixed-budget and 960 SWAP. It remains an amended, exploratory study
and **cannot** be described as original six-city independent confirmation.
With four city-level independent units, the exact two-sided sign-flip test
has minimum attainable p-value **2/16 = 0.125** and cannot pass alpha=0.05.

**Important:** Use a **new** `--output-root`. Earlier failed freezes may
have left incomplete map/checkpoint files, and frozen source hashes disallow
resuming an old study after code changes. Preserve the prior audit data,
unchanged real city caches and all map projection coordinates for reviewers.
