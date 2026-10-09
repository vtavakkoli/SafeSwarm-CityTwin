# EARS v8 — experimental topology-aware coordination

**Status:** implemented research prototype, not a validated replacement for EARS v7 publication results. v6/v7 checkpoints, eight-city results, and paper statistics remain unchanged.

## What changed

- New controller at src/agents/ears_v8.py (EARSTopoSafePolicy). It inherits the original ant rule, event detection, and runtime safety projection without changing the v7 implementation.
- Graph-distance routing now rejects unreachable goals and goals that cannot be reached and returned from within an estimated battery budget. Idle movement costs and a reserve margin are included.
- Optional utility gate compares a proposed relocation with local observable utility. This is a *heuristic proxy*, not predicted hidden mission success. Parameters are development defaults, not independently validated.
- No new global relocation when the robot currently reports communications offline.
- Strict SWAP gate optionally checks sufficient source targets, changed views, and unique seed signatures before running experiments. Degenerate city snapshots cause a hard failure; they are never silently regenerated.
- Paired v8 comparison reports individual episodes, equal-city-weighted results, topology metrics, exact city-level sign-flip tests, city/episode hierarchical bootstrap uncertainty, leave-one-city-out sensitivity, and a standalone HTML report.
- An independent validator checks score recomputation, pairing, source conditions, safety results, and checkpoint/protocol hashes.

## Run the v8 comparison

Requires Python dependencies from requirements-dev.txt and the existing frozen checkpoint at results/train/checkpoints/ears_safe.json.

A local synthetic smoke run (not publication evidence):

    python experiments/run_train_test_pipeline.py --offline --quick
    python experiments/benchmark_ears_v8.py --offline --quick
    python experiments/validate_publication_v8.py --output-root results/v8/development

Development and scalability/communication-dropout sensitivity on existing v7 cities:

    python experiments/benchmark_ears_v8.py --agents-grid 4,8,16,32 --dropout-grid 0,0.03,0.15,0.30 --require-real-data
    python experiments/validate_publication_v8.py

The eight v7 cities are now **development evidence for v8** because their results motivated this design.

Independent new-city confirmation (after freezing v8 code and parameters):

    python experiments/benchmark_ears_v8.py --confirm
    python experiments/validate_publication_v8.py --confirm --output-root results/v8/confirmation

This fixes Zurich, Madrid, Lisbon, Stockholm, Helsinki, and Copenhagen as candidate confirmatory cities. If map adequacy fails, the run stops. Replacements must be documented before examining any method scores. These are geographic units, not 120 independent city samples.

Strict SWAP on the existing publication protocol (may fail Chicago with one target):

    python experiments/test_swap_protocol.py --protocol configs/publication_protocol_v7.json --require-real-data --strict-dataset-gate --min-original-targets 6 --output-root results/v8/swap-quality

The old v7 SWAP command is unchanged and only logs its quality findings. Never relabel old v7 statistics as data-adequacy-verified.

## Artifact contract

The comparison creates episodes.csv, city_means.csv, summary.csv, topology.csv, paired_city_tests.csv, manifest.json, and index.html in the output directory. The evidence validator creates quality_report.json. Every method shares the same city, start zone, episode seed, robot count, dropout probability, base and mission configuration.

## Limitations and reviewer precautions

- This is OSM-derived grid simulation, not validation on real robots or aircraft.
- Communication scenarios currently vary dropout probability. They do not yet simulate stale messages, radio propagation, Byzantine participants, or real distributed global coordination.
- The runtime monitor enforces grid-level constraints. That is not formal safety certification under physical dynamics and sensing errors.
- The new gate and route penalties may *worsen* discovery. No claim of superiority should be made without independent confirmation.
- Report both the route-only and gated variants, city-specific failures, effect sizes and confidence intervals. Do not choose the winner from confirmatory test data.
