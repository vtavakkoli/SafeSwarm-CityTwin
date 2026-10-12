# EARS v9: information routing and validation-only selection

Status: experimental candidate. A large real-city improvement has **not** been
established. v7, v8, their scores, simulator and historical outputs are unchanged.

## Why this change exists

The supplied full v8 development run contains 10,240 rows: eight cities, four
robot counts, four dropout rates, 20 episodes and four methods. Every matched
key is present exactly once, there are no missing core metrics, and recalculating
J agrees within 3.4e-16. Its `episodes.csv` SHA-256 is
`a913fe7697aee33e554a75fdcaa8e43f9b824f374cafe48ddeb5dd918155d3d4`.

Equal-weight means across that balanced grid:

| Method | J | Discovery | Coverage | Energy | Seconds/episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| AntSwarmSafe | 0.774209 | 0.878975 | 0.595542 | 833.489590 | 1.876893 |
| EARS-Safe-v7 | 0.776424 | 0.878991 | 0.590000 | 802.945918 | 7.581535 |
| EARS-TopoSafe | 0.775980 | 0.878976 | 0.596699 | 820.932148 | 6.824798 |
| EARS-TopoSafe-Gated | 0.774961 | 0.879962 | 0.590018 | 808.871387 | 29.371532 |

All recorded physical incidents and battery failures are zero. v8-versus-v7
city-block sign-flip p-values, averaging the 16 settings within each city, are
0.6953125 (ungated) and 0.234375 (gated). Neither supports superiority.
These are descriptive analyses of already inspected development data.

At 32 robots gated v8 averages 103.94 seconds per episode, of which 100.93
seconds are planning; it evaluates approximately 2.18 million goal candidates.
Code inspection identifies repeated searches after rejection, timestep-scoped
graph caches, goal spacing restricted to same-step allocations, and a route
score whose visit penalty can outweigh progress. These are actionable mechanisms,
not proof that any single one explains the score regressions.

## New policy

`src/agents/ears_v9.py` introduces `EARSInformationPolicy`:

- Scores the marginal newly sensed area in the actual Manhattan sensor footprint.
  It never reads hidden mission coordinates, target weights or city identities.
- Reserves sensing footprints across same-step moves and ongoing relocation
  goals. Drops completed, stale, expired or energy-infeasible options.
- Caches graph distances across steps with bounded LRU storage, and invalidates
  them when the environment, obstacles, restricted zones or bases change.
- Uses multi-source graph distances for return planning. Route progress takes
  priority over accumulated visit counts. Duration depends on route length.
- Applies a retry delay to unsuccessful allocation and records history every
  step. Congestion uses nearby teammates rather than all distant robots.
- Keeps the existing runtime monitor, energy costs, score and episode termination.

This changes the local controller as well as relocation. It must be described
as a new controller; it is not a pure relocation ablation of v7. The two matched
local-only variants isolate the additional relocation layer within v9.

## Run selection first

Prerequisite: the existing frozen `results/train/checkpoints/ears_safe.json`.
Real-data selection deliberately fails if that checkpoint or real OSM maps are
missing. It never silently creates a replacement checkpoint.

From the repository root, run:

```bash
docker compose run --rm --build publication python experiments/benchmark_ears_v9.py --mode select --agents-grid 4,8,16,32 --dropout-grid 0,0.03,0.15,0.30 --episodes 4 --require-real-data --output-root results/v9/selection-20261012
```

This is 768 episodes: two validation cities (Amsterdam, Prague), all four
validation start zones, 16 settings and six methods. Four episodes means one
per start zone, not four independent map replicates. Use additional repetitions
(episode counts must be a multiple of four)
if needed before freezing a stronger claim. The candidate set is fixed in code:

| Candidate | Information weight | Coverage weight | Relocation |
| --- | ---: | ---: | --- |
| EARS-Info-v9 | 2.0 | 0.35 | enabled |
| EARS-Info-Local-v9 | 2.0 | 0.35 | disabled |
| EARS-Balanced-v9 | 1.0 | 1.5 | enabled |
| EARS-Balanced-Local-v9 | 1.0 | 1.5 | disabled |

The two remaining methods are unchanged AntSwarmSafe and frozen EARS-Safe-v7.
All methods share maps, initial state, episode seeds, safety monitor rules,
dropout rates, stopping rules and score weights. Execution order rotates by
episode to reduce systematic timing bias.

A candidate is eligible only when its validation mean J exceeds v7 by at least
0.002, no city/setting loses more than 0.005, mean discovery does not regress,
mean J is at least Ant's, mean runtime is no more than twice v7's, and no actual
incident, exhausted battery or blocked move is recorded. The best eligible mean
is selected. If none passes, `selected_model.json` retains **v7**. These are
declared engineering gates, not a statistical generalization claim from two cities.

The alias `docker compose run --rm --build ears-v9-select` uses the same defaults
and creates a unique timestamped output directory.

## Where outputs are saved

The command above creates `results/v9/selection-20261012/` on the host through
the existing `./results:/app/results` bind mount. `--rm` removes the container,
not these host files. The runner prints its output directory at start and finish.

- `episodes.csv`: flushed after every completed episode, including failed runs.
- `diagnostics.jsonl`: paired per-policy diagnostics, flushed with each episode.
- `manifest.json`: running/completed/failed/interrupted status, counts, code and
  checkpoint hashes, actual start-zone map hashes, versions and provenance.
- `city_means.csv`, `summary.csv`, `paired_city_tests.csv`, `index.html`.
- `selected_model.json`: only after complete real-data validation selection.

The selected artifact freezes a controller choice and parameters, not trained
neural weights; it also requires the original v7 checkpoint. An existing
nonempty output directory is rejected. Use a new directory for a new run.
Interrupted runs preserve completed episodes but do not automatically resume.
The manifest's completed count is refreshed after each full method group and
at handled exceptions; after an uncatchable process kill, CSV may be slightly
ahead of that count. A partial run never creates a selected model.

## Evaluate the frozen choice

First inspect the validation decision and its rejection reasons. If a new
candidate passes, use the exact selected artifact. Do not edit it to force a win.

Optional regression comparison on the eight already inspected development cities:

```bash
docker compose run --rm --build publication python experiments/benchmark_ears_v9.py --mode development --selection results/v9/selection-20261012/selected_model.json --episodes 20 --require-real-data --output-root results/v9/development-20261012
```

Independent confirmation, after freezing the choice:

```bash
docker compose run --rm --build publication python experiments/benchmark_ears_v9.py --mode confirm --selection results/v9/selection-20261012/selected_model.json --episodes 20 --require-real-data --output-root results/v9/confirmation-20261012
```

`confirmation_protocol_v9.json` reserves the six cities already set aside by v8:
Zurich, Madrid, Lisbon, Stockholm, Helsinki and Copenhagen. If they were inspected
elsewhere, they are no longer untouched confirmation geography. Do not tune on them.
Confirmation checks reachable missions against the **actual start-zone bases**.
Data-quality failures remain failures; targets are not dropped or replaced.

The runner rejects code/checkpoint changes after selection and grid/horizon
changes. Robot/dropout settings must have been included in selection. Paired
reports compare candidates directly with v7 and Ant. Per-setting p-values are
unadjusted/exploratory; publication claims need a prespecified primary comparison
and appropriate uncertainty and multiplicity analysis.

## Tests and limits of current evidence

```bash
python -m pytest -q tests/test_ears_v9.py tests/test_benchmark_ears_v9.py tests/test_ears_v8.py tests/test_ears_v6.py tests/test_runtime_monitor.py tests/test_safety_rules.py
docker compose run --rm --build ears-v9-smoke
```

Smoke mode may use explicit synthetic maps/default EARS parameters, clearly
records those sources, and cannot write a selected model. Initial synthetic
checks show substantially reduced runtime but mixed J: some sparse-team cases
improve and some larger-team cases regress. They do not establish real-city gains.
The uploaded archive contains metrics, not the frozen checkpoint or city snapshots,
so the real validation selection has not been run in the implementation environment.

The legacy simulator terminates when all targets are found. Earlier discovery
can therefore lower coverage-at-termination; examine discovery, energy, distance,
steps and J together. Do not change the score or termination to manufacture a win.
Communication dropout still uses globally shared maps; it does not model a fully
distributed sensor network. Legacy map caches may contain synthetic target
fallbacks without an explicit target-provenance flag. OSM-derived grid results
are not physical-robot evidence, and passing the unchanged safety monitor is
not a formal safety guarantee.
