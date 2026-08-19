# SafeSwarm-CityTwin

A reproducible research benchmark for **safety-constrained multi-agent urban search** on cached real OpenStreetMap city snapshots. SafeSwarm separates training, robust validation, primary held-out testing, and post-selection SWAP stress testing. Primary policies obey an observable-only action contract.

## v6: EARS — improve Ant without destroying what makes Ant strong

The v5 real-city run exposed a useful result: `PRISM-Ant-Safe` could approach AntSwarm's discovery and even exceed it on some SWAP views, yet `AntSwarmSafe` still won overall because it used **less distance, less energy, and less redundant coverage**.

v6 therefore changes the research question.

Instead of continuously blending more global planning into Ant, v6 keeps **AntSwarmSafe unchanged as the reference** and introduces global intelligence only when observable search quality degrades.

### EARS-Safe

**EARS — Event-driven Ant Reallocation Search** uses the original Ant-style local rule by default. Global relocation is triggered only when an agent shows evidence of:

- trajectory stagnation;
- excessive local revisits;
- local swarm congestion.

When triggered, EARS assigns a new observable frontier/evidence goal using a utility that explicitly penalizes:

- path distance;
- predicted movement energy;
- revisited cells;
- agent congestion;
- goals that would compromise safe battery return.

The goal is not “more exploration.” It is **higher marginal target discovery per unit movement**.

### EARS-NP-Safe

`EARS-NP-Safe` adds a repulsive **negative-pheromone exclusion field**. Repeatedly visited and congested cells deposit negative pheromone; the field decays and diffuses into a small spatial halo. This discourages agents from tracing the same corridors or nearly parallel paths.

Positive observable evidence remains in the existing environment pheromone/frontier signals. The negative field is a separate anti-overlap mechanism.

### H-MAPPO-EARS-Safe

`H-MAPPO-EARS-Safe` is a **MAPPO-assisted hierarchical option controller**, not a claim of a new MAPPO implementation. It treats the trained MAPPO checkpoint as one option inside an explicit hierarchy:

1. safe return to base;
2. EARS global relocation;
3. negative-pheromone escape;
4. Ant evidence exploitation;
5. MAPPO local action when the observable state is suitable;
6. Ant local exploration fallback.

This avoids forcing MAPPO to relearn the low-level Ant heuristic that already generalizes well.

## Existing v5 components retained

### PRISM

**PRISM — Probability-guided Region-Integrated Search with Memory** remains the explicit global-search comparison. It uses observable signed memory, normalized search utility, region allocation, and X / Plus / Star local search geometries. Pattern and parameter selection use validation only.

### PRISM-Ant

`PRISM-Ant-Safe` continuously fuses PRISM global allocation with Ant local search. v6 retains it because it is the direct predecessor to the event-driven EARS hypothesis.

### SWAP

**SWAP — Seeded World Alternate Protocol** keeps the same unseen real-city OSM geometry but generates deterministic seed-indexed hidden mission subsets. SWAP is post-selection only and cannot tune any model, PRISM pattern, EARS threshold, or pheromone parameter.

## Experimental protocol

| Split | Cities | Starting zones | May select a model? |
|---|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center | training only |
| Validation | Amsterdam, Prague | west, east, north, south | **yes** |
| Primary v6 test | San Francisco, Paris | north-east, south-west | **no** |
| SWAP test | test cities with seed-indexed alternate mission views | test zones | **no** |

Ground-truth mission coordinates and hidden priority labels are evaluator data. Policies may use sensed observations, uncertainty/frontiers, pheromones, visits, known constraints, battery, communication, and shared observable swarm state.

## v6 pipeline

```text
prepare real OSM snapshots
  ↓
base PPO/GRPO training + robust validation checkpoints
  ↓
v5 coordination + teacher-distillation upgrade
  ↓
PRISM X / Plus / Star + PRISM-Ant validation selection
  ↓
EARS / EARS-NP / H-MAPPO-EARS train-shortlist + validation selection
  ↓
frozen primary held-out test
  ↓
SWAP multi-seed alternate-target stress test
  ↓
combined report
```

Run everything:

```bash
docker compose up --build pipeline
```

## v7 publication-validation suite

v7 does **not** train a new method. It freezes the v6 checkpoints and adds the evidence reviewers typically need before a publication claim.

### Expanded post-selection geography

`configs/publication_protocol_v7.json` keeps the original train/validation cities unchanged and expands the evaluation-only test set to eight cities:

- San Francisco and Paris (the original v6 held-out cities);
- Barcelona and Rome (dense/irregular European street structure);
- Manhattan/New York and Chicago (US grid-dominant urban structure);
- Tokyo and Melbourne (additional geographically distinct urban layouts).

These cities may not select or update EARS.

### Paired and hierarchical statistics

`experiments/analyze_publication_statistics_v7.py` pairs EARS and Ant on the same city, start zone, episode and seed, then reports:

- paired bootstrap 95% confidence intervals;
- a two-sided paired randomization/permutation p-value;
- hierarchical bootstrap 95% intervals with episodes nested inside cities;
- paired effect size and win fraction;
- episode-level and across-city **mean ± standard deviation** tables.

### EARS mechanism ablations

`experiments/test_ears_ablations_v7.py` uses the same frozen EARS checkpoint and evaluates:

- full EARS;
- stagnation-only triggering;
- revisit-only triggering;
- congestion-only triggering;
- relocation with energy and battery-return awareness removed;
- the unchanged `AntSwarmSafe` baseline.

Trigger counts and Ant-vs-relocation fractions are reported per city so the paper can test the mechanism explanation rather than only the final score.

### Operational-score sensitivity

`experiments/score_sensitivity_v7.py` retains the published operational score as the primary metric, then perturbs component weights by ±25% and renormalizes them. It reports how often EARS still beats Ant and how the EARS-minus-Ant margin changes. This guards against a result that exists only for one hand-picked composite weighting.

### Full publication SWAP

The publication pipeline runs SWAP again on the expanded frozen test set. Multiple deterministic alternate hidden-target views are evaluated after all checkpoints are fixed.

### Qualitative winner episode

`experiments/visualize_winner_episode_v7.py` creates one reproducible frozen winner episode with:

- `winner_episode.gif` — animated agent motion and detection progression;
- `winner_visit_heatmap.png` — visit intensity / overlap;
- `winner_trajectory_map.png` — static trajectories and detection order;
- `detection_events.csv` and `summary.json`.

Mission target markers in these figures are explicitly post-evaluation overlays; policies never receive hidden target coordinates.

### Run the publication suite

If v6 checkpoints already exist, they are reused. Otherwise the original v6 train/validation pipeline runs first and is frozen before the v7 evaluation starts.

```bash
docker compose up --build publication
```

For a deterministic CI-sized synthetic smoke test:

```bash
docker compose up --build publication-offline
```

Final outputs are collected under:

```text
results/publication/test/
results/publication/swap-test/
results/publication/statistics/
results/publication/ablations/
results/publication/sensitivity/
results/publication/visualization/
results/publication/report.html
```

## Useful v6 outputs

```text
results/train/ears_candidate_history.csv
results/train/ears_summary.csv
results/train/ears_manifest.json
results/train/checkpoints/ears_safe.json
results/train/checkpoints/ears_np_safe.json
results/train/checkpoints/h_mappo_ears_safe.json
results/test/tables/overall_ranking.csv
results/swap-test/tables/seed_ranking.csv
results/swap-test/tables/overall_ranking.csv
results/report.html
```

## Scientific interpretation

`AntSwarmSafe` is intentionally left unchanged. EARS is successful only if the **frozen** event-driven variants improve the held-out/SWAP result without using test feedback—especially by lowering redundancy, energy, and distance while preserving or improving Ant-level target discovery.

The v7 publication report adds an evidence gate, not a model-selection gate. Strong publication support requires consistent multi-city/SWAP behavior, paired and hierarchical confidence intervals, raw-metric agreement with the proposed mechanism, mechanism ablations, and score-weight robustness. The repository never forces EARS, PRISM, MAPPO, or any other method to win.

## License and data

Repository code is licensed under the project license. Real geographic snapshots are derived from OpenStreetMap and must retain **© OpenStreetMap contributors** attribution and applicable ODbL terms. Synthetic mode exists for deterministic CI/development and must not be presented as real-world evidence.
