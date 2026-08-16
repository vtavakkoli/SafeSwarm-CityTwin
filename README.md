# SafeSwarm-CityTwin

A reproducible, safety-constrained multi-agent city-search benchmark with **real OpenStreetMap city snapshots**, explicit **train/validation/test separation**, runtime safety assurance, and a trainable GRPO-Safe policy with shared swarm memory and geographic propagation.

## Why this version exists

The original benchmark compared 15 algorithms fairly at inference time, but the PPO/GRPO entries were lightweight execution baselines rather than independently trained policies. That made the previous real-city report useful for execution benchmarking but not for claims about learned generalization.

This repository now separates the workflow into four auditable stages:

1. **Prepare real city data** and freeze/cache the OSM-derived snapshots.
2. **Train** PPO-family residual policies only on the training cities and training start zones.
3. **Test** all algorithms on held-out cities and geographically unseen start zones.
4. **Combine** the training and held-out results into one publication-oriented HTML report.

The test ranking is never hard-coded. `GRPO-Safe` is given the intended mechanism—shared swarm memory, frontier propagation, group-relative behavior selection, and a clipped-PPO residual—but it is declared best only if the held-out metrics actually support that conclusion.

## Real-city protocol

`configs/real_city_protocol.json` defines the experiment contract.

| Split | Cities | Starting zones |
|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center |
| Validation | Amsterdam | west |
| Test | San Francisco, Paris | north-east, south-west |

The city sets are disjoint and the test start zones never appear in training. This creates two simultaneous generalization tests: **unseen city structure** and **unseen deployment geography**.

Real snapshots are downloaded with OSMnx, cached under `data/cache/`, and attributed to © OpenStreetMap contributors. Synthetic fallback is clearly labelled and is rejected by the real-data Docker workflow.

## GRPO-Safe: swarm memory + geographic propagation

`TrainableGRPOMemoryPolicy` extends the existing GRPO-Safe execution policy with a shared spatial memory field. The memory is updated only from information available to the swarm during execution:

- sensed observation values;
- uncertainty and exploration-frontier state;
- pheromone traces;
- visit history;
- inter-agent spatial context.

The map decays over time and diffuses to neighboring cells to create a **geographic propagation field**. Agents can therefore move along promising search frontiers instead of repeatedly making only local independent decisions. Memory and frontier values are also inputs to the learnable PPO residual.

The GRPO behavior group still chooses among exploration, high-priority exploitation, pheromone following, communication, unresolved-target revisit, redundant-coverage reduction, and energy saving. A lightweight clipped-PPO residual is trained over interpretable action features. The residual is deliberately linear/NumPy-based so checkpoints remain inspectable and training remains reproducible without a heavyweight neural stack.

## Trainable policies

The training stage fits:

- `GRPO-Safe` — group-relative behavior selection + swarm memory + propagation + PPO residual
- `IPPO-Safe` — independent PPO-style baseline + PPO residual
- `MAPPO-Safe` — centralized-context MAPPO-style baseline + PPO residual
- `HAPPO-Safe` — heterogeneous-agent PPO-style baseline + PPO residual

The remaining primary methods stay fixed baselines during testing. The held-out evaluation additionally creates two controlled GRPO ablations from the same trained checkpoint:

- `GRPO-Safe-Ablation-NoMemory` — disables memory, frontier, and propagation influence;
- `GRPO-Safe-Ablation-NoPropagation` — keeps memory/frontier use but disables geographic diffusion/gradient propagation.

This makes it possible to test whether a GRPO gain actually comes from the swarm-memory and propagation mechanisms.

## Docker workflow

### 1. Prepare real data

```bash
docker compose up --build prepare-data
```

Outputs `results/prepare-data/city_data_manifest.csv` and `manifest.json`.

### 2. Train

```bash
docker compose up --build train
```

Docker runs 18 episodes per trainable strategy by default, giving two passes over every train-city/start-zone pairing. It writes checkpoints, `training_history.csv`, `training_summary.csv`, and a manifest under `results/train/`.

For a larger publication run:

```bash
TRAIN_EPISODES=27 AGENTS=8 GRID_SIZE=40 MAX_STEPS=200 docker compose up --build train
```

### 3. Held-out test

```bash
docker compose up --build test
```

The test step requires the four trained checkpoints; it fails instead of silently substituting untrained PPO policies. By default it evaluates the 15 primary strategies plus two GRPO mechanism ablations for 20 episodes per held-out city on San Francisco and Paris while alternating the unseen north-east and south-west start zones.

Outputs:

```text
results/test/
├── report.html
├── manifest.json
└── tables/
    ├── episode_results.csv
    ├── city_ranking.csv
    └── overall_ranking.csv
```

The held-out report includes a 95% confidence interval for the operational score.

### 4. Complete pipeline + combined report

```bash
docker compose up --build pipeline
```

This runs prepare → train → test → combined report and writes `results/report.html`.

For deterministic CI/development without network access:

```bash
docker compose up --build pipeline-offline
```

## Existing inference benchmark

The original all-city execution benchmark remains available:

```bash
docker compose up --build benchmark-real-cities
docker compose up --build benchmark-offline
```

It is intentionally kept separate from the train/test protocol so old results remain reproducible.

## Evaluation score

Every strategy is ranked with the same hardware-independent operational score:

| Component | Weight |
|---|---:|
| Weighted priority-target discovery | 35% |
| Traversable-area coverage | 20% |
| Actual safety | 20% |
| Energy efficiency | 10% |
| Coordination / low redundant coverage | 10% |
| Communication availability | 5% |

Runtime is reported separately. Safety-filter interventions are diagnostics; actual incidents determine the safety term.

## GRPO hypothesis and scientific integrity

The intended research hypothesis is:

> Shared swarm memory + geographic frontier propagation + group-relative behavior selection should improve GRPO-Safe on difficult unseen-city search tasks.

The repository tests that hypothesis without manipulating the ranking:

- no test-city metric is used during training;
- test cities are disjoint from train cities;
- test start zones are unseen during training;
- episodic swarm memory is reset between cities/episodes;
- every baseline receives the same environment, seeds, agent count, safety monitor, and episode budget;
- GRPO memory and propagation ablations isolate mechanism contribution;
- confidence intervals are reported from independent held-out episodes;
- the combined report explicitly shows the held-out winner even if it is not GRPO-Safe.

For a paper, use 27–30 independent training episodes and 20–30 test episodes per held-out city, archive `results/`, and report both checkpoint metadata and the OSM provenance manifest.

## Repository layout

```text
configs/
  real_cities.json
  real_city_protocol.json
experiments/
  prepare_real_city_data.py
  train_real_city_policies.py
  test_real_city_policies.py
  build_train_test_report.py
  run_train_test_pipeline.py
  run_city_benchmark.py
src/
  agents/
    marl_baselines.py
    trainable_policies.py
  environment/
  evaluation/
  safety/
  training/
    geography.py
tests/
results/
data/cache/
docker-compose.yaml
```

## Tests

```bash
docker compose up --build unit-test
```

The tests cover protocol separation, geographic spawn zones, GRPO memory propagation, checkpoint round-tripping, PPO residual updates, safety rules, city layers, ranking, and the original baselines. GitHub Actions also runs the offline prepare→train→test→combined-report smoke pipeline.

## Data provenance and licensing

Real snapshots come from OpenStreetMap through OSMnx. Reports and manifests retain the required attribution:

> © OpenStreetMap contributors

OpenStreetMap data is available under the Open Data Commons Open Database License. The software in this repository is released under the MIT License.

## Research scope

The repository supports reproducible comparative research under the configured cities, start zones, and operational score. It does **not** claim universal superiority of any algorithm, and the NumPy PPO residual should not be presented as a full deep-neural reproduction of the original IPPO/MAPPO/HAPPO papers. Its purpose is transparent, controlled policy learning within the SafeSwarm CityTwin benchmark.
