# SafeSwarm-CityTwin

A reproducible, safety-constrained multi-agent city-search benchmark with **real OpenStreetMap city snapshots**, explicit **train/validation/test separation**, runtime safety assurance, and a trainable GRPO-Safe policy with shared swarm memory and geographic propagation.

## Why this version exists

The original benchmark compared 15 algorithms at inference time, but the PPO/GRPO entries were lightweight execution baselines rather than independently trained policies. The first train/test extension added checkpoints and GRPO memory, but real training exposed an important methodological flaw: PPO could log one proposed action while the runtime monitor executed a different safe replacement. That makes credit assignment noisy and affected GRPO particularly strongly because it triggered many safety interventions.

The current training stack fixes that problem by making safety part of the trainable action distribution itself. Trainable policies **mask unsafe actions before sampling**, so a PPO decision is trained only when that same action can be executed.

The repository separates the workflow into four auditable stages:

1. **Prepare real city data** and freeze/cache OSM-derived snapshots.
2. **Train** only on training cities/start zones using balanced geography batches.
3. **Select** checkpoints only on the disjoint validation split.
4. **Test** frozen checkpoints on held-out cities and geographically unseen start zones, then build one publication-oriented report.

The test ranking is never hard-coded. `GRPO-Safe` receives the intended memory/search mechanism, but it is declared best only if the held-out metrics actually support that conclusion.

## Real-city protocol

`configs/real_city_protocol.json` defines the experiment contract.

| Split | Cities | Starting zones |
|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center |
| Validation | Amsterdam | west |
| Test | San Francisco, Paris | north-east, south-west |

The city sets are disjoint and the test start zones never appear in training. This creates two simultaneous generalization tests: **unseen city structure** and **unseen deployment geography**.

Real snapshots are downloaded with OSMnx, cached under `data/cache/`, and attributed to © OpenStreetMap contributors. Synthetic fallback is labelled and rejected by the real-data Docker workflow.

## PPO v2: correct safety-aware credit assignment

Trainable PPO-family policies use a sequential runtime-safe action mask before stochastic sampling:

1. enumerate candidate actions;
2. remove actions violating boundary, obstacle, restricted-zone, collision, battery-reserve, or communication rules;
3. normalize the policy only over the remaining safe actions;
4. sample one action and store its exact probability;
5. execute that same action in the environment.

If no legal action exists, the runtime monitor can still apply a last-resort fallback, but that forced fallback does **not** create a PPO training sample. `mask_rejections` and true runtime `safety_interventions` are reported separately.

The actor also removes oracle target information: trainable execution uses sensed observations rather than hidden target labels. A lightweight centralized linear critic is used only during training, with GAE-style advantages, while execution remains decentralized.

## GRPO-Safe: learned behavior + signed swarm memory

`TrainableGRPOMemoryPolicy` now trains two policy levels:

- a high-level GRPO behavior distribution over exploration, exploitation, pheromone following, communication, revisit handling, redundancy reduction, and energy-safe return;
- a spatial action policy whose learned residual includes swarm memory, frontier utility, propagation gradient, and return-to-base progress.

The memory uses only execution-available evidence:

- sensed observations;
- uncertainty and exploration-frontier state;
- pheromone traces;
- visit history;
- known obstacles/restricted regions;
- current swarm occupancy.

It is a **signed search-utility field**, not a hidden target map. Unexplored frontiers and useful nearby evidence are attractive; revisits, congestion, discovered targets, and unsafe boundaries are repulsive. Positive utility propagates only through traversable unexplored geography.

The previous fixed memory/frontier/propagation bonuses are now trainable policy weights. The `save_energy` behavior also prefers progress toward a base rather than simply remaining stationary.

## Trainable policies

The training stage fits:

- `GRPO-Safe` — trainable behavior selector + signed swarm memory + propagation + PPO residual;
- `IPPO-Safe` — independent PPO-style baseline + safety-masked PPO residual;
- `MAPPO-Safe` — centralized-context MAPPO-style baseline + safety-masked PPO residual;
- `HAPPO-Safe` — heterogeneous-agent PPO-style baseline + safety-masked PPO residual.

The held-out evaluation additionally creates two controlled GRPO ablations from the same selected checkpoint:

- `GRPO-Safe-Ablation-NoMemory` — disables memory, frontier, and propagation influence;
- `GRPO-Safe-Ablation-NoPropagation` — keeps local memory/frontier use but disables geographic propagation.

A GRPO gain is therefore useful scientifically only if the full policy also beats these ablations on the held-out split.

## Balanced training and checkpoint selection

Normal training performs updates only after a complete **city × start-zone batch**. With 3 training cities and 3 training zones, one epoch contains 9 geography scenarios. Environment seeds are paired across strategies to reduce comparison variance.

After each epoch, the current checkpoint is evaluated on the disjoint Amsterdam validation split. The checkpoint with the highest mean validation operational score is copied to `results/train/checkpoints/`. **San Francisco and Paris are never consulted for checkpoint selection.**

`final_train_score` now means the mean score of the final balanced training batch rather than the score of one stochastic final episode. `training_summary.csv` additionally reports the best validation score, confidence interval, and selected epoch.

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

Docker requests **54 training episodes per strategy by default**, equivalent to six complete 3-city × 3-zone batches, plus 6 validation episodes after each epoch. It writes:

```text
results/train/
├── checkpoints/          # best validation-selected checkpoints
├── candidates/           # latest epoch candidates
├── training_history.csv
├── validation_history.csv
├── training_summary.csv
└── manifest.json
```

For a larger publication run:

```bash
TRAIN_EPISODES=90 VALIDATION_EPISODES=12 AGENTS=8 GRID_SIZE=40 MAX_STEPS=200 \
  docker compose up --build train
```

### 3. Held-out test

```bash
docker compose up --build test
```

The test step requires all four trained checkpoints and fails instead of silently substituting untrained policies. By default it evaluates the primary strategies plus the two GRPO mechanism ablations for 20 episodes per held-out city on San Francisco and Paris while alternating the unseen north-east and south-west start zones.

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

The held-out report includes 95% confidence intervals for the operational score.

### 4. Complete pipeline + combined report

```bash
docker compose up --build pipeline
```

This runs prepare → train/validate → held-out test → combined report and writes `results/report.html`.

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

It is intentionally separate from the train/test protocol so legacy results remain reproducible.

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

Runtime is reported separately. Masked unsafe candidates and runtime interventions are diagnostics; actual incidents determine the safety term.

## GRPO hypothesis and scientific integrity

The intended research hypothesis is:

> Learned shared swarm memory + geographic frontier propagation + group-relative behavior selection should improve GRPO-Safe on difficult unseen-city search tasks.

The repository tests that hypothesis without manipulating the ranking:

- train, validation, and test cities are disjoint;
- test start zones are unseen during training;
- no test metric selects a checkpoint;
- train updates use balanced geography batches and paired environment seeds;
- PPO samples correspond to executed safe actions;
- forced safety fallbacks do not receive PPO credit;
- trainable actor inputs do not expose hidden target labels;
- episodic swarm memory is reset between cities/episodes;
- GRPO memory/propagation ablations isolate mechanism contribution;
- confidence intervals are reported on held-out episodes;
- the combined report shows the actual held-out winner even if it is not GRPO-Safe.

For a paper, use multiple complete geography batches (for example 90+ training episodes per strategy), at least 20–30 held-out episodes per city, archive `results/`, and report both checkpoint metadata and the OSM provenance manifest. Do not reuse numerical results produced by an older training implementation.

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
    safe_ppo_core.py
    grpo_memory_v2.py
    trainable_policies.py
  environment/
  evaluation/
  safety/
    runtime_monitor.py
  training/
    geography.py
    policy_learning.py
tests/
results/
data/cache/
docker-compose.yaml
```

## Tests

```bash
docker compose up --build unit-test
```

Tests cover protocol separation, geographic spawn zones, safety-masked PPO credit assignment, GRPO memory propagation, learned behavior updates, GAE targets, checkpoint round-tripping, safety rules, city layers, ranking, and the original baselines. GitHub Actions also runs the offline prepare → train → test → combined-report smoke pipeline.

## Data provenance and licensing

Real snapshots come from OpenStreetMap through OSMnx. Reports and manifests retain the required attribution:

> © OpenStreetMap contributors

OpenStreetMap data is available under the Open Data Commons Open Database License. The software in this repository is released under the MIT License.

## Research scope

The repository supports reproducible comparative research under the configured cities, start zones, and operational score. It does **not** claim universal superiority of any algorithm, and the NumPy policy/critic stack should not be presented as a full deep-neural reproduction of the original IPPO/MAPPO/HAPPO papers. Its purpose is transparent, controlled, safety-aware policy learning within the SafeSwarm CityTwin benchmark.
