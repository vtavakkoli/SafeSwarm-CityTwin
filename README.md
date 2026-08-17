# SafeSwarm-CityTwin

A reproducible research benchmark for **safety-constrained multi-agent urban search** on real OpenStreetMap city snapshots. SafeSwarm separates training, robust validation, and held-out testing; enforces an observable-only policy contract; and compares learned MARL controllers with bio-inspired swarm search.

## v4: generalization-first model selection + SPARX

The v3 real-city experiment exposed a second failure mode after the earlier safety/credit problems were fixed: PPO-family controllers could look strong on the validation domain and still generalize poorly to unseen cities. The held-out policies travelled extensively but produced high redundant coverage, while AntSwarmSafe spread the team more effectively and discovered far more targets.

v4 addresses that problem in two complementary ways:

1. **robust checkpoint selection** across multiple unseen validation cities and start zones; and
2. a new explicit swarm-search algorithm, **SPARX — Swarm Probability-map Allocation & Region eXploration**.

The held-out test still decides the winner. The repository does not alter scores or select a model from test results.

## Experimental protocol

`configs/real_city_protocol.json` is the experiment contract.

| Split | Cities | Starting zones |
|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center |
| Validation | Amsterdam, Prague | west, east, north, south |
| Test | San Francisco, Paris | north-east, south-west |

The city sets are disjoint. Validation start zones are unseen during training and disjoint from test start zones. Test cities and test starts are never consulted for checkpoint or pattern selection.

Real snapshots are downloaded with OSMnx, cached under `data/cache/`, and attributed to © OpenStreetMap contributors. Synthetic fallback is labelled and rejected by the real-data Docker workflow.

## Fair observation contract

Primary benchmark policies may use only information available at execution time:

- sensed observations;
- uncertainty/frontier state;
- pheromone and visit history;
- current swarm positions and communication state;
- battery/base information;
- known obstacles and restricted zones.

Ground-truth mission coordinates and hidden priority labels are reserved for evaluation. This applies to learned policies and the primary Ant/Bee/PSO/Greedy/SafeSwarm baselines.

## PPO/GRPO v4 checkpoint selection

The v3 PPO/GRPO learner remains in place: safe-action masking before sampling, per-agent difference rewards, per-agent GAE, centralized value baseline during training, minibatch PPO, entropy annealing, gradient clipping, teacher bootstrap for GRPO, and validation-based early stopping.

v4 changes **which weight is promoted**. A validation candidate receives the conservative score

```text
robust = mean
         - 0.50 × CI95
         - 0.20 × domain_std
         - 0.10 × (mean - worst_domain)
```

The final checkpoint is updated only when this robust validation score improves. Every improvement is archived under:

```text
results/train/validation-improvements/<strategy>/epoch_XXX.json
```

This avoids treating one unusually favorable validation topology as sufficient evidence of generalization.

## SPARX — probability memory + region allocation + pattern search

SPARX targets the high-redundancy failure mode directly.

### 1. Shared search-probability map

At each step SPARX builds a normalized search-utility probability map from observable evidence:

- confidence-weighted sensed priority;
- unexplored uncertainty;
- exploration frontiers;
- pheromones;
- visit novelty/revisit pressure;
- signed shared swarm memory;
- current swarm occupancy.

Resolved targets become repulsive in memory so agents move away after a successful discovery. Positive evidence can diffuse through traversable four-neighbor geography but never through known obstacles/restricted cells.

The map is a **search-allocation probability**, not a calibrated probability that a hidden target occupies a cell.

### 2. City segmentation

The traversable grid is divided into approximately `ceil(sqrt(number_of_agents))²` spatial regions. Each region is scored by probability mass, frontier density, unexplored share, revisit pressure, agent distance, and battery state.

Agents are assigned to different promising regions before local movement is selected. This explicit swarm-level task allocation is the major difference from purely local PPO action preferences.

### 3. X / + / ★ search patterns

Three pattern mechanisms are tuned and reported separately:

```text
SPARX-X-Safe          diagonal X rays

\   /
 \ /
  X
 / \
/   \

SPARX-Plus-Safe       cardinal + rays

  |
--+--
  |

SPARX-Star-Safe       X + Plus, eight rays

\ | /
- ★ -
/ | \
```

`SPARX-Safe` is the pattern selected **only from validation**. The held-out X/Plus/Star rows remain visible for mechanism analysis but are never used retrospectively to select the final pattern.

Full methodology: [`docs/SPARX_V4.md`](docs/SPARX_V4.md).

## Algorithms

The benchmark includes:

- **SPARX-Safe** — validation-selected probability-memory/region-pattern search;
- **SPARX-X-Safe / SPARX-Plus-Safe / SPARX-Star-Safe** — pattern mechanism variants;
- **GRPO-Safe** — learned behavior selection + signed swarm memory + propagation;
- **IPPO-Safe, MAPPO-Safe, HAPPO-Safe** — trained PPO-family policies;
- **AntSwarmSafe, BeeSwarmSafe, PSOSwarmSafe, UA-HBAS-Safe**;
- observable fixed GRPO/IPPO/MAPPO/QMIX/MADDPG/HAPPO/MAT baselines;
- Greedy, SafeSwarm and random/reference policies.

GRPO ablations remain available:

- `GRPO-Safe-Ablation-NoMemory`;
- `GRPO-Safe-Ablation-NoPropagation`;
- `GRPO-Safe-Ablation-NoLearnedBehavior`.

## Reproduce

### 1. Prepare real city data

```bash
docker compose up --build prepare-data
```

### 2. Train PPO/GRPO and save every validation improvement

```bash
docker compose up --build train
```

Outputs include:

```text
results/train/
├── checkpoints/
├── candidates/
├── validation-improvements/
├── training_history.csv
├── validation_history.csv
├── validation_improvements.json
├── training_summary.csv
└── teacher_bootstrap.json
```

### 3. Tune X / + / ★ SPARX

```bash
docker compose up --build train-sparx
```

Outputs include:

```text
results/train/
├── checkpoints/
│   ├── sparx_safe.json
│   ├── sparx_x_safe.json
│   ├── sparx_plus_safe.json
│   └── sparx_star_safe.json
├── sparx-validation-improvements/
├── sparx_tuning_history.csv
├── sparx_pattern_summary.csv
└── sparx_manifest.json
```

### 4. Frozen held-out test

```bash
docker compose up --build test
```

The test evaluates San Francisco and Paris with unseen north-east/south-west starts and writes 95% confidence intervals, per-city rankings, pattern diagnostics, discovery, coverage, redundancy, energy, safety and runtime.

### 5. Complete pipeline

```bash
docker compose up --build pipeline
```

For deterministic CI without network access:

```bash
docker compose up --build pipeline-offline
```

## Evaluation score

Every strategy uses the same hardware-independent operational score:

| Component | Weight |
|---|---:|
| Weighted target discovery | 35% |
| Traversable-area coverage | 20% |
| Actual safety | 20% |
| Energy efficiency | 10% |
| Coordination / low redundancy | 10% |
| Communication availability | 5% |

Runtime is reported separately. A credible improvement should therefore be visible not only in the aggregate score but also in discovery, coverage, redundancy and energy.

## Research integrity

SafeSwarm v4 deliberately separates optimization from evaluation:

- training uses only training cities/start zones;
- PPO/GRPO weights are selected by robust validation only;
- each validation-improving checkpoint is retained;
- SPARX weights are tuned on training and gated by validation improvement;
- X/Plus/Star selection uses validation only;
- San Francisco/Paris never influence model or pattern selection;
- hidden target labels are unavailable to primary policies;
- confidence intervals are reported on held-out episodes;
- reports always show the actual held-out winner, even if it is not SPARX or GRPO.

For publication claims, run several independent top-level training/tuning seeds and report per-city/per-start-zone distributions as well as 95% confidence intervals.

## Repository layout

```text
configs/
  real_city_protocol.json
experiments/
  prepare_real_city_data.py
  train_real_city_policies.py       # stable entry point → v4
  train_real_city_policies_v4.py
  train_sparx_patterns.py
  test_real_city_policies.py
  run_train_test_pipeline.py
  build_train_test_report.py
src/
  agents/
    grpo_v3.py
    ppo_v3.py
    sparx_pattern.py
    observable_utils.py
    bio_swarm_agents.py
  environment/
  evaluation/
  safety/
  training/
    geography.py
    policy_learning.py
    validation_selection.py
docs/
  SPARX_V4.md
tests/
results/
data/cache/
docker-compose.yaml
```

## Tests

```bash
docker compose up --build unit-test
```

Tests cover safety-aware credit, GRPO state/memory behavior, geographic split integrity, SPARX probability normalization, hidden-target leakage protection, region assignment, X/Plus/Star geometry, SPARX checkpoint round-tripping and robust validation scoring. GitHub Actions also runs the complete offline v4 smoke pipeline and verifies the generated SPARX artifacts.

## Scope

This is a research benchmark, not a claim of universal algorithm superiority. The NumPy PPO/critic stack is intentionally auditable and should not be presented as a full deep-neural reproduction of the original IPPO/MAPPO/HAPPO papers. SPARX is a new benchmark algorithm whose superiority, if any, must be established by the frozen held-out experiment rather than assumed from its design.
