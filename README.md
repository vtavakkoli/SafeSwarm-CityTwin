# SafeSwarm-CityTwin

A reproducible research benchmark for **safety-constrained multi-agent urban search** on real OpenStreetMap city snapshots, with explicit train/validation/test separation, runtime assurance, partially observable policy inputs, and a trainable GRPO-Safe controller with shared swarm memory.

## What changed in v3

The latest real-city run showed that the previous training stack was still not learning the intended GRPO behavior well. GRPO-Safe reached only about 0.32 on the held-out operational score while the fixed AntSwarmSafe baseline was substantially stronger. The failure was not one missing weight. It came from several interacting methodological problems.

### 1. The benchmark was not observationally fair

Some fixed baselines could read the ground-truth `priority_map`, `mission_zones`, or `remaining_missions()` when choosing actions, while the trainable GRPO actor had deliberately been restricted to sensed `observation_map` values. That made the old ranking partly an **oracle-vs-partially-observed** comparison.

v3 introduces an explicit partial-observability contract:

> Ground-truth mission coordinates and priority labels are evaluation data. Primary benchmark policies may act only on information available at runtime.

`AntSwarmSafe`, `BeeSwarmSafe`, `PSOSwarmSafe`, `UA-HBAS-Safe`, Greedy/SafeSwarm baselines, and the fixed MARL baselines now use observable guidance derived from sensed evidence, uncertainty, pheromone, visit history, and exploration frontiers. The legacy implementations remain in the repository for reproducibility but are no longer the primary fair-comparison path.

### 2. PPO used team-level credit for every agent

The previous trainer computed one team reward/advantage at each timestep and attached that same advantage to every agent decision. With eight agents, useful and harmful actions therefore received almost identical credit and policy gradients frequently cancelled.

v3 performs **per-agent difference-style credit assignment**. Each agent receives local credit for:

- weighted target discoveries it actually helped sense;
- first visits and local uncertainty reduction;
- useful swarm spread;
- safe progress toward a base;
- successful safe return;
- repeated/redundant visits;
- unnecessary energy use and idle behavior.

A smaller cooperative team component preserves the multi-agent objective. GAE is then computed independently for each agent trajectory.

### 3. Training was too weak compared with BioSwarm

`BioSwarm-Urban-Monitoring` succeeds partly because its PPO workflow has a much richer optimization lifecycle: teacher imitation, minibatch PPO, entropy/diversity pressure, gradient clipping, learning-rate scheduling, repeated validation, and a much larger training budget.

SafeSwarm v3 imports the most relevant ideas while keeping the implementation auditable and NumPy-only:

- shuffled **minibatch PPO** updates;
- clipped policy objective;
- centralized value baseline with per-agent GAE targets;
- entropy regularization and entropy annealing;
- gradient clipping;
- learning-rate decay;
- KL-based early stopping inside PPO updates;
- repeated stochastic validation;
- validation-based checkpoint selection and training early stopping;
- observable AntSwarm/UA-HBAS **teacher bootstrap for GRPO**.

The NumPy actor/critic is intentionally lightweight and inspectable. It should not be described as a full deep-neural reproduction of the original IPPO/MAPPO/HAPPO papers.

### 4. GRPO behavior selection was not sufficiently state-dependent

v2 learned only a small global bias over seven hand-designed GRPO behaviors. The same correction was therefore applied in Vienna and London, at high and low battery, and in explored and unexplored regions.

v3 adds a **state-conditioned behavior policy** over:

- local/neighbor observed priority;
- uncertainty and frontier strength;
- pheromone intensity;
- novelty;
- team spread and nearby-agent density;
- battery level and normalized base distance;
- unexplored-map fraction.

The seven high-level behaviors are still interpretable, but the policy can now learn when each behavior is useful rather than only learning one global preference.

### 5. Battery dynamics made failure nearly inevitable

The previous CityTwin consumed 1.5 battery units for each move while episodes could last 160 steps. In the real training logs all eight agents repeatedly reached zero battery, and the terminal battery penalty dominated the reward. The safety monitor then spent the end of many trajectories repeatedly handling reserve violations.

v3 calibrates energy to the mission horizon and adds operationally meaningful return semantics:

- movement cost defaults to `0.55` and idle cost to `0.10`;
- a proactive safety guard constrains low-battery actions toward a base before hard reserve violation;
- an agent that reaches a base with low reserve is **safely parked** and stops consuming energy;
- multiple safely parked agents may share the base without being treated as active collisions;
- an episode can terminate immediately when all mission cells are discovered.

This does not remove battery safety. It makes safe return a learnable outcome instead of making depletion the default terminal state.

---

## Real-city protocol

`configs/real_city_protocol.json` defines the experiment contract.

| Split | Cities | Starting zones |
|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center |
| Validation | Amsterdam | west |
| Test | San Francisco, Paris | north-east, south-west |

Train, validation, and test cities are disjoint. Test start zones are never used in training. Checkpoint selection uses validation only; San Francisco and Paris are not consulted until the final held-out test.

Real snapshots are downloaded with OSMnx, cached under `data/cache/`, and attributed to © OpenStreetMap contributors. `--require-real-data` rejects synthetic fallback for publication runs.

## GRPO-Safe v3

`TrainableGRPOMemoryPolicy` combines three learned/interpretable levels:

1. **State-conditioned high-level GRPO behavior selection** among exploration, observed-priority exploitation, pheromone following, communication-aware search, revisit handling, redundancy reduction, and energy-safe return.
2. **Signed swarm memory and geographic propagation** built only from runtime-observable information.
3. **Safety-masked PPO action residual** over candidate grid moves.

The spatial memory contains sensed evidence, uncertainty/frontier state, pheromone traces, visit history, known blocked areas, and current swarm occupancy. It never stores unseen ground-truth target labels. Useful unexplored regions are attractive; revisits, congestion, discovered targets, and unsafe boundaries are repulsive.

### GRPO teacher bootstrap

Before PPO fine-tuning, GRPO can be warm-started from the strongest observable swarm heuristics (`AntSwarmSafe` and `UA-HBAS-Safe`). The teacher rollout is filtered through the same runtime-safety contract and produces supervised action/behavior examples. PPO then remains free to improve beyond the teachers on the training cities.

This follows the useful training idea in BioSwarm without transferring hidden target information.

## Safety and action credit

Trainable policies use **safe action masking before sampling**:

1. enumerate candidate actions;
2. remove boundary, obstacle, restricted-zone, collision, battery-reserve, communication, and proactive return-guard violations;
3. normalize the trainable policy over the remaining action set;
4. sample and store the exact action probability;
5. execute that same action.

Last-resort runtime fallback remains available, but a forced fallback does not receive PPO credit. Mask rejections, return-guard interventions, and executed safety interventions are reported separately.

## Training lifecycle

A normal training epoch covers the complete training geography: 3 cities × 3 starting zones = 9 scenarios. PPO is updated after the geography batch, then the candidate checkpoint is evaluated repeatedly on the disjoint Amsterdam validation split.

Default v3 training requests:

- **108 training episodes per strategy**;
- 12 validation episodes × 2 validation repeats per checkpoint;
- 6 PPO update epochs with shuffled minibatches;
- entropy annealing and learning-rate decay;
- GRPO teacher bootstrap across the training geography;
- validation-based early stopping.

The selected checkpoint is the highest validation operational score, never the best test score.

## Controlled GRPO ablations

Held-out evaluation produces the full trained GRPO checkpoint plus:

- `GRPO-Safe-Ablation-NoMemory`;
- `GRPO-Safe-Ablation-NoPropagation`;
- `GRPO-Safe-Ablation-NoLearnedBehavior`.

This makes the research claim falsifiable. A strong result should show not only that GRPO performs well, but that removing the proposed mechanisms measurably degrades held-out performance.

## Docker workflow

### Prepare real data

```bash
docker compose up --build prepare-data
```

### Train + validate

```bash
docker compose up --build train
```

Outputs include:

```text
results/train/
├── checkpoints/
├── candidates/
├── training_history.csv
├── validation_history.csv
├── training_summary.csv
├── teacher_bootstrap.json
└── manifest.json
```

### Held-out test

```bash
docker compose up --build test
```

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

### Complete pipeline

```bash
docker compose up --build pipeline
```

For deterministic offline CI/development:

```bash
docker compose up --build pipeline-offline
```

### Larger publication run

```bash
TRAIN_EPISODES=180 \
VALIDATION_EPISODES=24 \
VALIDATION_REPEATS=3 \
TEST_EPISODES=30 \
TEACHER_BOOTSTRAP_SCENARIOS=9 \
AGENTS=8 GRID_SIZE=40 MAX_STEPS=160 \
  docker compose up --build pipeline
```

Use several independent top-level experiment seeds for a final paper; do not treat one training seed as a confidence interval over model training.

## Evaluation score

Every primary strategy is ranked with the same hardware-independent operational score:

| Component | Weight |
|---|---:|
| Weighted priority-target discovery | 35% |
| Traversable-area coverage | 20% |
| Actual safety | 20% |
| Energy efficiency | 10% |
| Coordination / low redundant coverage | 10% |
| Communication availability | 5% |

Runtime is reported separately. Masked candidates and preventive interventions are diagnostics; actual executed incidents determine the safety component.

## Scientific integrity

The v3 experimental contract requires:

- no hidden mission/priority labels in primary policy action selection;
- disjoint train/validation/test cities;
- unseen held-out start zones;
- validation-only checkpoint selection;
- same city snapshot, start zone, seed, agent count, horizon, observation contract, and runtime safety rules across compared strategies;
- per-agent training credit for trainable policies;
- episodic memory reset;
- explicit GRPO mechanism ablations;
- confidence intervals from independent held-out episodes;
- full reporting even when GRPO is not the winner.

**Important:** numerical results generated by the v2 implementation should not be reused as v3 results. The observation contract, energy semantics, optimizer, reward/credit assignment, and GRPO controller changed materially. Re-run prepare → train → test and archive the new `results/` directory.

## Repository layout

```text
configs/
  real_city_protocol.json
experiments/
  prepare_real_city_data.py
  train_real_city_policies.py          # stable entry point
  train_real_city_policies_v3.py       # v3 implementation
  test_real_city_policies.py
  build_train_test_report.py
  run_train_test_pipeline.py
src/
  agents/
    observable_utils.py
    observable_marl.py
    safe_ppo_core.py
    ppo_v3.py
    grpo_memory_v2.py
    grpo_v3.py
    trainable_policies.py
  environment/
    city_twin.py
  safety/
    rules.py
    runtime_monitor.py
  training/
    geography.py
    policy_learning.py
    teacher_bootstrap.py
tests/
docker-compose.yaml
```

## Tests

```bash
docker compose up --build unit-test
```

Regression tests cover protocol separation, observable-only baseline behavior, safety-masked action credit, proactive safe return, mission-completion termination, per-agent credit assignment, GRPO memory propagation, state-conditioned behavior learning, teacher imitation, checkpoint round-tripping, safety rules, city layers, and ranking. GitHub Actions also runs the offline benchmark and the complete offline prepare → train → validation → held-out test → report smoke pipeline.

## Data provenance and licensing

Real snapshots come from OpenStreetMap through OSMnx. Reports/manifests retain:

> © OpenStreetMap contributors

OpenStreetMap data is available under the Open Data Commons Open Database License. Repository software is MIT licensed.

## Research scope

SafeSwarm-CityTwin is a controlled research benchmark, not a claim of universal algorithm superiority. The v3 NumPy PPO stack is designed for transparent experimentation and mechanism studies. For a publication claiming full neural PPO/GRPO equivalence, add a neural implementation and compare it as a separate model family rather than overstating this lightweight implementation.
