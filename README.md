# SafeSwarm-CityTwin

A reproducible research benchmark for **safety-constrained multi-agent urban search** on cached real OpenStreetMap city snapshots. SafeSwarm separates training, robust validation, primary held-out testing, and post-selection stress testing; all primary policies obey an observable-only action contract.

## v5: PRISM, PRISM-Ant, SWAP, and stronger learned baselines

SafeSwarm v5 is driven by the real v4 finding that `AntSwarmSafe` generalized substantially better than the learned PPO-family controllers. The learned policies travelled far but revisited a large fraction of already-covered cells; AntSwarm's novelty, pheromone and anti-clustering inductive bias produced much stronger held-out target discovery.

v5 addresses that failure without changing the test score or forcing a preferred winner.

### 1. SPARX is renamed PRISM

The v4 search controller is now **PRISM — Probability-guided Region-Integrated Search with Memory**. The rename avoids confusion with unrelated uses of the SPARX acronym.

PRISM retains the scientifically relevant mechanism:

- observable-only signed shared swarm memory;
- normalized search-utility probability map;
- uncertainty/frontier/pheromone/novelty evidence;
- repulsion from revisited and resolved cells;
- spatial region segmentation;
- distance/battery-aware multi-agent region assignment;
- X, Plus, and Star local search geometries;
- validation-only weight and pattern selection.

The legacy `sparx_pattern.py` module remains only as a compatibility alias for reproducing v4 checkpoints. **No v5 benchmark strategy is named SPARX.**

### 2. PRISM-Ant-Safe

**PRISM-Ant-Safe** combines complementary strengths instead of replacing one heuristic with another:

- **PRISM** decides *where the team should search*: probability memory, map segmentation, diverse region assignments, and structured pattern goals;
- **AntSwarm** influences *how an agent searches locally*: observed priority, uncertainty×novelty, pheromone following, strong revisit suppression, and anti-clustering.

The fusion strength is selected on the validation split only. The held-out test and SWAP cannot tune it.

### 3. SWAP: Seeded World Alternate Protocol

Changing only the episode RNG seed does **not** create a different real OSM dataset because the cached physical city layer is keyed by place/grid/radius. v5 therefore adds **SWAP — Seeded World Alternate Protocol**.

For each SWAP dataset seed, SafeSwarm keeps the same real city geometry—obstacles, restricted cells and bases—but deterministically builds a different priority-stratified hidden mission subset. Each view records a stable signature and target count.

SWAP is deliberately **post-selection only**. It answers a stronger question: does the frozen ranking survive different hidden mission layouts in the same unseen cities?

### 4. Why IPPO/MAPPO/HAPPO/GRPO were weak, and what v5 changes

The v4 implementation exposed several repository-level limitations beyond algorithm theory:

1. loaded PPO checkpoints still sampled actions stochastically during evaluation;
2. IPPO/MAPPO/HAPPO had no explicit global frontier/goal allocation, leaving the memory/frontier coordination feature slots effectively unused;
3. only GRPO received teacher imitation;
4. local PPO preferences could therefore become long high-redundancy walks even after robust validation selection.

v5 adds:

- **deterministic checkpoint inference** for learned-policy validation/test;
- an observable **shared frontier/evidence goal coordinator** for IPPO/MAPPO/HAPPO and an auxiliary coordination signal for GRPO;
- distinct multi-agent global goals to reduce overlap;
- generic safe-action distillation from `AntSwarmSafe` and `UA-HBAS-Safe` for **all four trainable policies**;
- a separate validation-gated upgrade stage: a distilled checkpoint replaces the base checkpoint only if the multi-domain robust validation score improves.

This is still an auditable NumPy research implementation, not a claim of full deep-neural reproduction of the original IPPO/MAPPO/HAPPO papers.

## Experimental protocol

| Split | Cities | Starting zones | May select a model? |
|---|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center | training only |
| Validation | Amsterdam, Prague | west, east, north, south | **yes** |
| Primary test | San Francisco, Paris | north-east, south-west | **no** |
| SWAP test | test cities with seed-indexed alternate mission views | test zones | **no** |

Ground-truth mission coordinates and priority labels are evaluator data. Policies may use sensed observations, uncertainty/frontiers, pheromones, visits, known constraints, battery, communication and shared observable swarm state.

## v5 pipeline

```text
prepare real OSM snapshots
  ↓
base PPO/GRPO training + robust validation checkpoints
  ↓
v5 coordination + teacher-distillation upgrade
  ↓
PRISM X / Plus / Star tuning + validation pattern selection
  ↓
PRISM-Ant validation fusion selection
  ↓
frozen primary held-out test
  ↓
SWAP seed-shift stress test
  ↓
combined report
```

Run everything:

```bash
docker compose up --build pipeline
```

Or run stages explicitly:

```bash
docker compose up --build prepare-data
docker compose up --build train
docker compose up --build upgrade-ppo
docker compose up --build train-prism
docker compose up --build test
docker compose up --build test-swap
```

Useful outputs:

```text
results/train/training_summary.csv
results/train/coordination_upgrade_summary.csv
results/train/prism_pattern_summary.csv
results/train/prism_ant_summary.csv
results/train/prism_manifest.json
results/train/checkpoints/prism_safe.json
results/train/checkpoints/prism_ant_safe.json
results/test/tables/overall_ranking.csv
results/swap-test/tables/seed_ranking.csv
results/swap-test/tables/overall_ranking.csv
results/swap-test/manifest.json
results/report.html
```

## Scientific interpretation

The repository never assumes PRISM-Ant, PRISM, GRPO, or a PPO baseline must beat AntSwarm. The correct result is whichever frozen method wins the primary held-out and SWAP evaluations. If AntSwarm remains strongest after v5, that is evidence for the value of its search inductive bias rather than a reason to alter the benchmark.

For publication results, run multiple independent top-level training seeds and report 95% confidence intervals, per-city/per-SWAP-seed rankings, discovery, coverage, redundancy, energy, safety, and ablations—not only the aggregate operational score.

## License and data

Repository code is licensed under the project license. Real geographic snapshots are derived from OpenStreetMap and must retain **© OpenStreetMap contributors** attribution and applicable ODbL terms. Synthetic mode exists for deterministic CI/development and must not be presented as real-world evidence.
