# SafeSwarm v4 — SPARX

## SPARX = Swarm Probability-map Allocation & Region eXploration

SPARX is a new observable-only search controller designed for the generalization failure exposed by the v3 real-city run. PPO-family policies achieved strong validation scores on Amsterdam but lost much of that performance on San Francisco and Paris. The held-out behavior showed high movement and energy use together with high redundant coverage, indicating that the missing capability was not simply a larger actor update: the swarm lacked explicit global search allocation.

SPARX therefore separates **where the swarm should search** from **how each agent should cover its assigned area**.

## Observable search-probability map

At each environment step SPARX builds a normalized map from runtime-available evidence only:

- confidence-weighted sensed priority evidence;
- unexplored uncertainty;
- exploration-frontier strength;
- pheromone history;
- visit novelty and revisit pressure;
- signed shared swarm memory;
- current agent occupancy;
- known obstacles and restricted zones.

The map is called a probability map for search allocation, but it is deliberately documented as a **normalized search-utility probability**. It is not a calibrated estimate of the hidden probability that a target occupies a cell, and it never reads hidden mission coordinates or priority labels.

Shared memory decays over time, diffuses positive evidence through traversable four-neighbor geography, and makes already resolved targets repulsive. This discourages the failure mode where many agents repeatedly return to a region that was useful earlier in the episode.

## City segmentation and agent allocation

The traversable grid is partitioned into approximately `ceil(sqrt(n_agents)) × ceil(sqrt(n_agents))` spatial regions. For eight agents this creates up to nine regions.

Each region receives a score from:

1. probability mass;
2. frontier density;
3. unexplored share;
4. revisit pressure.

Active agents are greedily assigned to different high-value regions while considering distance and battery. The assignment is refreshed periodically so the team can adapt as observations and shared memory change.

This is the main conceptual difference from local PPO action selection: SPARX explicitly solves a swarm-level **task allocation problem** before selecting individual movements.

## Three scan geometries

Each assigned region is searched around its probability anchor with three independently evaluated patterns.

### SPARX-X-Safe

Uses diagonal rays around the anchor:

```text
\   /
 \ /
  X
 / \
/   \
```

Useful when relevant cells align along diagonal or mixed street/block geometry.

### SPARX-Plus-Safe

Uses cardinal rays:

```text
  |
--+--
  |
```

Useful for rectilinear city structure and direct north/south/east/west expansion.

### SPARX-Star-Safe

Combines X and Plus into eight rays:

```text
\ | /
- ★ -
/ | \
```

It provides the densest local angular search but can be less energy-efficient, which is why it is not assumed to be the winner.

## Validation-selected SPARX-Safe

`experiments/train_sparx_patterns.py` tunes all three patterns without accessing held-out test cities.

For each pattern:

1. evaluate paired weight perturbations on training city/start-zone scenarios;
2. take the best training candidate;
3. evaluate that candidate on the disjoint validation domains;
4. save a checkpoint **only when robust validation improves**;
5. archive every validation-improving checkpoint;
6. after tuning, select X, Plus or Star by validation robust score only.

The resulting checkpoint is `results/train/checkpoints/sparx_safe.json` and is evaluated as `SPARX-Safe` on the held-out test.

The three pattern-specific checkpoints are retained and tested as mechanism analysis. Their held-out scores must never be used retrospectively to change the selected SPARX pattern.

## Robust PPO/GRPO checkpoint selection

The same v4 branch also improves PPO-family checkpoint selection. v3 selected the highest mean score on a single validation city and a single start zone. v4 expands validation to multiple cities and multiple unseen validation start zones and uses:

```text
robust = mean
         - 0.50 × CI95
         - 0.20 × domain_std
         - 0.10 × (mean - worst_domain)
```

A checkpoint is promoted only when this score improves. Every promotion is copied into `results/train/validation-improvements/`.

This does not guarantee better test performance, but it reduces the chance of selecting a checkpoint that is unusually good on one Amsterdam topology while fragile elsewhere.

## Protocol v3.0

| Split | Cities | Start zones |
|---|---|---|
| Train | Vienna, London, Berlin | north-west, south-east, center |
| Validation | Amsterdam, Prague | west, east, north, south |
| Test | San Francisco, Paris | north-east, south-west |

Validation cities and validation starts are disjoint from training and test. Test metrics never select weights or search patterns.

## Reproduction

```bash
# Cache every city, including the expanded validation set
docker compose up --build prepare-data

# PPO/GRPO with robust validation checkpoint gating
docker compose up --build train

# Tune X / + / Star and create validation-selected SPARX-Safe
docker compose up --build train-sparx

# Frozen held-out evaluation
docker compose up --build test
```

Or run the whole chain:

```bash
docker compose up --build pipeline
```

## Scientific claim boundary

SPARX is designed to attack the strongest remaining v3 failure mode: high redundant coverage and weak spatial generalization. The repository must not claim SPARX is the winner until the held-out real-city run demonstrates it. A paper should report:

- SPARX-Safe selected on validation;
- SPARX-X-Safe / Plus / Star as mechanism comparisons;
- AntSwarmSafe and UA-HBAS-Safe;
- PPO/GRPO-family baselines;
- per-city and per-start-zone results;
- 95% confidence intervals;
- redundancy, discovery, coverage, energy, distance and safety;
- multiple top-level training/tuning seeds for publication claims.
