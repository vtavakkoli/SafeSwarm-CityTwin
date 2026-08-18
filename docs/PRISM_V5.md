# SafeSwarm v5 — PRISM, PRISM-Ant and SWAP

## PRISM

**PRISM = Probability-guided Region-Integrated Search with Memory.**

PRISM is the canonical v5 name for the probability-memory search controller. It maintains a signed observable swarm-memory field, converts runtime evidence into a normalized search-utility distribution, segments traversable geography, assigns different agents to high-value regions, and executes X, Plus, or Star local search patterns. Hidden mission coordinates and ground-truth labels are never policy inputs.

The final pattern and all PRISM weights are selected on the disjoint validation domains only.

## PRISM-Ant

`PRISM-Ant-Safe` combines two empirically complementary mechanisms:

- PRISM provides global probability-mass allocation and structured region goals;
- AntSwarm provides local novelty, pheromone following, strong revisit suppression and anti-clustering.

An adaptive fusion uses more Ant-style local search near sensed evidence and more PRISM guidance while the map remains highly unexplored. Fusion presets are compared only on validation, after the PRISM pattern has been selected.

## PPO-family v5 coordination upgrade

The v4 real-city result showed that the learned policies still revisited too much of the map. v5 therefore adds an observable global goal coordinator to the PPO-family actor feature path and deterministic checkpoint evaluation. All trainable policies can also distill safe AntSwarm/UA-HBAS actions from training cities. A candidate replaces the existing checkpoint only when the robust multi-domain validation score improves.

This keeps the comparison falsifiable: the held-out test cannot rescue a weaker candidate.

## SWAP

**SWAP = Seeded World Alternate Protocol.**

A different episode RNG seed alone does not alter a cached real OSM target map. SWAP explicitly creates deterministic, priority-stratified alternate hidden mission subsets from the same physical OSM city snapshot. Obstacles, restricted geography, bases and provenance remain fixed.

SWAP views are generated only after all model, pattern and hybrid selection is frozen. They are a post-selection robustness test, not another validation set.

## Selection boundary

```text
training cities
    ↓ fit weights / teacher distillation
validation cities + validation-only starts
    ↓ accept checkpoints / select PRISM pattern / select PRISM-Ant fusion
freeze everything
    ↓
primary held-out test
    ↓
SWAP alternate-seed held-out stress test
```

No primary-test or SWAP metric may change a model checkpoint, PRISM pattern, or hybrid fusion parameter.
