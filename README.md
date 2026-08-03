# SafeSwarm-CityTwin

A reproducible, safety-constrained multi-agent benchmark for comparing urban exploration and monitoring algorithms on **small, cached real-city snapshots**.

The repository combines the original SafeSwarm runtime-assurance layer with portable bio-inspired and multi-agent policy implementations from **BioSwarm-Urban-Monitoring**. Every strategy is evaluated on the same city cells, seeds, agent count, battery budget, communication model, and episode length.

## What is included

- Real urban layers downloaded from OpenStreetMap through OSMnx.
- Persistent JSON caching so algorithms never receive different map snapshots in the same benchmark.
- Explicit provenance: every run is labelled `openstreetmap` or `synthetic`; fallback data is never presented as real.
- Weighted monitoring targets derived from emergency services, schools, public transport, parks, tourism, shops, and offices.
- Obstacles and restricted areas derived from buildings, water/wetlands, and industrial or railway land use.
- Runtime safety enforcement for boundaries, obstacles, restricted zones, collisions, battery return reserve, and prolonged communication loss.
- Fifteen comparable strategies:
  - Classical and SafeSwarm: `RandomAgent`, `GreedyAgent`, `SafetyFilteredGreedy`, `SafeSwarmAgent`
  - Bio-inspired: `AntSwarmSafe`, `BeeSwarmSafe`, `PSOSwarmSafe`, `UA-HBAS-Safe`
  - GRPO and established MARL: `GRPO-Safe`, `IPPO-Safe`, `MAPPO-Safe`, `QMIX-Safe`, `MADDPG-Safe`, `HAPPO-Safe`, `MAT-Safe`
- CSV rankings, experiment manifest, PNG figures, and a self-contained HTML report.
- Unit tests and GitHub Actions validation.

## GRPO and MARL implementations

`GRPO-Safe` ports the group-relative behavior selection used in BioSwarm into the CityTwin environment. It scores a group of high-level behaviors, normalizes them relative to the group mean and standard deviation, samples a behavior with a softmax policy, and then selects a city-grid action through the SafeSwarm runtime monitor.

The repository also includes executable NumPy policy implementations for IPPO, MAPPO, QMIX, MADDPG, HAPPO, and MAT. They preserve the characteristic decision structure of each method—independent actors, centralized team context, monotonic value decomposition, continuous-actor-inspired movement, heterogeneous sequential roles, and attention-based coordination—while sharing the same discrete grid interface and safety filter.

These entries are **lightweight inference/benchmark implementations**, not claims of reproducing the complete neural training stacks from the original papers. Their purpose is to provide transparent, deterministic, directly comparable execution baselines. Optional JSON policy parameters can be loaded through `model_path` for fitted or tuned runs.

## Real-city benchmark

The default benchmark evaluates central snapshots of Vienna, London, and San Francisco.

```bash
python experiments/run_city_benchmark.py \
  --agents 8 \
  --grid-size 40 \
  --episodes 10 \
  --max-steps 160
```

Run only selected cities:

```bash
python experiments/run_city_benchmark.py --cities Vienna London
```

Require real data and fail instead of falling back:

```bash
python experiments/run_city_benchmark.py --require-real-data
```

Quick deterministic development run without network access:

```bash
python experiments/run_city_benchmark.py --offline --quick
```

## Docker

```bash
docker compose up --build benchmark-real-cities
```

For tests and the small offline benchmark:

```bash
docker compose up --build test
docker compose up --build benchmark-offline
```

Generated artifacts are written to:

```text
results/real_city_benchmark/
├── report.html
├── manifest.json
├── figures/
│   ├── overall_score.png
│   ├── target_discovery.png
│   └── safety_incidents.png
└── tables/
    ├── episode_results.csv
    ├── city_ranking.csv
    └── overall_ranking.csv
```

## Fair-comparison protocol

For each city and episode, the city layer is loaded once and reused by every algorithm. The benchmark holds constant:

- cached city snapshot;
- seed;
- grid size;
- number of agents;
- mission and safety zones;
- communication dropout process;
- battery model;
- sensor radius;
- maximum steps.

Fresh policy objects are created for every episode to prevent state leakage.

## Ranking

The operational score is bounded to `[0, 1]` and weights:

| Component | Weight |
|---|---:|
| Weighted priority-target discovery | 35% |
| Traversable-area coverage | 20% |
| Actual safety | 20% |
| Energy efficiency | 10% |
| Coordination / low redundant coverage | 10% |
| Communication availability | 5% |

Runtime is reported separately because it depends on hardware. Safety-filter interventions are reported as useful diagnostics, while **actual incidents** are used in the score.

## Data provenance and licensing

Real snapshots are obtained from OpenStreetMap with OSMnx and cached as a derived grid abstraction. Reports and manifests retain the required attribution:

> © OpenStreetMap contributors

OpenStreetMap data is available under the Open Data Commons Open Database License. See `https://www.openstreetmap.org/copyright`.

The cache is intentionally excluded from Git so each user can create or refresh local snapshots and review the corresponding data obligations.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pytest -q
```

## Repository layout

```text
configs/                    Multi-city benchmark configuration
experiments/                Single-city and multi-city runners
src/agents/                 SafeSwarm, BioSwarm, GRPO and MARL policies
src/environment/            City ingestion, cache, and digital-twin simulation
src/evaluation/             Metrics and ranking
src/safety/                 Runtime safety rules and monitor
tests/                      Unit and integration-oriented tests
data/cache/                 Local OSM-derived snapshots (ignored)
results/                    Generated benchmark artifacts
```

## Research use

The benchmark identifies the strongest algorithm **under the configured cities and conditions**. It does not claim universal superiority. For publication-quality results, increase episodes, retain the generated manifest, report whether every city used real data, and run statistical analysis over the episode-level CSV.

## License

The software is released under the MIT License. OpenStreetMap-derived data remains subject to the ODbL and its attribution requirements.
