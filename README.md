# SafeSwarm-CityTwin

Safety-constrained multi-agent exploration prototype for **smart-city digital twins**.

This repository provides a reproducible Python 3.12 research baseline for feasibility studies. It implements a 2D city-grid abstraction with optional ingestion of **real city data** from OpenStreetMap (OSM), runtime safety monitoring, and comparative agent strategies.

## Research Goal
Evaluate whether runtime safety filtering can improve mission reliability of heterogeneous multi-agent exploration in urban digital twins while preserving acceptable runtime overhead.

## Features
- 2D smart-city twin grid with:
  - obstacles
  - restricted zones
  - mission zones
  - charging/base stations
- Multi-agent state:
  - position
  - battery level
  - current task
  - communication status
  - trajectory history
- Safety constraints:
  - no restricted-zone entry
  - no inter-agent collision
  - battery reserve for return-to-base
  - communication-loss timeout
  - operational boundary compliance
- Runtime safety monitor (`RuntimeSafetyMonitor`)
- Four exploration strategies:
  - `RandomAgent` (unsafe baseline)
  - `GreedyAgent` (unsafe baseline)
  - `SafetyFilteredAgent`
  - `SafeSwarmAgent` (task allocation + safety filtering)
- Evaluation metrics + tables/plots/report generation
- Docker + docker-compose support

## Repository Layout
```
SafeSwarm-CityTwin/
├── README.md
├── LICENSE
├── requirements.txt
├── docker/
│   └── Dockerfile
├── docker-compose.yaml
├── src/
│   ├── environment/
│   │   ├── city_twin.py
│   │   └── obstacles.py
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── greedy_agent.py
│   │   ├── safety_filtered_agent.py
│   │   └── safe_swarm_agent.py
│   ├── safety/
│   │   ├── rules.py
│   │   └── runtime_monitor.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── visualization/
│       └── plots.py
├── experiments/
│   └── run_safety_experiment.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── reports/
└── tests/
```

## Setup
### Local (Python 3.12)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker
```bash
docker compose up --build
```

## Run experiment
```bash
python experiments/run_safety_experiment.py --agents 10 --grid-size 50 --episodes 100 --seed 42
```

Outputs:
- `results/tables/safety_experiment_results.csv`
- `results/figures/safety_violations_comparison.png`
- `results/figures/mission_success_comparison.png`
- `results/figures/trajectories_with_restricted_zones.png`
- `results/figures/runtime_overhead.png`
- `results/reports/safety_feasibility_report.md`

## Real city data usage (v1)
The environment attempts to load real geospatial context from OpenStreetMap (city/place configurable through CLI):
- building footprints -> obstacles
- parks/water/industrial polygons -> restricted zones
- points of interest -> mission zones
- fire/police/hospital amenities -> base/charging stations

If OSM download is unavailable, the prototype falls back to deterministic synthetic zones to preserve experiment reproducibility.

## Tests
```bash
pytest -q
```

## Notes for feasibility-paper usage
- Metrics and artifacts are produced in a publication-friendly folder structure.
- A markdown report template is generated automatically for direct iteration into   manuscript sections.
