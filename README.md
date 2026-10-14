<div align="center">

# SafeSwarm-CityTwin

**Reproducible multi-robot exploration on urban map abstractions**

[![CI](https://github.com/vtavakkoli/SafeSwarm-CityTwin/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/SafeSwarm-CityTwin/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Quick start](#quick-start) · [Experiments](#experiments) · [Results](#results-and-reproducibility) · [Development](#development) · [Citation](#citation-and-license)

</div>

SafeSwarm-CityTwin is a research benchmark for comparing swarm controllers under
shared sensing, energy, and runtime safety constraints. It converts cached
OpenStreetMap features into reproducible grid environments and evaluates policies
with matched scenarios, frozen checkpoints, and explicit geographic protocols.

- **Urban exploration:** hidden mission targets, spatial coverage, battery use, and team coordination.
- **Controlled comparisons:** ant-inspired search, event-driven relocation, information routing, and mechanism ablations.
- **Reproducible evaluation:** immutable map snapshots, paired seeds, source hashes, and resumable episode journals.
- **Auditable results:** raw metrics, city-level comparisons, runtime diagnostics, and generated reports.

## Quick start

Install Docker with Docker Compose, then run a small synthetic verification study:

```bash
git clone https://github.com/vtavakkoli/SafeSwarm-CityTwin.git
cd SafeSwarm-CityTwin
docker compose run --rm --build ears-v9-publication-smoke
```

Outputs are written to `results/publication-v9-smoke/`. This command verifies the
software workflow using synthetic maps; it is not a real-city evaluation.

All commands below run from the repository root. Docker mounts `results/` and
`data/cache/` to retain outputs and downloaded map snapshots on the host.

## Experiments

### Controlled routing comparison

The mechanism experiment compares four variants with the same checkpoint,
scenario seeds, return logic, and runtime safety monitor:

| Variant | Role |
| --- | --- |
| **EARS-Info-v9** | Information-aware local exploration with event-driven relocation |
| **EARS-Info-Local-v9** | Matched control with relocation disabled |
| **EARS-Info-Periodic16** | Relocation requests on a fixed 16-step schedule |
| **EARS-Info-NoSensingReservations** | Sensing-overlap reservations disabled; collision reservations retained |

The input is a completed frozen study. For the supplied four-city dataset, extract
its archive so that `results/publication-v9-sparse-filter-v2/frozen/study.json`
exists. The source directory must also contain its episode journal, CSV,
diagnostics, manifest, checkpoint, and frozen maps.

**Audit the completed source study:**

```bash
docker compose run --rm --build ears-v9-audit
```

**Run the 256-episode pilot:**

```bash
docker compose run --rm --build ears-v9-mechanism-probe
```

**Run the full 10,240-episode comparison:**

```bash
docker compose run --rm --build ears-v9-mechanism-probe python tools/mechanism_probe_v9.py --study-root results/publication-v9-sparse-filter-v2 --output-root results/publication-v9-mechanism-full --profile full
```

For this four-city input, the full profile evaluates 4/8/16/32 robots, dropout
probabilities 0/0.03/0.15/0.30, and 20 paired episodes per setting. Both mission-stop
and fixed-budget panels are included. Fixed-budget disables mission-completion
stopping; all-agent termination and safety checks remain active.

Use a separate output directory for each profile. To resume an interrupted run,
repeat the identical command with the same code, dependencies, and thread settings.
Completed episodes are retained. Keep the existing image for a resumed study;
source or dependency changes are rejected. Do not run multiple writers against
the same study directory or edit its frozen artifacts.

The default audit and pilot output directories are `results/publication-v9-audit/`
and `results/publication-v9-mechanism-pilot/`. The full command above writes to
`results/publication-v9-mechanism-full/`. All three preserve the input study.

### Additional workflows

| Workflow | Command | Prerequisite |
| --- | --- | --- |
| Calibrate a base EARS controller | `docker compose run --rm --build ears-v9-train` | Real training and validation maps; new training output directory |
| Freeze and evaluate the v9 study | `docker compose run --rm --build ears-v9-publication` | Base checkpoint at `results/train/checkpoints/ears_safe.json` and verified OSM maps |
| Regenerate a completed study report | `docker compose run --rm --build ears-v9-report` | Complete study at `results/publication-v9/` with matching source hashes |
| Render the four-method city demo | `docker compose run --rm --build demo-top4` | Trained checkpoints and real-city maps |

The base calibration selects discrete controller parameters; it does not train a
new neural network. Its output checkpoint is
`results/publication-v9/training/checkpoints/ears_safe.json`. To evaluate that new
checkpoint, use a new study directory:

```bash
docker compose run --rm --build ears-v9-publication python experiments/publication_ears_v9.py all --checkpoint results/publication-v9/training/checkpoints/ears_safe.json --output-root results/publication-v9-new
```

Real-city workflows require verified OSM geography and mission provenance. First-time
map preparation may need network access. Frozen evaluations reuse their saved maps.
Use each script's `--help` for configuration options and earlier experiment workflows.

## Results and reproducibility

| Artifact | Purpose |
| --- | --- |
| `frozen/` | Study or probe specification, checkpoint, map snapshots, and hashes |
| `episodes.sqlite` | Committed episode journal used for recovery |
| `episodes.csv` | Raw paired outcomes, completion metrics, and timings |
| `summary.csv`, `city_means.csv` | Aggregate and city-level results |
| `contrasts.csv` | Mechanism comparisons and descriptive uncertainty |
| `completion.json` | Mechanism-run completion status and output hashes |
| `report.md` | Generated mechanism report |

The publication workflow additionally writes statistical tables, figures, LaTeX,
and PDF/HTML reports under its `report/` directory. Preserve the complete output
folder with its exact code revision. Source hashes normalize CRLF/LF where the
original workflow specifies; the mechanism tools also record their own file hashes.

Archived measurements are available in [`evidence/`](evidence/):
[`validation-20261012/`](evidence/validation-20261012/) contains the historical
selection record; [`review-20261014/`](evidence/review-20261014/) contains audit tables
and the 256-episode mechanism pilot, including its raw journal and frozen inputs.

**Research scope.** EARS-Info-v9 remains a research candidate; historical validation
retained EARS-Safe-v7. The archived four-city comparisons are exploratory. Reused
maps do not provide independent geographic confirmation, and periodic interventions
are not matched to the event-driven method's accepted intervention budget. The
simulator uses known grid geometry and shared state; communication dropout counters
and zero observed incidents do not establish distributed or physical safety.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Controllers, city environment, safety monitor, training utilities, and metrics |
| [`experiments/`](experiments/) | Calibration, evaluation, visualization, and report entry points |
| [`tools/`](tools/) | Read-only evidence audit and resumable mechanism experiment |
| [`configs/`](configs/) | City definitions and geographic protocols |
| [`tests/`](tests/) | Unit and experiment-integrity tests |
| [`evidence/`](evidence/) | Archived raw measurements and provenance |
| [`docker/`](docker/) | Container build definition |
| `data/cache/`, `results/` | Local map caches and generated experiment outputs |

## Development

Python 3.12 or later is required for a native environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in PowerShell, then install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Alternatively, run the test suite in Docker:

```bash
docker compose run --rm --build unit-test
```

## Citation and license

Use GitHub's **Cite this repository** action or [`CITATION.cff`](CITATION.cff) when
referencing this software. Include the commit and frozen study identifier for
reported experiments.

Code is released under the [MIT License](LICENSE). Map snapshots are derived from
**© OpenStreetMap contributors** and retain their separate
[Open Database License and attribution requirements](https://www.openstreetmap.org/copyright).
Per-map provenance is recorded with the frozen data.
