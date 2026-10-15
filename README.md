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

### Train, test, and report everything

For a **new study without the previous candidate archive**, run from the repository
root (the same commands work in Windows Command Prompt and PowerShell):

```bash
git pull --ff-only
docker compose build ears-v9-train-all
docker compose run --rm ears-v9-train-all
```

This single run prepares maps, trains/selects all supported models, evaluates all
30 implementations on the original eight cities, and creates the comprehensive
publication evidence report. It does not require
`results/publication-v9-sparse-filter-v2/frozen/ears_safe.json` and does not modify
previous studies or `results/train/checkpoints/`.

| Stage | Work performed |
| --- | --- |
| Geography | Copy available OSM caches; download missing snapshots into the new study; reject synthetic fallback |
| Learning | Train GRPO, IPPO, MAPPO and HAPPO on Vienna/London/Berlin; select on Amsterdam/Prague |
| Coordination | Run the existing teacher-assisted coordination upgrade with validation gating |
| PRISM | Select X/Plus/Star patterns, the preferred pattern, and PRISM-Ant fusion |
| EARS | Select EARS, negative-pheromone and hierarchical MAPPO/EARS configurations |
| Frozen evaluation | Run all 26 original implementations plus four EARS-Info variants: 168,000 episodes |
| Reporting | Generate PDF/HTML/Markdown reports, LaTeX results text/tables, figures, full CSVs, and a reproducibility ZIP |

Fixed heuristics are evaluated as fixed methods; they have no learned weights.
GRPO ablations derive from the trained GRPO checkpoint. All four v9 variants share
the newly selected EARS-Safe checkpoint and fixed routing settings. The eight test
cities never select checkpoints. Matching evaluation scenarios does not establish
equal training budgets or faithful external SOTA implementations.

**Resume:** repeat only `docker compose run --rm ears-v9-train-all`, without rebuilding
or changing machines. Completed stages are verified and skipped. An interrupted
training stage is preserved under `attempts/` and restarted from the preceding
completed stage; optimizer-step resumption is not supported. Testing resumes from
the last committed episode. Keep the same image throughout the study. Run without
other intensive experiments for comparable runtime measurements.

Outputs are under `results/eight-city-publication/`:

- `evaluation/report/publication_report.pdf`: comprehensive measured-results report.
- `evaluation/report/index.html`: all-method tables and figures.
- `evaluation/report/results_section.tex`: measured results text for manuscript integration.
- `evaluation/report/setting_means.csv`, `setting_city_means.csv`, `between_city_sd.csv`: detailed comparisons.
- `training_manifest.json` and each training-stage directory: checkpoints, budgets, logs and selection histories.
- `../eight-city-publication.zip`: complete training and evaluation evidence package.

The report preserves negative findings and limitations. It does not automatically
certify an IEEE manuscript as submission-ready. Integrate the actual outcomes into
the original paper and review its methods, references, and claims before submission.

To exercise the entire workflow with short **synthetic software checks**:

```bash
docker compose run --rm ears-v9-train-all python tools/train_all_eight_city.py --smoke --profile core --output-root results/eight-city-pipeline-smoke
```

Use `--offline` to require existing cached maps, `--stop-after ears` to stop after
training, or `--max-evaluation-jobs N` to pause testing. Use a fresh output root for
a different profile or training study. The default historical evaluator below
still requires its exact old checkpoint; a new training study is labelled and
hash-locked separately rather than treated as a replication of that checkpoint.

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

### Eight-city comparison: all 30 implementations

The new evaluator reruns the **26 original implementations and four EARS-Info
variants** together in San Francisco, Paris, Barcelona, Rome, New York
(Manhattan), Chicago, Tokyo, and Melbourne. It preserves the core controllers
and simulator. Historical measurements are not pooled into this comparison.

| Panel | Settings | Episodes in the full profile |
| --- | --- | ---: |
| Mission-stop | 8 cities × 30 methods × 4 team sizes × 4 dropout levels × 20 episodes | 76,800 |
| Fixed-budget | Same matrix; mission-completion stopping disabled | 76,800 |
| SWAP | 8 cities × 30 methods × 3 target views × 20 episodes; 8 robots, dropout 0.03 | 14,400 |
| **Total** | **All methods at every setting within each panel** | **168,000** |

Team sizes are 4/8/16/32 and dropout probabilities are 0/0.03/0.15/0.30.
Each setting has 10 north-east and 10 south-west starts. The `core` profile is
a separate 24,000-episode study at 8 robots and dropout 0.03.

**Required inputs:** restore all 12 original selected checkpoint files to
`results/train/checkpoints/`, the eight original OSM snapshots to `data/cache/`,
and the previously evaluated candidate to
`results/publication-v9-sparse-filter-v2/frozen/ears_safe.json`.
The preflight lists missing files and rejects synthetic checkpoints/maps and a
changed candidate. No automatic training or fallback occurs. Checkpoints record
provenance but their training history still needs to be documented by the author.
Use `--allow-download` only if deliberately starting a study with new map snapshots.

Build once, check the inputs, then start the full study:

```bash
docker compose build ears-v9-eight-city
docker compose run --rm ears-v9-eight-city python tools/eight_city_comparison.py preflight --profile full --output-root results/eight-city-v9-full
docker compose run --rm ears-v9-eight-city
```

Repeat the last command to resume. **Retain the same image, code, machine, and
output directory; do not rebuild during a study.** Evaluation runs serially and
prints progress and a running time estimate. A journal commits every episode;
partial studies cannot produce final rankings. The new directory is independent
of existing publication and mechanism runs. For meaningful runtime comparisons,
run the full study after other CPU-intensive experiments finish.

To verify the software without the original inputs:

```bash
docker compose run --rm ears-v9-eight-city python tools/eight_city_comparison.py smoke --profile core --output-root results/eight-city-smoke
```

This uses untrained policies, one synthetic map, and 180 short episodes. Its
outputs are labelled **software checks, not paper results**.

The full study freezes source files, maps, checkpoints, package versions, thread
settings, and hardware metadata. Every method receives the same observed-state
interface, paired starts, and communication draws indexed by step and robot.
A fixture audit checks that the interface preserves native actions and policy
memory; it is not a formal proof of information isolation. Native safety and
return logic remain different, with RandomAgent and GreedyAgent identified as
unfiltered controls. QMIX, MADDPG, and MAT are explicitly labelled lightweight
policy-style adaptations, not full reproductions of the published algorithms.

Analysis uses equal city weights, a declared primary EARS-Info-v9 versus EARS-Safe
score contrast at the original setting, exact city-level sign-flip tests, and
Holm adjustment across secondary contrasts. City bootstrap intervals are
descriptive. Reused geography makes this exploratory evidence. Unchanged or
duplicate SWAP views remain in the all-city descriptive panel; a separate
changed-view analysis reports which cities qualify. Targets are never added or
projected to improve results.

On completion, `results/eight-city-v9-full.zip` contains the frozen inputs, raw
journal, metrics, integrity hashes, and `report/` with all-method tables, LaTeX,
PDF/PNG figures, and an HTML report. The report does not certify publication
readiness or automatically declare the candidate a winner.

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
| [`tools/`](tools/) | Evidence audit, resumable mechanism and eight-city comparisons |
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
