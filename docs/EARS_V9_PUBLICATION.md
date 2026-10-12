# EARS v9 publication workflow

## What is final, and what is still evidence to collect?

The final research candidate is **EARS-Info-v9**, using the unchanged information
routing defaults and a frozen EARS base checkpoint. It is a controller, not a new
neural network. “Training” below means matched discrete parameter calibration.
The historical validation selected **EARS-Safe-v7** because every v9 candidate
failed a promotion gate. That decision and all original rows are retained in
`evidence/validation-20261012/`. Mean validation gains do not erase regressions.

The new study evaluates the explicitly chosen research candidate, including if it
loses. It does not automatically replace v7, tune on confirmation cities, change
J, or promise publishability. No new independent real-city results are bundled.

## Recommended: continue with the existing checkpoint

Run from the repository root after checking out this version. Preserve your
existing `results/train/checkpoints/ears_safe.json`; its historical SHA-256 is
`3023a2cb283206b9dfd0c0679bdfbb29aaad9e633b599c2470cd4b196bf326bf`.

```bash
# Optional software check: clearly labeled synthetic evidence.
docker compose run --rm --build ears-v9-publication-smoke

# Freeze the existing model and maps, evaluate all panels, then generate the report.
docker compose run --rm --build ears-v9-publication
```

The default output is `results/publication-v9/` on your host. The command prints
progress for every episode. Re-run the identical command to resume; committed
rows are skipped. Keep the same checkout, container dependencies and machine for
comparable runtime measurements. Do not run another benchmark concurrently.
A process lock prevents simultaneous writers to the same output folder.

The first run needs online OSM access unless all required snapshots have explicit
OSM target provenance. Legacy caches without that field are refreshed online.
Missing/invalid real maps cause a clear failure: no synthetic fallback, city
omission, target deletion, or post-hoc protocol change is allowed. Review a data
failure before rerunning; do not change a map after observing its performance.

If any of Zurich, Madrid, Lisbon, Stockholm, Helsinki or Copenhagen have already
been inspected for tuning, they are not untouched confirmation cities. Designate
an untouched protocol in `configs/confirmation_protocol_v9.json` **before**
freezing or viewing any new outcomes, and document that amendment.

## Optional: train a new base controller from scratch

Use this only to create a **new model lineage**. It cannot reproduce the supplied
validation numbers without the original checkpoint. The calibration uses Vienna,
London and Berlin for training; Amsterdam and Prague for validation. It never
loads test geography or needs MAPPO/PPO checkpoints.

```bash
docker compose run --rm --build ears-v9-train

# Explicitly use the new checkpoint. This works in Bash, PowerShell and cmd.
docker compose run --rm --build ears-v9-publication python experiments/publication_ears_v9.py all --checkpoint results/publication-v9/training/checkpoints/ears_safe.json
```

Training evaluates four fixed EARS parameter candidates on the same 36 training
scenarios (144 episodes). The best two safety-eligible candidates are evaluated
on the same 16 validation scenarios (32 episodes). The winner is selected by
mean city score with deterministic ties; physical incidents, battery failures or
blocked moves disqualify it. All 176 rows and calibration snapshots are saved.
Training is exclusive: to repeat it, use a new output root. The v9 information
routing configuration is fixed, not tuned by this training command.

**Never retrain after inspecting confirmation results and reuse those cities as
untouched tests.** Archive the earlier run and describe any subsequent study as
exploratory unless new untouched data are available.

## Staged execution and recovery

The generic `publication` service can also run each stage explicitly:

```bash
docker compose run --rm --build publication python experiments/publication_ears_v9.py freeze
docker compose run --rm publication python experiments/publication_ears_v9.py test --offline
docker compose run --rm publication python experiments/publication_ears_v9.py report --offline
```

`test` also generates the final report after all rows finish. Evaluation uses
frozen snapshots, so it needs no network. `freeze` is idempotent only for the
same frozen study; a later `--checkpoint` argument never silently replaces a
frozen checkpoint. For a different study use `--output-root results/publication-v9-new`.

Interrupted test episodes are rerun from the beginning; completed episodes are
committed atomically in `episodes.sqlite`. Partial CSV exports are available on
a handled interruption, and the journal remains authoritative after a forced
termination. A final report is refused until the **exact** planned key set is
complete. Do not delete the journal to “fix” unfavorable results. Source,
checkpoint, protocol and map modifications invalidate a frozen study.

## Complete, prespecified evaluation

| Panel | Scope | Rows |
|---|---|---:|
| Mission-stop (primary) | 6 cities x 16 settings x 20 episodes x 4 methods | 7,680 |
| Fixed-budget | Same matrix; mission completion does not end an episode | 7,680 |
| SWAP target subsets | 6 cities x 3 target seeds x 20 episodes x 4 methods; 8 agents, dropout .03 | 1,440 |
| Total | All panels required | 16,800 |

Settings: 4/8/16/32 agents and dropout 0/.03/.15/.30; grid 40, horizon 160.
Twenty episodes alternate north-east and south-west start zones, with matched
seeds across methods. Methods: AntSwarmSafe, EARS-Safe-v7, EARS-Info-v9, and the
Info no-relocation ablation. Execution order rotates deterministically.
All-agent termination and safety behavior remain unchanged in fixed-budget;
this panel supplies a common horizon ceiling, not guaranteed equal movement.
SWAP uses seeds 2042/3042/4042 and the existing 72% stratified target sampler;
the views must be distinct. All OSM targets must be reachable from actual starts.

Primary: equal-city mean Info-v9 minus v7 **original operational score J**, in
mission-stop, with a two-sided exact city sign-flip test (alpha .05). Each city
has equal weight; episodes/configurations are not independent-city replicates.
The sign-flip inference assumes symmetric/exchangeable city differences under
the null. With six cities its two-sided resolution is 2/64; interpretation is
limited to this modest geography sample. 95% bootstrap intervals resample whole
cities and are approximate/descriptive, not multiplicity-adjusted.

All 89 non-primary panel/metric/comparator tests form one Holm family. Comparators
are v7, Ant and the no-relocation ablation. Outcomes are J, discovery, coverage,
energy, distance, runtime, full completion, capped completion steps, incidents
and battery failures. Per-city/setting regressions are also shown descriptively.
Completion failures receive the full 160-step budget, even if all agents stop
early. Successful-only completion time is not used to hide failures.

Sensitivity reports J under original weights, a +.10 discovery/-.10 coverage
shift, and the reverse. These do not replace the original primary metric.
No result is called universal superiority; an average advantage can coexist
with local regressions or safety failures, all of which must be reported.

## Results and final publication report

```text
results/publication-v9/
  frozen/study.json              immutable study specification and source hashes
  frozen/study.sha256            exact study checksum
  frozen/ears_safe.json          copied base checkpoint
  frozen/map-*.json              actual evaluation snapshots
  frozen/historical_selection.json
  episodes.sqlite               crash-resilient authoritative episode journal
  episodes.csv                  raw metrics, paired keys, completion and runtime
  diagnostics.jsonl             per-episode controller diagnostics
  manifest.json                 status and execution environment
  report/report.pdf             generated complete-study report
  report/report.md              editable report text
  report/index.html             browsable report
  report/paper_table.tex         LaTeX table (requires booktabs)
  report/summary.csv
  report/city_means.csv
  report/paired_tests.csv        raw and Holm-adjusted p-values
  report/stratum_deltas.csv
  report/score_sensitivity.csv
  report/score_deltas.pdf         vector figure
  report/score_deltas.png
  report/conclusion.json        machine-readable claim limits
```

Archive the whole study directory with the exact repository revision. Frozen
source hashes normalize CRLF to LF for Windows/Linux consistency; checkpoint
and data hashes cover exact bytes. Preserve OSM attribution and review dataset
redistribution obligations before publicly redistributing snapshots.

The report is a manuscript evidence package, not a substitute for related-work
review, venue formatting or peer review. Report the shared-map simulator,
communication abstraction, old target-provenance limitation, validation selection
failure, checkpoint lineage and every new negative result. Zero observed
incidents is not proof of physical or formal safety.
