# Publication readiness review — 14 October 2026

**Recommendation: revise before IEEE Access submission.** The evidence is promising,
but adding its numbers to the attached v7 manuscript would misdescribe the method
and overstate geographic confirmation. This is a technical assessment, not an
acceptance prediction or a numerical quality rating.

IEEE Access asks for technically sound experiments and conclusions supported by
the data; it does not require every study to establish statistically significant
superiority. A carefully scoped paper about a reproducible design and its tradeoffs
can include null and negative findings. The present blockers are the mismatch
between method and manuscript, the strength of mechanism claims, incomplete
geographic confirmation, and the remaining provenance limitations.

Official criteria checked for this review:

- [Reviewer guidelines](https://ieeeaccess.ieee.org/reviewers/reviewer-guidelines/)
- [Article preparation](https://ieeeaccess.ieee.org/authors/preparing-your-article/)

## Evidence checked

- Repository base: `dc1a926` (full base SHA is in the accompanying review manifest).
- Attached `latex_IEEE_ACCESS_SR.zip`, including `EARS.tex`, previous analysis,
  and tables. Its main method and eight-city results concern the earlier EARS controller.
- Attached `publication-v9-sparse-filter-v2.zip`: all 11,200 raw rows, SQLite
  payloads, frozen job matrix, map/checkpoint/study hashes and diagnostic exports.
- The new audit did not rewrite the original run or alter its selection decision.

| Check | Finding | Implication |
| --- | --- | --- |
| Raw evidence | 11,200/11,200 rows, unique matched keys, score and export parity verified | Arithmetic and artifact consistency pass |
| Primary v9 versus v7 score | +0.014470; exact city sign-flip p=0.25 | Average gain is descriptive |
| Geographic replication | Zurich, Madrid, Lisbon, Helsinki | Four cities; smallest possible two-sided p=0.125 |
| Robustness | 6/64 settings below -0.005; worst -0.147417 | Original robustness requirement fails |
| Worst regression | Zurich, 4 agents, dropout 0 | Needs a mechanism explanation, not selective exclusion |
| Historical selection | v7 retained; v9 promotion failed | Do not relabel the research candidate as the selected winner |
| Geography amendment | Copenhagen absent, Stockholm excluded; 7 Lisbon targets relocated | Not the original six-city confirmation |
| Post hoc omission of Lisbon | Mean gain +0.005922; p=0.50 across three cities | Composition sensitivity, not proof of projection bias |
| Safety | Zero recorded incidents, battery failures, blocked moves | Observation under the simulator contract only |
| Source provenance | All recorded Python source hashes match; Dockerfile differs | Recover evaluated Dockerfile/image for an exact environment replay |

The original run's positive city-bootstrap interval does not override the exact
test or the limited geographic replication. Repeated seeds do not create new city
units. Six cities are merely the minimum for this two-sided test to attain p<0.05,
not a power analysis, a guarantee of significance, or an IEEE rule.

The audited mean runtime is 0.5657 s for v9 versus 6.7052 s for v7; seconds per
environment step are 0.006596 versus 0.079149 on the original recorded platform.
These are useful computational findings, but episode duration, hardware, software
and load constrain their interpretation. New pilot runtime is measured in a
different environment and must not be pooled with the original measurements.

## New repository experiment and completed pilot

`tools/mechanism_probe_v9.py` adds four matched controls with unchanged safety
monitor, checkpoint, return logic, routing score and stopping semantics:

| Variant | Change relative to full v9 |
| --- | --- |
| Full v9 | Existing information routing |
| Local v9 | Disable relocation only |
| Periodic16 | Replace degradation-triggered requests with a fixed 16-step request schedule; retain warmup, communication and cooldown eligibility |
| NoSensingReservations | Remove sensing-footprint overlap suppression in local actions and relocation allocation; keep collision reservations |

The pilot was frozen before execution and completed **256 fresh simulation
episodes**: four supplied maps, two agent counts (4 and 8), dropout 0.03,
four episodes alternating two start zones, four methods and two stopping panels.
The same scenario seeds are paired across variants. Method order rotates. This is
an exploratory diagnostic on inspected maps, not new independent geographic evidence.

| Mission-stop mean | Full v9 | Local only | Periodic16 | No sensing reservations |
| --- | ---: | ---: | ---: | ---: |
| Operational score J | 0.722727 | 0.558257 | 0.749909 | 0.721468 |
| Weighted discovery | 0.772908 | 0.408502 | 0.857956 | 0.777618 |

The pilot supports investigating relocation: disabling it substantially reduces
the observed mean. It **does not validate event triggering as the best schedule**.
Periodic16 has a higher overall observed mean, driven by Zurich: at four agents,
its J is 0.793332 versus full v9's 0.460074. Full v9 nevertheless beats Periodic16
on three of the four city means, illustrating why a single overall average is
insufficient. The city-level two-sided test for full-minus-periodic J is p=1.0;
this tiny pilot neither establishes periodic superiority nor equivalence.
The sensing-reservation contrast is small and mixed. No observed safety incidents,
battery failures or blocked moves occurred in the pilot.

Accepted relocation counts and planning time are exported. The periodic control
does not match the accepted intervention budget; it changes request timing and
may change intervention count. A future matched-budget study must calibrate its
schedule on development data before untouched-map evaluation. Do not promote
Periodic16 solely because of this exploratory result.

Read the [audit](../evidence/review-20261014/audit/report.md) and
[pilot report](../evidence/review-20261014/pilot/report.md). Raw pilot episodes,
SQLite journal, source/dependency manifest, frozen maps/checkpoint, city means and
all contrasts are retained in the same evidence folder. Bootstrap intervals and
Holm-adjusted tests are descriptive; no significance-based selection is performed.

## Run the new checks

Extract the supplied result archive so that this file exists:
`results/publication-v9-sparse-filter-v2/frozen/study.json`.
Do not put the outer ZIP folder inside a second folder of the same name.
The source study must be stopped and its SQLite journal checkpointed before audit.

```bash
docker compose run --rm --build ears-v9-audit
docker compose run --rm --build ears-v9-mechanism-probe
```

The audit outputs to `results/publication-v9-audit`; the default 256-episode pilot
outputs to `results/publication-v9-mechanism-pilot`. They do not modify the source
study. The runner commits each row, resumes completed rows without rerunning them,
and rejects changed code, software, thread settings or frozen artifacts.

To run the **10,240-episode full mechanism matrix** on these four maps, use a NEW
output root. It uses all 4/8/16/32 agent counts, all four dropout settings, 20 paired
episodes, four methods, and both panels. This remains exploratory.

PowerShell:

```powershell
$env:EARS_PROBE_PROFILE = "full"
$env:EARS_PROBE_ROOT = "results/publication-v9-mechanism-full"
docker compose run --rm --build ears-v9-mechanism-probe
```

Bash:

```bash
EARS_PROBE_PROFILE=full EARS_PROBE_ROOT=results/publication-v9-mechanism-full docker compose run --rm --build ears-v9-mechanism-probe
```

Native Python (same requirements as the repository):

```bash
python tools/publication_readiness_v9.py --study-root results/publication-v9-sparse-filter-v2 --output-root results/publication-v9-audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/mechanism_probe_v9.py --study-root results/publication-v9-sparse-filter-v2 --output-root results/publication-v9-mechanism-full --profile full
```

These tools live outside `experiments/` so their addition leaves the historical
source manifest intact. The probe hashes both new tools explicitly. The audit
reports source differences rather than forging a source match; the probe requires
matching historical Python sources and records its new execution environment.
The older publication runner retains its strict source checks and can still reject
the supplied archive's Dockerfile mismatch. Do not edit its frozen hashes to bypass it.

## What is needed before rewriting for submission

1. Choose the paper's actual contribution. For v9, describe marginal sensing gain,
   traversable-graph distance fields, battery-aware feasibility, persistent routes,
   cancellation, reservations and information-based local actions. Do not reuse
   v7 equations, trigger claims or baseline rankings as if they evaluated v9.
2. Use the full mechanism matrix to characterize the Zurich failure and periodic
   control. If tuning follows, do it only as development work. Keep every negative
   result and freeze the final candidate before another geographic evaluation.
3. For claims about geographic superiority, preregister genuinely untouched maps,
   map sampling, preprocessing and sparse-target handling before viewing outcomes.
   Determine sample size prospectively from effect size and geographic variability;
   do not add cities or change exclusions until p<0.05. Retain the original four-city
   findings as exploratory. A tradeoff paper can instead explicitly narrow its claims.
4. Include matched classical/frontier or information-gain comparators appropriate
   to the proposed novelty. The 26 older labels include lightweight MARL-style
   adaptations, so they do not support state-of-the-art MARL superiority. A complete
   current literature/novelty review remains necessary; this code audit does not certify it.
5. Restore the historical v7 held-out and ablation raw artifacts if those old claims
   remain central. The manuscript archive contains SWAP raw rows but documents
   missing primary held-out and ablation rows. New v9 evidence does not fill that gap.
6. Correct the safety-score definition: current J counts collisions and restricted
   entries in `actual_safety_incidents`; battery failures are separate. Preserve
   historical scores and state this definition, or rerun a prospectively changed
   metric as a distinct study. No numerical difference occurs in the supplied zero-failure run.
7. Limit the scope to known-grid, shared-state simulation. Communication counters
   do not model withheld map information; zero incidents are not a physical safety
   guarantee. Explain map projection, exclusions and runtime provenance.
8. Then rewrite title, abstract, algorithm, equations, results, figures, discussion
   and AI-assistance disclosure together. The attached statement that no simulations
   were generated during revision would need updating if this pilot is included.
   Confirm author details, references, artwork and final IEEE submission requirements.

The supplied LaTeX archive is left unchanged because the requested readiness
condition was not met. This review supplies the alternative requested action:
implemented, tested repository checks and new measured diagnostic evidence.
