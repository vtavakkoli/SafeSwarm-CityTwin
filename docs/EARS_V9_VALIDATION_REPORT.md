# EARS v9 - Validation evidence and publication protocol


## Evidence status

This report audits completed validation evidence and specifies the final evaluation workflow. Independent six-city confirmation has not been supplied or executed for this report. It is not evidence that the candidate is ready for an unconditional superiority claim.

The original selection retained EARS-Safe-v7. EARS-Info-v9 is frozen separately as a research candidate because of its average validation gains. This post-validation choice does not reverse the failed promotion decision. All four v9 variants failed at least one declared gate.


## Completed validation results

The supplied run contains 768 matched episodes: Amsterdam and Prague, four start zones, 4/8/16/32 agents, dropout probabilities 0/0.03/0.15/0.30, and six methods. Each method has 128 episodes. These cities were used for selection, so these are descriptive results.

| Method | J | Discovery | Coverage | Energy | Time (s) |
| --- | --- | --- | --- | --- | --- |
| AntSwarmSafe | 0.795545 | 0.908257 | 0.585287 | 673.969531 | 1.599126 |
| EARS-Balanced-Local-v9 | 0.745275 | 0.832818 | 0.549613 | 795.367187 | 0.200841 |
| EARS-Balanced-v9 | 0.795920 | 0.916470 | 0.562039 | 659.860547 | 0.702024 |
| EARS-Info-Local-v9 | 0.744722 | 0.832818 | 0.544086 | 781.682422 | 0.196582 |
| EARS-Info-v9 | 0.816120 | 0.957449 | 0.542815 | 581.267578 | 0.455820 |
| EARS-Safe-v7 | 0.799021 | 0.909288 | 0.591576 | 662.408203 | 5.546917 |

Relative to v7, Info-v9 increased J by 0.017099 and weighted discovery by 4.816 percentage points. Energy decreased by 12.25% and runtime by 91.78% on the reported machine. Coverage decreased by 4.876 percentage points.

The worst city/setting J change was -0.012176, below the -0.005 regression limit. Zero physical incidents and battery failures were recorded in this sample; this does not establish a universal safety guarantee.


## Interpretation and paper claim

A defensible current statement is: In two validation cities, information-aware routing improved mean discovery, operational score, energy consumption and measured runtime, while reducing coverage at termination and failing the prespecified robustness gate. Independent confirmation remains pending.

At larger team sizes, both methods can finish all targets. Earlier termination gives the faster method less time to accumulate area coverage. The fixed-budget panel tests this explanation while retaining the original mission-stop score as the primary endpoint.

The controller is a calibrated, event-driven policy, not a newly trained neural network. The simulation uses shared maps and a communication dropout abstraction. Do not describe it as a validated physical deployment or a fully decentralized communication protocol.


## Frozen final evaluation

The default complete study uses Zurich, Madrid, Lisbon, Stockholm, Helsinki and Copenhagen, with 20 paired episodes per setting, alternating north-east and south-west starts. These geographies are independent only if they have not been inspected for tuning in another run.

Mission-stop: 6 cities x 16 agent/dropout settings x 20 episodes x 4 methods = 7,680 rows. Fixed-budget: the same 7,680 rows with only mission-completion stopping disabled. SWAP: 6 cities x 3 target-subset seeds x 20 episodes x 4 methods at 8 agents and 0.03 dropout = 1,440 rows. Total: 16,800 episodes. Four methods are AntSwarmSafe, EARS-Safe-v7, EARS-Info-v9, and its no-relocation ablation.

Primary endpoint: the equal-city mean Info-v9 minus v7 operational-score difference under mission-stop, tested two-sided by exact city sign flipping at alpha 0.05. Secondary panel/metric/comparator tests are corrected together with Holm's method. Six cities provide limited inferential resolution; city bootstrap intervals are approximate and the exchangeability assumption must be considered.

Completion is reported as full-target success probability and first completion step capped at the horizon for failures. Fixed-budget still allows all-agent termination and retains all safety rules. Original, discovery-emphasis and coverage-emphasis weights are reported separately; the primary weights never change.


## Reproduction and provenance

Use docker compose run --rm --build ears-v9-publication to freeze the existing checkpoint, run or resume all evaluation panels, and create the final report. Output is under results/publication-v9/. Use ears-v9-train only for an explicitly new controller calibration; this creates a different model lineage. The detailed commands and recovery procedure are in docs/EARS_V9_PUBLICATION.md.

Frozen artifacts include the exact controller, normalized source hashes, immutable map snapshots, evaluation settings and historical selection. SQLite commits each completed episode. Final deliverables include raw episodes, diagnostics, city summaries, paired tests, score sensitivity, figures, LaTeX tables, and reports in PDF, HTML and Markdown.

Historical checkpoint SHA-256: 3023a2cb283206b9dfd0c0679bdfbb29aaad9e633b599c2470cd4b196bf326bf

Historical episode SHA-256: e8ec5e4bb4ff3767118818969957014413cf9e1230a032e0b709665cbb38bec8

Historical maps were labeled OpenStreetMap but lack explicit target provenance. The final pipeline requires verified OpenStreetMap targets, refreshes legacy snapshots online and rejects fallback data. The historic values above remain unchanged.


## Before submission

Run the entire frozen study, retain null results and regressions, and report every prespecified panel. If the reserved cities were previously inspected, designate a genuinely untouched protocol before running. Use the generated report to update the manuscript's abstract, tables and limitations. Do not merge validation and test outcomes into one claimed held-out performance number.

Archive the checkpoint, study manifest, map snapshots, raw journal/CSV and exact code revision. Check the target venue's formatting and reporting requirements, OSM attribution and data distribution obligations, and include a literature/novelty assessment. This technical report is not a completed venue-specific manuscript or an acceptance prediction.
