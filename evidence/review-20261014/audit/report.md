# EARS v9 publication evidence audit

Recommendation: revise evidence and align the manuscript before submission. Null results can be publishable; a significant superiority result is not a universal IEEE requirement. This audit assesses the supplied evidence, not acceptance or originality.

Verified 11,200 journal rows against the frozen job matrix, payload hashes, raw CSV, diagnostic export, score formula and artifact hashes. Original evidence was opened read-only.

Primary delta J = 0.01447037; exact two-sided city sign-flip p = 0.2500. The 4-city test cannot attain p < 0.1250, regardless of episode count. The positive percentile bootstrap interval does not overturn this small-city limitation. All added sensitivity analyses are post hoc and descriptive.

## Geographic concentration and target projection

| city | targets | moved_targets | protocol_amended | delta_J |
| --- | --- | --- | --- | --- |
| Zurich | 12 | 0 | True | -0.001495 |
| Madrid | 585 | 0 | True | 0.006813 |
| Lisbon | 550 | 7 | True | 0.040115 |
| Helsinki | 441 | 0 | True | 0.012448 |

| analysis | delta | p_two_sided | city_bootstrap_low | city_bootstrap_high | cities | cities_positive |
| --- | --- | --- | --- | --- | --- | --- |
| all cities | 0.014470 | 0.250000 | 0.001991 | 0.031790 | 4 | 3 |
| no targets physically moved (post hoc) | 0.005922 | 0.500000 | -0.001495 | 0.012448 | 3 | 2 |

Target projection sensitivity removes entire cities with moved targets; it does NOT isolate the causal effect of moving targets and is not a new confirmatory test.

## Settings below the original -0.005 robustness threshold

| city | agents | dropout | delta_J |
| --- | --- | --- | --- |
| Zurich | 4 | 0.000000 | -0.147417 |
| Zurich | 4 | 0.030000 | -0.129997 |
| Zurich | 4 | 0.150000 | -0.043768 |
| Zurich | 4 | 0.300000 | -0.037893 |
| Madrid | 4 | 0.300000 | -0.011119 |
| Madrid | 16 | 0.150000 | -0.006266 |

## Leave-one-city-out sensitivity

| omitted_city | remaining_cities | delta_J |
| --- | --- | --- |
| Helsinki | 3 | 0.015144 |
| Lisbon | 3 | 0.005922 |
| Madrid | 3 | 0.017023 |
| Zurich | 3 | 0.019792 |

## Runtime on the original recorded platform

| strategy | runtime_seconds | runtime_overhead | steps |
| --- | --- | --- | --- |
| AntSwarmSafe | 1.868094 | 0.016581 | 124.785156 |
| EARS-Info-Local-v9 | 0.263454 | 0.001987 | 145.357031 |
| EARS-Info-v9 | 0.565693 | 0.006596 | 113.853906 |
| EARS-Safe-v7 | 6.705173 | 0.079149 | 120.112500 |

runtime_overhead is seconds per simulated environment step, not per robot decision. Episode runtime is affected by stopping time; compare both measures on the same platform.

## Provenance and claim boundaries

Files differing from frozen hashes: ['docker/Dockerfile']. Additional hashed-source files: none. Hash consistency supports artifact integrity, not historical independence or a fresh replay in the original software environment. Archive the evaluated Dockerfile/image and resolved dependency versions; never edit the original frozen manifest to make checks pass.

The attached manuscript describes v7, while these data evaluate v9 information gain, graph routing, persistent routes and sensing reservations. Rewrite the algorithm, equations and contributions before using v9 numbers. Keep the older eight-city comparison separate. The score implementation uses collisions plus restricted entries in its safety component; battery failures are exported separately, contrary to the manuscript's broader definition. This did not affect this run's J because all those counts were zero.

The previous validation selection retained v7. Four-city exploratory findings cannot be relabeled as independent six-city confirmation. Reused cities may support mechanism diagnostics; a later confirmatory study needs a prospectively frozen sampling and analysis protocol. At least six cities are needed merely for p-value resolution, not guaranteed power or significance.

Zero recorded incidents is a simulation observation. Shared-map access during communication dropout does not validate distributed communication failure robustness or physical robot safety.
