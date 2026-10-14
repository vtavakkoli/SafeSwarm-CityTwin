# v9 exploratory mechanism probe

Reused geography; no promotion or confirmation claim.

All variants retain the same safety monitor, return logic, routing score, checkpoint and paired scenario seeds. Local disables relocation. Periodic16 replaces the event request schedule. NoSensingReservations removes local and relocation sensing-footprint overlap suppression; collision avoidance reservations stay active. Safety, score, routing and stopping rules remain unchanged.

| panel | strategy | operational_score | weighted_target_discovery | runtime_overhead | accepted_relocations | relocation_steps |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_budget | EARS-Info-Local-v9 | 0.558257 | 0.408502 | 0.000485 | 0.000000 | 0.000000 |
| fixed_budget | EARS-Info-NoSensingReservations | 0.720148 | 0.777618 | 0.001398 | 25.375000 | 164.218750 |
| fixed_budget | EARS-Info-Periodic16 | 0.746962 | 0.857956 | 0.001456 | 24.187500 | 151.312500 |
| fixed_budget | EARS-Info-v9 | 0.719829 | 0.772908 | 0.001321 | 23.343750 | 167.468750 |
| mission_stop | EARS-Info-Local-v9 | 0.558257 | 0.408502 | 0.000455 | 0.000000 | 0.000000 |
| mission_stop | EARS-Info-NoSensingReservations | 0.721468 | 0.777618 | 0.001188 | 24.250000 | 157.250000 |
| mission_stop | EARS-Info-Periodic16 | 0.749909 | 0.857956 | 0.001457 | 22.187500 | 137.406250 |
| mission_stop | EARS-Info-v9 | 0.722727 | 0.772908 | 0.001211 | 22.531250 | 160.156250 |

Periodic interventions are not budget-matched to the event-driven method. Differences test the whole request schedule, not a causal advantage at equal intervention budgets. Future work should calibrate matched budgets on development maps and freeze them before untouched-map evaluation.

Fixed-budget disables mission completion stopping only; all-agent termination remains possible. All tests and intervals are descriptive, with cities as units and Holm adjustment across all reported panel/metric/comparator contrasts. Small samples and convenience maps limit generalization.
