# Publication review evidence

See [the readiness review](../../docs/EARS_V9_PUBLICATION_READINESS.md) for
interpretation, limitations, exact commands and the distinction between the
attached v7 manuscript, original v9 study and this new pilot.

- `review_manifest.json`: input archive hashes, base code revision and verification scope.
- `audit/`: independently recomputed diagnostics of the supplied 11,200 original
  episodes. These files do not replace or modify the original study. Reproduction
  requires the supplied `publication-v9-sparse-filter-v2.zip`, which is not duplicated here.
- `pilot/`: 256 newly executed, paired simulation rollouts on the four supplied
  OSM-derived maps. This is exploratory reused-geography evidence. The complete
  raw journal, CSV, frozen checkpoint/maps, new source/environment hashes, summaries
  and descriptive comparisons are included.

All map data: © OpenStreetMap contributors; [OpenStreetMap copyright and license](https://www.openstreetmap.org/copyright).
Per-map original metadata and projection audits are retained in `pilot/frozen/plan.json`
and the frozen map files. Code uses the repository MIT license; OSM data use their
separate data license.

Runtime measurements in the pilot belong to its recorded platform, not the original
study's platform. They cannot be pooled to claim a new speedup. No physical robot
experiment, independent confirmatory study, or full 10,240-episode follow-up was run.

The original source manifest hashes `src/`, `experiments/`, dependencies and Dockerfile.
The new tools do not change that source set; their own hashes are recorded in the
pilot plan. Tests, documentation and Compose additions do not change either controller.
The original Dockerfile hash mismatch remains explicitly reported.
