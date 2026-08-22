# Top-4 real-city visualization demo

The Docker demo replays the four leading frozen SafeSwarm/EARS policies on the **same OpenStreetMap-derived city twin**, start zone, and random seed:

| Strategy | Held-out operational score | Weighted discovery | Coverage | Mean runtime (s) |
|---|---:|---:|---:|---:|
| H-MAPPO-EARS-Safe | 0.7845 | 0.9160 | 0.5417 | 5.49 |
| EARS-Safe | 0.7805 | 0.9123 | 0.5139 | 2.50 |
| EARS-NP-Safe | 0.7797 | 0.9008 | 0.5454 | 2.55 |
| AntSwarmSafe | 0.7754 | 0.8979 | 0.5056 | 1.81 |

If `results/publication/test/tables/overall_ranking.csv` exists, the dashboard reads the frozen publication values from that file. The values above are used only as a fallback for display.

## Prerequisite

The three trained EARS-family policies must already have their frozen checkpoints in `results/train/checkpoints/`:

- `h_mappo_ears_safe.json`
- `mappo_safe.json`
- `ears_safe.json`
- `ears_np_safe.json`

`AntSwarmSafe` is the fixed interpretable controller and therefore has no learned checkpoint.

The easiest way to create/cache the required publication artifacts is:

```bash
docker compose up --build publication
```

The demo refuses to silently use synthetic geography when `--require-real-data` is enabled.

## One real city

```bash
docker compose up --build demo-top4
```

The default city is San Francisco. Select another publication test city with an environment variable:

```bash
DEMO_CITIES="Tokyo" docker compose up --build demo-top4
```

Multiple cities can be requested in one run:

```bash
DEMO_CITIES="San Francisco,Barcelona,Tokyo,Rome" docker compose up --build demo-top4
```

On PowerShell:

```powershell
$env:DEMO_CITIES="San Francisco,Barcelona,Tokyo,Rome"
docker compose up --build demo-top4
```

## All eight held-out cities

```bash
docker compose up --build demo-top4-all-cities
```

This uses the v7 publication test cities: San Francisco, Paris, Barcelona, Rome, New York/Manhattan, Chicago, Tokyo, and Melbourne.

## Optional controls

The Compose service accepts:

```text
DEMO_CITIES          default: San Francisco
DEMO_START_ZONE      default: north_east
DEMO_SEED            default: 1042
DEMO_FRAME_STRIDE    default: 2
DEMO_FPS             default: 8
AGENTS               default: 8
GRID_SIZE            default: 40
MAX_STEPS            default: 160
```

The start zone must be one of the frozen publication test zones (`north_east`, `south_west`).

## Outputs

All artifacts are written below:

```text
results/demo/top4-real-city/
```

For each city there is a polished `index.html` comparison dashboard and a `demo_metrics.csv`. Each strategy gets its own folder containing:

```text
animation.gif
trajectory_map.png
visit_heatmap.png
final_snapshot.png
detection_events.csv
summary.json
```

The root `results/demo/top4-real-city/index.html` links all generated city dashboards.

### Visualization semantics

- **Animation:** robot trails, current robot positions, detected targets, and the real-map-derived blocked/restricted/base grid.
- **Trajectory map:** complete robot paths with detection order and run metrics.
- **Visit heat map:** spatial revisit intensity over the OSM-derived traversability layer.
- **Final snapshot:** final robot positions and detected/undetected mission overlay.

Hidden mission markers are **post-evaluation overlays only**. They are never exposed to the policy and therefore do not change the repository's observable-only scientific contract.

All real geographic outputs retain **© OpenStreetMap contributors** attribution.
