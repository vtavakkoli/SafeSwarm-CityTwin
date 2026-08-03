# City data cache

`data/cache/` stores compact grid snapshots created from OpenStreetMap through OSMnx.

The JSON snapshots are ignored by Git because they are generated data and remain subject to the OpenStreetMap ODbL. Each benchmark manifest records source, place, radius, feature count, retrieval time, cache path, and attribution.

Use `--force-refresh` to replace a cached snapshot or `--offline` to disable downloads and use the labelled deterministic fallback.
