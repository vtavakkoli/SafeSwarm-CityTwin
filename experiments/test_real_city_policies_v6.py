"""Evaluate frozen SafeSwarm v6 policies using the v5-compatible evaluator.

The underlying episode/ranking code is intentionally reused so EARS cannot
silently change the benchmark metric. v6 extends the policy registry/manifest
and corrects the inherited HTML metadata to describe EARS explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_real_city_policies_v5 import main as _v5_main
from test_real_city_policies_v5 import parse_args


def main() -> None:
    args = parse_args()
    _v5_main()

    output = Path(args.output_root)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ears_manifest_path = Path(args.model_dir).parent / "ears_manifest.json"
    ears_manifest = (
        json.loads(ears_manifest_path.read_text(encoding="utf-8"))
        if ears_manifest_path.exists()
        else {}
    )
    manifest.update(
        {
            "training_version": "SafeSwarm PPO/GRPO/PRISM/EARS v6",
            "ears_algorithm": "EARS: Event-driven Ant Reallocation Search",
            "ears_np_algorithm": "EARS-NP: EARS + negative-pheromone exclusion halo",
            "h_mappo_ears_algorithm": "MAPPO-assisted hierarchical EARS option controller",
            "ant_reference_unchanged": bool(ears_manifest.get("ant_reference_unchanged", True)),
            "ears_selection": ears_manifest.get("selection_rule"),
            "checkpoint_selection": (
                "validation-only; held-out test and SWAP never select PPO/GRPO weights, "
                "PRISM pattern/fusion, or EARS event/negative-pheromone/hierarchical parameters"
            ),
            "evaluation_city_count": int(len(manifest.get("cities", []))),
            "evaluation_scope": (
                "post-selection held-out evaluation; when publication_protocol_v7 is used, "
                "the expanded test cities remain evaluation-only"
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # v6 intentionally reuses the v5 evaluator implementation, but the generated
    # HTML must not mislabel a v6 EARS result as v5.
    report_path = output / "report.html"
    if report_path.exists():
        html = report_path.read_text(encoding="utf-8")
        html = html.replace("SafeSwarm v5 Held-out Test", "SafeSwarm v6 Held-out Test")
        html = html.replace("SafeSwarm v5 · Held-out Real-City Test", "SafeSwarm v6 · Held-out Real-City Test")
        html = html.replace(
            "PRISM + PRISM-Ant · deterministic learned-policy inference · unseen cities/starts",
            "EARS + PRISM-Ant · frozen post-validation policies · unseen cities/starts",
        )
        html = html.replace(
            "PPO/GRPO weights, PRISM pattern/weights and PRISM-Ant fusion parameters are selected on training + validation only. This held-out test cannot change them.",
            "PPO/GRPO weights, PRISM parameters, and EARS trigger/relocation parameters are selected on training + validation only. This held-out test cannot change them.",
        )
        report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
