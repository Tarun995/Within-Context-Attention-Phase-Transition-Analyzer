#!/usr/bin/env python3
"""
export_dashboard_data.py

Run from your project root (after any `attn-phase run`):
    python export_dashboard_data.py

Reads:
  - results/phase_c1_gpt2_seed42_results.json  (per-task results + Mann-Whitney test_results)
  - results/patch_manifest.csv                  (P1 causal patch pairs, single-position)
  - results/patch_manifest_range.csv            (P1 causal patch pairs, range-position)

Writes:
  - dashboard/public/data.json

App.jsx can then fetch('/data.json') at runtime instead of using the
hardcoded HEADLINE_STATS array. This script is safe to re-run any time
your results/ folder is updated by a new run — it always overwrites
dashboard/public/data.json fully.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_PATH = ROOT / "dashboard" / "public" / "data.json"

C1_RESULTS_FILE = RESULTS_DIR / "phase_c1_gpt2_seed42_results.json"
PATCH_MANIFEST = RESULTS_DIR / "patch_manifest.csv"
PATCH_MANIFEST_RANGE = RESULTS_DIR / "patch_manifest_range.csv"


def load_c1_results():
    if not C1_RESULTS_FILE.exists():
        print(f"WARNING: {C1_RESULTS_FILE} not found — skipping C1 data.")
        return {"config": None, "per_task": [], "test_results": []}
    with open(C1_RESULTS_FILE) as f:
        data = json.load(f)
    return {
        "config": data.get("config"),
        "per_task": data.get("results", []),
        "test_results": data.get("test_results", []),
    }


def load_csv_rows(path: Path):
    if not path.exists():
        print(f"WARNING: {path} not found — skipping.")
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast obvious booleans/numbers so the frontend doesn't have
            # to parse "True"/"False" strings.
            cleaned = {}
            for k, v in row.items():
                if v == "True":
                    cleaned[k] = True
                elif v == "False":
                    cleaned[k] = False
                else:
                    try:
                        cleaned[k] = int(v)
                    except (ValueError, TypeError):
                        try:
                            cleaned[k] = float(v)
                        except (ValueError, TypeError):
                            cleaned[k] = v
            rows.append(cleaned)
    return rows


def summarize_patch_manifest(rows, label):
    """Quick aggregate: how often did patching flip the outcome as predicted?"""
    if not rows:
        return {"label": label, "n_pairs": 0}
    n = len(rows)
    n_shift = sum(1 for r in rows if r.get("shift_observed") is True)
    by_direction = {}
    for r in rows:
        d = r.get("direction", "unknown")
        by_direction.setdefault(d, {"n": 0, "n_shift": 0})
        by_direction[d]["n"] += 1
        if r.get("shift_observed") is True:
            by_direction[d]["n_shift"] += 1
    return {
        "label": label,
        "n_pairs": n,
        "n_shift_observed": n_shift,
        "shift_rate": round(n_shift / n, 4) if n else None,
        "by_direction": by_direction,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    c1 = load_c1_results()
    patch_rows = load_csv_rows(PATCH_MANIFEST)
    patch_range_rows = load_csv_rows(PATCH_MANIFEST_RANGE)

    payload = {
        "generated_from": {
            "c1_results_file": str(C1_RESULTS_FILE.relative_to(ROOT)) if C1_RESULTS_FILE.exists() else None,
            "patch_manifest_file": str(PATCH_MANIFEST.relative_to(ROOT)) if PATCH_MANIFEST.exists() else None,
            "patch_manifest_range_file": str(PATCH_MANIFEST_RANGE.relative_to(ROOT)) if PATCH_MANIFEST_RANGE.exists() else None,
        },
        "c1": {
            "config": c1["config"],
            "per_task": c1["per_task"],
            "test_results": c1["test_results"],
        },
        "p1_patching": {
            "single_position": {
                "rows": patch_rows,
                "summary": summarize_patch_manifest(patch_rows, "single_position"),
            },
            "range_position": {
                "rows": patch_range_rows,
                "summary": summarize_patch_manifest(patch_range_rows, "range_position"),
            },
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Wrote {OUT_PATH}")
    print(f"  C1 per-task records: {len(c1['per_task'])}")
    print(f"  C1 test results: {len(c1['test_results'])}")
    print(f"  P1 single-position patch pairs: {len(patch_rows)}")
    print(f"  P1 range-position patch pairs: {len(patch_range_rows)}")


if __name__ == "__main__":
    main()