#!/usr/bin/env python3
"""
verify_dashboard.py

Run from your project root:
    python verify_dashboard.py

Automates the verification checklist from DASHBOARD_UPDATE.md as far as
that's possible without a browser. It:

  1. Runs export_dashboard_data.py and checks its own reported counts.
  2. Loads dashboard/public/data.json and checks its structure/counts.
  3. Sanity-checks the known C1 statistic (post_plateau_var p_corrected)
     against the value reported earlier in this project, as an
     informational cross-check (not a hard failure if it's changed --
     re-running the pipeline with new data is expected to change it).
  4. Statically checks dashboard/src/App.jsx for the six edits described
     in DASHBOARD_UPDATE.md -- that the fallback consts were renamed
     correctly, the live-data helpers exist, and the fetch('/data.json')
     wiring is present.
  5. (Optional, slower) Runs `npm run build` inside dashboard/ to catch
     any JSX syntax errors the static checks can't. Skip with --skip-build.

What this CANNOT do: confirm the dashboard renders correctly in an actual
browser, or that the numbers displayed visually match data.json. That
part of the checklist still needs a manual look at http://localhost:5173
(or whatever port `npm run dev` reports) after this script passes.

Exit code 0 if every automated check passes, 1 otherwise.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
DATA_JSON = DASHBOARD_DIR / "public" / "data.json"
APP_JSX = DASHBOARD_DIR / "src" / "App.jsx"
EXPORT_SCRIPT = ROOT / "export_dashboard_data.py"

# Known-good value from the C1 run this integration was built against.
# Purely informational -- flagged as INFO, not FAIL, if it's different,
# since a new pipeline run is expected to change these numbers.
EXPECTED_POST_PLATEAU_VAR_P = 0.042

PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"
SKIP = "SKIP"

results = []  # list of (status, label, detail)


def record(status, label, detail=""):
    results.append((status, label, detail))
    marker = {"PASS": "✓", "FAIL": "✗", "INFO": "i", "SKIP": "-"}.get(status, "?")
    print(f"[{marker}] {status:<4} {label}")
    if detail:
        for line in detail.splitlines():
            print(f"           {line}")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Run export_dashboard_data.py
# ---------------------------------------------------------------------------
def check_export_script():
    section("1. Running export_dashboard_data.py")
    if not EXPORT_SCRIPT.exists():
        record(FAIL, "export_dashboard_data.py found", f"Not found at {EXPORT_SCRIPT}")
        return None

    try:
        proc = subprocess.run(
            [sys.executable, str(EXPORT_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        record(FAIL, "export script ran within 120s", "Timed out.")
        return None

    if proc.returncode != 0:
        record(FAIL, "export script exited cleanly", proc.stderr[-1500:])
        return None
    record(PASS, "export script exited cleanly")

    out = proc.stdout
    counts = {}
    for label, pattern in [
        ("c1_per_task", r"C1 per-task records:\s*(\d+)"),
        ("c1_test_results", r"C1 test results:\s*(\d+)"),
        ("p1_single", r"P1 single-position patch pairs:\s*(\d+)"),
        ("p1_range", r"P1 range-position patch pairs:\s*(\d+)"),
    ]:
        m = re.search(pattern, out)
        counts[label] = int(m.group(1)) if m else None

    for label, val in counts.items():
        if val and val > 0:
            record(PASS, f"{label} > 0", f"{label} = {val}")
        else:
            record(FAIL, f"{label} > 0", f"{label} = {val!r} (check WARNING lines above)\n{out[-800:]}")

    if "WARNING" in out:
        for line in out.splitlines():
            if "WARNING" in line:
                record(INFO, "export script warning", line)

    return counts


# ---------------------------------------------------------------------------
# 2. Inspect dashboard/public/data.json
# ---------------------------------------------------------------------------
def check_data_json():
    section("2. Checking dashboard/public/data.json")
    if not DATA_JSON.exists():
        record(FAIL, "data.json exists", f"Not found at {DATA_JSON}")
        return None

    try:
        data = json.loads(DATA_JSON.read_text())
    except Exception as e:
        record(FAIL, "data.json is valid JSON", str(e))
        return None
    record(PASS, "data.json is valid JSON")

    c1 = data.get("c1", {})
    per_task = c1.get("per_task", [])
    test_results = c1.get("test_results", [])
    p1 = data.get("p1_patching", {})
    sp_rows = p1.get("single_position", {}).get("rows", [])
    rp_rows = p1.get("range_position", {}).get("rows", [])

    if per_task:
        record(PASS, "c1.per_task is non-empty", f"{len(per_task)} records")
    else:
        record(FAIL, "c1.per_task is non-empty")

    if len(test_results) == 3:
        record(PASS, "c1.test_results has 3 entries")
    else:
        record(FAIL, "c1.test_results has 3 entries", f"found {len(test_results)}")

    if sp_rows:
        record(PASS, "p1_patching.single_position.rows is non-empty", f"{len(sp_rows)} rows")
    else:
        record(FAIL, "p1_patching.single_position.rows is non-empty")

    if rp_rows:
        record(PASS, "p1_patching.range_position.rows is non-empty", f"{len(rp_rows)} rows")
    else:
        record(FAIL, "p1_patching.range_position.rows is non-empty")

    # Required fields on a per_task record -- sample the first one.
    if per_task:
        required_fields = {"task_id", "solved", "plateau_onset_fraction", "post_plateau_var"}
        missing = required_fields - set(per_task[0].keys())
        if missing:
            record(FAIL, "per_task records have expected fields", f"missing: {missing}")
        else:
            record(PASS, "per_task records have expected fields")

    return data


# ---------------------------------------------------------------------------
# 3. Sanity-check the known statistic
# ---------------------------------------------------------------------------
def check_known_statistic(data):
    section("3. Cross-checking post_plateau_var statistic (informational)")
    if not data:
        record(SKIP, "statistic cross-check", "data.json unavailable")
        return

    test_results = data.get("c1", {}).get("test_results", [])
    row = next((t for t in test_results if t.get("metric") == "post_plateau_var"), None)
    if not row:
        record(FAIL, "post_plateau_var present in test_results")
        return

    p = row.get("p_corrected")
    if p is None:
        record(FAIL, "post_plateau_var has p_corrected value")
        return

    if abs(p - EXPECTED_POST_PLATEAU_VAR_P) < 1e-6:
        record(PASS, "post_plateau_var p_corrected matches known value", f"{p}")
    else:
        record(
            INFO,
            "post_plateau_var p_corrected differs from prior known value",
            f"found {p}, expected {EXPECTED_POST_PLATEAU_VAR_P} "
            "(expected to differ if the pipeline was re-run with new data -- not a failure)",
        )


# ---------------------------------------------------------------------------
# 4. Static-check App.jsx for the six edits
# ---------------------------------------------------------------------------
def check_app_jsx():
    section("4. Checking dashboard/src/App.jsx for the live-data wiring")
    if not APP_JSX.exists():
        record(FAIL, "App.jsx exists", f"Not found at {APP_JSX}")
        return

    text = APP_JSX.read_text(errors="replace")

    required_tokens = [
        ("useEffect import", r"import\s+React.*useEffect"),
        ("useRef import", r"import\s+React.*useRef"),
        ("FALLBACK_HEADLINE_STATS defined", r"const\s+FALLBACK_HEADLINE_STATS\s*="),
        ("FALLBACK_P1_PILOTS defined", r"const\s+FALLBACK_P1_PILOTS\s*="),
        ("mapTestResultsToHeadline defined", r"function\s+mapTestResultsToHeadline"),
        ("buildPilotsFromLive defined", r"function\s+buildPilotsFromLive"),
        ("fetch('/data.json') present", r"fetch\(\s*[\"']/data\.json[\"']\s*\)"),
        ("Overview receives headlineStats/p1Pilots props", r"function\s+Overview\(\{\s*headlineStats,\s*p1Pilots\s*\}\)"),
        ("CausalPanel receives pilots prop", r"function\s+CausalPanel\(\{\s*pilots\s*\}\)"),
        ("RunTab receives initialData prop", r"function\s+RunTab\(\{\s*initialData\s*\}\)"),
        ("RunTab auto-loads initialData via useEffect", r"useEffect\(\(\)\s*=>\s*\{\s*if\s*\(initialData"),
    ]
    for label, pattern in required_tokens:
        if re.search(pattern, text):
            record(PASS, label)
        else:
            record(FAIL, label, "pattern not found -- edit may be missing or reworded")

    # No leftover un-renamed references to the old const names.
    stray_headline = re.findall(r"(?<!FALLBACK_)\bHEADLINE_STATS\b", text)
    if stray_headline:
        record(FAIL, "no stray HEADLINE_STATS references", f"found {len(stray_headline)} occurrence(s) -- these will throw ReferenceError")
    else:
        record(PASS, "no stray HEADLINE_STATS references")

    stray_pilots = re.findall(r"(?<!FALLBACK_)\bP1_PILOTS\b", text)
    if stray_pilots:
        record(FAIL, "no stray P1_PILOTS references", f"found {len(stray_pilots)} occurrence(s) -- these will throw ReferenceError")
    else:
        record(PASS, "no stray P1_PILOTS references")


# ---------------------------------------------------------------------------
# 5. Optional: npm run build
# ---------------------------------------------------------------------------
def check_npm_build(skip):
    section("5. Running `npm run build` in dashboard/ (catches JSX syntax errors)")
    if skip:
        record(SKIP, "npm run build", "skipped via --skip-build")
        return

    npm = shutil.which("npm")
    if not npm:
        record(SKIP, "npm run build", "npm not found on PATH -- run manually: cd dashboard && npm run build")
        return

    if not DASHBOARD_DIR.exists():
        record(FAIL, "dashboard/ directory exists", f"Not found at {DASHBOARD_DIR}")
        return

    try:
        proc = subprocess.run(
            [npm, "run", "build"],
            cwd=str(DASHBOARD_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            shell=(sys.platform == "win32"),
        )
    except subprocess.TimeoutExpired:
        record(FAIL, "npm run build completed within 300s", "Timed out -- try running manually.")
        return
    except FileNotFoundError as e:
        record(SKIP, "npm run build", f"Could not invoke npm: {e}")
        return

    if proc.returncode == 0:
        record(PASS, "npm run build succeeded")
    else:
        tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
        record(FAIL, "npm run build succeeded", tail)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify the dashboard live-data integration.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the npm run build step (faster).")
    args = parser.parse_args()

    print(f"Project root: {ROOT}\n")

    check_export_script()
    data = check_data_json()
    check_known_statistic(data)
    check_app_jsx()
    check_npm_build(args.skip_build)

    section("Summary")
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_info = sum(1 for s, _, _ in results if s == INFO)
    n_skip = sum(1 for s, _, _ in results if s == SKIP)
    print(f"{n_pass} passed, {n_fail} failed, {n_info} informational, {n_skip} skipped.\n")

    if n_fail:
        print("Failed checks:")
        for status, label, detail in results:
            if status == FAIL:
                print(f"  - {label}")
        print(
            "\nStill needs a manual look: start `npm run dev` inside dashboard/, "
            "open the app in a browser, check the console for errors, and confirm "
            "the Finding/Notebook/Run tabs show real numbers."
        )
        sys.exit(1)
    else:
        print("All automated checks passed.")
        print(
            "\nStill needs a manual look: start `npm run dev` inside dashboard/, "
            "open the app in a browser, and visually confirm the tabs render "
            "correctly -- this script can't check that part."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()