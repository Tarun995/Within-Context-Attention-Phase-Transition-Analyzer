"""
run_p1_pilot_range.py — Phase P1 extension: post-plateau RANGE patching.

WHY THIS EXISTS: the first pilot (run_p1_pilot.py) patched only the final
token's attention-layer output and found zero shift across 10/10 pairs. But
C1's significant metric, post_plateau_var, is a VARIANCE measured across the
whole span of positions from plateau onset to the end of the sequence — not
a single point. Patching only the last token never touched most of that
span. This pilot closes that specific gap: for each pair, it finds the
recipient's own plateau onset position (same computation metrics.py uses
for post_plateau_var) and patches the attention-layer output at EVERY
position from onset to end.

This is a single, pre-registered extension decided in advance because of a
specific, named mismatch between what was measured and what was tested —
not a parameter sweep hunting for a hit. If this also shows no shift, that
is a materially stronger null than the first pilot alone, and the honest
conclusion is that post_plateau_var is very likely a correlate of solving,
not a cause of it. Do not follow this with a third pilot on a different
layer or task type without deciding why in advance, in writing, before
running it.

Results go to results/patch_manifest_range.csv — a SEPARATE manifest from
the final-token pilot's, since this is a genuinely different experiment and
mixing the two would blur what was actually tested.

USAGE:
    python -m attn_phase.run_p1_pilot_range --n-pairs 5
    python -m attn_phase.run_p1_pilot_range --force
"""

import argparse
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from attn_phase.tasks import generate_task_set
from attn_phase.patch import (
    get_plateau_onset_position,
    capture_range_outputs,
    build_post_plateau_patch_map,
    run_with_range_patch,
    next_token_matches_expected,
    score_full_generation,
    load_manifest,
    append_manifest_row as _append_row_generic,
    check_coverage,
    manifest_bool,
    now_iso,
)

TASK_TYPE = "easy_mod_arith"
MOD_ARITH_CONFIG = dict(n_examples=20)
LAYER_IDX = -1  # same layer as the first pilot and as C1's default

RANGE_MANIFEST_COLUMNS = [
    "pair_id", "task_family", "layer", "position",
    "recipient_onset_pos", "n_positions_patched",
    "direction", "seed", "pre_patch_output_correct",
    "post_patch_output_correct", "shift_observed", "timestamp",
]


def append_range_manifest_row(path: str, row: dict) -> None:
    """Same crash-safe append-and-flush pattern as patch.py's
    append_manifest_row, but with this experiment's own column set."""
    import csv
    file_exists = os.path.exists(path)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANGE_MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def build_pairs_with_onsets(model, tokenizer, device, n_pairs, base_seed=42):
    """
    Generates the pool, scores solved/failed status, AND computes each
    instance's own plateau onset position up front. Instances with no
    detectable onset (curve never reaches threshold) are excluded — there's
    no well-defined post-plateau region to patch for them.
    """
    pool_size = max(40, n_pairs * 8)
    pool = generate_task_set(TASK_TYPE, n_instances=pool_size,
                              base_seed=base_seed, **MOD_ARITH_CONFIG)

    print(f"Scoring {len(pool)} pool instances (solved status + plateau onset)...")
    solved_tasks, failed_tasks = [], []
    for t in pool:
        t["solved"] = score_full_generation(model, tokenizer, t, device)
        onset_pos, seq_len = get_plateau_onset_position(
            model, tokenizer, t["prompt"], LAYER_IDX, device)
        t["onset_pos"] = onset_pos
        t["seq_len"] = seq_len
        tag = "solved" if t["solved"] else "failed"
        if onset_pos is None:
            print(f"  {t['task_id']}: {tag}, NO PLATEAU DETECTED — excluded")
            continue
        print(f"  {t['task_id']}: {tag}, onset={onset_pos}/{seq_len}")
        (solved_tasks if t["solved"] else failed_tasks).append(t)

    print(f"Eligible pool: {len(solved_tasks)} solved, {len(failed_tasks)} failed")
    n = min(n_pairs, len(solved_tasks), len(failed_tasks))
    if n < n_pairs:
        print(f"WARNING: only {n} pairs possible per direction (requested {n_pairs}).")

    pairs = []
    for i in range(n):
        pairs.append(("solved_to_failed", solved_tasks[i], failed_tasks[i]))
        pairs.append(("failed_to_solved", failed_tasks[i], solved_tasks[i]))
    return pairs


def run_pilot(n_pairs: int, force: bool, output_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Layer: {LAYER_IDX} | Pairs per direction: {n_pairs}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    model.to(device)
    model.eval()

    manifest_path = os.path.join(output_dir, "patch_manifest_range.csv")
    manifest = {} if force else load_manifest(manifest_path)

    pairs = build_pairs_with_onsets(model, tokenizer, device, n_pairs)
    expected_pair_ids = [
        f"{direction}_{donor['task_id']}_to_{recipient['task_id']}"
        for direction, donor, recipient in pairs
    ]

    for direction, donor, recipient in pairs:
        pair_id = f"{direction}_{donor['task_id']}_to_{recipient['task_id']}"
        if not force and pair_id in manifest:
            print(f"  skip (already done): {pair_id}")
            continue

        recipient_onset = recipient["onset_pos"]
        recipient_len = recipient["seq_len"]
        donor_len = donor["seq_len"]

        # Patch region: recipient's own post-plateau span, capped at
        # whatever the donor actually has captured outputs for.
        end_pos = min(recipient_len, donor_len)
        if recipient_onset >= end_pos:
            print(f"  skip (no valid patch range): {pair_id}")
            continue

        pre_logits = run_with_range_patch(model, tokenizer, recipient["prompt"],
                                           LAYER_IDX, {}, device)
        pre_correct = next_token_matches_expected(
            pre_logits, tokenizer, recipient["expected_answer"])

        donor_outputs = capture_range_outputs(model, tokenizer, donor["prompt"],
                                               LAYER_IDX, device)
        patch_map = build_post_plateau_patch_map(
            donor_outputs, recipient_onset, end_pos)

        post_logits = run_with_range_patch(model, tokenizer, recipient["prompt"],
                                            LAYER_IDX, patch_map, device)
        post_correct = next_token_matches_expected(
            post_logits, tokenizer, recipient["expected_answer"])

        shift_observed = post_correct != pre_correct

        row = {
            "pair_id": pair_id,
            "task_family": TASK_TYPE,
            "layer": LAYER_IDX,
            "position": "post_plateau_range",
            "recipient_onset_pos": recipient_onset,
            "n_positions_patched": len(patch_map),
            "direction": direction,
            "seed": recipient.get("instance_idx"),
            "pre_patch_output_correct": pre_correct,
            "post_patch_output_correct": post_correct,
            "shift_observed": shift_observed,
            "timestamp": now_iso(),
        }
        append_range_manifest_row(manifest_path, row)
        manifest[pair_id] = {k: str(v) for k, v in row.items()}
        print(f"  {pair_id}: onset={recipient_onset} n_patched={len(patch_map)} "
              f"pre={pre_correct} post={post_correct} shift={shift_observed}")

    is_complete, missing = check_coverage(manifest, expected_pair_ids)
    if not is_complete:
        print(f"\nINCOMPLETE: {len(missing)} pairs missing outcomes: {missing}")
        print("Re-run the same command to resume.")
        return

    n_shifts = sum(
        1 for pid in expected_pair_ids
        if pid in manifest and manifest_bool(manifest[pid]["shift_observed"])
    )
    total = len(expected_pair_ids)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {n_shifts}/{total} pairs showed a shift toward/away "
          f"from correct when patching the full post-plateau span.")
    print(f"{'=' * 60}")
    print(
        "\nThis is the pre-registered extension test. Whatever this shows, "
        "the honest next step is Phase W (write-up) — not another patching "
        "configuration."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase P1 extension: post-plateau range patching")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_pilot(args.n_pairs, args.force, args.output_dir)