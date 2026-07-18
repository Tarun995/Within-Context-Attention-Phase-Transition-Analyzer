"""
run_p1_pilot.py — Phase P1 causal patching pilot.

Scope (per roadmap, deliberately kept small):
    - One task family: mod_arith (same config Phase C1 already used, so
      results are directly comparable — not a new task design)
    - One layer (default: last layer, override with --layer)
    - ~5 patch pairs per direction (solved->failed, failed->solved),
      not 30 — this pilot is about proving the method works before
      scaling, per the roadmap's explicit guardrail

Every pair's result is written to results/patch_manifest.csv immediately
after that pair completes — a killed process loses at most one pair.
Re-running skips already-completed pairs unless --force is passed.

BEFORE RUNNING THIS FOR REAL: run `pytest tests/test_patch.py -v` first
and confirm both tests pass. If they don't, the patch hook isn't actually
intervening in the forward pass for your transformers version, and every
result from this script would be meaningless.

USAGE:
    python -m attn_phase.run_p1_pilot                  # default: 5 pairs, last layer
    python -m attn_phase.run_p1_pilot --n-pairs 5 --layer -1
    python -m attn_phase.run_p1_pilot --force           # redo everything

WHAT COUNTS AS THE PRE-REGISTERED TEST (decided here, not iterated on):
    task family = mod_arith, layer = --layer argument (default -1, i.e.
    the same last-layer default C1 used), position = final prompt token.
    Do not re-run this script with a sweep of layers looking for a
    significant one — that is exactly the layer-10 false positive pattern
    this project has already caught and corrected for once. If the
    default layer shows no effect, that is the answer for this pilot.
"""

import argparse
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from attn_phase.tasks import generate_task_set
from attn_phase.patch import (
    capture_final_position_output,
    run_with_patch,
    next_token_matches_expected,
    score_full_generation,
    load_manifest,
    append_manifest_row,
    check_coverage,
    manifest_bool,
    now_iso,
)

# NOTE: "mod_arith" (modulus=97, n_digits=2) is the HARD config Phase C1
# uses as its failed task type — GPT-2 small essentially never solves it,
# so it can't supply solved instances for patching.
# "easy_mod_arith" (modulus=10, n_digits=1) is the config that actually
# produces a solved/failed MIX — this is the exact task type FINDINGS.md's
# caveat names as "mod_arith_m10_d1" (seed instances 0 and 4 solved,
# 1-3 failed in the C1 run). That's the pool this pilot needs.
TASK_TYPE = "easy_mod_arith"
MOD_ARITH_CONFIG = dict(n_examples=20)


def build_pairs(model, tokenizer, device, n_pairs, base_seed=42):
    """
    Generates a pool of mod_arith instances, scores each with full-generation
    correctness, and pairs up solved <-> failed instances in both directions.
    Pool size is oversized relative to n_pairs since we don't know the
    solved/failed split in advance (C1 found this task type mixed within
    itself across seeds — see FINDINGS.md's mod_arith_m10_d1 caveat).
    """
    pool_size = max(40, n_pairs * 8)
    pool = generate_task_set(TASK_TYPE, n_instances=pool_size,
                              base_seed=base_seed, **MOD_ARITH_CONFIG)

    print(f"Scoring {len(pool)} pool instances for solved/failed status...")
    solved_tasks, failed_tasks = [], []
    for t in pool:
        t["solved"] = score_full_generation(model, tokenizer, t, device)
        (solved_tasks if t["solved"] else failed_tasks).append(t)
        print(f"  {t['task_id']}: solved={t['solved']}")

    print(f"Pool result: {len(solved_tasks)} solved, {len(failed_tasks)} failed")
    if len(solved_tasks) < n_pairs or len(failed_tasks) < n_pairs:
        print(
            f"WARNING: requested {n_pairs} pairs per direction but only "
            f"have {len(solved_tasks)} solved / {len(failed_tasks)} failed "
            f"instances. Proceeding with min({n_pairs}, available)."
        )

    n = min(n_pairs, len(solved_tasks), len(failed_tasks))
    pairs = []
    for i in range(n):
        pairs.append(("solved_to_failed", solved_tasks[i], failed_tasks[i]))
        pairs.append(("failed_to_solved", failed_tasks[i], solved_tasks[i]))
    return pairs


def run_pilot(n_pairs: int, layer_idx: int, force: bool, output_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Layer: {layer_idx} | Pairs per direction: {n_pairs}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    manifest_path = os.path.join(output_dir, "patch_manifest.csv")
    manifest = {} if force else load_manifest(manifest_path)

    pairs = build_pairs(model, tokenizer, device, n_pairs)
    expected_pair_ids = [
        f"{direction}_{donor['task_id']}_to_{recipient['task_id']}"
        for direction, donor, recipient in pairs
    ]

    for direction, donor, recipient in pairs:
        pair_id = f"{direction}_{donor['task_id']}_to_{recipient['task_id']}"
        if not force and pair_id in manifest:
            print(f"  skip (already done): {pair_id}")
            continue

        pre_logits = run_with_patch(model, tokenizer, recipient["prompt"],
                                     layer_idx, None, device)
        pre_correct = next_token_matches_expected(
            pre_logits, tokenizer, recipient["expected_answer"])

        donor_vec = capture_final_position_output(
            model, tokenizer, donor["prompt"], layer_idx, device)

        post_logits = run_with_patch(model, tokenizer, recipient["prompt"],
                                      layer_idx, donor_vec, device)
        post_correct = next_token_matches_expected(
            post_logits, tokenizer, recipient["expected_answer"])

        shift_observed = post_correct != pre_correct

        row = {
            "pair_id": pair_id,
            "task_family": TASK_TYPE,
            "layer": layer_idx,
            "position": "final",
            "direction": direction,
            "seed": recipient.get("instance_idx"),
            "pre_patch_output_correct": pre_correct,
            "post_patch_output_correct": post_correct,
            "shift_observed": shift_observed,
            "timestamp": now_iso(),
        }
        append_manifest_row(manifest_path, row)
        manifest[pair_id] = {k: str(v) for k, v in row.items()}
        print(f"  {pair_id}: pre={pre_correct} post={post_correct} "
              f"shift={shift_observed}")

    is_complete, missing = check_coverage(manifest, expected_pair_ids)
    if not is_complete:
        print(f"\nINCOMPLETE: {len(missing)} pairs missing outcomes: {missing}")
        print("Re-run the same command to resume — completed pairs are skipped.")
        return

    n_shifts = sum(
        1 for pid in expected_pair_ids
        if manifest_bool(manifest[pid]["shift_observed"])
    )
    total = len(expected_pair_ids)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {n_shifts}/{total} pairs showed a shift toward/away "
          f"from correct when patched.")
    print(f"{'=' * 60}")
    print(
        "\nSTOP/GO (per roadmap Phase P1):\n"
        "  - Consistent, repeated shift across pairs -> real causal signal\n"
        "    -> worth scaling to a properly powered run.\n"
        "  - No consistent shift (roughly what noise alone would produce)\n"
        "    -> the C1 post_plateau_var effect is likely a correlate of\n"
        "    solving, not a cause of it. That is a complete, legitimate,\n"
        "    citable finding — write it up honestly rather than sweeping\n"
        "    layers/positions looking for one that shows something."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase P1 causal patching pilot")
    parser.add_argument("--n-pairs", type=int, default=5,
                         help="Patch pairs per direction (default: 5)")
    parser.add_argument("--layer", type=int, default=-1,
                         help="Layer index to patch (default: -1, last layer)")
    parser.add_argument("--force", action="store_true",
                         help="Redo all pairs, ignoring existing manifest")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_pilot(args.n_pairs, args.layer, args.force, args.output_dir)

    