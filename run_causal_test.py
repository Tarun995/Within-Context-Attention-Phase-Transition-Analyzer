#!/usr/bin/env python3
"""
run_causal_test.py — The actual causal audit run: is the layer-3 hidden
state that predicted correctness (AUC ~0.83) actually load-bearing, or
just correlational?

Usage:
    python run_causal_test.py --device cuda --layer 3 --n-questions 817 --n-pairs 50
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=3,
                         help="Layer to test — default 3, the validated "
                              "AUC peak from the probe layer sweep")
    parser.add_argument("--n-questions", type=int, default=817,
                         help="How many TruthfulQA questions to score "
                              "first, to build the confident-correct / "
                              "confident-incorrect pools")
    parser.add_argument("--n-pairs", type=int, default=50,
                         help="Donor/recipient pairs to test PER "
                              "direction (so total forward-pass work is "
                              "roughly proportional to 2x this number)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="single", choices=["single", "range"],
                         help="'single' patches only the last prefix "
                              "token; 'range' patches every prefix token "
                              "(closes the bypass where the model can "
                              "still read untouched question tokens). "
                              "Run 'single' first; escalate to 'range' if "
                              "it comes back null, same as P1's own "
                              "single-position -> post-plateau-range "
                              "escalation.")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available. "
              "Falling back to cpu.")
        args.device = "cpu"

    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.causal_test import (
        build_confidence_pools, run_causal_test,
    )
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[1/4] Loading {args.n_questions} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n_questions)
    print(f"  Loaded {len(questions)} questions.")

    print(f"[2/4] Loading {args.model} on {args.device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    print(f"[3/4] Scoring all questions to build confident-correct / "
          f"confident-incorrect pools...")
    correct_pool, incorrect_pool = build_confidence_pools(
        model, tokenizer, questions, args.device)
    print(f"  {len(correct_pool)} confidently-scored correct answers, "
          f"{len(incorrect_pool)} confidently-scored incorrect answers "
          f"available as donor/recipient pools.")

    if len(correct_pool) < 5 or len(incorrect_pool) < 5:
        print("\nERROR: not enough questions in one of the pools to run "
              "a meaningful test. Try a larger --n-questions.")
        return

    print(f"\n[4/4] Running causal test at layer {args.layer}, "
          f"mode={args.mode}, {args.n_pairs} pairs per direction...\n")

    print("--- Direction 1: donor=CORRECT -> recipient=INCORRECT "
          "(does injecting a 'confident correct' representation FIX a "
          "wrong answer?) ---")
    result_fix = run_causal_test(
        model, tokenizer, donor_pool=correct_pool,
        recipient_pool=incorrect_pool, layer_idx=args.layer,
        device=args.device, n_pairs=args.n_pairs, seed=args.seed,
        direction="incorrect_to_correct", mode=args.mode,
    )
    print(f"  donor flip rate:   {result_fix.donor_flip_rate:.2%} "
          f"({result_fix.donor_flip_count}/{result_fix.n_pairs})")
    print(f"  control flip rate: {result_fix.control_flip_rate:.2%} "
          f"({result_fix.control_flip_count}/{result_fix.n_pairs})")
    print(f"  McNemar stat: {result_fix.mcnemar_stat}, "
          f"p-value: {result_fix.p_value:.4f}")

    print("\n--- Direction 2: donor=INCORRECT -> recipient=CORRECT "
          "(does injecting a 'confident incorrect' representation BREAK "
          "a right answer?) ---")
    result_break = run_causal_test(
        model, tokenizer, donor_pool=incorrect_pool,
        recipient_pool=correct_pool, layer_idx=args.layer,
        device=args.device, n_pairs=args.n_pairs, seed=args.seed,
        direction="correct_to_incorrect", mode=args.mode,
    )
    print(f"  donor flip rate:   {result_break.donor_flip_rate:.2%} "
          f"({result_break.donor_flip_count}/{result_break.n_pairs})")
    print(f"  control flip rate: {result_break.control_flip_rate:.2%} "
          f"({result_break.control_flip_count}/{result_break.n_pairs})")
    print(f"  McNemar stat: {result_break.mcnemar_stat}, "
          f"p-value: {result_break.p_value:.4f}")

    print("\n--- How to read this ---")
    print("If donor flip rate is meaningfully HIGHER than control flip "
          "rate, AND p-value < 0.05 (correct for testing 2 directions: "
          "use 0.025 as your threshold, Bonferroni), that's evidence the "
          "layer-3 representation is causally load-bearing, not just "
          "correlational — in that direction.")
    print("If donor and control flip rates are similar, that's a genuine "
          "null result: the AUC-0.83 probe signal predicts correctness "
          "without being the mechanism that PRODUCES it. That's still a "
          "publishable, honest finding — exactly what happened with "
          "post_plateau_var in Phase P1.")


if __name__ == "__main__":
    main()
