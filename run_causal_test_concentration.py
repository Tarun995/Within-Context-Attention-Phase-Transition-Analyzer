#!/usr/bin/env python3
"""
run_causal_test_concentration.py — Baseline 2's causal test: does patching
the representation behind the attention-concentration score actually
shift the recipient's own concentration, and/or its correctness?

Usage:
    python run_causal_test_concentration.py --device cuda --layer 3 \
        --n-questions 817 --n-pairs 50
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, required=True,
                         help="The layer to patch — use whichever layer "
                              "run_attention_score_probe.py found had the "
                              "highest AUC. Must NOT be 11 (the last "
                              "layer) unless you also pass a valid "
                              "--read-layer, since concentration shift is "
                              "measured at a layer AFTER the patched one.")
    parser.add_argument("--read-layer", type=int, default=-1,
                         help="Layer to measure the concentration SHIFT "
                              "in — must differ from --layer. Defaults to "
                              "-1 (last layer), which works for any "
                              "--layer except 11/-1 itself.")
    parser.add_argument("--n-questions", type=int, default=817)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available. "
              "Falling back to cpu.")
        args.device = "cpu"

    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.attention_score import build_concentration_dataset
    from attn_phase.audit.causal_test_concentration import (
        build_concentration_pools, run_concentration_causal_test,
    )
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[1/4] Loading {args.n_questions} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n_questions)
    print(f"  Loaded {len(questions)} questions.")

    print(f"[2/4] Loading {args.model} on {args.device} "
          f"(attn_implementation='eager')...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model,
                                             attn_implementation="eager")
    model.to(args.device)
    model.eval()

    print(f"[3/4] Scoring all questions for concentration at layer "
          f"{args.layer}...")
    scores, labels, mcqs = build_concentration_dataset(
        model, tokenizer, questions, args.layer, args.device)
    low_pool, high_pool = build_concentration_pools(scores, labels, mcqs)
    print(f"  {len(low_pool)} low-concentration, {len(high_pool)} "
          f"high-concentration questions in the tercile pools.")

    print(f"\n[4/4] Running causal test: patch layer {args.layer}, "
          f"read shift at layer {args.read_layer}, "
          f"{args.n_pairs} pairs per direction...\n")

    print("--- Direction 1: donor=HIGH concentration -> recipient=LOW "
          "(does injecting a 'sharply focused' representation sharpen a "
          "diffuse recipient's own attention?) ---")
    result_sharpen = run_concentration_causal_test(
        model, tokenizer, donor_pool=high_pool, recipient_pool=low_pool,
        layer_idx=args.layer, read_layer_idx=args.read_layer,
        device=args.device, n_pairs=args.n_pairs, seed=args.seed,
        direction="low_to_high",
    )
    print(f"  concentration shift — donor: "
          f"{result_sharpen.donor_concentration_shift_count}/{result_sharpen.n_pairs}, "
          f"control: {result_sharpen.control_concentration_shift_count}/{result_sharpen.n_pairs}, "
          f"p={result_sharpen.concentration_shift_p_value:.4f}")
    print(f"  correctness flip — donor: "
          f"{result_sharpen.donor_correctness_flip_count}/{result_sharpen.n_pairs}, "
          f"control: {result_sharpen.control_correctness_flip_count}/{result_sharpen.n_pairs}, "
          f"p={result_sharpen.correctness_flip_p_value:.4f}")

    print("\n--- Direction 2: donor=LOW concentration -> recipient=HIGH "
          "(does injecting a 'diffuse' representation flatten a sharply-"
          "focused recipient's own attention?) ---")
    result_diffuse = run_concentration_causal_test(
        model, tokenizer, donor_pool=low_pool, recipient_pool=high_pool,
        layer_idx=args.layer, read_layer_idx=args.read_layer,
        device=args.device, n_pairs=args.n_pairs, seed=args.seed,
        direction="high_to_low",
    )
    print(f"  concentration shift — donor: "
          f"{result_diffuse.donor_concentration_shift_count}/{result_diffuse.n_pairs}, "
          f"control: {result_diffuse.control_concentration_shift_count}/{result_diffuse.n_pairs}, "
          f"p={result_diffuse.concentration_shift_p_value:.4f}")
    print(f"  correctness flip — donor: "
          f"{result_diffuse.donor_correctness_flip_count}/{result_diffuse.n_pairs}, "
          f"control: {result_diffuse.control_correctness_flip_count}/{result_diffuse.n_pairs}, "
          f"p={result_diffuse.correctness_flip_p_value:.4f}")

    print("\n--- How to read this ---")
    print("Two separate questions are being tested: does patching shift "
          "the recipient's OWN concentration score (a direct test of "
          "whether the representation controls the attention pattern "
          "itself), and does it flip correctness (comparable to baseline "
          "1's test). Donor rate meaningfully above control rate, with "
          "p < 0.025 (Bonferroni for 2 directions), is evidence of a real "
          "effect for that specific outcome. Donor ~= control is a null "
          "result for that outcome — report it honestly either way.")


if __name__ == "__main__":
    main()
