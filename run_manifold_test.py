#!/usr/bin/env python3
"""
run_manifold_test.py — Does perturbation severity scale with how
"in-distribution" the injected representation is? Tests donor (real,
on-topic) vs. unrelated (real, off-topic) vs. noise (not real at all).

Usage:
    python run_manifold_test.py --device cuda --layer 3 --n-questions 817 --n-pairs 50
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=3,
                         help="Default 3 — Baseline 1's validated AUC "
                              "peak, for direct comparability with "
                              "already-reported single-patch results.")
    parser.add_argument("--mode", default="single", choices=["single", "range"])
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
    from attn_phase.audit.causal_test_manifold import run_manifold_test
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"[1/3] Loading {args.n_questions} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n_questions)
    print(f"  Loaded {len(questions)} questions.")

    print(f"[2/3] Loading {args.model} on {args.device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    print(f"\n[3/3] Running manifold test: layer {args.layer}, mode="
          f"{args.mode}, {args.n_pairs} recipients...\n")
    result = run_manifold_test(
        model, tokenizer, question_pool=questions, layer_idx=args.layer,
        device=args.device, n_questions=args.n_pairs, seed=args.seed,
        mode=args.mode,
    )

    print("\n--- Mean disruption magnitude (L2 distance in choice-loglik "
          "space; SMALLER = gentler perturbation) ---")
    for cond, val in result["mean_disruption"].items():
        print(f"  {cond}: {val:.4f}")

    print("\n--- Correctness flips (out of {} recipients) ---".format(
        result["n_questions"]))
    for cond, val in result["flip_counts"].items():
        print(f"  {cond}: {val}/{result['n_questions']}")

    print("\n--- Wilcoxon signed-rank p-values (pairwise) ---")
    for pair, p in result["wilcoxon_p"].items():
        print(f"  {pair}: p={p:.4f}")

    print("\n--- How to read this ---")
    print("On-manifold hypothesis predicts: donor < unrelated < noise, in "
          "mean disruption magnitude, with donor_vs_noise reaching "
          "significance (p<0.05) and ideally donor_vs_unrelated and "
          "unrelated_vs_noise showing the same direction even if weaker. "
          "If 'unrelated' behaves like noise instead of like donor, the "
          "effect isn't about manifold geometry generally — something "
          "more specific to QA-prompt structure is going on, and that's "
          "worth investigating separately, not the same finding restated.")


if __name__ == "__main__":
    main()
