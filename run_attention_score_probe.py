#!/usr/bin/env python3
"""
run_attention_score_probe.py — Baseline 2: validate the attention-
concentration score's AUC, per layer, on real TruthfulQA + GPT-2.

Usage:
    python run_attention_score_probe.py --device cuda --n 817
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n", type=int, default=817,
                         help="Number of TruthfulQA questions to test with")
    parser.add_argument("--layers", type=int, nargs="+",
                         default=[1, 2, 3, 4, 6, 9, 11],
                         help="Which layers to sweep (GPT-2 small has 12, "
                              "indices 0-11). Defaults roughly mirror "
                              "baseline 1's own sweep for comparability, "
                              "using 11 instead of -1 since this script "
                              "needs an explicit index for read_layer_idx "
                              "math later in the causal test.")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available. "
              "Falling back to cpu.")
        args.device = "cpu"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.attention_score import (
        build_concentration_dataset, evaluate_concentration_auc,
    )

    print(f"[1/3] Loading {args.n} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n)
    print(f"  Loaded {len(questions)} questions.")

    print(f"[2/3] Loading {args.model} on {args.device} "
          f"(attn_implementation='eager' — REQUIRED for output_attentions "
          f"to actually return weights on recent transformers versions)...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model,
                                             attn_implementation="eager")
    model.to(args.device)
    model.eval()

    print(f"[3/3] Sweeping layers {args.layers}...\n")
    results = {}
    for layer in args.layers:
        print(f"--- Layer {layer} ---")
        scores, labels, _ = build_concentration_dataset(
            model, tokenizer, questions, layer, args.device)
        auc = evaluate_concentration_auc(scores, labels)
        results[layer] = auc
        print(f"  AUC: {auc}\n")

    print("--- Summary ---")
    for layer, auc in results.items():
        marker = " <-- BEST" if auc == max(v for v in results.values()
                                            if v is not None) else ""
        print(f"  layer {layer}: AUC {auc}{marker}")
    print("\nPick the best layer above, then run the causal test with "
          "run_causal_test_concentration.py --layer <best>.")


if __name__ == "__main__":
    main()
