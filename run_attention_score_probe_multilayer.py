#!/usr/bin/env python3
"""
run_attention_score_probe_multilayer.py — Baseline 2, take 2: aggregate
the concentration score across ALL layers (and heads), not just one
layer's heads. Checks whether the weak per-layer AUCs (~0.51-0.55) from
run_attention_score_probe.py were a genuinely weak signal, or an
under-built reproduction that only did half of "aggregated across
heads/layers."

Usage:
    python run_attention_score_probe_multilayer.py --device cuda --n 817
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n", type=int, default=817)
    parser.add_argument("--layers", type=int, nargs="+",
                         default=list(range(12)),
                         help="Which layers to aggregate over — defaults "
                              "to all 12 layers of GPT-2 small.")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available. "
              "Falling back to cpu.")
        args.device = "cpu"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.attention_score import (
        build_concentration_dataset_multilayer, evaluate_concentration_auc,
    )

    print(f"[1/3] Loading {args.n} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n)
    print(f"  Loaded {len(questions)} questions.")

    print(f"[2/3] Loading {args.model} on {args.device} "
          f"(attn_implementation='eager')...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model,
                                             attn_implementation="eager")
    model.to(args.device)
    model.eval()

    print(f"[3/3] Computing multi-layer concentration score "
          f"(layers {args.layers}, one forward pass per question)...")
    scores, labels, _ = build_concentration_dataset_multilayer(
        model, tokenizer, questions, args.layers, args.device)
    auc = evaluate_concentration_auc(scores, labels)

    print(f"\n--- Multi-layer aggregated concentration score ---")
    print(f"  Layers aggregated: {args.layers}")
    print(f"  AUC: {auc}")
    print(f"  Base rate (model's own accuracy): {labels.mean():.3f}")
    print("\nCompare this to the per-layer sweep's best single-layer AUC "
          "(~0.55, layer 1) already on record. If this is meaningfully "
          "higher, the single-layer version was under-built and this is "
          "the number to use going forward for the causal test's "
          "correlational baseline. If it's similar or lower, the weak "
          "signal itself is the real finding, not a reproduction gap.")


if __name__ == "__main__":
    main()
