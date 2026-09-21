#!/usr/bin/env python3
"""
verify_audit_pipeline.py — End-to-end smoke test with REAL GPT-2 weights and
a REAL slice of TruthfulQA. Run this from your repo root, after `pip install
-e .` and after adding the audit/ module (see integration notes).

This is NOT a substitute for tests/test_audit_activations.py and
tests/test_audit_probe.py (run those first — `pytest tests/ -v` — they
catch hook/math bugs with zero network dependency). This script instead
answers a different question: "does the whole pipeline actually work
end-to-end against the real benchmark and real model," on a small slice
(20 questions) so it runs in under a minute before you commit to a full run.

Usage:
    python verify_audit_pipeline.py                  # CPU, gpt2 small
    python verify_audit_pipeline.py --device cuda     # your RTX 2050
    python verify_audit_pipeline.py --n 50            # more questions
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n", type=int, default=20,
                         help="Number of TruthfulQA questions to test with")
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA not available. "
              "Falling back to cpu.")
        args.device = "cpu"

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.activations import predict_answer
    from attn_phase.audit.linear_probe import (
        build_probe_dataset, train_and_evaluate_probe,
    )

    print(f"[1/4] Loading {args.model} on {args.device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    print(f"[2/4] Loading {args.n} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n)
    print(f"  Loaded {len(questions)} questions. "
          f"Example: {questions[0].question!r}")

    print(f"[3/4] Scoring model answers + building probe dataset "
          f"(layer {args.layer})...")
    dataset = build_probe_dataset(model, tokenizer, questions,
                                   layer_idx=args.layer, device=args.device)
    print(f"  Model's own accuracy on these {len(questions)} questions: "
          f"{dataset.y.mean():.1%}")

    if len(set(dataset.y.tolist())) < 2:
        print("\nWARNING: model got every question right or every question "
              "wrong on this small slice — can't train/evaluate a probe "
              "with only one class present. Re-run with a larger --n.")
        sys.exit(0)

    print("[4/4] Training linear probe (small split, just for a smoke "
          "signal — not a real reported number at n=20)...")
    result = train_and_evaluate_probe(dataset, test_size=0.3)

    print("\n--- Smoke test result (NOT a real finding — n too small) ---")
    print(f"  base rate (model's own accuracy): {result['base_rate']:.2f}")
    print(f"  probe accuracy on held-out set:    {result['accuracy']:.2f}")
    print(f"  probe AUC:                         {result['auc']}")
    print(f"  n_train={result['n_train']}, n_test={result['n_test']}")
    print("\nPipeline ran end-to-end without error. Re-run with a larger "
          "--n (e.g. 200+) once your compute budget for a real reported "
          "number is decided.")


if __name__ == "__main__":
    main()
