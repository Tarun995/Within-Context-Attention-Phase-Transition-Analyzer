#!/usr/bin/env python3
"""
run_manifold_sweep.py — on-manifold test across LAYERS, for ANY supported
model (GPT-2 family, Pythia/GPT-NeoX family).

    python run_manifold_sweep.py --model gpt2 --device cuda
    python run_manifold_sweep.py --model gpt2-medium --device cuda
    python run_manifold_sweep.py --model EleutherAI/pythia-70m --device cuda
    python run_manifold_sweep.py --model EleutherAI/pythia-410m --device cuda --n-pairs 100

Add `--n-pairs 5 --n-questions 50` first as a quick smoke run.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Default: auto spread (first, 25%%, 50%%, 75%%, last)")
    p.add_argument("--mode", default="single", choices=["single", "range"])
    p.add_argument("--n-questions", type=int, default=817)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to cpu.")
        args.device = "cpu"

    # dataset first (avoids the pyarrow/CUDA load-order crash seen on Windows)
    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.causal_test_manifold import run_manifold_test, _PAIRS
    from attn_phase.audit.model_adapter import get_num_layers, auto_layers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/3] Loading {args.n_questions} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n_questions)

    print(f"[2/3] Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device).eval()

    n_layers = get_num_layers(model)
    layers = args.layers if args.layers else auto_layers(n_layers)
    bad = [l for l in layers if not 0 <= l < n_layers]
    if bad:
        raise SystemExit(f"Layers {bad} out of range: {args.model} has "
                         f"{n_layers} layers (indices 0-{n_layers - 1}).")

    # every pairwise comparison reported per layer -> Bonferroni over all of them
    alpha = 0.05 / (len(_PAIRS) * len(layers))
    print(f"[3/3] {args.model}: {n_layers} layers, testing {layers}, "
          f"mode={args.mode}, {args.n_pairs} recipients/layer, "
          f"Bonferroni alpha={alpha:.5f}\n")

    rows = []
    for layer in layers:
        print(f"--- layer {layer} ---")
        r = run_manifold_test(model, tokenizer, questions, layer,
                              args.device, n_questions=args.n_pairs,
                              seed=args.seed, mode=args.mode)
        md, pv, fl = r["mean_disruption"], r["wilcoxon_p"], r["flip_counts"]
        graded = (md["donor"] < md["unrelated"] < md["noise"]
                  and pv["donor_vs_unrelated"] < alpha
                  and pv["unrelated_vs_noise"] < alpha)
        template_gap = pv["donor_vs_unrelated_template"] < alpha
        structure_gap = pv["donor_vs_shuffled_donor"] < alpha
        rows.append((layer, md, fl, graded, template_gap, structure_gap, pv))

    n = args.n_pairs
    print("\n=== SUMMARY (mean disruption; smaller = gentler) ===")
    print("layer |  donor | tmplQ  | unrel  | shuff  | noise  | "
          "flips d/t/u/s/n | graded | tmpl!=donor | shuf!=donor")
    for layer, md, fl, graded, tgap, sgap, pv in rows:
        print(f"{layer:5d} | {md['donor']:.4f} | "
              f"{md['unrelated_template']:.4f} | {md['unrelated']:.4f} | "
              f"{md['shuffled_donor']:.4f} | {md['noise']:.4f} | "
              f"{fl['donor']}/{fl['unrelated_template']}/{fl['unrelated']}/"
              f"{fl['shuffled_donor']}/{fl['noise']} of {n} | "
              f"{'YES' if graded else 'no':6s} | {'YES' if tgap else 'no':11s} | "
              f"{'YES' if sgap else 'no'}")
    held = sum(1 for r in rows if r[3])
    print(f"\nGraded ordering (donor < unrelated < noise, adjacent "
          f"p<{alpha:.5f}) held at {held}/{len(rows)} layers.")
    print("\ntmpl!=donor: 'no' -> donor ~= same-template off-topic question, "
          "i.e. the plain-sentence donor-vs-unrelated gap was FORMAT/TOKEN "
          "mismatch, not topic distance. 'YES' -> disruption tracks context "
          "relevance beyond format alone.")
    print("shuf!=donor: 'YES' -> a vector with the DONOR's own values in "
          "random dimension order is still more disruptive than the donor "
          "itself, i.e. dimension-specific STRUCTURE matters, not just the "
          "set of values/scale present. 'no' -> scale/values alone explain "
          "the donor's gentleness, structure doesn't add anything further.")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
run_manifold_sweep.py — on-manifold test across LAYERS, for ANY supported
model (GPT-2 family, Pythia/GPT-NeoX family).

    python run_manifold_sweep.py --model gpt2 --device cuda
    python run_manifold_sweep.py --model gpt2-medium --device cuda
    python run_manifold_sweep.py --model EleutherAI/pythia-70m --device cuda
    python run_manifold_sweep.py --model EleutherAI/pythia-410m --device cuda --n-pairs 100

Add `--n-pairs 5 --n-questions 50` first as a quick smoke run.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Default: auto spread (first, 25%%, 50%%, 75%%, last)")
    p.add_argument("--mode", default="single", choices=["single", "range"])
    p.add_argument("--n-questions", type=int, default=817)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to cpu.")
        args.device = "cpu"

    # dataset first (avoids the pyarrow/CUDA load-order crash seen on Windows)
    from attn_phase.audit.data import load_truthful_qa_mc1
    from attn_phase.audit.causal_test_manifold import run_manifold_test, _PAIRS
    from attn_phase.audit.model_adapter import get_num_layers, auto_layers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/3] Loading {args.n_questions} TruthfulQA mc1 questions...")
    questions = load_truthful_qa_mc1(limit=args.n_questions)

    print(f"[2/3] Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device).eval()

    n_layers = get_num_layers(model)
    layers = args.layers if args.layers else auto_layers(n_layers)
    bad = [l for l in layers if not 0 <= l < n_layers]
    if bad:
        raise SystemExit(f"Layers {bad} out of range: {args.model} has "
                         f"{n_layers} layers (indices 0-{n_layers - 1}).")

    # every pairwise comparison reported per layer -> Bonferroni over all of them
    alpha = 0.05 / (len(_PAIRS) * len(layers))
    print(f"[3/3] {args.model}: {n_layers} layers, testing {layers}, "
          f"mode={args.mode}, {args.n_pairs} recipients/layer, "
          f"Bonferroni alpha={alpha:.5f}\n")

    rows = []
    for layer in layers:
        print(f"--- layer {layer} ---")
        r = run_manifold_test(model, tokenizer, questions, layer,
                              args.device, n_questions=args.n_pairs,
                              seed=args.seed, mode=args.mode)
        md, pv, fl = r["mean_disruption"], r["wilcoxon_p"], r["flip_counts"]
        graded = (md["donor"] < md["unrelated"] < md["noise"]
                  and pv["donor_vs_unrelated"] < alpha
                  and pv["unrelated_vs_noise"] < alpha)
        template_gap = pv["donor_vs_unrelated_template"] < alpha
        structure_gap = pv["donor_vs_shuffled_donor"] < alpha
        rows.append((layer, md, fl, graded, template_gap, structure_gap, pv))

    n = args.n_pairs
    print("\n=== SUMMARY (mean disruption; smaller = gentler) ===")
    print("layer |  donor | tmplQ  | unrel  | shuff  | noise  | "
          "flips d/t/u/s/n | graded | tmpl!=donor | shuf!=donor")
    for layer, md, fl, graded, tgap, sgap, pv in rows:
        print(f"{layer:5d} | {md['donor']:.4f} | "
              f"{md['unrelated_template']:.4f} | {md['unrelated']:.4f} | "
              f"{md['shuffled_donor']:.4f} | {md['noise']:.4f} | "
              f"{fl['donor']}/{fl['unrelated_template']}/{fl['unrelated']}/"
              f"{fl['shuffled_donor']}/{fl['noise']} of {n} | "
              f"{'YES' if graded else 'no':6s} | {'YES' if tgap else 'no':11s} | "
              f"{'YES' if sgap else 'no'}")
    held = sum(1 for r in rows if r[3])
    print(f"\nGraded ordering (donor < unrelated < noise, adjacent "
          f"p<{alpha:.5f}) held at {held}/{len(rows)} layers.")
    print("\ntmpl!=donor: 'no' -> donor ~= same-template off-topic question, "
          "i.e. the plain-sentence donor-vs-unrelated gap was FORMAT/TOKEN "
          "mismatch, not topic distance. 'YES' -> disruption tracks context "
          "relevance beyond format alone.")
    print("shuf!=donor: 'YES' -> a vector with the DONOR's own values in "
          "random dimension order is still more disruptive than the donor "
          "itself, i.e. dimension-specific STRUCTURE matters, not just the "
          "set of values/scale present. 'no' -> scale/values alone explain "
          "the donor's gentleness, structure doesn't add anything further.")


if __name__ == "__main__":
    main()