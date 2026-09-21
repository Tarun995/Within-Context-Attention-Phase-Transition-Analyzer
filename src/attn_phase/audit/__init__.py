"""
attn_phase.audit — Causal Validity Audit of LLM Confidence-Probing Methods.

Extends the project's activation-patching methodology (proven out in
Phase P1 on the synthetic mod_arith task family) to test whether published
LLM confidence/correctness-probing methods are causally load-bearing or
merely correlational, on real benchmarks (TruthfulQA first).

Modules:
    data          — TruthfulQA mc1 loading, prompt formatting
    activations   — hidden-state capture/patching, choice log-lik scoring
    linear_probe  — Baseline 1: hidden-state linear probe (Azaria & Mitchell-style)
"""
