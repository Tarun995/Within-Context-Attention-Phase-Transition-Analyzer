"""
attn_phase — Attention Phase Analyzer

A pipeline for detecting and statistically testing within-context attention
dynamics in transformer models.

Quick start:
    from attn_phase.runner import run_experiment
    results = run_experiment({"mode": "single", "model_name": "gpt2"})

Or via the command line after installation:
    attn-phase run --mode single
    attn-phase run --mode phase_c1 --instances 5
    attn-phase run --mode layer_sweep
"""

from attn_phase.runner import run_experiment, DEFAULT_CONFIG
from attn_phase.metrics import compute_all_metrics
from attn_phase.tasks import generate_task_set, answer_matches
from attn_phase.stats import run_full_test_battery, print_test_results

__version__ = "0.1.0"
__all__ = [
    "run_experiment",
    "DEFAULT_CONFIG",
    "compute_all_metrics",
    "generate_task_set",
    "answer_matches",
    "run_full_test_battery",
    "print_test_results",
]