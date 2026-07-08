"""
runner.py — Core experiment runner for the Attention Phase Analyzer.

Replaces run_experiment_v2.py and run_layer_sweep.py with a single
configurable function that handles both single-layer and multi-layer
analysis, and both single-seed and multi-seed (Phase C1) task sets.

Entry point for the CLI (cli.py) and for direct use in scripts/notebooks.

Typical usage:
    from attn_phase.runner import run_experiment
    results = run_experiment(config)
"""

import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from attn_phase.tasks import (
    make_copy_task_matched,
    make_transduction_task_matched,
    make_modular_arithmetic_task,
    make_easy_modular_arithmetic_task,
    make_lookup_task,
    make_sorting_task,
    sweep_difficulty_for_failure,
    extract_model_answer,
    answer_matches,
    generate_task_set,
)
from attn_phase.metrics import compute_all_metrics
from attn_phase.stats import run_full_test_battery, print_test_results, summarize_results_to_dict


# ---------------------------------------------------------------------------
# Default config — override any key via the config dict passed to run_experiment
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Model
    "model_name": "gpt2",          # any HuggingFace causal LM identifier

    # Experiment mode
    "mode": "phase_c1",            # "single" | "layer_sweep" | "phase_c1"
                                    # single     : last layer, original 7 tasks
                                    # layer_sweep: all layers, original 7 tasks
                                    # phase_c1   : last layer, multi-seed, stats

    # Layer selection (used in single and phase_c1 modes)
    "layer_idx": -1,               # -1 = last layer; int for specific layer

    # Layer sweep range (used in layer_sweep mode only)
    "layer_range": None,           # None = all layers; or list e.g. [6,7,8,9]

    # Task config
    "base_seed": 42,
    "target_tokens": 504,          # length-matching target for copy/transduction
    "n_instances": 5,              # instances per task type (phase_c1 only)

    # Statistical testing (phase_c1 only)
    "stat_metrics": [
        "plateau_onset_fraction",
        "post_plateau_var",
        "entropy_rise_rate",
    ],
    "alpha": 0.05,

    # Output
    "output_dir": "results",
    "run_label": None,             # if None, auto-generated from mode + model
    "save_plots": True,
    "verbose": True,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def make_generate_fn(tokenizer, model, device: str):
    def generate_fn(prompt: str, max_new_tokens: int = 10) -> str:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    return generate_fn


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_single_task(task: dict, tokenizer, model, generate_fn,
                     device: str, layers: list[int]) -> dict:
    """
    Runs one task through generation (solved/failed check) and a single
    forward pass, then computes metrics for every layer in `layers`.

    If layers has one element, metrics are stored at the top level of the
    task dict (backward-compatible with single-layer analysis).
    If layers has multiple elements, metrics are stored under task["layers"][i].

    Raw curve arrays (_entropy_smoothed, _envelope_curve) are always stored
    at the top level for plotting, keyed by layer index.
    """
    prompt = task["prompt"]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    n_tokens = input_ids.shape[1]

    generated = generate_fn(prompt, max_new_tokens=10)
    model_answer = extract_model_answer(generated, prompt)
    solved, model_answer = answer_matches(model_answer, task["expected_answer"])

    with torch.no_grad():
        output = model(input_ids, output_attentions=True)

    task["actual_tokens"] = int(n_tokens)
    task["solved"] = bool(solved)
    task["model_answer"] = model_answer
    task["raw_continuation"] = generated[len(prompt):].strip()
    task["_curves"] = {}   # keyed by layer_idx, popped before JSON save

    def _safe(v, cast):
        return cast(v) if v is not None else None

    multi_layer = len(layers) > 1

    if multi_layer:
        task["layers"] = {}

    for layer_idx in layers:
        m = compute_all_metrics(output.attentions, input_ids[0],
                                layer_idx=layer_idx)

        layer_data = {
            "plateau_onset_pos":      _safe(m["plateau_onset_pos"], int),
            "plateau_onset_fraction": _safe(m["plateau_onset_fraction"], float),
            "post_plateau_var":       _safe(m["post_plateau_var"], float),
            "has_plateau":            bool(m["has_plateau"]),
            "entropy_rise_rate":      _safe(m["entropy_rise_rate"], float),
            "inflection_position":    _safe(m["inflection_position"], int),
            "inflection_fraction":    _safe(m["inflection_fraction"], float),
            "inflection_effect_size": _safe(m["inflection_effect_size"], float),
            "inflection_high_confidence": bool(m["inflection_high_confidence"]),
            "envelope_growth_pct":    _safe(m["envelope_growth_pct"], float),
        }

        if multi_layer:
            task["layers"][layer_idx] = layer_data
        else:
            task.update(layer_data)

        task["_curves"][layer_idx] = {
            "entropy_smoothed": m["entropy_smoothed"].tolist(),
            "envelope_curve":   m["envelope_curve"].tolist(),
        }

    return task


# ---------------------------------------------------------------------------
# Task set builders per mode
# ---------------------------------------------------------------------------

def build_single_mode_tasks(config: dict, tokenizer, model,
                              generate_fn, device: str) -> tuple:
    """Original 7 deconfounded tasks from Phase 0.5."""
    seed = config["base_seed"]
    target = config["target_tokens"]
    tasks = []

    tasks.append(make_copy_task_matched(seed, tokenizer, target))
    tasks.append(make_transduction_task_matched(seed, tokenizer, target))

    hard_failed, sweep_log = sweep_difficulty_for_failure(
        tokenizer, model, generate_fn,
        seed=seed, target_tokens=target,
    )
    if hard_failed is not None:
        tasks.append(hard_failed)
    else:
        print("  WARNING: sweep found no hard-failed length-matched task.")

    tasks.append(make_modular_arithmetic_task(
        seed=seed, modulus=97, n_digits=2, n_examples=50, op="+"))
    tasks.append(make_easy_modular_arithmetic_task(seed, n_examples=20))
    tasks.append(make_lookup_task(seed, n_keys=8, n_examples=20))
    tasks.append(make_sorting_task(seed, n_numbers=4, n_examples=15))

    return tasks, sweep_log


def build_phase_c1_tasks(config: dict, tokenizer, model,
                          generate_fn, device: str) -> tuple:
    """
    Multi-seed task set for Phase C1 statistical analysis.
    Generates n_instances independently-seeded instances per task type.
    """
    n = config["n_instances"]
    seed = config["base_seed"]
    tasks = []

    # Solved task types
    tasks += generate_task_set("easy_mod_arith", n, seed,
                                n_examples=20)
    tasks += generate_task_set("lookup", n, seed,
                                n_keys=8, n_examples=20)

    # Failed task types
    tasks += generate_task_set("sorting", n, seed,
                                n_numbers=4, n_examples=15)
    tasks += generate_task_set("mod_arith", n, seed,
                                modulus=97, n_digits=2, n_examples=50)

    # Length-matched tasks (need tokenizer — generate individually)
    for i in range(n):
        s = seed * 1000 + i
        t = make_copy_task_matched(s, tokenizer, config["target_tokens"])
        t["task_id"] += f"_inst{i}"
        t["instance_idx"] = i
        tasks.append(t)

        t2 = make_transduction_task_matched(s, tokenizer, config["target_tokens"])
        t2["task_id"] += f"_inst{i}"
        t2["instance_idx"] = i
        tasks.append(t2)

    return tasks, []


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(results: list[dict], output_dir: str,
               run_label: str, layers: list[int]) -> None:
    """Save entropy curve plots for all tasks."""
    os.makedirs(output_dir, exist_ok=True)
    primary_layer = layers[0] if len(layers) == 1 else layers[-1]

    fig, axes = plt.subplots(len(results), 2,
                              figsize=(16, 3 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for i, r in enumerate(results):
        curves = r.get("_curves", {}).get(primary_layer, {})
        entropy = curves.get("entropy_smoothed")
        envelope = curves.get("envelope_curve")

        onset = (r.get("plateau_onset_pos") if len(layers) == 1
                 else r.get("layers", {}).get(primary_layer, {})
                    .get("plateau_onset_pos"))
        label = f"{r['task_id']} | solved={r['solved']}"
        if onset is not None:
            label += f" | onset={onset}"

        if entropy:
            axes[i][0].plot(entropy, linewidth=0.8)
            if onset is not None:
                axes[i][0].axvline(onset, color="green", linestyle="--",
                                   linewidth=1, label=f"onset={onset}")
            axes[i][0].set_title(label, fontsize=7)
            axes[i][0].set_ylabel("entropy")
            axes[i][0].legend(fontsize=6)

        if envelope:
            axes[i][1].plot(envelope, linewidth=0.8, color="orange")
            axes[i][1].set_title("induction envelope", fontsize=8)
            axes[i][1].set_ylabel("amplitude")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{run_label}_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved {plot_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(config: dict | None = None) -> dict:
    """
    Run the full experiment pipeline.

    config : dict of config overrides on top of DEFAULT_CONFIG.
             Pass None to use all defaults.

    Returns a dict with keys:
        "results"      : list of task result dicts
        "sweep_log"    : sweep log from difficulty sweep (may be empty)
        "test_results" : list of statistical test result dicts (phase_c1 only)
        "config"       : the full resolved config used for this run
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["verbose"]:
        print(f"Device: {device}")
        print(f"Model:  {cfg['model_name']}")
        print(f"Mode:   {cfg['mode']}")

    # Seed everything
    random.seed(cfg["base_seed"])
    torch.manual_seed(cfg["base_seed"])
    np.random.seed(cfg["base_seed"])

    # Run label for output files
    run_label = cfg["run_label"] or (
        f"{cfg['mode']}_{cfg['model_name'].replace('/', '_')}"
        f"_seed{cfg['base_seed']}"
    )

    # Load model
    if cfg["verbose"]:
        print(f"Loading {cfg['model_name']}...")
    tokenizer, model = load_model(cfg["model_name"], device)
    generate_fn = make_generate_fn(tokenizer, model, device)

    # Determine which layers to analyze
    if cfg["mode"] == "layer_sweep":
        n_layers = model.config.n_layer
        layers = cfg["layer_range"] or list(range(n_layers))
    else:
        layers = [cfg["layer_idx"]]

    # Build task set
    if cfg["verbose"]:
        print("Building tasks...")
    if cfg["mode"] == "phase_c1":
        tasks, sweep_log = build_phase_c1_tasks(
            cfg, tokenizer, model, generate_fn, device)
    else:
        tasks, sweep_log = build_single_mode_tasks(
            cfg, tokenizer, model, generate_fn, device)

    if cfg["verbose"]:
        print(f"Running {len(tasks)} tasks across {len(layers)} layer(s)...")

    # Run tasks
    results = []
    for t in tasks:
        if cfg["verbose"]:
            print(f"  {t['task_id']}...")
        results.append(run_single_task(
            t, tokenizer, model, generate_fn, device, layers))

    # Save plots (before popping curves)
    if cfg["save_plots"]:
        save_plots(results, cfg["output_dir"], run_label, layers)

    # Pop raw curves before JSON save
    for r in results:
        r.pop("_curves", None)

    # Statistical testing (phase_c1 only)
    test_results_dicts = []
    if cfg["mode"] == "phase_c1":
        if cfg["verbose"]:
            print("\nRunning statistical tests...")
        layer_arg = None if len(layers) == 1 else layers
        stat_results = run_full_test_battery(
            results,
            metrics=cfg["stat_metrics"],
            layers=layer_arg,
            alpha=cfg["alpha"],
        )
        print_test_results(stat_results)
        test_results_dicts = summarize_results_to_dict(stat_results)

    # Save JSON
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], f"{run_label}_results.json")
    payload = {
        "config": cfg,
        "results": results,
        "sweep_log": sweep_log,
        "test_results": test_results_dicts,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    if cfg["verbose"]:
        print(f"Saved {out_path}")

    return payload