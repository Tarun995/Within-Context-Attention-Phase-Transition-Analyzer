"""
cli.py — Command-line interface for the Attention Phase Analyzer.

Usage:
    attn-phase run --mode single
    attn-phase run --mode phase_c1
    attn-phase run --mode layer_sweep
    attn-phase run --mode phase_c1 --model gpt2-medium --instances 5
    attn-phase run --config configs/phase_c1.yaml
"""

import argparse
import sys
import yaml

from attn_phase.runner import run_experiment, DEFAULT_CONFIG


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="attn-phase",
        description=(
            "Attention Phase Analyzer — detects and statistically tests "
            "within-context attention dynamics in transformer models."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run an experiment.",
    )
    run_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. CLI flags override config file values.",
    )
    run_parser.add_argument(
        "--mode", type=str,
        choices=["single", "layer_sweep", "phase_c1"],
        default=None,
        help=(
            "Experiment mode. "
            "single: last layer, original 7 deconfounded tasks. "
            "layer_sweep: all layers, original 7 tasks, heatmap output. "
            "phase_c1: last layer, multi-seed tasks, Mann-Whitney stats test."
        ),
    )
    run_parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model identifier (default: gpt2).",
    )
    run_parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (default: 42).",
    )
    run_parser.add_argument(
        "--instances", type=int, default=None,
        help="Number of seeded instances per task type for phase_c1 (default: 5).",
    )
    run_parser.add_argument(
        "--layer", type=int, default=None,
        help="Single layer index to analyze for 'single' mode (default: -1 = last).",
    )
    run_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results and plots (default: results/).",
    )
    run_parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving plots.",
    )
    run_parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args(argv)


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_config(args) -> dict:
    """Merge DEFAULT_CONFIG <- yaml config <- CLI flags (highest priority)."""
    config = dict(DEFAULT_CONFIG)

    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        config.update(yaml_cfg)

    # CLI flags override everything
    if args.mode is not None:
        config["mode"] = args.mode
    if args.model is not None:
        config["model_name"] = args.model
    if args.seed is not None:
        config["base_seed"] = args.seed
    if args.instances is not None:
        config["n_instances"] = args.instances
    if args.layer is not None:
        config["layer_idx"] = args.layer
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.no_plots:
        config["save_plots"] = False
    if args.quiet:
        config["verbose"] = False

    return config


def main(argv=None):
    args = parse_args(argv)

    if args.command is None:
        print("Usage: attn-phase run --mode [single|layer_sweep|phase_c1]")
        print("       attn-phase run --help")
        sys.exit(0)

    if args.command == "run":
        config = build_config(args)

        if config.get("verbose"):
            print("\nAttention Phase Analyzer")
            print("=" * 40)
            print(f"Mode     : {config['mode']}")
            print(f"Model    : {config['model_name']}")
            print(f"Seed     : {config['base_seed']}")
            if config["mode"] == "phase_c1":
                print(f"Instances: {config['n_instances']} per task type")
            print("=" * 40 + "\n")

        run_experiment(config)


if __name__ == "__main__":
    main()