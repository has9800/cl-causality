"""Measure base model: probe accuracy + trace quality."""

import json
from pathlib import Path

import yaml

from src.evaluator import generate_traces, score_with_nemotron
from src.model import CausalModel
from src.probe import train_probe


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    model = CausalModel(cfg)

    print("\n--- Probe Accuracy (Base Model) ---")
    probe_acc, _ = train_probe(model, cfg)

    print("\n--- Generating Traces (Base Model) ---")
    traces = generate_traces(model, cfg)

    print("\n--- Scoring with Nemotron ---")
    scored = score_with_nemotron(traces, cfg)

    output_dir = Path("outputs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "probe_accuracy": probe_acc,
        "mean_nemotron_score": sum(d["mean_score"] for d in scored) / len(scored),
        "mean_best_score": sum(d["best_score"] for d in scored) / len(scored),
        "scored_traces": scored,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nBaseline probe accuracy: {probe_acc:.4f}")
    print(f"Baseline mean Nemotron score: {results['mean_nemotron_score']:.4f}")
    print(f"Baseline mean best score: {results['mean_best_score']:.4f}")
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
