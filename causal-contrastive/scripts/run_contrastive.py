"""Run contrastive training and evaluate."""

import json
from pathlib import Path

import yaml

from src.contrastive_trainer import ContrastiveTrainer
from src.dataset import TripletDataset
from src.evaluator import generate_traces, score_with_nemotron
from src.model import CausalModel
from src.probe import train_probe


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    print("=" * 60)
    print("CONTRASTIVE TRAINING + EVALUATION")
    print("=" * 60)

    model = CausalModel(cfg)

    print("\n--- Probe Accuracy (Before Contrastive) ---")
    pre_acc, _ = train_probe(model, cfg)

    if cfg["lora"]["enabled"]:
        model.apply_lora()

    dataset = TripletDataset(cfg["data"]["probe_sentences"])
    print(f"Triplet dataset: {len(dataset)} triplets")

    print("\n--- Contrastive Training ---")
    trainer = ContrastiveTrainer(model, cfg)
    trainer.train(dataset)

    output_dir = Path("outputs/contrastive")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output_dir / "model")
    model.tokenizer.save_pretrained(output_dir / "model")

    print("\n--- Probe Accuracy (After Contrastive) ---")
    post_acc, _ = train_probe(model, cfg)

    print("\n--- Generating Traces (Contrastive Model) ---")
    traces = generate_traces(model, cfg)

    print("\n--- Scoring with Nemotron ---")
    scored = score_with_nemotron(traces, cfg)

    results = {
        "probe_accuracy_before": pre_acc,
        "probe_accuracy_after": post_acc,
        "probe_accuracy_change": post_acc - pre_acc,
        "mean_nemotron_score": sum(d["mean_score"] for d in scored) / len(scored),
        "mean_best_score": sum(d["best_score"] for d in scored) / len(scored),
        "scored_traces": scored,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nProbe accuracy: {pre_acc:.4f} -> {post_acc:.4f} (delta: {post_acc - pre_acc:+.4f})")
    print(f"Mean Nemotron score: {results['mean_nemotron_score']:.4f}")
    print(f"Mean best score: {results['mean_best_score']:.4f}")
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
