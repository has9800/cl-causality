"""Compare baseline vs contrastive results."""

import json
from pathlib import Path


def main():
    baseline = json.loads(Path("outputs/baseline/results.json").read_text(encoding="utf-8"))
    contrastive = json.loads(Path("outputs/contrastive/results.json").read_text(encoding="utf-8"))

    print("=" * 60)
    print("COMPARISON: Baseline vs Contrastive")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'Contrastive':>12} {'Delta':>12}")
    print("-" * 66)

    b_probe = baseline["probe_accuracy"]
    c_probe = contrastive["probe_accuracy_after"]
    print(f"{'Probe Accuracy':<30} {b_probe:>12.4f} {c_probe:>12.4f} {c_probe - b_probe:>+12.4f}")

    b_nem = baseline["mean_nemotron_score"]
    c_nem = contrastive["mean_nemotron_score"]
    print(f"{'Mean Nemotron Score':<30} {b_nem:>12.4f} {c_nem:>12.4f} {c_nem - b_nem:>+12.4f}")

    b_best = baseline["mean_best_score"]
    c_best = contrastive["mean_best_score"]
    print(f"{'Mean Best Trace Score':<30} {b_best:>12.4f} {c_best:>12.4f} {c_best - b_best:>+12.4f}")

    print("\n" + "=" * 60)
    print("EXAMPLE TRACE COMPARISON")
    print("=" * 60)
    for i in range(min(5, len(baseline["scored_traces"]))):
        base = baseline["scored_traces"][i]
        cont = contrastive["scored_traces"][i]
        print(f"\nPrompt: {base['prompt']}")
        print(f"  Baseline best (score={base['best_score']:.2f}): {base['best_trace'][:200]}")
        print(f"  Contrastive best (score={cont['best_score']:.2f}): {cont['best_trace'][:200]}")


if __name__ == "__main__":
    main()
