"""Linear probe for measuring causal geometry."""

import json
from typing import Tuple

import torch
import torch.nn as nn


class CausalProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(model, cfg) -> Tuple[float, CausalProbe]:
    """
    Extract activations from current model state, train a linear probe.
    Returns (accuracy, trained_probe).
    """
    probe_cfg = cfg["probe"]
    label_map = {"causal": 0, "correlational": 1, "unrelated": 2}

    sentences, labels = [], []
    with open(cfg["data"]["probe_sentences"], encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            sentences.append(row["sentence"])
            labels.append(label_map[row["label"]])

    model.model.eval()
    model.attach_hooks()
    all_features = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = model.tokenize([sentence], max_length=128)
            features = model.extract_activations(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            all_features.append(features.cpu().float())

    model._clear_hooks()

    X = torch.cat(all_features, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    n = len(y)
    perm = torch.randperm(n)
    split = int(probe_cfg["train_split"] * n)
    X_train, X_val = X[perm[:split]], X[perm[split:]]
    y_train, y_val = y[perm[:split]], y[perm[split:]]

    probe = CausalProbe(input_dim=X.shape[1], num_classes=probe_cfg["num_classes"])
    optimizer = torch.optim.Adam(probe.parameters(), lr=probe_cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for _ in range(probe_cfg["train_epochs"]):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(X_val).argmax(dim=-1)
            acc = (val_preds == y_val).float().mean().item()

        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    if best_state:
        probe.load_state_dict(best_state)

    print(f"Probe accuracy: {best_acc:.4f}")
    return best_acc, probe
