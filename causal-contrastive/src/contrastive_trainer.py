"""Contrastive training on reasoning layer activations."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import TripletDataset
from src.model import CausalModel


def collate_triplets(batch):
    """Unzip triplets into three lists."""
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


class ContrastiveTrainer:
    def __init__(self, model: CausalModel, cfg: dict):
        self.model = model
        self.cfg = cfg
        cont_cfg = cfg["contrastive"]

        self.margin = cont_cfg["margin"]
        self.distance_fn = cont_cfg.get("distance", "cosine")
        self.max_seq_length = cont_cfg["max_seq_length"]

        trainable_params = [p for p in model.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cont_cfg["learning_rate"],
            weight_decay=0.01,
        )

        if self.distance_fn == "cosine":
            self.loss_fn = nn.TripletMarginWithDistanceLoss(
                distance_function=lambda a, b: 1 - nn.functional.cosine_similarity(a, b),
                margin=self.margin,
            )
        else:
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin)

    def _extract(self, texts):
        """Tokenize and extract pooled activations."""
        inputs = self.model.tokenize(texts, max_length=self.max_seq_length)
        return self.model.extract_activations(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    def train(self, dataset: TripletDataset):
        """Run contrastive training loop."""
        cont_cfg = self.cfg["contrastive"]
        loader = DataLoader(
            dataset,
            batch_size=cont_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_triplets,
        )

        self.model.model.train()
        self.model.attach_hooks()

        for epoch in range(cont_cfg["num_epochs"]):
            total_loss = 0.0
            num_batches = 0

            for anchors, positives, negatives in loader:
                self.optimizer.zero_grad()

                anchor_acts = self._extract(anchors)
                positive_acts = self._extract(positives)
                negative_acts = self._extract(negatives)

                loss = self.loss_fn(anchor_acts, positive_acts, negative_acts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{cont_cfg['num_epochs']}: loss={avg_loss:.4f}")

        self.model._clear_hooks()
        self.model.model.eval()
