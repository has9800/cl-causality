"""Triplet dataset for contrastive training."""

import json
import random
from typing import Tuple

from torch.utils.data import Dataset


LABEL_MAP = {"causal": 0, "correlational": 1, "unrelated": 2}


class TripletDataset(Dataset):
    """
    Returns (anchor, positive, negative) triplets.
    Anchor and positive share the same label.
    Negative has a different label.
    For causal contrastive: anchor=causal, positive=causal, negative=correlational.
    We focus on causal vs correlational separation primarily.
    """

    def __init__(self, data_path: str, focus_labels=("causal", "correlational")):
        self.sentences = []
        self.labels = []

        with open(data_path, encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                self.sentences.append(row["sentence"])
                self.labels.append(row["label"])

        self.by_label = {}
        for sentence, label in zip(self.sentences, self.labels):
            self.by_label.setdefault(label, []).append(sentence)

        self.focus_labels = focus_labels
        self.anchor_indices = [
            i for i, label in enumerate(self.labels) if label in focus_labels
        ]

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx) -> Tuple[str, str, str]:
        anchor_idx = self.anchor_indices[idx]
        anchor_sent = self.sentences[anchor_idx]
        anchor_label = self.labels[anchor_idx]

        positives = [s for s in self.by_label[anchor_label] if s != anchor_sent]
        positive_sent = random.choice(positives)

        neg_labels = [label for label in self.focus_labels if label != anchor_label]
        neg_label = random.choice(neg_labels)
        negative_sent = random.choice(self.by_label[neg_label])

        return anchor_sent, positive_sent, negative_sent
