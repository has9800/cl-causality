"""Model loading and activation extraction."""

from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalModel:
    """Wraps GPT-Neo with activation extraction from target layers."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["name"],
            torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.target_layers = cfg["model"]["target_layers"]
        self._hooks = []
        self._activations = {}

    def attach_hooks(self):
        """Register forward hooks on target layers to capture activations."""
        self._clear_hooks()
        blocks = self.model.transformer.h

        for layer_idx in self.target_layers:

            def hook_fn(module, input_, output, idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                self._activations[idx] = hidden

            self._hooks.append(blocks[layer_idx].register_forward_hook(hook_fn))

    def _clear_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def extract_activations(self, input_ids, attention_mask=None):
        """
        Forward pass, return pooled activations from target layers.
        Returns: tensor of shape (batch_size, hidden_size * num_layers)
        """
        self._activations = {}
        with torch.set_grad_enabled(self.model.training):
            self.model(input_ids=input_ids, attention_mask=attention_mask)

        pooled_layers = []
        pooling = self.cfg["contrastive"].get("pooling", "mean")
        for idx in self.target_layers:
            hidden = self._activations[idx]  # (batch, seq_len, hidden_size)
            if pooling == "mean":
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden.mean(dim=1)
            else:
                pooled = hidden[:, -1, :]
            pooled_layers.append(pooled)

        return torch.cat(pooled_layers, dim=-1)

    def apply_lora(self):
        """Attach LoRA adapters for training."""
        lora_cfg = self.cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def tokenize(self, texts: List[str], max_length=128):
        """Tokenize a list of strings."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
