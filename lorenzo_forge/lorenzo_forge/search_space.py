"""Discrete architecture search space and the ArchitectureSpec encoding used
both as search-candidate genome and as the meta-model's prediction target."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

NUM_BLOCKS_CHOICES = [1, 2, 3, 4]
UNITS_CHOICES = [32, 64, 128, 256]
KERNEL_CHOICES = [3, 5]
ACTIVATION_CHOICES = ["relu", "tanh"]
DROPOUT_CHOICES = [0.0, 0.2, 0.4]
OPTIMIZER_CHOICES = ["adam", "sgd"]
LR_CHOICES = [1e-2, 1e-3, 1e-4]
# Text-only axes (like kernel is image-only): fixed at index 0 for other tasks.
EMBEDDING_CHOICES = [16, 32, 64]
ENCODER_CHOICES = ["lstm", "gru", "conv1d"]

# Fixed head order/sizes. Used as the one-hot layout for architecture encoding
# and (historically) as the classification meta-model's output heads.
HEADS = {
    "num_blocks": len(NUM_BLOCKS_CHOICES),
    "units": len(UNITS_CHOICES),
    "kernel": len(KERNEL_CHOICES),
    "activation": len(ACTIVATION_CHOICES),
    "dropout": len(DROPOUT_CHOICES),
    "optimizer": len(OPTIMIZER_CHOICES),
    "lr": len(LR_CHOICES),
    "embedding": len(EMBEDDING_CHOICES),
    "encoder": len(ENCODER_CHOICES),
}

# Length of the one-hot architecture feature vector fed to the scorer meta-model.
ARCH_FEATURE_DIM = sum(HEADS.values())


@dataclass(frozen=True)
class ArchitectureSpec:
    task_type: str  # "tabular" | "image" | "text"
    num_blocks: int
    units: int
    activation: str
    dropout: float
    optimizer: str
    lr: float
    kernel_size: Optional[int] = None  # image conv kernel, or text conv1d kernel
    embedding_dim: Optional[int] = None  # text only
    encoder: Optional[str] = None  # text only: lstm | gru | conv1d

    def label_indices(self) -> dict[str, int]:
        return {
            "num_blocks": NUM_BLOCKS_CHOICES.index(self.num_blocks),
            "units": UNITS_CHOICES.index(self.units),
            "kernel": KERNEL_CHOICES.index(self.kernel_size) if self.kernel_size is not None else 0,
            "activation": ACTIVATION_CHOICES.index(self.activation),
            "dropout": DROPOUT_CHOICES.index(self.dropout),
            "optimizer": OPTIMIZER_CHOICES.index(self.optimizer),
            "lr": LR_CHOICES.index(self.lr),
            "embedding": EMBEDDING_CHOICES.index(self.embedding_dim) if self.embedding_dim is not None else 0,
            "encoder": ENCODER_CHOICES.index(self.encoder) if self.encoder is not None else 0,
        }

    def to_dict(self) -> dict:
        if self.task_type == "image":
            blocks = [
                {"type": "conv2d", "filters": self.units, "kernel_size": self.kernel_size, "activation": self.activation}
                for _ in range(self.num_blocks)
            ]
        elif self.task_type == "text":
            blocks = [
                {"type": self.encoder, "units": self.units,
                 **({"kernel_size": self.kernel_size} if self.encoder == "conv1d" else {})}
                for _ in range(self.num_blocks)
            ]
        else:
            blocks = [
                {"type": "dense", "units": self.units, "activation": self.activation}
                for _ in range(self.num_blocks)
            ]
        out = {
            "task_type": self.task_type,
            "blocks": blocks,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
        }
        if self.task_type == "text":
            out["embedding_dim"] = self.embedding_dim
        return out

    def to_feature_vector(self) -> np.ndarray:
        """One-hot encoding of the architecture, laid out per HEADS order.
        This is the architecture half of the scorer meta-model's input."""
        idx = self.label_indices()
        parts: list[np.ndarray] = []
        for name, size in HEADS.items():
            v = np.zeros(size, dtype=np.float32)
            v[idx[name]] = 1.0
            parts.append(v)
        return np.concatenate(parts)

    def complexity_proxy(self) -> int:
        """Cheap ordering key for tie-breaking: prefer smaller/faster models."""
        k = self.kernel_size if self.kernel_size is not None else 1
        conv_cost = k * k if self.task_type == "image" else 1
        return self.num_blocks * self.units * conv_cost

    def to_raw_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "num_blocks": self.num_blocks,
            "units": self.units,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "embedding_dim": self.embedding_dim,
            "encoder": self.encoder,
        }

    @classmethod
    def from_raw_dict(cls, d: dict) -> "ArchitectureSpec":
        fields = {
            "task_type", "num_blocks", "units", "kernel_size", "activation",
            "dropout", "optimizer", "lr", "embedding_dim", "encoder",
        }
        return cls(**{k: v for k, v in d.items() if k in fields})

    def describe(self) -> str:
        if self.task_type == "text":
            kind = self.encoder if self.encoder else "seq"
            extra = f", kernel={self.kernel_size}" if self.encoder == "conv1d" else ""
            return (
                f"[text] embed{self.embedding_dim} -> {self.num_blocks}x {kind}(units={self.units}{extra}) "
                f"-> dropout={self.dropout} -> optimizer={self.optimizer}(lr={self.lr})"
            )
        kind = "Conv2D" if self.task_type == "image" else "Dense"
        extra = f", kernel={self.kernel_size}" if self.task_type == "image" else ""
        return (
            f"[{self.task_type}] {self.num_blocks}x {kind}(units={self.units}{extra}, "
            f"act={self.activation}) -> dropout={self.dropout} -> "
            f"optimizer={self.optimizer}(lr={self.lr})"
        )


def sample_random_spec(task_type: str, rng: np.random.Generator) -> ArchitectureSpec:
    encoder = str(rng.choice(ENCODER_CHOICES)) if task_type == "text" else None
    if task_type == "image" or (task_type == "text" and encoder == "conv1d"):
        kernel_size = int(rng.choice(KERNEL_CHOICES))
    else:
        kernel_size = None
    return ArchitectureSpec(
        task_type=task_type,
        num_blocks=int(rng.choice(NUM_BLOCKS_CHOICES)),
        units=int(rng.choice(UNITS_CHOICES)),
        kernel_size=kernel_size,
        activation=str(rng.choice(ACTIVATION_CHOICES)),
        dropout=float(rng.choice(DROPOUT_CHOICES)),
        optimizer=str(rng.choice(OPTIMIZER_CHOICES)),
        lr=float(rng.choice(LR_CHOICES)),
        embedding_dim=int(rng.choice(EMBEDDING_CHOICES)) if task_type == "text" else None,
        encoder=encoder,
    )


def enumerate_specs(task_type: str) -> Iterator[ArchitectureSpec]:
    """Every architecture in the search space for a task type. Small enough to
    score exhaustively at recommend time (tabular=576, image=1152, text=6912)."""
    kernels = KERNEL_CHOICES if task_type == "image" else [None]
    embeddings = EMBEDDING_CHOICES if task_type == "text" else [None]
    encoders = ENCODER_CHOICES if task_type == "text" else [None]
    for nb, u, k, act, do, opt, lr, emb, enc in itertools.product(
        NUM_BLOCKS_CHOICES, UNITS_CHOICES, kernels, ACTIVATION_CHOICES,
        DROPOUT_CHOICES, OPTIMIZER_CHOICES, LR_CHOICES, embeddings, encoders,
    ):
        # For text, kernel_size is only meaningful for conv1d; enumerate it there.
        if task_type == "text":
            k_vals = KERNEL_CHOICES if enc == "conv1d" else [None]
        else:
            k_vals = [k]
        for kv in k_vals:
            yield ArchitectureSpec(
                task_type=task_type, num_blocks=nb, units=u, kernel_size=kv,
                activation=act, dropout=do, optimizer=opt, lr=lr,
                embedding_dim=emb, encoder=enc,
            )


def decode_from_indices(task_type: str, indices: dict[str, int]) -> ArchitectureSpec:
    encoder = ENCODER_CHOICES[indices["encoder"]] if task_type == "text" else None
    if task_type == "image" or (task_type == "text" and encoder == "conv1d"):
        kernel_size = KERNEL_CHOICES[indices["kernel"]]
    else:
        kernel_size = None
    return ArchitectureSpec(
        task_type=task_type,
        num_blocks=NUM_BLOCKS_CHOICES[indices["num_blocks"]],
        units=UNITS_CHOICES[indices["units"]],
        kernel_size=kernel_size,
        activation=ACTIVATION_CHOICES[indices["activation"]],
        dropout=DROPOUT_CHOICES[indices["dropout"]],
        optimizer=OPTIMIZER_CHOICES[indices["optimizer"]],
        lr=LR_CHOICES[indices["lr"]],
        embedding_dim=EMBEDDING_CHOICES[indices["embedding"]] if task_type == "text" else None,
        encoder=encoder,
    )
