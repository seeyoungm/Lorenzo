"""Discrete architecture search space and the ArchitectureSpec encoding used
both as search-candidate genome and as the meta-model's prediction target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

NUM_BLOCKS_CHOICES = [1, 2, 3, 4]
UNITS_CHOICES = [32, 64, 128, 256]
KERNEL_CHOICES = [3, 5]
ACTIVATION_CHOICES = ["relu", "tanh"]
DROPOUT_CHOICES = [0.0, 0.2, 0.4]
OPTIMIZER_CHOICES = ["adam", "sgd"]
LR_CHOICES = [1e-2, 1e-3, 1e-4]

# Fixed head order/sizes shared by the search space and the meta-model output heads.
HEADS = {
    "num_blocks": len(NUM_BLOCKS_CHOICES),
    "units": len(UNITS_CHOICES),
    "kernel": len(KERNEL_CHOICES),
    "activation": len(ACTIVATION_CHOICES),
    "dropout": len(DROPOUT_CHOICES),
    "optimizer": len(OPTIMIZER_CHOICES),
    "lr": len(LR_CHOICES),
}


@dataclass(frozen=True)
class ArchitectureSpec:
    task_type: str  # "tabular" | "image"
    num_blocks: int
    units: int
    activation: str
    dropout: float
    optimizer: str
    lr: float
    kernel_size: Optional[int] = None  # only meaningful when task_type == "image"

    def label_indices(self) -> dict[str, int]:
        return {
            "num_blocks": NUM_BLOCKS_CHOICES.index(self.num_blocks),
            "units": UNITS_CHOICES.index(self.units),
            "kernel": KERNEL_CHOICES.index(self.kernel_size) if self.kernel_size is not None else 0,
            "activation": ACTIVATION_CHOICES.index(self.activation),
            "dropout": DROPOUT_CHOICES.index(self.dropout),
            "optimizer": OPTIMIZER_CHOICES.index(self.optimizer),
            "lr": LR_CHOICES.index(self.lr),
        }

    def to_dict(self) -> dict:
        if self.task_type == "image":
            blocks = [
                {"type": "conv2d", "filters": self.units, "kernel_size": self.kernel_size, "activation": self.activation}
                for _ in range(self.num_blocks)
            ]
        else:
            blocks = [
                {"type": "dense", "units": self.units, "activation": self.activation}
                for _ in range(self.num_blocks)
            ]
        return {
            "task_type": self.task_type,
            "blocks": blocks,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "learning_rate": self.lr,
        }

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
        }

    @classmethod
    def from_raw_dict(cls, d: dict) -> "ArchitectureSpec":
        return cls(**d)

    def describe(self) -> str:
        kind = "Conv2D" if self.task_type == "image" else "Dense"
        extra = f", kernel={self.kernel_size}" if self.task_type == "image" else ""
        return (
            f"[{self.task_type}] {self.num_blocks}x {kind}(units={self.units}{extra}, "
            f"act={self.activation}) -> dropout={self.dropout} -> "
            f"optimizer={self.optimizer}(lr={self.lr})"
        )


def sample_random_spec(task_type: str, rng: np.random.Generator) -> ArchitectureSpec:
    return ArchitectureSpec(
        task_type=task_type,
        num_blocks=int(rng.choice(NUM_BLOCKS_CHOICES)),
        units=int(rng.choice(UNITS_CHOICES)),
        kernel_size=int(rng.choice(KERNEL_CHOICES)) if task_type == "image" else None,
        activation=str(rng.choice(ACTIVATION_CHOICES)),
        dropout=float(rng.choice(DROPOUT_CHOICES)),
        optimizer=str(rng.choice(OPTIMIZER_CHOICES)),
        lr=float(rng.choice(LR_CHOICES)),
    )


def decode_from_indices(task_type: str, indices: dict[str, int]) -> ArchitectureSpec:
    return ArchitectureSpec(
        task_type=task_type,
        num_blocks=NUM_BLOCKS_CHOICES[indices["num_blocks"]],
        units=UNITS_CHOICES[indices["units"]],
        kernel_size=KERNEL_CHOICES[indices["kernel"]] if task_type == "image" else None,
        activation=ACTIVATION_CHOICES[indices["activation"]],
        dropout=DROPOUT_CHOICES[indices["dropout"]],
        optimizer=OPTIMIZER_CHOICES[indices["optimizer"]],
        lr=LR_CHOICES[indices["lr"]],
    )
