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
# Sequence axes: encoder is used by text + timeseries; embedding is text-only
# (timeseries feeds real-valued sequences straight in, no token embedding).
EMBEDDING_CHOICES = [16, 32, 64]
ENCODER_CHOICES = ["lstm", "gru", "conv1d", "bilstm", "bigru"]
ENCODER_TASKS = ("text", "timeseries")
EMBEDDING_TASKS = ("text",)
# Modernization axes (Track 2): skip connections and pooling style. Meaningful
# for tabular Dense / image Conv2D / conv1d-encoded text+timeseries; present
# but ignored elsewhere (same convention as kernel_size being unused for
# non-conv tasks) so the one-hot layout stays uniform across task types.
BLOCK_STYLE_CHOICES = ["plain", "residual"]
POOL_STYLE_CHOICES = ["standard", "gap"]

# Search-space constraints (restrictions only -- they narrow which axis
# combinations are legal but add no new axes, so HEADS/ARCH_FEATURE_DIM and the
# scorer encoding are unchanged; only the corpus/scorer need rebuilding to
# reflect the narrower distribution). See is_valid_spec for the rationale.
RECURRENT_ENCODERS = ("lstm", "gru", "bilstm", "bigru")
DEEP_STACK_MIN_BLOCKS = 3      # 3+ recurrent blocks is where high-lr collapse appears
DEEP_RECURRENT_MAX_LR = 1e-3   # deep recurrent stacks may not use lr=1e-2
IMAGE_RESIDUAL_MAX_UNITS = 128  # image residual conv2d capped below the costly 256-wide variant

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
    "block_style": len(BLOCK_STYLE_CHOICES),
    "pool_style": len(POOL_STYLE_CHOICES),
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
    block_style: str = "plain"  # plain | residual
    pool_style: str = "standard"  # standard (flatten/max) | gap

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
            "block_style": BLOCK_STYLE_CHOICES.index(self.block_style),
            "pool_style": POOL_STYLE_CHOICES.index(self.pool_style),
        }

    def to_dict(self) -> dict:
        if self.task_type == "image":
            blocks = [
                {"type": "conv2d", "filters": self.units, "kernel_size": self.kernel_size, "activation": self.activation}
                for _ in range(self.num_blocks)
            ]
        elif self.task_type in ("text", "timeseries"):
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
            "block_style": self.block_style,
            "pool_style": self.pool_style,
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
        block_mult = 2 if self.block_style == "residual" else 1
        return self.num_blocks * self.units * conv_cost * block_mult

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
            "block_style": self.block_style,
            "pool_style": self.pool_style,
        }

    @classmethod
    def from_raw_dict(cls, d: dict) -> "ArchitectureSpec":
        fields = {
            "task_type", "num_blocks", "units", "kernel_size", "activation",
            "dropout", "optimizer", "lr", "embedding_dim", "encoder",
            "block_style", "pool_style",
        }
        return cls(**{k: v for k, v in d.items() if k in fields})

    def _style_suffix(self, pool_applicable: bool) -> str:
        """block_style/pool_style are only meaningful for tabular Dense, image
        Conv2D, and conv1d-encoded text/timeseries (BatchNorm is applied
        unconditionally there and isn't itself worth describing)."""
        bits = []
        if self.block_style == "residual":
            bits.append("residual")
        if pool_applicable and self.pool_style == "gap":
            bits.append("gap")
        return f" [{'+'.join(bits)}]" if bits else ""

    def describe(self) -> str:
        if self.task_type in ENCODER_TASKS:
            kind = self.encoder if self.encoder else "seq"
            extra = f", kernel={self.kernel_size}" if self.encoder == "conv1d" else ""
            prefix = f"embed{self.embedding_dim} -> " if self.embedding_dim is not None else ""
            suffix = self._style_suffix(pool_applicable=self.encoder == "conv1d") if self.encoder == "conv1d" else ""
            return (
                f"[{self.task_type}] {prefix}{self.num_blocks}x {kind}(units={self.units}{extra}) "
                f"-> dropout={self.dropout} -> optimizer={self.optimizer}(lr={self.lr}){suffix}"
            )
        kind = "Conv2D" if self.task_type == "image" else "Dense"
        extra = f", kernel={self.kernel_size}" if self.task_type == "image" else ""
        suffix = self._style_suffix(pool_applicable=self.task_type == "image")
        return (
            f"[{self.task_type}] {self.num_blocks}x {kind}(units={self.units}{extra}, "
            f"act={self.activation}) -> dropout={self.dropout} -> "
            f"optimizer={self.optimizer}(lr={self.lr}){suffix}"
        )


def is_valid_spec(spec: ArchitectureSpec) -> bool:
    """Whether an architecture is inside the (constrained) search space.

    Both rules exclude architectures that empirically waste search compute
    without ever winning a release:

    - Deep recurrent stacks at lr=1e-2 blow up (gradient explosion -> a
      dead network stuck at chance-level accuracy). clipnorm=1.0 rescued
      shallow bidirectional stacks but not 3+ block ones, so those are
      excluded from the space outright.
    - Wide (256-unit) residual Conv2D is ~50min per candidate on full-res
      images and never won, so image residual blocks are capped at 128 units.
    """
    if (
        spec.encoder in RECURRENT_ENCODERS
        and spec.num_blocks >= DEEP_STACK_MIN_BLOCKS
        and spec.lr > DEEP_RECURRENT_MAX_LR
    ):
        return False
    if (
        spec.task_type == "image"
        and spec.block_style == "residual"
        and spec.units > IMAGE_RESIDUAL_MAX_UNITS
    ):
        return False
    return True


def sample_random_spec(task_type: str, rng: np.random.Generator) -> ArchitectureSpec:
    # Rejection-sample so the sampler and the exhaustive enumerator draw from
    # exactly the same constrained space. The excluded fraction is small, so
    # this terminates quickly; the bound is just a safety net.
    spec = None
    for _ in range(1000):
        encoder = str(rng.choice(ENCODER_CHOICES)) if task_type in ENCODER_TASKS else None
        if task_type == "image" or (task_type in ENCODER_TASKS and encoder == "conv1d"):
            kernel_size = int(rng.choice(KERNEL_CHOICES))
        else:
            kernel_size = None
        spec = ArchitectureSpec(
            task_type=task_type,
            num_blocks=int(rng.choice(NUM_BLOCKS_CHOICES)),
            units=int(rng.choice(UNITS_CHOICES)),
            kernel_size=kernel_size,
            activation=str(rng.choice(ACTIVATION_CHOICES)),
            dropout=float(rng.choice(DROPOUT_CHOICES)),
            optimizer=str(rng.choice(OPTIMIZER_CHOICES)),
            lr=float(rng.choice(LR_CHOICES)),
            embedding_dim=int(rng.choice(EMBEDDING_CHOICES)) if task_type in EMBEDDING_TASKS else None,
            encoder=encoder,
            block_style=str(rng.choice(BLOCK_STYLE_CHOICES)),
            pool_style=str(rng.choice(POOL_STYLE_CHOICES)),
        )
        if is_valid_spec(spec):
            return spec
    return spec


def enumerate_specs(task_type: str) -> Iterator[ArchitectureSpec]:
    """Every (valid) architecture in the search space for a task type. Small
    enough to score exhaustively at recommend time. Post-constraint sizes:
    tabular=2304, image=4032, text=36864, timeseries=12288 (see is_valid_spec
    for what the constraints exclude)."""
    kernels = KERNEL_CHOICES if task_type == "image" else [None]
    embeddings = EMBEDDING_CHOICES if task_type in EMBEDDING_TASKS else [None]
    encoders = ENCODER_CHOICES if task_type in ENCODER_TASKS else [None]
    for nb, u, k, act, do, opt, lr, emb, enc, bstyle, pstyle in itertools.product(
        NUM_BLOCKS_CHOICES, UNITS_CHOICES, kernels, ACTIVATION_CHOICES,
        DROPOUT_CHOICES, OPTIMIZER_CHOICES, LR_CHOICES, embeddings, encoders,
        BLOCK_STYLE_CHOICES, POOL_STYLE_CHOICES,
    ):
        # For sequence encoders, kernel_size is only meaningful for conv1d.
        if task_type in ENCODER_TASKS:
            k_vals = KERNEL_CHOICES if enc == "conv1d" else [None]
        else:
            k_vals = [k]
        for kv in k_vals:
            spec = ArchitectureSpec(
                task_type=task_type, num_blocks=nb, units=u, kernel_size=kv,
                activation=act, dropout=do, optimizer=opt, lr=lr,
                embedding_dim=emb, encoder=enc,
                block_style=bstyle, pool_style=pstyle,
            )
            if is_valid_spec(spec):
                yield spec


def decode_from_indices(task_type: str, indices: dict[str, int]) -> ArchitectureSpec:
    encoder = ENCODER_CHOICES[indices["encoder"]] if task_type in ENCODER_TASKS else None
    if task_type == "image" or (task_type in ENCODER_TASKS and encoder == "conv1d"):
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
        embedding_dim=EMBEDDING_CHOICES[indices["embedding"]] if task_type in EMBEDDING_TASKS else None,
        encoder=encoder,
        block_style=BLOCK_STYLE_CHOICES[indices["block_style"]],
        pool_style=POOL_STYLE_CHOICES[indices["pool_style"]],
    )
