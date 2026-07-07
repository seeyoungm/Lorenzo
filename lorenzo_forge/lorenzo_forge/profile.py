"""DataProfile: the fixed-length description of a dataset that the meta-model
conditions its architecture prediction on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "is_tabular",
    "is_image",
    "is_text",
    "is_timeseries",
    "log_input_size",
    "log_output_dim",
    "log_num_samples",
    "class_balance_entropy",
    "noise_level",
    "log_vocab_size",
]


@dataclass(frozen=True)
class DataProfile:
    task_type: str  # "tabular" | "image" | "text"
    input_shape: tuple[int, ...]
    output_dim: int
    num_samples: int
    class_balance_entropy: float  # 0..1, 1 = perfectly balanced classes
    noise_level: float  # 0..1, higher = noisier / harder signal
    vocab_size: int = 0  # text only: size of the token vocabulary

    @property
    def input_size(self) -> int:
        size = 1
        for d in self.input_shape:
            size *= d
        return size

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                1.0 if self.task_type == "tabular" else 0.0,
                1.0 if self.task_type == "image" else 0.0,
                1.0 if self.task_type == "text" else 0.0,
                1.0 if self.task_type == "timeseries" else 0.0,
                np.log10(self.input_size + 1),
                np.log10(self.output_dim + 1),
                np.log10(self.num_samples + 1),
                self.class_balance_entropy,
                self.noise_level,
                np.log10(self.vocab_size + 1),
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "input_shape": list(self.input_shape),
            "output_dim": self.output_dim,
            "num_samples": self.num_samples,
            "class_balance_entropy": self.class_balance_entropy,
            "noise_level": self.noise_level,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataProfile":
        return cls(
            task_type=d["task_type"],
            input_shape=tuple(d["input_shape"]),
            output_dim=d["output_dim"],
            num_samples=d["num_samples"],
            class_balance_entropy=d["class_balance_entropy"],
            noise_level=d["noise_level"],
            vocab_size=d.get("vocab_size", 0),
        )

    @classmethod
    def from_arrays(
        cls, X: np.ndarray, y: np.ndarray, task_type: str, noise_level: float = 0.3, vocab_size: int = 0
    ) -> "DataProfile":
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        k = len(classes)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0
        return cls(
            task_type=task_type,
            input_shape=tuple(X.shape[1:]),
            output_dim=int(k),
            num_samples=int(X.shape[0]),
            class_balance_entropy=entropy,
            noise_level=float(noise_level),
            vocab_size=int(vocab_size),
        )
