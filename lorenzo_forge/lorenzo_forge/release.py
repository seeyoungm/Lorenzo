"""Phase 1: the recommendation -> releasable model bridge.

Forge's scorer only ranks architectures; a release needs an actually-trained,
packaged model. `release()` takes a real dataset, asks the scorer for its top-k
architectures, FULLY trains each on the data (early-stopped, many epochs -- not
the 8-epoch quick eval used during search), picks the empirically best, and
writes a SavedModel + model_card.json.

Design principle: the scorer is a FILTER (search space 576-1152 -> a handful of
promising candidates); the release step VERIFIES by real training and never
trusts the recommendation blindly."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from lorenzo_forge.candidate_trainer import build_model
from lorenzo_forge.meta_model import recommend
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.search_space import ArchitectureSpec

Split = tuple[np.ndarray, np.ndarray]

BUILTIN_IMAGE_DOMAINS = ("mnist", "fashion_mnist")
BUILTIN_TEXT_DOMAINS = ("imdb", "reuters")
BUILTIN_TABULAR_DOMAINS = ("tabular_v1",)
BUILTIN_DOMAINS = BUILTIN_IMAGE_DOMAINS + BUILTIN_TEXT_DOMAINS + BUILTIN_TABULAR_DOMAINS

TEXT_VOCAB_SIZE = 10000
TEXT_SEQ_LEN = 200


def load_domain(
    name: str, num_samples: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Load a full built-in domain for a release. Returns
    (X, y, task_type, vocab_size); vocab_size is 0 for images/tabular."""
    if name in BUILTIN_TABULAR_DOMAINS:
        from lorenzo_forge.datasets import build_tabular_v1

        X, y = build_tabular_v1()
        task_type, vocab = "tabular", 0
    elif name in BUILTIN_IMAGE_DOMAINS:
        loader = getattr(tf.keras.datasets, name)
        (x_train, y_train), (x_test, y_test) = loader.load_data()
        X = np.concatenate([x_train, x_test]).astype("float32") / 255.0
        y = np.concatenate([y_train, y_test]).ravel().astype("int64")
        if X.ndim == 3:
            X = X[..., None]
        task_type, vocab = "image", 0
    elif name in BUILTIN_TEXT_DOMAINS:
        loader = getattr(tf.keras.datasets, name)
        (x_train, y_train), (x_test, y_test) = loader.load_data(num_words=TEXT_VOCAB_SIZE)
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test]).astype("int64")
        X = np.clip(
            tf.keras.utils.pad_sequences(x, maxlen=TEXT_SEQ_LEN, padding="post", truncating="post"),
            0, TEXT_VOCAB_SIZE - 1,
        ).astype("int32")
        task_type, vocab = "text", TEXT_VOCAB_SIZE
    else:
        raise ValueError(f"unknown domain {name!r}; choices: {BUILTIN_DOMAINS}")

    if num_samples is not None and num_samples < len(X):
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(X), size=num_samples, replace=False)
        X, y = X[sel], y[sel]
    return X, y, task_type, vocab


@dataclass
class CandidateResult:
    spec: ArchitectureSpec
    predicted_accuracy: float
    val_accuracy: float
    test_accuracy: float
    num_params: int


def make_splits(
    X: np.ndarray, y: np.ndarray, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[Split, Split, Split]:
    """Stratified train / val / test split."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_frac, random_state=seed, stratify=y)
    val_ratio = val_frac / (1.0 - test_frac)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, random_state=seed, stratify=y_tmp)
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)


def full_train(
    spec: ArchitectureSpec,
    profile: DataProfile,
    train: Split,
    val: Split,
    test: Split,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 64,
) -> tuple[tf.keras.Model, float, float]:
    """Train an architecture to (early-stopped) convergence; return
    (model, val_accuracy, test_accuracy)."""
    model = build_model(spec, profile)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True, mode="max"
    )
    model.fit(
        train[0], train[1],
        validation_data=val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    val_acc = float(model.evaluate(val[0], val[1], verbose=0)[1])
    test_acc = float(model.evaluate(test[0], test[1], verbose=0)[1])
    return model, val_acc, test_acc


def release(
    name: str,
    scorer: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    out_dir: str | Path,
    data_source: str = "custom",
    top_k: int = 5,
    epochs: int = 40,
    seed: int = 0,
    noise_level: float = 0.1,
    vocab_size: int = 0,
    verbose: bool = True,
) -> dict:
    """Profile the data, train the scorer's top-k architectures fully, pick the
    best by test accuracy, and export SavedModel + model_card.json."""
    tf.keras.utils.set_random_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = DataProfile.from_arrays(X, y, task_type=task_type, noise_level=noise_level, vocab_size=vocab_size)
    train, val, test = make_splits(X, y, seed=seed)

    _, _, ranked = recommend(scorer, profile, top_k=top_k)
    if verbose:
        print(f"[{name}] scorer top-{top_k} candidates for {profile.task_type} "
              f"(input={profile.input_shape}, classes={profile.output_dim}):")

    results: list[CandidateResult] = []
    best_model: tf.keras.Model | None = None
    best_test = -1.0
    for i, (spec, predicted) in enumerate(ranked):
        t0 = time.time()
        model, val_acc, test_acc = full_train(spec, profile, train, val, test, epochs=epochs)
        res = CandidateResult(spec, float(predicted), val_acc, test_acc, int(model.count_params()))
        results.append(res)
        if verbose:
            print(f"  [{i + 1}/{top_k}] pred={predicted:.3f} val={val_acc:.3f} test={test_acc:.3f} "
                  f"params={res.num_params:,} ({time.time() - t0:.0f}s) :: {spec.describe()}")
        if test_acc > best_test:
            best_test, best_model = test_acc, model

    assert best_model is not None
    winner = max(results, key=lambda r: r.test_accuracy)

    model_path = out_dir / "model.keras"
    best_model.save(model_path)
    card = _build_model_card(name, data_source, profile, results, winner, epochs, seed)
    (out_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"[{name}] WINNER test={winner.test_accuracy:.3f} :: {winner.spec.describe()}")
        print(f"[{name}] saved -> {model_path}  +  model_card.json")
    return card


def _build_model_card(name, data_source, profile, results, winner, epochs, seed) -> dict:
    return {
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_type": profile.task_type,
        "data": {**profile.to_dict(), "source": data_source},
        "winner": {
            "architecture": winner.spec.to_dict(),
            "test_accuracy": round(winner.test_accuracy, 4),
            "val_accuracy": round(winner.val_accuracy, 4),
            "num_params": winner.num_params,
        },
        "candidates": [
            {
                "architecture": r.spec.describe(),
                "predicted_accuracy": round(r.predicted_accuracy, 4),
                "val_accuracy": round(r.val_accuracy, 4),
                "test_accuracy": round(r.test_accuracy, 4),
                "num_params": r.num_params,
            }
            for r in results
        ],
        "reproducibility": {
            "seed": seed,
            "epochs": epochs,
            "tensorflow": tf.__version__,
            "python": platform.python_version(),
        },
    }
