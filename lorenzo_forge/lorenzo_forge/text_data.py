"""Real text (sequence) profiles for the corpus — the third domain.

Sequence classification is a genuinely different model family (Embedding ->
recurrent/conv1d encoder -> dense), so it exercises the text-only search axes
(embedding_dim, encoder type). Profiles are drawn from real IMDB (binary
sentiment) and Reuters (multi-class topic), varied by vocab size, sequence
length, class subset (Reuters), and sample count for diversity."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lorenzo_forge.profile import DataProfile

SOURCES = ("imdb", "reuters")
VOCAB_CHOICES = (5000, 10000, 20000)
SEQ_LEN_CHOICES = (100, 200)  # keep short: recurrent candidates train slowly on long sequences
NUM_SAMPLES_RANGE = (1500, 3000)  # text needs enough data for a good arch to separate from a weak one
REUTERS_CLASS_RANGE = (3, 47)  # use 3..46 of the most frequent topics (46 = real Reuters release shape)

_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _load_source(name: str, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    key = (name, vocab_size)
    if key not in _CACHE:
        loader = getattr(tf.keras.datasets, name)
        (x_train, y_train), (x_test, y_test) = loader.load_data(num_words=vocab_size)
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test]).astype("int64")
        _CACHE[key] = (x, y)
    return _CACHE[key]


def sample_text_profile(
    rng: np.random.Generator,
) -> tuple[DataProfile, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    source = str(rng.choice(SOURCES))
    vocab_size = int(rng.choice(VOCAB_CHOICES))
    x_all, y_all = _load_source(source, vocab_size)

    if source == "reuters":
        classes, counts = np.unique(y_all, return_counts=True)
        top = classes[np.argsort(-counts)]
        n_classes = int(rng.integers(*REUTERS_CLASS_RANGE))
        chosen = top[:n_classes]
        remap = {int(c): i for i, c in enumerate(chosen)}
        mask = np.isin(y_all, chosen)
        x_sub, y_sub = x_all[mask], np.array([remap[int(v)] for v in y_all[mask]], dtype="int64")
    else:  # imdb: binary
        x_sub, y_sub = x_all, y_all

    num_samples = int(min(rng.integers(*NUM_SAMPLES_RANGE), len(x_sub)))
    sel = rng.choice(len(x_sub), size=num_samples, replace=False)
    x_sub, y_sub = x_sub[sel], y_sub[sel]

    seq_len = int(rng.choice(SEQ_LEN_CHOICES))
    x_pad = tf.keras.utils.pad_sequences(x_sub, maxlen=seq_len, padding="post", truncating="post")
    x_pad = np.clip(x_pad, 0, vocab_size - 1).astype("int32")

    classes, counts = np.unique(y_sub, return_counts=True)
    probs = counts / counts.sum()
    k = len(classes)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(k)) if k > 1 else 1.0

    profile = DataProfile(
        task_type="text",
        input_shape=(seq_len,),
        output_dim=k,
        num_samples=len(x_pad),
        class_balance_entropy=entropy,
        noise_level=0.0,
        vocab_size=vocab_size,
    )

    split = int(len(x_pad) * 0.75)
    perm = rng.permutation(len(x_pad))
    tr, va = perm[:split], perm[split:]
    return profile, (x_pad[tr], y_sub[tr], x_pad[va], y_sub[va])
