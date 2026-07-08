"""CLI: lorenzo-forge build-corpus | train-meta | recommend | release"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from lorenzo_forge.dataset_builder import build_training_corpus, load_training_corpus
from lorenzo_forge.meta_model import load_meta_model, recommend, save_meta_model, train_scorer_model
from lorenzo_forge.profile import DataProfile
from lorenzo_forge.release import BUILTIN_DOMAINS, load_domain, release

DEFAULT_CORPUS = "lorenzo_forge/data/meta_training_corpus.jsonl"
DEFAULT_MODEL = "lorenzo_forge/artifacts/meta_model.keras"


def _cmd_build_corpus(args: argparse.Namespace) -> None:
    kwargs = {}
    if args.domains is not None:
        kwargs["domains"] = tuple(d.strip() for d in args.domains.split(","))
    if args.base_corpus is not None:
        kwargs["base_corpus_path"] = args.base_corpus
    build_training_corpus(
        num_profiles=args.profiles,
        candidates_per_profile=args.candidates,
        search_epochs=args.search_epochs,
        seed=args.seed,
        out_path=args.out,
        **kwargs,
    )
    print(f"corpus written to {args.out}")


def _cmd_train_meta(args: argparse.Namespace) -> None:
    records = load_training_corpus(args.corpus)
    model = train_scorer_model(records, epochs=args.epochs, verbose=1 if args.verbose else 0)
    save_meta_model(model, args.out)
    print(f"meta-model saved to {args.out}")


def _cmd_recommend(args: argparse.Namespace) -> None:
    model = load_meta_model(args.model)
    if args.task == "tabular":
        input_shape = (args.input_dim,)
    else:
        input_shape = (args.side, args.side, args.channels)

    profile = DataProfile(
        task_type=args.task,
        input_shape=input_shape,
        output_dim=args.output_dim,
        num_samples=args.num_samples,
        class_balance_entropy=args.class_balance,
        noise_level=args.noise,
    )
    spec, predicted_acc, ranked = recommend(model, profile, top_k=args.top_k)
    print(spec.describe())
    print(
        json.dumps(
            {
                "architecture": spec.to_dict(),
                "predicted_accuracy": round(predicted_acc, 4),
                "top_k": [{"arch": s.describe(), "predicted_accuracy": round(p, 4)} for s, p in ranked],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _cmd_release(args: argparse.Namespace) -> None:
    scorer = load_meta_model(args.scorer)
    vocab_size = 0
    if args.domain:
        X, y, task_type, vocab_size = load_domain(args.domain, num_samples=args.num_samples, seed=args.seed)
        source = args.domain
    else:
        if not args.task:
            raise SystemExit("--task is required when using --data-npz")
        blob = np.load(args.data_npz)
        X, y, task_type = blob["X"], blob["y"], args.task
        vocab_size = int(blob["vocab_size"]) if "vocab_size" in blob else 0
        source = args.data_npz

    card = release(
        name=args.name,
        scorer=scorer,
        X=X,
        y=y,
        task_type=task_type,
        out_dir=args.out_dir,
        data_source=source,
        top_k=args.top_k,
        epochs=args.epochs,
        seed=args.seed,
        vocab_size=vocab_size,
    )
    print(json.dumps(card["winner"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lorenzo-forge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-corpus", help="run empirical architecture search across random profiles")
    p_build.add_argument("--profiles", type=int, default=80)
    p_build.add_argument("--candidates", type=int, default=8)
    p_build.add_argument("--search-epochs", type=int, default=4)
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--out", default=DEFAULT_CORPUS)
    p_build.add_argument("--domains", default=None,
                          help="comma-separated subset to (re)search, e.g. 'text'; default: all 4 domains")
    p_build.add_argument("--base-corpus", default=None,
                          help="existing corpus jsonl to carry over records for domains NOT in --domains "
                               "(incremental rebuild -- skips re-searching untouched domains)")
    p_build.set_defaults(func=_cmd_build_corpus)

    p_train = sub.add_parser("train-meta", help="train the meta-model on a corpus")
    p_train.add_argument("--corpus", default=DEFAULT_CORPUS)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--out", default=DEFAULT_MODEL)
    p_train.add_argument("--verbose", action="store_true")
    p_train.set_defaults(func=_cmd_train_meta)

    p_rec = sub.add_parser("recommend", help="recommend an architecture for a dataset profile")
    p_rec.add_argument("--model", default=DEFAULT_MODEL)
    p_rec.add_argument("--task", choices=["tabular", "image"], required=True)
    p_rec.add_argument("--input-dim", type=int, default=20, help="tabular feature count")
    p_rec.add_argument("--side", type=int, default=32, help="image height/width")
    p_rec.add_argument("--channels", type=int, default=3, help="image channels")
    p_rec.add_argument("--output-dim", type=int, required=True)
    p_rec.add_argument("--num-samples", type=int, required=True)
    p_rec.add_argument("--class-balance", type=float, default=1.0)
    p_rec.add_argument("--noise", type=float, default=0.2)
    p_rec.add_argument("--top-k", type=int, default=3)
    p_rec.set_defaults(func=_cmd_recommend)

    p_rel = sub.add_parser("release", help="train the scorer's top-k architectures fully and export the best as a release")
    p_rel.add_argument("--name", required=True, help='release name, e.g. "Lorenzo Image Alpha"')
    p_rel.add_argument("--scorer", default=DEFAULT_MODEL, help="trained scorer meta-model")
    src = p_rel.add_mutually_exclusive_group(required=True)
    src.add_argument("--domain", choices=list(BUILTIN_DOMAINS), help="built-in dataset (image: mnist/fashion_mnist, text: imdb/reuters)")
    src.add_argument("--data-npz", help="custom .npz with arrays X and y (use with --task)")
    p_rel.add_argument("--task", choices=["tabular", "image", "text", "timeseries"], help="task type for --data-npz")
    p_rel.add_argument("--num-samples", type=int, default=8000, help="subsample size for --domain (None-like: use all if larger)")
    p_rel.add_argument("--top-k", type=int, default=5)
    p_rel.add_argument("--epochs", type=int, default=40)
    p_rel.add_argument("--seed", type=int, default=0)
    p_rel.add_argument("--out-dir", default="lorenzo_forge/releases/latest")
    p_rel.set_defaults(func=_cmd_release)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
