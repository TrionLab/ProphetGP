from __future__ import annotations

import argparse
import json

from prophet_gp.config import load_config
from prophet_gp.data.dataset import ReactionDatasetService
from prophet_gp.pipeline.trainer import ProphetGPPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prophet-gp")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train GP surrogate from reaction dataset.")
    train.add_argument("--data", required=True, help="CSV path.")
    train.add_argument("--config", required=True, help="YAML config path.")

    suggest = sub.add_parser("suggest", help="Suggest next experiments using BO.")
    suggest.add_argument("--data", required=True, help="CSV path.")
    suggest.add_argument("--config", required=True, help="YAML config path.")
    suggest.add_argument("--n-candidates", type=int, default=5)

    append = sub.add_parser("append", help="Append newly observed batch dataset.")
    append.add_argument("--base-data", required=True)
    append.add_argument("--new-data", required=True)
    append.add_argument("--out", required=True)
    append.add_argument("--config", required=True)

    list_feat = sub.add_parser("list-featurisers", help="Show available featurisers.")
    list_feat.add_argument("--config", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "append":
        ds = ReactionDatasetService(config.data)
        merged = ds.append_csv(args.base_data, args.new_data, args.out)
        print(json.dumps({"rows": int(len(merged)), "output": args.out}, indent=2))
        return

    pipeline = ProphetGPPipeline(config)
    if args.command == "list-featurisers":
        print(json.dumps({"featurisers": pipeline.featurizers.available()}, indent=2))
        return

    artifacts = pipeline.train_from_csv(args.data)
    if args.command == "train":
        print(
            json.dumps(
                {
                    "status": "trained",
                    "rows": int(artifacts.x_train.shape[0]),
                    "features": int(artifacts.x_train.shape[1]),
                },
                indent=2,
            )
        )
        return

    if args.command == "suggest":
        suggestions = pipeline.suggest_next_experiments(
            artifacts,
            n_candidates=args.n_candidates,
        )
        print(json.dumps({"candidates": suggestions.tolist()}, indent=2))
        return

    raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
