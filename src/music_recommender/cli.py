"""Command-line entry point: ``music-rec <command>``."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from music_recommender import data


def _cmd_download(args: argparse.Namespace) -> None:
    path = data.download(force=args.force)
    print(f"Raw data at {path}")


def _cmd_build(args: argparse.Namespace) -> None:
    data.download()
    report = data.build()
    print(report.to_frame().to_string(index=False))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from music_recommender import reporting

    results = reporting.run(data.ensure_tracks(), n_tasks=args.n_tasks)
    cols = ["hit_rate", "ndcg", "genre_precision", "diversity", "coverage"]
    print(results[cols].round(3).to_string())
    print("\nFull report written to reports/evaluation.md")


def _cmd_recommend(args: argparse.Namespace) -> None:
    from music_recommender import ContentRecommender

    rec = ContentRecommender(diversity=args.diversity).fit(data.ensure_tracks())
    seeds = []
    for query in args.query:
        hits = rec.search(query, limit=1)
        if hits.empty:
            raise SystemExit(f"No track matches '{query}'")
        seeds.append(hits.iloc[0])
        print(f"Seed: {hits.iloc[0].track_name} - {hits.iloc[0].artists}")
    out = rec.recommend(
        [s.track_id for s in seeds], k=args.k, exclude_seed_artists=args.new_artists
    )
    with pd.option_context("display.max_colwidth", 40, "display.width", 160):
        print(out[["similarity", "track_name", "artists", "genres", "popularity"]].to_string())


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _unit_float(value: str) -> float:
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return x


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="music-rec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download the raw dataset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_download)

    sub.add_parser("build", help="clean raw data into data/processed").set_defaults(func=_cmd_build)

    p = sub.add_parser("evaluate", help="run offline evaluation and write reports/")
    p.add_argument("--n-tasks", type=_positive_int, default=1000)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("recommend", help="recommend from one or more seed songs")
    p.add_argument("query", nargs="+", help='seed song searches, e.g. "blinding lights"')
    p.add_argument("-k", type=_positive_int, default=10)
    p.add_argument("--diversity", type=_unit_float, default=0.3)
    p.add_argument("--new-artists", action="store_true", help="exclude the seed artists")
    p.set_defaults(func=_cmd_recommend)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
