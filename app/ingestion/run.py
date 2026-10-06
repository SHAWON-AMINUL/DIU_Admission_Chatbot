"""CLI entrypoint for the ingest pipeline.

Testing this slice must never require the Backend teammate's verify endpoint
to exist. Their handler, when it lands, calls the same ingest_post() function
this wraps.

    uv run python -m app.ingestion.run --post-id 3
    uv run python -m app.ingestion.run --all-eligible
"""

import argparse
import json
import logging
import sys

from app.ingestion.pipeline import ingest_all_eligible, ingest_post, reclaim_stale


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DIU admission ingest pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--post-id", type=int, help="ingest a single post")
    group.add_argument(
        "--all-eligible",
        action="store_true",
        help="ingest every approved+verified post not yet ingested",
    )
    parser.add_argument(
        "--reclaim-after",
        type=int,
        default=30,
        metavar="MINUTES",
        help="reset posts stuck in 'AI training in progress' older than this "
        "(default 30; a killed worker leaves its post claimed forever)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    reclaim_stale(args.reclaim_after)

    results = [ingest_post(args.post_id)] if args.post_id else ingest_all_eligible()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    failed = [r for r in results if r.get("status") == "AI training failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
