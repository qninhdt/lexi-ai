"""Explicit generated-database migration CLI: python -m lexi_ai.migrations."""

import argparse

from . import inspect_current, inspect_head, upgrade_to_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upgrade", "head", "current"))
    parser.add_argument("db_url", nargs="?")
    parser.add_argument("--db-schema")
    args = parser.parse_args()
    if args.action == "head":
        if args.db_url or args.db_schema:
            parser.error("head does not accept database arguments")
        print(inspect_head())
        return
    if not args.db_url:
        parser.error(f"{args.action} requires an explicit generated-database URL")
    if args.action == "upgrade":
        upgrade_to_head(args.db_url, db_schema=args.db_schema)
        print("Dictionary schema is at head.")
    else:
        print(inspect_current(args.db_url, db_schema=args.db_schema))


if __name__ == "__main__":
    main()
