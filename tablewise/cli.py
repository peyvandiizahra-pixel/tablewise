"""tablewise app --data FOLDER          a folder of CSV / Excel / Parquet files, analysed on your computer
   tablewise app --bigquery PROJECT.DATASET
   tablewise demo                       the made-up finance dataset in examples/finance"""
import argparse
import re
import sys
from pathlib import Path


def _project(args):
    from .backends import BigQueryBackend, DuckDBBackend
    from .project import Project, Settings
    s = Settings(ai_budget_usd=args.budget, stale_data_alerts=args.stale_data_alerts)
    if args.bigquery:
        project, _, dataset = args.bigquery.partition(".")
        if not dataset:
            sys.exit("--bigquery needs PROJECT.DATASET")
        backend = BigQueryBackend(project, dataset)
        name = f"bq_{project}_{dataset}"
    else:
        data = Path(args.data).expanduser().resolve()
        name = re.sub(r"\W+", "_", data.name)
        ws = Path(args.workspace or Path("tablewise_workspace") / name)
        backend = DuckDBBackend(data, ws / "local.duckdb")
    ws = Path(args.workspace or Path("tablewise_workspace") / name)
    print(f"Data: {backend.label}   Decisions are saved in: {ws}")
    return Project(backend, ws, s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tablewise", description="From messy tables to trusted answers and alerts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("app", "demo"):
        sp = sub.add_parser(name, help="open the app" if name == "app" else "open the app on the demo data")
        if name == "app":
            src = sp.add_mutually_exclusive_group(required=True)
            src.add_argument("--data", help="a folder of CSV / Excel / Parquet files")
            src.add_argument("--bigquery", help="PROJECT.DATASET (needs: pip install 'tablewise[bigquery]')")
        sp.add_argument("--workspace", help="where decisions are saved (default: ./tablewise_workspace/<name>)")
        sp.add_argument("--port", type=int, default=None)
        sp.add_argument("--budget", type=float, default=0.50, help="AI spending limit per session, in USD")
        sp.add_argument("--stale-data-alerts", action="store_true", help="alert when new data stops arriving")
        sp.add_argument("--share", action="store_true", help="a public link for 72 hours. Never with real personal data.")
        sp.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        here = Path(__file__).resolve().parents[1] / "examples"
        data = here / "finance"
        if not any(data.glob("*.csv")):
            sys.path.insert(0, str(here))
            import make_finance_data
            make_finance_data.make(data)
        args.data, args.bigquery = str(data), None
    from .app import launch
    launch(_project(args), share=args.share, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
