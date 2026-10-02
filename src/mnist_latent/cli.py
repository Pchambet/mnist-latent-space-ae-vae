"""Command line entry point: ``uv run mnist-latent {data,run,figures,report}``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mnist-latent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download MNIST into data/raw (cached)")
    run = sub.add_parser("run", help="train (cached), evaluate, write results/ and figures")
    run.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    run.add_argument("--retrain", action="store_true", help="ignore cached checkpoints")
    run.add_argument("--epochs", type=int, default=None, help="override the epoch budget")
    run.add_argument("--threads", type=int, default=None, help="cap the CPU threads torch uses")
    sub.add_parser("figures", help="redraw docs/figures from results/ and cached arrays")
    sub.add_parser("report", help="build site/index.html and refresh the README tables")
    args = parser.parse_args(argv)

    if args.cmd == "data":
        from .data import download

        print(f"MNIST cached in {download()}")
    elif args.cmd == "run":
        from dataclasses import replace

        from .experiments import Config, run_all
        from .figures import draw_all

        cfg = Config() if args.epochs is None else replace(Config(), epochs=args.epochs)
        run_all(cfg, device=args.device, retrain=args.retrain, threads=args.threads)
        draw_all()
    elif args.cmd == "figures":
        from .figures import draw_all

        draw_all()
    elif args.cmd == "report":
        from .report import build

        build()


if __name__ == "__main__":
    main()
