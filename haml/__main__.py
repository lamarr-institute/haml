import logging
import os
import sys

from argparse import ArgumentParser

import numpy as np

from . import parse_file
from .haml import RANDOM_VALUE_LIMIT
from .runtime import run_file


def remove_empty_lines(s: str):
    """Drop blank lines from generated YAML files."""
    return "\n".join([line for line in s.splitlines() if line.strip()])


def build_generate_parser() -> ArgumentParser:
    """Create the legacy YAML generation CLI parser."""
    parser = ArgumentParser()
    parser.add_argument("file", help="input HAML file")
    parser.add_argument("-d", "--directory", default=".", help="output directory for generated files")
    parser.add_argument("-a", "--all", action="store_true", help="generate all possible files")
    parser.add_argument("-s", "--sample", type=int, default=0, help="randomly sample files")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    parser.add_argument(
        "--rvlimit",
        type=int,
        default=RANDOM_VALUE_LIMIT,
        help="number of samples when running `all` on random variables with infinite support",
    )
    parser.add_argument("--keep-empty-lines", action="store_true")
    return parser


def build_run_parser() -> ArgumentParser:
    """Create the runtime execution CLI parser."""
    parser = ArgumentParser(prog="python -m haml run")
    parser.add_argument("file", help="input HAML file")
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="number of sampled configs to generate",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        nargs="*",
        default=None,
        help="CUDA device ids to schedule onto; omit the flag or pass it without values to leave CUDA_VISIBLE_DEVICES unset",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="number of CPU-only runs to execute in parallel when CUDA devices are not set",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="directory for generated configs; defaults to a folder below tempfile.gettempdir()",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="skip configs already present on disk and do not relaunch them",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="do not redirect tmux stdout/stderr into per-run log files",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    parser.add_argument(
        "--rvlimit",
        type=int,
        default=RANDOM_VALUE_LIMIT,
        help="number of samples when expanding random variables with infinite support",
    )
    parser.add_argument("--keep-empty-lines", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity",
    )
    return parser


def run_generate_mode(args) -> None:
    """Handle the legacy generate-only CLI mode."""
    obj = parse_file(args.file)

    import haml.haml as haml_module

    haml_module.RANDOM_VALUE_LIMIT = args.rvlimit

    rng = np.random.default_rng(args.seed)
    basename = os.path.basename(args.file).rsplit(".", 1)[0]

    os.makedirs(args.directory, exist_ok=True)
    if args.all:
        if input(f"Proceed to create {obj.num_combinations()} files? [y/n] ").lower() != "y":
            raise SystemExit(0)
        for i, s in enumerate(obj.all(random_state=rng)):
            with open(os.path.join(args.directory, f"{basename}_{i}.yml"), "w", encoding="utf-8") as handle:
                handle.write(s if args.keep_empty_lines else remove_empty_lines(s))

    elif args.sample > 0:
        i = 0
        for _ in range(args.sample):
            for content in obj.sample(random_state=rng):
                with open(os.path.join(args.directory, f"{basename}_{i}.yml"), "w", encoding="utf-8") as handle:
                    handle.write(content if args.keep_empty_lines else remove_empty_lines(content))
                i += 1


def run_executor_mode(args) -> None:
    """Handle the tmux-based runtime execution mode."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import haml.haml as haml_module

    haml_module.RANDOM_VALUE_LIMIT = args.rvlimit

    generated, output_dir = run_file(
        haml_file=args.file,
        num_samples=args.num_samples,
        cuda_devices=args.cuda_visible_devices or [],
        cpu_workers=args.cpu_workers,
        temp_dir=args.temp_dir,
        skip_existing=args.skip,
        enable_logging=not args.no_log_file,
        seed=args.seed,
        keep_empty_lines=args.keep_empty_lines,
    )
    logging.info("Prepared %d configs in %s", generated, output_dir)


def main(argv=None):
    """Dispatch between generation and execution CLI modes."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "run":
        parser = build_run_parser()
        args = parser.parse_args(argv[1:])
        run_executor_mode(args)
        return

    parser = build_generate_parser()
    args = parser.parse_args(argv)
    run_generate_mode(args)


if __name__ == "__main__":
    main()
