import logging
import os
import sys

from argparse import ArgumentParser

import numpy as np

from . import parse_file
from .haml import RANDOM_VALUE_LIMIT
from .runtime import load_runtime_config, run_file


def remove_empty_lines(s: str):
    """Drop blank lines from generated YAML files."""
    return "\n".join([line for line in s.splitlines() if line.strip()])


def build_generate_parser() -> ArgumentParser:
    """Create the legacy YAML generation CLI parser."""
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", help="input HAML file")
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
    parser.add_argument("files", nargs="+", help="input HAML file(s)")
    parser.add_argument(
        "-r",
        "--runtime-config",
        required=True,
        help="YAML file containing runtime-only execution settings such as script and CUDA devices",
    )
    return parser


def run_generate_mode(args) -> None:
    """Handle the legacy generate-only CLI mode."""
    import haml.haml as haml_module

    haml_module.RANDOM_VALUE_LIMIT = args.rvlimit

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.directory, exist_ok=True)

    parsed_files = [(haml_file, parse_file(haml_file)) for haml_file in args.files]

    if args.all:
        total_combinations = sum(obj.num_combinations() for _, obj in parsed_files)
        if input(f"Proceed to create {total_combinations} files? [y/n] ").lower() != "y":
            raise SystemExit(0)
        for haml_file, obj in parsed_files:
            basename = os.path.basename(haml_file).rsplit(".", 1)[0]
            for i, s in enumerate(obj.all(random_state=rng)):
                with open(os.path.join(args.directory, f"{basename}_{i}.yml"), "w", encoding="utf-8") as handle:
                    handle.write(s if args.keep_empty_lines else remove_empty_lines(s))

    elif args.sample > 0:
        for haml_file, obj in parsed_files:
            basename = os.path.basename(haml_file).rsplit(".", 1)[0]
            i = 0
            for _ in range(args.sample):
                for content in obj.sample(random_state=rng):
                    with open(os.path.join(args.directory, f"{basename}_{i}.yml"), "w", encoding="utf-8") as handle:
                        handle.write(content if args.keep_empty_lines else remove_empty_lines(content))
                    i += 1


def run_executor_mode(args) -> None:
    """Handle the tmux-based runtime execution mode."""
    runtime_config = load_runtime_config(args.runtime_config)
    logging.basicConfig(
        level=getattr(logging, runtime_config.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import haml.haml as haml_module

    if runtime_config.rvlimit is not None:
        haml_module.RANDOM_VALUE_LIMIT = runtime_config.rvlimit

    total_generated = 0
    for haml_file in args.files:
        generated, output_dir = run_file(
            haml_file=haml_file,
            runtime_config_path=args.runtime_config,
        )
        total_generated += generated
        logging.info("Prepared %d configs from %s in %s", generated, haml_file, output_dir)
    logging.info("Prepared %d configs total from %d HAML file(s)", total_generated, len(args.files))


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
