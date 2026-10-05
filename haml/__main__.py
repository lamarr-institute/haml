import logging
import hashlib
import os
import signal
import sys

from argparse import ArgumentParser, REMAINDER
from pathlib import Path

import numpy as np

from . import parse_file
from . import haml as haml_module
from .haml import RANDOM_VALUE_LIMIT
from .runtime import default_temp_dir, load_runtime_config, run_files


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
    parser.add_argument("files", nargs="+", help="HAML files, YAML configs, or directories of YAML configs")
    parser.add_argument(
        "-r",
        "--runtime-config",
        required=True,
        help="YAML file containing runtime-only execution settings such as script and CUDA devices",
    )
    parser.add_argument("--cuda-visible-devices", nargs="+", help="Override the GPU slots in the runtime YAML.")
    parser.add_argument("--backend", choices=["tmux", "direct", "slurm"], help="Override the execution backend.")
    parser.add_argument("--temp-dir", help="Override the directory containing runs/<id>/.")
    parser.add_argument("--script", help="Override the execution command (shell-style argument string).")
    parser.add_argument("--skip", action="store_true", default=None, help="Skip existing run directories and watch matching live workers.")
    parser.add_argument("--args", nargs=REMAINDER, help="Extra job arguments; must be the final runtime option.")
    return parser


def run_generate_mode(args) -> None:
    """Handle the legacy generate-only CLI mode."""
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


def prepare_inputs(files, config):
    """Expand HAML in the CLI; the executor receives only concrete config paths."""
    inputs = []
    output = Path(config["temp_dir"]).expanduser() / "configs"
    rng = np.random.default_rng(config.get("seed"))
    if "rvlimit" in config:
        haml_module.RANDOM_VALUE_LIMIT = config["rvlimit"]
    for source in map(Path, files):
        if source.is_dir() or source.suffix.lower() in {".yaml", ".yml"}:
            inputs.append(str(source))
            continue
        obj = parse_file(str(source))
        samples = config.get("num_samples")
        contents = obj.all(random_state=rng) if samples is None else (content for _ in range(samples) for content in obj.sample(random_state=rng))
        output.mkdir(parents=True, exist_ok=True)
        for content in contents:
            if not config.get("keep_empty_lines", False):
                content = remove_empty_lines(content)
            path = output / (hashlib.sha256(content.encode("utf-8")).hexdigest() + ".yaml")
            path.write_text(content, encoding="utf-8")
            if str(path) not in inputs:
                inputs.append(str(path))
    return inputs


def run_executor_mode(args) -> None:
    config = load_runtime_config(args.runtime_config)
    for key, value in {"cuda_visible_devices": args.cuda_visible_devices, "backend": args.backend, "temp_dir": args.temp_dir, "script": args.script, "args": args.args, "skip": args.skip}.items():
        if value is not None:
            config[key] = value
    config.setdefault("temp_dir", str(default_temp_dir(args.files[0])))
    logging.basicConfig(level=config.get("log_level", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    completed, output_dirs = run_files(prepare_inputs(args.files, config), runtime=config)
    logging.info("Finished %d runs; outputs: %s", completed, ", ".join(map(str, output_dirs)))


def terminate_runtime(number, frame):
    """Let the runtime's finally blocks clean up direct workers on SIGTERM."""
    raise SystemExit(128 + number)


def main(argv=None):
    """Dispatch between generation and execution CLI modes."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "run":
        parser = build_run_parser()
        args = parser.parse_args(argv[1:])
        previous = signal.signal(signal.SIGTERM, terminate_runtime)
        try:
            run_executor_mode(args)
        finally:
            signal.signal(signal.SIGTERM, previous)
        return

    parser = build_generate_parser()
    args = parser.parse_args(argv)
    run_generate_mode(args)


if __name__ == "__main__":
    main()
