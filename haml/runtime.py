import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from tqdm import tqdm

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, IO, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from . import haml as haml_module
from .haml import parse_file

LOG = logging.getLogger(__name__)


def remove_empty_lines(s: str) -> str:
    """Drop blank lines from generated YAML content."""
    return "\n".join(line for line in s.splitlines() if line.strip())


def default_temp_dir(source_file: str) -> Path:
    """Return the default directory used to store generated run configs."""
    stem = Path(source_file).stem
    return Path(tempfile.gettempdir()) / "haml-runs" / stem


def build_cli_args(raw_args) -> List[str]:
    """Convert YAML `args` content into a flat CLI argument list."""
    if raw_args is None:
        return []

    if isinstance(raw_args, list):
        return [str(item) for item in raw_args]

    if isinstance(raw_args, dict):
        cli_args: List[str] = []
        for key, value in raw_args.items():
            flag = key if str(key).startswith("-") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(flag)
                continue
            if value is None:
                continue
            if isinstance(value, list):
                cli_args.append(flag)
                cli_args.extend(str(item) for item in value)
                continue
            cli_args.extend([flag, str(value)])
        return cli_args

    raise TypeError("`args` must be a mapping or list")


def parse_script_command(raw_script) -> List[str]:
    """Parse the configured script into a command list."""
    if isinstance(raw_script, str):
        command = shlex.split(raw_script)
    elif isinstance(raw_script, list):
        command = [str(part) for part in raw_script]
    else:
        raise TypeError("`script` must be a string or list")

    if not command:
        raise ValueError("`script` must not be empty")
    return command


def parse_env(raw_env) -> Dict[str, str]:
    """Normalize the optional YAML `env` mapping into string pairs."""
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise TypeError("`env` must be a mapping")
    return {str(key): str(value) for key, value in raw_env.items()}


def read_optional_pass_config(data: dict, config_path: Path) -> Optional[bool]:
    """Read pass-config using either supported YAML key spelling, if present."""
    dashed = data.get("pass-config")
    underscored = data.get("pass_config")

    if dashed is not None and underscored is not None and dashed != underscored:
        raise ValueError(f"{config_path} contains conflicting `pass-config` and `pass_config` values")

    value = underscored if underscored is not None else dashed
    return bool(value) if value is not None else None


def read_mapping_file(config_path: Path) -> dict:
    """Load a YAML file and require a top-level mapping."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping")
    return data


def compute_run_id(content: str) -> str:
    """Compute the stable run id for one generated YAML config."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_session_name(script_command: Sequence[str], run_id: str) -> str:
    """Build a tmux-safe session name from the script and run id."""
    script_name = Path(script_command[0]).stem or "run"
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in script_name)
    return f"{safe_name}-{run_id[:12]}"


def build_shell_command(
    command: Sequence[str],
    env: Dict[str, str],
    log_path: Path,
) -> str:
    """Build a shell command that mirrors stdout/stderr to the pane and logfile."""
    env_prefix = shlex.join(["env", *[f"{key}={value}" for key, value in env.items()]]) if env else "env"
    command_str = shlex.join(command)
    log_target = shlex.quote(str(log_path))
    inner_command = f"set -o pipefail; {env_prefix} {command_str} 2>&1 | tee -a {log_target}; exit ${{PIPESTATUS[0]}}"
    return shlex.join(["bash", "-lc", inner_command])


@dataclass
class RunSpec:
    """Concrete execution plan for one generated YAML config."""
    run_id: str
    config_path: Path
    log_path: Optional[Path]
    script_command: List[str]
    env: Dict[str, str]
    cli_args: List[str]
    pass_config: bool
    session_name: str


@dataclass
class RuntimeConfig:
    """Runtime-only settings used to launch generated configs."""
    script_command: List[str]
    env: Dict[str, str]
    cli_args: List[str]
    pass_config: bool
    num_samples: Optional[int] = None
    seed: Optional[int] = None
    rvlimit: Optional[int] = None
    keep_empty_lines: bool = False
    cuda_devices: Optional[List[str]] = None
    cpu_workers: Optional[int] = None
    temp_dir: Optional[str] = None
    skip_existing: Optional[bool] = None
    enable_logging: Optional[bool] = None
    progress_bar: Optional[bool] = None
    log_level: str = "INFO"
    backend: str = "tmux"


def parse_cuda_devices(raw_devices, config_path: Path) -> Optional[List[str]]:
    """Normalize CUDA device config into a list of strings."""
    if raw_devices is None:
        return None
    if isinstance(raw_devices, list):
        return [str(device) for device in raw_devices]
    raise TypeError(f"`cuda_visible_devices` in {config_path} must be a list")


def load_runtime_config(runtime_config_path: str) -> RuntimeConfig:
    """Load runtime-only settings from a YAML file."""
    config_path = Path(runtime_config_path)
    data = read_mapping_file(config_path)
    if "runtime" in data:
        runtime_data = data["runtime"]
        if not isinstance(runtime_data, dict):
            raise TypeError(f"`runtime` in {config_path} must be a mapping")
        data = runtime_data

    raw_args = data.get("args", data.get("cli_args", data.get("cli")))
    no_log_file = data.get("no_log_file", data.get("no-log-file"))
    enable_logging = data.get("enable_logging", data.get("enable-log-file"))
    progress_bar = data.get("progress_bar", data.get("progress-bar"))
    if no_log_file is not None and enable_logging is not None:
        raise ValueError(f"{config_path} contains both `no_log_file` and `enable_logging`")
    if "script" not in data:
        raise ValueError(f"{config_path} is missing required key `script`")
    pass_config = read_optional_pass_config(data, config_path)
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError(f"`log_level` in {config_path} must be one of DEBUG, INFO, WARNING, ERROR")
    backend = str(data.get("backend", "tmux")).lower()
    if backend not in {"tmux", "direct"}:
        raise ValueError(f"`backend` in {config_path} must be either tmux or direct")

    return RuntimeConfig(
        script_command=parse_script_command(data["script"]),
        env=parse_env(data.get("env")),
        cli_args=build_cli_args(raw_args),
        pass_config=pass_config if pass_config is not None else True,
        num_samples=int(data["num_samples"]) if data.get("num_samples") is not None else None,
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        rvlimit=int(data["rvlimit"]) if data.get("rvlimit") is not None else None,
        keep_empty_lines=bool(data.get("keep_empty_lines", False)),
        cuda_devices=parse_cuda_devices(data.get("cuda_visible_devices"), config_path),
        cpu_workers=int(data["cpu_workers"]) if data.get("cpu_workers") is not None else None,
        temp_dir=str(data["temp_dir"]) if data.get("temp_dir") is not None else None,
        skip_existing=bool(data["skip"]) if data.get("skip") is not None else None,
        enable_logging=(not bool(no_log_file)) if no_log_file is not None else (
            bool(enable_logging) if enable_logging is not None else None
        ),
        progress_bar=bool(progress_bar) if progress_bar is not None else None,
        log_level=log_level,
        backend=backend,
    )


def generate_run_specs(
    haml_file: str,
    num_samples: Optional[int],
    temp_dir: Optional[str],
    skip_existing: bool,
    enable_logging: bool,
    runtime_config: RuntimeConfig,
    seed: Optional[int] = None,
    keep_empty_lines: bool = False,
) -> Tuple[List[RunSpec], int, Path]:
    """Expand one HAML file into executable run specs and persisted YAML configs."""
    rng = np.random.default_rng(seed)
    haml_object = parse_file(haml_file)
    output_dir = Path(temp_dir) if temp_dir else default_temp_dir(haml_file)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if enable_logging:
        log_dir.mkdir(parents=True, exist_ok=True)

    def iter_contents() -> Iterator[str]:
        if num_samples is None:
            yield from haml_object.all(random_state=rng)
            return
        for _ in range(num_samples):
            yield from haml_object.sample(random_state=rng)

    generated = 0
    seen_run_ids = set()
    specs: List[RunSpec] = []

    for raw_content in iter_contents():
        content = raw_content if keep_empty_lines else remove_empty_lines(raw_content)
        run_id = compute_run_id(content)
        config_path = output_dir / f"{run_id}.yaml"

        if run_id in seen_run_ids:
            LOG.info("Skipping duplicate config for run_id=%s", run_id)
            continue
        seen_run_ids.add(run_id)

        if skip_existing and config_path.exists():
            LOG.info("Skipping existing config %s", config_path)
            continue

        config_path.write_text(content, encoding="utf-8")
        specs.append(
            RunSpec(
                run_id=run_id,
                config_path=config_path,
                log_path=(log_dir / f"{run_id}.log") if enable_logging else None,
                script_command=runtime_config.script_command,
                env=runtime_config.env,
                cli_args=runtime_config.cli_args,
                pass_config=runtime_config.pass_config,
                session_name=build_session_name(runtime_config.script_command, run_id),
            )
        )
        generated += 1

    return specs, generated, output_dir


def build_run_command(spec: RunSpec) -> List[str]:
    """Build the script command for one run."""
    command = list(spec.script_command)
    if spec.pass_config:
        command.extend(["--config", str(spec.config_path), "--id", spec.run_id])
    else:
        command.extend(["--id", spec.run_id])
        command.extend(spec.cli_args)
    return command


def build_run_env(spec: RunSpec, cuda_device: Optional[str]) -> Dict[str, str]:
    """Build the environment for one run."""
    env = dict(spec.env)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
    return env


def launch_tmux_session(spec: RunSpec, cuda_device: Optional[str], quiet: bool = False) -> None:
    """Launch one run inside a detached tmux session."""
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for execution but was not found in PATH")

    if spec.log_path is not None:
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_run_command(spec)
    env = build_run_env(spec, cuda_device)

    shell_command = (
        build_shell_command(command=command, env=env, log_path=spec.log_path)
        if spec.log_path is not None
        else shlex.join(["env", *[f"{key}={value}" for key, value in env.items()], *command])
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            spec.session_name,
            shell_command,
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "set-option",
            "-w",
            "-t",
            f"{spec.session_name}:0",
            "remain-on-exit",
            "failed",
        ],
        check=True,
    )

    if not quiet:
        LOG.info(
            "Launched %s on CUDA_VISIBLE_DEVICES=%s using %s%s",
            spec.session_name,
            cuda_device if cuda_device is not None else "<unset>",
            spec.config_path,
            f" (tmux log: {spec.log_path})" if spec.log_path is not None else "",
        )


def launch_direct(
    spec: RunSpec,
    cuda_device: Optional[str],
    quiet: bool = False,
) -> Tuple[subprocess.Popen, Optional[IO[str]]]:
    """Launch one run directly and return its process and optional log handle."""
    if spec.log_path is not None:
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = None if spec.log_path is None else spec.log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        build_run_command(spec),
        env={**os.environ, **build_run_env(spec, cuda_device)},
        stdout=log_handle,
        stderr=subprocess.STDOUT if log_handle is not None else None,
    )
    if not quiet:
        LOG.info(
            "Launched %s on CUDA_VISIBLE_DEVICES=%s using %s%s",
            spec.session_name,
            cuda_device if cuda_device is not None else "<unset>",
            spec.config_path,
            f" (log: {spec.log_path})" if spec.log_path is not None else "",
        )
    return process, log_handle


def tmux_session_alive(session_name: str) -> bool:
    """Return whether the tmux session still has a live pane."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return False

    result = subprocess.run(
        ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_dead}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False

    pane_states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not pane_states:
        return False
    return any(state == "0" for state in pane_states)


def execute_runs(
    specs: Sequence[RunSpec],
    cuda_devices: Sequence[str],
    cpu_workers: int = 1,
    poll_interval_seconds: float = 2.0,
    progress_bar: bool = False,
    backend: str = "tmux",
) -> None:
    """Execute run specs on a fixed pool of CUDA or CPU slots."""
    if cpu_workers < 1:
        raise ValueError("cpu_workers must be >= 1")
    if cuda_devices and cpu_workers != 1:
        raise ValueError("cpu_workers cannot be set together with cuda_devices")
    if not specs:
        LOG.info("No runs to execute.")
        return

    pending: Deque[RunSpec] = deque(specs)
    slot_devices: List[Optional[str]] = list(cuda_devices) if cuda_devices else [None] * cpu_workers
    slot_ids = list(range(len(slot_devices)))
    active: Dict[int, Tuple[RunSpec, Optional[subprocess.Popen], Optional[IO[str]]]] = {}
    failures: List[Tuple[RunSpec, int]] = []
    completed = 0
    old_completed = 0
    reported_started = False
    total = len(specs)
    progress = None
    if progress_bar and total > 1:
        progress = tqdm(total=total, desc="Runs", unit="run")

    try:
        while pending or active:
            for slot_id, cuda_device in zip(slot_ids, slot_devices):
                if slot_id in active or not pending:
                    continue
                spec = pending.popleft()
                if backend == "direct":
                    process, log_handle = launch_direct(spec, cuda_device, quiet=progress is not None)
                    active[slot_id] = (spec, process, log_handle)
                else:
                    launch_tmux_session(spec, cuda_device, quiet=progress is not None)
                    active[slot_id] = (spec, None, None)

            if progress is not None and not reported_started:
                reported_started = True
                message = (
                    f"Started runs: {completed}/{total} complete, "
                    f"{len(active)} active, {len(pending)} pending"
                )
                tqdm.write(message)

            for slot_id, (spec, process, log_handle) in list(active.items()):
                if process is None:
                    if tmux_session_alive(spec.session_name):
                        continue
                    returncode = 0
                else:
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    if log_handle is not None:
                        log_handle.close()

                del active[slot_id]
                completed += 1
                if returncode != 0:
                    failures.append((spec, returncode))
                    LOG.error("Failed %s with exit code %d (%d/%d)", spec.session_name, returncode, completed, total)
                    continue

                if progress is None:
                    LOG.info("Completed %s (%d/%d)", spec.session_name, completed, total)
                else:
                    progress.update(1)

            if pending or active:
                if progress is None and old_completed != completed:
                    old_completed = completed
                    LOG.info(
                        "Progress: %d/%d complete, %d active, %d pending",
                        completed,
                        total,
                        len(active),
                        len(pending),
                    )
                time.sleep(poll_interval_seconds)
    finally:
        if progress is not None:
            progress.close()
        for _, process, log_handle in active.values():
            if process is not None:
                process.terminate()
            if log_handle is not None:
                log_handle.close()

    if failures:
        details = ", ".join(
            f"{spec.session_name}=exit {code}{f' log {spec.log_path}' if spec.log_path is not None else ''}"
            for spec, code in failures
        )
        raise RuntimeError(f"{len(failures)} direct run(s) failed: {details}")


def run_file(
    haml_file: str,
    num_samples: Optional[int] = None,
    cuda_devices: Optional[Sequence[str]] = None,
    cpu_workers: Optional[int] = None,
    temp_dir: Optional[str] = None,
    runtime_config_path: Optional[str] = None,
    skip_existing: Optional[bool] = None,
    enable_logging: Optional[bool] = None,
    progress_bar: Optional[bool] = None,
    seed: Optional[int] = None,
    keep_empty_lines: Optional[bool] = None,
) -> Tuple[int, Path]:
    """Generate, persist, and execute runs derived from one HAML file."""
    if runtime_config_path is None:
        raise ValueError("runtime_config_path is required")
    runtime_config = load_runtime_config(runtime_config_path)
    if runtime_config.rvlimit is not None:
        haml_module.RANDOM_VALUE_LIMIT = runtime_config.rvlimit
    if num_samples is None:
        num_samples = runtime_config.num_samples
    if cuda_devices is None and cpu_workers is None:
        cuda_devices = runtime_config.cuda_devices
    if cpu_workers is None and not cuda_devices:
        cpu_workers = runtime_config.cpu_workers
    if temp_dir is None:
        temp_dir = runtime_config.temp_dir
    if skip_existing is None:
        skip_existing = runtime_config.skip_existing
    if skip_existing is None:
        skip_existing = False
    if enable_logging is None:
        enable_logging = runtime_config.enable_logging
    if enable_logging is None:
        enable_logging = True
    if progress_bar is None:
        progress_bar = runtime_config.progress_bar
    if progress_bar is None:
        progress_bar = False
    if seed is None:
        seed = runtime_config.seed
    if keep_empty_lines is None:
        keep_empty_lines = runtime_config.keep_empty_lines

    if cpu_workers is not None and cpu_workers < 1:
        raise ValueError("cpu_workers must be >= 1")
    if cuda_devices and cpu_workers is not None:
        raise ValueError("cpu_workers cannot be set together with cuda_devices")

    specs, generated, output_dir = generate_run_specs(
        haml_file=haml_file,
        num_samples=num_samples,
        temp_dir=temp_dir,
        skip_existing=skip_existing,
        enable_logging=enable_logging,
        runtime_config=runtime_config,
        seed=seed,
        keep_empty_lines=keep_empty_lines,
    )
    execute_runs(
        specs,
        cuda_devices or [],
        cpu_workers=cpu_workers or 1,
        progress_bar=progress_bar,
        backend=runtime_config.backend,
    )
    return generated, output_dir
