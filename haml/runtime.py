import hashlib
import logging
import shlex
import shutil
import subprocess
import tempfile
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml

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


def load_run_config(config_path: Path) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Load one generated YAML config and extract execution settings."""
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping")

    if "script" not in data:
        raise ValueError(f"{config_path} is missing required key `script`")

    raw_args = data.get("args", data.get("cli_args", data.get("cli")))
    return (
        parse_script_command(data["script"]),
        parse_env(data.get("env")),
        build_cli_args(raw_args),
    )


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
    session_name: str


def generate_run_specs(
    haml_file: str,
    num_samples: Optional[int],
    temp_dir: Optional[str],
    skip_existing: bool,
    enable_logging: bool,
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
            yield haml_object.random(random_state=rng)

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
        script_command, env, cli_args = load_run_config(config_path)
        specs.append(
            RunSpec(
                run_id=run_id,
                config_path=config_path,
                log_path=(log_dir / f"{run_id}.log") if enable_logging else None,
                script_command=script_command,
                env=env,
                cli_args=cli_args,
                session_name=build_session_name(script_command, run_id),
            )
        )
        generated += 1

    return specs, generated, output_dir


def launch_tmux_session(spec: RunSpec, cuda_device: Optional[str]) -> None:
    """Launch one run inside a detached tmux session."""
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for execution but was not found in PATH")

    if spec.log_path is not None:
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(spec.env)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device

    command = list(spec.script_command)
    command.extend(["--id", spec.run_id])
    command.extend(spec.cli_args)

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

    LOG.info(
        "Launched %s on CUDA_VISIBLE_DEVICES=%s using %s%s",
        spec.session_name,
        cuda_device if cuda_device is not None else "<unset>",
        spec.config_path,
        f" (tmux log: {spec.log_path})" if spec.log_path is not None else "",
    )


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
    poll_interval_seconds: float = 2.0,
) -> None:
    """Execute run specs on a fixed pool of CUDA or CPU slots."""
    if not specs:
        LOG.info("No runs to execute.")
        return

    pending: Deque[RunSpec] = deque(specs)
    slot_devices: List[Optional[str]] = list(cuda_devices) if cuda_devices else [None]
    slot_ids = list(range(len(slot_devices)))
    active: Dict[int, RunSpec] = {}
    completed = 0
    old_completed = 0
    total = len(specs)

    while pending or active:
        for slot_id, cuda_device in zip(slot_ids, slot_devices):
            if slot_id in active or not pending:
                continue
            spec = pending.popleft()
            launch_tmux_session(spec, cuda_device)
            active[slot_id] = spec

        finished_slots: List[int] = []
        for slot_id, spec in active.items():
            if tmux_session_alive(spec.session_name):
                continue
            finished_slots.append(slot_id)
            completed += 1
            LOG.info("Completed %s (%d/%d)", spec.session_name, completed, total)

        for slot_id in finished_slots:
            del active[slot_id]

        if pending or active:
            if old_completed != completed:
                old_completed = completed
                LOG.info("Progress: %d/%d complete, %d active, %d pending", completed, total, len(active), len(pending))
            time.sleep(poll_interval_seconds)


def run_file(
    haml_file: str,
    num_samples: Optional[int] = None,
    cuda_devices: Optional[Sequence[str]] = None,
    temp_dir: Optional[str] = None,
    skip_existing: bool = False,
    enable_logging: bool = True,
    seed: Optional[int] = None,
    keep_empty_lines: bool = False,
) -> Tuple[int, Path]:
    """Generate, persist, and execute runs derived from one HAML file."""
    specs, generated, output_dir = generate_run_specs(
        haml_file=haml_file,
        num_samples=num_samples,
        temp_dir=temp_dir,
        skip_existing=skip_existing,
        enable_logging=enable_logging,
        seed=seed,
        keep_empty_lines=keep_empty_lines,
    )
    execute_runs(specs, cuda_devices or [])
    return generated, output_dir
