import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from tqdm import tqdm

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, IO, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from . import haml as haml_module
from .haml import parse_file

LOG = logging.getLogger(__name__)
YAML_SUFFIXES = {".yaml", ".yml"}


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


class Heartbeat:
    """Small helper for scripts launched by the HAML runtime."""

    def __init__(self, path: Optional[str], interval_seconds: float = 5.0):
        self.path = Path(path) if path else None
        self.interval_seconds = interval_seconds
        self._last_write = 0.0
        self._last_data: Dict[str, Any] = {}

    def __enter__(self) -> "Heartbeat":
        self.update(state="running", force=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.update(state="failed" if exc_type else "done", force=True)

    def update(
        self,
        step: Optional[float] = None,
        total: Optional[float] = None,
        message: Optional[str] = None,
        state: str = "running",
        force: bool = False,
        **extra: Any,
    ) -> None:
        """Write one heartbeat update, rate-limited by interval_seconds."""
        if self.path is None:
            return

        now = time.time()
        if not force and now - self._last_write < self.interval_seconds:
            return

        data: Dict[str, Any] = dict(self._last_data)
        data.update({"time": now, "state": state})
        if step is not None:
            data["step"] = step
        if total is not None:
            data["total"] = total
        if message is not None:
            data["message"] = message
        data.update(extra)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)
        self._last_write = now
        self._last_data = data


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
    heartbeat_path: Optional[Path] = None
    append_runtime_args: bool = True


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
    heartbeat: bool = False
    heartbeat_timeout: float = 300.0
    inner_cpu_workers: Optional[int] = None
    configs_per_container: Optional[int] = None
    slurm_memory: str = "32GB"
    slurm_cpus: int = 32
    slurm_partition: str = "CPU"
    slurm_container_image: str = "nvcr.io/ml2r/interactive_ubuntu"
    slurm_container_prefix: str = "haml"
    slurm_job_prefix: str = "haml"
    slurm_mail_user: Optional[str] = None
    slurm_mail_type: Optional[str] = None
    slurm_wrapper: str = "scripts/haml_run_configs.sh"
    slurm_gateway_prefix: Optional[str] = None
    slurm_path_prefix: Optional[str] = None
    workdir: str = "."
    setup: List[str] = field(default_factory=list)


def parse_cuda_devices(raw_devices, config_path: Path) -> Optional[List[str]]:
    """Normalize CUDA device config into a list of strings."""
    if raw_devices is None:
        return None
    if isinstance(raw_devices, list):
        return [str(device) for device in raw_devices]
    raise TypeError(f"`cuda_visible_devices` in {config_path} must be a list")


def parse_setup(raw_setup, config_path: Path) -> List[str]:
    """Normalize optional container setup commands into shell fragments."""
    if raw_setup is None:
        return []
    if isinstance(raw_setup, str):
        return [raw_setup]
    if isinstance(raw_setup, list):
        return [str(command) for command in raw_setup]
    raise TypeError(f"`setup` in {config_path} must be a string or list")


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
    heartbeat = data.get("heartbeat", False)
    if no_log_file is not None and enable_logging is not None:
        raise ValueError(f"{config_path} contains both `no_log_file` and `enable_logging`")
    if "script" not in data:
        raise ValueError(f"{config_path} is missing required key `script`")
    pass_config = read_optional_pass_config(data, config_path)
    log_level = str(data.get("log_level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError(f"`log_level` in {config_path} must be one of DEBUG, INFO, WARNING, ERROR")
    backend = str(data.get("backend", "tmux")).lower()
    if backend not in {"tmux", "direct", "slurm"}:
        raise ValueError(f"`backend` in {config_path} must be one of tmux, direct, or slurm")

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
        heartbeat=bool(heartbeat),
        heartbeat_timeout=float(data.get("heartbeat_timeout", data.get("heartbeat-timeout", 300.0))),
        inner_cpu_workers=int(data["inner_cpu_workers"]) if data.get("inner_cpu_workers") is not None else None,
        configs_per_container=(
            int(data["configs_per_container"]) if data.get("configs_per_container") is not None else None
        ),
        slurm_memory=str(data.get("slurm_memory", "32GB")),
        slurm_cpus=int(data.get("slurm_cpus", 32)),
        slurm_partition=str(data.get("slurm_partition", "CPU")),
        slurm_container_image=str(data.get("slurm_container_image", "nvcr.io/ml2r/interactive_ubuntu")),
        slurm_container_prefix=str(data.get("slurm_container_prefix", "haml")),
        slurm_job_prefix=str(data.get("slurm_job_prefix", data.get("slurm_container_prefix", "haml"))),
        slurm_mail_user=str(data["slurm_mail_user"]) if data.get("slurm_mail_user") is not None else None,
        slurm_mail_type=str(data["slurm_mail_type"]) if data.get("slurm_mail_type") is not None else None,
        slurm_wrapper=str(data.get("slurm_wrapper", "scripts/haml_run_configs.sh")),
        slurm_gateway_prefix=(
            str(data["slurm_gateway_prefix"]) if data.get("slurm_gateway_prefix") is not None else None
        ),
        slurm_path_prefix=str(data["slurm_path_prefix"]) if data.get("slurm_path_prefix") is not None else None,
        workdir=str(data.get("workdir", ".")),
        setup=parse_setup(data.get("setup"), config_path),
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
    heartbeat_dir = output_dir / "heartbeats"
    output_dir.mkdir(parents=True, exist_ok=True)
    if enable_logging:
        log_dir.mkdir(parents=True, exist_ok=True)
    if runtime_config.heartbeat:
        heartbeat_dir.mkdir(parents=True, exist_ok=True)

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
                heartbeat_path=(heartbeat_dir / f"{run_id}.json") if runtime_config.heartbeat else None,
            )
        )
        generated += 1

    return specs, generated, output_dir


def existing_config_run_specs(
    config_dir: str,
    temp_dir: Optional[str],
    enable_logging: bool,
    runtime_config: RuntimeConfig,
) -> Tuple[List[RunSpec], int, Path]:
    """Build executable run specs from YAML files that already exist in a directory."""
    source_dir = Path(config_dir)
    output_dir = Path(temp_dir) if temp_dir else source_dir
    log_dir = output_dir / "logs"
    heartbeat_dir = output_dir / "heartbeats"
    output_dir.mkdir(parents=True, exist_ok=True)
    if enable_logging:
        log_dir.mkdir(parents=True, exist_ok=True)
    if runtime_config.heartbeat:
        heartbeat_dir.mkdir(parents=True, exist_ok=True)

    config_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in YAML_SUFFIXES
    )
    if not config_paths:
        raise ValueError(f"{source_dir} contains no .yaml or .yml config files")

    seen_run_ids = set()
    specs: List[RunSpec] = []
    for config_path in config_paths:
        run_id = config_path.stem
        if run_id in seen_run_ids:
            raise ValueError(f"{source_dir} contains multiple YAML configs with run id {run_id!r}")
        seen_run_ids.add(run_id)
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
                heartbeat_path=(heartbeat_dir / f"{run_id}.json") if runtime_config.heartbeat else None,
            )
        )

    return specs, len(specs), output_dir


def build_run_command(spec: RunSpec) -> List[str]:
    """Build the script command for one run."""
    command = list(spec.script_command)
    if not spec.append_runtime_args:
        return command
    if spec.pass_config:
        command.extend(["--config", str(spec.config_path), "--id", spec.run_id])
    else:
        command.extend(["--id", spec.run_id])
        command.extend(spec.cli_args)
    if spec.heartbeat_path is not None:
        command.extend(["--heartbeat", str(spec.heartbeat_path)])
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
    output_target = log_handle if log_handle is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        build_run_command(spec),
        env={**os.environ, **build_run_env(spec, cuda_device)},
        stdout=output_target,
        stderr=subprocess.STDOUT,
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


def read_heartbeat(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Read one heartbeat file if it exists and contains a JSON object."""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def heartbeat_fraction(data: Dict[str, Any]) -> float:
    """Return completed fraction from optional heartbeat step/total fields."""
    if data.get("state") == "done":
        return 1.0
    try:
        step = float(data["step"])
        total = float(data["total"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    return max(0.0, min(step / total, 1.0))


def format_duration(seconds: Optional[float]) -> str:
    """Return a compact human-readable duration for progress postfixes."""
    if seconds is None:
        return "n/a"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def heartbeat_counts(data: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Return raw step/total counts for tqdm slot bars."""
    if data is None:
        return 0.0, 1.0
    try:
        step = float(data["step"])
        total = float(data["total"])
    except (KeyError, TypeError, ValueError):
        return heartbeat_fraction(data), 1.0
    if total <= 0:
        return 0.0, 1.0
    return max(0.0, min(step, total)), total


def heartbeat_eta(start_time: Optional[float], fraction: float, now: float) -> Optional[float]:
    """Estimate remaining slot time from slot start and current heartbeat fraction."""
    if start_time is None or fraction <= 0.0 or fraction >= 1.0:
        return None
    elapsed = max(0.0, now - start_time)
    return elapsed * (1.0 - fraction) / fraction


def update_heartbeat_progress(
    overall_progress,
    slot_progresses: Dict[int, Any],
    completed: int,
    active: Dict[int, Tuple[RunSpec, Optional[subprocess.Popen], Optional[IO[str]]]],
    pending_count: int,
    heartbeat_timeout: float,
    warned_stale: Set[str],
    slot_started_at: Dict[int, float],
) -> None:
    """Update total and per-slot heartbeat progress bars."""
    now = time.time()
    slot_etas = []
    stale_slots = 0
    missing_slots = 0
    for slot_id, (spec, _, _) in active.items():
        data = read_heartbeat(spec.heartbeat_path)
        fraction = 0.0
        status = ""
        message = ""
        if data is None:
            missing_slots += 1
            status = "no heartbeat"
        else:
            fraction = heartbeat_fraction(data)
            message = str(data.get("message", ""))
            eta = heartbeat_eta(slot_started_at.get(slot_id), fraction, now)
            if eta is not None:
                slot_etas.append(eta)
            try:
                heartbeat_time = float(data["time"])
            except (KeyError, TypeError, ValueError):
                heartbeat_time = None
            if data.get("state") in {"done", "failed"}:
                warned_stale.discard(spec.run_id)
            elif heartbeat_time is not None and now - heartbeat_time > heartbeat_timeout:
                stale_slots += 1
                status = f"! stalled {format_duration(now - heartbeat_time)}"
                if spec.run_id not in warned_stale:
                    warned_stale.add(spec.run_id)
                    suffix = f", last message: {message}" if message else ""
                    warning = (
                        f"Stale heartbeat for slot {slot_id} {spec.run_id}: "
                        f"{now - heartbeat_time:.0f}s since last update{suffix}"
                    )
                    if overall_progress is None and not slot_progresses:
                        LOG.warning(warning)
                    else:
                        tqdm.write(warning)
            else:
                warned_stale.discard(spec.run_id)
                status = str(data.get("state", "running"))

        slot_progress = slot_progresses.get(slot_id)
        if slot_progress is not None:
            prefix = "!" if status.startswith("!") else " "
            current, total = heartbeat_counts(data)
            status_text = status if status else "running"
            message_text = message if message else "-"
            slot_progress.desc = (
                f"{prefix} Slot {slot_id} {spec.config_path.name} | "
                f"{status_text} | {message_text}"
            )
            slot_progress.total = total
            slot_progress.n = current
            slot_progress.refresh()

    if overall_progress is None:
        return

    target = min(overall_progress.total, completed)
    eta_variance = float(np.var(slot_etas)) if len(slot_etas) > 1 else 0.0
    overall_progress.set_postfix(
        active=len(active),
        pending=pending_count,
        stale=stale_slots,
        missing=missing_slots,
        eta_var=f"{eta_variance:.1f}s^2" if slot_etas else "n/a",
        eta_min=format_duration(min(slot_etas) if slot_etas else None),
        eta_max=format_duration(max(slot_etas) if slot_etas else None),
        refresh=False,
    )
    if target != overall_progress.n:
        overall_progress.n = target
    overall_progress.refresh()


def reset_slot_progresses_if_needed(
    slot_progresses: Dict[int, Any],
    active: Dict[int, Tuple[RunSpec, Optional[subprocess.Popen], Optional[IO[str]]]],
    slot_run_ids: Dict[int, Optional[str]],
) -> Dict[int, Optional[str]]:
    """Reset each slot bar when the assigned run changes."""
    for slot_id, slot_progress in slot_progresses.items():
        spec = active[slot_id][0] if slot_id in active else None
        current_run_id = spec.run_id if spec is not None else None
        if slot_run_ids.get(slot_id) == current_run_id:
            continue
        slot_progress.reset(total=1)
        slot_progress.n = 0
        if spec is None:
            slot_progress.desc = f"  Slot {slot_id} idle | idle | -"
        else:
            slot_progress.desc = f"  Slot {slot_id} {spec.config_path.name} | starting | -"
        slot_progress.refresh()
        slot_run_ids[slot_id] = current_run_id
    return slot_run_ids


def chunk_run_specs(specs: Sequence[RunSpec], chunk_size: int) -> List[List[RunSpec]]:
    """Split run specs into fixed-size chunks."""
    if chunk_size < 1:
        raise ValueError("configs_per_container must be >= 1")
    return [list(specs[index:index + chunk_size]) for index in range(0, len(specs), chunk_size)]


def build_container_inner_command(
    chunk: Sequence[RunSpec],
    runtime_config: RuntimeConfig,
    inner_cpu_workers: int,
    enable_logging: bool,
) -> str:
    """Build the shell command executed inside one Slurm container."""
    if not chunk:
        raise ValueError("Cannot build a container command for an empty chunk")

    first_spec = chunk[0]
    wrapper = os.path.expanduser(runtime_config.slurm_wrapper)
    workdir = os.path.expanduser(runtime_config.workdir)
    log_dir = first_spec.log_path.parent if first_spec.log_path is not None else None
    heartbeat_dir = first_spec.heartbeat_path.parent if first_spec.heartbeat_path is not None else None

    wrapper_args = [
        wrapper,
        "--workers",
        str(inner_cpu_workers),
        "--script",
        shlex.join(runtime_config.script_command),
        "--pass-config",
        "true" if runtime_config.pass_config else "false",
        "--enable-logging",
        "true" if enable_logging else "false",
    ]
    if runtime_config.cli_args:
        wrapper_args.extend(["--cli-args", shlex.join(runtime_config.cli_args)])
    if log_dir is not None:
        wrapper_args.extend(["--log-dir", str(log_dir)])
    if heartbeat_dir is not None:
        wrapper_args.extend(["--heartbeat-dir", str(heartbeat_dir)])
    for key, value in runtime_config.env.items():
        wrapper_args.extend(["--env", f"{key}={value}"])
    wrapper_args.append("--")
    wrapper_args.extend(str(spec.config_path) for spec in chunk)

    commands = [f"cd {shlex.quote(workdir)}"]
    commands.extend(runtime_config.setup)
    commands.append(shlex.join(wrapper_args))
    return " && ".join(commands)


def build_slurm_chunk_spec(
    chunk: Sequence[RunSpec],
    runtime_config: RuntimeConfig,
    chunk_index: int,
    inner_cpu_workers: int,
    enable_logging: bool,
) -> RunSpec:
    """Build one synthetic run spec that launches a Slurm container for a chunk."""
    if not chunk:
        raise ValueError("Cannot build a Slurm chunk spec for an empty chunk")

    first_spec = chunk[0]
    shard_id = f"shard-{chunk_index:04d}-{first_spec.run_id[:8]}"
    container_name = f"{runtime_config.slurm_container_prefix}-{shard_id}"
    job_name = f"{runtime_config.slurm_job_prefix}-{shard_id}"
    inner_command = build_container_inner_command(
        chunk=chunk,
        runtime_config=runtime_config,
        inner_cpu_workers=inner_cpu_workers,
        enable_logging=enable_logging,
    )

    command = [
        "srun",
        "--mem",
        runtime_config.slurm_memory,
        "--export",
        "ALL",
        "-c",
        str(runtime_config.slurm_cpus),
        f"--container-name={container_name}",
        "-p",
        runtime_config.slurm_partition,
        f"--job-name={job_name}",
        f"--container-image={runtime_config.slurm_container_image}",
    ]
    if runtime_config.slurm_mail_user is not None:
        command.append(f"--mail-user={runtime_config.slurm_mail_user}")
    if runtime_config.slurm_mail_type is not None:
        command.append(f"--mail-type={runtime_config.slurm_mail_type}")
    command.extend(["--pty", "/bin/bash", "-lc", inner_command])

    return RunSpec(
        run_id=shard_id,
        config_path=first_spec.config_path,
        log_path=None,
        script_command=command,
        env={},
        cli_args=[],
        pass_config=False,
        session_name=shard_id,
        heartbeat_path=None,
        append_runtime_args=False,
    )


def build_slurm_run_specs(
    specs: Sequence[RunSpec],
    runtime_config: RuntimeConfig,
    enable_logging: bool,
) -> List[RunSpec]:
    """Build synthetic tmux-launched Slurm container specs from concrete runs."""
    inner_cpu_workers = runtime_config.inner_cpu_workers or 1
    if inner_cpu_workers < 1:
        raise ValueError("inner_cpu_workers must be >= 1")
    chunk_size = runtime_config.configs_per_container or inner_cpu_workers
    chunks = chunk_run_specs(specs, chunk_size)
    return [
        build_slurm_chunk_spec(
            chunk=chunk,
            runtime_config=runtime_config,
            chunk_index=index,
            inner_cpu_workers=inner_cpu_workers,
            enable_logging=enable_logging,
        )
        for index, chunk in enumerate(chunks)
    ]


def execute_runs(
    specs: Sequence[RunSpec],
    cuda_devices: Sequence[str],
    cpu_workers: int = 1,
    poll_interval_seconds: float = 1.0,
    progress_bar: bool = False,
    backend: str = "tmux",
    seed: Optional[int] = 0,
    heartbeat_timeout: float = 300.0,
    runtime_config: Optional[RuntimeConfig] = None,
    enable_logging: bool = True,
) -> None:
    """Execute run specs on a fixed pool of CUDA or CPU slots."""
    if backend == "slurm":
        if runtime_config is None:
            raise ValueError("runtime_config is required for the slurm backend")
        if cuda_devices:
            raise ValueError("cuda_visible_devices cannot be used with the slurm backend")
        specs = list(specs)
        rng = np.random.default_rng(seed)
        rng.shuffle(specs)
        specs = build_slurm_run_specs(specs, runtime_config, enable_logging=enable_logging)
        backend = "tmux"

    if cpu_workers < 1:
        raise ValueError("cpu_workers must be >= 1")
    if cuda_devices and cpu_workers != 1:
        raise ValueError("cpu_workers cannot be set together with cuda_devices")
    if not specs:
        LOG.info("No runs to execute.")
        return

    specs = list(specs)
    rng = np.random.default_rng(seed)
    rng.shuffle(specs)

    pending: Deque[RunSpec] = deque(specs)
    slot_devices: List[Optional[str]] = list(cuda_devices) if cuda_devices else [None] * cpu_workers
    slot_ids = list(range(len(slot_devices)))
    active: Dict[int, Tuple[RunSpec, Optional[subprocess.Popen], Optional[IO[str]]]] = {}
    failures: List[Tuple[RunSpec, int]] = []
    warned_stale: Set[str] = set()
    completed = 0
    old_completed = 0
    reported_started = False
    total = len(specs)
    heartbeat_enabled = any(spec.heartbeat_path is not None for spec in specs)
    progress = None
    spacer_progress = None
    slot_progresses: Dict[int, Any] = {}
    slot_run_ids: Dict[int, Optional[str]] = {}
    slot_started_at: Dict[int, float] = {}
    if progress_bar and total > 1 and heartbeat_enabled:
        slot_bar_format = "{desc} |{bar}| {n_fmt}/{total_fmt} [{remaining}]"
        slot_progresses = {
            slot_id: tqdm(
                total=1,
                desc=f"  Slot {slot_id} idle | idle | -",
                position=index,
                leave=True,
                bar_format=slot_bar_format,
            )
            for index, slot_id in enumerate(slot_ids)
        }
        spacer_progress = tqdm(total=0, desc="", bar_format="{desc}", position=len(slot_ids), leave=True)
        progress = tqdm(total=total, desc="Total", unit="config", position=len(slot_ids) + 1)
    elif progress_bar and total > 1:
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
                slot_started_at[slot_id] = time.time()

            if progress is not None and not reported_started:
                reported_started = True
                message = (
                    f"Started runs: {completed}/{total} complete, "
                    f"{len(active)} active, {len(pending)} pending"
                )
                tqdm.write(message)

            slot_run_ids = reset_slot_progresses_if_needed(slot_progresses, active, slot_run_ids)
            update_heartbeat_progress(
                progress if heartbeat_enabled else None,
                slot_progresses,
                completed,
                active,
                len(pending),
                heartbeat_timeout,
                warned_stale,
                slot_started_at,
            )

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
                slot_started_at.pop(slot_id, None)
                completed += 1
                if returncode != 0:
                    failures.append((spec, returncode))
                    LOG.error("Failed %s with exit code %d (%d/%d)", spec.session_name, returncode, completed, total)
                    continue

                if progress is None:
                    LOG.info("Completed %s (%d/%d)", spec.session_name, completed, total)
                elif not heartbeat_enabled:
                    target = max(progress.n, completed)
                    progress.update(min(target, total) - progress.n)
                else:
                    target = max(progress.n, completed)
                    progress.update(min(target, total) - progress.n)
                    progress.set_postfix(active=len(active), pending=len(pending), refresh=False)

            if pending or active:
                slot_run_ids = reset_slot_progresses_if_needed(slot_progresses, active, slot_run_ids)
                update_heartbeat_progress(
                    progress if heartbeat_enabled else None,
                    slot_progresses,
                    completed,
                    active,
                    len(pending),
                    heartbeat_timeout,
                    warned_stale,
                    slot_started_at,
                )
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
        for slot_progress in slot_progresses.values():
            slot_progress.close()
        if spacer_progress is not None:
            spacer_progress.close()
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

    source_path = Path(haml_file)
    if source_path.is_dir():
        specs, generated, output_dir = existing_config_run_specs(
            config_dir=haml_file,
            temp_dir=temp_dir,
            enable_logging=enable_logging,
            runtime_config=runtime_config,
        )
    else:
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
        seed=seed,
        heartbeat_timeout=runtime_config.heartbeat_timeout,
        runtime_config=runtime_config,
        enable_logging=enable_logging,
    )
    return generated, output_dir
