"""Execute config files through a bounded direct, tmux or Slurm queue."""

import logging
import os
import shlex
import signal
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml
from tqdm import tqdm

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    config_path: Path


def parse_script_command(value):
    command = shlex.split(value) if isinstance(value, str) else [str(part) for part in value]
    if not command:
        raise ValueError("script must contain a command")
    return command


def load_runtime_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Runtime config must be a mapping")
    allowed = {"script", "args", "env", "pass_config", "backend", "cpu_workers", "cuda_visible_devices", "temp_dir", "workdir", "skip", "enable_logging", "progress_bar", "log_level", "slurm_args", "num_samples", "seed", "rvlimit", "keep_empty_lines"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unsupported runtime settings: {', '.join(sorted(unknown))}")
    return config


def default_temp_dir(source):
    return Path(tempfile.gettempdir()) / "haml-runs" / Path(source).stem


def collect_runs(inputs):
    specs, seen = [], set()
    for source in map(Path, inputs):
        paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}) if source.is_dir() else [source]
        if not paths:
            raise ValueError(f"{source} contains no .yaml or .yml config files")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"Expected a YAML config: {path}")
            if path.stem in seen:
                raise ValueError(f"Duplicate run ID {path.stem!r}; use unique filenames")
            seen.add(path.stem)
            specs.append(RunSpec(path.stem, path.resolve()))
    return specs


def build_command(spec, config):
    command = parse_script_command(config["script"])
    if config.get("pass_config", True):
        command.extend(["--config", str(spec.config_path)])
    command.extend(["--id", spec.run_id])
    args = config.get("args", [])
    if isinstance(args, dict):
        for key, value in args.items():
            flag = str(key) if str(key).startswith("-") else f"--{key}"
            if value is False or value is None:
                continue
            command.append(flag)
            if value is not True:
                command.extend(map(str, value if isinstance(value, list) else [value]))
    else:
        command.extend(map(str, args))
    return command


def session_name(directory):
    return "haml-" + quote(str(directory), safe="/-_").replace(".", "%2E")


def tmux(args, check=True):
    result = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result


def running_task(directory, backend):
    if not directory.exists():
        return False
    if (directory / "exitcode").exists():
        return False
    if (directory / "backend").exists():
        backend = (directory / "backend").read_text().strip()
    if backend == "tmux":
        result = tmux(["display-message", "-p", "-t", f"={session_name(directory)}:run", "#{pane_dead}"], check=False)
        return result.returncode == 0 and result.stdout.strip() == "0"
    pid_file = directory / "pid"
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return True  # /proc may not expose the process; kill(pid, 0) succeeded.


def worker_shell(command, directory, logging_enabled):
    """Workers report completion themselves, allowing manual queue recovery."""
    pid = shlex.quote(str(directory / "pid"))
    exitcode = shlex.quote(str(directory / "exitcode"))
    prefix = f"echo $$ > {pid}.tmp; mv {pid}.tmp {pid}; "
    invocation = shlex.join(command)
    if logging_enabled:
        invocation += f" 2>&1 | tee {shlex.quote(str(directory / 'run.log'))}"
        completion = 'code=${PIPESTATUS[0]}; '
    else:
        invocation += " >/dev/null 2>&1"
        completion = 'code=$?; '
    return prefix + invocation + "; " + completion + f"echo $code > {exitcode}.tmp; mv {exitcode}.tmp {exitcode}; exit $code"


def worker_env(config, device):
    env = {**os.environ, **{str(key): str(value) for key, value in config.get("env", {}).items()}}
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env


def launch_direct(command, directory, config, device):
    shell = worker_shell(command, directory, config.get("enable_logging", True))
    return subprocess.Popen(["bash", "-c", shell], cwd=Path(config.get("workdir", ".")).expanduser(), env=worker_env(config, device), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def launch_slurm(command, directory, config, device):
    args = config.get("slurm_args", [])
    args = shlex.split(args) if isinstance(args, str) else list(map(str, args))
    return launch_direct(["srun", *args, *command], directory, config, device)


def launch_tmux(command, directory, config, device):
    name = session_name(directory)
    target = f"={name}:run"
    workdir = str(Path(config.get("workdir", ".")).expanduser().resolve())
    if tmux(["has-session", "-t", f"={name}"], check=False).returncode:
        tmux(["new-session", "-d", "-s", name, "-n", "run", "-c", workdir])
    tmux(["set-option", "-w", "-t", target, "remain-on-exit", "on"])
    env_args = [f"{key}={value}" for key, value in worker_env(config, device).items() if key not in {"TMUX", "TMUX_PANE"}]
    command = ["env", *env_args, "bash", "-c", worker_shell(command, directory, config.get("enable_logging", True))]
    tmux(["respawn-pane", "-k", "-t", target, "-c", workdir, shlex.join(command)])


def execute_runs(specs, config):
    parse_script_command(config["script"])
    backend = config.get("backend", "tmux")
    launchers = {"direct": launch_direct, "tmux": launch_tmux, "slurm": launch_slurm}
    if backend not in launchers:
        raise ValueError(f"Unknown backend {backend!r}")
    devices = list(map(str, config.get("cuda_visible_devices") or []))
    workers = int(config.get("cpu_workers", 1))
    if workers < 1 or len(set(devices)) != len(devices):
        raise ValueError("Worker count must be positive and GPU slots must be unique")
    if devices and ("cpu_workers" in config or backend == "slurm"):
        raise ValueError("GPU slots cannot be combined with cpu_workers or the Slurm backend")
    slots = devices or [None] * workers
    output = Path(config["temp_dir"]).expanduser().resolve()
    active, pending, skipped = {}, deque(), 0
    # Reserve recovered slots before launching any new work.
    for spec in specs:
        directory = output / "runs" / spec.run_id
        if running_task(directory, backend):
            if not config.get("skip", False):
                raise ValueError(f"Run {spec.run_id!r} is already running; use skip to watch it")
            recorded_device = (directory / "device").read_text().strip() or None
            slot = next((index for index, device in enumerate(slots) if device == recorded_device and index not in active), None)
            if slot is None:
                raise ValueError(f"Running task {spec.run_id!r} needs an available slot for {recorded_device!r}")
            active[slot] = (spec, None)
            LOG.info("Watching running task %s", spec.run_id)
        elif config.get("skip", False) and directory.exists():
            skipped += 1
        else:
            pending.append(spec)
    failures, completed = [], 0
    with tqdm(total=len(pending) + len(active), desc="Runs", unit="config", disable=not config.get("progress_bar", True)) as progress:
        try:
            while pending or active:
                for slot, device in enumerate(slots):
                    if slot in active or not pending:
                        continue
                    spec = pending.popleft()
                    directory = output / "runs" / spec.run_id
                    directory.mkdir(parents=True, exist_ok=True)
                    for name in ("exitcode", "pid", "run.log"):
                        path = directory / name
                        if path.exists():
                            path.unlink()
                    (directory / "device").write_text(device or "")
                    (directory / "backend").write_text(backend)
                    process = launchers[backend](build_command(spec, config), directory, config, device)
                    active[slot] = (spec, process)
                for slot, (spec, process) in list(active.items()):
                    directory = output / "runs" / spec.run_id
                    exitfile = directory / "exitcode"
                    if exitfile.exists():
                        code = int(exitfile.read_text())
                    elif process is not None:
                        code = process.poll()
                        if code is None:
                            continue
                    elif running_task(directory, backend):
                        continue
                    else:
                        code = 125  # A disappeared worker did not report completion.
                    if process is not None:
                        process.wait()
                    exitfile.write_text(str(code) + "\n")
                    if code:
                        failures.append((spec.run_id, code))
                        LOG.error("%s exited with code %s", spec.run_id, code)
                    del active[slot]
                    completed += 1
                    progress.update(1)
                progress.set_postfix(failed=len(failures), skipped=skipped, refresh=False)
                if pending or active:
                    time.sleep(0.1)
        finally:
            for spec, process in active.values():
                directory = output / "runs" / spec.run_id
                if (directory / "backend").read_text().strip() == "tmux":
                    continue
                if process is None:
                    pid = int((directory / "pid").read_text())
                elif process.poll() is None:
                    pid = process.pid
                else:
                    continue
                try:
                    os.killpg(pid, signal.SIGTERM)
                    if process is not None:
                        process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)
                    process.wait()
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed: {failures}")
    return completed


def run_files(files, runtime_config_path=None, *, runtime=None):
    config = dict(runtime if runtime is not None else load_runtime_config(runtime_config_path))
    if not files:
        return 0, [Path(config["temp_dir"]).expanduser().resolve()]
    config.setdefault("temp_dir", str(default_temp_dir(files[0])))
    specs = collect_runs(files)
    return execute_runs(specs, config), [Path(config["temp_dir"]).expanduser().resolve()]


def run_file(file, runtime_config_path=None, *, runtime=None):
    completed, directories = run_files([file], runtime_config_path, runtime=runtime)
    return completed, directories[0]
