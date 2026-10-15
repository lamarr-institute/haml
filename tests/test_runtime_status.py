import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import haml.runtime as runtime
from haml.__main__ import prepare_inputs


@pytest.fixture
def batch(tmp_path):
    worker = tmp_path / "worker.py"
    events = tmp_path / "events.jsonl"
    worker.write_text("""import argparse, json, os, pathlib, time, yaml
p = argparse.ArgumentParser()
p.add_argument('--config')
p.add_argument('--id')
p.add_argument('--extra', default='default')
a = p.parse_args()
c = yaml.safe_load(pathlib.Path(a.config).read_text())
def event(kind):
    with open(os.environ['EVENTS'], 'a') as f:
        f.write(json.dumps({'id': a.id, 'kind': kind, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'time': time.time(), 'extra': a.extra, 'cwd': os.getcwd(), 'haml_env': [k for k in os.environ if k.startswith('HAML_')]}) + '\\n')
event('start')
time.sleep(c.get('delay', 0.15))
print(a.id + ' ' + a.extra, flush=True)
event('end')
raise SystemExit(c.get('exit_code', 0))
""")
    files = []
    for index in range(4):
        path = tmp_path / f"job-{index}.yml"
        path.write_text("exit_code: 0\n")
        files.append(str(path))
    config = dict(script=[sys.executable, str(worker)], backend="direct", cpu_workers=2, workdir=str(tmp_path), env={"EVENTS": str(events)}, temp_dir=str(tmp_path / "output"), progress_bar=False)
    return files, config, events


def test_shared_queue_limits_gpu_concurrency_and_preserves_names(batch):
    files, config, events = batch
    config.pop("cpu_workers")
    config.update(cuda_visible_devices=["1", "2"], args=["--extra", "literal $HOME with spaces"])
    assert runtime.run_files(files, runtime=config)[0] == 4
    active = set()
    peak = 0
    for row in map(json.loads, events.read_text().splitlines()):
        if row["kind"] == "start":
            assert row["gpu"] not in active
            active.add(row["gpu"])
            peak = max(peak, len(active))
        else:
            active.remove(row["gpu"])
        assert row["extra"] == "literal $HOME with spaces"
        assert row["cwd"] == config["workdir"]
        assert row["haml_env"] == []
    assert peak == 2
    for file in files:
        directory = Path(config["temp_dir"]) / "runs" / Path(file).stem
        assert (directory / "exitcode").read_text().strip() == "0"
        assert Path(file).read_text() == "exit_code: 0\n"


def test_skip_uses_directory_existence_and_overwrite_ignores_changed_settings(batch):
    files, config, events = batch
    Path(files[0]).write_text("exit_code: 7\n")
    with pytest.raises(RuntimeError, match="run\\(s\\) failed"):
        runtime.run_files(files, runtime=config)
    assert len(events.read_text().splitlines()) == 8
    config.update(skip=True, args=["--extra", "changed"])
    Path(files[0]).write_text("exit_code: 0\n")
    before = events.read_text()
    assert runtime.run_files(files, runtime=config)[0] == 0
    assert events.read_text() == before
    directory = Path(config["temp_dir"]) / "runs" / "job-0"
    (directory / "run.log").write_text("stale log\n")
    config["skip"] = False
    assert runtime.run_files(files, runtime=config)[0] == 4
    assert (directory / "run.log").read_text() == "job-0 changed\n"
    assert (directory / "exitcode").read_text().strip() == "0"


def test_skip_also_skips_incomplete_existing_directories(batch):
    files, config, events = batch
    directory = Path(config["temp_dir"]) / "runs" / "job-0"
    directory.mkdir(parents=True)
    config["skip"] = True
    assert runtime.run_files(files[:1], runtime=config)[0] == 0
    assert not events.exists()


@pytest.mark.parametrize("backend", ["direct", "tmux", "slurm"])
def test_skip_reserves_running_tasks_before_launching_pending_jobs(batch, monkeypatch, backend):
    files, config, _ = batch
    config.update(backend=backend, skip=True)
    output = Path(config["temp_dir"])
    live = output / "runs" / "job-0"
    live.mkdir(parents=True)
    (live / "device").write_text("")
    (live / "run.log").write_text("existing worker log")
    monkeypatch.setattr(runtime, "running_task", lambda directory, backend: directory == live and not (live / "exitcode").exists())
    launches = []

    def launch(command, directory, config, device):
        assert directory != live
        if len(launches) == 0:
            assert command[command.index("--id") + 1] == "job-1"
            (live / "exitcode").write_text("0\n")
        launches.append(directory.name)
        (directory / "exitcode").write_text("0\n")

    monkeypatch.setattr(runtime, "launch_" + backend, launch)
    assert runtime.run_files(files, runtime=config)[0] == 4
    assert launches == ["job-1", "job-2", "job-3"]
    assert (live / "run.log").read_text() == "existing worker log"


def test_running_task_conflicts_without_skip(batch, monkeypatch):
    files, config, _ = batch
    monkeypatch.setattr(runtime, "running_task", lambda *args: True)
    with pytest.raises(ValueError, match="already running"):
        runtime.run_files(files, runtime=config)
    assert not Path(config["temp_dir"]).exists()


def test_recovery_preserves_gpu_assignment_and_rejects_unavailable_slot(batch, monkeypatch):
    files, config, _ = batch
    config.pop("cpu_workers")
    config.update(backend="tmux", skip=True, cuda_visible_devices=["1", "2"])
    live = Path(config["temp_dir"]) / "runs" / "job-0"
    live.mkdir(parents=True)
    (live / "device").write_text("2")
    monkeypatch.setattr(runtime, "running_task", lambda directory, backend: directory == live and not (live / "exitcode").exists())

    def launch(command, directory, config, device):
        assert device == "1"
        (live / "exitcode").write_text("0")
        (directory / "exitcode").write_text("0")

    monkeypatch.setattr(runtime, "launch_tmux", launch)
    assert runtime.run_files(files[:2], runtime=config)[0] == 2
    (live / "exitcode").unlink()
    config["cuda_visible_devices"] = ["1"]
    with pytest.raises(ValueError, match="available slot"):
        runtime.run_files(files[:1], runtime=config)


def test_direct_worker_can_be_watched_by_new_queue(batch):
    files, config, events = batch
    Path(files[0]).write_text("delay: 0.5\n")
    directory = Path(config["temp_dir"]) / "runs" / "job-0"
    directory.mkdir(parents=True)
    (directory / "device").write_text("")
    (directory / "backend").write_text("direct")
    spec = runtime.collect_runs(files[:1])[0]
    process = runtime.launch_direct(runtime.build_command(spec, config), directory, config, None)
    try:
        deadline = time.monotonic() + 3
        while not (directory / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.running_task(directory, "direct")
        config["skip"] = True
        assert runtime.run_files(files, runtime=config)[0] == 4
        starts = [json.loads(row)["id"] for row in events.read_text().splitlines() if json.loads(row)["kind"] == "start"]
        assert sorted(starts) == ["job-0", "job-1", "job-2", "job-3"]
    finally:
        process.wait(timeout=5)


def test_slurm_runs_one_srun_per_config(batch, tmp_path):
    files, config, events = batch
    executable = tmp_path / "srun"
    record = tmp_path / "srun.jsonl"
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\nwith open(os.environ['SRUN_RECORD'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\nassert sys.argv[1:3] == ['--partition', 'test']\nos.execvp(sys.argv[3], sys.argv[3:])\n")
    executable.chmod(0o755)
    config.update(backend="slurm", slurm_args=["--partition", "test"])
    config["env"].update(PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], SRUN_RECORD=str(record))
    assert runtime.run_files(files, runtime=config)[0] == 4
    calls = [json.loads(row) for row in record.read_text().splitlines()]
    assert len(calls) == 4
    assert {call[call.index("--id") + 1] for call in calls} == {"job-0", "job-1", "job-2", "job-3"}


def test_haml_expansion_is_preparation_not_run_existence(batch, tmp_path):
    _, config, _ = batch
    source = tmp_path / "choice.hml"
    source.write_text("exit_code: [[ 0 || 7 ]]\n")
    configs = prepare_inputs([str(source)], config)
    assert len(configs) == 2
    assert not (Path(config["temp_dir"]) / "runs").exists()
    with pytest.raises(RuntimeError, match="run\\(s\\) failed"):
        runtime.run_files(configs, runtime={**config, "skip": True})
    assert len(list((Path(config["temp_dir"]) / "runs").iterdir())) == 2


def test_duplicate_names_fail_before_launch(batch, tmp_path):
    files, config, _ = batch
    duplicate = tmp_path / "other" / "job-0.yml"
    duplicate.parent.mkdir()
    duplicate.write_text("exit_code: 0\n")
    with pytest.raises(ValueError, match="Duplicate run ID"):
        runtime.run_files([files[0], str(duplicate)], runtime=config)
    assert not Path(config["temp_dir"]).exists()


def test_missing_tmux_worker_is_failure(batch, monkeypatch):
    files, config, _ = batch
    config["backend"] = "tmux"
    monkeypatch.setattr(runtime, "running_task", lambda *args: False)
    monkeypatch.setattr(runtime, "launch_tmux", lambda *args: None)
    with pytest.raises(RuntimeError, match="125"):
        runtime.run_files(files[:1], runtime=config)
