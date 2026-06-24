import sys
import time

from haml import Heartbeat
from haml.runtime import (
    RunSpec,
    heartbeat_fraction,
    read_heartbeat,
    reset_slot_progresses_if_needed,
    run_file,
    update_heartbeat_progress,
)


class DummyProgress:
    def __init__(self, total, n=0):
        self.total = total
        self.n = n
        self.desc = ""
        self.postfix = {}
        self.refreshed = 0
        self.reset_count = 0

    def update(self, amount):
        self.n += amount

    def reset(self, total=None):
        self.total = total
        self.n = 0
        self.reset_count += 1

    def set_postfix(self, refresh=False, **kwargs):
        self.postfix = kwargs

    def refresh(self):
        self.refreshed += 1


def make_run_spec(tmp_path, run_id, heartbeat_path):
    return RunSpec(
        run_id=run_id,
        config_path=tmp_path / f"{run_id}.yaml",
        log_path=None,
        script_command=[sys.executable],
        env={},
        cli_args=[],
        pass_config=False,
        session_name=run_id,
        heartbeat_path=heartbeat_path,
    )


def test_heartbeat_helper_writes_progress(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(str(path), interval_seconds=0)

    heartbeat.update(step=2, total=4, message="half", force=True)
    data = read_heartbeat(path)

    assert data["step"] == 2
    assert data["total"] == 4
    assert data["message"] == "half"
    assert heartbeat_fraction(data) == 0.5


def test_heartbeat_done_counts_as_complete(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(str(path), interval_seconds=0)

    heartbeat.update(state="done", force=True)

    assert heartbeat_fraction(read_heartbeat(path)) == 1.0


def test_heartbeat_update_preserves_previous_progress_fields(tmp_path):
    path = tmp_path / "heartbeat.json"
    heartbeat = Heartbeat(str(path), interval_seconds=0)

    heartbeat.update(step=3, total=10, message="epoch", force=True)
    heartbeat.update(message="still running", force=True)
    data = read_heartbeat(path)

    assert data["step"] == 3
    assert data["total"] == 10
    assert data["message"] == "still running"
    assert heartbeat_fraction(data) == 0.3


def test_heartbeat_progress_can_correct_total_to_current_snapshot(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.json"
    heartbeat_path.write_text('{"time": 1, "state": "running", "step": 1, "total": 10}', encoding="utf-8")
    spec = make_run_spec(tmp_path, "run-a", heartbeat_path)
    overall = DummyProgress(total=20, n=5)
    slot_progress = DummyProgress(total=1, n=0)

    update_heartbeat_progress(
        overall,
        {0: slot_progress},
        completed=2,
        active={0: (spec, None, None)},
        pending_count=17,
        heartbeat_timeout=9999999999,
        warned_stale=set(),
        slot_started_at={0: time.time() - 10},
    )

    assert overall.n == 2.1
    assert overall.postfix["active"] == 1
    assert overall.postfix["pending"] == 17
    assert overall.postfix["eta_var"] == "0.0s^2"
    assert overall.postfix["eta_min"] != "n/a"
    assert slot_progress.n == 0.1
    assert slot_progress.total == 1


def test_slot_progress_reports_each_active_slot(tmp_path):
    heartbeat_a = tmp_path / "a.json"
    heartbeat_b = tmp_path / "b.json"
    missing_heartbeat = tmp_path / "missing.json"
    heartbeat_a.write_text('{"time": 1, "state": "running", "step": 4, "total": 10}', encoding="utf-8")
    heartbeat_b.write_text('{"time": 1, "state": "running", "step": 7, "total": 10}', encoding="utf-8")
    slot_progresses = {0: DummyProgress(total=1), 1: DummyProgress(total=1), 2: DummyProgress(total=1)}

    update_heartbeat_progress(
        None,
        slot_progresses,
        completed=0,
        active={
            0: (make_run_spec(tmp_path, "run-a", heartbeat_a), None, None),
            1: (make_run_spec(tmp_path, "run-b", heartbeat_b), None, None),
            2: (make_run_spec(tmp_path, "run-c", missing_heartbeat), None, None),
        },
        pending_count=0,
        heartbeat_timeout=9999999999,
        warned_stale=set(),
        slot_started_at={0: time.time() - 10, 1: time.time() - 10, 2: time.time() - 10},
    )

    assert slot_progresses[0].n == 0.4
    assert slot_progresses[0].postfix["progress"] == "4/10 (40.0%)"
    assert slot_progresses[1].n == 0.7
    assert slot_progresses[2].n == 0.0
    assert slot_progresses[2].postfix["status"] == "no heartbeat"


def test_slot_progress_resets_when_assignment_changes(tmp_path):
    old_spec = make_run_spec(tmp_path, "old", tmp_path / "old.json")
    new_spec = make_run_spec(tmp_path, "new", tmp_path / "new.json")
    slot_progresses = {0: DummyProgress(total=1, n=0.8)}
    slot_run_ids = {0: old_spec.run_id}

    reset_slot_progresses_if_needed(slot_progresses, {0: (new_spec, None, None)}, slot_run_ids)

    assert slot_progresses[0].reset_count == 1
    assert slot_progresses[0].n == 0
    assert slot_progresses[0].desc == f"  Slot 0 {new_spec.config_path.name}"
    assert slot_progresses[0].postfix["status"] == "starting"
    assert slot_run_ids[0] == new_spec.run_id


def test_heartbeat_adds_cli_argument(tmp_path):
    script = tmp_path / "worker.py"
    haml_file = tmp_path / "config.hml"
    runtime_file = tmp_path / "runtime.yml"
    output_dir = tmp_path / "out"

    script.write_text(
        "\n".join(
            [
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--config')",
                "parser.add_argument('--id')",
                "parser.add_argument('--heartbeat')",
                "parser.parse_args()",
            ]
        ),
        encoding="utf-8",
    )
    haml_file.write_text("value: 1\n", encoding="utf-8")
    runtime_file.write_text(
        "\n".join(
            [
                f"script: {sys.executable} {script}",
                "backend: direct",
                f"temp_dir: {output_dir}",
                "heartbeat: true",
                "progress_bar: false",
            ]
        ),
        encoding="utf-8",
    )

    generated, _ = run_file(str(haml_file), runtime_config_path=str(runtime_file))

    assert generated == 1


def test_heartbeat_cli_drives_executor_progress(tmp_path):
    script = tmp_path / "worker.py"
    haml_file = tmp_path / "config.hml"
    runtime_file = tmp_path / "runtime.yml"
    output_dir = tmp_path / "out"

    script.write_text(
        "\n".join(
            [
                "import argparse",
                "from haml import Heartbeat",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--config')",
                "parser.add_argument('--id')",
                "parser.add_argument('--heartbeat')",
                "args = parser.parse_args()",
                "with Heartbeat(args.heartbeat, interval_seconds=0) as heartbeat:",
                "    heartbeat.update(step=1, total=1, message='done')",
            ]
        ),
        encoding="utf-8",
    )
    haml_file.write_text("value: 1\n", encoding="utf-8")
    runtime_file.write_text(
        "\n".join(
            [
                f"script: {sys.executable} {script}",
                "backend: direct",
                f"temp_dir: {output_dir}",
                "heartbeat: true",
                "heartbeat_timeout: 2",
                "progress_bar: true",
            ]
        ),
        encoding="utf-8",
    )

    generated, _ = run_file(str(haml_file), runtime_config_path=str(runtime_file))

    assert generated == 1
    assert len(list((output_dir / "heartbeats").glob("*.json"))) == 1


def test_direct_backend_discards_child_output_when_logging_disabled(tmp_path, capsys):
    script = tmp_path / "worker.py"
    haml_file = tmp_path / "config.hml"
    runtime_file = tmp_path / "runtime.yml"
    output_dir = tmp_path / "out"

    script.write_text(
        "\n".join(
            [
                "import argparse",
                "import sys",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--config')",
                "parser.add_argument('--id')",
                "parser.parse_args()",
                "print('child stdout')",
                "print('child stderr', file=sys.stderr)",
            ]
        ),
        encoding="utf-8",
    )
    haml_file.write_text("value: 1\n", encoding="utf-8")
    runtime_file.write_text(
        "\n".join(
            [
                f"script: {sys.executable} {script}",
                "backend: direct",
                f"temp_dir: {output_dir}",
                "enable_logging: false",
                "progress_bar: false",
            ]
        ),
        encoding="utf-8",
    )

    generated, _ = run_file(str(haml_file), runtime_config_path=str(runtime_file))
    captured = capsys.readouterr()

    assert generated == 1
    assert "child stdout" not in captured.out
    assert "child stderr" not in captured.err
