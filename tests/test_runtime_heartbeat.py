import sys

from haml import Heartbeat
from haml.runtime import heartbeat_fraction, read_heartbeat, run_file


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
