import json
import sys

from haml.runtime import run_file


def test_run_file_accepts_directory_of_existing_yaml_configs(tmp_path):
    script = tmp_path / "worker.py"
    config_dir = tmp_path / "configs"
    runtime_file = tmp_path / "runtime.yml"
    output_dir = tmp_path / "runtime-out"
    results_file = tmp_path / "results.jsonl"
    config_dir.mkdir()
    first = config_dir / "first.yaml"
    second = config_dir / "second.yml"
    first.write_text("value: 1\n", encoding="utf-8")
    second.write_text("value: 2\n", encoding="utf-8")
    (config_dir / "ignore.txt").write_text("value: 3\n", encoding="utf-8")
    script.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "import os",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--config')",
                "parser.add_argument('--id')",
                "args = parser.parse_args()",
                "with open(os.environ['RESULTS_FILE'], 'a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps({'config': args.config, 'id': args.id}) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    runtime_file.write_text(
        "\n".join(
            [
                f"script: {sys.executable} {script}",
                "backend: direct",
                "cpu_workers: 1",
                f"temp_dir: {output_dir}",
                "enable_logging: false",
                "progress_bar: false",
                "env:",
                f"  RESULTS_FILE: {results_file}",
            ]
        ),
        encoding="utf-8",
    )

    generated, returned_output_dir = run_file(str(config_dir), runtime_config_path=str(runtime_file))

    rows = sorted(
        (json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()),
        key=lambda row: row["id"],
    )
    assert generated == 2
    assert returned_output_dir == output_dir
    assert rows == [
        {"config": str(first), "id": "first"},
        {"config": str(second), "id": "second"},
    ]


def test_run_file_rejects_directory_without_yaml_configs(tmp_path):
    config_dir = tmp_path / "configs"
    runtime_file = tmp_path / "runtime.yml"
    script = tmp_path / "worker.py"
    config_dir.mkdir()
    script.write_text("print('unused')\n", encoding="utf-8")
    runtime_file.write_text(
        "\n".join(
            [
                f"script: {sys.executable} {script}",
                "backend: direct",
            ]
        ),
        encoding="utf-8",
    )

    try:
        run_file(str(config_dir), runtime_config_path=str(runtime_file))
    except ValueError as exc:
        assert "contains no .yaml or .yml config files" in str(exc)
    else:
        raise AssertionError("expected ValueError")
