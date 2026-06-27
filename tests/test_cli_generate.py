from argparse import Namespace

from haml.__main__ import run_generate_mode


def test_generate_mode_samples_multiple_files(tmp_path):
    first = tmp_path / "first.hml"
    second = tmp_path / "second.hml"
    output_dir = tmp_path / "out"
    first.write_text("value: {{ a || b }}\n", encoding="utf-8")
    second.write_text("name: {{ x || y }}\n", encoding="utf-8")

    run_generate_mode(
        Namespace(
            files=[str(first), str(second)],
            directory=str(output_dir),
            all=False,
            sample=2,
            seed=1,
            rvlimit=1,
            keep_empty_lines=False,
        )
    )

    outputs = sorted(path.name for path in output_dir.glob("*.yml"))
    assert outputs == ["first_0.yml", "first_1.yml", "second_0.yml", "second_1.yml"]


def test_generate_mode_expands_all_multiple_files(tmp_path, monkeypatch):
    first = tmp_path / "first.hml"
    second = tmp_path / "second.hml"
    output_dir = tmp_path / "out"
    first.write_text("value: [[ a || b ]]\n", encoding="utf-8")
    second.write_text("name: [[ x || y || z ]]\n", encoding="utf-8")
    prompts = []

    def accept(prompt):
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", accept)

    run_generate_mode(
        Namespace(
            files=[str(first), str(second)],
            directory=str(output_dir),
            all=True,
            sample=0,
            seed=None,
            rvlimit=1,
            keep_empty_lines=False,
        )
    )

    outputs = sorted(path.name for path in output_dir.glob("*.yml"))
    assert outputs == [
        "first_0.yml",
        "first_1.yml",
        "second_0.yml",
        "second_1.yml",
        "second_2.yml",
    ]
    assert prompts == ["Proceed to create 5 files? [y/n] "]
