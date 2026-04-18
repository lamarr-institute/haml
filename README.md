# Hyper-YAML (HAML)

<p align="center">
  <img src="logo.png" alt="haml logo" width="200"/>
</p>

HAML is an extension of YAML providing syntax to make parts of the file optional
or generate values.
This is particularly useful for generating YAML files defining the hyperparameters
of ML experiments. The repository also includes a simple runtime executor that can
expand one `.hml` file into concrete YAML configs and launch each config in its own
`tmux` session.

## Installation

This package is currently not on PiPy and has to be installed directly from github:
```bash
pip install git+https://github.com/lamarr-institute/haml.git
```

Alternatively, you can clone this repository and install it from there:
```bash
git clone git@github.com:lamarr-institute/haml.git
cd haml
pip install .
```

## Syntax

### Choice Lists

Choice lists are the most important and versatile syntax element of HAML files.
They define blocks of text of which only one is selected for the resulting YAML file.

#### Inline
```
model:
  loss: cross-entropy
  dropout_p: {{ 0.0 || 0.05 }}
  norm: channel 
  activation: elu
  mlp_size: {{ 128 || 512 }}
  name: rutime
```

#### Multi-Line
```
channels: {{ ["E1-M2", "E2-M1"]
|| ["E1-M2", "E2-M1", "1-F", "1-2", "2-F"]
|| ["E1-M2", "E2-M1", "1-F", "1-2", "2-F", "Resp Rate", "Pulse Waveform", "Heart Rate"]
}}
```
By default, line breaks are preserved exactly as they occur in the HAML file, i.e., newline characters between the list item separators (`{{`, `||` and `}}`) are part of the items' content!
When not used carefully, this can result in unexpected line breaks in the generated YAML files.
The command line interface of HAML removes empty lines in the generated files by default.


#### Weighted Choice Lists

An optional weight can be added after `{{` or `||`, separated from the main content by `%`. By default, each list item has weight 1.
When sampling randomly from a HAML file, the probability of choosing each item is given by its weight divided by the sum of all items' weights.
Consequently, assigning weight 2 to an item (e.g., through ``||2% ...`) doubles its probability to be sampled.
The weights can be non-negative floats or integers.

```
channels: {{2% ["E1-M2", "E2-M1"]
||3% ["E1-M2", "E2-M1", "1-F", "1-2", "2-F"]
||5% ["E1-M2", "E2-M1", "1-F", "1-2", "2-F", "Resp Rate", "Pulse Waveform", "Heart Rate"]}}
```

#### Optional Blocks

The weighted choice list can be used to create optional content blocks by adding an empty item:

```
{{
option: debug
||}}
```

### Multiple Choice Lists

```
{{1-2%
  key: foo
  value: 3
||
  key: bar
  value: 4
||
  key: baz
  value: -29
}}
```

### Random Values

HAML can insert random values through a special syntax:
```
intensity: {{%normal(loc=10, scale=2)%}}
saturation: {{%uniform(low=0, high=10)%}}
num-scans: {{%integers(high=20)%}}
```

In place of `normal` or `uniform`, every method provided by a NumPy `Generator` object can be invoked this way (e.g., `triangular`, `poisson`, `triangular`, etc.), with keyword arguments provided in parenthesis.
Note that random functions called without arguments still require parentheses (e.g., `{{%normal()%}}`).


## Usage

Write your HAML file using the syntax described above. The recommended file ending is `.hml`.
The `haml` package provides methods for parsing such files into a `HAMLObject`, which allows to (1) generate all possible YAML files matching the HAML file, or (2) sample random YAML files.
You can use the package both from the command line or within a Python script. The
command line also provides a lightweight runtime executor for launching generated
configs through `tmux`.

### Command Line: Generate YAML

The `haml` package can be used from the command line using the following syntax:

```
usage: python -m haml [-h] [-d DIR] [-a] [-s NUM] [--seed SEED] [--rvlimit RVLIMIT]
                      [--keep-empty-lines]
                      file

positional arguments:
  file                  input HAML file

options:
  -h, --help            show this help message and exit
  -d DIR, --directory DIR
                        output directory for generated files
  -a, --all             generate all possible files
  -s NUM, --sample NUM  randomly sample files
  --seed SEED           random seed for sampling
  --rvlimit RVLIMIT     number of samples when running `all` on random variables with infinite
                        support
  --keep-empty-lines    do not remove empty lines from the output (this is done by default)
```

### Command Line: Execute YAML Runs

The runtime executor expands a single `.hml` file into YAML configs, hashes each
generated YAML file, stores it under a temp directory, and launches one `tmux`
session per run.

```
usage: python -m haml run [-h] [-n NUM_SAMPLES]
                          [--cuda-visible-devices [CUDA_VISIBLE_DEVICES ...]]
                          [--temp-dir TEMP_DIR] [--skip] [--seed SEED]
                          [--rvlimit RVLIMIT] [--keep-empty-lines]
                          [--log-level {DEBUG,INFO,WARNING,ERROR}]
                          file
```

Example:

```bash
python -m haml run experiments/train.hml \
  --num-samples 8 \
  --cuda-visible-devices 0 1 \
  --temp-dir /tmp/haml-train \
  --skip
```

The runtime uses these rules:

- The generated YAML must contain a top-level `script` entry.
- The generated YAML may contain a top-level `env` mapping.
- The generated YAML may contain CLI parameters under `args`. `args` can be either:
  a mapping like `{"lr": 0.001, "batch-size": 64, "use-ema": true}` or a list like
  `["--lr", "0.001", "--batch-size", "64"]`.
- The runtime computes `sha256(yaml_content)` and uses the resulting hex digest as the
  run id.
- Every launched script receives `--id <run_id>` in addition to the configured CLI args.
- Generated configs are written to `<temp_dir>/<run_id>.yaml`. If `--temp-dir` is not
  provided, HAML uses a folder below `tempfile.gettempdir()` such as `/tmp/haml-runs/<stem>`.
- Each tmux-backed run also writes pane output to `<temp_dir>/logs/<run_id>.log`.
- If `--skip` is enabled, existing `<run_id>.yaml` files are assumed to have been run
  successfully already. They are neither rewritten nor relaunched.
- If multiple CUDA devices are provided, the runtime schedules one active job per device.
  If the flag is omitted, or passed without values, runs are launched sequentially with
  `CUDA_VISIBLE_DEVICES` left unset.

Example generated YAML:

```yaml
script: python train.py
env:
  WANDB_MODE: offline
args:
  config: configs/model-a.yaml
  epochs: 100
  use-ema: true
```

### Runtime Assumptions

- Your script must accept an `--id` CLI argument.
- Your script is responsible for storing checkpoints, metrics, and any other results locally.
- The current backend is intentionally simple: `tmux` for process management and optional
  `CUDA_VISIBLE_DEVICES` assignment for GPU selection.
- Existing config files are treated as completed runs when `--skip` is used. There is no
  separate result verification layer yet.

### Python Module

The following example shows how to parse a HAML file and work with the resulting `HAMLObject`.

Content of the HAML file `foo.hml`:
```yaml
# A sample yaml file
company: spacelift
domain:
{{2%
 - devops
||
 - devsecops
}}
tutorial:
{{2-2%
  - yaml:
      name: "YAML Ain't Markup Language"
      type: awesome
      born: 2001
||
  - json:
      name: JavaScript Object Notation
      type: great
      born: 2001
||
  - xml:
      name: Extensible Markup Language
      type: good
      born: 1996
}}
author: omkarbirade
published: true
```

Python script:
```python
import haml
import numpy as np

# parse a HAML file
h = haml.parse_file('foo.hml')

rng = np.random.default_rng(2993644)
for i in range(10):
    # generate a random YAML string from HAML file
    s = h.random(random_state=rng)
    # write to YAML file
    with open(f'foo_random_{i}.yaml', 'w') as f:
        f.write(s)

# generate all possible YAML files matching the HAML file
# (this can be a large number, so check before jampacking your harddisk!)
print(f'Generating {h.num_combinations()} files')
for i, s in enumerate(h.all()):
    with open(f'foo_all_{i}.yaml', 'w') as f:
        f.write(s)
```
