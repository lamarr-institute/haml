# Hyper-YAML (HAML)

<p align="center">
  <img src="logo.png" alt="haml logo" width="200"/>
</p>

HAML is an extension of YAML providing syntax to make parts of the file optional
or generate values.
This is particularly useful for generating YAML files defining the hyperparameters
of ML experiments. The repository also includes a small queue executor for running
config files in parallel through direct processes, tmux, or Slurm.

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

#### Exhaustive Sample Choice Lists

Use `[[ ... || ... ]]` when a choice list should be expanded exhaustively even during random sampling:

```yaml
optimizer: [[ adam || sgd || rmsprop ]]
lr: {{ 0.001 || 0.0003 || 0.0001 }}
batch_size: {{ 32 || 64 }}
```

With `-s 20`, HAML samples the regular `{{ ... }}` choices 20 times and combines
each random sample with every `optimizer` value, producing `20 * 3` YAML files.
Multiple `[[]]` lists are combined as a Cartesian product. When using `--all`,
`{{}}` and `[[]]` are both expanded exhaustively.

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
  -s NUM, --sample NUM  randomly sample files; `[[]]` lists multiply the output
  --seed SEED           random seed for sampling
  --rvlimit RVLIMIT     number of samples when running `all` on random variables with infinite
                        support
  --keep-empty-lines    do not remove empty lines from the output (this is done by default)
```

### Command Line: Execute YAML Runs

The runtime does one thing: execute config files through a bounded worker queue.
Tasks are identified by their config filename stem. All inputs share one queue
and one output directory.

    python -m haml run configs/sleep.yml configs/arousal.yml -r runtime.yml
    python -m haml run configs/ -r runtime.yml

Directory inputs include immediate .yaml and .yml files in sorted order.
Duplicate filename stems in the same queue are rejected before launching work.

The CLI can also expand HAML first and then pass the generated configs to the
same executor:

    python -m haml run experiments/train.hml -r runtime.yml

These generated configs are stored in <temp_dir>/configs/ with filenames based
on SHA-256 of the generated YAML contents. Expansion settings (num_samples,
seed, rvlimit, keep_empty_lines) belong to this preparation step. The executor
does not parse HAML or inspect workload YAML contents.

Example runtime YAML:

    script: [python, train.py]
    backend: direct
    cpu_workers: 4
    temp_dir: ./results
    workdir: .
    skip: false
    progress_bar: true
    enable_logging: true

The command receives --config <absolute-config-path> --id <filename-stem>,
followed by optional args. Set pass_config: false to omit --config.
args accepts a list of arguments or a mapping of flags to values.

Runtime settings:

- script: required command, as an argument list or shell-style string.
- backend: direct, tmux (default), or slurm.
- cpu_workers: maximum parallel workers; defaults to one.
- cuda_visible_devices: optional GPU slots, one worker per listed device.
  These replace cpu_workers and cannot be combined with Slurm. HAML sets the
  standard CUDA_VISIBLE_DEVICES variable for each GPU worker.
- temp_dir: batch output directory. Defaults to /tmp/haml-runs/<first-input-stem>
  using the system temporary directory.
- workdir: worker working directory; defaults to the current directory.
- skip: skip existing run directories; defaults to false.
- enable_logging: write combined stdout/stderr to run.log; defaults to true.
- progress_bar: one tqdm bar counting completed tasks, with failed/skipped
  counts; defaults to true.
- env: optional ordinary environment overrides. Workers inherit the caller's
  environment. HAML supplies config paths and IDs through arguments.
- pass_config and args: control the worker command described above.
- slurm_args: raw srun options as a list or shell-style string.
- log_level: standard Python logging level; defaults to INFO.

CLI overrides are --backend, --cuda-visible-devices, --temp-dir, --script,
--skip and --args. Place --args last; everything after it goes to the worker.

For tmux GPU execution:

    script: [python, train.py]
    backend: tmux
    cuda_visible_devices: [0, 1]
    temp_dir: ./results

For Slurm:

    script: [python, train.py]
    backend: slurm
    cpu_workers: 4
    slurm_args: [--partition, GPU, --gres, gpu:1, --cpus-per-task, "8"]
    temp_dir: ./results

Each Slurm task is one srun invocation. Slurm handles resource allocation;
HAML limits concurrent invocations. Container flags can be supplied in
slurm_args. Setup and application-specific behavior belong in the launched
command. There is no container batching or inner worker scheduler.

Bookkeeping lives in <temp_dir>/runs/<id>/:

- run.log: combined worker output, when logging is enabled.
- exitcode: the reported integer exit code.
- pid, device, backend: minimal information for watching an existing worker.

Run directories are created when launching tasks. Generated config existence
does not count as an existing run.

With skip enabled, the queue first checks for matching running tasks and reserves
their worker/GPU slots. It watches those tasks until completion and starts queued
work as slots become free. Other existing run directories are skipped, including
failed or incomplete runs. No config/command comparison is performed.
Restart the queue on the same host or Slurm submission node; retain enough
workers and the GPUs used by running tasks.

Without skip, HAML overwrites its log and exit information in existing run
directories. An already-running task is a conflict rather than a second launch.
The executed program decides how to handle its own artifacts.

Failed tasks release their slots and the queue continues. The runtime exits
nonzero after the queue finishes if any watched or newly launched task failed.
A worker that disappears without reporting completion is a failure.

Tmux workers remain detached and their panes stay available for inspection.
Interrupting a queue leaves tmux workers alive and stops pending launches.
Direct and srun worker process groups are stopped on interruption.
Run the scheduler itself in tmux to keep pending work progressing after a
terminal disconnect. Manual recovery uses skip; there is no automatic restart
service, fingerprinting, or heartbeat progress protocol.

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
