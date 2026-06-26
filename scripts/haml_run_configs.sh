#! /usr/bin/env bash
set -euo pipefail

workers=1
script=""
cli_args=""
pass_config=true
log_dir=""
heartbeat_dir=""
enable_logging=true
env_args=()
configs=()

usage() {
  cat >&2 <<'EOF'
Usage:
  haml_run_configs.sh --workers N --script "python run.py" [options] -- CONFIG.yaml...

Options:
  --cli-args "..."          Extra arguments used when --pass-config false.
  --env KEY=VALUE           Environment variable for child runs. May be repeated.
  --pass-config true|false  Whether to pass --config CONFIG --id RUN_ID.
  --log-dir DIR             Directory for per-run logs.
  --heartbeat-dir DIR       Directory for per-run heartbeat files.
  --enable-logging true|false
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      workers="$2"
      shift 2
      ;;
    --script)
      script="$2"
      shift 2
      ;;
    --cli-args)
      cli_args="$2"
      shift 2
      ;;
    --env)
      env_args+=("$2")
      shift 2
      ;;
    --pass-config)
      pass_config="$2"
      shift 2
      ;;
    --log-dir)
      log_dir="$2"
      shift 2
      ;;
    --heartbeat-dir)
      heartbeat_dir="$2"
      shift 2
      ;;
    --enable-logging)
      enable_logging="$2"
      shift 2
      ;;
    --)
      shift
      configs=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$script" ]]; then
  echo "--script is required" >&2
  usage
  exit 2
fi

if ! [[ "$workers" =~ ^[0-9]+$ ]] || (( workers < 1 )); then
  echo "--workers must be a positive integer" >&2
  exit 2
fi

if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "At least one config path is required" >&2
  usage
  exit 2
fi

if [[ "$enable_logging" == "true" && -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
fi
if [[ -n "$heartbeat_dir" ]]; then
  mkdir -p "$heartbeat_dir"
fi

quote() {
  printf '%q' "$1"
}

run_one() {
  local config="$1"
  local file_name run_id command log_path heartbeat_path

  file_name="$(basename "$config")"
  run_id="${file_name%.*}"
  command="$script"

  if [[ "$pass_config" == "true" ]]; then
    command+=" --config $(quote "$config") --id $(quote "$run_id")"
  else
    command+=" --id $(quote "$run_id")"
    if [[ -n "$cli_args" ]]; then
      command+=" $cli_args"
    fi
  fi

  if [[ -n "$heartbeat_dir" ]]; then
    heartbeat_path="${heartbeat_dir}/${run_id}.json"
    command+=" --heartbeat $(quote "$heartbeat_path")"
  fi

  if [[ "${#env_args[@]}" -gt 0 ]]; then
    command="$(printf ' %q' "${env_args[@]}") $command"
    command="env $command"
  fi

  if [[ "$enable_logging" == "true" && -n "$log_dir" ]]; then
    log_path="${log_dir}/${run_id}.log"
    bash -lc "$command" >> "$log_path" 2>&1
  elif [[ "$enable_logging" == "true" ]]; then
    bash -lc "$command"
  else
    bash -lc "$command" >/dev/null 2>&1
  fi
}

failures=0

for config in "${configs[@]}"; do
  while (( $(jobs -pr | wc -l) >= workers )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
  done
  run_one "$config" &
done

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  echo "$failures run(s) failed" >&2
  exit 1
fi
