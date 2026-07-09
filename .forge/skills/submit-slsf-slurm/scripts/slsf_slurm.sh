#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "run this command inside the SLSF-rs repository" >&2
    exit 2
}
cd "$ROOT"

SSH_TARGET=${SLSF_SSH_TARGET:-m9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com}
SSH_PORT=${SLSF_SSH_PORT:-22}
REMOTE_ROOT=${SLSF_REMOTE_ROOT:-/public1/home/m9s004715/SLSF-rs}
SLURM_SCRIPT=${SLSF_SLURM_SCRIPT:-examples/run_theta_small_amd512.slurm}
STATE_DIR=${SLSF_STATE_DIR:-.forge/slurm-runs}
CURRENT_RECORD="$STATE_DIR/current.json"
LEGACY_JOB_ID=.forge/slsf-current-job-id
LOGIN_SHELL=${SLSF_LOGIN_SHELL:-0}
mkdir -p "$STATE_DIR/jobs"

usage() {
    cat <<'EOF'
Usage: slsf_slurm.sh <command> [arguments]

Commands:
  doctor
  sync [--source head|worktree]
  submit <config.toml> [--source head|worktree] [--wait-start]
  status [job-id] [--json]
  wait [job-id] [--attempts N] [--interval SECONDS]
  logs [job-id]
  fetch [job-id] [--destination DIR]
EOF
}

quote() { printf '%q' "$1"; }

remote() {
    local command=$1
    if [[ "$LOGIN_SHELL" == 1 ]]; then
        command="bash -lc $(quote "$command")"
    fi
    ssh -p "$SSH_PORT" "$SSH_TARGET" "$command"
}

current_job_id() {
    if [[ -r "$CURRENT_RECORD" ]]; then
        python3 - "$CURRENT_RECORD" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["job_id"])
PY
        return
    fi
    if [[ -r "$LEGACY_JOB_ID" ]] && [[ $(<"$LEGACY_JOB_ID") =~ ^[0-9]+$ ]]; then
        cat "$LEGACY_JOB_ID"
        return
    fi
    echo "no tracked Slurm run; submit a job or pass a job id" >&2
    return 2
}

record_path() {
    printf '%s/jobs/%s.json\n' "$STATE_DIR" "$1"
}

resolve_job_id() {
    if [[ ${1:-} =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$1"
    else
        current_job_id
    fi
}

sync_source() {
    local source=$1 revision dirty manifest remote_manifest
    revision=$(git rev-parse --short HEAD)
    dirty=false
    [[ -z $(git status --short) ]] || dirty=true

    case "$source" in
        head)
            git archive --format=tar.gz HEAD | remote "mkdir -p $(quote "$REMOTE_ROOT") && tar -xzf - -C $(quote "$REMOTE_ROOT")"
            manifest="head:$revision"
            ;;
        worktree)
            command -v rsync >/dev/null || {
                echo "rsync is required for --source worktree" >&2
                return 2
            }
            manifest="worktree:$revision:$(python3 - <<'PY'
import hashlib, pathlib, subprocess
files = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard"], text=True
).splitlines()
digest = hashlib.sha256()
for name in sorted(files):
    if name.startswith(("target/", "runs/", ".forge/slurm-runs/")) or name == ".forge/slsf-current-job-id":
        continue
    path = pathlib.Path(name)
    if not path.is_file():
        continue
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"
            rsync -az --checksum \
                --exclude=.git/ --exclude=target/ --exclude=runs/ \
                --exclude=.forge/slurm-runs/ --exclude=.forge/slsf-current-job-id \
                -e "ssh -p $SSH_PORT" ./ "$SSH_TARGET:$REMOTE_ROOT/"
            ;;
        *)
            echo "unsupported source mode: $source (expected head or worktree)" >&2
            return 2
            ;;
    esac

    remote "test -f $(quote "$REMOTE_ROOT/Cargo.toml") && test -f $(quote "$REMOTE_ROOT/$SLURM_SCRIPT") && printf '%s\\n' $(quote "$manifest") > $(quote "$REMOTE_ROOT/.slsf-source-manifest")"
    remote_manifest=$(remote "cat $(quote "$REMOTE_ROOT/.slsf-source-manifest")")
    [[ "$remote_manifest" == "$manifest" ]] || {
        echo "remote source manifest verification failed" >&2
        return 1
    }
    printf 'source=%s\nrevision=%s\ndirty=%s\nmanifest=%s\n' "$source" "$revision" "$dirty" "$manifest"
}

record_run() {
    local job_id=$1 config=$2 source=$3 revision=$4 dirty=$5 manifest=$6
    local path tmp
    path=$(record_path "$job_id")
    tmp="$path.tmp"
    python3 - "$job_id" "$config" "$SLURM_SCRIPT" "$source" "$revision" "$dirty" "$manifest" "$tmp" <<'PY'
import datetime, json, pathlib, sys, tomllib
job_id, config, script, source, revision, dirty, manifest, output = sys.argv[1:]
with open(config, "rb") as handle:
    spec = tomllib.load(handle)
name = spec.get("name") or pathlib.Path(config).stem
output_dir = spec.get("output_dir") or "results"
merged = spec.get("merged_output_file") or str(pathlib.PurePosixPath(output_dir) / f"{pathlib.PurePosixPath(name).name}.results.json")
record = {
    "job_id": int(job_id),
    "config": config,
    "slurm_script": script,
    "name": name,
    "output_dir": output_dir,
    "result_path": merged,
    "revision": revision,
    "dirty": dirty == "true",
    "source": source,
    "manifest": manifest,
    "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(output, "w") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
    mv "$tmp" "$path"
    cp "$path" "$CURRENT_RECORD"
    printf '%s\n' "$job_id" > "$LEGACY_JOB_ID"
}

record_field() {
    local job_id=$1 field=$2 path
    path=$(record_path "$job_id")
    [[ -r "$path" ]] || {
        echo "no local run record for job $job_id" >&2
        return 2
    }
    python3 - "$path" "$field" <<'PY'
import json, sys
value = json.load(open(sys.argv[1])).get(sys.argv[2])
if value is None:
    raise SystemExit(2)
print(value)
PY
}

status_raw() {
    local job_id=$1 queue accounting
    queue=$(remote "squeue -j $(quote "$job_id") -h -o '%i|%T|%r|%M|%D|%R'" || true)
    if [[ -n "$queue" ]]; then
        printf 'queue|%s\n' "$queue"
        return
    fi
    accounting=$(remote "sacct -j $(quote "$job_id") --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NTasks,NodeList -n -P" || true)
    printf 'accounting|%s\n' "$accounting"
}

status_state() {
    local job_id=$1 raw state
    raw=$(status_raw "$job_id")
    if [[ "$raw" == queue\|* ]]; then
        state=$(printf '%s' "${raw#queue|}" | cut -d'|' -f2)
    else
        state=$(printf '%s\n' "${raw#accounting|}" | while IFS='|' read -r id _ _ candidate exit_code _; do
            if [[ "$id" == "$job_id" ]]; then
                if [[ "$candidate" == COMPLETED && "$exit_code" != 0:0 ]]; then
                    printf 'FAILED'
                else
                    printf '%s' "$candidate"
                fi
                break
            fi
        done)
        [[ -n "$state" ]] || state=UNKNOWN
    fi
    printf '%s\n' "$state"
}

cmd=${1:-}
[[ -n "$cmd" ]] || { usage; exit 2; }
shift

case "$cmd" in
    doctor)
        echo "local_root=$ROOT"
        echo "ssh_target=$SSH_TARGET"
        echo "remote_root=$REMOTE_ROOT"
        remote "test -d $(quote "$REMOTE_ROOT") && printf 'remote_root=ok\\n'; command -v sbatch; command -v squeue; command -v sacct; sinfo -h -o '%P|%a|%l|%D|%t'"
        ;;
    sync)
        source=head
        while (($#)); do
            case "$1" in
                --source) source=${2:?missing source mode}; shift 2 ;;
                *) echo "unknown sync argument: $1" >&2; exit 2 ;;
            esac
        done
        sync_source "$source"
        ;;
    submit)
        config=${1:?submit requires a config path}
        shift
        source=head
        wait_start=0
        while (($#)); do
            case "$1" in
                --source) source=${2:?missing source mode}; shift 2 ;;
                --wait-start) wait_start=1; shift ;;
                *) echo "unknown submit argument: $1" >&2; exit 2 ;;
            esac
        done
        [[ -f "$config" ]] || { echo "config not found: $config" >&2; exit 2; }
        config=${config#./}
        if [[ "$source" == head ]] && ! git cat-file -e "HEAD:$config" 2>/dev/null; then
            echo "$config is not present in HEAD; use --source worktree or commit it first" >&2
            exit 2
        fi
        cargo +nightly run --quiet -- check --config "$config"
        sync_output=$(sync_source "$source")
        printf '%s\n' "$sync_output"
        revision=
        dirty=
        manifest=
        while IFS='=' read -r key value; do
            case "$key" in
                revision) revision=$value ;;
                dirty) dirty=$value ;;
                manifest) manifest=$value ;;
            esac
        done <<< "$sync_output"
        submit_output=$(remote "cd $(quote "$REMOTE_ROOT") && sbatch --export=ALL,CONFIG=$(quote "$config") $(quote "$SLURM_SCRIPT")")
        printf '%s\n' "$submit_output"
        job_id=$(python3 - "$submit_output" <<'PY'
import re, sys
matches = re.findall(r"Submitted batch job (\d+)", sys.argv[1])
if matches:
    print(matches[-1])
PY
)
        [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "could not parse Slurm job id" >&2; exit 1; }
        record_run "$job_id" "$config" "$source" "$revision" "$dirty" "$manifest"
        echo "tracked_record=$(record_path "$job_id")"
        if [[ "$wait_start" == 1 ]]; then
            for _ in {1..30}; do
                state=$(status_state "$job_id")
                echo "state=$state"
                [[ "$state" != UNKNOWN ]] && break
                sleep 2
            done
        fi
        ;;
    status)
        json=0
        requested=
        while (($#)); do
            case "$1" in
                --json) json=1; shift ;;
                *) requested=$1; shift ;;
            esac
        done
        job_id=$(resolve_job_id "$requested")
        raw=$(status_raw "$job_id")
        state=$(status_state "$job_id")
        if [[ "$json" == 1 ]]; then
            python3 - "$job_id" "$state" "$raw" <<'PY'
import json, sys
print(json.dumps({"job_id": int(sys.argv[1]), "state": sys.argv[2], "raw": sys.argv[3]}))
PY
        else
            echo "job_id=$job_id"
            echo "state=$state"
            printf '%s\n' "$raw"
        fi
        ;;
    wait)
        requested=
        attempts=60
        interval=10
        while (($#)); do
            case "$1" in
                --attempts) attempts=${2:?missing attempts}; shift 2 ;;
                --interval) interval=${2:?missing interval}; shift 2 ;;
                *) requested=$1; shift ;;
            esac
        done
        job_id=$(resolve_job_id "$requested")
        for ((i=1; i<=attempts; i++)); do
            state=$(status_state "$job_id")
            echo "attempt=$i state=$state"
            case "$state" in
                COMPLETED) exit 0 ;;
                PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|UNKNOWN) sleep "$interval" ;;
                *) exit 1 ;;
            esac
        done
        echo "wait timed out; job $job_id continues" >&2
        exit 124
        ;;
    logs)
        job_id=$(resolve_job_id "${1:-}")
        job_name=$(remote "sacct -j $(quote "$job_id") --format=JobName -n -X | sed -n '1{s/^[[:space:]]*//;s/[[:space:]]*$//;p;}'")
        [[ -n "$job_name" ]] || job_name=slsf-small
        remote "cd $(quote "$REMOTE_ROOT") && { printf '%s\\n' '--- stdout ---'; tail -n 80 $(quote "runs/slurm-$job_name-$job_id.out") 2>/dev/null || true; printf '%s\\n' '--- stderr ---'; tail -n 80 $(quote "runs/slurm-$job_name-$job_id.err") 2>/dev/null || true; }"
        ;;
    fetch)
        requested=
        destination=
        while (($#)); do
            case "$1" in
                --destination) destination=${2:?missing destination}; shift 2 ;;
                *) requested=$1; shift ;;
            esac
        done
        job_id=$(resolve_job_id "$requested")
        state=$(status_state "$job_id")
        [[ "$state" == COMPLETED ]] || { echo "job $job_id is $state; refusing to fetch an incomplete result" >&2; exit 1; }
        result_path=$(record_field "$job_id" result_path)
        config=$(record_field "$job_id" config)
        destination=${destination:-$(dirname "$result_path")}
        mkdir -p "$destination"
        scp -P "$SSH_PORT" "$SSH_TARGET:$REMOTE_ROOT/$result_path" "$destination/"
        scp -P "$SSH_PORT" "$SSH_TARGET:$REMOTE_ROOT/$config" "$destination/$(basename "$config").submitted" || true
        echo "result=$destination/$(basename "$result_path")"
        ;;
    -h|--help|help) usage ;;
    *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
