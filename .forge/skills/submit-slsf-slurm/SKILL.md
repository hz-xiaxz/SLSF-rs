---
name: submit-slsf-slurm
description: Sync the current SLSF-rs working tree to the existing remote cluster directory even when the remote checkout has no usable git history, run local and remote validation, create trial or formal theta job TOML configs, submit rank-64 SLURM jobs on the BSCC-M9 Paracloud account, monitor startup/completion, and report result paths. Use when the user asks to upload/sync SLSF-rs code, run small tests, submit formal production jobs with specified temperatures/samples/sweeps, rebuild remotely, inspect SLSF-rs Slurm failures, or check job status/results on the remote cluster.
---

# Submit SLSF-rs SLURM Jobs

Use this workflow for this repository's Rust theta/XY simulation jobs on the Paracloud SLURM cluster.

Guiding rule from HPC tooling practice: automate deterministic mechanics, keep judgment in the agent. Use fixed commands/scripts for probing, submitting, polling, and parsing status; use judgment for whether to sync dirty code, choose parameters, retry, or report uncertainty.

## Repository facts

- Local repository: `/home/hzxiaxz/Projects/SLSF-rs`
- Usual local branch: `nicolas`
- Remote SSH target: `m9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com -p 22`
- Remote project path: `/public1/home/m9s004715/SLSF-rs`
- Scheduler commands may depend on the remote login environment. If `sbatch`/`squeue`/`sacct` are missing through plain non-interactive ssh, retry the remote command through `bash -lc '<command>'` and record that login-shell wrapping is required.
- Do not create a second remote project directory. Sync code into the existing remote `SLSF-rs` directory.
- The remote `SLSF-rs` directory may be a plain directory or a checkout without useful git history. Do not rely on remote git commit history, `git pull`, or remote branch state to determine freshness.
- Preserve remote outputs and build/cache directories such as `runs/`, `target/`, logs, checkpoints, and results.
- Local helper script: `.forge/scripts/slsf_slurm.sh` fixes the mechanical `status`, `submit`, and bounded `wait` operations. It is a local SSH driver only: run it from `/home/hzxiaxz/Projects/SLSF-rs`; do not upload it to the cluster, install it remotely, or submit it with `sbatch`. Prefer it for these operations unless you need to debug or extend the mechanics.

## Job tracking rule

- Keep the authoritative job id for status checks in the local repository file `.forge/slsf-current-job-id`.
- After every successful `sbatch` submission, immediately overwrite `.forge/slsf-current-job-id` with the submitted numeric job id plus a trailing newline.
- When the user asks whether the SLSF task is done, running, failed, or asks to inspect the current task, read `.forge/slsf-current-job-id` first and check only that job id with `squeue -j <jobid>`, `sacct -j <jobid>`, and `runs/slurm-slsf-small-<jobid>.out/err`.
- Do not infer the current task from the newest `squeue`, newest `sacct`, newest `runs/` directory, or all user jobs unless `.forge/slsf-current-job-id` is missing/unreadable; if it is missing, report that the tracked job id is unavailable before doing any fallback search.
- If `squeue`/`sacct` says the tracked job is still `RUNNING` or `PENDING`, report that it is not finished even if a plausible `<output_dir>/<name>.results.json` exists. Result files can be stale from an earlier run or partial/intermediate; only treat them as completion evidence after the tracked job leaves the queue and accounting/logs show successful completion.

## Core workflow

1. Confirm local state:
   - Run `git status --short --branch`.
   - Run `git rev-parse --short HEAD`.
   - Run `cargo +nightly test` before uploading unless the user only asks for status.
   - If local uncommitted changes exist, decide whether the user requested tracked committed code only or the full working tree.

2. Inspect relevant files before changing/submitting:
   - Rank-64 SLURM script: `examples/run_theta_small_amd512.slurm`.
   - Production templates: `examples/theta_jxy12_prod_small.toml`, `examples/theta_jxy12_prod_mid.toml`, `examples/theta_jxy12_prod_large.toml`.
   - CLI/config parsing: `src/job/cli.rs`, `src/job/config.rs`.

3. Connect and verify the existing remote path and scheduler visibility:
   ```bash
   ssh -p 22 'm9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com' 'pwd; ls; test -d SLSF-rs && echo SLSF-rs-exists; command -v sbatch; command -v squeue; command -v sacct'
   ```
   Use the full login string with `@BSCC-M9@...`.
   If scheduler commands are not visible, retry the same checks as:
   ```bash
   ssh -p 22 'm9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com' "bash -lc 'command -v sbatch; command -v squeue; command -v sacct'"
   ```
   Then wrap later remote scheduler commands with `bash -lc` when needed.

4. Sync code carefully:
   - Prefer a local-source-of-truth sync. The remote repository may not have git records, so never use remote git history as freshness proof.
   - Safe pattern for committed tracked code:
     ```bash
     git archive --format=tar.gz HEAD | ssh -p 22 'm9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com' 'mkdir -p SLSF-rs && tar -xzf - -C SLSF-rs'
     ```
   - This extracts into the existing remote `SLSF-rs` directory and preserves files not present in the archive, including `target/` and `runs/`.
   - If the user needs uncommitted files, use a non-destructive copy/rsync pattern from the local working tree that excludes `.git`, `target/`, `runs/`, logs, and result files. Do not delete remote outputs unless explicitly requested.
   - To guarantee freshness for a job, sync immediately before the remote build/submission, then verify the remote source using a sentinel generated from local state: local short HEAD plus `git status --short` summary for committed-only sync, or a manifest/checksum of the copied working-tree files for uncommitted-file sync. If the sentinel cannot be verified, report uncertainty and do not claim the remote is latest.
   - Keep live per-user cluster state out of git: local tracking files such as `.forge/slsf-current-job-id` and remote outputs are operational state, not source/config changes to commit.

5. Create or choose a TOML config:
   - Use `[model] T = [...]` for temperatures; five temperatures in `1.8-2.2` usually means `[1.8, 1.9, 2.0, 2.1, 2.2]`.
   - For small rank-64 smoke tests, use small values such as `samples = 2`, `sweeps = 20000`, `thermalization = 5000`, `binsize = 100`.
   - For formal jobs requested in this repo, common parameters are `samples = 50`, `sweeps = 1000000`, `thermalization = 100000`, `binsize = 1000`.
   - Use distinct names/output directories so trial and formal results do not mix, e.g.:
     - trial: `theta_jxy12_small_test_t18_22`
     - formal: `theta_jxy12_prod_t18_22_s50`

6. Submit from the remote project directory:
   - Prefer the helper:
     ```bash
     .forge/scripts/slsf_slurm.sh submit examples/<config>.toml
     ```
   - Manual equivalent:
     ```bash
     ssh -p 22 'm9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com' \
       'cd SLSF-rs && sbatch --export=ALL,CONFIG=examples/<config>.toml examples/run_theta_small_amd512.slurm'
     ```
   If the scheduler needs a login shell, set `SLSF_LOGIN_SHELL=1` for the helper or use:
   ```bash
   ssh -p 22 'm9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com' \
     "bash -lc 'cd SLSF-rs && sbatch --export=ALL,CONFIG=examples/<config>.toml examples/run_theta_small_amd512.slurm'"
   ```
   - The script builds release binary remotely before running.
   - `examples/run_theta_small_amd512.slurm` sets `#SBATCH --ntasks=64` and partition `amd_512`.
   - Rank 0 merges all rank JSON files after all ranks finish.
   - Capture `Submitted batch job <id>` and immediately update the local repository tracker:
     ```bash
     printf '%s\n' '<id>' > .forge/slsf-current-job-id
     ```
     Use only the numeric id, with no comments or extra text.

7. Monitor beyond initial submission:
   - Read the authoritative job id from local `.forge/slsf-current-job-id`; do not substitute another job id just because it appears newer in Slurm output.
   - Prefer the helper for routine status checks:
     ```bash
     .forge/scripts/slsf_slurm.sh status
     ```
   - For short smoke tests only, prefer the helper bounded wait. Stop and report if the wait times out; a timeout means the job continues, not failure:
     ```bash
     .forge/scripts/slsf_slurm.sh wait <jobid> --attempts 60 --interval 10
     ```
   - Manual status checks, when debugging the helper:
     ```bash
     squeue -j <jobid> -h -o '%i|%T|%r|%M|%D|%R'
     ```
   - Check accounting with parseable output when possible:
     ```bash
     sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NTasks,NodeList -P
     ```
   - Interpret terminal states: `COMPLETED` = success only with `ExitCode 0:0`; `FAILED` with `0:0` can still be a logic/application failure if logs say so; `OUT_OF_MEMORY`/`OOM` = memory; `TIMEOUT` = walltime; `CANCELLED*` = cancelled; `RUNNING`/`PENDING`/`REQUEUED` = in progress.
   - Inspect logs in remote `SLSF-rs/runs/`:
     - `runs/slurm-slsf-small-<jobid>.out`
     - `runs/slurm-slsf-small-<jobid>.err`
   - Do not inspect a hard-coded result directory unless you first identify the tracked job's config/output from the submission context, script output, or config file. If unsure, say the config/output path is uncertain instead of guessing.
   - Interpret status conservatively:
     - `RUNNING`/`PENDING` in `squeue` or `sacct` means the tracked job is not complete, regardless of any existing result JSON.
     - Existing rank/result JSON files are not completion proof unless their freshness is tied to the tracked job and the job has completed successfully.
     - Use result file mtimes only as supporting evidence; compare them with the job start/end time when available.
   - Success signs for completed jobs:
     - `sacct` final state is `COMPLETED` with `ExitCode` `0:0` for the batch and MPI step.
     - stderr contains `Finished release profile` or no fatal Rust/MPI/build error.
     - stdout contains rank completion lines like `MPI rank X/64 completed ...`.
     - rank 0 reports `merged 64 rank file(s)` into `<output_dir>/<name>.results.json`.

## Common configs

### Trial: five temperatures, rank 64

Create remotely as `examples/theta_jxy12_small_test_t18_22.toml` when the user asks for a small test:

```toml
name = "theta_jxy12_small_test_t18_22"
output_dir = "runs/theta_jxy12_small_test_t18_22"
checkpoint = true
run_time = "02:00:00"
checkpoint_time = "00:15:00"

[model]
L = [4, 6, 8, 10]
T = [1.8, 1.9, 2.0, 2.1, 2.2]
samples = 2
base_seed = 20260627
j_xy = 1.5
delta_j_xy = [0.5]
j_z = 0.1
delta_j_z = [0.0]

[run]
sweeps = 20000
thermalization = 5000
binsize = 100
proposal_width = 3.141592653589793
wolff_steps = 1

[measure]
corr_rmax_xy = 0
corr_rmax_z = 0
```

### Formal: five temperatures, sample 50, sweep 1e6

Create remotely as `examples/theta_jxy12_prod_t18_22_s50.toml` when the user says to submit formal with `sample=50, sweep=1e6`:

```toml
name = "theta_jxy12_prod_t18_22_s50"
output_dir = "runs/theta_jxy12_prod_t18_22_s50"
checkpoint = true
run_time = "48:00:00"
checkpoint_time = "00:30:00"

[model]
L = [4, 6, 8, 10]
T = [1.8, 1.9, 2.0, 2.1, 2.2]
samples = 50
base_seed = 20260627
j_xy = 1.5
delta_j_xy = [0.5]
j_z = 0.1
delta_j_z = [0.0]

[run]
sweeps = 1000000
thermalization = 100000
binsize = 1000
proposal_width = 3.141592653589793
wolff_steps = 1

[measure]
corr_rmax_xy = 0
corr_rmax_z = 0
```

## Cluster probe and smoke-test practice

- Before relying on a new cluster/session, prefer one probe step that captures scheduler visibility, partition availability, module availability, and basic internet reachability rather than hand-running commands piecemeal.
- For this fixed SLSF Paracloud workflow, at minimum verify: `command -v sbatch squeue sacct`, `sinfo -o '%P %a %.10l %.6D %.6t'`, `module avail 2>&1 | head` if modules are needed, and a lightweight internet check only if builds/downloads require it.
- For any new script/config path, run a small end-to-end smoke job before formal production: ship/sync, submit, bounded wait, inspect logs, verify a clear PASS condition. Scheduler acceptance is not evidence that the scientific job works.
- Avoid long blocking waits for formal jobs. Use bounded polling and return status/result paths to the user when the job is still running.
- Prefer parseable scheduler formats (`squeue -h -o`, `sacct -P`) and deterministic state classification over free-form summaries. In this repository, use `.forge/scripts/slsf_slurm.sh status` for the fixed implementation.

## If a job fails

- Always diagnose the job id stored in `.forge/slsf-current-job-id`; do not switch to a different job unless the user explicitly provides one or asks to change the tracked job.
- `Permission denied` on SSH: ensure the target is exactly `m9s004715@BSCC-M9@ssh.cn-hongkong-1.paracloud.com` with `-p 22`.
- Job missing from `squeue`: check `sacct`; it may have completed or failed quickly. Do not call it complete until `sacct` reports `COMPLETED`/`0:0` and logs/results corroborate.
- Build failure: inspect `runs/slurm-slsf-small-<jobid>.err`, then fix locally, run `cargo +nightly test`, resync code, and resubmit.
- No output directory or missing merge: inspect stdout/stderr and rank JSON files under the configured `output_dir`.
- Do not report success just because `sbatch` returned a job id. Confirm with `squeue`/`sacct` and logs.

## Reporting format

End with a concise Chinese or user-language summary containing:

- Whether code was synced into the existing remote `SLSF-rs` directory.
- Freshness proof: sync method, local revision/dirty-state or manifest/checksum, and how it was verified on the remote after sync.
- Local branch/revision and local test result if applicable.
- Config file path and key parameters: `L`, `T`, `samples`, `sweeps`, `thermalization`, `binsize`.
- Slurm script, partition, ranks/tasks, job id, node if known.
- Tracked job id source: `.forge/slsf-current-job-id` and its current value.
- Current `squeue` or `sacct` status and exit code for that tracked job id if available; state clearly if it is still `RUNNING`/`PENDING`, and include pending-reason classification when relevant.
- Log paths and merged result path for that tracked job id, or explicitly say the output path is uncertain if it was not verified from the tracked job's config/logs.
- Whether the job is trial/small test or formal production.
- Any uncertainty, including login-shell requirement, output path ambiguity, stale result-file possibility, remote git-history absence, sentinel/checksum verification failure, or bounded wait timeout.
