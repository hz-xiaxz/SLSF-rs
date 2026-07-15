---
name: submit-slsf-slurm
description: Validate, sync, submit, monitor, diagnose, and fetch SLSF-rs theta jobs on the BSCC-M9 Paracloud Slurm cluster. Use when the user asks to upload code, run a smoke or production job, inspect the current Slurm job, diagnose a failure, or download completed results.
---

# Submit SLSF-rs Slurm Jobs

Use the bundled controller for deterministic mechanics. Keep judgment in the agent: choose the source mode, config, trial/production scope, and whether a retry is safe.

## Controller

Run from the repository root:

```bash
CTRL=.forge/skills/submit-slsf-slurm/scripts/slsf_slurm.sh
```

Supported lifecycle:

```bash
$CTRL doctor
$CTRL sync --source head
$CTRL submit examples/<config>.toml --source head --wait-start
$CTRL status
$CTRL status --json
$CTRL wait --attempts 60 --interval 10
$CTRL logs
$CTRL fetch
```

The controller owns SSH details, remote path selection, scheduler parsing, source sync, and local run records. Do not duplicate those commands manually unless debugging the controller.

## Defaults and state

- Remote project: `/public1/home/m9s004715/SLSF-rs`.
- Slurm script: `examples/run_theta_small_amd512.slurm`.
- Current partition: `amd_m9_768`; ranks/tasks: 64.
- Build occurs on the allocated node with nightly Rust and `target-cpu=native`.
- Run records are local operational state under `.forge/slurm-runs/`:
  - `current.json`: current job.
  - `jobs/<job-id>.json`: immutable submission context used by status, logs, and fetch.
- Never infer the current job from the newest queue entry when a current record exists.

Environment overrides are available for debugging or migration: `SLSF_SSH_TARGET`, `SLSF_SSH_PORT`, `SLSF_REMOTE_ROOT`, `SLSF_SLURM_SCRIPT`, `SLSF_STATE_DIR`, and `SLSF_LOGIN_SHELL=1`.

## Workflow

### Status-only request

Run `$CTRL status`. If failed or uncertain, run `$CTRL logs`. Do not test or sync code for a status-only request.

### New submission

1. Inspect `git status --short --branch` and the requested config.
2. Select source mode:
   - `head`: committed tracked content only; default for a clean tree.
   - `worktree`: include local tracked/untracked work while excluding git metadata, build products, results, and run records.
   - If relevant uncommitted changes exist and intent is unclear, ask once which source to run.
3. Run `cargo +nightly test`.
4. Submit atomically through the controller for normal jobs. For an explicitly requested benchmark/profile job, use the checked-in Slurm script with `PROFILE=benchmark|stat|record`; never emulate profiling with an ad-hoc `srun`, `salloc`, or `sbatch --wrap` command.

   ```bash
   $CTRL submit examples/<config>.toml --source <head|worktree> --wait-start
   ```

   Submission runs `slsf check`, syncs and verifies the remote tree, submits with `sbatch`, and atomically records the job id plus config, output, revision, dirty state, source mode, and manifest.
5. Before any profile submission, require exactly one expanded task and select:
   - `PROFILE=benchmark`: single-rank release binary without counters.
   - `PROFILE=stat`: single-rank profiling binary with hardware counters; add `CARGO_FEATURES=profile-stats` when phase/histogram statistics are required.
   - `PROFILE=record`: single-rank call-graph recording.
   Never use `PROFILE=none` for a one-task benchmark because it launches the production MPI rank count.
6. Set an explicit semantic top-level job name derived from the config, such as `theta-l64-s1000-benchmark`. Reject generic names including `bash`, `sh`, `slsf`, `test`, and `job`.
7. Immediately after submission, query the top-level allocation and verify all of the following before reporting startup:
   - `JobName` equals the intended semantic name and is not `bash`.
   - partition and requested task count match the intended mode.
   - benchmark/profile modes execute one application rank.
   - command points to the checked-in Slurm script, not a shell allocation.
   Cancel the new job immediately if any check fails, then fix the submission path before retrying.
8. For a smoke job, use bounded waiting. A timeout means the job continues; it is not failure.
9. For a production job, confirm startup and return without a long blocking wait.

### Completed result

1. Require `$CTRL status` to report `COMPLETED`.
2. Run `$CTRL fetch`; it refuses incomplete jobs and resolves the result path from the run record.
3. If visualization was requested, invoke the `plot-slsf-results` workflow on the downloaded merged JSON.

## Config policy

Prefer existing TOML configs and edit or create a new config only when requested parameters do not match one. Before submission, `slsf check --config <path>` must succeed and report the expanded task count and output path.

Treat `[measure]` correlation limits as per-lattice arrays. For every configured or expanded lattice specification, require exactly one value in each present `corr_rmax`, `corr_rmax_xy`, or `corr_rmax_z` array, preserving the same order as `L` (or the expanded `l_x`/`l_y`/`l_z` specifications). Never write a scalar correlation limit and never reuse one lattice's limit for another. For example:

```toml
[model]
L = [16, 32]

[measure]
corr_rmax_xy = [8, 16]
corr_rmax_z = [8, 16]
```

Use zero per lattice to disable correlation output, for example `corr_rmax_xy = [0, 0]`. Reject an array-length mismatch before syncing or submitting; rely on `slsf check` as the final validation.

Typical intent:

- Smoke: small samples/sweeps and a bounded wait.
- Production: user-requested scientific parameters, distinct name/output directory, startup confirmation only.
- Profile: submit with the Slurm script's profiling environment only when explicitly requested.

Ensure `name` and `output_dir` are unique enough that trial and production results cannot mix. Never silently overwrite an existing formal result directory.

## Failure handling

- Missing scheduler commands: retry with `SLSF_LOGIN_SHELL=1`.
- SSH failure: run `$CTRL doctor` and verify the configured target.
- Missing from `squeue`: trust the controller's `sacct` fallback; do not call it complete without `COMPLETED` and exit code `0:0`.
- Build/application failure: run `$CTRL logs`, fix locally, rerun `cargo +nightly test`, then submit a new job. Do not mutate an existing run record.
- Generic `bash`/`sh` allocation: cancel it immediately. Never leave an interactive or diagnostic shell allocation queued, and never use a shell allocation to inspect a running production node.
- Compute-node inspection: perform it only inside a purpose-named checked-in Slurm job. Do not attach `srun --jobid ... bash` to an existing allocation.
- One-task benchmark launched with multiple ranks: cancel it and resubmit with `PROFILE=benchmark`, `stat`, or `record`; its timing and merge result are invalid.
- `PENDING` or `RUNNING`: never treat an existing result JSON as completion evidence.
- Fetch refusal: inspect status/logs rather than bypassing the completion guard.

## Report

End in the user's language with:

- Source mode, local revision, dirty state, and local test result.
- Config path and key `L`, `T`, samples, sweeps, thermalization, and binsize values.
- Slurm script, partition, tasks, job id, and current state.
- Run record path and whether this is smoke or production.
- Verified log/result paths, or explicit uncertainty.
- Any bounded-wait timeout, login-shell requirement, or failure diagnosis.
