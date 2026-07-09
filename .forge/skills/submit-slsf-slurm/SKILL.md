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
4. Submit atomically:

   ```bash
   $CTRL submit examples/<config>.toml --source <head|worktree> --wait-start
   ```

   Submission runs `slsf check`, syncs and verifies the remote tree, submits with `sbatch`, and atomically records the job id plus config, output, revision, dirty state, source mode, and manifest.
5. For a smoke job, use bounded waiting. A timeout means the job continues; it is not failure.
6. For a production job, confirm startup and return without a long blocking wait.

### Completed result

1. Require `$CTRL status` to report `COMPLETED`.
2. Run `$CTRL fetch`; it refuses incomplete jobs and resolves the result path from the run record.
3. If visualization was requested, invoke the `plot-slsf-results` workflow on the downloaded merged JSON.

## Config policy

Prefer existing TOML configs and edit or create a new config only when requested parameters do not match one. Before submission, `slsf check --config <path>` must succeed and report the expanded task count and output path.

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
