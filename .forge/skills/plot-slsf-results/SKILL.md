---
name: plot-slsf-results
description: Generate stable matplotlib plots for SLSF-rs theta job result JSON files. Use when the user asks to plot, visualize, graph, or inspect pulled-down SLSF theta results, especially files like runs/<job>/<job>.results.json, using uv with matplotlib/numpy.
---

# Plot SLSF Results

Use the bundled Python script instead of rewriting plotting code inline.

## Workflow

1. Ensure the result JSON exists locally. If the user references a remote job, pull the tracked job result first.
2. Run the script with `uv` from the repository root:

```bash
uv run --with matplotlib --with numpy python .forge/skills/plot-slsf-results/scripts/plot_theta_results.py runs/<job>/<job>.results.json
```

3. Report the generated PNG paths and the grouping used.

## Script options

```bash
uv run --with matplotlib --with numpy python .forge/skills/plot-slsf-results/scripts/plot_theta_results.py \
  runs/<job>/<job>.results.json \
  --output-dir runs/<job> \
  --prefix <job>
```

The script groups tasks by `(L, T)`, averages over samples, and plots mean with SEM error bars.

## Outputs

By default the script writes three PNG files next to the input result JSON:

- `<prefix>_observables.png`: `RhoXY`, `RhoZ`, `Energy`, `MagnetizationSquared`
- `<prefix>_fluctuations.png`: `SpecificHeat`, `Chi`
- `<prefix>_acceptance.png`: Metropolis acceptance

Do not create extra summary or documentation files unless the user explicitly requests them.
