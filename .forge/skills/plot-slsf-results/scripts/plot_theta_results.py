#!/usr/bin/env python3
"""Plot SLSF-rs theta result JSON files.

This script expects a merged ThetaJobResult JSON with a top-level `tasks` array.
It groups by (L, T), averages over samples, and writes stable matplotlib PNGs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAIN_OBSERVABLES = [
    ("RhoXY", r"$\rho_{XY}$"),
    ("RhoZ", r"$\rho_Z$"),
    ("Energy", "Energy"),
    ("MagnetizationSquared", r"$M^2$"),
]
FLUCTUATION_OBSERVABLES = [
    ("SpecificHeat", "Specific heat"),
    ("Chi", r"$\chi$"),
]


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem


def load_tasks(result_json: Path) -> list[dict]:
    with result_json.open() as handle:
        data = json.load(handle)
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise SystemExit(f"{result_json} does not look like a merged theta result JSON")
    return tasks


def collect(tasks: list[dict]) -> tuple[list[int], list[float], dict[str, defaultdict], defaultdict]:
    all_observables = [name for name, _ in MAIN_OBSERVABLES + FLUCTUATION_OBSERVABLES]
    grouped = {obs: defaultdict(list) for obs in all_observables}
    acceptance = defaultdict(list)
    lattice_sizes = set()
    temperatures = set()

    for item in tasks:
        task = item.get("task", {})
        lattice_size = task.get("l")
        temperature = task.get("temperature")
        if not finite_number(lattice_size) or not finite_number(temperature):
            continue
        lattice_size = int(lattice_size)
        temperature = float(temperature)
        key = (lattice_size, temperature)
        lattice_sizes.add(lattice_size)
        temperatures.add(temperature)

        acc = item.get("acceptance")
        if finite_number(acc):
            acceptance[key].append(float(acc))

        observables = item.get("observables", {})
        if not isinstance(observables, dict):
            continue
        for obs in all_observables:
            value = observables.get(obs, {}).get("mean") if isinstance(observables.get(obs), dict) else None
            if finite_number(value):
                grouped[obs][key].append(float(value))

    return sorted(lattice_sizes), sorted(temperatures), grouped, acceptance


def plot_errorbar_grid(
    path: Path,
    title: str,
    specs: list[tuple[str, str]],
    lattice_sizes: list[int],
    temperatures: list[float],
    grouped: dict[str, defaultdict],
    ncols: int,
) -> None:
    nrows = int(math.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), constrained_layout=True)
    axes_array = np.array(axes, dtype=object).reshape(-1)

    for ax, (obs, ylabel) in zip(axes_array, specs):
        for lattice_size in lattice_sizes:
            means = []
            errors = []
            for temperature in temperatures:
                mean, sem = mean_sem(grouped[obs].get((lattice_size, temperature), []))
                means.append(mean)
                errors.append(sem)
            ax.errorbar(
                temperatures,
                means,
                yerr=errors,
                marker="o",
                ms=3,
                lw=1.2,
                capsize=2,
                label=f"L={lattice_size}",
            )
        ax.set_xlabel("T")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes_array[len(specs) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_acceptance(
    path: Path,
    prefix: str,
    lattice_sizes: list[int],
    temperatures: list[float],
    acceptance: defaultdict,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for lattice_size in lattice_sizes:
        means = []
        errors = []
        for temperature in temperatures:
            mean, sem = mean_sem(acceptance.get((lattice_size, temperature), []))
            means.append(mean)
            errors.append(sem)
        ax.errorbar(
            temperatures,
            means,
            yerr=errors,
            marker="o",
            ms=3,
            lw=1.2,
            capsize=2,
            label=f"L={lattice_size}",
        )
    ax.set_xlabel("T")
    ax.set_ylabel("Metropolis acceptance")
    ax.set_title(f"{prefix}: acceptance vs T")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot merged SLSF-rs theta result JSON files")
    parser.add_argument("result_json", type=Path, help="merged theta result JSON")
    parser.add_argument("--output-dir", type=Path, help="directory for generated PNG files")
    parser.add_argument("--prefix", help="output filename prefix; defaults to result JSON stem without .results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_json = args.result_json
    output_dir = args.output_dir or result_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or result_json.name.removesuffix(".results.json").removesuffix(".json")

    tasks = load_tasks(result_json)
    lattice_sizes, temperatures, grouped, acceptance = collect(tasks)
    if not lattice_sizes or not temperatures:
        raise SystemExit("no plottable (L, T) tasks found")

    observable_path = output_dir / f"{prefix}_observables.png"
    fluctuation_path = output_dir / f"{prefix}_fluctuations.png"
    acceptance_path = output_dir / f"{prefix}_acceptance.png"

    plot_errorbar_grid(
        observable_path,
        f"{prefix}: observables vs T (mean over samples)",
        MAIN_OBSERVABLES,
        lattice_sizes,
        temperatures,
        grouped,
        ncols=2,
    )
    plot_errorbar_grid(
        fluctuation_path,
        f"{prefix}: fluctuation observables",
        FLUCTUATION_OBSERVABLES,
        lattice_sizes,
        temperatures,
        grouped,
        ncols=2,
    )
    plot_acceptance(acceptance_path, prefix, lattice_sizes, temperatures, acceptance)

    print(f"tasks={len(tasks)}")
    print(f"L={lattice_sizes}")
    print(f"T_count={len(temperatures)}")
    print(observable_path)
    print(fluctuation_path)
    print(acceptance_path)


if __name__ == "__main__":
    main()
