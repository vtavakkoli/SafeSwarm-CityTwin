"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    agg = df.groupby("strategy", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg["strategy"], agg[metric], color=["#6baed6", "#74c476", "#fd8d3c", "#9e9ac8"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} by strategy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_trajectories_with_restricted_zones(env, trajectories: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    if env.restricted_zones:
        rx, ry = zip(*env.restricted_zones)
        ax.scatter(rx, ry, c="red", s=10, alpha=0.5, label="restricted")

    for aid, tr in trajectories.items():
        if not tr:
            continue
        x, y = zip(*tr)
        ax.plot(x, y, linewidth=1.2, alpha=0.8, label=f"agent {aid}")

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_title("Trajectories with restricted zones")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
