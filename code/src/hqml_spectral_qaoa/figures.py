"""Generate the six quantitative figures for the collision-aware audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "mean": "#b35c1e",
    "ridge": "#2a6fbb",
    "spectral": "#238b57",
    "closed_form": "#6a3d9a",
    "gate": "#111111",
}
LABELS = {
    "mean": "training mean",
    "ridge": "ridge",
    "spectral": "spectral prior",
    "closed_form": "exact p=1 formula",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.3,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_figures(result: dict, output: Path) -> None:
    _style()
    output.mkdir(parents=True, exist_ok=True)

    # 1. Class multiplicities and leakage in the original row-wise split.
    sizes = np.asarray(result["isomorphism_class_sizes"])
    audit = result["old_row_fold_audit"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    order = np.argsort(-sizes)
    axes[0].bar(np.arange(1, len(sizes) + 1), sizes[order], color="#4c78a8")
    axes[0].axhline(1, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("isomorphism class (decreasing size)")
    axes[0].set_ylabel("registered tasks")
    axes[0].set_xticks([1, 4, 8, 12, 16])
    folds = np.arange(1, 5)
    width = 0.24
    for offset, (key, label, color) in enumerate(
        (
            ("topology_matches", "isomorphic", "#d95f02"),
            ("feature_matches", "same features", "#7570b3"),
            ("ordered_edge_repeats", "same edge list", "#1b9e77"),
        )
    ):
        axes[1].bar(
            folds + (offset - 1) * width,
            [row[key] for row in audit],
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(folds)
    axes[1].set_xlabel("old row-wise fold")
    axes[1].set_ylabel("test tasks matched in training (of 12)")
    axes[1].set_ylim(0, 12)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    _save(fig, output, "collision_audit")

    # 2. Sensitivity of the headline one-query metric to the split contract.
    methods = ("mean", "ridge", "spectral")
    old = result["old_row_aggregate_curves"]
    grouped = result["grouped_aggregate_curves"]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(
        x - 0.19,
        [old[method]["mean"][0] for method in methods],
        0.38,
        label="row-wise split",
        color="#9ecae1",
        edgecolor="#2a6fbb",
    )
    ax.bar(
        x + 0.19,
        [grouped[method]["mean"][0] for method in methods],
        0.38,
        label="isomorphism-grouped split",
        color="#fdae6b",
        edgecolor="#b35c1e",
    )
    ax.set_xticks(x, [LABELS[method] for method in methods])
    ax.set_ylabel("mean normalized one-query regret")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    _save(fig, output, "split_sensitivity")

    # 3. Grouped-fold query curves, with the exact formula at zero regret.
    budgets = np.asarray(result["budgets"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.9, 3.1))
    for method in ("mean", "ridge", "spectral", "closed_form"):
        mean = np.asarray(grouped[method]["mean"])
        std = np.asarray(grouped[method]["std"])
        ax.plot(
            budgets,
            mean,
            marker="o",
            markersize=3.2,
            color=COLORS[method],
            label=LABELS[method],
        )
        if method != "closed_form":
            ax.fill_between(
                budgets,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
    ax.set_xscale("log", base=2)
    maximum = max(float(np.max(grouped[method]["mean"])) for method in grouped)
    ax.set_ylim(0.0, 1.12 * maximum)
    ax.set_xticks(budgets, [str(int(value)) for value in budgets])
    ax.set_xlabel("queried target objective values after ranking")
    ax.set_ylabel("normalized best-so-far regret")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.22)
    _save(fig, output, "grouped_query_curves")

    # 4. Empirical residual gate and the PSD spike non-certification theorem.
    grouped_folds = result["grouped_folds"]
    residual = np.asarray(
        [fold["operator"]["sampled_row_relative_error"] for fold in grouped_folds]
    )
    full = np.asarray(
        [fold["operator"]["full_frobenius_relative_error"] for fold in grouped_folds]
    )
    spike = result["spike_counterexample"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(folds, residual, marker="o", label="registered sampled rows")
    axes[0].plot(folds, full, marker="s", label="full 48-row audit")
    axes[0].axhline(
        result["spectral_residual_gate"],
        color=COLORS["gate"],
        ls="--",
        lw=1,
        label="5% gate",
    )
    axes[0].set_xticks(folds)
    axes[0].set_xlabel("isomorphism-grouped fold")
    axes[0].set_ylabel("relative Frobenius residual")
    axes[0].legend(frameon=False)
    tau = np.asarray([row["tau"] for row in spike])
    sampled = np.asarray([row["sampled_row_operator_error"] for row in spike])
    global_error = np.asarray([row["global_operator_error"] for row in spike])
    axes[1].plot(tau, sampled, marker="o", label="observed sampled rows")
    axes[1].plot(tau, global_error, marker="s", label="global operator error")
    axes[1].set_xlabel(r"PSD spike amplitude $\tau$")
    axes[1].set_ylabel("operator-norm difference")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    _save(fig, output, "residual_noncertification")

    # 5. Formula/statevector agreement and matched full-surface runtime.
    validation = result["formula_validation"]
    runtime = result["timing_formula_runtime"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    task_error = np.maximum(
        np.asarray(validation["per_task_max_absolute_error"]), 1e-18
    )
    axes[0].plot(np.arange(1, len(task_error) + 1), task_error, color="#6a3d9a")
    axes[0].axhline(1e-12, color="black", ls="--", lw=1, label=r"$10^{-12}$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("authenticated G1-derived task")
    axes[0].set_ylabel("max absolute surface discrepancy")
    axes[0].legend(frameon=False)
    vertices = np.asarray([row["vertices"] for row in runtime])
    formula_ms = 1000 * np.asarray([row["timing_formula_seconds"] for row in runtime])
    statevector_ms = 1000 * np.asarray(
        [row["timing_statevector_seconds"] for row in runtime]
    )
    axes[1].plot(vertices, formula_ms, marker="o", label="exact p=1 formula")
    axes[1].plot(vertices, statevector_ms, marker="s", label="statevector")
    axes[1].set_yscale("log")
    axes[1].set_xticks(vertices)
    axes[1].set_xlabel("vertices in G1-induced graph")
    axes[1].set_ylabel("81-point surface time (ms)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    _save(fig, output, "formula_validation_runtime")

    # 6. Fair dense-versus-Nystrom build-plus-apply audit on G1 vertices.
    kernel = result["timing_kernel_runtime"]
    records = np.asarray([row["records"] for row in kernel])
    dense_ms = 1000 * np.asarray([row["timing_dense_seconds"] for row in kernel])
    spectral_ms = 1000 * np.asarray(
        [row["timing_spectral_seconds"] for row in kernel]
    )
    error = np.asarray([row["application_relative_error"] for row in kernel])
    storage = np.asarray(
        [
            row["spectral_storage_scalars"] / row["dense_storage_scalars"]
            for row in kernel
        ]
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0))
    axes[0].plot(records, dense_ms, marker="o", label="dense build + apply")
    axes[0].plot(records, spectral_ms, marker="s", label="Nystrom build + apply")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("authenticated G1 vertex rows")
    axes[0].set_ylabel("matched wall time (ms)")
    axes[0].legend(frameon=False)
    axes[1].plot(records, error, marker="o", label="application error")
    axes[1].plot(records, storage, marker="s", label="stored / dense scalars")
    axes[1].axhline(0.05, color="black", ls="--", lw=1, label="5% reference")
    axes[1].set_xlabel("authenticated G1 vertex rows")
    axes[1].set_ylabel("fraction")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    _save(fig, output, "matched_kernel_runtime")
