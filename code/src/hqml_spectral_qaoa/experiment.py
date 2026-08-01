"""Collision-aware negative audit of singular-value priors for depth-one QAOA.

The experiment has one deliberately narrow target: unweighted MaxCut at
depth one.  It authenticates Stanford Gset G1, derives forty-eight registered
ten-vertex tasks, groups graph-isomorphic tasks before splitting, and compares
learned surface rankings with the established exact depth-one formula of Wang
et al.  Timings always include construction and application for both paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path

import networkx as nx
import numpy as np

from .data import SOURCES, authenticated_gset
from .matrix_free import build_spectral_nystrom, dense_kernel_audit

SOURCE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RESULTS = SOURCE_ROOT / "results"
WORKTREE_RESULTS = CODE_ROOT / "results"

CONFIG = {
    "seed": 1202,
    "package": "hqml_spectral_qaoa",
    "title": (
        "Collision Leakage and an Exact Depth-One Baseline: "
        "A Negative Audit of Singular-Value Priors for QAOA MaxCut"
    ),
}
DATA_SHA256 = SOURCES["G1"].sha256
BUDGETS = np.array([1, 2, 4, 8, 12, 20, 40, 81], dtype=int)
SELECTION_COLUMNS = np.array([0, 1, 2], dtype=int)
REGULARIZATION_GRID = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0)
SPECTRAL_RANK = 6
LANDMARK_COUNT = 8
KERNEL_BANDWIDTH = 4.0
OPERATOR_TOLERANCE = 0.05


def parse_gset(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    """Parse a Gset edge list after the caller has authenticated its bytes."""

    with path.open(encoding="utf-8") as handle:
        n, m = map(int, handle.readline().split()[:2])
        edges = []
        for line in handle:
            u, v, weight = line.split()
            edges.append((int(u) - 1, int(v) - 1, float(weight)))
    if len(edges) != m:
        raise ValueError(f"Gset header declares {m} edges but parsed {len(edges)}")
    if any(u < 0 or v < 0 or u >= n or v >= n for u, v, _ in edges):
        raise ValueError("Gset endpoint is outside the declared vertex range")
    if any(u == v for u, v, _ in edges):
        raise ValueError("Gset input contains a self-loop")
    return n, edges


def bfs_subgraph(adj: list[list[int]], start: int, size: int) -> list[int]:
    """Return the first ``size`` vertices in a deterministic BFS."""

    seen = {start}
    queue = [start]
    for u in queue:
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if len(queue) == size:
                    return queue
    for v in range(len(adj)):
        if v not in seen:
            queue.append(v)
            if len(queue) == size:
                break
    return queue


def induced_edges(
    adjacency: list[list[int]], vertices: list[int]
) -> list[tuple[int, int]]:
    local = {vertex: index for index, vertex in enumerate(vertices)}
    return [
        (local[u], local[v])
        for u in vertices
        for v in adjacency[u]
        if u < v and v in local
    ]


def statevector_surface(
    edges: list[tuple[int, int]],
    n: int,
    gamma_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate the complete depth-one surface by an exact statevector.

    The cost vector is constructed once, so the timing comparator includes
    matched setup rather than rebuilding the diagonal Hamiltonian per angle.
    """

    states = np.arange(1 << n, dtype=np.uint32)
    cost = np.zeros(len(states), dtype=float)
    for u, v in edges:
        cost += ((states >> u) ^ (states >> v)) & 1
    output = []
    normalization = math.sqrt(len(states))
    for gamma in gamma_grid:
        phase_state = np.exp(-1j * float(gamma) * cost) / normalization
        for beta in beta_grid:
            state = phase_state.copy()
            cosine, sine = math.cos(float(beta)), -1j * math.sin(float(beta))
            for qubit in range(n):
                block = 1 << qubit
                for start in range(0, len(states), 2 * block):
                    left = state[start : start + block].copy()
                    right = state[start + block : start + 2 * block].copy()
                    state[start : start + block] = cosine * left + sine * right
                    state[start + block : start + 2 * block] = sine * left + cosine * right
            output.append(float(np.dot(np.abs(state) ** 2, cost)) / len(edges))
    return np.asarray(output)


def exact_p1_surface(
    edges: list[tuple[int, int]],
    n: int,
    gamma_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate Wang et al.'s exact p=1 MaxCut formula.

    Degrees ``du`` and ``dv`` exclude the active edge; ``lam`` is the number
    of common neighbours.  The output is normalized by the edge count.
    """

    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    edge_statistics = [
        (len(adjacency[u]) - 1, len(adjacency[v]) - 1, len(adjacency[u] & adjacency[v]))
        for u, v in edges
    ]
    output = []
    for gamma in gamma_grid:
        sine_gamma = math.sin(float(gamma))
        cosine_gamma = math.cos(float(gamma))
        cosine_two_gamma = math.cos(2.0 * float(gamma))
        for beta in beta_grid:
            total = 0.0
            for du, dv, lam in edge_statistics:
                total += (
                    0.5
                    + 0.25
                    * math.sin(4.0 * float(beta))
                    * sine_gamma
                    * (cosine_gamma**du + cosine_gamma**dv)
                    - 0.25
                    * math.sin(2.0 * float(beta)) ** 2
                    * cosine_gamma ** (du + dv - 2 * lam)
                    * (1.0 - cosine_two_gamma**lam)
                )
            output.append(total / len(edges))
    return np.asarray(output)


def graph_features(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1.0
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    spectrum = np.linalg.eigvalsh(laplacian)
    triangles = np.trace(adjacency @ adjacency @ adjacency) / 6.0
    return np.array(
        [
            1.0,
            len(edges) / max(n * (n - 1) / 2, 1),
            degree.mean(),
            degree.std(),
            degree.max(),
            triangles,
            spectrum[1] if len(spectrum) > 1 else 0.0,
            spectrum[-1],
        ],
        dtype=float,
    )


def _gset_adjacency(
    path: Path,
) -> tuple[int, list[tuple[int, int, float]], list[list[int]]]:
    n, weighted_edges = parse_gset(path)
    adjacency = [[] for _ in range(n)]
    for u, v, _ in weighted_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbours in adjacency:
        neighbours.sort()
    return n, weighted_edges, adjacency


def build_qaoa_dataset(path: Path) -> dict:
    """Build the registered tasks and independently evaluate both surfaces."""

    n, weighted_edges, adjacency = _gset_adjacency(path)
    gamma_grid = np.linspace(0.08, math.pi - 0.08, 9)
    beta_grid = np.linspace(0.04, math.pi / 2.0 - 0.04, 9)
    features = []
    statevector_surfaces = []
    formula_surfaces = []
    edge_sets = []
    starts = []
    per_task_error = []
    for sample in range(48):
        start = (sample * 37) % n
        vertices = bfs_subgraph(adjacency, start, 10)
        edges = induced_edges(adjacency, vertices)
        if not edges:
            raise ValueError("registered neighbourhood has no edges")
        formula = exact_p1_surface(edges, 10, gamma_grid, beta_grid)
        statevector = statevector_surface(edges, 10, gamma_grid, beta_grid)
        features.append(graph_features(edges, 10))
        statevector_surfaces.append(statevector)
        formula_surfaces.append(formula)
        edge_sets.append(edges)
        starts.append(start)
        per_task_error.append(float(np.max(np.abs(formula - statevector))))
    return {
        "features": np.asarray(features),
        "surfaces": np.asarray(statevector_surfaces),
        "formula_surfaces": np.asarray(formula_surfaces),
        "formula_per_task_max_error": np.asarray(per_task_error),
        "edge_sets": edge_sets,
        "starts": starts,
        "adjacency": adjacency,
        "weighted_edges": weighted_edges,
        "gamma_grid": gamma_grid,
        "beta_grid": beta_grid,
        "records": n,
    }


def query_regret(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ordering = np.argsort(-predicted, axis=1, kind="mergesort")
    optimum = truth.max(axis=1)
    curves = []
    for budget in BUDGETS:
        chosen = np.take_along_axis(truth, ordering[:, :budget], axis=1)
        best = chosen.max(axis=1)
        curves.append(np.maximum(optimum - best, 0.0) / np.maximum(optimum, 1e-12))
    return np.column_stack(curves)


def _selection_score(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(query_regret(predicted, truth)[:, SELECTION_COLUMNS]))


def _standardize(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean = features[train].mean(axis=0)
    std = features[train].std(axis=0)
    std[std < 1e-9] = 1.0
    return (features - mean) / std


def _ridge_predictions(
    features: np.ndarray,
    surfaces: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, float]:
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    candidates = []
    for regularization in REGULARIZATION_GRID:
        coefficient = np.linalg.solve(
            design[fit].T @ design[fit]
            + regularization * penalty
            + 1e-12 * np.eye(design.shape[1]),
            design[fit].T @ surfaces[fit],
        )
        candidates.append(
            (
                _selection_score(
                    design[development] @ coefficient, surfaces[development]
                ),
                float(regularization),
            )
        )
    regularization = min(candidates)[1]
    coefficient = np.linalg.solve(
        design[train].T @ design[train]
        + regularization * penalty
        + 1e-12 * np.eye(design.shape[1]),
        design[train].T @ surfaces[train],
    )
    return design @ coefficient, regularization


def _spectral_predictions(
    features: np.ndarray,
    surfaces: np.ndarray,
    fit: np.ndarray,
    development: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    development_operator = build_spectral_nystrom(
        features,
        fit,
        rank=SPECTRAL_RANK,
        landmark_count=LANDMARK_COUNT,
        bandwidth=KERNEL_BANDWIDTH,
        sample_rows=development,
        row_chunk=16,
    )
    candidates = []
    for regularization in REGULARIZATION_GRID:
        phi = development_operator.phi[fit]
        coefficient = np.linalg.solve(
            phi.T @ phi
            + regularization * np.eye(development_operator.rank)
            + 1e-12 * np.eye(development_operator.rank),
            phi.T @ surfaces[fit],
        )
        prediction = development_operator.phi @ coefficient
        candidates.append(
            (
                _selection_score(prediction[development], surfaces[development]),
                float(regularization),
            )
        )
    regularization = min(candidates)[1]
    operator = build_spectral_nystrom(
        features,
        train,
        rank=SPECTRAL_RANK,
        landmark_count=LANDMARK_COUNT,
        bandwidth=KERNEL_BANDWIDTH,
        sample_rows=test,
        row_chunk=16,
    )
    phi_train = operator.phi[train]
    coefficient = np.linalg.solve(
        phi_train.T @ phi_train
        + regularization * np.eye(operator.rank)
        + 1e-12 * np.eye(operator.rank),
        phi_train.T @ surfaces[train],
    )
    prediction = operator.phi @ coefficient
    metrics = operator.metrics(batch_width=surfaces.shape[1])
    exact = dense_kernel_audit(
        features, maximum_records=128, bandwidth=KERNEL_BANDWIDTH
    )
    approximate = operator.phi @ operator.phi.T
    metrics.update(
        {
            "sampled_row_relative_error": operator.sampled_relative_error,
            "development_sampled_row_relative_error": (
                development_operator.sampled_relative_error
            ),
            "full_frobenius_relative_error": float(
                np.linalg.norm(exact - approximate) / np.linalg.norm(exact)
            ),
            "full_spectral_relative_error": float(
                np.linalg.norm(exact - approximate, 2) / np.linalg.norm(exact, 2)
            ),
            "operator_tolerance": OPERATOR_TOLERANCE,
            "operator_tolerance_pass": bool(
                operator.sampled_relative_error <= OPERATOR_TOLERANCE
            ),
        }
    )
    return prediction, regularization, metrics


def isomorphism_groups(edge_sets: list[list[tuple[int, int]]]) -> list[list[int]]:
    """Group tasks by exact graph isomorphism in first-seen order."""

    graphs = []
    for edges in edge_sets:
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from(edges)
        graphs.append(graph)
    groups: list[list[int]] = []
    for index, graph in enumerate(graphs):
        for group in groups:
            if nx.is_isomorphic(graph, graphs[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def balanced_group_folds(groups: list[list[int]]) -> list[list[list[int]]]:
    """Assign whole isomorphism groups by decreasing size to four folds."""

    folds: list[list[list[int]]] = [[] for _ in range(4)]
    loads = [0, 0, 0, 0]
    for group in sorted(groups, key=lambda values: (-len(values), values[0])):
        fold = min(range(4), key=lambda index: (loads[index], index))
        folds[fold].append(group)
        loads[fold] += len(group)
    return folds


def old_row_fold_audit(
    edge_sets: list[list[tuple[int, int]]], features: np.ndarray
) -> list[dict]:
    # The feature audit is descriptive rather than an arithmetic-identity test.
    # Quantize the two Laplacian eigenvalue coordinates before comparison so
    # that equal graph descriptors remain equal across compliant eigensolvers.
    feature_keys = [tuple(np.round(row, decimals=12)) for row in features]
    graphs = []
    for edges in edge_sets:
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from(edges)
        graphs.append(graph)
    rows = []
    all_indices = np.arange(len(edge_sets))
    for fold in range(4):
        test = all_indices[all_indices % 4 == fold]
        train = all_indices[all_indices % 4 != fold]
        topology = sum(
            any(nx.is_isomorphic(graphs[i], graphs[j]) for j in train) for i in test
        )
        feature = sum(
            any(feature_keys[i] == feature_keys[j] for j in train) for i in test
        )
        ordered = sum(
            any(edge_sets[i] == edge_sets[j] for j in train) for i in test
        )
        rows.append(
            {
                "fold": fold + 1,
                "test_tasks": len(test),
                "topology_matches": int(topology),
                "feature_matches": int(feature),
                "ordered_edge_repeats": int(ordered),
            }
        )
    return rows


def _evaluate_split(
    features: np.ndarray,
    surfaces: np.ndarray,
    formula_surfaces: np.ndarray,
    *,
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    development: np.ndarray,
) -> dict:
    fit = np.setdiff1d(train, development)
    standardized = _standardize(features, train)
    mean_prediction = np.tile(surfaces[train].mean(axis=0), (len(features), 1))
    ridge_prediction, ridge_regularization = _ridge_predictions(
        standardized, surfaces, fit, development, train
    )
    spectral_prediction, spectral_regularization, operator = _spectral_predictions(
        standardized, surfaces, fit, development, train, test
    )
    task_curves = {
        "mean": query_regret(mean_prediction[test], surfaces[test]),
        "ridge": query_regret(ridge_prediction[test], surfaces[test]),
        "spectral": query_regret(spectral_prediction[test], surfaces[test]),
        "closed_form": query_regret(formula_surfaces[test], surfaces[test]),
    }
    return {
        "fold": fold + 1,
        "fit_indices": fit.tolist(),
        "development_indices": development.tolist(),
        "test_indices": test.tolist(),
        "fit_instances": len(fit),
        "development_instances": len(development),
        "train_instances": len(train),
        "test_instances": len(test),
        "ridge_regularization": ridge_regularization,
        "spectral_regularization": spectral_regularization,
        "mean_curves": {
            name: values.mean(axis=0).tolist() for name, values in task_curves.items()
        },
        "std_curves": {
            name: values.std(axis=0).tolist() for name, values in task_curves.items()
        },
        "operator": operator,
    }


def grouped_evaluation(
    features: np.ndarray,
    surfaces: np.ndarray,
    formula_surfaces: np.ndarray,
    fold_groups: list[list[list[int]]],
) -> list[dict]:
    all_indices = np.arange(len(features))
    folds = []
    for fold in range(4):
        test = np.asarray(sorted(i for group in fold_groups[fold] for i in group))
        train = np.setdiff1d(all_indices, test)
        train_groups = [
            group
            for other_fold in range(4)
            if other_fold != fold
            for group in fold_groups[other_fold]
        ]
        development = np.asarray(
            sorted(i for group in train_groups[::4] for i in group), dtype=int
        )
        folds.append(
            _evaluate_split(
                features,
                surfaces,
                formula_surfaces,
                fold=fold,
                train=train,
                test=test,
                development=development,
            )
        )
    return folds


def row_fold_evaluation(
    features: np.ndarray,
    surfaces: np.ndarray,
    formula_surfaces: np.ndarray,
) -> list[dict]:
    indices = np.arange(len(features))
    folds = []
    for fold in range(4):
        test = indices[indices % 4 == fold]
        train = indices[indices % 4 != fold]
        development = train[np.arange(len(train)) % 5 == fold % 5]
        folds.append(
            _evaluate_split(
                features,
                surfaces,
                formula_surfaces,
                fold=fold,
                train=train,
                test=test,
                development=development,
            )
        )
    return folds


def _aggregate_curves(folds: list[dict]) -> dict:
    output = {}
    for method in ("mean", "ridge", "spectral", "closed_form"):
        values = np.asarray([fold["mean_curves"][method] for fold in folds])
        output[method] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }
    return output


def vertex_features(adjacency: list[list[int]]) -> np.ndarray:
    degree = np.array([len(neighbours) for neighbours in adjacency], dtype=float)
    neighbour_sets = [set(neighbours) for neighbours in adjacency]
    rows = []
    for vertex, neighbours in enumerate(adjacency):
        if neighbours:
            neighbour_degree = degree[neighbours]
            links = sum(
                1
                for offset, u in enumerate(neighbours)
                for v in neighbours[offset + 1 :]
                if v in neighbour_sets[u]
            )
            possible = len(neighbours) * (len(neighbours) - 1) / 2
            rows.append(
                [
                    degree[vertex],
                    neighbour_degree.mean(),
                    neighbour_degree.std(),
                    links / max(possible, 1),
                ]
            )
        else:
            rows.append([0.0, 0.0, 0.0, 0.0])
    return np.asarray(rows)


def formula_runtime_audit(
    adjacency: list[list[int]], gamma_grid: np.ndarray, beta_grid: np.ndarray
) -> list[dict]:
    rows = []
    for n in (6, 8, 10, 12):
        edges = induced_edges(adjacency, bfs_subgraph(adjacency, 0, n))
        exact_p1_surface(edges, n, gamma_grid, beta_grid)
        statevector_surface(edges, n, gamma_grid, beta_grid)
        analytic_times = []
        statevector_times = []
        errors = []
        for _ in range(5):
            start = time.perf_counter()
            analytic = exact_p1_surface(edges, n, gamma_grid, beta_grid)
            analytic_times.append(time.perf_counter() - start)
        for _ in range(3):
            start = time.perf_counter()
            statevector = statevector_surface(edges, n, gamma_grid, beta_grid)
            statevector_times.append(time.perf_counter() - start)
            errors.append(float(np.max(np.abs(analytic - statevector))))
        analytic_seconds = float(np.median(analytic_times))
        statevector_seconds = float(np.median(statevector_times))
        rows.append(
            {
                "vertices": n,
                "edges": len(edges),
                "angle_pairs": len(gamma_grid) * len(beta_grid),
                "timing_formula_seconds": analytic_seconds,
                "timing_statevector_seconds": statevector_seconds,
                "timing_statevector_over_formula": (
                    statevector_seconds / max(analytic_seconds, 1e-15)
                ),
                "max_absolute_error": max(errors),
            }
        )
    return rows


def kernel_runtime_audit(adjacency: list[list[int]]) -> list[dict]:
    """Match dense and Nyström construction plus one 16-column application."""

    raw = vertex_features(adjacency)
    rows = []
    rng = np.random.default_rng(CONFIG["seed"] + 91)
    for records in (100, 200, 400, 800):
        features = raw[:records].copy()
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        features = (features - features.mean(axis=0)) / scale
        batch = rng.normal(size=(records, 16))
        indices = np.arange(records)
        sample_rows = np.unique(
            np.linspace(0, records - 1, min(32, records)).round().astype(int)
        )

        def dense_path() -> np.ndarray:
            delta = features[:, None, :] - features[None, :, :]
            kernel = np.exp(
                -np.sum(delta * delta, axis=2) / (2.0 * KERNEL_BANDWIDTH)
            )
            return kernel @ batch

        def spectral_path():
            operator = build_spectral_nystrom(
                features,
                indices,
                rank=16,
                landmark_count=min(64, records),
                bandwidth=KERNEL_BANDWIDTH,
                sample_rows=sample_rows,
                row_chunk=64,
            )
            approximation, _ = operator.apply(batch)
            return operator, approximation

        dense_path()
        spectral_path()
        dense_times = []
        spectral_times = []
        exact = None
        approximation = None
        operator = None
        for repeat in range(7):
            if repeat % 2 == 0:
                start = time.perf_counter()
                exact = dense_path()
                dense_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                operator, approximation = spectral_path()
                spectral_times.append(time.perf_counter() - start)
            else:
                start = time.perf_counter()
                operator, approximation = spectral_path()
                spectral_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                exact = dense_path()
                dense_times.append(time.perf_counter() - start)
        if exact is None or approximation is None or operator is None:
            raise RuntimeError("timing audit did not execute")
        dense_seconds = float(np.median(dense_times))
        spectral_seconds = float(np.median(spectral_times))
        rows.append(
            {
                "records": records,
                "feature_dimension": features.shape[1],
                "batch_width": batch.shape[1],
                "rank": operator.rank,
                "landmarks": len(operator.landmarks),
                "timing_dense_seconds": dense_seconds,
                "timing_spectral_seconds": spectral_seconds,
                "timing_dense_over_spectral": (
                    dense_seconds / max(spectral_seconds, 1e-15)
                ),
                "application_relative_error": float(
                    np.linalg.norm(approximation - exact) / np.linalg.norm(exact)
                ),
                "sampled_row_relative_error": operator.sampled_relative_error,
                "dense_storage_scalars": records * records,
                "spectral_storage_scalars": operator.storage,
            }
        )
    return rows


def spike_counterexample() -> list[dict]:
    """PSD matrices agreeing on sampled rows but diverging elsewhere."""

    n = 16
    sampled = np.arange(8)
    j = 12
    base = np.eye(n)
    rows = []
    for tau in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        perturbed = base.copy()
        perturbed[j, j] += tau
        sampled_error = np.linalg.norm((perturbed - base)[sampled])
        global_error = np.linalg.norm(perturbed - base, 2)
        rows.append(
            {
                "tau": tau,
                "sampled_row_operator_error": float(sampled_error),
                "global_operator_error": float(global_error),
            }
        )
    return rows


def reproduce(*, g1_path: str | Path | None = None, fetch: bool = False) -> dict:
    g1 = authenticated_gset("G1", explicit=g1_path, fetch=fetch)
    data = build_qaoa_dataset(g1)
    groups = isomorphism_groups(data["edge_sets"])
    fold_groups = balanced_group_folds(groups)
    grouped_folds = grouped_evaluation(
        data["features"], data["surfaces"], data["formula_surfaces"], fold_groups
    )
    row_folds = row_fold_evaluation(
        data["features"], data["surfaces"], data["formula_surfaces"]
    )
    grouped_curves = _aggregate_curves(grouped_folds)
    row_curves = _aggregate_curves(row_folds)
    residual_passes = sum(
        fold["operator"]["operator_tolerance_pass"] for fold in grouped_folds
    )
    return {
        "schema_version": 5,
        "seed": CONFIG["seed"],
        "package": CONFIG["package"],
        "title": CONFIG["title"],
        "dataset": "Stanford Gset G1",
        "dataset_url": SOURCES["G1"].url,
        "dataset_sha256": DATA_SHA256,
        "records": int(data["records"]),
        "edges": len(data["weighted_edges"]),
        "unique_edge_weights": sorted({weight for _, _, weight in data["weighted_edges"]}),
        "learning_instances": len(data["surfaces"]),
        "query_candidates": data["surfaces"].shape[1],
        "budgets": BUDGETS.tolist(),
        "isomorphism_class_count": len(groups),
        "isomorphism_class_sizes": [len(group) for group in groups],
        "isomorphism_groups": groups,
        "grouped_fold_loads": [sum(map(len, fold)) for fold in fold_groups],
        "old_row_fold_audit": old_row_fold_audit(
            data["edge_sets"], data["features"]
        ),
        "old_row_folds": row_folds,
        "old_row_aggregate_curves": row_curves,
        "grouped_folds": grouped_folds,
        "grouped_aggregate_curves": grouped_curves,
        "formula_validation": {
            "values_checked": int(data["surfaces"].size),
            "max_absolute_error": float(
                np.max(data["formula_per_task_max_error"])
            ),
            "per_task_max_absolute_error": (
                data["formula_per_task_max_error"].tolist()
            ),
            "target_quantum_objective_queries": 0,
            "classical_evaluation_work": "O(precompute + m q)",
            "classical_storage": "O(n + m + q), or O(n + m) when streamed",
        },
        "spectral_residual_gate": OPERATOR_TOLERANCE,
        "spectral_residual_passes": int(residual_passes),
        "spectral_residual_total_folds": len(grouped_folds),
        "timing_formula_runtime": formula_runtime_audit(
            data["adjacency"], data["gamma_grid"], data["beta_grid"]
        ),
        "timing_kernel_runtime": kernel_runtime_audit(data["adjacency"]),
        "spike_counterexample": spike_counterexample(),
        "runtime_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "timing_scope": "matched construction plus evaluation/application",
        },
    }


def rounded(value):
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.15g}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def _write_tex(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    curves = result["grouped_aggregate_curves"]
    error_mantissa, error_exponent = (
        f"{result['formula_validation']['max_absolute_error']:.2e}".split("e")
    )
    error_tex = rf"\ensuremath{{{error_mantissa}\times 10^{{{int(error_exponent)}}}}}"
    macros = [
        rf"\newcommand{{\DatasetName}}{{{result['dataset']}}}",
        rf"\newcommand{{\TaskCount}}{{{result['learning_instances']}}}",
        rf"\newcommand{{\ClassCount}}{{{result['isomorphism_class_count']}}}",
        rf"\newcommand{{\FormulaChecks}}{{{result['formula_validation']['values_checked']}}}",
        rf"\newcommand{{\FormulaMaxError}}{{{error_tex}}}",
        rf"\newcommand{{\ResidualPasses}}{{{result['spectral_residual_passes']}}}",
        rf"\newcommand{{\GroupedMeanOne}}{{{curves['mean']['mean'][0]:.6f}}}",
        rf"\newcommand{{\GroupedRidgeOne}}{{{curves['ridge']['mean'][0]:.6f}}}",
        rf"\newcommand{{\GroupedSpectralOne}}{{{curves['spectral']['mean'][0]:.6f}}}",
        rf"\newcommand{{\ClosedFormOne}}{{{curves['closed_form']['mean'][0]:.1f}}}",
    ]
    (output / "results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    collision_rows = []
    for row in result["old_row_fold_audit"]:
        collision_rows.append(
            f"{row['fold']} & {row['test_tasks']} & {row['topology_matches']} & "
            f"{row['feature_matches']} & {row['ordered_edge_repeats']}"
        )
    (tables / "collision_rows.tex").write_text(
        " \\\\\n".join(collision_rows) + "\n", encoding="utf-8"
    )

    fold_rows = []
    for fold in result["grouped_folds"]:
        residual = fold["operator"]["sampled_row_relative_error"]
        fold_rows.append(
            f"{fold['fold']} & {fold['fit_instances']}/{fold['development_instances']}/"
            f"{fold['test_instances']} & {fold['mean_curves']['mean'][0]:.6f} & "
            f"{fold['mean_curves']['ridge'][0]:.6f} & "
            f"{fold['mean_curves']['spectral'][0]:.6f} & "
            f"{fold['mean_curves']['closed_form'][0]:.6f} & {residual:.4f} & "
            f"{'pass' if residual <= result['spectral_residual_gate'] else 'fail'}"
        )
    (tables / "grouped_fold_rows.tex").write_text(
        " \\\\\n".join(fold_rows) + "\n", encoding="utf-8"
    )

    formula_rows = []
    for row in result["timing_formula_runtime"]:
        formula_rows.append(
            f"{row['vertices']} & {row['edges']} & "
            f"{1000 * row['timing_formula_seconds']:.3f} & "
            f"{1000 * row['timing_statevector_seconds']:.3f} & "
            f"{row['timing_statevector_over_formula']:.1f} & "
            f"{row['max_absolute_error']:.1e}"
        )
    (tables / "formula_runtime_rows.tex").write_text(
        " \\\\\n".join(formula_rows) + "\n", encoding="utf-8"
    )

    kernel_rows = []
    for row in result["timing_kernel_runtime"]:
        kernel_rows.append(
            f"{row['records']} & {1000 * row['timing_dense_seconds']:.3f} & "
            f"{1000 * row['timing_spectral_seconds']:.3f} & "
            f"{row['timing_dense_over_spectral']:.2f} & "
            f"{row['application_relative_error']:.4f} & "
            f"{100 * row['spectral_storage_scalars'] / row['dense_storage_scalars']:.2f}"
        )
    (tables / "kernel_runtime_rows.tex").write_text(
        " \\\\\n".join(kernel_rows) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    payload = {
        "schema_version": 1,
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_outputs(result: dict) -> None:
    from .figures import write_figures

    result = rounded(result)
    output = WORKTREE_RESULTS
    output.mkdir(parents=True, exist_ok=True)
    for child in list(output.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_tex(result, output)
    write_figures(result, output / "figures")
    _write_manifest(output)

    if PACKAGE_RESULTS.resolve() != output.resolve():
        if PACKAGE_RESULTS.exists():
            shutil.rmtree(PACKAGE_RESULTS)
        shutil.copytree(output, PACKAGE_RESULTS)


def _compare(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            if str(key).startswith("timing_") or key == "runtime_environment":
                continue
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-10):
            raise AssertionError(f"{path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def verify(*, g1_path: str | Path | None = None, fetch: bool = False) -> dict:
    actual = rounded(reproduce(g1_path=g1_path, fetch=fetch))
    locked = json.loads((PACKAGE_RESULTS / "results.json").read_text(encoding="utf-8"))
    _compare(actual, locked)
    if actual["isomorphism_class_count"] != 16:
        raise AssertionError("registered tasks no longer form sixteen classes")
    if actual["grouped_fold_loads"] != [12, 12, 12, 12]:
        raise AssertionError("grouped folds are no longer balanced")
    if actual["formula_validation"]["max_absolute_error"] > 1e-12:
        raise AssertionError("closed form disagrees with the exact statevector")
    if actual["unique_edge_weights"] != [1.0]:
        raise AssertionError("registered exact formula requires unit G1 weights")
    if actual["formula_validation"]["target_quantum_objective_queries"] != 0:
        raise AssertionError("closed-form baseline must use zero target queries")
    if actual["spectral_residual_passes"] != 2:
        raise AssertionError("registered 5% residual gate must pass exactly two folds")
    if actual["grouped_aggregate_curves"]["closed_form"]["mean"][0] > 1e-12:
        raise AssertionError("closed-form baseline must attain zero one-query regret")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fetch-data", action="store_true")
    parser.add_argument("--g1-path", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.write:
        result = reproduce(g1_path=arguments.g1_path, fetch=arguments.fetch_data)
        write_outputs(result)
        print(json.dumps(rounded(result), indent=2, sort_keys=True))
        return 0
    if arguments.verify:
        result = verify(g1_path=arguments.g1_path, fetch=arguments.fetch_data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = rounded(reproduce(g1_path=arguments.g1_path, fetch=arguments.fetch_data))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
