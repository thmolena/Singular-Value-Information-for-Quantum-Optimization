"""Matrix-free spectral Nyström operators for query-priority learning.

The proposed path exposes a kernel only through requested blocks.  It stores a
truncated feature factor ``Phi`` and applies ``Phi @ (Phi.T @ B)``; it never
forms the global ``n x n`` kernel.  A deliberately small dense audit belongs
to a separately guarded routine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EntryStats:
    calls: int = 0
    queried_entries: int = 0
    peak_block_entries: int = 0


class FeatureKernelOracle:
    """Gaussian kernel over fixed feature rows, exposed only by block."""

    def __init__(self, features: np.ndarray, bandwidth: float):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("features must be a matrix")
        self.features = matrix
        self.bandwidth = float(bandwidth)
        self.stats = EntryStats()

    @property
    def n(self) -> int:
        return len(self.features)

    def block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        count = int(len(rows) * len(cols))
        self.stats.calls += 1
        self.stats.queried_entries += count
        self.stats.peak_block_entries = max(self.stats.peak_block_entries, count)
        delta = self.features[rows, None, :] - self.features[None, cols, :]
        distance = np.sum(delta * delta, axis=2)
        return np.exp(-distance / (2.0 * self.bandwidth))


def fwht_rows(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Padded orthonormal Walsh--Hadamard transform and product count."""

    source = np.asarray(values, dtype=float)
    width = 1 << max(0, int(np.ceil(np.log2(max(source.shape[1], 1)))))
    transformed = np.zeros((len(source), width), dtype=float)
    transformed[:, : source.shape[1]] = source
    step = 1
    operations = 0
    while step < width:
        for start in range(0, width, 2 * step):
            left = transformed[:, start : start + step].copy()
            right = transformed[:, start + step : start + 2 * step].copy()
            transformed[:, start : start + step] = left + right
            transformed[:, start + step : start + 2 * step] = left - right
            operations += 2 * len(source) * step
        step *= 2
    transformed /= np.sqrt(width)
    return transformed, operations


def butterfly_key(features: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a deterministic scalar ordering after a fast butterfly map."""

    matrix = np.asarray(features, dtype=float)
    transformed, work = fwht_rows(matrix)
    covariance = transformed.T @ transformed / max(len(transformed), 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return transformed @ direction, work


def chebyshev_landmarks(
    eligible: np.ndarray,
    key: np.ndarray,
    count: int,
) -> np.ndarray:
    """Choose deterministic endpoint-resolving landmarks from eligible rows."""

    eligible = np.asarray(eligible, dtype=int)
    order = eligible[np.argsort(key[eligible], kind="mergesort")]
    count = min(int(count), len(order))
    if count == len(order):
        return order
    angle = np.linspace(0.0, np.pi, count)
    raw = 0.5 * (1.0 - np.cos(angle)) * (len(order) - 1)
    positions = np.unique(np.rint(raw).astype(int))
    return order[positions]


@dataclass
class SpectralNystrom:
    phi: np.ndarray
    eigenvalues: np.ndarray
    landmarks: np.ndarray
    stats: EntryStats
    sampled_relative_error: float
    butterfly_work: int
    global_dense_matrix_materialized: bool = False

    @property
    def rank(self) -> int:
        return int(self.phi.shape[1])

    @property
    def storage(self) -> int:
        return int(self.phi.size + self.eigenvalues.size)

    def apply(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        batch = np.asarray(batch, dtype=float)
        if batch.ndim == 1:
            batch = batch[:, None]
        result = self.phi @ (self.phi.T @ batch)
        work = 2 * len(self.phi) * self.rank * batch.shape[1]
        return result, int(work)

    def metrics(self, batch_width: int) -> dict[str, float | int | bool]:
        n = len(self.phi)
        dense_work = n * n * int(batch_width)
        structured_work = 2 * n * self.rank * int(batch_width)
        return {
            "matrix_free": True,
            "global_dense_matrix_materialized": False,
            "operator_records": n,
            "spectral_rank": self.rank,
            "spectral_landmarks": len(self.landmarks),
            "sampled_row_relative_error": self.sampled_relative_error,
            "storage_scalars": self.storage,
            "storage_fraction": self.storage / max(n * n, 1),
            "matmat_products": structured_work,
            "dense_matmat_products": dense_work,
            "matmat_work_ratio": dense_work / max(structured_work, 1),
            "entry_query_calls": self.stats.calls,
            "entry_query_requests": self.stats.queried_entries,
            "entry_query_fraction": self.stats.queried_entries / max(n * n, 1),
            "peak_block_entries": self.stats.peak_block_entries,
            "peak_block_fraction": self.stats.peak_block_entries / max(n * n, 1),
            "feature_butterfly_work": self.butterfly_work,
        }


def build_spectral_nystrom(
    features: np.ndarray,
    eligible_landmarks: np.ndarray,
    *,
    rank: int,
    landmark_count: int,
    bandwidth: float,
    sample_rows: np.ndarray,
    row_chunk: int = 64,
) -> SpectralNystrom:
    """Build a spectral Nyström factor and stream a sampled-row residual."""

    matrix = np.asarray(features, dtype=float)
    n = len(matrix)
    all_rows = np.arange(n, dtype=int)
    key, butterfly_work = butterfly_key(matrix)
    landmarks = chebyshev_landmarks(
        np.asarray(eligible_landmarks, dtype=int), key, landmark_count
    )
    if rank > len(landmarks):
        raise ValueError("rank exceeds landmark count")
    oracle = FeatureKernelOracle(matrix, bandwidth)
    intersection = oracle.block(landmarks, landmarks)
    values, vectors = np.linalg.eigh(intersection)
    take = np.argsort(values)[::-1][:rank]
    retained = np.maximum(values[take], 1e-10)
    basis = vectors[:, take]
    columns = oracle.block(all_rows, landmarks)
    phi = columns @ basis / np.sqrt(retained)[None, :]

    sample_rows = np.unique(np.asarray(sample_rows, dtype=int))
    numerator = 0.0
    denominator = 0.0
    for start in range(0, n, row_chunk):
        cols = all_rows[start : min(start + row_chunk, n)]
        exact = oracle.block(sample_rows, cols)
        approximate = phi[sample_rows] @ phi[cols].T
        numerator += float(np.sum((exact - approximate) ** 2))
        denominator += float(np.sum(exact**2))
    error = float(np.sqrt(numerator / max(denominator, 1e-30)))
    return SpectralNystrom(
        phi=phi,
        eigenvalues=retained,
        landmarks=landmarks,
        stats=oracle.stats,
        sampled_relative_error=error,
        butterfly_work=butterfly_work,
    )


def spectral_surface_prediction(
    operator: SpectralNystrom,
    known: np.ndarray,
    surfaces: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Fit all query-candidate scores in the retained spectral coordinates."""

    known = np.asarray(known, dtype=int)
    phi = operator.phi[known]
    penalty = float(regularization) * np.eye(operator.rank)
    coefficient = np.linalg.solve(
        phi.T @ phi + penalty + 1e-12 * np.eye(operator.rank),
        phi.T @ np.asarray(surfaces, dtype=float)[known],
    )
    return operator.phi @ coefficient


def dense_kernel_audit(
    features: np.ndarray,
    *,
    maximum_records: int = 128,
    bandwidth: float,
) -> np.ndarray:
    """Materialize a guarded dense kernel for an explicitly named audit."""

    matrix = np.asarray(features, dtype=float)
    if len(matrix) > maximum_records:
        raise ValueError("dense audit is guarded above 128 records")
    delta = matrix[:, None, :] - matrix[None, :, :]
    return np.exp(-np.sum(delta * delta, axis=2) / (2.0 * bandwidth))


def streamed_exact_apply(
    features: np.ndarray,
    batch: np.ndarray,
    *,
    bandwidth: float,
    row_chunk: int = 256,
) -> tuple[np.ndarray, EntryStats]:
    """Apply the exact kernel in bounded row blocks for a timing reference.

    This reference avoids a resident global matrix, but it still requests all
    ``n^2`` entries.  It is deliberately separate from the proposed path.
    """

    matrix = np.asarray(features, dtype=float)
    right = np.asarray(batch, dtype=float)
    if right.ndim == 1:
        right = right[:, None]
    if len(right) != len(matrix):
        raise ValueError("batch and feature row counts differ")
    oracle = FeatureKernelOracle(matrix, bandwidth)
    output = np.empty((len(matrix), right.shape[1]), dtype=float)
    columns = np.arange(len(matrix), dtype=int)
    for start in range(0, len(matrix), row_chunk):
        rows = np.arange(start, min(start + row_chunk, len(matrix)), dtype=int)
        output[rows] = oracle.block(rows, columns) @ right
    return output, oracle.stats
