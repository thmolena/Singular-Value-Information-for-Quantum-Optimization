# hqml-spectral-qaoa

This wheel reproduces the collision-aware negative audit in the manuscript.
It authenticates Stanford Gset G1, derives 48 deterministic ten-vertex tasks,
checks exact graph isomorphism, evaluates all 3,888 depth-one graph--angle
pairs by both a statevector and the established closed form, and runs the
grouped-fold baselines.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
spectral-qaoa-reproduce --fetch-data --verify
python -m pytest -q tests
```

`--fetch-data` accepts only the registered Stanford HTTPS host and verifies
the pinned SHA-256 digest before caching the file outside the package. Use
`--g1-path /path/to/G1` or `HQML_G1_PATH` for an existing copy. The verifier
recomputes scientific fields and ignores equality only for keys prefixed by
`timing_` and the runtime-environment record.

## What is tested

- 16 first-seen graph-isomorphism classes and balanced 12-task outer folds;
- the old split's topology, feature, and ordered-edge collision counts;
- exact formula/statevector agreement over 48 tasks by 81 angle pairs;
- grouped one-query regrets and two failures of the 5% residual gate;
- the positive-semidefinite spike counterexample and ranking bound;
- matched construction-plus-application timing metadata;
- the six-figure, four-table artifact and SHA-256 manifest.

## Files

- `src/hqml_spectral_qaoa/data.py`: authenticated G1 resolution.
- `src/hqml_spectral_qaoa/experiment.py`: task construction, formula,
  statevector, grouped evaluation, timings, tables, and manifests.
- `src/hqml_spectral_qaoa/matrix_free.py`: audited Gaussian-kernel and
  Nystrom routines.
- `src/hqml_spectral_qaoa/figures.py`: six quantitative figures.
- `results/`: locked JSON, TeX rows, figures, and content hashes.
- `tests/test_reproduction.py`: independent contracts.

The exact $p=1$ baseline uses no target quantum-objective evaluations. The
timing audit is classical CPU evidence and does not establish quantum
advantage, end-to-end training speedup, or higher-depth behaviour.
