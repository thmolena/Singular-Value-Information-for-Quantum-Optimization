# Zero-Query Depth-One QAOA Surfaces from Local Edge Statistics

This repository implements the exact depth-one MaxCut QAOA surface formula and
tests it before learned singular-value priors. On 48 checksum-authenticated
Gset G1 neighborhoods, all 3,888 grid values agree with independent
statevectors to maximum absolute error `1.33e-15`.

The formula uses zero target objective queries, $O(mq)$ work for a $q$-point
surface after local-statistic preprocessing, and no statevector. The 48 tasks
form only 16 graph-isomorphism classes; grouped folds replace a leaky row split.

```bash
cd code
python -m pip install .
spectral-qaoa-reproduce --fetch-data --verify
python -m pytest -q tests
python scripts/validate_release.py
```

The result is restricted to unweighted depth one. Higher depth, weighted
MaxCut, continuous optimization, and hardware execution are outside scope.
