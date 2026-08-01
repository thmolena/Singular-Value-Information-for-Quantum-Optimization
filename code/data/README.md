# Data provenance

The only external input is Stanford Gset G1:

- source: <https://web.stanford.edu/~yyye/yyye/Gset/G1>
- records: 800 vertices and 19,176 unit-weight edges
- SHA-256: `73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665`

The package does not redistribute G1. It derives 48 deterministic ten-vertex
induced neighbourhoods with starts `37*i mod 800` for `i=0,...,47` and then
assigns whole graph-isomorphism classes to balanced outer folds. No random
train/test split is used.

```bash
python -m pip install .
spectral-qaoa-reproduce --fetch-data --verify
```
