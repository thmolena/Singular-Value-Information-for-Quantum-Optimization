"""Runtime scaling and the zero-query comparison against learned angle priors."""
from __future__ import annotations
import json, time, pathlib
import numpy as np, networkx as nx
from sklearn.linear_model import Ridge
from svi.core import (load_gset, edge_triples, statistic, surface, grid,
                      statevector_surface)
from svi.corpus import build, unweighted_gset

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results_full"
R = json.loads((OUT / "results.json").read_text())
gq, bq = grid(9)

# ------------------------------------------------ E6 runtime scaling
print("E6 runtime", flush=True)
rows = []
for n in range(8, 19, 2):
    G = nx.random_regular_graph(3, n, seed=11 * n)
    e = [tuple(map(int, x)) for x in G.edges()]
    t0 = time.perf_counter()
    st = statistic(edge_triples(n, e)); f = surface(st, gq, bq)
    t_f = time.perf_counter() - t0
    t0 = time.perf_counter(); s = statevector_surface(n, e, gq, bq)
    t_s = time.perf_counter() - t0
    assert np.abs(f - s).max() < 1e-10
    rows.append({"n": n, "m": len(e), "formula_s": t_f, "statevector_s": t_s})
    print("  n=%2d  formula %.5fs  statevector %.4fs" % (n, t_f, t_s), flush=True)
R["runtime_small"] = rows

print("E6b full-size Gset", flush=True)
big = []
for name, (n, e) in unweighted_gset().items():
    t0 = time.perf_counter(); st = statistic(edge_triples(n, e))
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter(); surface(st, gq, bq)
    t_ev = time.perf_counter() - t0
    big.append({"name": name, "n": n, "m": len(e), "unique_triples": int(len(st[0])),
                "preprocess_s": t_pre, "surface_s": t_ev})
R["runtime_gset"] = big
b = max(big, key=lambda d: d["m"])
print("  largest: %s n=%d m=%d  pre %.2fs  surface %.4fs (%d unique triples)"
      % (b["name"], b["n"], b["m"], b["preprocess_s"], b["surface_s"], b["unique_triples"]), flush=True)

# ---------------------------- E7 zero-query vs learned angle priors
print("E7 learned priors", flush=True)
corpus = build()
S, feats, cls = [], [], []
for nm, fam, G in corpus:
    n = G.number_of_nodes(); e = [tuple(map(int, x)) for x in G.edges()]
    u, w = statistic(edge_triples(n, e))
    S.append(surface((u, w), gq, bq))
    cls.append((tuple(map(tuple, u)), tuple(np.round(w, 12))))
    deg = np.array([d for _, d in G.degree()], float)
    A = nx.to_numpy_array(G)
    sv = np.sort(np.linalg.svd(A, compute_uv=False))[::-1]
    sv = np.pad(sv, (0, max(0, 8 - len(sv))))[:8] / max(1.0, n)
    feats.append(np.concatenate([[n, len(e), 2 * len(e) / (n * (n - 1)),
                                  deg.mean(), deg.std(), deg.max(), deg.min(),
                                  sum(nx.triangles(G).values()) / 3.0,
                                  nx.average_clustering(G)], sv]))
S = np.array(S); X = np.array(feats)
X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
uc = {c: i for i, c in enumerate(dict.fromkeys(cls))}
cid = np.array([uc[c] for c in cls])
print("  %d graphs, %d distinct depth-one landscapes" % (len(S), len(uc)), flush=True)
R["corpus_landscapes"] = {"graphs": int(len(S)), "distinct_landscapes": int(len(uc))}

BUDGETS = [1, 2, 4, 8, 16]
def regret(pred, true, b):
    order = np.argsort(-pred)[:b]
    return (true.max() - true[order].max()) / true.max()

def evaluate(folds):
    acc = {k: {b: [] for b in BUDGETS} for k in ("mean", "ridge", "spectral", "exact")}
    for te in folds:
        tr = np.setdiff1d(np.arange(len(S)), te)
        mu = S[tr].mean(0)
        rg = Ridge(alpha=1.0).fit(X[tr], S[tr])
        sp = Ridge(alpha=1.0).fit(X[tr][:, 9:], S[tr])
        for i in te:
            for b in BUDGETS:
                acc["mean"][b].append(regret(mu, S[i], b))
                acc["ridge"][b].append(regret(rg.predict(X[i:i+1])[0], S[i], b))
                acc["spectral"][b].append(regret(sp.predict(X[i:i+1, 9:])[0], S[i], b))
                acc["exact"][b].append(0.0)
    return {k: {b: float(np.mean(v[b])) for b in BUDGETS} for k, v in acc.items()}

rng = np.random.default_rng(7)
order = rng.permutation(len(S))
naive = [order[i::4] for i in range(4)]
classes = rng.permutation(len(uc))
grouped = [np.where(np.isin(cid, classes[i::4]))[0] for i in range(4)]
R["regret_naive"] = evaluate(naive)
R["regret_grouped"] = evaluate(grouped)
leak = sum(int(np.isin(cid[te], cid[np.setdiff1d(np.arange(len(S)), te)]).sum()) for te in naive)
R["naive_split_leakage"] = {"test_graphs_with_identical_landscape_in_training": leak,
                            "total_test_graphs": int(len(S))}
print("  naive  ", {k: round(v[1], 5) for k, v in R["regret_naive"].items()}, flush=True)
print("  grouped", {k: round(v[1], 5) for k, v in R["regret_grouped"].items()}, flush=True)
print("  leakage", R["naive_split_leakage"], flush=True)

(OUT / "results.json").write_text(json.dumps(R, indent=2))
np.savez_compressed(OUT / "surfaces81.npz", S=S, X=X, cid=cid,
                    names=np.array([c[0] for c in corpus]),
                    fams=np.array([c[1] for c in corpus]))
print("done", flush=True)
