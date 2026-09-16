"""Supplementary measurements: consistent degeneracy metric and O(m) scaling."""
from __future__ import annotations
import json, time, pathlib
import numpy as np, networkx as nx
from svi.core import edge_triples, statistic, surface, grid
from svi.corpus import unweighted_gset

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results_full"
R = json.loads((OUT / "results.json").read_text())
gq, bq = grid(9)

def key_of(n, e):
    u, w = statistic(edge_triples(n, e))
    return (tuple(map(tuple, u)), tuple(np.round(w, 12)))

# consistent metric: fraction of graphs sharing a landscape with another sample
print("degeneracy metric", flush=True)
for d in (3, 4, 5):
    keys = {}
    for nn in range(d + 1 if (d + 1) % 2 == 0 else d + 2, 31, 2):
        for s in range(40):
            try: G = nx.random_regular_graph(d, nn, seed=7000 + 131 * nn + s)
            except Exception: continue
            k = key_of(nn, [tuple(map(int, x)) for x in G.edges()])
            keys[k] = keys.get(k, 0) + 1
    tot = sum(keys.values())
    shar = sum(v for v in keys.values() if v > 1)
    R["regular_benchmark"][str(d)]["graphs_sharing_a_landscape"] = shar
    R["regular_benchmark"][str(d)]["shared_fraction"] = shar / tot
    print("  d=%d %d/%d share (%.1f%%)" % (d, shar, tot, 100 * shar / tot), flush=True)
for n in R["small_graph_census"]:
    c = R["small_graph_census"][n]
    c["shared_fraction"] = c["graphs_in_colliding_classes"] / c["connected_graphs"]

# O(m) scaling across decades
print("scaling", flush=True)
scal = []
for nn, deg in [(200, 4), (600, 4), (2000, 4), (6000, 4), (20000, 4),
                (60000, 4), (2000, 20), (6000, 20), (20000, 20), (50000, 24)]:
    G = nx.barabasi_albert_graph(nn, deg, seed=17 * nn + deg)
    e = [tuple(map(int, x)) for x in G.edges()]
    t0 = time.perf_counter(); st = statistic(edge_triples(nn, e)); t_p = time.perf_counter() - t0
    t0 = time.perf_counter(); surface(st, gq, bq); t_s = time.perf_counter() - t0
    scal.append({"n": nn, "m": len(e), "unique_triples": int(len(st[0])),
                 "preprocess_s": t_p, "surface_s": t_s})
    print("  n=%6d m=%8d T=%6d  %.3fs + %.4fs" % (nn, len(e), len(st[0]), t_p, t_s), flush=True)
R["scaling"] = scal
comp = [(r["m"], r["unique_triples"]) for r in R["runtime_gset"] + scal]
R["compression"] = {"max_m": max(c[0] for c in comp),
                    "median_ratio_m_over_T": float(np.median([c[0] / c[1] for c in comp]))}
print("  compression", R["compression"], flush=True)
(OUT / "results.json").write_text(json.dumps(R, indent=2))
print("done", flush=True)
