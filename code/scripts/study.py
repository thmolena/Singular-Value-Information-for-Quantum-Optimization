"""Full study: sufficient statistic, certified transfer, collision audit."""
from __future__ import annotations
import json, time, pathlib, itertools
import numpy as np, networkx as nx
from svi.core import (load_gset, edge_triples, statistic, surface, grid,
                      statevector_surface, tv_distance, adjacency)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results_full"; OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
R = {}

# fine angle grid; gamma has period 2*pi, beta has period pi/2
GF, BF = np.meshgrid(np.linspace(0, 2*np.pi, 96, endpoint=False),
                     np.linspace(0, np.pi/2, 48, endpoint=False), indexing="ij")
GF, BF = GF.ravel(), BF.ravel()

def stat_of(G):
    n = G.number_of_nodes(); e = [tuple(map(int, x)) for x in G.edges()]
    return n, e, statistic(edge_triples(n, e))

# ================================================== E1  verification
print("E1 verification", flush=True)
rng = np.random.default_rng(20260907); errs = []
for _ in range(60):
    n = int(rng.integers(4, 15))
    G = nx.gnp_random_graph(n, float(rng.uniform(.2, .85)), seed=int(rng.integers(1 << 30)))
    if G.number_of_edges() == 0: continue
    n, e, st = stat_of(G)
    errs.append(float(np.abs(surface(st, GF, BF) - statevector_surface(n, e, GF, BF)).max()))
R["verification"] = {"n_graphs": len(errs), "n_values": len(errs) * len(GF),
                     "max_abs_error": max(errs), "median_abs_error": float(np.median(errs))}
print("  max err", max(errs), flush=True)

# =========================================== E2  statistic collisions
print("E2 collisions", flush=True)
gq, bq = grid(9)
pairs = [("Petersen graph", nx.petersen_graph(), "pentagonal prism", nx.circular_ladder_graph(5)),
         ("6-cycle", nx.cycle_graph(6), "8-cycle", nx.cycle_graph(8)),
         ("Heawood graph", nx.heawood_graph(), "7-prism", nx.circular_ladder_graph(7))]
R["collision_pairs"] = []
for na, A, nb, B in pairs:
    sa, sb = stat_of(A), stat_of(B)
    d = float(np.abs(statevector_surface(*sa[:2], GF, BF)
                     - statevector_surface(*sb[:2], GF, BF)).max()) if sa[0] <= 16 else None
    R["collision_pairs"].append({
        "a": na, "b": nb, "n_a": sa[0], "n_b": sb[0],
        "m_a": len(sa[1]), "m_b": len(sb[1]),
        "isomorphic": bool(nx.is_isomorphic(A, B)),
        "girth_a": int(nx.girth(A)), "girth_b": int(nx.girth(B)),
        "tv": float(tv_distance(sa[2], sb[2])),
        "max_surface_gap_statevector": d})
    print("  ", na, "vs", nb, "TV", R["collision_pairs"][-1]["tv"], "gap", d, flush=True)

# how often do non-isomorphic graphs collide?  exhaustive over small graphs
print("E2b exhaustive small-graph census", flush=True)
census = {}
for n in (5, 6, 7):
    seen = {}
    for G in nx.graph_atlas_g():
        if G.number_of_nodes() != n or G.number_of_edges() == 0: continue
        if not nx.is_connected(G): continue
        u, w = statistic(edge_triples(n, [tuple(map(int, x)) for x in G.edges()]))
        key = (tuple(map(tuple, u)), tuple(np.round(w, 12)))
        seen.setdefault(key, []).append(G)
    tot = sum(len(v) for v in seen.values())
    census[n] = {"connected_graphs": tot, "statistic_classes": len(seen),
                 "graphs_in_colliding_classes": sum(len(v) for v in seen.values() if len(v) > 1),
                 "largest_class": max(len(v) for v in seen.values())}
    print("  n=%d" % n, census[n], flush=True)
R["small_graph_census"] = census

# ==================================== E3  corpus, transfer, leakage
print("E3 corpus", flush=True)
gset_dir = ROOT / "data" / "gset"
gset = {}
for p in sorted(gset_dir.iterdir()):
    if not p.name.startswith("G"): continue
    n, e, wt = load_gset(p)
    if not wt: gset[p.name] = (n, e)
R["gset_unweighted"] = sorted(gset, key=lambda s: int(s[1:]))
print("  unweighted Gset graphs:", len(gset), flush=True)

def bfs_neighbourhood(n, adj, start, size):
    order, seen = [start], {start}
    qi = 0
    while qi < len(order) and len(order) < size:
        for w in sorted(adj[order[qi]]):
            if w not in seen and len(order) < size:
                seen.add(w); order.append(w)
        qi += 1
    return order

corpus = []   # (name, family, nx graph)
for name in R["gset_unweighted"][:12]:
    n, e = gset[name]; adj = adjacency(n, e)
    for i in range(24):
        nodes = bfs_neighbourhood(n, adj, (37 * i) % n, 10)
        H = nx.Graph(); H.add_nodes_from(range(len(nodes)))
        idx = {v: k for k, v in enumerate(nodes)}
        for u, v in e:
            if u in idx and v in idx: H.add_edge(idx[u], idx[v])
        if H.number_of_edges(): corpus.append((f"{name}:bfs{i}", f"gset:{name}", H))
for k, (nn, p) in enumerate(itertools.product((10, 12, 14), (.25, .4, .6))):
    for s in range(8):
        corpus.append((f"er{nn}_{p}_{s}", f"ER(p={p})", nx.gnp_random_graph(nn, p, seed=1000 + 97 * k + s)))
for k, (nn, d) in enumerate(itertools.product((10, 12, 14), (3, 4))):
    for s in range(8):
        corpus.append((f"rr{nn}_{d}_{s}", f"random {d}-regular", nx.random_regular_graph(d, nn, seed=2000 + 97 * k + s)))
for k, nn in enumerate((10, 12, 14)):
    for s in range(8):
        corpus.append((f"ba{nn}_{s}", "Barabasi-Albert", nx.barabasi_albert_graph(nn, 2, seed=3000 + 97 * k + s)))
corpus = [(nm, fam, G) for nm, fam, G in corpus if G.number_of_edges() > 0]
print("  corpus size", len(corpus), flush=True)

names = [c[0] for c in corpus]; fams = [c[1] for c in corpus]
stats = [statistic(edge_triples(G.number_of_nodes(), [tuple(map(int, x)) for x in G.edges()]))
         for _, _, G in corpus]
S = np.array([surface(st, GF, BF) for st in stats])          # (K, Q)

vocab = sorted({tuple(t) for st in stats for t in st[0]})
vi = {t: i for i, t in enumerate(vocab)}
M = np.zeros((len(stats), len(vocab)))
for r, (u, w) in enumerate(stats):
    for t, wt in zip(map(tuple, u), w): M[r, vi[t]] = wt
TV = 0.5 * np.abs(M[:, None, :] - M[None, :, :]).sum(-1)
GAP = np.zeros_like(TV)
for i in range(len(S)):
    GAP[i] = np.abs(S[i][None, :] - S).max(1)
best = S.max(1); arg = S.argmax(1)
# loss from optimising angles on i and applying them to j
XFER = np.array([[best[j] - S[j, arg[i]] for j in range(len(S))] for i in range(len(S))])

R["transfer"] = {
    "corpus_size": len(corpus), "angle_points": int(len(GF)),
    "bound_violations_sup": int((GAP > TV + 1e-12).sum()),
    "bound_violations_transfer": int((XFER > 2 * TV + 1e-12).sum()),
    "n_pairs": int(TV.size),
    "median_ratio_gap_over_tv": float(np.median((GAP / np.maximum(TV, 1e-15))[TV > 1e-9])),
    "spearman_gap_tv": float(__import__("scipy.stats", fromlist=["x"]).spearmanr(
        TV[np.triu_indices(len(TV), 1)], GAP[np.triu_indices(len(TV), 1)]).statistic)}
print("  ", R["transfer"], flush=True)

# exact-duplicate leakage: statistic collisions vs isomorphism collisions
print("E4 leakage audit", flush=True)
iso_pairs = stat_pairs = 0
K = len(corpus)
for i in range(K):
    for j in range(i + 1, K):
        if TV[i, j] < 1e-12:
            stat_pairs += 1
            if nx.is_isomorphic(corpus[i][2], corpus[j][2]): iso_pairs += 1
R["leakage"] = {"corpus_size": K, "pairs": K * (K - 1) // 2,
                "statistic_identical_pairs": stat_pairs,
                "isomorphic_pairs": iso_pairs,
                "collisions_missed_by_isomorphism": stat_pairs - iso_pairs}
print("  ", R["leakage"], flush=True)

# ============ E5  the standard QAOA benchmark family: random d-regular
print("E5 regular-graph benchmark census", flush=True)
reg = {}
for d in (3, 4, 5):
    keys, tri_free = {}, 0
    for nn in range(d + 1 if (d + 1) % 2 == 0 else d + 2, 31, 2):
        for s_ in range(40):
            try: G = nx.random_regular_graph(d, nn, seed=7000 + 131 * nn + s_)
            except Exception: continue
            u, w = statistic(edge_triples(nn, [tuple(map(int, x)) for x in G.edges()]))
            k = (tuple(map(tuple, u)), tuple(np.round(w, 12)))
            keys.setdefault(k, 0); keys[k] += 1
            if sum(t[2] for t in map(tuple, u)) == 0: tri_free += 1
    tot = sum(keys.values())
    reg[d] = {"graphs": tot, "sizes": "n=%d..30 even" % (d + 1),
              "statistic_classes": len(keys),
              "largest_class": max(keys.values()),
              "largest_class_fraction": max(keys.values()) / tot,
              "triangle_free_graphs": tri_free,
              "triangle_free_fraction": tri_free / tot}
    print("  d=%d" % d, reg[d], flush=True)
R["regular_benchmark"] = reg

np.savez_compressed(OUT / "matrices.npz", TV=TV, GAP=GAP, XFER=XFER, S=S,
                    names=np.array(names), fams=np.array(fams))
(OUT / "results.json").write_text(json.dumps(R, indent=2))
print("done", flush=True)
