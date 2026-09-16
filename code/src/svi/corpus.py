"""Deterministic graph corpus shared by every experiment."""
from __future__ import annotations
import itertools, pathlib
import networkx as nx
from svi.core import load_gset, adjacency

ROOT = pathlib.Path(__file__).resolve().parents[2]


def unweighted_gset():
    out = {}
    for p in sorted((ROOT / "data" / "gset").iterdir(), key=lambda q: int(q.name[1:])):
        n, e, wt = load_gset(p)
        if not wt:
            out[p.name] = (n, e)
    return out


def bfs_neighbourhood(adj, start, size):
    order, seen, qi = [start], {start}, 0
    while qi < len(order) and len(order) < size:
        for w in sorted(adj[order[qi]]):
            if w not in seen and len(order) < size:
                seen.add(w)
                order.append(w)
        qi += 1
    return order


def build():
    """Return [(name, family, networkx graph)] -- fully deterministic."""
    corpus, gset = [], unweighted_gset()
    for name in list(gset)[:12]:
        n, e = gset[name]
        adj = adjacency(n, e)
        for i in range(24):
            nodes = bfs_neighbourhood(adj, (37 * i) % n, 10)
            idx = {v: k for k, v in enumerate(nodes)}
            H = nx.Graph()
            H.add_nodes_from(range(len(nodes)))
            for u, v in e:
                if u in idx and v in idx:
                    H.add_edge(idx[u], idx[v])
            corpus.append((f"{name}:bfs{i}", f"gset:{name}", H))
    for k, (nn, p) in enumerate(itertools.product((10, 12, 14), (.25, .4, .6))):
        for s in range(8):
            corpus.append((f"er{nn}_{p}_{s}", f"ER(p={p})",
                           nx.gnp_random_graph(nn, p, seed=1000 + 97 * k + s)))
    for k, (nn, d) in enumerate(itertools.product((10, 12, 14), (3, 4))):
        for s in range(8):
            corpus.append((f"rr{nn}_{d}_{s}", f"random {d}-regular",
                           nx.random_regular_graph(d, nn, seed=2000 + 97 * k + s)))
    for k, nn in enumerate((10, 12, 14)):
        for s in range(8):
            corpus.append((f"ba{nn}_{s}", "Barabasi-Albert",
                           nx.barabasi_albert_graph(nn, 2, seed=3000 + 97 * k + s)))
    return [(nm, fam, G) for nm, fam, G in corpus if G.number_of_edges() > 0]
