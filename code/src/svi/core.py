"""Exact depth-one MaxCut QAOA surfaces from edge statistics.

The depth-one edge value is the Wang-Hadfield-Jiang-Rieffel identity
[Phys. Rev. A 97, 022304 (2018)].  Everything here is built on the
observation that the identity depends on the graph only through the
empirical distribution of the edge triple (deg u - 1, deg v - 1,
|N(u) cap N(v)|).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- graphs

def load_gset(path):
    """Return (n, edges, weighted) for a Gset file."""
    with open(path) as fh:
        head = fh.readline().split()
        n, m = int(head[0]), int(head[1])
        edges, weighted = [], False
        for line in fh:
            p = line.split()
            if len(p) < 3:
                continue
            u, v, w = int(p[0]) - 1, int(p[1]) - 1, int(p[2])
            if w != 1:
                weighted = True
            edges.append((u, v))
    assert len(edges) == m, (path, len(edges), m)
    return n, edges, weighted


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def edge_triples(n, edges):
    """(deg u - 1, deg v - 1, |N(u) cap N(v)|) for every edge, sorted per edge."""
    adj = adjacency(n, edges)
    out = np.empty((len(edges), 3), dtype=np.int64)
    for i, (u, v) in enumerate(edges):
        a, b = adj[u], adj[v]
        if len(a) > len(b):
            a, b = b, a
        lam = sum(1 for w in a if w in b)
        du, dv = len(adj[u]) - 1, len(adj[v]) - 1
        out[i] = (min(du, dv), max(du, dv), lam)
    return out


def statistic(triples):
    """Empirical measure mu_G over edge triples: (unique triples, weights)."""
    uniq, counts = np.unique(triples, axis=0, return_counts=True)
    return uniq, counts / counts.sum()


# ------------------------------------------------------- exact surface

def edge_value(triples, gamma, beta):
    """WHJR depth-one edge value, vectorised over triples and angles.

    triples : (T, 3) int array of (d_u, d_v, lambda)
    gamma, beta : (Q,) arrays.  Returns (T, Q).
    """
    du = triples[:, 0][:, None].astype(float)
    dv = triples[:, 1][:, None].astype(float)
    lam = triples[:, 2][:, None].astype(float)
    g = np.asarray(gamma, float)[None, :]
    b = np.asarray(beta, float)[None, :]

    cg = np.cos(g)
    term1 = 0.25 * np.sin(4 * b) * np.sin(g) * (cg ** du + cg ** dv)
    term2 = (0.25 * np.sin(2 * b) ** 2 * cg ** (du + dv - 2 * lam)
             * (1.0 - np.cos(2 * g) ** lam))
    return 0.5 + term1 - term2


def surface(stat, gamma, beta):
    """Exact F_G on the given angle points, from the statistic (uniq, w)."""
    uniq, w = stat
    return w @ edge_value(uniq, gamma, beta)


def grid(q):
    """The registered q x q angle grid, returned flattened."""
    gs = np.linspace(0.0, np.pi, q, endpoint=False)
    bs = np.linspace(0.0, np.pi / 2, q, endpoint=False)
    G, B = np.meshgrid(gs, bs, indexing="ij")
    return G.ravel(), B.ravel()


# ------------------------------------------------- reference simulator

def statevector_surface(n, edges, gamma, beta):
    """Independent statevector evaluation of F_G; O(2^n) memory."""
    N = 1 << n
    idx = np.arange(N)
    cost = np.zeros(N)
    for u, v in edges:
        cost += 0.5 * (1.0 - (1 - 2 * ((idx >> u) & 1)) * (1 - 2 * ((idx >> v) & 1)))
    out = np.empty(len(gamma))
    for k, (g, b) in enumerate(zip(gamma, beta)):
        psi = np.exp(-1j * g * cost) / np.sqrt(N)
        psi = psi.reshape([2] * n)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(n):
            psi = np.moveaxis(psi, n - 1 - q, 0)
            a0, a1 = psi[0].copy(), psi[1].copy()
            psi[0] = c * a0 + s * a1
            psi[1] = s * a0 + c * a1
            psi = np.moveaxis(psi, 0, n - 1 - q)
        psi = psi.reshape(N)
        out[k] = np.real(np.vdot(psi, cost * psi))
    return out / len(edges)


# ------------------------------------------------------------- metrics

def tv_distance(stat_a, stat_b):
    """Total variation distance between two edge-triple measures."""
    ua, wa = stat_a
    ub, wb = stat_b
    keys = {}
    for u, w in zip(map(tuple, ua), wa):
        keys[u] = keys.get(u, 0.0) + w
    for u, w in zip(map(tuple, ub), wb):
        keys[u] = keys.get(u, 0.0) - w
    return 0.5 * sum(abs(v) for v in keys.values())
