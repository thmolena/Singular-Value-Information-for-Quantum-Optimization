"""Publication figures for the depth-one sufficient-statistic study."""
from __future__ import annotations
import json, pathlib
import numpy as np, networkx as nx
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from svi.core import edge_triples, statistic, surface, statevector_surface

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results_full" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "results_full" / "results.json").read_text())
Z = np.load(ROOT / "results_full" / "matrices.npz", allow_pickle=True)

mpl.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.1, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6,
})
C = {"a": "#2166ac", "b": "#b2182b", "c": "#1b7837", "d": "#e08214", "g": "#666666"}

def grid2d(nq=121, nb=61):
    g = np.linspace(0, 2 * np.pi, nq)
    b = np.linspace(0, np.pi / 2, nb)
    G, B = np.meshgrid(g, b, indexing="ij")
    return g, b, G.ravel(), B.ravel()

def stat_of(G):
    n = G.number_of_nodes(); e = [tuple(map(int, x)) for x in G.edges()]
    return n, e, statistic(edge_triples(n, e))

# ---------------------------------------------------- Fig 1  statistic
def fig1():
    g, b, GF, BF = grid2d()
    P, Q = nx.petersen_graph(), nx.circular_ladder_graph(5)
    (nP, eP, sP), (nQ, eQ, sQ) = stat_of(P), stat_of(Q)
    FP = statevector_surface(nP, eP, GF, BF).reshape(len(g), len(b))
    FQ = statevector_surface(nQ, eQ, GF, BF).reshape(len(g), len(b))

    fig = plt.figure(figsize=(7.1, 2.5))
    gs = GridSpec(1, 5, width_ratios=[1.15, 1.15, 1.3, 1.3, 1.25], wspace=.5)
    for k, (G_, ttl, sub) in enumerate([(P, "Petersen graph", "girth 5"),
                                        (Q, "pentagonal prism", "girth 4")]):
        ax = fig.add_subplot(gs[0, k])
        pos = nx.shell_layout(G_, [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]])
        nx.draw_networkx_edges(G_, pos, ax=ax, width=.8, edge_color=C["g"])
        nx.draw_networkx_nodes(G_, pos, ax=ax, node_size=26,
                               node_color=C["a"] if k == 0 else C["b"], linewidths=0)
        ax.set_title(f"{ttl}\n{sub}", pad=3); ax.axis("off")
        ax.set_aspect("equal"); ax.margins(.12)
    ext = [b[0], b[-1], g[0], g[-1]]
    vmin, vmax = min(FP.min(), FQ.min()), max(FP.max(), FQ.max())
    for k, (F, ttl) in enumerate([(FP, r"$F_{\rm Petersen}$"), (FQ, r"$F_{\rm prism}$")]):
        ax = fig.add_subplot(gs[0, 2 + k])
        im = ax.imshow(F, origin="lower", aspect="auto", extent=ext,
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$\beta$"); ax.set_title(ttl, pad=3)
        if k == 0: ax.set_ylabel(r"$\gamma$")
        else: ax.set_yticklabels([])
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)
    ax = fig.add_subplot(gs[0, 4])
    D = np.abs(FP - FQ)
    im = ax.imshow(D, origin="lower", aspect="auto", extent=ext, cmap="magma")
    ax.set_xlabel(r"$\beta$"); ax.set_yticklabels([])
    ax.set_title("$|F_{\\rm Petersen}-F_{\\rm prism}|$\nmax $%.1e$" % D.max(), pad=3)
    cb = plt.colorbar(im, ax=ax, fraction=.046, pad=.03,
                      format=mpl.ticker.FuncFormatter(lambda z, _: "%.0f" % (z * 1e15)))
    cb.set_label(r"$\times10^{-15}$", fontsize=6, labelpad=1)
    fig.savefig(OUT / "fig1_sufficient_statistic.pdf")
    plt.close(fig); print("fig1", D.max())

# ---------------------------------------------------- Fig 2  transfer
def fig2():
    TV, GAP, XFER = Z["TV"], Z["GAP"], Z["XFER"]
    iu = np.triu_indices(len(TV), 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    panels = [(axes[0], TV[iu], GAP[iu], r"$D_{\rm TV}(\mu_G,\mu_H)$",
               r"$\sup_\theta\,|F_G(\theta)-F_H(\theta)|$",
               "(a)  landscape distance", 1.0, r"bound: $D_{\rm TV}$"),
              (axes[1], 2 * TV.ravel(), XFER.ravel(), r"$2\,D_{\rm TV}(\mu_G,\mu_H)$",
               r"transfer loss $F_H(\theta^*_H)-F_H(\theta^*_G)$",
               "(b)  transfer loss", 2.0, r"bound: $2D_{\rm TV}$")]
    for ax, x, y, xl, yl, ttl, xmax, blab in panels:
        m = (x > 1e-3) & (y > 1e-7)
        hb = ax.hexbin(x[m], y[m], xscale="log", yscale="log", gridsize=44,
                       bins="log", cmap="Blues", mincnt=1, linewidths=0,
                       extent=(np.log10(1e-2), np.log10(xmax * 1.15), -6, 0.1))
        ax.plot([1e-2, xmax], [1e-2, xmax], color=C["b"], lw=1.2, zorder=5, label=blab)
        ax.fill_between([1e-2, xmax], [1e-2, xmax], 3, color=C["b"], alpha=.06, zorder=0)
        ax.text(.97, .93, "forbidden by Thm. 2", transform=ax.transAxes, ha="right",
                fontsize=6, color=C["b"], style="italic")
        ax.set_xlim(1e-2, xmax * 1.15); ax.set_ylim(1e-6, 1.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", pad=3)
        ax.legend(loc="lower right", frameon=False)
        cb = plt.colorbar(hb, ax=ax, fraction=.045, pad=.02)
        cb.set_label("graph pairs", fontsize=6)
    v = R["transfer"]
    fig.text(.5, -.05, "%s graph pairs, %s bound violations"
             % (format(v["n_pairs"], ","), v["bound_violations_sup"] + v["bound_violations_transfer"]),
             ha="center", fontsize=6.5, color=C["g"])
    fig.tight_layout(); fig.savefig(OUT / "fig2_transfer_bound.pdf")
    plt.close(fig); print("fig2")

# -------------------------------------------------- Fig 3  degeneracy
def fig3():
    reg = R["regular_benchmark"]
    cen = R["small_graph_census"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    ax = axes[0]
    ds = sorted(reg, key=int)
    xx = np.arange(len(ds))
    ax.bar(xx - .2, [reg[d]["graphs"] for d in ds], .38, color=C["g"], label="graphs sampled")
    ax.bar(xx + .2, [reg[d]["statistic_classes"] for d in ds], .38, color=C["a"],
           label="distinct depth-one landscapes")
    for i, d in enumerate(ds):
        ax.text(i + .2, reg[d]["statistic_classes"] + 12, str(reg[d]["statistic_classes"]),
                ha="center", fontsize=6)
        ax.text(i - .2, reg[d]["graphs"] + 12, str(reg[d]["graphs"]), ha="center", fontsize=6)
    ax.set_xticks(xx); ax.set_xticklabels([f"random\n{d}-regular" for d in ds])
    ax.set_ylabel("count"); ax.set_ylim(0, 660)
    ax.set_title("(a)  benchmark landscape degeneracy", loc="left", pad=3)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax = axes[1]
    ns = sorted(cen, key=int)
    frac = [100 * cen[n]["shared_fraction"] for n in ns]
    fr2 = [100 * reg[d]["shared_fraction"] for d in ds]
    ax.bar(np.arange(len(ns)), frac, .55, color=C["d"])
    ax.bar(np.arange(len(ds)) + len(ns) + .6, fr2, .55, color=C["b"])
    ax.set_xticks(list(range(len(ns))) + [len(ns) + .6 + i for i in range(len(ds))])
    ax.set_xticklabels([f"all\n$n={n}$" for n in ns] + [f"{d}-reg.\n$n\\leq30$" for d in ds])
    ax.set_ylabel("% sharing a landscape with another sample")
    ax.set_ylim(0, 108)
    for i, v_ in enumerate(frac + fr2):
        xx_ = i if i < len(frac) else len(frac) + .6 + (i - len(frac))
        ax.text(xx_, v_ + 2.5, "%.1f" % v_, ha="center", fontsize=6)
    ax.set_title("(b)  degeneracy is a property of the family", loc="left", pad=3)
    fig.tight_layout(); fig.savefig(OUT / "fig3_degeneracy.pdf")
    plt.close(fig); print("fig3")

# ------------------------------------------------------ Fig 4 regret
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    style = {"mean": ("training mean", C["g"], "o"), "ridge": ("ridge on graph features", C["a"], "s"),
             "spectral": ("singular-value prior", C["d"], "^"), "exact": ("exact formula (0 queries)", C["b"], "D")}
    for ax, key, ttl in [(axes[0], "regret_naive", "(a)  random split (leaky)"),
                         (axes[1], "regret_grouped", "(b)  landscape-grouped split")]:
        d = R[key]
        bs = sorted(int(b) for b in d["mean"])
        for k, (lab, col, mk) in style.items():
            y = [d[k][str(b)] if str(b) in d[k] else d[k][b] for b in bs]
            ax.plot(bs, np.maximum(y, 1e-6), marker=mk, ms=3.2, color=col, label=lab)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("target objective queries $b$")
        ax.set_title(ttl, loc="left", pad=3)
        ax.set_xticks(bs); ax.set_xticklabels([str(b) for b in bs])
    for ax in axes:
        ax.set_ylim(5e-7, 6e-2)
        ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        ax.set_yticklabels(["$0$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
        ax.axhline(1e-6, color=C["g"], lw=.5, ls=":", zorder=0)
    axes[0].set_ylabel("mean normalised regret")
    axes[0].legend(frameon=False, loc="lower left")
    fig.text(.5, -.06, "the bottom gridline is exact zero regret",
             ha="center", fontsize=6.5, color=C["g"])
    fig.tight_layout(); fig.savefig(OUT / "fig4_regret.pdf")
    plt.close(fig); print("fig4")

# ----------------------------------------------------- Fig 5 runtime
def fig5():
    rs, bg = R["runtime_small"], R["runtime_gset"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    ax = axes[0]
    n = [r["n"] for r in rs]
    ax.plot(n, [r["statevector_s"] for r in rs], marker="o", ms=3.2, color=C["b"],
            label="statevector, $O(2^n)$")
    ax.plot(n, [r["formula_s"] for r in rs], marker="s", ms=3.2, color=C["a"],
            label="edge statistics, $O(m)$")
    ax.set_yscale("log"); ax.set_xlabel("vertices $n$")
    ax.set_ylabel("time for an $81$-point surface (s)")
    ax.set_title("(a)  matched full-surface cost", loc="left", pad=3)
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    sc = R["scaling"]
    for src, col, mk, lab in [(sc, C["c"], "o", "Barabasi-Albert"), (bg, C["a"], "s", "Gset")]:
        m = np.array([r["m"] for r in src], float)
        t = np.array([r["preprocess_s"] + r["surface_s"] for r in src])
        o = np.argsort(m)
        ax.plot(m[o], t[o], marker=mk, ms=3.2, lw=.9, color=col, label=lab)
    mm = np.array([1e3, 1.3e6])
    ref = mm * (max(r["preprocess_s"] + r["surface_s"] for r in sc)
                / max(r["m"] for r in sc))
    ax.plot(mm, ref, color=C["g"], lw=.9, ls="--", label=r"$O(m)$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("edges $m$"); ax.set_ylabel("time for an $81$-point surface (s)")
    ax.set_title("(b)  exact surfaces at scale", loc="left", pad=3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig5_runtime.pdf")
    plt.close(fig); print("fig5")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
