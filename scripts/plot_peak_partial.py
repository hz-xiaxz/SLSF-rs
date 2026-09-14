# /// script
# dependencies = ["matplotlib>=3.9", "numpy>=2.0"]
# ///
"""Plot the near-peak layer-resolved scans (partial or final extracted JSON)."""
import json, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {16: "#2a78d6", 32: "#eb6834", 64: "#1baf7a"}   # fixed categorical order
J1, J2 = "#4a3aa7", "#eda100"

def load(path):
    rows = json.load(open(path))
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["L"], round(r["T"], 4))].append(r)
    return g

def stat(v, key, fn=lambda x: x):
    a = np.array([fn(x[key]) for x in v]); return a.mean(), a.std(ddof=1) / np.sqrt(len(a))

def series(g, L, key, fn=lambda x: x):
    Ts = sorted(t for (l, t) in g if l == L)
    m, e = zip(*[stat(g[(L, T)], key, fn) for T in Ts])
    return np.array(Ts), np.array(m), np.array(e)

def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def overview(wide_files, out, peaks):
    g = collections.defaultdict(list)
    for f in wide_files:
        for e in json.load(open(f))["tasks"]:
            t, o = e["task"], e.get("observables") or {}
            if "SpecificHeat" in o:
                g[(t["l"], round(t["temperature"], 4))].append(dict(C=o["SpecificHeat"]["mean"], RhoXY=o["RhoXY"]["mean"], RhoZ=o["RhoZ"]["mean"]))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for L, c in [(8, "#9ec5f4"), (16, COL[16]), (32, COL[32]), (64, COL[64])]:
        T, m, e = series(g, L, "C"); axs[0].errorbar(T, m, e, color=c, lw=1.6, ms=4, marker="o", label=f"L={L}")
        T, m, e = series(g, L, "RhoXY"); axs[1].plot(T, m, color=c, lw=1.6, marker="o", ms=4, label=f"ρxy L={L}")
    T, m, e = series(g, 64, "RhoZ"); axs[1].plot(T, m * 10, color="#52514e", lw=1.6, ls="--", marker="s", ms=4, label="10·ρz L=64")
    for ax in axs:
        for p in peaks: ax.axvline(p, color="#c3c2b7", lw=1, ls=":")
    style(axs[0], "T", "specific heat C", "Wide scan: two non-divergent peaks")
    style(axs[1], "T", "stiffness", "In-plane vs stacking stiffness")
    axs[0].legend(frameon=False, fontsize=8); axs[1].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=160)

def peak1(g, out):
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7), constrained_layout=True)
    a = axs.ravel()
    for L in (16, 32, 64):
        T, m, e = series(g, L, "C"); a[0].errorbar(T, m, e, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        T, m, e = series(g, L, "RhoZ"); a[1].errorbar(T, m * L, e * L, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        T, m, e = series(g, L, "RhoXY"); a[2].errorbar(T, m, e, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        h = L // 2
        T, m1, e1 = series(g, L, "CorrXY_J", lambda d: d["1"][h - 1]); T, m2, e2 = series(g, L, "CorrXY_J", lambda d: d["2"][h - 1])
        a[3].errorbar(T, m1, e1, color=COL[L], lw=1.6, marker="o", ms=4, label=f"J=1, L={L}")
        a[3].errorbar(T, m2, e2, color=COL[L], lw=1.6, marker="s", ms=4, ls="--", label=f"J=2, L={L}")
        T, m4, e4 = series(g, L, "CorrXY_J", lambda d: d["1"][3])
        a[4].errorbar(T, m4, e4, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        v = g[(L, 1.6)]; cz = np.array([x["CorrZ"] for x in v]); r = np.arange(1, cz.shape[1] + 1)
        a[5].errorbar(r, cz.mean(0), cz.std(0, ddof=1) / np.sqrt(len(v)), color=COL[L], lw=1.6, marker="o", ms=3, label=f"L={L}")
    a[1].axhline(0, color="#c3c2b7", lw=1)
    for ax in a[:5]: ax.axvline(1.325, color="#c3c2b7", lw=1, ls=":")
    style(a[0], "T", "C", "(a) Specific heat: peak at T≈1.30–1.35, no L dependence")
    style(a[1], "T", "ρz · L", "(b) ρz·L grows with L up to T≈1.55, flat by 1.6")
    style(a[2], "T", "ρxy", "(c) In-plane stiffness stays finite")
    style(a[3], "T", "Cxy(L/2)", "(d) In-plane plateau: J=1 layers (solid) vs J=2 (dashed)")
    style(a[4], "T", "Cxy(r=4), J=1 layers", "(e) J=1 layers lose short-range order across the peak")
    style(a[5], "r (layers)", "Cz(r)", "(f) Stacking correlation at T=1.6")
    a[5].set_yscale("log")
    for ax in a: ax.legend(frameon=False, fontsize=7.5, ncols=2 if ax is a[3] else 1)
    fig.savefig(out, dpi=160)

def peak2(g, out):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    for L in (16, 32, 64):
        T, m, e = series(g, L, "C"); axs[0].errorbar(T, m, e, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        T, m, e = series(g, L, "RhoXY"); axs[1].errorbar(T, m, e, color=COL[L], lw=1.6, marker="o", ms=4, label=f"L={L}")
        h = L // 2
        T, m2, e2 = series(g, L, "CorrXY_J", lambda d: d["2"][h - 1]); axs[2].errorbar(T, m2, e2, color=COL[L], lw=1.6, marker="s", ms=4, label=f"J=2, L={L}")
    axs[1].plot(T, T / np.pi, color="#52514e", lw=1, ls="--", label="T/π (KT jump if only J=2 layers stiff)")
    style(axs[0], "T", "C", "(a) Second peak at T≈2.3, height saturates")
    style(axs[1], "T", "ρxy", "(b) ρxy shrinks with L across the peak")
    style(axs[2], "T", "Cxy(L/2), J=2 layers", "(c) J=2 plateau collapses with L")
    axs[2].set_yscale("log")
    for ax in axs: ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(out, dpi=160)

if __name__ == "__main__":
    which, path, out = sys.argv[1:4]
    if which == "overview":
        overview(path.split(","), out, peaks=(1.325, 2.3))
    elif which == "peak1":
        peak1(load(path), out)
    else:
        peak2(load(path), out)
