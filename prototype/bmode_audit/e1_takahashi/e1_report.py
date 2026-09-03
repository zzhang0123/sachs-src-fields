"""Tables, figures and results.json for the E1 Takahashi B-mode audit.

Reads the caches written by e1_takahashi_bmode.py (outputs/spectra_*.npz,
outputs/floor_*.npz) and writes outputs/results.json, outputs/tables.md and
the figures. Every number in results.json carries its provenance string.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
BANDS = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1500), (1500, 2500), (2500, 4000)]
ELL_QUOTE = [60, 100, 300, 1000, 1500]
PLANES = {"zs10": 0.574, "zs16": 1.033, "zs38": 5.342}
# paper's FK prediction (results_a1_inventory.json; conclusion.tex:68-71; insights.tex:207-218)
FK_A = {"zs16": (0.009, 0.011), "zs38": (0.013, 0.017)}  # FK fraction of Order-0 in xi_+ (z_s = 1, 5)
FK_R = {60.0: 0.95, 1500.0: 0.42}  # FK's own C_BB/C_EE anchors
KH = {"z_le_1": 1.0 / 3000.0, "z_le_3": 2.0e-3}  # Krause & Hirata 2010 ceiling, as used by the Letter
GATE_BAND = (50, 1000)
INV = "/Users/zzhang/projects/SFT-WL-B/analysis/wpa/results_a1_inventory.json"
TH = None


def r_of_ell(ell):
    """FK's own C_BB/C_EE, linear in r against log ell between the anchors
    (identical to SFT-WL-B/analysis/wpa/make_a2_figure.py::r_of_ell)."""
    ell = np.asarray(ell, dtype=float)
    return FK_R[60.0] + (FK_R[1500.0] - FK_R[60.0]) * np.log(ell / 60.0) / np.log(1500.0 / 60.0)


def fk_band(plane, ell):
    a_lo, a_hi = FK_A[plane]
    r = r_of_ell(ell)
    w = r / (1.0 + r)
    return a_lo * w, a_hi * w


def sphere_factor(ell):
    """(l+2)(l-1)/(l(l+1)): C_EE = F C_kk and C_BB = F C_ww for shear derived
    from scalar / pseudo-scalar potentials on the sphere."""
    ell = np.asarray(ell, dtype=float)
    out = np.zeros_like(ell)
    ok = ell >= 2
    out[ok] = (ell[ok] + 2) * (ell[ok] - 1) / (ell[ok] * (ell[ok] + 1))
    return out


def load_spectra():
    files = sorted(glob.glob(os.path.join(OUT, "spectra_*.npz")))
    data = {}
    for fn in files:
        m = re.match(r"spectra_(zs\d+)_(fast|full)_nside(\d+)_lmax(\d+)\.npz", os.path.basename(fn))
        d = dict(np.load(fn, allow_pickle=False))
        d["file"] = fn
        data[(m.group(1), m.group(2))] = d
    return data


def band_mean(cl, lo, hi):
    return float(np.mean(cl[lo:hi]))


def band_cv_rel(cl, lo, hi):
    """Gaussian cosmic-variance relative sigma of the band mean (f_sky = 1)."""
    ell = np.arange(lo, hi)
    c = cl[lo:hi]
    var = np.sum(2.0 * c * c / (2.0 * ell + 1.0)) / len(ell) ** 2
    return float(np.sqrt(var) / abs(np.mean(c)))


def window_mean(cl, l0, frac=0.1):
    lo, hi = int(round(l0 * (1 - frac))), int(round(l0 * (1 + frac))) + 1
    return float(np.mean(cl[lo:hi])), lo, hi


def gate(d):
    """Pick the gamma2 sign convention: the one with C_EE ~ C_kk in GATE_BAND."""
    lo, hi = GATE_BAND
    out = {}
    for tag in ("plus", "minus"):
        if f"cl_{tag}_EE" not in d:
            continue
        ee, kk, bb = d[f"cl_{tag}_EE"], d[f"cl_{tag}_kk"], d[f"cl_{tag}_BB"]
        f = sphere_factor(d["ell"])
        out[tag] = dict(
            EE_over_kk=band_mean(ee[lo:hi] / (f[lo:hi] * kk[lo:hi]), 0, hi - lo),
            BB_over_EE=band_mean(bb, lo, hi) / band_mean(ee, lo, hi),
        )
    best = max(out, key=lambda t: out[t]["EE_over_kk"] - abs(out[t]["EE_over_kk"] - 1.0))
    best = min(out, key=lambda t: abs(out[t]["EE_over_kk"] - 1.0))
    return best, out


def spectra_of(d, tag):
    return {k[len(f"cl_{tag}_"):]: v for k, v in d.items() if k.startswith(f"cl_{tag}_")}


def band_table(cl, lmax):
    rows = []
    f = sphere_factor(np.arange(lmax + 1))
    for lo, hi in BANDS:
        if hi > lmax + 1:
            continue
        kk, ee, bb, ww = (band_mean(cl[k], lo, hi) for k in ("kk", "EE", "BB", "ww"))
        eb = band_mean(cl["EB"], lo, hi)
        ke = band_mean(cl["kE"], lo, hi)
        kw = band_mean(cl["kw"], lo, hi)
        fww = band_mean(f * cl["ww"], lo, hi)
        fkk = band_mean(f * cl["kk"], lo, hi)
        rows.append(dict(
            band=[lo, hi], ell_eff=float(np.mean(np.arange(lo, hi))),
            C_kk=kk, C_EE=ee, C_BB=bb, C_ww=ww, C_EB=eb, C_kE=ke, C_kw=kw,
            BB_over_EE=bb / ee, ww_over_kk=ww / kk, BB_over_ww=bb / ww,
            BB_over_Fww=bb / fww, EE_over_kk=ee / kk, EE_over_Fkk=ee / fkk,
            EB_over_sqrtEEBB=eb / np.sqrt(ee * bb), kE_over_sqrtkkEE=ke / np.sqrt(kk * ee),
            kw_over_sqrtkkww=kw / np.sqrt(kk * ww),
            cv_rel_EE=band_cv_rel(cl["EE"], lo, hi), cv_rel_BB=band_cv_rel(cl["BB"], lo, hi),
            cv_rel_ratio=float(np.hypot(band_cv_rel(cl["EE"], lo, hi), band_cv_rel(cl["BB"], lo, hi))),
        ))
    return rows


def quote_table(cl, lmax, plane):
    rows = []
    for l0 in ELL_QUOTE:
        if l0 * 1.1 + 1 > lmax:
            continue
        bb, lo, hi = window_mean(cl["BB"], l0)
        ee, _, _ = window_mean(cl["EE"], l0)
        ww, _, _ = window_mean(cl["ww"], l0)
        kk, _, _ = window_mean(cl["kk"], l0)
        meas = bb / ee
        cv = float(np.hypot(band_cv_rel(cl["EE"], lo, hi), band_cv_rel(cl["BB"], lo, hi)))
        row = dict(ell=l0, window=[lo, hi], BB_over_EE=meas, BB_over_EE_single_ell=float(cl["BB"][l0] / cl["EE"][l0]),
                   ww_over_kk=ww / kk, BB_over_ww=bb / ww, cv_rel_ratio=cv)
        if plane in FK_A:
            f_lo, f_hi = fk_band(plane, l0)
            row.update(paper_f_bb_lo=float(f_lo), paper_f_bb_hi=float(f_hi), paper_r_B_over_E=float(r_of_ell(l0)),
                       exclusion_lo=float(f_lo / meas), exclusion_hi=float(f_hi / meas))
        row["KH_ceiling_z_le_1"] = KH["z_le_1"]
        row["KH_ceiling_z_le_3"] = KH["z_le_3"]
        row["meas_over_KH_z_le_1"] = meas / KH["z_le_1"]
        row["meas_over_KH_z_le_3"] = meas / KH["z_le_3"]
        rows.append(row)
    return rows


def log_binned(ell, y, nbins=40, lmin=10, lmax=None):
    lmax = lmax or ell.max()
    edges = np.unique(np.geomspace(lmin, lmax, nbins + 1).astype(int))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        xs.append(np.exp(np.mean(np.log(np.arange(lo, hi)))))
        ys.append(np.mean(y[lo:hi]))
    return np.array(xs), np.array(ys)


def md_table(header, rows, fmt):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r)) + " |")
    return "\n".join(lines)


def fig_spectra(data, tags, floors):
    planes = sorted({p for p, _ in data})
    fig, axes = plt.subplots(1, len(planes), figsize=(6.2 * len(planes), 5.0), squeeze=False)
    for ax, plane in zip(axes[0], planes):
        for mode, ls in (("fast", "-"), ("full", "--")):
            if (plane, mode) not in data:
                continue
            d = data[(plane, mode)]
            cl = spectra_of(d, tags[(plane, mode)])
            ell = d["ell"]
            pref = ell * (ell + 1) / (2 * np.pi)
            for key, col, lab in (("kk", "k", r"$\kappa\kappa$"), ("EE", "C0", "EE"), ("BB", "C3", "BB"), ("ww", "C2", r"$\omega\omega$")):
                x, y = log_binned(ell, pref * cl[key], lmax=int(d["lmax"]))
                ax.plot(x, y, ls, color=col, label=f"{lab} ({mode}, Nside {int(d['nside'])})" if key != "kk" or mode == "fast" else None, lw=1.5 if mode == "fast" else 1.0)
            x, y = log_binned(ell, pref * np.abs(cl["EB"]), lmax=int(d["lmax"]))
            ax.plot(x, y, ls, color="C1", lw=0.8, label=f"|EB| ({mode})")
        if plane in floors:
            fd = floors[plane]
            for mode, ls in (("fast", ":"), ("full", "-.")):
                ell = fd[f"ell_{mode}"]
                pref = ell * (ell + 1) / (2 * np.pi)
                x, y = log_binned(ell, pref * fd[f"raw_{mode}_BB"], lmax=int(ell.max()))
                ax.plot(x, y, ls, color="C3", lw=1.2, label=f"BB pipeline floor, {mode} pass (pure-E synthesis)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
        ax.set_title(f"Takahashi 2017 {plane} ($z_s$ = {PLANES[plane]})")
        ax.set_xlim(10, 4096)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_spectra.png"), dpi=150)
    plt.close(fig)


def fig_overlay(data, tags, floors, tables):
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ell_fine = np.geomspace(50, 1500, 200)
    for plane, col, lab in (("zs16", "C0", r"paper FK prediction, $z_s=1$ (a = 0.9-1.1%)"), ("zs38", "C4", r"paper FK prediction, $z_s=5$ (a = 1.3-1.7%)")):
        lo, hi = fk_band(plane, ell_fine)
        ax.fill_between(ell_fine, lo, hi, color=col, alpha=0.25, label=lab)
    ax.axhline(KH["z_le_1"], color="0.3", ls="--", lw=1.0, label=r"Krause & Hirata 2010 ceiling, $z\leq1$ (3.3e-4)")
    ax.axhline(KH["z_le_3"], color="0.3", ls=":", lw=1.0, label=r"Krause & Hirata 2010 ceiling, $z\leq3$ (2.0e-3)")
    style = {"zs16": ("C0", "o"), "zs38": ("C4", "s"), "zs10": ("C2", "^")}
    for (plane, mode), d in sorted(data.items()):
        cl = spectra_of(d, tags[(plane, mode)])
        ell = d["ell"]
        x, num = log_binned(ell, cl["BB"], nbins=30, lmax=int(d["lmax"]))
        _, den = log_binned(ell, cl["EE"], nbins=30, lmax=int(d["lmax"]))
        col, mk = style[plane]
        ax.plot(x, num / den, "-" if mode == "fast" else "--", color=col, lw=1.6 if mode == "fast" else 1.0,
                label=f"measured $C^{{BB}}/C^{{EE}}$ {plane} ($z_s$={PLANES[plane]}), {mode} Nside {int(d['nside'])}")
        if mode == "fast":
            rows = tables[(plane, mode)]["bands"]
            xb = [r["ell_eff"] for r in rows]
            yb = [r["BB_over_EE"] for r in rows]
            eb = [r["BB_over_EE"] * r["cv_rel_ratio"] for r in rows]
            ax.errorbar(xb, yb, yerr=eb, fmt=mk, color=col, ms=5, capsize=2)
    for (plane, mode), d in sorted(data.items()):
        if mode != "full":
            continue
        cl = spectra_of(d, tags[(plane, mode)])
        ell = d["ell"]
        col, mk = style[plane]
        x, num = log_binned(ell, sphere_factor(ell) * cl["ww"], nbins=30, lmax=int(d["lmax"]))
        _, den = log_binned(ell, cl["EE"], nbins=30, lmax=int(d["lmax"]))
        ax.plot(x, num / den, "-", color=col, lw=2.6, alpha=0.45, label=f"physical (rotation-implied) B-mode $F_\\ell C^{{\\omega\\omega}}/C^{{EE}}$ {plane}")
        if TH is not None and plane in TH:
            et = np.array(TH[plane]["ell"])
            cwt = np.array(TH[plane]["C_ww_postborn"])
            eet = np.array([window_mean(cl["EE"], int(l))[0] for l in et])
            ax.plot(et, sphere_factor(et) * cwt / eet, "x--", color=col, lw=0.8, ms=5, label=f"post-Born lens-lens theory (CAMB halofit, this work) {plane}")
    for plane, fd in floors.items():
        col, _ = style[plane]
        for mode, ls in (("fast", ":"), ("full", "-.")):
            ell = fd[f"ell_{mode}"]
            x, num = log_binned(ell, fd[f"raw_{mode}_BB"], nbins=30, lmax=int(ell.max()))
            _, den = log_binned(ell, fd[f"raw_{mode}_EE"], nbins=30, lmax=int(ell.max()))
            ax.plot(x, num / den, ls, color="0.5", lw=1.2, label=f"pipeline B floor, {mode} pass (pure-E Gaussian synthesis, {plane} spectrum)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 4096)
    ax.set_ylim(1e-8, 3e-2)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^{BB}/C_\ell^{EE}$")
    ax.set_title("Shear B-mode fraction: Takahashi 2017 ray-traced maps vs the paper's FK prediction")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ratio_overlay.png"), dpi=150)
    plt.close(fig)


def fig_consistency(data, tags):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (plane, mode), d in sorted(data.items()):
        cl = spectra_of(d, tags[(plane, mode)])
        ell = d["ell"]
        f = sphere_factor(ell)
        ls = "-" if mode == "fast" else "--"
        x, a = log_binned(ell, cl["EE"], nbins=30, lmax=int(d["lmax"]))
        _, b = log_binned(ell, f * cl["kk"], nbins=30, lmax=int(d["lmax"]))
        axes[0].plot(x, a / b, ls, label=f"$C^{{EE}}/(F_\\ell C^{{\\kappa\\kappa}})$ {plane} {mode}")
        _, a = log_binned(ell, cl["BB"], nbins=30, lmax=int(d["lmax"]))
        _, b = log_binned(ell, f * cl["ww"], nbins=30, lmax=int(d["lmax"]))
        axes[0].plot(x, a / b, ls, lw=1.0, label=f"$C^{{BB}}/(F_\\ell C^{{\\omega\\omega}})$ {plane} {mode}")
        _, a = log_binned(ell, cl["EB"], nbins=30, lmax=int(d["lmax"]))
        _, b = log_binned(ell, np.sqrt(cl["EE"] * cl["BB"]), nbins=30, lmax=int(d["lmax"]))
        axes[1].plot(x, a / b, ls, label=f"$C^{{EB}}/\\sqrt{{C^{{EE}}C^{{BB}}}}$ {plane} {mode}")
        _, a = log_binned(ell, cl["kE"], nbins=30, lmax=int(d["lmax"]))
        _, b = log_binned(ell, np.sqrt(cl["kk"] * cl["EE"]), nbins=30, lmax=int(d["lmax"]))
        axes[1].plot(x, a / b, ls, lw=1.0, label=f"$C^{{\\kappa E}}/\\sqrt{{C^{{\\kappa\\kappa}}C^{{EE}}}}$ {plane} {mode}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(10, 4096)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        ax.set_xlabel(r"$\ell$")
    axes[0].set_ylim(0.0, 1.5)
    axes[0].set_title(r"E vs $\kappa$ and B vs $\omega$ ($F_\ell=(\ell+2)(\ell-1)/\ell(\ell+1)$)")
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_title("cross-correlation coefficients")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_consistency.png"), dpi=150)
    plt.close(fig)


def load_theory():
    fn = os.path.join(OUT, "post_born_theory.json")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)


def theory_rows(cl, plane, th):
    """Measured (window mean) vs post-Born theory at the theory multipoles."""
    rows = []
    if th is None or plane not in th:
        return rows
    ells = th[plane]["ell"]
    for l0, ckk_t, cww_t in zip(ells, th[plane]["C_kk_limber"], th[plane]["C_ww_postborn"]):
        l0 = int(l0)
        if l0 * 1.1 + 1 > len(cl["kk"]):
            continue
        kk, lo, hi = window_mean(cl["kk"], l0)
        ww, _, _ = window_mean(cl["ww"], l0)
        bb, _, _ = window_mean(cl["BB"], l0)
        ee, _, _ = window_mean(cl["EE"], l0)
        F = float(sphere_factor(np.array([l0]))[0])
        rows.append(dict(ell=l0, window=[lo, hi], C_kk_meas=kk, C_kk_theory=ckk_t, kk_ratio=kk / ckk_t,
                         C_ww_meas=ww, C_ww_theory=cww_t, ww_ratio=ww / cww_t,
                         physical_BB_over_EE_from_omega=F * ww / ee, theory_BB_over_EE=F * cww_t / ee,
                         total_BB_over_EE=bb / ee, cv_rel_ww=band_cv_rel(cl["ww"], lo, hi)))
    return rows


def coherence_rows(cl, lmax):
    """r_wB against -sqrt(F C_ww / C_BB): equality means B = -sqrt(F) omega + (uncorrelated) N."""
    rows = []
    f = sphere_factor(np.arange(lmax + 1))
    for lo, hi in BANDS:
        if hi > lmax + 1:
            continue
        ww, bb, wb = band_mean(cl["ww"], lo, hi), band_mean(cl["BB"], lo, hi), band_mean(cl["wB"], lo, hi)
        fww = band_mean(f * cl["ww"], lo, hi)
        rows.append(dict(band=[lo, hi], r_wB=wb / np.sqrt(ww * bb), sqrt_Fww_over_BB=np.sqrt(fww / bb),
                         excess_BB_minus_Fww=bb - fww, excess_over_EE=(bb - fww) / band_mean(cl["EE"], lo, hi)))
    return rows


def build_report():
    data = load_spectra()
    global TH
    TH = load_theory()
    floors = {}
    for fn in glob.glob(os.path.join(OUT, "floor_*.npz")):
        plane = os.path.basename(fn)[len("floor_"):-len(".npz")]
        floors[plane] = dict(np.load(fn))
    tags, tables, results = {}, {}, {"provenance": {}, "planes": {}}
    results["provenance"]["maps"] = "Takahashi et al. 2017 (arXiv:1706.01472) allskymap_nres12r000.zs{10,16,38}.mag.dat, Nside 4096, read with SFT-WL-B/analysis/wpc_routeb/cb2_takahashi_io.py"
    results["provenance"]["paper_fk"] = f"f_BB = a r/(1+r), a from {INV} fk.values.*.a_xi_plus_fraction, r from by_multipole.{{60,1500}}.r_B_over_E, r(ell) linear in log ell as SFT-WL-B/analysis/wpa/make_a2_figure.py::r_of_ell"
    results["provenance"]["KH"] = "Krause & Hirata 2010, A&A 523, A28: B/E <= 2.0e-3 (z<=3), 3.3e-4 (z<=1), from results_a1_inventory.json known_cosmological_ceiling"
    results["provenance"]["pixel_window"] = "healpy.pixwin(nside, pol=True); fast pass divides by pixwin(1024)*pixwin(4096) (T for kappa/omega, P for E/B); full pass by pixwin(4096). Ratios are window-independent. Takahashi's resolution damping (1+(ell/1.6 Nside)^2)^-1 is stored (damping_takahashi) but NOT applied."
    md = []
    for (plane, mode), d in sorted(data.items()):
        tag, gate_info = gate(d)
        tags[(plane, mode)] = tag
        cl = spectra_of(d, tag)
        lmax = int(d["lmax"])
        bands = band_table(cl, lmax)
        quotes = quote_table(cl, lmax, plane)
        theory = theory_rows(cl, plane, TH) if mode == "full" else []
        coh = coherence_rows(cl, lmax)
        tables[(plane, mode)] = dict(bands=bands, quotes=quotes, theory=theory, coherence=coh)
        stats = json.loads(str(d["stats_json"]))
        results["planes"].setdefault(plane, {})[mode] = dict(
            file=os.path.relpath(d["file"], HERE), z_s=PLANES[plane], nside=int(d["nside"]), lmax=lmax,
            n_iter=int(d["n_iter"]), wall_seconds=float(d["wall_seconds"]), field_stats=stats,
            gamma2_sign_gate=dict(chosen=tag, candidates=gate_info, band=list(GATE_BAND)),
            bands=bands, quotes=quotes, theory_comparison=theory, omega_B_coherence=coh,
        )
        md.append(f"\n### {plane} (z_s = {PLANES[plane]}), {mode} pass: Nside {int(d['nside'])}, lmax {lmax}, gamma2 sign convention = {tag}; gate {json.dumps(gate_info)}\n")
        md.append("Field statistics (native Nside 4096 map, float64 accumulation):\n")
        md.append(md_table(["field", "mean", "rms", "std", "min", "max", "non-finite"],
                           [dict(name=k, **v) for k, v in stats.items()],
                           lambda r: [r["name"], f"{r['mean']:.3e}", f"{r['rms']:.4e}", f"{r['std']:.4e}", f"{r['min']:.3e}", f"{r['max']:.3e}", str(r["n_nonfinite"])]))
        md.append("\nBand-averaged spectra (pixel-window corrected; C_ell mean over the band):\n")
        md.append(md_table(["band", "C_kk", "C_EE", "C_BB", "C_ww", "BB/EE", "ww/kk", "BB/(F ww)", "EE/(F kk)", "EB/sqrt(EE BB)", "kE/sqrt(kk EE)", "CV rel (ratio)"], bands,
                           lambda r: [f"{r['band'][0]}-{r['band'][1]}", f"{r['C_kk']:.3e}", f"{r['C_EE']:.3e}", f"{r['C_BB']:.3e}", f"{r['C_ww']:.3e}",
                                      f"{r['BB_over_EE']:.3e}", f"{r['ww_over_kk']:.3e}", f"{r['BB_over_Fww']:.3f}", f"{r['EE_over_Fkk']:.4f}",
                                      f"{r['EB_over_sqrtEEBB']:+.3e}", f"{r['kE_over_sqrtkkEE']:+.4f}", f"{r['cv_rel_ratio']:.3f}"]))
        md.append("\nQuoted multipoles (mean over ell in [0.9 l, 1.1 l]):\n")
        md.append(md_table(["ell", "measured BB/EE", "single-ell", "CV rel", "paper f_BB lo-hi", "paper r_B/E", "exclusion lo-hi", "meas/KH(z<=1)", "meas/KH(z<=3)", "BB/ww"], quotes,
                           lambda r: [str(r["ell"]), f"{r['BB_over_EE']:.3e}", f"{r['BB_over_EE_single_ell']:.3e}", f"{r['cv_rel_ratio']:.3f}",
                                      (f"{r['paper_f_bb_lo']:.2e}-{r['paper_f_bb_hi']:.2e}" if "paper_f_bb_lo" in r else "n/a"),
                                      (f"{r['paper_r_B_over_E']:.2f}" if "paper_r_B_over_E" in r else "n/a"),
                                      (f"{r['exclusion_lo']:.0f}-{r['exclusion_hi']:.0f}" if "exclusion_lo" in r else "n/a"),
                                      f"{r['meas_over_KH_z_le_1']:.3f}", f"{r['meas_over_KH_z_le_3']:.4f}", f"{r['BB_over_ww']:.3f}"]))
        md.append("\nomega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):\n")
        md.append(md_table(["band", "r_wB", "-sqrt(F ww/BB)", "excess C_BB - F C_ww", "excess / C_EE"], coh,
                           lambda r: [f"{r['band'][0]}-{r['band'][1]}", f"{r['r_wB']:+.4f}", f"{-r['sqrt_Fww_over_BB']:+.4f}", f"{r['excess_BB_minus_Fww']:.3e}", f"{r['excess_over_EE']:.3e}"]))
        if theory:
            md.append("\nMeasured vs post-Born theory (post_born_theory.py: CAMB halofit, flat-sky Limber lens-lens '22' term; window mean over [0.9 l, 1.1 l]):\n")
            md.append(md_table(["ell", "C_kk meas", "C_kk Limber", "ratio", "C_ww meas", "C_ww post-Born", "ratio", "CV rel ww", "physical BB/EE = F ww/EE", "theory F ww_th/EE", "total BB/EE"], theory,
                               lambda r: [str(r["ell"]), f"{r['C_kk_meas']:.3e}", f"{r['C_kk_theory']:.3e}", f"{r['kk_ratio']:.3f}", f"{r['C_ww_meas']:.3e}", f"{r['C_ww_theory']:.3e}", f"{r['ww_ratio']:.3f}", f"{r['cv_rel_ww']:.3f}",
                                          f"{r['physical_BB_over_EE_from_omega']:.3e}", f"{r['theory_BB_over_EE']:.3e}", f"{r['total_BB_over_EE']:.3e}"]))
    if ("zs16", "fast") in data and ("zs16", "full") in data:
        a, b = data[("zs16", "fast")], data[("zs16", "full")]
        ca, cb = spectra_of(a, tags[("zs16", "fast")]), spectra_of(b, tags[("zs16", "full")])
        rows = []
        for lo, hi in BANDS:
            if hi > 2049:
                continue
            rows.append(dict(band=[lo, hi], BB_fast_over_full=band_mean(ca["BB"], lo, hi) / band_mean(cb["BB"], lo, hi),
                             EE_fast_over_full=band_mean(ca["EE"], lo, hi) / band_mean(cb["EE"], lo, hi),
                             ww_fast_over_full=band_mean(ca["ww"], lo, hi) / band_mean(cb["ww"], lo, hi)))
        results["zs16_fast_vs_full"] = rows
        md.append("\n### zs16: fast (Nside 1024) / full (Nside 4096) band ratios, window-corrected\n")
        md.append(md_table(["band", "BB fast/full", "EE fast/full", "ww fast/full"], rows,
                           lambda r: [f"{r['band'][0]}-{r['band'][1]}", f"{r['BB_fast_over_full']:.3f}", f"{r['EE_fast_over_full']:.3f}", f"{r['ww_fast_over_full']:.3f}"]))
    for plane, fd in floors.items():
        for mode in ("full", "fast"):
            rows = []
            for lo, hi in BANDS:
                if hi > len(fd[f"ell_{mode}"]):
                    continue
                rows.append(dict(band=[lo, hi], floor_BB_over_EE=band_mean(fd[f"raw_{mode}_BB"], lo, hi) / band_mean(fd[f"raw_{mode}_EE"], lo, hi),
                                 floor_EB_over_EE=band_mean(fd[f"raw_{mode}_EB"], lo, hi) / band_mean(fd[f"raw_{mode}_EE"], lo, hi)))
            results.setdefault("floor", {}).setdefault(plane, {})[mode] = dict(rows=rows, seed=int(fd["seed"]), lmax_syn=int(fd["lmax_syn"]),
                note="Gaussian pure-E shear synthesised at Nside 4096 (lmax_syn) from the measured as-pixelised kappa spectrum (power-law extended), analysed exactly as this pass")
            md.append(f"\n### numerical B-mode floor of the {mode}-pass pipeline ({plane} spectrum, pure-E Gaussian synthesis)\n")
            md.append(md_table(["band", "floor BB/EE", "floor EB/EE"], rows, lambda r: [f"{r['band'][0]}-{r['band'][1]}", f"{r['floor_BB_over_EE']:.3e}", f"{r['floor_EB_over_EE']:+.3e}"]))
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    fig_spectra(data, tags, floors)
    fig_overlay(data, tags, floors, tables)
    fig_consistency(data, tags)
    print("\n".join(md))


if __name__ == "__main__":
    build_report()
