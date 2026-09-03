#!/usr/bin/env python
"""N2 products-level E/B audit of the paper's deployed 2PCF sweeps.

Tests the THEOREM (a pure-E linear response forces C_BB^FK = 0 identically,
i.e. the d^l_{2,2} transform of xi_+ and the d^l_{2,-2} transform of xi_-
must agree) on the manuscript's own production sweeps, with the manuscript's
own curved-sky (Wigner-d) transform, ported verbatim in
prototype/fk_mc/figure12.py.

Steps (numbered as in the task):
  1. load the four sweeps, tabulate xi_kk, xi_+, xi_-, xi_12, xi_01 vs gamma
  2. reproduce the paper's EE/BB (S = T22[xi_+], D = T2-2[xi_-])
  3. the E-consistency ratio D/S per ell
  4. robustness of the FK B-mode to the large-gamma treatment / DC subtraction
  5. vertex-version sensitivity (June cut1000 vs Aug cut15360 permfix)
  6. the pure-E prediction of xi_-^FK from xi_+^FK (curved-sky and flat-sky)
  7. note on kappa-E (not testable for B from xi_01 alone)

Run:  /opt/homebrew/Caskroom/miniconda/base/bin/python n2_products_eb.py
Writes results.npz, results.json, tables.md and fig_*.png next to this file.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc")
import figure12 as F12  # noqa: E402  (verbatim port of analysis3's transform)

ROOT = Path("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses"
            "/sachs_sft/sftwick_outputs/2PCF")
SWEEPS = {
    "O0": (ROOT / "C_corr_op_O0/xi_C_corr_op_O0.npz", 0),
    "FF": (ROOT / "C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz", 2),
    "FK": (ROOT / "C_corr_op_K_limber_FK_cut15360_permfix"
           / "xi_C_corr_op_K_limber_FK_cut15360_permfix.npz", 2),
    "FK_june": (ROOT / "C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz", 2),
}
ELL_PAPER = F12.ELL                                   # the manuscript's 30 multipoles
ELL = np.array(sorted(set(ELL_PAPER.astype(int).tolist()) | {300}), float)  # + ell=300
QUOTE_ELLS = (60, 300, 1500)
BAND = (50.0, 1500.0)                                 # the paper's quoted band
ARCMIN = math.pi / 180.0 / 60.0
DEG1 = 60.0                                           # the paper's convergence edge (1 deg)


# ----------------------------------------------------------------------------
# 1. loading, ported verbatim from analysis3/plot_analysis3_cl_decomposition.py
# ----------------------------------------------------------------------------
def _gamma_arcmin(xi, yi):
    xi = np.asarray(xi, float) / np.linalg.norm(xi)
    yi = np.asarray(yi, float) / np.linalg.norm(yi)
    return np.degrees(np.arccos(np.clip(np.dot(xi, yi), -1, 1))) * 60.0


def load_sweep_order(path: Path, order: int):
    """Verbatim port of analysis3.load_sweep_order (+ a grid-consistency check)."""
    # allow_pickle: x, y are object arrays of direction tuples in the operator's
    # own local sft-wick outputs (same trust statement as analysis3.LOAD_KW).
    with np.load(path, allow_pickle=True) as d:
        a, b, o = np.asarray(d["a"]), np.asarray(d["b"]), np.asarray(d["order"])
        v, x, y = np.asarray(d["value"], float), d["x"], d["y"]
    grouped: dict[tuple[int, int], np.ndarray] = {}
    g_ref = None
    for (pa, pb) in sorted({(int(ai), int(bi)) for ai, bi in zip(a, b)}):
        m = (a == pa) & (b == pb) & (o == order)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        g = np.array([_gamma_arcmin(x[i], y[i]) for i in idx])
        s = np.argsort(g)
        if g_ref is None:
            g_ref = g[s]
        else:
            assert np.allclose(g[s], g_ref), f"gamma grid differs for pair {(pa, pb)}"
        grouped[(pa, pb)] = v[m][s]
    if g_ref is None:
        raise ValueError(f"no rows at order={order} in {path}")
    return g_ref, grouped


def observables(grouped):
    g = grouped
    return {
        "xi_kk": g[(0, 0)], "xi_11": g[(1, 1)], "xi_22": g[(2, 2)],
        "xi_p": g[(1, 1)] + g[(2, 2)], "xi_m": g[(1, 1)] - g[(2, 2)],
        "xi_12": g[(1, 2)], "xi_01": g[(0, 1)], "xi_02": g[(0, 2)],
    }


# ----------------------------------------------------------------------------
# transform helpers (thin wrappers over the verbatim port)
# ----------------------------------------------------------------------------
_MAT_CACHE: dict = {}


def curved_setup(gamma, ell, m, n):
    key = (gamma.tobytes(), tuple(int(l) for l in ell), m, n)
    if key not in _MAT_CACHE:
        _MAT_CACHE[key] = F12.build_curved_matrix(gamma, ell, m, n)
    return _MAT_CACHE[key]


def eb_from_xi(gamma, xi_p, xi_m, ell=ELL, dc=True):
    """S = T22[xi_+] (=EE+BB), D = T2-2[xi_-] (=EE-BB), EE, BB."""
    S = F12.forward_curved(xi_p, curved_setup(gamma, ell, 2, 2), dc)
    D = F12.forward_curved(xi_m, curved_setup(gamma, ell, 2, -2), dc)
    return S, D, 0.5 * (S + D), 0.5 * (S - D)


def at_ell(ell, arr, l0):
    i = int(np.flatnonzero(ell == l0)[0])
    return float(arr[i])


def band_median(ell, arr, lo=BAND[0], hi=BAND[1]):
    m = (ell >= lo) & (ell <= hi)
    return float(np.median(arr[m]))


def summarize_ratio(ell, num, den):
    r = num / den
    out = {f"l{int(l)}": at_ell(ell, r, l) for l in QUOTE_ELLS}
    out["median_50_1500"] = band_median(ell, r)
    return out


# ----------------------------------------------------------------------------
# Wigner-d streaming recurrence (same recurrence as the port; no big matrix)
# ----------------------------------------------------------------------------
def wigner_iter(x, m, n, lmax):
    """Yield (L, d^L_{mn}(x)) for L = max(|m|,|n|) .. lmax (upward recurrence)."""
    x = np.asarray(x, float)
    s2sq = np.clip((1.0 - x) / 2.0, 0.0, None)
    c2sq = np.clip((1.0 + x) / 2.0, 0.0, None)
    if (m, n) == (2, 2):
        seed = c2sq * c2sq
    elif (m, n) == (2, -2):
        seed = s2sq * s2sq
    else:
        raise ValueError((m, n))
    lmin = 2
    dlm1, dl = np.zeros_like(x), np.array(seed, float)
    yield lmin, dl
    for L in range(lmin, lmax):
        a1 = (2 * L + 1) * (L * (L + 1) * x - m * n)
        t = (L * L - m * m) * (L * L - n * n)
        a2 = (L + 1) * math.sqrt(t) if t > 0 else 0.0
        denom = L * math.sqrt(((L + 1) ** 2 - m * m) * ((L + 1) ** 2 - n * n))
        dlp1 = (a1 * dl - a2 * dlm1) / denom
        dlm1, dl = dl, dlp1
        yield L + 1, dl


def dense_cl(gamma, xi, m, n, lmax, dc=True, n_fine=20000):
    """C_l for every integer l in [2, lmax] with the port's quadrature convention."""
    theta = gamma * ARCMIN
    lt = np.log(theta)
    lt_f = np.linspace(lt.min(), lt.max(), n_fine)
    theta_f = np.exp(lt_f)
    meas = 2.0 * math.pi * np.sin(theta_f) * theta_f * np.gradient(lt_f)
    xi = np.asarray(xi, float)
    if dc:
        xi = xi - xi[-1]
    xi_f = PchipInterpolator(lt, xi)(lt_f)
    w = meas * xi_f
    ells = np.arange(2, lmax + 1)
    cl = np.empty(ells.size)
    for L, d in wigner_iter(np.cos(theta_f), m, n, lmax):
        cl[L - 2] = float(d @ w)
    return ells, cl


def forward_sum(gamma, ells, cl, m, n, taper=None):
    """xi(gamma) = sum_l (2l+1)/(4 pi) C_l d^l_{mn}(gamma)  (optionally tapered)."""
    x = np.cos(gamma * ARCMIN)
    wl = np.ones_like(cl) if taper is None else taper
    out = np.zeros_like(x)
    lmax = int(ells[-1])
    for L, d in wigner_iter(x, m, n, lmax):
        out += (2 * L + 1) / (4.0 * math.pi) * cl[L - 2] * wl[L - 2] * d
    return out


# ----------------------------------------------------------------------------
# small-angle extension (below the 0.5' edge) and the flat-sky pure-E relation
# ----------------------------------------------------------------------------
def extend_small(gamma, xi_p, xi_m, g_lo=0.005, n_ext=20):
    """Extend the 2PCF below 0.5': xi_+ -> constant xi_+(0.5'), xi_- -> the
    0.5'-1.3' power law (band-limited d^l_{2,-2} behaviour).  Returns (g, xi_p, xi_m)."""
    g_ext = np.geomspace(g_lo, gamma[0], n_ext + 1)[:-1]
    A, p, ok = powerlaw_fit(gamma, xi_m, 0.0, 1.3)
    xm_ext = A * g_ext ** p if ok else np.full_like(g_ext, xi_m[0])
    return (np.concatenate([g_ext, gamma]),
            np.concatenate([np.full_like(g_ext, xi_p[0]), xi_p]),
            np.concatenate([xm_ext, xi_m]))


def flat_xi_minus_from_plus(gamma, xi_p, ext="const", n_fine=20000):
    """xi_-(t) = xi_+(t) + int_0^t (dt' t'/t^2) xi_+(t') [4 - 12 (t'/t)^2]  (flat sky,
    pure E).  xi_+ below gamma[0] is extended as a constant or as a power law."""
    lt = np.log(gamma)
    if ext == "const":
        g0, y0 = 1e-3 * gamma[0], xi_p[0]
    else:
        A, p, ok = powerlaw_fit(gamma, xi_p, 0.0, 1.3)
        g0 = 1e-3 * gamma[0]
        y0 = A * g0 ** p if ok else xi_p[0]
    lt_all = np.concatenate([[math.log(g0)], lt])
    xp_all = np.concatenate([[y0], xi_p])
    lt_f = np.linspace(lt_all.min(), lt_all.max(), n_fine)
    g_f = np.exp(lt_f)
    xp_f = PchipInterpolator(lt_all, xp_all)(lt_f)
    dl = lt_f[1] - lt_f[0]
    out = np.empty_like(gamma)
    for j, gj in enumerate(gamma):
        mm = g_f <= gj
        u = g_f[mm] / gj
        # int_0^t dt' t'/t^2 f(t') [4 - 12 u^2] = int dln t' u^2 f [4 - 12 u^2]
        out[j] = xi_p[j] + np.sum(xp_f[mm] * u * u * (4.0 - 12.0 * u * u)) * dl
    return out


# ----------------------------------------------------------------------------
# step 4 helpers: large-gamma treatments of a 2PCF
# ----------------------------------------------------------------------------
def truncate(gamma, xi_dict, gmax):
    m = gamma <= gmax * 1.005
    return gamma[m], {k: v[m] for k, v in xi_dict.items()}


def powerlaw_fit(gamma, y, lo, hi):
    """Fit y = A gamma^p on lo <= gamma <= hi (requires one sign); returns (A, p, ok)."""
    m = (gamma >= lo) & (gamma <= hi)
    yy = y[m]
    if yy.size < 3 or not (np.all(yy > 0) or np.all(yy < 0)):
        return math.nan, math.nan, False
    sgn = 1.0 if yy[0] > 0 else -1.0
    p, lnA = np.polyfit(np.log(gamma[m]), np.log(sgn * yy), 1)
    return sgn * math.exp(lnA), float(p), True


def replace_tail(gamma, xi, mode, edge=DEG1, fit_lo=15.0):
    """Replace xi(gamma > edge) by: 'zero', 'const' (=xi at last gamma<=edge),
    or 'powerlaw' (fit on [fit_lo, edge], fallback const)."""
    xi = np.array(xi, float)
    tail = gamma > edge
    last = int(np.flatnonzero(~tail)[-1])
    if mode == "zero":
        xi[tail] = 0.0
    elif mode == "const":
        xi[tail] = xi[last]
    elif mode == "powerlaw":
        A, p, ok = powerlaw_fit(gamma, xi, fit_lo, edge)
        xi[tail] = A * gamma[tail] ** p if ok else xi[last]
    else:
        raise ValueError(mode)
    return xi


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    assert F12.selftest_wigner()
    R: dict = {"ELL": ELL.tolist(), "ELL_paper": ELL_PAPER.tolist()}
    npz: dict = {"ELL": ELL, "ELL_paper": ELL_PAPER}

    # ---- 1. load + tabulate ----------------------------------------------
    data = {}
    gamma_ref = None
    for tag, (path, order) in SWEEPS.items():
        g, grouped = load_sweep_order(path, order)
        if gamma_ref is None:
            gamma_ref = g
        assert np.allclose(g, gamma_ref), f"{tag}: gamma grid differs from O0"
        data[tag] = observables(grouped)
        for k, v in data[tag].items():
            npz[f"{tag}_{k}"] = v
    gamma = gamma_ref
    npz["gamma_arcmin"] = gamma
    R["gamma_arcmin"] = gamma.tolist()
    R["n_gamma"] = int(gamma.size)
    R["parity_null_max_abs"] = {tag: {"xi_12": float(np.max(np.abs(d["xi_12"]))),
                                      "xi_02": float(np.max(np.abs(d["xi_02"])))}
                                for tag, d in data.items()}
    print(f"gamma grid: {gamma.min():.4f}..{gamma.max():.2f} arcmin, {gamma.size} pts")
    print("parity nulls max|xi_12|, max|xi_02|:", R["parity_null_max_abs"])

    # ---- 2. reproduce the paper's EE/BB with its transform ------------------
    EB = {}
    for tag in SWEEPS:
        S, D, EE, BB = eb_from_xi(gamma, data[tag]["xi_p"], data[tag]["xi_m"])
        EB[tag] = {"S": S, "D": D, "EE": EE, "BB": BB}
        for k, v in EB[tag].items():
            npz[f"EB_{tag}_{k}"] = v
    R["step2_BB_over_EE"] = {tag: summarize_ratio(ELL, EB[tag]["BB"], EB[tag]["EE"])
                             for tag in SWEEPS}
    R["step2_FK_over_O0"] = {
        "EE_FK_over_EE_O0": summarize_ratio(ELL, EB["FK"]["EE"], EB["O0"]["EE"]),
        "BB_FK_over_EE_O0 (Letter f_BB)": summarize_ratio(ELL, EB["FK"]["BB"], EB["O0"]["EE"]),
        "S_FK_over_S_O0": summarize_ratio(ELL, EB["FK"]["S"], EB["O0"]["S"]),
        "D_FK_over_D_O0": summarize_ratio(ELL, EB["FK"]["D"], EB["O0"]["D"]),
    }
    # the same on the paper's exact ELL grid (no ell=300) -> byte-level reproduction
    S_p, D_p, EE_p, BB_p = eb_from_xi(gamma, data["FK"]["xi_p"], data["FK"]["xi_m"], ELL_PAPER)
    m50 = (ELL_PAPER >= 50) & (ELL_PAPER <= 1500)
    R["step2_paper_grid_FK"] = {
        "BB_over_EE_l60": at_ell(ELL_PAPER, BB_p / EE_p, 60),
        "BB_over_EE_l1500": at_ell(ELL_PAPER, BB_p / EE_p, 1500),
        "BB_over_EE_median_50_1500": float(np.median((BB_p / EE_p)[m50])),
    }
    S_f, D_f, EE_f, BB_f = eb_from_xi(gamma, data["FF"]["xi_p"], data["FF"]["xi_m"], ELL_PAPER)
    R["step2_paper_grid_FF"] = {
        "BB_over_EE_median_50_1500": float(np.median((BB_f / EE_f)[m50])),
        "EE_over_BB_median_50_1500": float(np.median((EE_f / BB_f)[m50])),
    }
    print("\n[2] BB/EE (this transform):")
    for tag in SWEEPS:
        print(f"   {tag:8s}", {k: f"{v:+.4f}" for k, v in R["step2_BB_over_EE"][tag].items()})
    print("   paper-grid FK:", R["step2_paper_grid_FK"], " FF:", R["step2_paper_grid_FF"])

    # cross-check against the analysis3 module (run n2_crosscheck_analysis3.py first)
    xc = HERE / "crosscheck_analysis3.npz"
    if xc.exists():
        with np.load(xc) as c:
            diffs = {}
            for tag in SWEEPS:
                S, D, _, _ = eb_from_xi(gamma, data[tag]["xi_p"], data[tag]["xi_m"], ELL_PAPER)
                diffs[tag] = {"max_abs_dS": float(np.max(np.abs(S - c[f"S_{tag}"]))),
                              "max_abs_dD": float(np.max(np.abs(D - c[f"D_{tag}"]))),
                              "max_rel_dS": float(np.max(np.abs(S / c[f"S_{tag}"] - 1))),
                              "max_rel_dD": float(np.max(np.abs(D / c[f"D_{tag}"] - 1)))}
            R["step2_crosscheck_vs_analysis3"] = diffs
            if "ckk_pyccl" in c.files:
                ckk = c["ckk_pyccl"]
                S0, D0, _, _ = eb_from_xi(gamma, data["O0"]["xi_p"], data["O0"]["xi_m"], ELL_PAPER)
                Ckk0 = F12.forward_curved(data["O0"]["xi_kk"], curved_setup(gamma, ELL_PAPER, 0, 0))
                R["step2_O0_vs_pyccl"] = {
                    "ell": ELL_PAPER.tolist(),
                    "S_O0_over_pyccl": (S0 / ckk).tolist(),
                    "D_O0_over_pyccl": (D0 / ckk).tolist(),
                    "Ckk_O0_over_pyccl": (Ckk0 / ckk).tolist(),
                }
        print("   cross-check vs analysis3:", diffs)

    # ---- 3. E-consistency ratio D/S per ell ---------------------------------
    R["step3_D_over_S"] = {"ell": ELL.tolist()}
    for tag in SWEEPS:
        r = EB[tag]["D"] / EB[tag]["S"]
        R["step3_D_over_S"][tag] = r.tolist()
        npz[f"DoverS_{tag}"] = r
    R["step3_summary"] = {tag: summarize_ratio(ELL, EB[tag]["D"], EB[tag]["S"]) for tag in SWEEPS}
    R["step3_departure_from_1"] = {
        tag: {"max_abs_50_1500": float(np.max(np.abs(
            (EB[tag]["D"] / EB[tag]["S"] - 1)[(ELL >= 50) & (ELL <= 1500)])))}
        for tag in SWEEPS}
    print("\n[3] D/S:", {t: {k: f"{v:.4f}" for k, v in s.items()} for t, s in R["step3_summary"].items()})

    # ---- 4. robustness of the FK B-mode ---------------------------------------
    rows = []

    def add_row(label, g, xp, xm, dc, note=""):
        S, D, EE, BB = eb_from_xi(g, xp, xm, ELL, dc)
        ratio = summarize_ratio(ELL, BB, EE)
        fbb = summarize_ratio(ELL, BB, EB["O0"]["EE"])
        fee = summarize_ratio(ELL, EE, EB["O0"]["EE"])
        rows.append({"variant": label, "dc": dc, "gamma_max_arcmin": float(g.max()),
                     "BB_over_EE": ratio, "BB_over_EE_O0": fbb, "EE_over_EE_O0": fee,
                     "note": note, "_EE": EE, "_BB": BB})

    fk = data["FK"]
    add_row("FK deployed (83 deg)", gamma, fk["xi_p"], fk["xi_m"], True)
    add_row("FK deployed (83 deg)", gamma, fk["xi_p"], fk["xi_m"], False)
    for gmax_deg in (40, 20, 10, 5, 2, 1):
        g_t, d_t = truncate(gamma, fk, gmax_deg * 60.0)
        add_row(f"FK truncated gamma_max={gmax_deg} deg", g_t, d_t["xi_p"], d_t["xi_m"], True)
        add_row(f"FK truncated gamma_max={gmax_deg} deg", g_t, d_t["xi_p"], d_t["xi_m"], False)
    fits = {}
    for mode in ("zero", "const", "powerlaw"):
        xp = replace_tail(gamma, fk["xi_p"], mode)
        xm = replace_tail(gamma, fk["xi_m"], mode)
        note = ""
        if mode == "powerlaw":
            for nm, arr in (("xi_p", fk["xi_p"]), ("xi_m", fk["xi_m"])):
                A, p, ok = powerlaw_fit(gamma, arr, 15.0, DEG1)
                fits[nm] = {"A": A, "p": p, "ok": ok}
            note = f"fit 15'-60': xi_+ ~ g^{fits['xi_p']['p']:.2f}, xi_- ~ g^{fits['xi_m']['p']:.2f}"
        add_row(f"FK tail(>1 deg) -> {mode}", gamma, xp, xm, True, note)
        add_row(f"FK tail(>1 deg) -> {mode}", gamma, xp, xm, False, note)
    # O0 control rows (the transform's own floor under the same surgery)
    o0 = data["O0"]
    add_row("O0 deployed (83 deg)", gamma, o0["xi_p"], o0["xi_m"], True)
    add_row("O0 deployed (83 deg)", gamma, o0["xi_p"], o0["xi_m"], False)
    for gmax_deg in (10, 2, 1):
        g_t, d_t = truncate(gamma, o0, gmax_deg * 60.0)
        add_row(f"O0 truncated gamma_max={gmax_deg} deg", g_t, d_t["xi_p"], d_t["xi_m"], True)
    for mode in ("zero", "const"):
        add_row(f"O0 tail(>1 deg) -> {mode}", gamma, replace_tail(gamma, o0["xi_p"], mode),
                replace_tail(gamma, o0["xi_m"], mode), True)
    # the transform's small-angle floor: extend both 2PCFs below 0.5' (see extend_small)
    for tag in ("O0", "FK", "FK_june"):
        g_e, xp_e, xm_e = extend_small(gamma, data[tag]["xi_p"], data[tag]["xi_m"])
        _, p_m, _ = powerlaw_fit(gamma, data[tag]["xi_m"], 0.0, 1.3)
        add_row(f"{tag} extended below 0.5' (xi_+ const, xi_- ~ g^{p_m:.2f})", g_e, xp_e, xm_e, True,
                "removes the missing [0,0.5'] piece of S")
    R["step4_powerlaw_tail_fits"] = fits
    R["step4_rows"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    npz["step4_EE"] = np.array([r["_EE"] for r in rows])
    npz["step4_BB"] = np.array([r["_BB"] for r in rows])
    npz["step4_labels"] = np.array([f"{r['variant']} | dc={r['dc']}" for r in rows])
    print("\n[4] FK robustness (BB/EE at 60/300/1500, median):")
    for r in rows:
        q = r["BB_over_EE"]
        print(f"   {r['variant']:36s} dc={str(r['dc']):5s} "
              f"{q['l60']:+.3f} {q['l300']:+.3f} {q['l1500']:+.3f} med {q['median_50_1500']:+.3f}  {r['note']}")

    # ---- 5. vertex-version sensitivity + small-gamma exponent ----------------
    R["step5"] = {"BB_over_EE": {t: R["step2_BB_over_EE"][t] for t in ("FK", "FK_june")},
                  "BB_FK_june_over_BB_FK": summarize_ratio(ELL, EB["FK_june"]["BB"], EB["FK"]["BB"]),
                  "EE_FK_june_over_EE_FK": summarize_ratio(ELL, EB["FK_june"]["EE"], EB["FK"]["EE"]),
                  "xi_kk_june_over_aug_at_0.5arcmin": float(data["FK_june"]["xi_kk"][0] / fk["xi_kk"][0])}
    small = {}
    for tag in ("FK", "FK_june", "O0", "FF"):
        d = data[tag]
        ratio = d["xi_m"] / d["xi_p"]
        loc_slope_m = np.gradient(np.log(np.abs(d["xi_m"])), np.log(gamma))
        loc_slope_r = np.gradient(np.log(np.abs(ratio)), np.log(gamma))
        _, p_m, ok_m = powerlaw_fit(gamma, d["xi_m"], 0.0, 2.1)
        _, p_r, ok_r = powerlaw_fit(gamma, ratio, 0.0, 2.1)
        small[tag] = {
            "ratio_xim_over_xip": ratio.tolist(),
            "local_slope_xi_m": loc_slope_m.tolist(),
            "local_slope_ratio": loc_slope_r.tolist(),
            "fit_exponent_xi_m_0.5to2arcmin": p_m if ok_m else None,
            "fit_exponent_ratio_0.5to2arcmin": p_r if ok_r else None,
        }
        npz[f"ratio_xim_xip_{tag}"] = ratio
    R["step5_small_gamma"] = small
    print("\n[5] small-gamma exponents (0.5'-2'):",
          {t: (s["fit_exponent_xi_m_0.5to2arcmin"], s["fit_exponent_ratio_0.5to2arcmin"]) for t, s in small.items()})

    # ---- 6. pure-E prediction of xi_- from xi_+ -------------------------------
    R["step6"] = {}
    LMAX_DIRECT = 3000      # direct Wigner transform of the 40-point 2PCF up to here
    for tag in ("O0", "FK", "FK_june"):
        d = data[tag]
        ells, cl_p = dense_cl(gamma, d["xi_p"], 2, 2, LMAX_DIRECT)
        _, cl_m = dense_cl(gamma, d["xi_m"], 2, -2, LMAX_DIRECT)
        npz[f"dense_ell"] = ells
        npz[f"dense_S_{tag}"] = cl_p
        npz[f"dense_D_{tag}"] = cl_m
        # power-law extension of S_l beyond LMAX_DIRECT, fitted on [1000, 3000]
        mfit = (ells >= 1000) & (ells <= LMAX_DIRECT)
        pl, lnA = np.polyfit(np.log(ells[mfit]), np.log(np.abs(cl_p[mfit])), 1)
        sgn = np.sign(np.mean(cl_p[mfit]))
        preds = {}
        for lsum in (3000, 6000, 12000, 24000):
            ells_x = np.arange(2, lsum + 1)
            cl_x = np.empty(ells_x.size)
            cl_x[: cl_p.size] = cl_p
            cl_x[cl_p.size:] = sgn * np.exp(lnA) * ells_x[cl_p.size:] ** pl
            # cos^2 taper over the last 25% of the ell range (kills the hard-edge ringing)
            wl = np.ones(ells_x.size)
            l0 = 0.75 * lsum
            tt = ells_x > l0
            wl[tt] = np.cos(0.5 * math.pi * (ells_x[tt] - l0) / (lsum - l0)) ** 2
            xi_m_pred = forward_sum(gamma, ells_x, cl_x, 2, -2, wl)
            xi_p_back = forward_sum(gamma, ells_x, cl_x, 2, 2, wl)
            preds[lsum] = (xi_m_pred, xi_p_back)
            npz[f"pred_xim_{tag}_l{lsum}"] = xi_m_pred
            npz[f"pred_xip_{tag}_l{lsum}"] = xi_p_back
        xi_m_dep = d["xi_m"] - d["xi_m"][-1]   # DC-subtracted deployed (same convention)
        xi_p_dep = d["xi_p"] - d["xi_p"][-1]
        # flat-sky pure-E relation (Crittenden+2002 / Schneider+2002), xi_- from xi_+:
        #   xi_-(t) = xi_+(t) + int_0^t (dt' t'/t^2) xi_+(t') [4 - 12 (t'/t)^2]
        # (a constant xi_+ maps to xi_- = 0, so no DC issue; only xi_+ at t' <= t
        # is needed, i.e. only the CONVERGED small-angle FK data for t <= 1 deg).
        # xi_+ below 0.5' is extended as a constant or as the 0.5'-1.3' power law.
        xi_m_flat = {}
        for ext in ("const", "powerlaw"):
            xi_m_flat[ext] = flat_xi_minus_from_plus(gamma, d["xi_p"], ext)
            npz[f"pred_xim_flat_{ext}_{tag}"] = xi_m_flat[ext]
        np.seterr(divide="ignore", invalid="ignore")
        R["step6"][tag] = {
            "S_powerlaw_slope_1000_3000": float(pl),
            "ratio_pred_over_deployed_xi_m": {
                f"lsum{lsum}": (preds[lsum][0] / xi_m_dep).tolist() for lsum in preds},
            "ratio_backtransformed_over_deployed_xi_p": {
                f"lsum{lsum}": (preds[lsum][1] / xi_p_dep).tolist() for lsum in preds},
            "ratio_flatsky_pred_over_deployed_xi_m": {
                ext: (xi_m_flat[ext] / d["xi_m"]).tolist() for ext in xi_m_flat},
            "xi_m_deployed_dc": xi_m_dep.tolist(),
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            r6 = {lsum: preds[lsum][0] / xi_m_dep for lsum in preds}
            rf = {ext: xi_m_flat[ext] / d["xi_m"] for ext in xi_m_flat}
        qg = (1, 2, 5, 10, 30, 60)
        print(f"\n[6] {tag}: xi_-^pred/xi_-^deployed at gamma = {qg} arcmin")
        for lsum in preds:
            print(f"     curved lsum={lsum:5d}: " + " ".join(f"{np.interp(gg, gamma, r6[lsum]):8.3f}" for gg in qg))
        for ext in rf:
            print(f"     flat-sky ({ext:8s}): " + " ".join(f"{np.interp(gg, gamma, rf[ext]):8.3f}" for gg in qg))

    # ---- 7. kappa-E cross (note only) -----------------------------------------
    R["step7_note"] = ("C_l^{kappa E} from xi_01 (kernel d^l_{2,0}) is a single spectrum; "
                       "the theorem's C^{kappa B}=0 needs <kappa gamma_x> at a rotated frame, "
                       "not available from the sweeps' xi_01/xi_02 (xi_02 is the parity null). Skipped.")
    for tag in ("O0", "FK"):
        CkE = F12.forward_curved(-data[tag]["xi_01"], curved_setup(gamma, ELL, 2, 0))
        npz[f"CkE_{tag}"] = CkE
    R["step7_CkE_FK_over_O0"] = summarize_ratio(ELL, npz["CkE_FK"], npz["CkE_O0"])

    # ---- save ------------------------------------------------------------------
    np.savez(HERE / "results.npz", **npz)
    (HERE / "results.json").write_text(json.dumps(R, indent=1, default=float))
    write_tables(HERE / "tables.md", gamma, data, EB, rows, R)
    make_figures(gamma, data, EB, rows, npz, R)
    print(f"\n-> {HERE/'results.npz'}, results.json, tables.md, fig_*.png")
    return 0


# ----------------------------------------------------------------------------
# tables / figures
# ----------------------------------------------------------------------------
def write_tables(path, gamma, data, EB, rows, R):
    L = []
    L.append("# N2 tables (auto-generated by n2_products_eb.py)\n")
    L.append("## 1. 2PCF sweeps versus gamma (subset of rows)\n")
    sel = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
    for tag in SWEEPS:
        d = data[tag]
        L.append(f"\n### {tag}\n")
        L.append("| gamma ['] | xi_kk | xi_+ | xi_- | xi_12 | xi_01 | xi_-/xi_+ |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|")
        for i in sel:
            L.append(f"| {gamma[i]:.3f} | {d['xi_kk'][i]:+.4e} | {d['xi_p'][i]:+.4e} | {d['xi_m'][i]:+.4e} "
                     f"| {d['xi_12'][i]:+.2e} | {d['xi_01'][i]:+.4e} | {d['xi_m'][i]/d['xi_p'][i]:+.4e} |")
    L.append("\n## 2/3. E/B from the paper's transform (S=T22[xi_+], D=T2-2[xi_-]; EE=(S+D)/2, BB=(S-D)/2)\n")
    L.append("| ell | S_O0 | D_O0 | BB/EE O0 | D/S O0 | BB/EE FF | D/S FF | S_FK | D_FK | BB/EE FK | D/S FK | BB/EE FK_june | D/S FK_june | f_BB=BB_FK/EE_O0 | EE_FK/EE_O0 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, l in enumerate(ELL):
        e = {t: EB[t] for t in SWEEPS}
        L.append(f"| {int(l)} | {e['O0']['S'][i]:+.3e} | {e['O0']['D'][i]:+.3e} | {e['O0']['BB'][i]/e['O0']['EE'][i]:+.4f} | {e['O0']['D'][i]/e['O0']['S'][i]:.4f} "
                 f"| {e['FF']['BB'][i]/e['FF']['EE'][i]:+.4f} | {e['FF']['D'][i]/e['FF']['S'][i]:.4f} "
                 f"| {e['FK']['S'][i]:+.3e} | {e['FK']['D'][i]:+.3e} | {e['FK']['BB'][i]/e['FK']['EE'][i]:+.4f} | {e['FK']['D'][i]/e['FK']['S'][i]:.4f} "
                 f"| {e['FK_june']['BB'][i]/e['FK_june']['EE'][i]:+.4f} | {e['FK_june']['D'][i]/e['FK_june']['S'][i]:.4f} "
                 f"| {e['FK']['BB'][i]/e['O0']['EE'][i]:+.2e} | {e['FK']['EE'][i]/e['O0']['EE'][i]:+.4f} |")
    L.append("\n## 4. FK B-mode robustness (BB/EE at ell = 60 / 300 / 1500, median 50-1500; f_BB = BB/EE_O0)\n")
    L.append("| variant | DC sub | gamma_max ['] | BB/EE l=60 | l=300 | l=1500 | median | f_BB l=60 | l=300 | l=1500 | EE/EE_O0 l=1500 | note |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        q, f, e = r["BB_over_EE"], r["BB_over_EE_O0"], r["EE_over_EE_O0"]
        L.append(f"| {r['variant']} | {r['dc']} | {r['gamma_max_arcmin']:.0f} | {q['l60']:+.3f} | {q['l300']:+.3f} | {q['l1500']:+.3f} | {q['median_50_1500']:+.3f} "
                 f"| {f['l60']:+.2e} | {f['l300']:+.2e} | {f['l1500']:+.2e} | {e['l1500']:+.4f} | {r['note']} |")
    L.append("\n## 6. pure-E prediction xi_-^pred / xi_-^deployed: curved-sky (lsum 6000 / 12000, DC-subtracted) and flat-sky (const / power-law extension below 0.5')\n")
    L.append("| gamma ['] | O0 c6000 | O0 c12000 | O0 flat-c | O0 flat-p | FK c6000 | FK c12000 | FK flat-c | FK flat-p | FK_june c12000 | FK_june flat-c |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    s6 = R["step6"]
    for i in range(len(gamma)):
        c = lambda t, k: s6[t]["ratio_pred_over_deployed_xi_m"][k][i]  # noqa: E731
        f = lambda t, k: s6[t]["ratio_flatsky_pred_over_deployed_xi_m"][k][i]  # noqa: E731
        L.append(f"| {gamma[i]:.3f} | {c('O0','lsum6000'):+.3f} | {c('O0','lsum12000'):+.3f} | {f('O0','const'):+.3f} | {f('O0','powerlaw'):+.3f} "
                 f"| {c('FK','lsum6000'):+.3f} | {c('FK','lsum12000'):+.3f} | {f('FK','const'):+.3f} | {f('FK','powerlaw'):+.3f} "
                 f"| {c('FK_june','lsum12000'):+.3f} | {f('FK_june','const'):+.3f} |")
    path.write_text("\n".join(L) + "\n")


def _signed(ax, x, y, **kw):
    y = np.asarray(y)
    ax.plot(x, np.abs(y), **kw)
    neg = y < 0
    if neg.any():
        ax.plot(x[neg], np.abs(y[neg]), "o", mfc="none", color=kw.get("color", "k"), ms=6, lw=0)


def make_figures(gamma, data, EB, rows, npz, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"O0": "C0", "FF": "C1", "FK": "C2", "FK_june": "C3"}
    # fig 1: the 2PCFs
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, ttl in zip(axes, ("xi_p", "xi_m", "xi_kk"), (r"$\xi_+$", r"$\xi_-$", r"$\xi_{\kappa\kappa}$")):
        for tag in SWEEPS:
            _signed(ax, gamma, data[tag][key], color=cols[tag], label=tag, lw=1.4)
        ax.axvline(60, color="0.6", ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(ttl + "  (hollow = negative)")
        ax.set_xlabel(r"$\gamma$ [arcmin]")
    axes[0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_2pcf.png", dpi=130); plt.close(fig)

    # fig 2: EE/BB reproduction + BB/EE ratio + D/S
    pref = ELL * (ELL + 1) / (2 * math.pi)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for tag in ("FF", "FK", "FK_june"):
        _signed(ax, ELL, pref * EB[tag]["EE"], color=cols[tag], label=f"{tag} EE", lw=1.6)
        _signed(ax, ELL, pref * EB[tag]["BB"], color=cols[tag], label=f"{tag} BB", lw=1.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-16, 1e-5)
    ax.set_title(r"$\ell(\ell+1)\Delta C_\ell/2\pi$ (paper Fig. cl_EB, hollow = negative)"); ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel(r"$\ell$")
    ax = axes[1]
    for tag in SWEEPS:
        ax.plot(ELL, EB[tag]["BB"] / EB[tag]["EE"], "-o", color=cols[tag], label=tag, ms=4)
    ax.axhline(0, color="k", lw=0.8); ax.axvspan(3, 50, color="0.9")
    ax.set_xscale("log"); ax.set_ylim(-0.5, 1.5); ax.set_title(r"$C^{BB}/C^{EE}$ per term"); ax.legend(fontsize=9)
    ax.set_xlabel(r"$\ell$")
    ax = axes[2]
    for tag in SWEEPS:
        ax.plot(ELL, EB[tag]["D"] / EB[tag]["S"], "-o", color=cols[tag], label=tag, ms=4)
    ax.axhline(1, color="k", lw=0.8); ax.axvspan(3, 50, color="0.9")
    ax.set_xscale("log"); ax.set_ylim(-1, 2); ax.set_title(r"E-consistency  $D/S = T_{2,-2}[\xi_-]/T_{2,2}[\xi_+]$ (pure E: 1)")
    ax.set_xlabel(r"$\ell$"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_eb_DS.png", dpi=130); plt.close(fig)

    # fig 3: robustness
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = npz["step4_labels"]
    for i, lab in enumerate(labels):
        if not lab.startswith("FK"):
            continue
        dc = "dc=True" in lab
        y = npz["step4_BB"][i] / npz["step4_EE"][i]
        ax = axes[0] if "truncated" in lab or "deployed" in lab else axes[1]
        ax.plot(ELL, y, "-" if dc else "--", label=lab.replace("FK ", ""), lw=1.3, ms=3, marker="o" if dc else None)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8); ax.axvspan(3, 50, color="0.9"); ax.set_xscale("log")
        ax.set_ylim(-0.6, 1.6); ax.set_xlabel(r"$\ell$"); ax.legend(fontsize=7)
    axes[0].set_title(r"FK $C^{BB}/C^{EE}$: truncation angle (solid = DC sub, dashed = none)")
    axes[1].set_title(r"FK $C^{BB}/C^{EE}$: tail $\gamma>1^\circ$ replaced")
    fig.tight_layout(); fig.savefig(HERE / "fig_robustness.png", dpi=130); plt.close(fig)

    # fig 4: pure-E prediction of xi_-
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, tag in enumerate(("O0", "FK", "FK_june")):
        d = data[tag]
        dep = d["xi_m"] - d["xi_m"][-1]
        ax = axes[0, j]
        _signed(ax, gamma, dep, color="k", label=r"deployed $\xi_-$ (DC-sub)", lw=2)
        for lsum, c in ((3000, "C0"), (6000, "C1"), (12000, "C2"), (24000, "C3")):
            _signed(ax, gamma, npz[f"pred_xim_{tag}_l{lsum}"], color=c, label=f"pure-E pred, lsum={lsum}", lw=1.1)
        _signed(ax, gamma, npz[f"pred_xim_flat_const_{tag}"], color="C4", label="pure-E pred, flat-sky (const ext)", lw=1.1, ls="--")
        _signed(ax, gamma, npz[f"pred_xim_flat_powerlaw_{tag}"], color="C5", label="pure-E pred, flat-sky (power-law ext)", lw=1.1, ls="--")
        _signed(ax, gamma, d["xi_p"] - d["xi_p"][-1], color="0.5", label=r"deployed $\xi_+$ (DC-sub)", lw=1, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{tag}: $\\xi_-$ deployed vs pure-E prediction")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            for lsum, c in ((3000, "C0"), (6000, "C1"), (12000, "C2"), (24000, "C3")):
                ax.plot(gamma, npz[f"pred_xim_{tag}_l{lsum}"] / dep, "-", color=c, lw=1.1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(gamma, npz[f"pred_xim_flat_const_{tag}"] / d["xi_m"], "--", color="C4", lw=1.1)
            ax.plot(gamma, npz[f"pred_xim_flat_powerlaw_{tag}"] / d["xi_m"], "--", color="C5", lw=1.1)
        ax.axhline(1, color="k", lw=0.8); ax.axvline(60, color="0.6", ls=":")
        ax.set_xscale("log"); ax.set_ylim(-1, 6); ax.set_xlabel(r"$\gamma$ [arcmin]")
        ax.set_title(r"$\xi_-^{\rm pred}/\xi_-^{\rm deployed}$")
    fig.tight_layout(); fig.savefig(HERE / "fig_pureE_prediction.png", dpi=130); plt.close(fig)

    # fig 5: dense S_l, D_l and their ratio
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ells = npz["dense_ell"]
    pref_d = ells * (ells + 1) / (2 * math.pi)
    for tag in ("O0", "FK", "FK_june"):
        _signed(axes[0], ells, pref_d * npz[f"dense_S_{tag}"], color=cols[tag], label=f"{tag} S=T22[xi+]", lw=1.2)
        _signed(axes[0], ells, pref_d * npz[f"dense_D_{tag}"], color=cols[tag], label=f"{tag} D=T2-2[xi-]", lw=1.2, ls="--")
        axes[1].plot(ells, npz[f"dense_D_{tag}"] / npz[f"dense_S_{tag}"], color=cols[tag], label=tag, lw=1.2)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].legend(fontsize=7); axes[0].set_xlabel(r"$\ell$")
    axes[0].set_title(r"dense integer-$\ell$ transforms (hollow = negative)")
    axes[1].axhline(1, color="k", lw=0.8); axes[1].set_xscale("log"); axes[1].set_ylim(-1, 2)
    axes[1].set_title(r"$D_\ell/S_\ell$ (pure E: 1)"); axes[1].legend(fontsize=8); axes[1].set_xlabel(r"$\ell$")
    fig.tight_layout(); fig.savefig(HERE / "fig_dense_DS.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
