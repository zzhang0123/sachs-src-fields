"""
Agent N1 -- direct numerical test of the "no FK B-mode" theorem with sachsray.

Setup
-----
* A scalar potential psi(lambda, n) on the full sky (HEALPix), Gaussian,
  band-limited (l <= L_cut), lambda-correlated (AR(1) chain, corr. length lc).
* The optical tidal matrix is the screen Hessian of psi, T_ab = D_a D_b psi, so
      Phi00      = tr T / 2 = (1/2) Lap psi        -> alm * (-l(l+1)/2)
      W1 + i W2  = (1/2) eth eth psi (pure E)      -> alm_E = -(1/2) sqrt((l+2)!/(l-2)!) psi_lm, alm_B = 0
  The minus sign in alm_E is the healpy convention (checked numerically in
  convention_check(): healpy's spin-2 synthesis of +sqrt(...) psi_lm returns
  MINUS the trace-free Hessian components (H_tt - H_pp, 2 H_tp) in the
  orthonormal (e_theta, e_phi) basis). Overall sign of E is irrelevant to the
  theorem; it is fixed only so that T is literally Hess(psi).
* sachsray integrates J'' = T J per pixel (exact nonlinear Jacobi form, float64),
  A = J/D_bg, kappa = 1 - tr A/2, gamma1 = -(A00-A11)/2, gamma2 = -(A01+A10)/2.
  sachsray defines NO screen basis of its own: (W1, W2) and (gamma1, gamma2)
  refer to whatever dyad the input maps use -- here HEALPix (e_theta, e_phi).
* Order separation: trace with eps in {+1, -1, +1/2, -1/2} (same realisation)
  and Richardson-extract the homogeneous orders g1..g4 of the field expansion
  (odd/even antithetic combinations, then 1/2-scaling to split g1|g3, g2|g4).
* E/B with healpy map2alm_spin (iterated), cross spectra hp.alm2cl.
* Non-Gaussian variant: psi -> psi + q (psi^2 - <psi^2>) (skewness ~ 6 q sigma),
  low-passed back to l <= L_cut; Psi0 = (1/2) eth eth psi_NG is STILL pure E.

Usage
-----
  python n1_theorem_sachsray.py run --nside 64 --seed 1
  python n1_theorem_sachsray.py aggregate --nside 64        (tables + figure)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import healpy as hp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # src-field root (sachsray package)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

ORDERS = ("1", "2", "3", "4", "odd", "even")
FIELDS = ("E", "B", "K")


# --------------------------------------------------------------------------
# spectra / generation
# --------------------------------------------------------------------------
def band_limits(nside: int) -> tuple[int, int, int]:
    """(lmax, l_cut, L_cut): taper scale l_cut = nside; hard cut L_cut with
    2 L_cut <= lmax so quadratic products of band-limited maps are alias-free."""
    lmax = 3 * nside - 1
    l_cut = nside
    L_cut = min(int(1.4 * l_cut), lmax // 2)
    return lmax, l_cut, L_cut


def cl_psi_shape(lmax: int, l_cut: int, L_cut: int) -> np.ndarray:
    """C_l^psi (unit amplitude) such that C_l^Phi00 = (l(l+1)/2)^2 C_l^psi
    = 1/(l(l+1)) * exp(-(l/l_cut)^6), l >= 2, hard cut at L_cut."""
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1)
    L2 = ell[2:] * (ell[2:] + 1.0)
    cl[2:] = 4.0 / L2**3 * np.exp(-((ell[2:] / l_cut) ** 6))
    cl[L_cut + 1:] = 0.0
    return cl


def gaussian_psi_alm(rng, ts, lc, cl_psi, lmax):
    """(n_lam, nalm) complex alm of a Gaussian psi, AR(1)-correlated in lambda."""
    n_lam = len(ts)
    nalm = hp.Alm.getsize(lmax)
    l_idx, m_idx = hp.Alm.getlm(lmax)
    z = (rng.standard_normal((n_lam, nalm)) + 1j * rng.standard_normal((n_lam, nalm))) / np.sqrt(2.0)
    m0 = m_idx == 0
    z[:, m0] = rng.standard_normal((n_lam, int(m0.sum())))
    rho = np.exp(-np.abs(ts[:, None] - ts[None, :]) / lc)
    L = np.linalg.cholesky(rho + 1e-12 * np.eye(n_lam))
    alm = L @ z
    alm *= np.sqrt(cl_psi)[l_idx][None, :]
    return alm


def alm_to_maps(alm, nside, lmax):
    return np.stack([hp.alm2map(a, nside, lmax=lmax) for a in alm])


def maps_to_alm_lowpass(maps, lmax, L_cut, iter=3):
    ell = np.arange(lmax + 1)
    keep = (ell <= L_cut).astype(float)
    return np.stack([hp.almxfl(hp.map2alm(m, lmax=lmax, iter=iter), keep) for m in maps])


def deform_ng(psi_maps, a):
    """psi -> psi + q (psi^2 - <psi^2>), q = a / sigma_psi.  Skewness ~ 6a + 8a^3."""
    sigma = psi_maps.std()
    q = a / sigma
    return psi_maps + q * (psi_maps**2 - np.mean(psi_maps**2))


def driving_from_psi_alm(alm, nside, lmax):
    """Phi00 = (1/2) Lap psi ; (W1, W2) = (1/2) eth eth psi (pure E). Shapes (n_lam, npix)."""
    ell = np.arange(lmax + 1, dtype=float)
    f_lap = -0.5 * ell * (ell + 1.0)
    f_E = np.zeros(lmax + 1)
    f_E[2:] = -0.5 * np.sqrt((ell[2:] + 2) * (ell[2:] + 1) * ell[2:] * (ell[2:] - 1))
    phi00, w1, w2 = [], [], []
    for a in alm:
        phi00.append(hp.alm2map(hp.almxfl(a, f_lap), nside, lmax=lmax))
        q, u = hp.alm2map_spin([hp.almxfl(a, f_E), np.zeros_like(a)], nside, 2, lmax)
        w1.append(q)
        w2.append(u)
    return np.stack(phi00), np.stack(w1), np.stack(w2)


def born(ts, lam_s, phi00, w1, w2):
    """Linear (Born) maps: kappa = -int K Phi00, gamma = -int K (W1 + i W2), K = (lam_s - l) l / lam_s."""
    K = (lam_s - ts) * ts / lam_s
    kb = -np.trapezoid(K[:, None] * phi00, ts, axis=0)
    g1b = -np.trapezoid(K[:, None] * w1, ts, axis=0)
    g2b = -np.trapezoid(K[:, None] * w2, ts, axis=0)
    return kb, g1b, g2b


# --------------------------------------------------------------------------
# ray tracing + order extraction
# --------------------------------------------------------------------------
def trace(ts, phi00, w1, w2, eps, lam_s, chunk, rtol, atol):
    import sachsray as sr
    field = sr.driving_from_components(
        ts, eps * phi00, eps * w1, eps * w2, np.zeros(len(ts)), dtype=np.float64
    )
    obs = sr.trace_rays(field, lam_s, chunk=chunk, rtol=rtol, atol=atol)
    return {k: np.asarray(obs[k], dtype=np.float64) for k in ("kappa", "gamma1", "gamma2", "omega")}


def extract_orders(res: dict) -> dict:
    """res[eps] -> dict of maps. Returns orders['1'|'2'|'3'|'4'|'odd'|'even'][field]."""
    out = {}
    keys = ("kappa", "gamma1", "gamma2", "omega")
    odd1 = {k: 0.5 * (res[1.0][k] - res[-1.0][k]) for k in keys}
    even1 = {k: 0.5 * (res[1.0][k] + res[-1.0][k]) for k in keys}
    out["odd"], out["even"] = odd1, even1
    if 0.5 in res and -0.5 in res:
        oddh = {k: 0.5 * (res[0.5][k] - res[-0.5][k]) for k in keys}
        evenh = {k: 0.5 * (res[0.5][k] + res[-0.5][k]) for k in keys}
        out["1"] = {k: (8.0 * oddh[k] - odd1[k]) / 3.0 for k in keys}
        out["3"] = {k: 4.0 * (odd1[k] - 2.0 * oddh[k]) / 3.0 for k in keys}
        out["2"] = {k: (16.0 * evenh[k] - even1[k]) / 3.0 for k in keys}
        out["4"] = {k: 4.0 * (even1[k] - 4.0 * evenh[k]) / 3.0 for k in keys}
    return out


# --------------------------------------------------------------------------
# E/B and spectra
# --------------------------------------------------------------------------
def map2alm_spin_iter(q, u, lmax, n_iter=3):
    alm = hp.map2alm_spin([q, u], 2, lmax)
    for _ in range(n_iter):
        qq, uu = hp.alm2map_spin(alm, hp.npix2nside(len(q)), 2, lmax)
        corr = hp.map2alm_spin([q - qq, u - uu], 2, lmax)
        alm = [alm[0] + corr[0], alm[1] + corr[1]]
    return alm


def eb_alms(maps: dict, lmax: int, variant: str = "id") -> dict:
    g1, g2 = maps["gamma1"], maps["gamma2"]
    if variant == "flip2":
        g2 = -g2
    elif variant == "swap":
        g1, g2 = g2, g1
    elif variant == "flip1":
        g1 = -g1
    E, B = map2alm_spin_iter(g1, g2, lmax)
    K = hp.map2alm(maps["kappa"], lmax=lmax, iter=3)
    return {"E": E, "B": B, "K": K}


def all_spectra(alms: dict, lmax: int) -> dict:
    """alms[order][field] -> dict 'C_XY_ab' of C_l arrays for all order pairs."""
    C = {}
    orders = [o for o in ORDERS if o in alms]
    for i, a in enumerate(orders):
        for b in orders[i:]:
            for X in FIELDS:
                for Y in FIELDS:
                    C[f"C_{X}{Y}_{a}{b}"] = hp.alm2cl(alms[a][X], alms[b][Y], lmax=lmax)
    return C


def bands_for(nside: int):
    lmax, _, L_cut = band_limits(nside)
    edges = sorted({2, 10, 30, 100, L_cut + 1, 2 * L_cut + 1})
    edges = [e for e in edges if e <= lmax + 1]
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def band_sum(cl, lo, hi):
    ell = np.arange(len(cl))
    m = (ell >= lo) & (ell <= hi)
    return float(np.sum((2 * ell[m] + 1) * cl[m]))


def band_ratios(C: dict, nside: int) -> dict:
    """The task's r0..r3 plus diagnostics, per band."""
    bands = bands_for(nside)
    out = {}
    ell = np.arange(3 * nside)
    kfac = (ell + 2.0) * (ell - 1.0) / np.where(ell > 0, ell * (ell + 1.0), 1.0)
    for lo, hi in bands:
        S = lambda key: band_sum(C[key], lo, hi)  # noqa: E731
        nmodes = float(np.sum(2 * ell[(ell >= lo) & (ell <= hi)] + 1))
        d = {"nmodes": nmodes}
        d["r0_BB11_over_EE11"] = S("C_BB_11") / S("C_EE_11")
        d["r1_BB21_over_EE21"] = S("C_BB_12") / S("C_EE_12")
        d["r2_BB21_norm"] = S("C_BB_12") / np.sqrt(S("C_BB_22") * S("C_EE_11"))
        d["r3_BB22_over_EE22"] = S("C_BB_22") / S("C_EE_22")
        d["rhoEE_21"] = S("C_EE_12") / np.sqrt(S("C_EE_22") * S("C_EE_11"))
        d["rhoBB_21_true"] = S("C_BB_12") / np.sqrt(S("C_BB_22") * S("C_BB_11"))
        d["fEE_21_over_EE11"] = S("C_EE_12") / S("C_EE_11")
        d["fEE_22_over_EE11"] = S("C_EE_22") / S("C_EE_11")
        d["fBB_22_over_EE11"] = S("C_BB_22") / S("C_EE_11")
        d["fBB_21_over_EE11"] = S("C_BB_12") / S("C_EE_11")
        d["EB_21_over_EE21"] = S("C_EB_12") / S("C_EE_12")
        d["BE_21_over_EE21"] = S("C_BE_12") / S("C_EE_12")
        # antithetic (raw odd/even) versions, as defined in the task
        d["r0_raw_odd"] = S("C_BB_oddodd") / S("C_EE_oddodd")
        d["r1_raw_evenodd"] = S("C_BB_oddeven") / S("C_EE_oddeven")
        d["r2_raw_evenodd"] = S("C_BB_oddeven") / np.sqrt(S("C_BB_eveneven") * S("C_EE_oddodd"))
        d["r3_raw_even"] = S("C_BB_eveneven") / S("C_EE_eveneven")
        # higher-order cross pieces (5th order): <g2 g3>, <g4 g1>
        d["BB23_over_EE12"] = S("C_BB_23") / S("C_EE_12")
        d["BB14_over_EE12"] = S("C_BB_14") / S("C_EE_12")
        d["EE23_over_EE12"] = S("C_EE_23") / S("C_EE_12")
        d["EE14_over_EE12"] = S("C_EE_14") / S("C_EE_12")
        # gate: EE(1,1) vs kappa(1,1) * (l+2)(l-1)/(l(l+1))
        m = (ell >= lo) & (ell <= hi)
        d["gate_EE11_over_KK11_expected"] = S("C_EE_11") / float(
            np.sum((2 * ell[m] + 1) * C["C_KK_11"][m] * kfac[m]))
        d["gate_KE11_corr"] = S("C_KE_11") / np.sqrt(S("C_KK_11") * S("C_EE_11"))
        # null-noise estimate for the BB(2,1) cross (independent maps): sqrt(sum (2l+1) C_BB^22 C_BB^11)
        d["null_sigma_r2"] = float(np.sqrt(np.sum((2 * ell[m] + 1) * C["C_BB_22"][m] * C["C_BB_11"][m]))
                                   / np.sqrt(S("C_BB_22") * S("C_EE_11")))
        d["null_sigma_r1"] = float(np.sqrt(np.sum((2 * ell[m] + 1) * C["C_BB_22"][m] * C["C_BB_11"][m]))
                                   / abs(S("C_EE_12")))
        out[f"{lo}-{hi}"] = d
    return out


# --------------------------------------------------------------------------
# convention checks
# --------------------------------------------------------------------------
def convention_check(nside=64):
    """healpy spin-2 synthesis of sqrt((l+2)!/(l-2)!) psi_lm vs the analytic
    trace-free Hessian of psi = sin^2(theta) cos(2 phi) in the (e_theta, e_phi)
    orthonormal basis. Returns the sign s such that alm2map_spin gives
    s * (H_tt - H_pp, 2 H_tp)."""
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    th, ph = hp.pix2ang(nside, np.arange(npix))
    ell = np.arange(lmax + 1)
    fac2 = np.zeros(lmax + 1)
    fac2[2:] = np.sqrt((ell[2:] + 2) * (ell[2:] + 1) * ell[2:] * (ell[2:] - 1))
    keep = (ell <= 2).astype(float)
    psi = np.sin(th) ** 2 * np.cos(2 * ph)
    alm = hp.almxfl(hp.map2alm(psi, lmax=lmax, iter=3), keep)
    Q, U = hp.alm2map_spin([hp.almxfl(alm, fac2), np.zeros_like(alm)], nside, 2, lmax)
    H_tt = 2 * np.cos(2 * th) * np.cos(2 * ph)
    H_pp = -4 * np.cos(2 * ph) + 2 * np.cos(th) ** 2 * np.cos(2 * ph)
    H_tp = -4 * np.cos(th) * np.sin(2 * ph) + 2 * np.cos(th) * np.sin(2 * ph)
    lap = hp.alm2map(hp.almxfl(alm, -ell * (ell + 1.0)), nside, lmax=lmax)
    ep = np.abs(Q - (H_tt - H_pp)).max() + np.abs(U - 2 * H_tp).max()
    em = np.abs(Q + (H_tt - H_pp)).max() + np.abs(U + 2 * H_tp).max()
    return {"sign_spin2": -1 if em < ep else +1, "err_plus": float(ep), "err_minus": float(em),
            "lap_err": float(np.abs(lap - (H_tt + H_pp)).max())}


# --------------------------------------------------------------------------
# run one seed
# --------------------------------------------------------------------------
def run(args):
    import jax
    jax.config.update("jax_enable_x64", True)

    t_start = time.perf_counter()
    nside = args.nside
    lmax, l_cut, L_cut = band_limits(nside)
    npix = hp.nside2npix(nside)
    ts = np.linspace(0.0, args.lam_s, args.n_lam)
    eps_list = [float(e) for e in args.eps.split(",")]
    rng = np.random.default_rng(args.seed)
    log = {"nside": nside, "lmax": lmax, "l_cut": l_cut, "L_cut": L_cut, "npix": npix,
           "n_lam": args.n_lam, "lam_s": args.lam_s, "lc": args.lc, "seed": args.seed,
           "eps": eps_list, "rtol": args.rtol, "atol": args.atol, "ng_a": args.ng_a}
    log["convention"] = convention_check(min(nside, 64))
    print("convention check:", log["convention"], flush=True)

    # --- Gaussian potential, normalised to the target Born kappa rms ---
    cl = cl_psi_shape(lmax, l_cut, L_cut)
    alm_g = gaussian_psi_alm(rng, ts, args.lc, cl, lmax)
    phi00, w1, w2 = driving_from_psi_alm(alm_g, nside, lmax)
    kb, _, _ = born(ts, args.lam_s, phi00, w1, w2)
    scale = args.kappa_rms / kb.std()
    alm_g *= scale
    psi_g = alm_to_maps(alm_g, nside, lmax)
    log["psi_rms"] = float(psi_g.std())
    log["scale"] = float(scale)

    cases = {"gauss": alm_g}
    if args.ng_a > 0:
        psi_ng = deform_ng(psi_g, args.ng_a)
        x = (psi_ng - psi_ng.mean()) / psi_ng.std()
        log["ng_skewness_psi"] = float(np.mean(x**3))
        log["ng_kurtosis_psi"] = float(np.mean(x**4) - 3)
        cases["ng"] = maps_to_alm_lowpass(psi_ng, lmax, L_cut)
        del psi_ng
    del psi_g

    results = {}
    for case, alm in cases.items():
        phi00, w1, w2 = driving_from_psi_alm(alm, nside, lmax)
        kb, g1b, g2b = born(ts, args.lam_s, phi00, w1, w2)
        xb = (phi00 - phi00.mean()) / phi00.std()
        stats = {"phi00_rms": float(phi00.std()), "w_rms": float(np.hypot(w1, w2).std()),
                 "phi00_skew": float(np.mean(xb**3)), "kappa_born_rms": float(kb.std())}
        print(f"[{case}] {stats}", flush=True)
        res = {}
        for eps in eps_list:
            t0 = time.perf_counter()
            res[eps] = trace(ts, phi00, w1, w2, eps, args.lam_s, args.chunk, args.rtol, args.atol)
            print(f"[{case}] eps={eps:+.2f} traced {npix} rays in {time.perf_counter()-t0:.1f} s; "
                  f"kappa rms {res[eps]['kappa'].std():.3e} omega rms {res[eps]['omega'].std():.2e}", flush=True)
        orders = extract_orders(res)
        # Born comparison for the linear order
        o1 = orders.get("1", orders["odd"])
        stats["born_corr_kappa"] = float(np.corrcoef(o1["kappa"], kb)[0, 1])
        stats["born_corr_gamma1"] = float(np.corrcoef(o1["gamma1"], g1b)[0, 1])
        stats["born_corr_gamma2"] = float(np.corrcoef(o1["gamma2"], g2b)[0, 1])
        stats["born_rms_ratio_kappa"] = float(o1["kappa"].std() / kb.std())
        stats["born_max_abs_diff_gamma"] = float(np.max(np.abs(o1["gamma1"] - g1b) + np.abs(o1["gamma2"] - g2b)))
        for n in ("1", "2", "3", "4"):
            if n in orders:
                stats[f"rms_gamma_{n}"] = float(np.hypot(orders[n]["gamma1"], orders[n]["gamma2"]).std())
                stats[f"rms_kappa_{n}"] = float(orders[n]["kappa"].std())
                stats[f"rms_omega_{n}"] = float(orders[n]["omega"].std())
        # E/B decomposition (identity convention) for all orders
        alms = {n: eb_alms(orders[n], lmax) for n in orders}
        C = all_spectra(alms, lmax)
        # Born gamma leakage floor (pure E by construction, only pixelisation)
        Eb, Bb = map2alm_spin_iter(g1b, g2b, lmax)
        C["C_EE_born"] = hp.alm2cl(Eb, lmax=lmax)
        C["C_BB_born"] = hp.alm2cl(Bb, lmax=lmax)
        # convention variants on the linear map (gate)
        conv = {}
        for v in ("id", "flip2", "swap", "flip1"):
            a = eb_alms(o1, lmax, variant=v)
            ce, cb = hp.alm2cl(a["E"], lmax=lmax), hp.alm2cl(a["B"], lmax=lmax)
            conv[v] = {f"{lo}-{hi}": band_sum(cb, lo, hi) / band_sum(ce, lo, hi) for lo, hi in bands_for(nside)}
        stats["convention_gate_BB_over_EE_linear"] = conv
        stats["bands"] = band_ratios(C, nside)
        results[case] = {"stats": stats, "C": C,
                         "maps": {n: {k: orders[n][k].astype(np.float32) for k in orders[n]} for n in ("1", "2", "3", "4") if n in orders}}
        del res, orders, alms

    log["runtime_s"] = time.perf_counter() - t_start
    tag = f"{args.tag}nside{nside}_seed{args.seed}"
    npz = {}
    for case, r in results.items():
        for k, v in r["C"].items():
            npz[f"{case}/{k}"] = v
        for n, mm in r["maps"].items():
            for k, v in mm.items():
                npz[f"{case}/map_{n}_{k}"] = v
    np.savez_compressed(os.path.join(OUT, f"{tag}.npz"), **npz)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump({"log": log, "stats": {c: r["stats"] for c, r in results.items()}}, f, indent=1)
    print(f"done {tag} in {log['runtime_s']:.0f} s", flush=True)


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def aggregate(args):
    import glob
    files = sorted(glob.glob(os.path.join(OUT, f"{args.tag}nside{args.nside}_seed*.json")))
    if not files:
        sys.exit("no results")
    runs = [json.load(open(f)) for f in files]
    bands = list(runs[0]["stats"]["gauss"]["bands"].keys())
    keys = ["r0_BB11_over_EE11", "r1_BB21_over_EE21", "r2_BB21_norm", "r3_BB22_over_EE22", "rhoEE_21",
            "fEE_21_over_EE11", "fEE_22_over_EE11", "fBB_22_over_EE11", "fBB_21_over_EE11",
            "EB_21_over_EE21", "r0_raw_odd", "r1_raw_evenodd", "r2_raw_evenodd", "r3_raw_even",
            "BB23_over_EE12", "BB14_over_EE12", "EE23_over_EE12", "EE14_over_EE12",
            "gate_EE11_over_KK11_expected", "gate_KE11_corr", "null_sigma_r2", "null_sigma_r1", "nmodes"]
    summary = {}
    for case in runs[0]["stats"]:
        summary[case] = {}
        for b in bands:
            summary[case][b] = {}
            for k in keys:
                vals = np.array([r["stats"][case]["bands"][b][k] for r in runs])
                summary[case][b][k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                                       "vals": vals.tolist()}
    with open(os.path.join(OUT, f"{args.tag}summary_nside{args.nside}.json"), "w") as f:
        json.dump({"n_seeds": len(runs), "files": files, "summary": summary,
                   "logs": [r["log"] for r in runs], "stats": [r["stats"] for r in runs]}, f, indent=1)
    # ---- text tables ----
    lines = [f"# nside={args.nside}, seeds={[r['log']['seed'] for r in runs]}\n"]
    for case in summary:
        lines.append(f"\n## case: {case}\n")
        lines.append("| band | Nmodes | r0=BB11/EE11 | r1=BB21/EE21 | r2=BB21/sqrt(BB22 EE11) | r3=BB22/EE22 | rhoEE21 | EE21/EE11 | EE22/EE11 | BB22/EE11 | gate EE11/KK11 | null sig r1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in bands:
            s = summary[case][b]
            f = lambda k: f"{s[k]['mean']:+.2e} +- {s[k]['std']:.1e}"  # noqa: E731
            lines.append(f"| {b} | {int(s['nmodes']['mean'])} | {f('r0_BB11_over_EE11')} | {f('r1_BB21_over_EE21')} | "
                         f"{f('r2_BB21_norm')} | {f('r3_BB22_over_EE22')} | {f('rhoEE_21')} | {f('fEE_21_over_EE11')} | "
                         f"{f('fEE_22_over_EE11')} | {f('fBB_22_over_EE11')} | {f('gate_EE11_over_KK11_expected')} | {s['null_sigma_r1']['mean']:.1e} |")
        lines.append("\nraw antithetic (eps=+-1 only) versions:\n")
        lines.append("| band | r0_raw(odd) | r1_raw(even x odd) | r2_raw | r3_raw(even) | BB23/EE12 | BB14/EE12 | EE23/EE12 | EE14/EE12 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for b in bands:
            s = summary[case][b]
            f = lambda k: f"{s[k]['mean']:+.2e} +- {s[k]['std']:.1e}"  # noqa: E731
            lines.append(f"| {b} | {f('r0_raw_odd')} | {f('r1_raw_evenodd')} | {f('r2_raw_evenodd')} | {f('r3_raw_even')} | "
                         f"{f('BB23_over_EE12')} | {f('BB14_over_EE12')} | {f('EE23_over_EE12')} | {f('EE14_over_EE12')} |")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, f"{args.tag}tables_nside{args.nside}.md"), "w") as f:
        f.write(txt)
    print(txt)
    make_figure(args, runs)


def make_figure(args, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nside = args.nside
    lmax, _, L_cut = band_limits(nside)
    ell = np.arange(lmax + 1)
    pref = ell * (ell + 1) / (2 * np.pi)
    cases = list(runs[0]["stats"].keys())
    fig, axes = plt.subplots(2, len(cases), figsize=(6.2 * len(cases), 9.5), squeeze=False)
    for j, case in enumerate(cases):
        # average spectra over seeds
        Cs = {}
        for r in runs:
            d = np.load(os.path.join(OUT, f"{args.tag}nside{nside}_seed{r['log']['seed']}.npz"))
            for k in d.files:
                if k.startswith(case + "/"):
                    Cs.setdefault(k.split("/")[1], []).append(d[k])
        Cm = {k: np.mean(v, axis=0) for k, v in Cs.items() if k.startswith("C_")}
        ax = axes[0, j]
        sel = ell >= 2
        ax.loglog(ell[sel], (pref * Cm["C_EE_11"])[sel], "k-", lw=2, label=r"$C^{EE}_{(1,1)}$ (O0)")
        ax.loglog(ell[sel], (pref * Cm["C_KK_11"])[sel], "k:", lw=1.2, label=r"$C^{\kappa\kappa}_{(1,1)}$")
        ax.loglog(ell[sel], (pref * Cm["C_BB_11"])[sel], "-", color="0.6", lw=1, label=r"$C^{BB}_{(1,1)}$ (leakage)")
        ax.loglog(ell[sel], (pref * Cm["C_EE_22"])[sel], "C0-", lw=1.5, label=r"$C^{EE}_{(2,2)}$ (FF-type)")
        ax.loglog(ell[sel], (pref * Cm["C_BB_22"])[sel], "C0--", lw=1.5, label=r"$C^{BB}_{(2,2)}$ (FF-type B)")
        y = pref * Cm["C_EE_12"]
        ax.loglog(ell[sel & (y > 0)], y[sel & (y > 0)], "C3.", ms=4, label=r"$C^{EE}_{(2,1)}$ (FK-type, +)")
        ax.loglog(ell[sel & (y < 0)], -y[sel & (y < 0)], "C3.", ms=4, mfc="none", label=r"$C^{EE}_{(2,1)}$ (FK-type, $-$)")
        y = pref * Cm["C_BB_12"]
        ax.loglog(ell[sel & (y > 0)], y[sel & (y > 0)], "C2x", ms=4, label=r"$|C^{BB}_{(2,1)}|$ (theorem: 0)")
        ax.loglog(ell[sel & (y < 0)], -y[sel & (y < 0)], "C2x", ms=4, alpha=0.5)
        ax.axvline(L_cut, color="0.8", ls=":")
        ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\ell(\ell+1)|C_\ell|/2\pi$")
        ax.set_title(f"{case}: seed-averaged spectra (nside={nside}, {len(runs)} seeds)")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        ax.set_xlim(2, lmax)
        # ratios per band
        ax = axes[1, j]
        bands = list(runs[0]["stats"][case]["bands"].keys())
        x = np.arange(len(bands))
        for k, lab, mk, col in (("r1_BB21_over_EE21", r"$r_1 = C^{BB}_{(2,1)}/C^{EE}_{(2,1)}$", "o", "C2"),
                                ("r2_BB21_norm", r"$r_2 = C^{BB}_{(2,1)}/\sqrt{C^{BB}_{(2,2)}C^{EE}_{(1,1)}}$", "s", "C4"),
                                ("r3_BB22_over_EE22", r"$r_3 = C^{BB}_{(2,2)}/C^{EE}_{(2,2)}$", "^", "C0"),
                                ("r0_BB11_over_EE11", r"$r_0 = C^{BB}_{(1,1)}/C^{EE}_{(1,1)}$", "v", "0.5"),
                                ("rhoEE_21", r"$\rho^{EE}_{(2,1)}$ (FK-type EE corr.)", "d", "C3")):
            vals = np.array([[r["stats"][case]["bands"][b][k] for b in bands] for r in runs])
            m, s = vals.mean(0), (vals.std(0, ddof=1) if len(runs) > 1 else 0 * vals.mean(0))
            ax.errorbar(x + 0.05 * (hash(k) % 5 - 2), np.abs(m), yerr=s, fmt=mk, color=col, label=lab, capsize=3)
        ax.axhspan(0.42, 0.95, color="C1", alpha=0.2, label="paper's claimed FK B/E (0.42-0.95)")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(bands)
        ax.set_xlabel(r"$\ell$ band"); ax.set_ylabel("|ratio| (band-summed $(2\\ell+1)C_\\ell$)")
        ax.set_title(f"{case}: B/E ratios (mean +- seed scatter)")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(1e-8, 3)
    fig.tight_layout()
    path = os.path.join(OUT, f"{args.tag}figure_nside{nside}.png")
    fig.savefig(path, dpi=130)
    print("->", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "aggregate"])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--n_lam", type=int, default=96)
    p.add_argument("--lam_s", type=float, default=2.0)
    p.add_argument("--lc", type=float, default=0.25)
    p.add_argument("--kappa_rms", type=float, default=0.03)
    p.add_argument("--ng_a", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eps", type=str, default="1,-1,0.5,-0.5")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    if args.mode == "run":
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
