#!/usr/bin/env python
"""E1: shear E/B, convergence and rotation power spectra of the Takahashi et al.
2017 full-sky ray-traced maps (allskymap_nres12r000.zs*.mag.dat, Nside 4096).

Usage (run with OMP_NUM_THREADS=24):
  python e1_takahashi_bmode.py measure --plane zs16 --mode fast   # ud_grade to Nside 1024, lmax 2048
  python e1_takahashi_bmode.py measure --plane zs16 --mode full   # native Nside 4096, lmax 4096
  python e1_takahashi_bmode.py floor   --plane zs16               # numerical B-mode floor (pure-E synthesis)
  python e1_takahashi_bmode.py report                              # tables, figures, results.json

Every step caches to outputs/*.npz and is idempotent (re-running skips work
whose cache exists unless --force).

Conventions: healpy map2alm_spin(spin=2) on (gamma1, gamma2) as (Q, U) in the
HEALPix polarization convention. Both signs of gamma2 are measured in the fast
pass; the one with C_EE ~ C_kk and C_BB << C_EE is the physical convention
(the wrong sign gives C_EE = C_BB = C_kk/2, see the synthetic test in REPORT.md).
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
OUT = os.path.join(HERE, "outputs")
DATA = "/Users/zzhang/projects/SFT-WL-B/data"
sys.path.insert(0, "/Users/zzhang/projects/SFT-WL-B/analysis/wpc_routeb")
from cb2_takahashi_io import read_field, read_header  # noqa: E402

PLANES = {"zs10": 0.574, "zs16": 1.033, "zs38": 5.342}
MODES = {"fast": (1024, 2048), "full": (4096, 4096)}
FIELDS = ("kappa", "gamma1", "gamma2", "omega")
NSIDE_NATIVE = 4096
ELL_RES = 1.6 * NSIDE_NATIVE  # Takahashi 2017 resolution damping (1 + (ell/ell_res)^2)^-1
N_ITER = 3  # Jacobi iterations for map2alm (scalar and spin alike)


def path_of(plane: str) -> str:
    return f"{DATA}/allskymap_nres12r000.{plane}.mag.dat"


def cache_path(kind: str, plane: str, mode: str) -> str:
    nside, lmax = MODES[mode]
    return os.path.join(OUT, f"{kind}_{plane}_{mode}_nside{nside}_lmax{lmax}.npz")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------
def field_stats(x: np.ndarray) -> dict:
    x64 = x.astype(np.float64)
    out = dict(
        mean=float(x64.mean()),
        rms=float(np.sqrt(np.mean(x64 * x64))),
        std=float(x64.std()),
        min=float(x64.min()),
        max=float(x64.max()),
        n_nonfinite=int(np.count_nonzero(~np.isfinite(x))),
    )
    del x64
    return out


def load_maps(plane: str, nside_out: int):
    """Read the four fields; degrade to nside_out (averaging) if requested."""
    hdr = read_header(path_of(plane))
    maps, stats = {}, {}
    for name in FIELDS:
        t0 = time.time()
        m = read_field(path_of(plane), name)
        stats[name] = field_stats(m)
        m = m.astype(np.float64)
        if nside_out != hdr.nside:
            m = hp.ud_grade(m, nside_out, order_in="RING", order_out="RING", power=None)
        maps[name] = m
        log(f"{plane}:{name} read+prep {time.time() - t0:.1f}s  stats={stats[name]}")
    return maps, stats, hdr.nside


# ----------------------------------------------------------------------------
# harmonic transforms
# ----------------------------------------------------------------------------
def map2alm_iter(m: np.ndarray, lmax: int, niter: int) -> np.ndarray:
    return hp.map2alm(m, lmax=lmax, iter=niter, use_pixel_weights=False)


def map2alm_spin_iter(q: np.ndarray, u: np.ndarray, lmax: int, niter: int):
    """map2alm_spin with the same Jacobi refinement healpy applies to scalars."""
    nside = hp.npix2nside(q.size)
    alm_e, alm_b = hp.map2alm_spin([q, u], 2, lmax=lmax)
    for _ in range(niter):
        q1, u1 = hp.alm2map_spin([alm_e, alm_b], nside, 2, lmax)
        de, db = hp.map2alm_spin([q - q1, u - u1], 2, lmax=lmax)
        alm_e = alm_e + de
        alm_b = alm_b + db
    return alm_e, alm_b


def all_spectra(alm: dict) -> dict:
    """Auto and cross spectra of kappa (k), omega (w), E, B."""
    keys = ["k", "w", "E", "B"]
    cl = {}
    for i, a in enumerate(keys):
        for b in keys[i:]:
            cl[a + b] = hp.alm2cl(alm[a], alm[b])
    return cl


def pixel_windows(nside_analysis: int, lmax: int) -> dict:
    """Pixel windows to divide out. For a degraded map the sub-pixel averaging
    of the native Nside-4096 map and the coarse-pixel window both act, so the
    product of the two windows is used (an approximation to the exact
    averaging kernel, adequate for ell << nside_analysis)."""
    pw_t, pw_p = hp.pixwin(nside_analysis, pol=True, lmax=lmax)
    if nside_analysis != NSIDE_NATIVE:
        pwn_t, pwn_p = hp.pixwin(NSIDE_NATIVE, pol=True, lmax=lmax)
        pw_t = pw_t * pwn_t
        pw_p = pw_p * pwn_p
    return dict(T=np.asarray(pw_t), P=np.asarray(pw_p))


def correct_windows(cl: dict, pw: dict) -> dict:
    win = {"k": pw["T"], "w": pw["T"], "E": pw["P"], "B": pw["P"]}
    out = {}
    for key, val in cl.items():
        w = win[key[0]] * win[key[1]]
        c = np.array(val, dtype=np.float64)
        good = w > 0
        c[good] = c[good] / w[good]
        c[~good] = 0.0
        out[key] = c
    return out


# ----------------------------------------------------------------------------
# measure
# ----------------------------------------------------------------------------
def measure(plane: str, mode: str, force: bool = False) -> str:
    nside, lmax = MODES[mode]
    fn = cache_path("spectra", plane, mode)
    if os.path.exists(fn) and not force:
        log(f"cache exists: {fn}")
        return fn
    t_start = time.time()
    maps, stats, nside_native = load_maps(plane, nside)
    assert nside_native == NSIDE_NATIVE
    t0 = time.time()
    alm = {}
    alm["k"] = map2alm_iter(maps["kappa"], lmax, N_ITER)
    log(f"map2alm kappa {time.time() - t0:.1f}s")
    t0 = time.time()
    alm["w"] = map2alm_iter(maps["omega"], lmax, N_ITER)
    log(f"map2alm omega {time.time() - t0:.1f}s")
    results = {}
    signs = (+1, -1) if mode == "fast" else (+1,)
    for sign in signs:
        t0 = time.time()
        alm["E"], alm["B"] = map2alm_spin_iter(maps["gamma1"], sign * maps["gamma2"], lmax, N_ITER)
        log(f"map2alm_spin (gamma2 sign {sign:+d}) {time.time() - t0:.1f}s")
        results[sign] = all_spectra(alm)
    pw = pixel_windows(nside, lmax)
    ell = np.arange(lmax + 1)
    damp = 1.0 / (1.0 + (ell / ELL_RES) ** 2)
    save = dict(
        ell=ell,
        pixwin_T=pw["T"],
        pixwin_P=pw["P"],
        damping_takahashi=damp,
        nside=nside,
        lmax=lmax,
        plane=plane,
        z_s=PLANES[plane],
        n_iter=N_ITER,
        stats_json=json.dumps(stats),
        wall_seconds=time.time() - t_start,
    )
    for sign, cl in results.items():
        tag = "plus" if sign > 0 else "minus"
        for key, val in cl.items():
            save[f"raw_{tag}_{key}"] = val
        for key, val in correct_windows(cl, pw).items():
            save[f"cl_{tag}_{key}"] = val
    np.savez(fn, **save)
    log(f"saved {fn}  ({time.time() - t_start:.0f}s total)")
    return fn


# ----------------------------------------------------------------------------
# numerical B-mode floor
# ----------------------------------------------------------------------------
def floor_test(plane: str, force: bool = False, seed: int = 12345) -> str:
    """Build a Gaussian, pure-E shear map whose E spectrum is the measured
    kappa spectrum of `plane` (as pixelised, extended as a power law beyond the
    measured range to lmax = 2 Nside), pixelise it at Nside 4096, and analyse it
    (a) exactly as the full pass (native, lmax 4096) and (b) exactly as the fast
    pass (ud_grade to 1024, lmax 2048). The B-mode that comes out is the
    pipeline's pixelisation/aliasing/quadrature floor for each pass."""
    fn = os.path.join(OUT, f"floor_{plane}.npz")
    if os.path.exists(fn) and not force:
        log(f"cache exists: {fn}")
        return fn
    src = cache_path("spectra", plane, "full")
    if not os.path.exists(src):
        src = cache_path("spectra", plane, "fast")
    d = np.load(src)
    cl_kk = d["cl_plus_kk"] * d["pixwin_T"] ** 2  # as-pixelised (raw) power
    lmax_syn = 2 * NSIDE_NATIVE
    ell = np.arange(lmax_syn + 1)
    cl_ext = np.zeros(lmax_syn + 1)
    n = min(len(cl_kk), lmax_syn + 1)
    cl_ext[:n] = cl_kk[:n]
    if n < lmax_syn + 1:
        l1, l2 = int(0.6 * (n - 1)), n - 1
        slope = np.log(cl_kk[l2] / cl_kk[l1]) / np.log(l2 / l1)
        cl_ext[n:] = cl_kk[l2] * (ell[n:] / l2) ** slope
    cl_ext[:2] = 0.0
    np.random.seed(seed)
    alm_k = hp.synalm(cl_ext, lmax=lmax_syn, new=True)
    l_of = hp.Alm.getlm(lmax_syn)[0].astype(float)
    with np.errstate(invalid="ignore"):
        fac = np.sqrt((l_of + 2.0) * (l_of - 1.0) / np.where(l_of > 1, l_of * (l_of + 1.0), 1.0))
    fac[l_of < 2] = 0.0
    t0 = time.time()
    q, u = hp.alm2map_spin([alm_k * fac, np.zeros_like(alm_k)], NSIDE_NATIVE, 2, lmax_syn)
    kap = hp.alm2map(alm_k, NSIDE_NATIVE, lmax=lmax_syn)
    log(f"floor synthesis at nside 4096 lmax {lmax_syn}: {time.time() - t0:.0f}s")
    del alm_k
    save = dict(cl_ext_input=cl_ext, seed=seed, lmax_syn=lmax_syn)
    for mode in ("full", "fast"):
        nside, lmax = MODES[mode]
        if nside != NSIDE_NATIVE:
            qq, uu, kk = (hp.ud_grade(x, nside, power=None) for x in (q, u, kap))
        else:
            qq, uu, kk = q, u, kap
        t0 = time.time()
        alm = {"k": map2alm_iter(kk, lmax, N_ITER)}
        alm["E"], alm["B"] = map2alm_spin_iter(qq, uu, lmax, N_ITER)
        cl = dict(kk=hp.alm2cl(alm["k"]), EE=hp.alm2cl(alm["E"]), BB=hp.alm2cl(alm["B"]),
                  EB=hp.alm2cl(alm["E"], alm["B"]), kE=hp.alm2cl(alm["k"], alm["E"]))
        save[f"ell_{mode}"] = np.arange(lmax + 1)
        for k, v in cl.items():
            save[f"raw_{mode}_{k}"] = v
        log(f"floor analysis ({mode}) {time.time() - t0:.0f}s: BB/EE at l=60 {cl['BB'][54:67].mean() / cl['EE'][54:67].mean():.2e}")
    np.savez(fn, **save)
    log(f"saved {fn}")
    return fn


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "floor", "report"])
    ap.add_argument("--plane", default="zs16", choices=list(PLANES))
    ap.add_argument("--mode", default="fast", choices=list(MODES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if args.step == "measure":
        measure(args.plane, args.mode, args.force)
    elif args.step == "floor":
        floor_test(args.plane, args.force)
    else:
        from e1_report import build_report  # noqa: WPS433
        build_report()


if __name__ == "__main__":
    main()
