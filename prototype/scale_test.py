"""
Device-agnostic scale test for the sachsray full-sky ray-tracer.

Runs on whatever JAX backend is present:
  * here (Apple M3 Ultra) -> CPU; mainstream JAX GPU is NVIDIA CUDA only, and
    the experimental jax-metal plugin is incompatible with this JAX and
    unreliable for diffrax, so the Mac GPU is not used.
  * on an NVIDIA box -> CUDA GPU, unchanged.

It (1) reports the device, (2) measures steady-state ray throughput with a
pre-compiled vmap, (3) does a memory-bounded end-to-end streaming run via
sachsray.trace_rays_streaming (peak memory ~ one chunk, full field never
materialised), and (4) projects cost/memory to nside >= 256.

    python prototype/scale_test.py
"""

import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import sachsray as sr

# float32 production mode (no x64): halves memory, fine for map-level observables.

N_LAM = 128
LAM_S = 2.0
CHUNK = 8192
RTOL, ATOL = 1e-5, 1e-7


def peak_mem_gb() -> float:
    """Peak RSS in GB (macOS reports ru_maxrss in bytes, Linux in KiB)."""
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if sys.platform == "darwin" else m * 1024 / 1e9


def make_gen(n_lam: int, lam_max: float):
    """Cheap distinct smooth-in-lambda per-ray driving (Phi00, W1, W2), float32."""
    lam = np.linspace(0.0, lam_max, n_lam).astype(np.float32)
    ks = np.arange(1, 12)
    amp = (0.02 / np.sqrt(ks)).astype(np.float32)

    def gen(start, size):
        rng = np.random.default_rng(start)  # deterministic per block
        ph = rng.uniform(0, 2 * np.pi, (size, 3, ks.size)).astype(np.float32)
        out = np.zeros((size, n_lam, 3), np.float32)
        for c in range(3):
            base = -0.05 if c == 0 else 0.0
            out[:, :, c] = base + (amp[None, None, :] * np.sin(
                ks[None, None, :] * lam[None, :, None] + ph[:, c, :][:, None, :])).sum(-1)
        return out

    return gen


def build_batched(ts, lam_s, phi00_bg):
    lam_eval = jnp.asarray([lam_s], dtype=ts.dtype)
    D_bg = sr.solvers.solve_jacobi_scalar(ts, phi00_bg, lam_eval, rtol=RTOL, atol=ATOL)[0, 0]

    def single_ray(drive):
        y = sr.solvers.solve_jacobi_matrix(ts, drive, lam_eval, rtol=RTOL, atol=ATOL)[0]
        return sr.physics.observables_from_jacobi(y[:4].reshape(2, 2), D_bg)

    return jax.jit(jax.vmap(single_ray))


def main():
    print("=" * 70)
    print(f"JAX {jax.__version__} | backend: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"n_lam={N_LAM}, chunk={CHUNK}, dtype=float32, rtol={RTOL}")
    print("=" * 70)

    ts = jnp.linspace(0.0, LAM_S, N_LAM, dtype=jnp.float32)
    phi00_bg = jnp.full((N_LAM,), -0.05, dtype=jnp.float32)
    gen = make_gen(N_LAM, LAM_S)
    batched = build_batched(ts, LAM_S, phi00_bg)

    # (2) steady-state throughput: warmup (compile) then time K chunks
    warm = jnp.asarray(gen(0, CHUNK))
    t0 = time.perf_counter()
    jax.block_until_ready(batched(warm))
    t_compile = time.perf_counter() - t0
    K = 4
    blocks = [jnp.asarray(gen(i * CHUNK, CHUNK)) for i in range(1, K + 1)]
    t0 = time.perf_counter()
    for b in blocks:
        jax.block_until_ready(batched(b))
    dt = time.perf_counter() - t0
    rate = K * CHUNK / dt
    print(f"\n[throughput]  compile {t_compile:.1f}s | steady {rate:,.0f} rays/s "
          f"({1e6/rate:.1f} us/ray)")
    drive_chunk_mb = CHUNK * N_LAM * 3 * 4 / 1e6
    print(f"  per-chunk driving on device: {drive_chunk_mb:.0f} MB "
          f"(state/scratch negligible) -> on GPU you can raise chunk a lot")

    # (3) memory-bounded end-to-end streaming run (real API)
    for nside in (16, 32, 64):
        npix = 12 * nside * nside
        gc_rss0 = peak_mem_gb()
        t0 = time.perf_counter()
        out = sr.trace_rays_streaming(ts, LAM_S, npix, gen, phi00_bg, chunk=CHUNK,
                                      rtol=RTOL, atol=ATOL)
        wall = time.perf_counter() - t0
        finite = bool(np.all(np.isfinite(np.asarray(out["kappa"]))))
        print(f"[stream]  nside={nside:>4} npix={npix:>7}  wall={wall:6.2f}s  "
              f"peakRSS={peak_mem_gb():5.2f}GB  all-finite={finite}  "
              f"kappa.std={float(np.std(np.asarray(out['kappa']))):.3e}")
        del out

    # (4) projection to large nside using measured steady-state rate
    print("\n[projection] (steady-state rate; field generated per chunk)")
    print(f"  {'nside':>6} {'npix':>10} {'full-field GB':>14} {'est. time':>12}")
    for nside in (128, 256, 512, 1024):
        npix = 12 * nside * nside
        full_gb = npix * N_LAM * 3 * 4 / 1e9
        est = npix / rate
        est_s = f"{est:.0f}s" if est < 120 else f"{est/60:.1f}min"
        print(f"  {nside:>6} {npix:>10,} {full_gb:>13.1f}  {est_s:>12}   "
              f"(working set ~{drive_chunk_mb:.0f} MB/chunk, never the full {full_gb:.1f} GB)")
    print("\nNote: 'full-field GB' is what you AVOID holding via streaming. On a CUDA GPU,")
    print("a larger chunk (e.g. 65536) and the GPU's parallelism cut est. time substantially.")


if __name__ == "__main__":
    main()
