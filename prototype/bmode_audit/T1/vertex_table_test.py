"""Per-shell pure-E consistency of the deployed kappa3 vertex table.

For the FK shear diagram the hard pair (Phi00, Psi0) sits at n1 and the soft
Psi0 leg at n2 (cosine triple (1, cos g, cos g) with legs (T, P, P)).  If the
soft leg is pure E, then in flat sky

    zeta_Bmod(g) = 2 pi int k dk s_E(k) J_0(k g),   zeta_TPP(g) = 2 pi int k dk s_E(k) J_4(k g)

for ONE real spectrum s_E, so zeta_TPP must equal the Schneider (J0 -> J4)
image of zeta_Bmod.  Tests that on every shell of the deployed table.
"""
import sys, math
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T1")
from eb_consistency import schneider_minus_from_plus

TAB = ("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/"
       "callables/kappa3_vertex/equal_time_limber_cut15360_permaware/table_permclosed.npz")

d = np.load(TAB, allow_pickle=True)
ct = d["cosine_triples"]
# collapsed family with legs (T at n1, P at n1, P at n2): cos01 = 1, cos12 = cos20 = cos g
m = (np.abs(ct[:, 0] - 1.0) < 1e-13) & (np.abs(ct[:, 1] - ct[:, 2]) < 1e-13) & (ct[:, 1] < 1.0 - 1e-15)
rows = np.flatnonzero(m)
g = np.degrees(np.arccos(np.clip(ct[rows, 1], -1, 1))) * 60.0
s = np.argsort(g)
rows, g = rows[s], g[s]
theta = np.radians(g / 60.0)
print(f"collapsed rows: {len(rows)}, gamma range {g[0]:.3f}' .. {g[-1]:.1f}'")
print("z_shells:", np.round(d["z_shells"], 2))

te_arcmin = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0])
te = np.radians(te_arcmin / 60.0)
sel = theta <= np.radians(2.0)   # only the converged small-angle part is needed
print("\nratio zeta_TPP(actual) / zeta_TPP(pred from zeta_Bmod via J0->J4), per shell:")
print("   z    " + "  ".join(f"{t:6.1f}'" for t in te_arcmin) + "     Bmod/TTT@0.5'")
for j, z in enumerate(d["z_shells"]):
    zb = d["zeta_Bmod"][rows, j]
    zt = d["zeta_TPP"][rows, j]
    ztt = d["zeta_TTT"][rows, j]
    pred = schneider_minus_from_plus(theta[sel], zb[sel], te, "const")
    act = PchipInterpolator(np.log(theta), zt)(np.log(te))
    print(f"  {z:5.2f} " + "  ".join(f"{a/p:+7.3f}" for a, p in zip(act, pred)) + f"      {zb[0]/ztt[0]:.4f}")
print("\n|zeta_TPP/zeta_Bmod| actual vs pure-E prediction, shell z~3.1:")
j = int(np.argmin(np.abs(d["z_shells"] - 3.1)))
zb = d["zeta_Bmod"][rows, j]; zt = d["zeta_TPP"][rows, j]
pred = schneider_minus_from_plus(theta[sel], zb[sel], te, "const")
act = PchipInterpolator(np.log(theta), zt)(np.log(te)); bz = PchipInterpolator(np.log(theta), zb)(np.log(te))
for t, a, p, b in zip(te_arcmin, act, pred, bz):
    print(f"  {t:5.1f}'  actual {a/b:+.3e}   pred {p/b:+.3e}")
# small-angle slope of both
lo = (g >= 0.4) & (g <= 1.0)
sl_act = np.polyfit(np.log(g[lo]), np.log(np.abs(zt[lo])), 1)[0]
print(f"  small-angle log-slope of zeta_TPP (0.4'-1'): {sl_act:.2f}")
