"""Closing the loop: the flat-sky Limber vertex WITH the pair phase (rezeta engine,
bmod_phase=True) must satisfy zeta_TPP = J0->J4[zeta_Bmod] exactly (pure-E soft leg),
while the deployed convention (bmod_phase=False, Bmod computed as TTT) must not.
Shells z = 0.45 and 3.1 from the deployed table."""
import sys, numpy as np
sys.path.insert(0, "/Users/zzhang/projects/rezeta")
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T1")
from rezeta.limber_vertex import limber_vertex, load_spectrum, ARCMIN, H
from eb_consistency import schneider_minus_from_plus
from scipy.interpolate import PchipInterpolator

TAB = ("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/"
       "callables/kappa3_vertex/equal_time_limber_cut15360_permaware/table_permclosed.npz")
d = np.load(TAB, allow_pickle=True)
P = load_spectrum()
g = np.geomspace(0.1, 120.0, 90)            # arcmin, dense to 0.1' so the [0,theta] integral is safe
theta = g * ARCMIN
te_arcmin = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0])
te = te_arcmin * ARCMIN
SLOT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("placement slot", SLOT, "(0: pair (T,P) at n1, soft P at n2 = the shear-2PCF placement)")
for sh in (2, 8):
    z = float(d["z_shells"][sh]); chi = float(d["chi_shells_Mpc"][sh]) * H
    print(f"\nshell {sh}: z = {z:.2f}, chi = {chi:.0f} Mpc/h, hard window (60, 15360]")
    for phase in (True, False):
        out = limber_vertex(chi * theta, chi, P, regulator="hard", bmod_phase=phase, slot=SLOT)
        zb, zt = out["Bmod"], out["TPP"]
        pred = schneider_minus_from_plus(theta, zb, te, "const")
        act = PchipInterpolator(np.log(theta), zt)(np.log(te))
        print(f"  bmod_phase={phase!s:5s}  TPP(actual)/TPP(pred from Bmod): " +
              "  ".join(f"{t:g}':{a/p:+.3f}" for t, a, p in zip(te_arcmin, act, pred)))
    print(f"  Bmod(phased)/Bmod(unphased): " + "  ".join(
        f"{t:g}':{v:.3f}" for t, v in zip(te_arcmin,
        PchipInterpolator(np.log(theta), limber_vertex(chi*theta, chi, P, bmod_phase=True, slot=SLOT)["Bmod"]
                          / limber_vertex(chi*theta, chi, P, bmod_phase=False, slot=SLOT)["Bmod"])(np.log(te)))))
