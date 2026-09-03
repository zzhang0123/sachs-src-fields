"""Pure-E consistency test on the deployed kappa3 vertex table.

For a pure-E spin-2 field P = eth^2 phi and any spin-2 field A(n1) = T(n1) P(n1),
  Bmod(g) = <A(n1) P*(n2)>  must be  sum_l (2l+1)/4pi C_l d^l_{22}(g)
  TPP(g)  = <A(n1) P(n2)>   must be  sum_l (2l+1)/4pi C_l d^l_{2,-2}(g)
with the SAME C_l.  So T22^{-1}[Bmod] == T_{2,-2}^{-1}[TPP] shell by shell.
This script reads the (1,c,c) family from the table and tests it, and also
prints Bmod vs TTT (the audit's 'pair phase dropped' finding).
"""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc")
from figure12 import wigner_d, build_curved_matrix, forward_curved, ELL

TAB = ("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/"
       "callables/kappa3_vertex/equal_time_limber_cut15360_permaware/table_permclosed.npz")
d = np.load(TAB, allow_pickle=False)
meta = json.loads(np.asarray(d["cosmo_meta"], dtype=np.uint8).tobytes().decode())
print("meta keys:", sorted(meta.keys())[:40])
for k in ("ell_high_max", "ell_cut", "ell_max", "build_date", "built", "date", "spt_kind"):
    if k in meta: print(" ", k, meta[k])
tr = np.asarray(d["cosine_triples"], float)
lam = np.asarray(d["lambda_shells_Mpc"], float)
# collapsed family with slot 0 = T coincident with slot 1: (1, c, c)
fam = np.isclose(tr[:, 0], 1.0) & np.isclose(tr[:, 1], tr[:, 2])
idx = np.flatnonzero(fam)
g = np.degrees(np.arccos(np.clip(tr[idx, 1], -1, 1))) * 60.0
o = np.argsort(g); idx, g = idx[o], g[o]
ch = {n: np.asarray(d["zeta_" + n], float)[idx] for n in ("TTT", "TTP", "TPP", "PPP", "Bmod", "Dmod")}
print(f"{len(idx)} collapsed rows, gamma {g[0]:.3f}' .. {g[-1]:.1f}'; {lam.size} shells")
s = -1
print("\nshell", s, "lambda", lam[s])
print(f"{'gamma':>9} {'TTT':>12} {'Bmod':>12} {'Bmod/TTT':>10} {'TPP':>12} {'TPP/Bmod':>10}")
for i in range(0, len(g), max(1, len(g)//25)):
    print(f"{g[i]:9.3f} {ch['TTT'][i,s]:12.4e} {ch['Bmod'][i,s]:12.4e} {ch['Bmod'][i,s]/ch['TTT'][i,s]:10.4f} "
          f"{ch['TPP'][i,s]:12.4e} {ch['TPP'][i,s]/ch['Bmod'][i,s]:10.4f}")
r = ch['Bmod'] / ch['TTT']
print("\nBmod/TTT over ALL collapsed rows & shells: min %.4f max %.4f median %.4f" % (r.min(), r.max(), np.median(r)))
print("max |Bmod-TTT|/|TTT| :", np.max(np.abs(ch['Bmod'] - ch['TTT']) / np.abs(ch['TTT'])))
