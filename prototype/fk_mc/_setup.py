"""Path wiring for the FK Monte-Carlo: import the paper's MC engine + canoes.

The paper's analysis code lives in the STF_lensing drafts repo and hardcodes a
stale canoes path; we prepend the correct one (canoes at angular_statistics) and
the paper MC directory, then re-export the engine + driver-stats. The cached
canoes Sigma2/zeta .npz tables load with numpy alone (no pyccl at runtime).
"""
from __future__ import annotations

import sys
from pathlib import Path

CANOES_SRC = "/Users/zzhang/projects/angular_statistics/canoes/src"
PAPER_MC = (
    "/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses"
    "/sachs_sft/analyses/mc_sachs_2pt"
)
PAPER_SACHS_SFT = (
    "/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft"
)

for _p in (CANOES_SRC, PAPER_MC):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import sachs_mc_core as mc  # noqa: E402
import driver_stats as ds  # noqa: E402
import background as bg  # noqa: E402

__all__ = ["mc", "ds", "bg", "PAPER_SACHS_SFT"]
