"""Focused MATZOV-classical BDD sweep for the P3 QP security gate."""

import json
import subprocess

from sage.all import RR, log, oo
from sage.env import SAGE_VERSION

from estimator import LWE, ND
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import MATZOV


out = {
    "schema": "mkf-p3-matzov-classical-bdd-threshold-v1",
    "tool": {
        "lattice_estimator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sage_version": SAGE_VERSION,
    },
    "points": {},
}

for logq in (1744, 1745, 1746, 1747, 1760):
    params = LWEParameters(
        n=2**16,
        q=2**logq,
        Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(RR("3.2")),
        m=oo,
        tag=f"N65536-logq{logq}-sigma3.2-ternary",
    )
    cost = LWE.primal_bdd(
        params,
        red_cost_model=MATZOV(nn="classical"),
        red_shape_model="gsa",
    )
    out["points"][str(logq)] = {
        "log2_rop": float(log(cost["rop"], 2)),
        "beta": int(cost["beta"]),
        "d": int(cost["d"]),
        "eta": int(cost["eta"]),
    }

print("P3_SECURITY_ESTIMATE_JSON_BEGIN")
print(json.dumps(out, indent=2, sort_keys=True))
print("P3_SECURITY_ESTIMATE_JSON_END")
