"""Fast, reproducible MATZOV primal-uSVP gate for PN16QP1761."""

import json
import subprocess

from sage.all import RealField, RR, log, oo, prod
from sage.env import SAGE_VERSION

from estimator import LWE, ND
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import MATZOV


Q = [
    0x80000000080001, 0x2000000A0001, 0x2000000E0001,
    0x1FFFFFC20001, 0x200000440001, 0x200000500001,
    0x200000620001, 0x1FFFFF980001, 0x2000006A0001,
    0x1FFFFF7E0001, 0x200000860001, 0x200000A60001,
    0x200000AA0001, 0x200000B20001, 0x200000C80001,
    0x1FFFFF360001, 0x200000E20001, 0x1FFFFF060001,
    0x200000FE0001, 0x1FFFFEDE0001, 0x1FFFFECA0001,
    0x1FFFFEB40001, 0x200001520001, 0x1FFFFE760001,
    0x2000019A0001, 0x1FFFFE640001, 0x200001A00001,
    0x1FFFFE520001, 0x200001E80001, 0x1FFFFE0C0001,
    0x1FFFFDEE0001, 0x200002480001, 0x1FFFFDB60001,
    0x200002560001,
]
P = [
    0x80000000440001, 0x7FFFFFFFBA0001,
    0x80000000500001, 0x7FFFFFFFAA0001,
]

N = 2**16
QP = prod(Q) * prod(P)
RF = RealField(256)

PARAMETERS = {
    "hes_v2_128_ternary_boundary": LWEParameters(
        n=N, q=2**1747, Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(RR("3.19")), m=oo,
        tag="HES-v2-N65536-logq1747-ternary",
    ),
    "p3_qp_single_ternary": LWEParameters(
        n=N, q=QP, Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(RR("3.2")), m=oo,
        tag="PN16QP1761-exact-QP-single-ternary",
    ),
}


def record(cost):
    out = {"log2_rop": float(log(cost["rop"], 2))}
    for key in ("beta", "d", "m"):
        if key in cost:
            try:
                out[key] = int(cost[key])
            except Exception:
                out[key] = str(cost[key])
    return out


result = {
    "schema": "mkf-p3-fast-matzov-primal-usvp-v1",
    "tool": {
        "lattice_estimator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sage_version": SAGE_VERSION,
    },
    "literal": {
        "log2_QP_exact": float(RF(QP).log2()),
        "QP_bit_length": int(QP.nbits()),
    },
    "parameters": {},
}

for name, params in PARAMETERS.items():
    result["parameters"][name] = {
        "log2_q": float(RF(params.q).log2()),
        "matzov_classical_primal_usvp": record(
            LWE.primal_usvp(
                params, red_cost_model=MATZOV(nn="classical"),
                red_shape_model="gsa",
            )
        ),
        "matzov_quantum_primal_usvp": record(
            LWE.primal_usvp(
                params, red_cost_model=MATZOV(nn="quantum"),
                red_shape_model="gsa",
            )
        ),
    }

print("P3_SECURITY_ESTIMATE_JSON_BEGIN")
print(json.dumps(result, indent=2, sort_keys=True))
print("P3_SECURITY_ESTIMATE_JSON_END")
