"""Concrete LWE estimates for the exact PN16QP1761 QP modulus.

This file is executed with SageMath and a fixed malb/lattice-estimator
checkout.  It intentionally reports ordinary LWE attack costs only.  It does
not prove Ring-LWE, KDM/circular security, or distributed-protocol simulation.
"""

import json
import os
import subprocess

from sage.all import RR, RealField, log, oo, prod, sqrt
from sage.env import SAGE_VERSION

from estimator import LWE, ND
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import ADPS16, MATZOV


Q = [
    0x80000000080001,
    0x2000000A0001,
    0x2000000E0001,
    0x1FFFFFC20001,
    0x200000440001,
    0x200000500001,
    0x200000620001,
    0x1FFFFF980001,
    0x2000006A0001,
    0x1FFFFF7E0001,
    0x200000860001,
    0x200000A60001,
    0x200000AA0001,
    0x200000B20001,
    0x200000C80001,
    0x1FFFFF360001,
    0x200000E20001,
    0x1FFFFF060001,
    0x200000FE0001,
    0x1FFFFEDE0001,
    0x1FFFFECA0001,
    0x1FFFFEB40001,
    0x200001520001,
    0x1FFFFE760001,
    0x2000019A0001,
    0x1FFFFE640001,
    0x200001A00001,
    0x1FFFFE520001,
    0x200001E80001,
    0x1FFFFE0C0001,
    0x1FFFFDEE0001,
    0x200002480001,
    0x1FFFFDB60001,
    0x200002560001,
]
P = [
    0x80000000440001,
    0x7FFFFFFFBA0001,
    0x80000000500001,
    0x7FFFFFFFAA0001,
]

N = 2**16
SIGMA_E = RR("3.2")
Q_PRODUCT = prod(Q)
P_PRODUCT = prod(P)
QP_PRODUCT = Q_PRODUCT * P_PRODUCT

# Each participant uses Lattigo's fully-entropic uniform ternary secret.  For
# two independent shares, Var(s1+s2)=2*(2/3)=4/3 and Pr[s1+s2 != 0]=2/3.
# Current lattice-estimator attacks consume the variance/density abstraction;
# they do not model the exact five-point convolution probabilities.  We thus
# label this estimate as an aggregate-moment model, not an exact reduction.
AGGREGATE_MOMENT = ND.NoiseDistribution(
    mean=0,
    stddev=sqrt(RR(4) / 3),
    bounds=(-2, 2),
    is_Gaussian_like=False,
    _density=RR(2) / 3,
)

PARAMETERS = {
    "hes_v2_128_ternary_boundary": LWEParameters(
        n=N,
        q=2**1747,
        Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(RR("3.19")),
        m=oo,
        tag="HES-v2-N65536-logq1747-ternary",
    ),
    "p3_qp_single_ternary": LWEParameters(
        n=N,
        q=QP_PRODUCT,
        Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(SIGMA_E),
        m=oo,
        tag="PN16QP1761-exact-QP-single-ternary",
    ),
    "p3_qp_two_party_aggregate_moment": LWEParameters(
        n=N,
        q=QP_PRODUCT,
        Xs=AGGREGATE_MOMENT,
        Xe=ND.DiscreteGaussian(SIGMA_E),
        m=oo,
        tag="PN16QP1761-exact-QP-two-party-aggregate-moment",
    ),
}


def scalar(value):
    if value is None:
        return None
    try:
        if value == oo:
            return "Infinity"
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return float(value)


def cost_record(cost):
    record = {"log2_rop": float(log(cost["rop"], 2))}
    for key in ("beta", "d", "m", "eta", "zeta", "t", "p"):
        if key in cost:
            record[key] = scalar(cost[key])
    return record


def estimate_selected(params, model):
    attacks = {
        "primal_usvp": lambda: LWE.primal_usvp(
            params, red_cost_model=model, red_shape_model="gsa"
        ),
        "primal_bdd": lambda: LWE.primal_bdd(
            params, red_cost_model=model, red_shape_model="gsa"
        ),
        "dual": lambda: LWE.dual(params, red_cost_model=model),
        "dual_hybrid_matzov": lambda: LWE.dual_hybrid(
            params, red_cost_model=model
        ),
    }
    result = {}
    for name, attack in attacks.items():
        try:
            result[name] = cost_record(attack())
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    finite = [v["log2_rop"] for v in result.values() if "log2_rop" in v]
    result["minimum_log2_rop"] = min(finite) if finite else None
    return result


RF = RealField(256)


def exact_log2(value):
    return float(RF(value).log2())


try:
    ESTIMATOR_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    ESTIMATOR_COMMIT = "unavailable"


COST_MODELS = {
    "matzov_classical": MATZOV(nn="classical"),
    "matzov_quantum": MATZOV(nn="quantum"),
    "adps16_core_svp_classical": ADPS16(mode="classical"),
    "adps16_core_svp_quantum": ADPS16(mode="quantum"),
}


output = {
    "schema": "mkf-p3-qp-security-estimate-v2",
    "tool": {
        "lattice_estimator_commit": ESTIMATOR_COMMIT,
        "sage_version": SAGE_VERSION,
        "working_directory": os.getcwd(),
    },
    "estimator_scope": (
        "LWE attack-cost estimate with MATZOV classical/quantum and ADPS16 "
        "classical/quantum Core-SVP cost models; "
        "does not cover Ring-LWE structure, circular/KDM evaluation-key security, "
        "or distributed-protocol transcript simulation"
    ),
    "literal": {
        "logN": 16,
        "N": N,
        "q_count": len(Q),
        "p_count": len(P),
        "log2_Q_exact": exact_log2(Q_PRODUCT),
        "log2_P_exact": exact_log2(P_PRODUCT),
        "log2_QP_exact": exact_log2(QP_PRODUCT),
        "Q_bit_length": int(Q_PRODUCT.nbits()),
        "P_bit_length": int(P_PRODUCT.nbits()),
        "QP_bit_length": int(QP_PRODUCT.nbits()),
        "sigma_error": float(SIGMA_E),
    },
    "secret_models": {
        "single_party": "uniform ternary: Pr[-1]=Pr[0]=Pr[1]=1/3",
        "two_party_exact": "convolution: Pr[-2,-1,0,1,2]=[1,2,3,2,1]/9",
        "two_party_estimator_abstraction": {
            "mean": 0.0,
            "variance": float(RR(4) / 3),
            "stddev": float(sqrt(RR(4) / 3)),
            "nonzero_density": float(RR(2) / 3),
            "bounds": [int(-2), int(2)],
        },
    },
    "parameters": {},
}

for parameter_name, parameter in PARAMETERS.items():
    parameter_output = {
        "n": int(parameter.n),
        "log2_q": exact_log2(parameter.q),
        "m": "Infinity",
        "secret_stddev": float(parameter.Xs.stddev),
        "error_stddev": float(parameter.Xe.stddev),
    }
    for model_name, model in COST_MODELS.items():
        parameter_output[model_name] = estimate_selected(parameter, model)
    output["parameters"][parameter_name] = parameter_output

def sage_json_default(value):
    """Convert residual Sage scalar types introduced by the preparser."""
    if type(value).__name__ == "Integer":
        return int(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


print("P3_SECURITY_ESTIMATE_JSON_BEGIN")
print(json.dumps(output, indent=int(2), sort_keys=True, default=sage_json_default))
print("P3_SECURITY_ESTIMATE_JSON_END")
