"""Fixed-source ordinary-LWE gate for the exact P2G-N16 candidate.

Run only with SageMath 10.5 from malb/lattice-estimator commit
d8c00b483f70b6298690386b1a3278b5167a2e42.  This script estimates the
maximum publicly exposed Q*guard*P modulus.  It does not prove Ring-LWE,
circular/KDM security, or distributed-protocol simulation.
"""

import json
import subprocess

from sage.all import RR, RealField, log, oo, prod, sqrt
from sage.env import SAGE_VERSION

from estimator import LWE, ND
from estimator.lwe_parameters import LWEParameters
from estimator.reduction import MATZOV


EXPECTED_COMMIT = "d8c00b483f70b6298690386b1a3278b5167a2e42"
N = 2**16
SIGMA_E = RR("3.2")

# Q contains the 14 persistent levels followed by the one-shot ingress guard.
Q = [
    0xFFFFFFFFFFC0001,
    0x3FFFFFFFD60001,
    0x3FFFFFFFCA0001,
    0x3FFFFFFF360001,
    0x3FFFFFFF2A0001,
    0x3FFFFFFF000001,
    0x3FFFFFFEFA0001,
    0x3FFFFFFEF40001,
    0x3FFFFFFEAA0001,
    0x3FFFFFFE920001,
    0x3FFFFFFE800001,
    0x3FFFFFFE320001,
    0x3FFFFFFE080001,
    0x3FFFFFFDCC0001,
    0x3FFFFFFDC20001,
]
P = [0x7FFFFFFFFCC0001, 0x7FFFFFFFFBA0001]
QP = prod(Q) * prod(P)

# Two independent fully-entropic ternary shares have exact coefficient law
# [1,2,3,2,1]/9 on [-2,-1,0,1,2].  The estimator accepts only a
# variance/density abstraction, hence this is explicitly a moment model.
AGGREGATE_MOMENT = ND.NoiseDistribution(
    mean=0,
    stddev=sqrt(RR(4) / 3),
    bounds=(-2, 2),
    is_Gaussian_like=False,
    _density=RR(2) / 3,
)


def current_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


commit = current_commit()
if commit != EXPECTED_COMMIT:
    raise RuntimeError(f"estimator commit {commit}, want {EXPECTED_COMMIT}")
if SAGE_VERSION != "10.5":
    raise RuntimeError(f"SageMath {SAGE_VERSION}, want exactly 10.5")


PARAMETERS = {
    "hes_v2_128_ternary_boundary": LWEParameters(
        n=N,
        q=2**1747,
        Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(RR("3.19")),
        m=oo,
        tag="HES-v2-N65536-logq1747-ternary",
    ),
    "p2g_n16_qp_single_ternary": LWEParameters(
        n=N,
        q=QP,
        Xs=ND.Ternary,
        Xe=ND.DiscreteGaussian(SIGMA_E),
        m=oo,
        tag="P2G-N16-exact-QGuardP-single-ternary",
    ),
    "p2g_n16_qp_two_party_aggregate_moment": LWEParameters(
        n=N,
        q=QP,
        Xs=AGGREGATE_MOMENT,
        Xe=ND.DiscreteGaussian(SIGMA_E),
        m=oo,
        tag="P2G-N16-exact-QGuardP-two-party-aggregate-moment",
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
    finite = {
        name: record["log2_rop"]
        for name, record in result.items()
        if "log2_rop" in record
    }
    result["minimum_log2_rop"] = min(finite.values()) if finite else None
    result["minimum_attack"] = min(finite, key=finite.get) if finite else None
    return result


RF = RealField(256)
output = {
    "schema": "mkf-p2g-n16-qp-security-estimate-v1",
    "tool": {
        "lattice_estimator_commit": commit,
        "sage_version": SAGE_VERSION,
    },
    "literal": {
        "N": int(N),
        "q_count_including_guard": len(Q),
        "persistent_q_count": len(Q) - 1,
        "p_count": len(P),
        "log2_Q_guard_P_exact": float(RF(QP).log2()),
        "Q_guard_P_bit_length": int(QP.nbits()),
        "sigma_error": float(SIGMA_E),
    },
    "secret_models": {
        "single_party": "uniform ternary [1,1,1]/3 on [-1,0,1]",
        "two_party_exact": "[1,2,3,2,1]/9 on [-2,-1,0,1,2]",
        "two_party_estimator_abstraction": {
            "variance": float(RR(4) / 3),
            "stddev": float(sqrt(RR(4) / 3)),
            "nonzero_density": float(RR(2) / 3),
        },
    },
    "parameters": {},
}

for name, params in PARAMETERS.items():
    output["parameters"][name] = {
        "log2_q": float(RF(params.q).log2()),
        "matzov_classical": estimate_selected(params, MATZOV(nn="classical")),
        "matzov_quantum": estimate_selected(params, MATZOV(nn="quantum")),
    }

def json_default(value):
    """Convert residual Sage scalar types without losing attack records."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return float(value)


print("P2G_N16_SECURITY_ESTIMATE_JSON_BEGIN")
print(json.dumps(output, indent=int(2), sort_keys=True, default=json_default))
print("P2G_N16_SECURITY_ESTIMATE_JSON_END")
