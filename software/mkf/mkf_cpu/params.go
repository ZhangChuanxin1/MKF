package mkfcpu

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/bfv"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// The literals below are the exact P1/P2 modulus chains used by the public
// KKLSS implementation. Keeping them here makes the MKF and KKLSS CPU runs
// comparable at the cryptographic-parameter level.

var ckksP1 = ckks.ParametersLiteral{
	LogN:     14,
	LogSlots: 13,
	Q: []uint64{
		0x7ffffffffe70001,
		0xffffffff00001, 0xfffffffe40001, 0xfffffffe20001,
		0xfffffffbe0001, 0xfffffffa60001,
	},
	P: []uint64{
		0xffffffffffc0001, 0xfffffffff840001,
	},
	Scale: 1 << 52,
	Sigma: rlwe.DefaultSigma,
}

var ckksP2 = ckks.ParametersLiteral{
	LogN:     15,
	LogSlots: 14,
	Q: []uint64{
		0xfffffffff6a0001,
		0x3fffffffd60001, 0x3fffffffca0001,
		0x3fffffff6d0001, 0x3fffffff5d0001,
		0x3fffffff550001, 0x3fffffff390001,
		0x3fffffff360001, 0x3fffffff2a0001,
		0x3fffffff000001, 0x3ffffffefa0001,
		0x3ffffffef40001, 0x3ffffffed70001,
		0x3ffffffed30001,
	},
	P: []uint64{
		0x7ffffffffe70001, 0x7ffffffffe10001,
	},
	Scale: 1 << 54,
	Sigma: rlwe.DefaultSigma,
}

// ckksGuardPrime is a 54-bit NTT prime compatible with LogN <= 15. The
// guarded experiment appends it as the top Q level solely for the initial
// FoldToAggregate conversion, then removes it before circuit evaluation.
const ckksGuardPrime uint64 = 0x3ffffffeaa0001

var bfvP1 = bfv.ParametersLiteral{
	LogN: 14,
	Q: []uint64{
		0x1fffffffe30001, 0x1fffffffd10001,
		0x1fffffffbf0001, 0x1fffffffb60001,
		0x1fffffff920001, 0x3fffffffd60001,
	},
	P: []uint64{
		0xffffffffffc0001, 0xfffffffff840001,
	},
	T:     65537,
	Sigma: rlwe.DefaultSigma,
}

var bfvP1QMul = []uint64{
	0x1fffffffd80001, 0x1fffffffc50001,
	0x1fffffffb90001, 0x1fffffffa50001,
	0x1fffffff900001, 0x3fffffffca0001,
}

var bfvP2 = bfv.ParametersLiteral{
	LogN: 15,
	Q: []uint64{
		0x3fffffffd60001, 0x3fffffff6d0001,
		0x3fffffff550001, 0x3fffffff360001,
		0x3fffffff000001, 0x3ffffffef40001,
		0x3ffffffed30001, 0x3ffffffe970001,
		0x3ffffffe800001, 0x3ffffffe410001,
		0x7fffffffe90001, 0x7fffffffbd0001,
		0x7fffffffaa0001, 0x7fffffff9f0001,
	},
	P: []uint64{
		0xffffffffffc0001, 0xfffffffff840001,
	},
	T:     65537,
	Sigma: rlwe.DefaultSigma,
}

var bfvP2QMul = []uint64{
	0x3fffffffca0001, 0x3fffffff5d0001,
	0x3fffffff390001, 0x3fffffff2a0001,
	0x3ffffffefa0001, 0x3ffffffed70001,
	0x3ffffffeaa0001, 0x3ffffffe920001,
	0x3ffffffe790001, 0x3ffffffe320001,
	0x7fffffffbf0001, 0x7fffffffba0001,
	0x7fffffffa50001, 0x7fffffff7e0001,
}

func CKKSParameters(name string) (ckks.Parameters, error) {
	var lit ckks.ParametersLiteral
	switch name {
	case "P1":
		lit = ckksP1
	case "P2":
		lit = ckksP2
	case "P1G":
		lit = ckksP1
		lit.Q = append(append([]uint64(nil), lit.Q...), ckksGuardPrime)
	case "P2G":
		lit = ckksP2
		lit.Q = append(append([]uint64(nil), lit.Q...), ckksGuardPrime)
	case "P3IRCandidate":
		// This is Lattigo v2.3.0's classical single-secret PN16QP1761
		// literal, reused only as a deep-chain *functional candidate* for
		// the interactive-refresh prototype. It is deliberately not called
		// "P3" because its concrete RLWE security and precision under the
		// aggregate-secret distribution have not yet been independently
		// validated for this MKF construction.
		lit = ckks.PN16QP1761
	default:
		return ckks.Parameters{}, fmt.Errorf("unknown CKKS parameter set %q", name)
	}
	return ckks.NewParametersFromLiteral(lit)
}

func BFVParameters(name string) (bfv.Parameters, []uint64, error) {
	var lit bfv.ParametersLiteral
	var qMul []uint64
	switch name {
	case "P1":
		lit, qMul = bfvP1, bfvP1QMul
	case "P2":
		lit, qMul = bfvP2, bfvP2QMul
	default:
		return bfv.Parameters{}, nil, fmt.Errorf("unknown BFV parameter set %q", name)
	}
	params, err := bfv.NewParametersFromLiteral(lit)
	return params, append([]uint64(nil), qMul...), err
}
