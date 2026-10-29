package mkfcpu

// This file is an isolated security-parameter/functionality gate.  It does
// not alter the published P1/P2 literals in params.go and it is not a timing
// benchmark.

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

const p2gN16GuardPrime uint64 = 0x3ffffffdc20001

var p2gN16SecurityCandidate = ckks.ParametersLiteral{
	LogN:     16,
	LogSlots: 14,
	Q: []uint64{
		0xffffffffffc0001,
		0x3fffffffd60001, 0x3fffffffca0001,
		0x3fffffff360001, 0x3fffffff2a0001,
		0x3fffffff000001, 0x3ffffffefa0001,
		0x3ffffffef40001, 0x3ffffffeaa0001,
		0x3ffffffe920001, 0x3ffffffe800001,
		0x3ffffffe320001, 0x3ffffffe080001,
		0x3ffffffdcc0001,
		p2gN16GuardPrime,
	},
	P: []uint64{
		0x7ffffffffcc0001, 0x7ffffffffba0001,
	},
	Scale: 1 << 54,
	Sigma: rlwe.DefaultSigma,
}

func p2gN16GuardedOwnerCarrier(params ckks.Parameters, encoder ckks.Encoder, public []*rlwe.PublicKey, foldKeys []*rlwe.SwitchingKey, owner int, values []complex128, switcher *switchWorkspace, profile *Profile) (*ckks.Ciphertext, error) {
	level := params.MaxLevel()
	guardedScale := params.Scale() * float64(p2gN16GuardPrime)
	plaintext := ckks.NewPlaintext(params, level, guardedScale)
	encoder.EncodeNTT(plaintext, values, params.LogSlots())
	fresh := ckks.NewEncryptor(params, public[owner]).EncryptNew(plaintext)
	multi := &multiCiphertext{
		C0:         fresh.Value[0].CopyNew(),
		Components: make([]*ring.Poly, len(public)),
		Level:      level,
		Scale:      fresh.Scale,
	}
	for party := range multi.Components {
		multi.Components[party] = params.RingQ().NewPolyLvl(level)
		multi.Components[party].Zero()
		multi.Components[party].IsNTT = true
	}
	multi.Components[owner].CopyValues(fresh.Value[1])
	multi.Components[owner].IsNTT = true
	multi.C0.IsNTT = true

	folded, err := fold(params.Parameters, multi, foldKeys, true, switcher, profile)
	if err != nil {
		return nil, err
	}
	compact := ckks.NewCiphertextAtLevelFromPoly(level, folded)
	compact.Scale = multi.Scale
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})
	if err := evaluator.Rescale(compact, params.Scale(), compact); err != nil {
		return nil, err
	}
	return compact, nil
}

func TestP2GN16SecurityCandidateGuardedFoldCompactMul(t *testing.T) {
	params, err := ckks.NewParametersFromLiteral(p2gN16SecurityCandidate)
	if err != nil {
		t.Fatal(err)
	}
	if params.LogN() != 16 || params.LogSlots() != 14 || params.MaxLevel() != 14 || len(params.P()) != 2 {
		t.Fatalf("candidate shape LogN/LogSlots/MaxLevel/P=%d/%d/%d/%d", params.LogN(), params.LogSlots(), params.MaxLevel(), len(params.P()))
	}
	if params.Alpha() != 2 || params.Beta() != 8 {
		t.Fatalf("candidate key-switch decomposition alpha/beta=%d/%d, want 2/8", params.Alpha(), params.Beta())
	}
	for index, modulus := range append(append([]uint64(nil), params.Q()...), params.P()...) {
		if !ring.IsPrime(modulus) || (modulus-1)%uint64(2*params.N()) != 0 {
			t.Fatalf("modulus[%d]=0x%x is not prime or 2N-compatible", index, modulus)
		}
	}
	if params.Q()[params.MaxLevel()] != p2gN16GuardPrime {
		t.Fatalf("guard mismatch: got 0x%x", params.Q()[params.MaxLevel()])
	}

	keyGenerator := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, 2)
	public := make([]*rlwe.PublicKey, 2)
	for party := range shares {
		shares[party], public[party] = keyGenerator.GenKeyPair()
	}
	aggregate := aggregateSecret(params.Parameters, shares)
	foldKeys := make([]*rlwe.SwitchingKey, len(shares))
	for party := range shares {
		foldKeys[party] = keyGenerator.GenSwitchingKey(shares[party], aggregate)
	}
	relinearizationKey := keyGenerator.GenRelinearizationKey(aggregate, 1)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: relinearizationKey})
	encoder := ckks.NewEncoder(params)
	switcher := newSwitchWorkspace(params.Parameters)
	profile := Profile{}

	leftValues := []complex128{0.125, -0.25, 0.375, -0.5}
	rightValues := []complex128{0.5, 0.25, -0.125, -0.5}
	left, err := p2gN16GuardedOwnerCarrier(params, encoder, public, foldKeys, 0, leftValues, switcher, &profile)
	if err != nil {
		t.Fatal(err)
	}
	right, err := p2gN16GuardedOwnerCarrier(params, encoder, public, foldKeys, 1, rightValues, switcher, &profile)
	if err != nil {
		t.Fatal(err)
	}
	if left.Degree() != 1 || right.Degree() != 1 || left.Level() != 13 || right.Level() != 13 {
		t.Fatalf("guard consumption/compact degree mismatch: left degree/level=%d/%d right=%d/%d", left.Degree(), left.Level(), right.Degree(), right.Level())
	}
	product := evaluator.MulRelinNew(left, right)
	if err := evaluator.Rescale(product, params.Scale(), product); err != nil {
		t.Fatal(err)
	}
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(product), params.LogSlots())
	var maxError float64
	for index := range leftValues {
		want := leftValues[index] * rightValues[index]
		maxError = math.Max(maxError, cmplx.Abs(decoded[index]-want))
	}
	if maxError >= 1e-5 {
		t.Fatalf("guarded fold+compact MulRelin max error=%g", maxError)
	}
	if product.Degree() != 1 || product.Level() != 12 {
		t.Fatalf("compact MulRelin output degree/level=%d/%d, want 1/12", product.Degree(), product.Level())
	}
	polyQBytes := uint64(params.N() * len(params.Q()) * 8)
	polyPBytes := uint64(params.N() * len(params.P()) * 8)
	switchingKeyPayloadBytes := uint64(params.Beta()) * 2 * (polyQBytes + polyPBytes)
	t.Logf("P2G_N16_GUARDED_FOLD_COMPACT_MUL_PASS max_error=%.17g input_level=13 output_level=12 logN=%d logSlots=%d q_count=%d p_count=%d alpha=%d beta=%d raw_switching_key_payload_bytes=%d", maxError, params.LogN(), params.LogSlots(), len(params.Q()), len(params.P()), params.Alpha(), params.Beta(), switchingKeyPayloadBytes)
}
