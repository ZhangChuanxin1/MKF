package mkfcpu

import (
	"fmt"
	"math/cmplx"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// collectiveRefreshSigmaSmudging matches the Lattigo v2.3.0 dckks Refresh
// example. It is used only by the functional prototype until a P3
// parameterization and statistical-hiding analysis are frozen.
const collectiveRefreshSigmaSmudging = 3.2

// RefreshValidation records the numerical check for a collective-refresh
// functional test. It is intentionally separate from the multiplication
// benchmark validation: Refresh is interactive and its communication is not
// part of the existing CPU multiplication timing boundary.
type RefreshValidation struct {
	Passed   bool
	MaxError float64
	LevelIn  int
	LevelOut int
	LogBound int
}

// collectiveRefreshCase is a two-or-more-party functional harness for the
// Lattigo dckks masked decrypt--recrypt protocol. The collective public key is
// generated with CKG from the shares. No aggregate secret is retained by the
// protocol; tests may reconstruct one only after completion as a local
// decryption oracle.
//
// This harness validates the interactive aggregate-domain refresh primitive.
// It does not itself implement MKF's distributed fold-key generation.
type collectiveRefreshCase struct {
	params   ckks.Parameters
	parties  int
	shares   []*rlwe.SecretKey
	input    *ckks.Ciphertext
	expected []complex128
	minLevel int
	logBound int
	crs      utils.PRNG
	encoder  ckks.Encoder
}

func newCollectiveRefreshCase(parameter string, parties int) (*collectiveRefreshCase, error) {
	if parties < 2 {
		return nil, fmt.Errorf("collective refresh requires at least two parties")
	}

	params, err := CKKSParameters(parameter)
	if err != nil {
		return nil, err
	}

	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), parties, params.Q())
	if !ok {
		return nil, fmt.Errorf("%s lacks modulus levels for 128-bit collective refresh with %d parties", parameter, parties)
	}

	crs, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create collective-refresh CRS: %w", err)
	}

	kgen := ckks.NewKeyGenerator(params)
	ckg := dckks.NewCKGProtocol(params)
	ckgCRP := ckg.SampleCRP(crs)

	shares := make([]*rlwe.SecretKey, parties)
	aggregatedCKGShare := ckg.AllocateShares()
	for i := range shares {
		shares[i] = kgen.GenSecretKey()
		partyShare := ckg.AllocateShares()
		ckg.GenShare(shares[i], ckgCRP, partyShare)
		if i == 0 {
			aggregatedCKGShare = partyShare
		} else {
			ckg.AggregateShares(aggregatedCKGShare, partyShare, aggregatedCKGShare)
		}
	}
	collectivePK := rlwe.NewPublicKey(params.Parameters)
	ckg.GenPublicKey(aggregatedCKGShare, ckgCRP, collectivePK)

	encoder := ckks.NewEncoder(params)
	expected := []complex128{
		complex(0.125, -0.0625),
		complex(-0.25, 0.03125),
		complex(0.375, 0.0),
	}
	plaintext := encoder.EncodeNTTAtLvlNew(params.MaxLevel(), expected, params.LogSlots())
	input := ckks.NewEncryptor(params, collectivePK).EncryptNew(plaintext)
	ckks.NewEvaluator(params, rlwe.EvaluationKey{}).DropLevel(input, input.Level()-minLevel)

	return &collectiveRefreshCase{
		params: params, parties: parties, shares: shares, input: input,
		expected: expected, minLevel: minLevel, logBound: logBound,
		crs: crs, encoder: encoder,
	}, nil
}

func runCollectiveRefresh(params ckks.Parameters, shares []*rlwe.SecretKey, input *ckks.Ciphertext, logBound int, crs utils.PRNG) (*ckks.Ciphertext, error) {
	if len(shares) < 2 {
		return nil, fmt.Errorf("collective refresh requires at least two secret shares")
	}
	if input.Degree() != 1 {
		return nil, fmt.Errorf("collective refresh requires a compact degree-1 ciphertext, got degree %d", input.Degree())
	}

	levelIn, levelOut := input.Level(), params.MaxLevel()
	coordinator := dckks.NewRefreshProtocol(params, logBound, collectiveRefreshSigmaSmudging)
	refreshCRP := coordinator.SampleCRP(levelOut, crs)

	var aggregatedShare *dckks.RefreshShare
	for i, sk := range shares {
		party := dckks.NewRefreshProtocol(params, logBound, collectiveRefreshSigmaSmudging)
		share := party.AllocateShare(levelIn, levelOut)
		party.GenShares(sk, logBound, params.LogSlots(), input, refreshCRP, share)
		if i == 0 {
			aggregatedShare = share
		} else {
			coordinator.Aggregate(share, aggregatedShare, aggregatedShare)
		}
	}

	output := ckks.NewCiphertext(params, 1, levelOut, params.Scale())
	coordinator.Finalize(input, params.LogSlots(), refreshCRP, aggregatedShare, output)
	// dckks.S2EProtocol.GetEncryption copies the CRS polynomial into c1
	// without preserving its representation flag. The protocol generated c1
	// and c0 in the NTT representation (as required by CKS/S2E); restore the
	// metadata so that the CKKS evaluator can safely consume the refreshed
	// ciphertext. This is a representation annotation, not an extra NTT.
	output.Value[0].IsNTT, output.Value[1].IsNTT = true, true
	return output, nil
}

func (c *collectiveRefreshCase) refresh() (*ckks.Ciphertext, error) {
	return runCollectiveRefresh(c.params, c.shares, c.input, c.logBound, c.crs)
}

// validateWithTestOnlyAggregate is deliberately a test oracle. It reconstructs
// the aggregate key only to check the result in-process and must never be used
// as part of the claimed distributed protocol or evaluation material.
func (c *collectiveRefreshCase) validateWithTestOnlyAggregate(output *ckks.Ciphertext) RefreshValidation {
	aggregate := aggregateSecret(c.params.Parameters, c.shares)
	plaintext := ckks.NewDecryptor(c.params, aggregate).DecryptNew(output)
	decoded := c.encoder.Decode(plaintext, c.params.LogSlots())

	var maxError float64
	for i, want := range c.expected {
		if got := decoded[i]; cmplx.Abs(got-want) > maxError {
			maxError = cmplx.Abs(got - want)
		}
	}

	return RefreshValidation{
		Passed: maxError < 1e-5, MaxError: maxError,
		LevelIn: c.input.Level(), LevelOut: output.Level(), LogBound: c.logBound,
	}
}

// foldedRefreshBridge connects the existing MKF functional fold primitive to
// the collective-refresh harness. It is a composition/correctness test only:
// CKKSCase currently creates Fold switching keys with test-process access to
// the aggregate secret, so this bridge cannot yet be cited as a distributed
// MKF key-generation implementation.
type foldedRefreshBridge struct {
	base     *CKKSCase
	minLevel int
	logBound int
	crs      utils.PRNG
}

func newFoldedRefreshBridge(parameter string, parties int) (*foldedRefreshBridge, error) {
	base, err := NewCKKSCase(parameter, parties)
	if err != nil {
		return nil, err
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, base.params.Scale(), parties, base.params.Q())
	if !ok {
		return nil, fmt.Errorf("%s lacks modulus levels for 128-bit collective refresh with %d parties", parameter, parties)
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create folded-refresh CRS: %w", err)
	}
	return &foldedRefreshBridge{base: base, minLevel: minLevel, logBound: logBound, crs: crs}, nil
}

func (b *foldedRefreshBridge) foldThenRefresh() (*outputCiphertext, RefreshValidation, error) {
	profile := Profile{}
	folded, err := fold(b.base.params.Parameters, b.base.left, b.base.keys.Fold, true, b.base.switcher, &profile)
	if err != nil {
		return nil, RefreshValidation{}, err
	}

	compact := ckks.NewCiphertextAtLevelFromPoly(b.base.left.Level, folded)
	compact.Scale = b.base.left.Scale
	b.base.evaluator.DropLevel(compact, compact.Level()-b.minLevel)

	refreshed, err := runCollectiveRefresh(b.base.params, b.base.secret, compact, b.logBound, b.crs)
	if err != nil {
		return nil, RefreshValidation{}, err
	}

	out := materialize(b.base.params.Parameters, refreshed.Value[0], refreshed.Value[1], b.base.parties, refreshed.Level(), refreshed.Scale, &profile)
	verificationCT := ckks.NewCiphertextAtLevelFromPoly(out.Level, [2]*ring.Poly{out.C0, out.Gamma})
	verificationCT.Scale = out.Scale
	plaintext := ckks.NewDecryptor(b.base.params, b.base.aggregate).DecryptNew(verificationCT)
	decoded := b.base.encoder.Decode(plaintext, b.base.params.LogSlots())
	maxError := cmplx.Abs(decoded[0] - complex(b.base.leftExpected, 0))

	return out, RefreshValidation{
		Passed: maxError < 1e-5, MaxError: maxError,
		LevelIn: compact.Level(), LevelOut: refreshed.Level(), LogBound: b.logBound,
	}, nil
}
