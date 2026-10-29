package mkfcpu

import (
	"math"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// TestDistributedPublicFoldKeyGenP2GN16OneCarrier is the actual LogN=16
// distributed FoldKeyGen micro gate. It is intentionally separate from the
// P2G/LogN=15 full functional run and from the centralized P2GN16 candidate
// gate. It evaluates one fresh owner carrier through exactly one public fold
// and one guard-prime rescale; it is functional/setup evidence, not timing.
func TestDistributedPublicFoldKeyGenP2GN16OneCarrier(t *testing.T) {
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	params := profile.guarded
	if params.LogN() != 16 || params.LogSlots() != 14 || params.Alpha() != 2 || params.Beta() != 8 || params.MaxLevel() != 14 || profile.base.MaxLevel() != 13 {
		t.Fatalf("P2GN16 parameter drift: logN/logSlots=%d/%d alpha/beta=%d/%d levels=%d/%d", params.LogN(), params.LogSlots(), params.Alpha(), params.Beta(), params.MaxLevel(), profile.base.MaxLevel())
	}

	keyGenerator := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, fullResNet20Parties)
	public := make([]*rlwe.PublicKey, fullResNet20Parties)
	for party := range shares {
		shares[party], public[party] = keyGenerator.GenKeyPair()
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	material, err := generateDistributedPublicFoldKeys(params, shares, crs)
	if err != nil {
		t.Fatal(err)
	}
	if material.Transcript.Parties != 2 || material.Transcript.Digits != 8 || material.Transcript.CKGRounds != 1 || material.Transcript.CKGShareMessages != 2 || material.Transcript.PublishedFoldKeyMessages != 2 {
		t.Fatalf("P2GN16 distributed transcript drift: %+v", material.Transcript)
	}

	// Test-only oracle reconstruction occurs after distributed key generation
	// and is never passed to that protocol or the online fold evaluator.
	aggregate := aggregateSecret(params.Parameters, shares)
	for party := range shares {
		maximum := switchingKeyRelationMaxCentered(params.Parameters, material.FoldKeys[party], shares[party], aggregate)
		if maximum == 0 || maximum > 1<<33 {
			t.Fatalf("P2GN16 party %d relation noise is implausible: %d", party, maximum)
		}
	}

	adapter, err := newGuardedCompactMKFLattigoAdapter(params, rlwe.EvaluationKey{}, material.FoldKeys, len(shares), guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	want := []complex128{0.125, -0.25, 0.375, -0.0625}
	ciphertext, err := adapter.ownerCarrier(encoder, public, 1, want)
	if err != nil {
		t.Fatal(err)
	}
	if ciphertext.Degree() != 1 || ciphertext.Level() != 13 {
		t.Fatalf("P2GN16 compact carrier degree/level=%d/%d, want 1/13", ciphertext.Degree(), ciphertext.Level())
	}
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(ciphertext), params.LogSlots())
	maximumError := 0.0
	for index := range want {
		maximumError = math.Max(maximumError, math.Abs(real(decoded[index]-want[index])))
	}
	if maximumError > 1e-6 {
		t.Fatalf("P2GN16 distributed one-carrier error=%g", maximumError)
	}
	t.Logf("P2GN16_DISTRIBUTED_PUBLIC_FOLD_PASS parameter=%s logN=%d parties=%d level=%d max_error=%.17g transcript=%+v", guardedCompactP2GN16ParameterID, params.LogN(), len(shares), ciphertext.Level(), maximumError, material.Transcript)
}
