package mkfcpu

import (
	"math"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// TestP2GN16LattigoLevelSemanticsNoCrypto freezes the level transitions used
// by the canonical P2GN16 ResNet-20 evaluator without generating keys or
// performing a cryptographic correctness/performance experiment. Degree-zero
// zero ciphertexts are sufficient because CKKS level and scale transitions
// depend only on the public operation schedule and metadata.
func TestP2GN16LattigoLevelSemanticsNoCrypto(t *testing.T) {
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	params := profile.base
	// A zero-valued structurally valid relinearization key is enough for this
	// metadata-only test. No decrypted value is asserted or published.
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlwe.NewRelinKey(params.Parameters, 1)})

	// referenceResNetBlockReLUPolynomial always has A=-1 and B=1. The
	// canonical map is therefore MultByConst(1) followed by AddConst(0).
	// Lattigo's following Rescale call is real but consumes no Q limb.
	polynomial := referenceResNetBlockReLUPolynomial(1)
	mapped := ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
	evaluator.MultByConst(mapped, 2/(polynomial.B-polynomial.A), mapped)
	evaluator.AddConst(mapped, (-polynomial.A-polynomial.B)/(polynomial.B-polynomial.A), mapped)
	beforeMapRescaleLevel := mapped.Level()
	beforeMapRescaleScale := mapped.Scale
	if err := evaluator.Rescale(mapped, params.Scale(), mapped); err != nil {
		t.Fatal(err)
	}
	if mapped.Level() != beforeMapRescaleLevel {
		t.Fatalf("canonical ReLU map Rescale dropped %d levels, want zero", beforeMapRescaleLevel-mapped.Level())
	}
	if relative := math.Abs(mapped.Scale-beforeMapRescaleScale) / beforeMapRescaleScale; relative > 1e-12 {
		t.Fatalf("canonical ReLU map scale changed by relative %g", relative)
	}

	activated, err := evaluator.EvaluatePoly(mapped, polynomial, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	if activated.Level() != 7 {
		t.Fatalf("degree-59 canonical activation output level=%d, want 7", activated.Level())
	}

	// The five public masks on a surviving Stage-2/3 packing path use a
	// common fifth-root scale whose product is the top three active Q primes.
	// Hence one logical Lattigo Rescale call consumes exactly three levels.
	packingInputLevel := params.MaxLevel() - 1
	q := params.Q()
	packingDivisor := float64(q[packingInputLevel]) * float64(q[packingInputLevel-1]) * float64(q[packingInputLevel-2])
	maskScale := math.Pow(packingDivisor, 1.0/5.0)
	packed := ckks.NewCiphertext(params, 0, packingInputLevel, params.Scale())
	for i := 0; i < 5; i++ {
		packed.Scale *= maskScale
	}
	beforePackingRescaleLevel := packed.Level()
	if err := evaluator.Rescale(packed, params.Scale(), packed); err != nil {
		t.Fatal(err)
	}
	if drops := beforePackingRescaleLevel - packed.Level(); drops != 3 {
		t.Fatalf("packing Rescale dropped %d levels, want three", drops)
	}
	if packed.Level() != 9 {
		t.Fatalf("packing output level=%d, want 9", packed.Level())
	}

	t.Logf("P2GN16_LATTIGO_LEVEL_SEMANTICS_PASS map_rescale_drops=0 activation_level=%d packing_rescale_drops=3 packing_level=%d performance_measured=false", activated.Level(), packed.Level())
}
