package mkfcpu

import (
	"math"
	"math/cmplx"
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const (
	resnetReferenceReLUDegree = 59
	resnetStemOutputScale     = 0.90
)

// evaluateChebyshevSeries evaluates the coefficient convention returned by
// ckks.Approximate. The input x is first mapped from [a,b] to [-1,1].
func evaluateChebyshevSeries(pol *ckks.Polynomial, x complex128) complex128 {
	t := (2*x - pol.A - pol.B) / (pol.B - pol.A)
	if len(pol.Coeffs) == 0 {
		return 0
	}
	tPrev := complex(1, 0)
	result := pol.Coeffs[0]
	if len(pol.Coeffs) == 1 {
		return result
	}
	tCurrent := t
	result += pol.Coeffs[1] * tCurrent
	for degree := 2; degree < len(pol.Coeffs); degree++ {
		tNext := 2*t*tCurrent - tPrev
		result += pol.Coeffs[degree] * tNext
		tPrev, tCurrent = tCurrent, tNext
	}
	return result
}

func runReferenceReLUGate(t *testing.T, parameter string) {
	t.Helper()
	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, 2)
	public := make([]*rlwe.PublicKey, 2)
	for party := range shares {
		shares[party], public[party] = kgen.GenKeyPair()
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, nil, crs)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)
	encoder := ckks.NewEncoder(params)

	values := make([]complex128, params.Slots())
	for i := range values {
		// Avoid making every slot equal while keeping the complete fixture in
		// the reference approximation interval [-1,1].
		values[i] = complex(-0.95+1.9*float64(i%257)/256.0, 0)
	}
	multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := interactiveOwnerFold(params, multi, shares, 0)
	if err != nil {
		t.Fatal(err)
	}
	inputLevel := compact.Level()

	referenceFunction := func(x complex128) complex128 {
		if real(x) < 0 {
			return 0
		}
		return x / complex(resnetStemOutputScale, 0)
	}
	polynomial := ckks.Approximate(referenceFunction, -1, 1, resnetReferenceReLUDegree)
	if polynomial.Depth() != 6 {
		t.Fatalf("unexpected degree-%d Chebyshev depth %d", resnetReferenceReLUDegree, polynomial.Depth())
	}

	// Lattigo v2.3 EvaluatePoly expects the Chebyshev change of variable to
	// be applied by the caller. It costs one additional level here.
	evaluator.MultByConst(compact, 2/(polynomial.B-polynomial.A), compact)
	evaluator.AddConst(compact, (-polynomial.A-polynomial.B)/(polynomial.B-polynomial.A), compact)
	if err := evaluator.Rescale(compact, params.Scale(), compact); err != nil {
		t.Fatal(err)
	}
	result, err := evaluator.EvaluatePoly(compact, polynomial, params.Scale())
	if err != nil {
		t.Fatal(err)
	}

	aggregate := aggregateSecret(params.Parameters, shares)
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(result), params.LogSlots())
	var maxFHEError, maxApproxError, sumSquaredFHE float64
	for i, x := range values {
		plainPolynomial := evaluateChebyshevSeries(polynomial, x)
		fheError := cmplx.Abs(decoded[i] - plainPolynomial)
		approxError := cmplx.Abs(plainPolynomial - referenceFunction(x))
		maxFHEError = math.Max(maxFHEError, fheError)
		maxApproxError = math.Max(maxApproxError, approxError)
		sumSquaredFHE += fheError * fheError
	}
	rmseFHE := math.Sqrt(sumSquaredFHE / float64(len(values)))
	if maxFHEError >= 1e-3 {
		t.Fatalf("reference ReLU FHE gate failed: parameter=%s degree=%d levels=%d->%d maxFHEError=%g maxApproxError=%g rmseFHE=%g", parameter, resnetReferenceReLUDegree, inputLevel, result.Level(), maxFHEError, maxApproxError, rmseFHE)
	}
	t.Logf("reference ReLU %s: PASS; degree=%d scale=%0.2f levels=%d->%d consumed=%d polynomialDepth=%d maxFHEError=%g rmseFHE=%g maxApproxErrorVsReLU=%g", parameter, resnetReferenceReLUDegree, resnetStemOutputScale, inputLevel, result.Level(), inputLevel-result.Level(), polynomial.Depth(), maxFHEError, rmseFHE, maxApproxError)
}

func TestReferenceResNet20ReLUP2TwoParty(t *testing.T) {
	runReferenceReLUGate(t, "P2")
}

func TestReferenceResNet20ReLUP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_RELU") != "1" {
		t.Skip("set MKF_RUN_P3_RELU=1 to run the P3 degree-59 reference ReLU gate")
	}
	runReferenceReLUGate(t, "P3IRCandidate")
}
