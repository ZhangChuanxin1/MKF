package mkfcpu

// This file mirrors Lattigo v2.3.0 ckks/polynomial_evaluation.go on the
// public-KKLSS degree-two carrier. Public linear operations remain on the
// carrier, while every ciphertext product is adapter.MulRelinNew and hence a
// real mkckks.MulRelinNew. Public KKLSS MulRelinNew already performs its CKKS
// rescale, so the two explicit post-product Rescale calls in upstream
// Lattigo are replaced by level alignment after that fused rescale.

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
)

const p2gn16Degree59ProductsPerActivation = 17

// kklssCarrierBSGSProductCount symbolically walks the same power-basis and
// recursive BSGS DAG as kklssCarrierEvaluatePolyBSGS. It is independent of
// ciphertext execution and lets the formal contract fail before allocating
// keys if the frozen coefficient vector or split schedule changes.
func kklssCarrierBSGSProductCount(polynomialInput *ckks.Polynomial) (int, error) {
	if polynomialInput == nil || len(polynomialInput.Coeffs) == 0 {
		return 0, fmt.Errorf("KKLSS BSGS polynomial is empty")
	}
	polynomial := paperMKFPolynomialCopy(polynomialInput)
	logDegree := bits.Len64(uint64(polynomial.Degree()))
	logSplit := logDegree >> 1
	odd, even := paperMKFOddOrEven(polynomial.Coeffs)
	seen := map[int]bool{1: true}
	var powerCount func(int) (int, error)
	powerCount = func(degree int) (int, error) {
		if degree < 1 {
			return 0, fmt.Errorf("invalid KKLSS BSGS power %d", degree)
		}
		if seen[degree] {
			return 0, nil
		}
		var leftDegree, rightDegree, subtractionDegree int
		if degree&(degree-1) == 0 {
			leftDegree, rightDegree = degree/2, degree/2
		} else {
			k := int(math.Ceil(math.Log2(float64(degree)))) - 1
			leftDegree = (1 << k) - 1
			rightDegree = degree + 1 - (1 << k)
			if polynomial.Basis == ckks.ChebyshevBasis {
				subtractionDegree = int(math.Abs(float64(leftDegree - rightDegree)))
			}
		}
		left, err := powerCount(leftDegree)
		if err != nil {
			return 0, err
		}
		right, err := powerCount(rightDegree)
		if err != nil {
			return 0, err
		}
		correction := 0
		if subtractionDegree != 0 {
			correction, err = powerCount(subtractionDegree)
			if err != nil {
				return 0, err
			}
		}
		seen[degree] = true
		return left + right + correction + 1, nil
	}
	total := 0
	for degree := 2; degree < 1<<logSplit; degree++ {
		if !(even || odd) || (degree&1 == 0 && even) || (degree&1 == 1 && odd) {
			count, err := powerCount(degree)
			if err != nil {
				return 0, err
			}
			total += count
		}
	}
	for index := logSplit; index < logDegree; index++ {
		count, err := powerCount(1 << index)
		if err != nil {
			return 0, err
		}
		total += count
	}
	var recurseCount func(int, int, *ckks.Polynomial) (int, error)
	recurseCount = func(currentSplit, currentDegree int, coefficients *ckks.Polynomial) (int, error) {
		if coefficients.Degree() < 1<<currentSplit {
			if coefficients.Lead && currentSplit > 1 && coefficients.MaxDeg%(1<<(currentSplit+1)) > 1<<(currentSplit-1) {
				currentDegree = bits.Len64(uint64(coefficients.Degree()))
				return recurseCount(currentDegree>>1, currentDegree, coefficients)
			}
			return 0, nil
		}
		nextPower := 1 << currentSplit
		for nextPower < (coefficients.Degree()>>1)+1 {
			nextPower <<= 1
		}
		quotient, remainder := paperMKFSplitCoefficients(coefficients, nextPower)
		left, err := recurseCount(currentSplit, currentDegree, quotient)
		if err != nil {
			return 0, err
		}
		right, err := recurseCount(currentSplit, currentDegree, remainder)
		if err != nil {
			return 0, err
		}
		return left + right + 1, nil
	}
	recursive, err := recurseCount(logSplit, logDegree, polynomial)
	if err != nil {
		return 0, err
	}
	return total + recursive, nil
}

func kklssCarrierEvaluatePolyBSGS(adapter *kklssLattigoAdapter, input *ckks.Ciphertext, polynomialInput *ckks.Polynomial, targetScale float64) (*ckks.Ciphertext, int, error) {
	if adapter == nil || input == nil || polynomialInput == nil || len(polynomialInput.Coeffs) == 0 {
		return nil, 0, fmt.Errorf("KKLSS BSGS polynomial input is incomplete")
	}
	if input.Level() < polynomialInput.Depth() {
		return nil, 0, fmt.Errorf("KKLSS BSGS has %d levels < polynomial depth %d", input.Level(), polynomialInput.Depth())
	}
	polynomial := paperMKFPolynomialCopy(polynomialInput)
	powers := map[int]*ckks.Ciphertext{1: input.CopyNew()}
	logDegree := bits.Len64(uint64(polynomial.Degree()))
	logSplit := logDegree >> 1
	odd, even := paperMKFOddOrEven(polynomial.Coeffs)
	products := 0
	for degree := 2; degree < 1<<logSplit; degree++ {
		if !(even || odd) || (degree&1 == 0 && even) || (degree&1 == 1 && odd) {
			if err := kklssCarrierComputePowerBSGS(adapter, degree, powers, targetScale, polynomial.Basis, &products); err != nil {
				return nil, 0, err
			}
		}
	}
	for index := logSplit; index < logDegree; index++ {
		if err := kklssCarrierComputePowerBSGS(adapter, 1<<index, powers, targetScale, polynomial.Basis, &products); err != nil {
			return nil, 0, err
		}
	}
	result, err := kklssCarrierRecurseBSGS(adapter, targetScale, logSplit, logDegree, polynomial, powers, &products)
	if err != nil {
		return nil, 0, err
	}
	result.Scale = targetScale
	return result, products, nil
}

func kklssCarrierComputePowerBSGS(adapter *kklssLattigoAdapter, degree int, powers map[int]*ckks.Ciphertext, scale float64, basis ckks.PolynomialBasis, products *int) error {
	if powers[degree] != nil {
		return nil
	}
	var leftDegree, rightDegree, subtractionDegree int
	if degree&(degree-1) == 0 {
		leftDegree, rightDegree = degree/2, degree/2
	} else {
		k := int(math.Ceil(math.Log2(float64(degree)))) - 1
		leftDegree = (1 << k) - 1
		rightDegree = degree + 1 - (1 << k)
		if basis == ckks.ChebyshevBasis {
			subtractionDegree = int(math.Abs(float64(leftDegree - rightDegree)))
		}
	}
	if err := kklssCarrierComputePowerBSGS(adapter, leftDegree, powers, scale, basis, products); err != nil {
		return err
	}
	if err := kklssCarrierComputePowerBSGS(adapter, rightDegree, powers, scale, basis, products); err != nil {
		return err
	}
	result := adapter.MulRelinNew(powers[leftDegree], powers[rightDegree])
	*products = *products + 1
	// mkckks.MulRelinNew already rescaled to params.Scale. The frozen
	// network targetScale equals that scale; check rather than rescale twice.
	if relative := math.Abs(result.Scale-scale) / scale; relative > 0.02 {
		return fmt.Errorf("KKLSS fused product scale=%g, target=%g", result.Scale, scale)
	}
	if basis == ckks.ChebyshevBasis {
		adapter.Evaluator.Add(result, result, result)
		if subtractionDegree == 0 {
			adapter.Evaluator.AddConst(result, -1, result)
		} else {
			if err := kklssCarrierComputePowerBSGS(adapter, subtractionDegree, powers, scale, basis, products); err != nil {
				return err
			}
			adapter.Evaluator.Sub(result, powers[subtractionDegree], result)
		}
	}
	powers[degree] = result
	return nil
}

func kklssCarrierRecurseBSGS(adapter *kklssLattigoAdapter, targetScale float64, logSplit, logDegree int, coefficients *ckks.Polynomial, powers map[int]*ckks.Ciphertext, products *int) (*ckks.Ciphertext, error) {
	if coefficients.Degree() < 1<<logSplit {
		if coefficients.Lead && logSplit > 1 && coefficients.MaxDeg%(1<<(logSplit+1)) > 1<<(logSplit-1) {
			logDegree = bits.Len64(uint64(coefficients.Degree()))
			return kklssCarrierRecurseBSGS(adapter, targetScale, logDegree>>1, logDegree, coefficients, powers, products)
		}
		return kklssCarrierEvaluateFromPowerBSGS(adapter, targetScale, coefficients, powers)
	}
	nextPower := 1 << logSplit
	for nextPower < (coefficients.Degree()>>1)+1 {
		nextPower <<= 1
	}
	quotient, remainder := paperMKFSplitCoefficients(coefficients, nextPower)
	level := powers[nextPower].Level() - 1
	if quotient.MaxDeg >= 1<<(logDegree-1) && quotient.Lead {
		level++
	}
	currentQi := adapter.params.QiFloat64(level)
	left, err := kklssCarrierRecurseBSGS(adapter, targetScale*currentQi/powers[nextPower].Scale, logSplit, logDegree, quotient, powers, products)
	if err != nil {
		return nil, err
	}
	right, err := kklssCarrierRecurseBSGS(adapter, targetScale, logSplit, logDegree, remainder, powers, products)
	if err != nil {
		return nil, err
	}
	if left.Level() > right.Level() {
		for left.Level() != right.Level()+1 {
			adapter.Evaluator.DropLevel(left, 1)
		}
	}
	left = adapter.MulRelinNew(left, powers[nextPower])
	*products = *products + 1
	// The fused KKLSS rescale has already consumed the level that upstream
	// Lattigo consumes immediately after this multiplication.
	if right.Level() > left.Level() {
		adapter.Evaluator.DropLevel(right, right.Level()-left.Level())
	}
	if left.Level() > right.Level() {
		adapter.Evaluator.DropLevel(left, left.Level()-right.Level())
	}
	adapter.Evaluator.Add(left, right, left)
	left.Scale = targetScale
	return left, nil
}

func kklssCarrierEvaluateFromPowerBSGS(adapter *kklssLattigoAdapter, targetScale float64, coefficients *ckks.Polynomial, powers map[int]*ckks.Ciphertext) (*ckks.Ciphertext, error) {
	maximumNonzero := 0
	for degree := coefficients.Degree(); degree > 0; degree-- {
		if paperMKFIsNotNegligible(coefficients.Coeffs[degree]) {
			maximumNonzero = degree
			break
		}
	}
	constant := coefficients.Coeffs[0]
	if maximumNonzero == 0 {
		result := ckks.NewCiphertext(adapter.params, len(adapter.context.partyIDs), powers[1].Level(), targetScale)
		if paperMKFIsNotNegligible(constant) {
			adapter.Evaluator.AddConst(result, constant, result)
		}
		return result, nil
	}
	maximumNonzero = coefficients.Degree()
	currentQi := adapter.params.QiFloat64(powers[maximumNonzero].Level())
	result := ckks.NewCiphertext(adapter.params, len(adapter.context.partyIDs), powers[maximumNonzero].Level(), targetScale*currentQi)
	if paperMKFIsNotNegligible(constant) {
		adapter.Evaluator.AddConst(result, constant, result)
	}
	realFloat, imaginaryFloat, constantScale := ring.NewFloat(0, 128), ring.NewFloat(0, 128), ring.NewFloat(0, 128)
	realInteger, imaginaryInteger := ring.NewUint(0), ring.NewUint(0)
	for degree := coefficients.Degree(); degree > 0; degree-- {
		coefficient := coefficients.Coeffs[degree]
		if !paperMKFIsNotNegligible(coefficient) {
			continue
		}
		realFloat.SetFloat64(real(coefficient))
		imaginaryFloat.SetFloat64(imag(coefficient))
		constantScale.SetFloat64(targetScale * currentQi / powers[degree].Scale)
		realFloat.Mul(realFloat, constantScale).Int(realInteger)
		imaginaryFloat.Mul(imaginaryFloat, constantScale).Int(imaginaryInteger)
		adapter.Evaluator.MultByGaussianIntegerAndAdd(powers[degree], realInteger, imaginaryInteger, result)
	}
	if err := adapter.Evaluator.Rescale(result, targetScale, result); err != nil {
		return nil, err
	}
	return result, nil
}

func TestKKLSSP2GN16Degree59BSGSMicroGate(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_P2GN16_BSGS_MICRO") != "1" {
		t.Skip("set MKF_RUN_KKLSS_P2GN16_BSGS_MICRO=1 for the real two-owner P2GN16 BSGS numerical gate")
	}
	context, err := newKKLSSN2Context(guardedCompactP2GN16ParameterID, nil)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := newKKLSSLattigoAdapter(profile.base, context, true)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]complex128, profile.base.Slots())
	for index := range values {
		values[index] = complex(-0.75+1.5*float64(index%257)/256, 0)
	}
	input, err := adapter.ownerCarrier(nil, nil, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	polynomial := referenceResNetBlockReLUPolynomial(1.0)
	beforeLevel := input.Level()
	output, products, err := kklssCarrierEvaluatePolyBSGS(adapter, input, polynomial, profile.base.Scale())
	if err != nil {
		t.Fatal(err)
	}
	if products != p2gn16Degree59ProductsPerActivation {
		t.Fatalf("KKLSS BSGS products=%d, want %d", products, p2gn16Degree59ProductsPerActivation)
	}
	if output.Level() != beforeLevel-polynomial.Depth() {
		t.Fatalf("KKLSS BSGS output level=%d, want %d", output.Level(), beforeLevel-polynomial.Depth())
	}
	if relative := math.Abs(output.Scale-profile.base.Scale()) / profile.base.Scale(); relative > 1e-12 {
		t.Fatalf("KKLSS BSGS output scale=%g", output.Scale)
	}
	state, err := adapter.state(output)
	if err != nil {
		t.Fatal(err)
	}
	decoded := context.decryptor.Decrypt(state, context.secret)
	maximum := 0.0
	for index, value := range decoded.Value {
		want := evaluateChebyshevSeries(polynomial, values[index])
		delta := math.Hypot(real(value-want), imag(value-want))
		if delta > maximum {
			maximum = delta
		}
	}
	if maximum >= fullResNet20FunctionalGate {
		t.Fatalf("KKLSS BSGS max error=%g", maximum)
	}
	t.Logf("KKLSS_P2GN16_BSGS_MICRO_PASS products=%d input_level=%d output_level=%d scale=%g max_error=%g", products, beforeLevel, output.Level(), output.Scale, maximum)
}
