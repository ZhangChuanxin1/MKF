package mkfcpu

// The Lattigo v2.3 polynomial evaluator has a concrete internal evaluator and
// cannot accept a replacement multiplication callback.  Calling it would
// silently route every internal product through ordinary single-secret
// MulRelin.  This file mirrors its balanced power-basis/BSGS schedule while
// routing every ciphertext product through paperMKFCKKSEvaluator.MulRelinNew.

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
)

func paperMKFPolynomialCopy(input *ckks.Polynomial) *ckks.Polynomial {
	result := &ckks.Polynomial{
		MaxDeg: input.MaxDeg,
		Lead:   input.Lead,
		A:      input.A,
		B:      input.B,
		Basis:  input.Basis,
		Coeffs: append([]complex128(nil), input.Coeffs...),
	}
	return result
}

func paperMKFCheckPolynomialLevels(levels int, polynomial *ckks.Polynomial) error {
	depth := polynomial.Depth()
	if levels < depth {
		return fmt.Errorf("%d levels < %d polynomial depth", levels, depth)
	}
	return nil
}

func paperMKFIsNotNegligible(coefficient complex128) bool {
	return math.Abs(real(coefficient)) > ckks.IsNegligbleThreshold || math.Abs(imag(coefficient)) > ckks.IsNegligbleThreshold
}

func paperMKFOddOrEven(coefficients []complex128) (odd, even bool) {
	odd, even = true, true
	for index, coefficient := range coefficients {
		nonzero := paperMKFIsNotNegligible(coefficient)
		odd = odd && !(index&1 == 0 && nonzero)
		even = even && !(index&1 == 1 && nonzero)
		if !odd && !even {
			break
		}
	}
	if odd || even {
		start := 0
		if even {
			start = 1
		}
		for index := start; index < len(coefficients); index += 2 {
			coefficients[index] = 0
		}
	}
	return odd, even
}

func (evaluator *paperMKFCKKSEvaluator) multByGaussianIntegerAndAdd(input *paperMKFCKKSCiphertext, realCoefficient, imaginaryCoefficient interface{}, target *paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	if err := evaluator.validateState(target); err != nil {
		return nil, err
	}
	inputLinear, targetLinear := input.asLinearCiphertext(evaluator.params), target.asLinearCiphertext(evaluator.params)
	evaluator.linear.MultByGaussianIntegerAndAdd(inputLinear, realCoefficient, imaginaryCoefficient, targetLinear)
	return paperMKFStateFromLinear(evaluator.params, targetLinear, evaluator.parties)
}

// EvaluatePoly evaluates an already basis-mapped input.  Its contract matches
// ckks.Evaluator.EvaluatePoly: callers of a Chebyshev polynomial must first
// apply x <- (2*x-a-b)/(b-a), including the associated rescale.
func (evaluator *paperMKFCKKSEvaluator) EvaluatePoly(input *paperMKFCKKSCiphertext, polynomialInput *ckks.Polynomial, targetScale float64) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	if polynomialInput == nil || len(polynomialInput.Coeffs) == 0 {
		return nil, fmt.Errorf("polynomial is empty")
	}
	polynomial := paperMKFPolynomialCopy(polynomialInput)
	if err := paperMKFCheckPolynomialLevels(input.Level, polynomial); err != nil {
		return nil, err
	}
	powers := map[int]*paperMKFCKKSCiphertext{1: input.copyNew()}
	logDegree := bits.Len64(uint64(polynomial.Degree()))
	logSplit := logDegree >> 1
	odd, even := paperMKFOddOrEven(polynomial.Coeffs)
	for degree := 2; degree < 1<<logSplit; degree++ {
		if !(even || odd) || (degree&1 == 0 && even) || (degree&1 == 1 && odd) {
			if err := evaluator.computePowerBasis(degree, powers, targetScale, polynomial.Basis); err != nil {
				return nil, err
			}
		}
	}
	for index := logSplit; index < logDegree; index++ {
		if err := evaluator.computePowerBasis(1<<index, powers, targetScale, polynomial.Basis); err != nil {
			return nil, err
		}
	}
	result, err := evaluator.recursePolynomial(targetScale, logSplit, logDegree, polynomial, powers)
	if err != nil {
		return nil, err
	}
	result.Scale = targetScale
	evaluator.counts.PolynomialCalls++
	return result, nil
}

func (evaluator *paperMKFCKKSEvaluator) computePowerBasis(degree int, powers map[int]*paperMKFCKKSCiphertext, scale float64, basis ckks.PolynomialBasis) error {
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
			subtractionDegree = int(math.Abs(float64(leftDegree) - float64(rightDegree)))
		}
	}
	if err := evaluator.computePowerBasis(leftDegree, powers, scale, basis); err != nil {
		return err
	}
	if err := evaluator.computePowerBasis(rightDegree, powers, scale, basis); err != nil {
		return err
	}
	product, err := evaluator.MulRelinNew(powers[leftDegree], powers[rightDegree])
	if err != nil {
		return err
	}
	if powers[degree], err = evaluator.RescaleNew(product, scale); err != nil {
		return err
	}
	if basis == ckks.ChebyshevBasis {
		if powers[degree], err = evaluator.AddNew(powers[degree], powers[degree]); err != nil {
			return err
		}
		if subtractionDegree == 0 {
			powers[degree], err = evaluator.AddConstNew(powers[degree], -1)
		} else {
			if err = evaluator.computePowerBasis(subtractionDegree, powers, scale, basis); err != nil {
				return err
			}
			powers[degree], err = evaluator.SubNew(powers[degree], powers[subtractionDegree])
		}
	}
	return err
}

func paperMKFSplitCoefficients(coefficients *ckks.Polynomial, split int) (quotient, remainder *ckks.Polynomial) {
	remainder = &ckks.Polynomial{Coeffs: make([]complex128, split), Basis: coefficients.Basis}
	if coefficients.MaxDeg == coefficients.Degree() {
		remainder.MaxDeg = split - 1
	} else {
		remainder.MaxDeg = coefficients.MaxDeg - (coefficients.Degree() - split + 1)
	}
	copy(remainder.Coeffs, coefficients.Coeffs[:split])
	quotient = &ckks.Polynomial{Coeffs: make([]complex128, coefficients.Degree()-split+1), MaxDeg: coefficients.MaxDeg, Basis: coefficients.Basis}
	quotient.Coeffs[0] = coefficients.Coeffs[split]
	if coefficients.Basis == ckks.StandardBasis {
		copy(quotient.Coeffs[1:], coefficients.Coeffs[split+1:])
	} else {
		for index, offset := split+1, 1; index < coefficients.Degree()+1; index, offset = index+1, offset+1 {
			quotient.Coeffs[index-split] = 2 * coefficients.Coeffs[index]
			remainder.Coeffs[split-offset] -= coefficients.Coeffs[index]
		}
	}
	quotient.Lead = coefficients.Lead
	return quotient, remainder
}

func (evaluator *paperMKFCKKSEvaluator) recursePolynomial(targetScale float64, logSplit, logDegree int, coefficients *ckks.Polynomial, powers map[int]*paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	if coefficients.Degree() < 1<<logSplit {
		if coefficients.Lead && logSplit > 1 && coefficients.MaxDeg%(1<<(logSplit+1)) > 1<<(logSplit-1) {
			logDegree = bits.Len64(uint64(coefficients.Degree()))
			logSplit = logDegree >> 1
			return evaluator.recursePolynomial(targetScale, logSplit, logDegree, coefficients, powers)
		}
		return evaluator.evaluateFromPowerBasis(targetScale, coefficients, powers)
	}
	nextPower := 1 << logSplit
	for nextPower < (coefficients.Degree()>>1)+1 {
		nextPower <<= 1
	}
	quotient, remainder := paperMKFSplitCoefficients(coefficients, nextPower)
	level := powers[nextPower].Level - 1
	if quotient.MaxDeg >= 1<<(logDegree-1) && quotient.Lead {
		level++
	}
	currentQi := evaluator.params.QiFloat64(level)
	left, err := evaluator.recursePolynomial(targetScale*currentQi/powers[nextPower].Scale, logSplit, logDegree, quotient, powers)
	if err != nil {
		return nil, err
	}
	right, err := evaluator.recursePolynomial(targetScale, logSplit, logDegree, remainder, powers)
	if err != nil {
		return nil, err
	}
	if left.Level > right.Level {
		for left.Level != right.Level+1 {
			if left, err = evaluator.DropLevelNew(left, 1); err != nil {
				return nil, err
			}
		}
	}
	left, err = evaluator.MulRelinNew(left, powers[nextPower])
	if err != nil {
		return nil, err
	}
	if left.Level > right.Level {
		if left, err = evaluator.RescaleNew(left, targetScale); err != nil {
			return nil, err
		}
		return evaluator.AddNew(left, right)
	}
	if left, err = evaluator.AddNew(left, right); err != nil {
		return nil, err
	}
	return evaluator.RescaleNew(left, targetScale)
}

func (evaluator *paperMKFCKKSEvaluator) evaluateFromPowerBasis(targetScale float64, coefficients *ckks.Polynomial, powers map[int]*paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	minimumDegree := 0
	for degree := coefficients.Degree(); degree > 0; degree-- {
		if paperMKFIsNotNegligible(coefficients.Coeffs[degree]) {
			minimumDegree = degree
			break
		}
	}
	constant := coefficients.Coeffs[0]
	if minimumDegree == 0 {
		result := evaluator.newZero(powers[1].Level, targetScale)
		if paperMKFIsNotNegligible(constant) {
			return evaluator.AddConstNew(result, constant)
		}
		return result, nil
	}
	minimumDegree = coefficients.Degree()
	currentQi := evaluator.params.QiFloat64(powers[minimumDegree].Level)
	result := evaluator.newZero(powers[minimumDegree].Level, targetScale*currentQi)
	var err error
	if paperMKFIsNotNegligible(constant) {
		if result, err = evaluator.AddConstNew(result, constant); err != nil {
			return nil, err
		}
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
		if result, err = evaluator.multByGaussianIntegerAndAdd(powers[degree], realInteger, imaginaryInteger, result); err != nil {
			return nil, err
		}
	}
	return evaluator.RescaleNew(result, targetScale)
}
