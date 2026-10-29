package mkfcpu

import (
	"math"
	"math/cmplx"
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

func decodePaperMKFState(t *testing.T, params ckks.Parameters, encoder ckks.Encoder, state *paperMKFCKKSCiphertext, sources []*rlwe.SecretKey) []complex128 {
	t.Helper()
	phase, err := paperMKFPhase(params, state, sources)
	if err != nil {
		t.Fatal(err)
	}
	zero := params.RingQ().NewPolyLvl(state.Level)
	zero.IsNTT = true
	ciphertext := ckks.NewCiphertextAtLevelFromPoly(state.Level, [2]*ring.Poly{phase, zero})
	ciphertext.Scale = state.Scale
	aggregate := aggregateSecret(params.Parameters, sources)
	return encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(ciphertext), params.LogSlots())
}

func foldPaperMKFStateForOracle(t *testing.T, params ckks.Parameters, state *paperMKFCKKSCiphertext, keys []*rlwe.SwitchingKey) *ckks.Ciphertext {
	t.Helper()
	folded, err := fold(params.Parameters, state.asMulti(), keys, true, newSwitchWorkspace(params.Parameters), &Profile{})
	if err != nil {
		t.Fatal(err)
	}
	result := ckks.NewCiphertextAtLevelFromPoly(state.Level, [2]*ring.Poly{folded[0], folded[1]})
	result.Scale = state.Scale
	return result
}

func maximumComplexDifference(left, right []complex128) float64 {
	maximum := 0.0
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}
	for i := 0; i < limit; i++ {
		maximum = math.Max(maximum, cmplx.Abs(left[i]-right[i]))
	}
	return maximum
}

func assertRepeatedGamma(t *testing.T, state *paperMKFCKKSCiphertext) {
	t.Helper()
	for i := 1; i < len(state.Components); i++ {
		if !equalPolyCoefficients(state.Components[0], state.Components[i]) {
			t.Fatalf("component %d is not the repeated Gamma materialization", i)
		}
	}
}

func TestPaperMKFCKKSCrossOwnerMultiplyAndRotationContract(t *testing.T) {
	params, err := CKKSParameters("P1")
	if err != nil {
		t.Fatal(err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	keys, aggregate, err := newPaperMKFCKKSTestMaterial(params, sources, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	leftValues, rightValues := make([]complex128, params.Slots()), make([]complex128, params.Slots())
	for i := range leftValues {
		leftValues[i] = complex(-0.15+0.3*float64(i%97)/96, 0)
		rightValues[i] = complex(0.05+0.2*float64(i%89)/88, 0)
	}
	leftMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, leftValues)
	if err != nil {
		t.Fatal(err)
	}
	rightMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, rightValues)
	if err != nil {
		t.Fatal(err)
	}
	left, err := newPaperMKFCKKSStateFromMulti(params, leftMulti)
	if err != nil {
		t.Fatal(err)
	}
	right, err := newPaperMKFCKKSStateFromMulti(params, rightMulti)
	if err != nil {
		t.Fatal(err)
	}
	product, err := evaluator.MulRelinNew(left, right)
	if err != nil {
		t.Fatal(err)
	}
	product, err = evaluator.RescaleNew(product, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	assertRepeatedGamma(t, product)
	observedProduct := decodePaperMKFState(t, params, encoder, product, sources)
	wantProduct := make([]complex128, len(leftValues))
	for i := range wantProduct {
		wantProduct[i] = leftValues[i] * rightValues[i]
	}
	if maximum := maximumComplexDifference(observedProduct, wantProduct); maximum >= 2e-6 {
		t.Fatalf("paper MKF cross-owner multiplication failed: max error=%g", maximum)
	}

	rotated, err := evaluator.RotateNew(product, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertRepeatedGamma(t, rotated)
	observedRotation := decodePaperMKFState(t, params, encoder, rotated, sources)
	standardRotationKeys := keyGenerator.GenRotationKeysForRotations([]int{1}, false, aggregate)
	standardEvaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rtks: standardRotationKeys})
	compactProduct := ckks.NewCiphertextAtLevelFromPoly(product.Level, [2]*ring.Poly{product.C0, product.Components[0]})
	compactProduct.Scale = product.Scale
	standardRotation := standardEvaluator.RotateNew(compactProduct, 1)
	wantRotation := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(standardRotation), params.LogSlots())
	if maximum := maximumComplexDifference(observedRotation, wantRotation); maximum >= 2e-6 {
		t.Fatalf("party-specific Galois accumulation failed: max error=%g", maximum)
	}
	if evaluator.profile.Decompositions != 7 {
		t.Fatalf("unexpected decomposition count: got %d want 7 (2n+1 multiply plus n rotation)", evaluator.profile.Decompositions)
	}
	t.Logf("paper MKF P1/n=2 multiply+rotation PASS: construction=%s decompositions=%d externalPolys=%d", paperMKFCKKSConstructionID, evaluator.profile.Decompositions, evaluator.profile.ExternalProductPolys)
}

func TestPaperMKFCKKSSyntheticStemContract(t *testing.T) {
	params, err := CKKSParameters("P1")
	if err != nil {
		t.Fatal(err)
	}
	rotations := []int{-33, -32, -31, -1, 1, 31, 32, 33}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	keys, aggregate, err := newPaperMKFCKKSTestMaterial(params, sources, rotations)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	inputValues := make([]complex128, params.Slots())
	for i := range inputValues {
		inputValues[i] = complex(-0.05+0.1*float64(i%251)/250, 0)
	}
	inputMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, inputValues)
	if err != nil {
		t.Fatal(err)
	}
	input, err := newPaperMKFCKKSStateFromMulti(params, inputMulti)
	if err != nil {
		t.Fatal(err)
	}
	compactInput := foldPaperMKFStateForOracle(t, params, input, keys.Fold)
	standardRotationKeys := keyGenerator.GenRotationKeysForRotations(rotations, false, aggregate)
	standardEvaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: keys.Relin, Rtks: standardRotationKeys})

	rotationForKernel := []int{-33, -32, -31, -1, 0, 1, 31, 32, 33}
	weights := []float64{0.08, -0.03, 0.02, 0.01, 0.11, -0.04, -0.02, 0.05, 0.07}
	var actual *paperMKFCKKSCiphertext
	var oracle *ckks.Ciphertext
	for kernel := range weights {
		shifted := input
		oracleShifted := compactInput
		if rotationForKernel[kernel] != 0 {
			shifted, err = evaluator.RotateNew(input, rotationForKernel[kernel])
			if err != nil {
				t.Fatal(err)
			}
			oracleShifted = standardEvaluator.RotateNew(compactInput, rotationForKernel[kernel])
		}
		weightValues := make([]complex128, params.Slots())
		for i := range weightValues {
			weightValues[i] = complex(weights[kernel], 0)
		}
		weightMulti, makeErr := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weightValues)
		if makeErr != nil {
			t.Fatal(makeErr)
		}
		weightState, makeErr := newPaperMKFCKKSStateFromMulti(params, weightMulti)
		if makeErr != nil {
			t.Fatal(makeErr)
		}
		term, multiplyErr := evaluator.MulRelinNew(shifted, weightState)
		if multiplyErr != nil {
			t.Fatal(multiplyErr)
		}
		term, multiplyErr = evaluator.RescaleNew(term, params.Scale())
		if multiplyErr != nil {
			t.Fatal(multiplyErr)
		}
		oracleWeight := foldPaperMKFStateForOracle(t, params, weightState, keys.Fold)
		oracleTerm := standardEvaluator.MulRelinNew(oracleShifted, oracleWeight)
		if rescaleErr := standardEvaluator.Rescale(oracleTerm, params.Scale(), oracleTerm); rescaleErr != nil {
			t.Fatal(rescaleErr)
		}
		if actual == nil {
			actual, oracle = term, oracleTerm
		} else {
			actual, err = evaluator.AddNew(actual, term)
			if err != nil {
				t.Fatal(err)
			}
			oracle = standardEvaluator.AddNew(oracle, oracleTerm)
		}
	}
	biasValues := make([]complex128, params.Slots())
	for i := range biasValues {
		biasValues[i] = 0.0125
	}
	biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, biasValues)
	if err != nil {
		t.Fatal(err)
	}
	biasState, err := newPaperMKFCKKSStateFromMulti(params, biasMulti)
	if err != nil {
		t.Fatal(err)
	}
	actual, err = evaluator.AddNew(actual, biasState)
	if err != nil {
		t.Fatal(err)
	}
	oracleBias := foldPaperMKFStateForOracle(t, params, biasState, keys.Fold)
	oracle = standardEvaluator.AddNew(oracle, oracleBias)
	observed := decodePaperMKFState(t, params, encoder, actual, sources)
	want := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(oracle), params.LogSlots())
	if maximum := maximumComplexDifference(observed, want); maximum >= 1e-5 {
		t.Fatalf("synthetic Stem equivalence failed: max error=%g", maximum)
	}
	t.Logf("paper MKF synthetic 3x3 Stem PASS: rotations=8 ct-ct-products=9 level=%d construction=%s", actual.Level, paperMKFCKKSConstructionID)
}

func TestPaperMKFCKKSPolynomialInternalProductsContract(t *testing.T) {
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	keys, _, err := newPaperMKFCKKSTestMaterial(params, sources, nil)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	values := make([]complex128, params.Slots())
	for i := range values {
		values[i] = complex(-0.2+0.4*float64(i%193)/192, 0)
	}
	multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	input, err := newPaperMKFCKKSStateFromMulti(params, multi)
	if err != nil {
		t.Fatal(err)
	}
	polynomial := ckks.NewPoly([]complex128{0.1, 0.2, -0.3, 0.4})
	result, err := evaluator.EvaluatePoly(input, polynomial, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	// The nonlinear product terms are repeated-Gamma outputs. The final
	// polynomial also includes a linear x term, so its complete n+1 state is
	// valid but need not have byte-identical nonconstant components.
	if len(result.Components) != 2 {
		t.Fatalf("polynomial result lost n+1 state: got %d nonconstant components", len(result.Components))
	}
	observed := decodePaperMKFState(t, params, encoder, result, sources)
	want := make([]complex128, len(values))
	for i, value := range values {
		want[i] = 0.1 + 0.2*value - 0.3*value*value + 0.4*value*value*value
	}
	if maximum := maximumComplexDifference(observed, want); maximum >= 2e-5 {
		t.Fatalf("paper MKF polynomial evaluator failed: max error=%g", maximum)
	}
	if evaluator.profile.Decompositions == 0 {
		t.Fatal("polynomial evaluation did not execute the paper MKF multiplication path")
	}
	t.Logf("paper MKF polynomial internal-product PASS: degree=3 levels=%d->%d decompositions=%d construction=%s", input.Level, result.Level, evaluator.profile.Decompositions, paperMKFCKKSConstructionID)
}

func TestPaperMKFCKKSDegree59ActivationContract(t *testing.T) {
	if os.Getenv("MKF_RUN_TRUE_DEG59") != "1" {
		t.Skip("set MKF_RUN_TRUE_DEG59=1 for the paper-MKF degree-59 single-operator gate")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	keys, _, err := newPaperMKFCKKSTestMaterial(params, sources, nil)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	values := make([]complex128, params.Slots())
	for i := range values {
		values[i] = complex(-0.9+1.8*float64(i%257)/256, 0)
	}
	multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	input, err := newPaperMKFCKKSStateFromMulti(params, multi)
	if err != nil {
		t.Fatal(err)
	}
	reference := func(value complex128) complex128 {
		if real(value) < 0 {
			return 0
		}
		return value / complex(resnetStemOutputScale, 0)
	}
	polynomial := ckks.Approximate(reference, -1, 1, resnetReferenceReLUDegree)
	mapped, err := evaluator.MultByConstNew(input, 2/(polynomial.B-polynomial.A))
	if err != nil {
		t.Fatal(err)
	}
	mapped, err = evaluator.AddConstNew(mapped, (-polynomial.A-polynomial.B)/(polynomial.B-polynomial.A))
	if err != nil {
		t.Fatal(err)
	}
	mapped, err = evaluator.RescaleNew(mapped, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	result, err := evaluator.EvaluatePoly(mapped, polynomial, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	observed := decodePaperMKFState(t, params, encoder, result, sources)
	want := make([]complex128, len(values))
	for i, value := range values {
		want[i] = evaluateChebyshevSeries(polynomial, value)
	}
	maximum := maximumComplexDifference(observed, want)
	if maximum >= 1e-3 {
		t.Fatalf("paper MKF degree-59 activation failed: max FHE error=%g", maximum)
	}
	if evaluator.profile.Decompositions == 0 {
		t.Fatal("degree-59 activation did not execute the paper MKF multiplication path")
	}
	t.Logf("paper MKF degree-59 activation PASS: levels=%d->%d decompositions=%d externalPolys=%d maxFHEError=%g construction=%s", input.Level, result.Level, evaluator.profile.Decompositions, evaluator.profile.ExternalProductPolys, maximum, paperMKFCKKSConstructionID)
}

func TestPaperMKFCKKSRealFoldedStemContract(t *testing.T) {
	if os.Getenv("MKF_RUN_TRUE_REAL_STEM") != "1" {
		t.Skip("set MKF_RUN_TRUE_REAL_STEM=1 with MKF_RESNET20_WEIGHTS_DIR for the checkpoint Stem gate")
	}
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Fatal("MKF_RESNET20_WEIGHTS_DIR is required")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	weights, err := loadRealResNetStemWeights(weightsDirectory, params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	image := deterministicResNetStemFixture().image
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	rotations := []int{-resnetImageWidth, -1, 1, resnetImageWidth}
	keys, _, err := newPaperMKFCKKSTestMaterial(params, sources, rotations)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	encoder := ckks.NewEncoder(params)
	inputs := make([]*paperMKFCKKSCiphertext, resnetImageChannels)
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		multi, makeErr := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, packReplicatedInputChannel(params, image, inputChannel))
		if makeErr != nil {
			t.Fatal(makeErr)
		}
		inputs[inputChannel], makeErr = newPaperMKFCKKSStateFromMulti(params, multi)
		if makeErr != nil {
			t.Fatal(makeErr)
		}
	}
	var sum *paperMKFCKKSCiphertext
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		var spatial [3][3]*paperMKFCKKSCiphertext
		spatial[1][1] = inputs[inputChannel]
		spatial[0][1], err = evaluator.RotateNew(inputs[inputChannel], -resnetImageWidth)
		if err != nil {
			t.Fatal(err)
		}
		spatial[2][1], err = evaluator.RotateNew(inputs[inputChannel], resnetImageWidth)
		if err != nil {
			t.Fatal(err)
		}
		spatial[1][0], err = evaluator.RotateNew(inputs[inputChannel], -1)
		if err != nil {
			t.Fatal(err)
		}
		spatial[1][2], err = evaluator.RotateNew(inputs[inputChannel], 1)
		if err != nil {
			t.Fatal(err)
		}
		spatial[0][0], err = evaluator.RotateNew(spatial[0][1], -1)
		if err != nil {
			t.Fatal(err)
		}
		spatial[0][2], err = evaluator.RotateNew(spatial[0][1], 1)
		if err != nil {
			t.Fatal(err)
		}
		spatial[2][0], err = evaluator.RotateNew(spatial[2][1], -1)
		if err != nil {
			t.Fatal(err)
		}
		spatial[2][2], err = evaluator.RotateNew(spatial[2][1], 1)
		if err != nil {
			t.Fatal(err)
		}
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weightMulti, makeErr := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[inputChannel][kernelY][kernelX])
				if makeErr != nil {
					t.Fatal(makeErr)
				}
				weightState, makeErr := newPaperMKFCKKSStateFromMulti(params, weightMulti)
				if makeErr != nil {
					t.Fatal(makeErr)
				}
				term, multiplyErr := evaluator.MulRelinNew(spatial[kernelY][kernelX], weightState)
				if multiplyErr != nil {
					t.Fatal(multiplyErr)
				}
				if sum == nil {
					sum = term
				} else if sum, err = evaluator.AddNew(sum, term); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	if sum, err = evaluator.RescaleNew(sum, params.Scale()); err != nil {
		t.Fatal(err)
	}
	biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.biasPacked)
	if err != nil {
		t.Fatal(err)
	}
	biasState, err := newPaperMKFCKKSStateFromMulti(params, biasMulti)
	if err != nil {
		t.Fatal(err)
	}
	if biasState.Level > sum.Level {
		biasState, err = evaluator.DropLevelNew(biasState, biasState.Level-sum.Level)
		if err != nil {
			t.Fatal(err)
		}
	}
	if sum, err = evaluator.AddNew(sum, biasState); err != nil {
		t.Fatal(err)
	}
	decoded := decodePaperMKFState(t, params, encoder, sum, sources)
	want := plainRealFoldedResNetStem(image, weights)
	maximum, sumSquares := 0.0, 0.0
	for i := range want {
		difference := cmplx.Abs(decoded[i] - want[i])
		maximum = math.Max(maximum, difference)
		sumSquares += difference * difference
	}
	rmse := math.Sqrt(sumSquares / float64(len(want)))
	if maximum >= 1e-4 {
		t.Fatalf("paper MKF real folded Stem failed: max error=%g rmse=%g", maximum, rmse)
	}
	if evaluator.profile.Decompositions != 183 {
		t.Fatalf("unexpected real Stem decomposition count: got %d want 183", evaluator.profile.Decompositions)
	}
	t.Logf("paper MKF checkpoint Stem PASS: inputCT=3 weightCT=27 biasCT=1 rotations=24 ct-ct-products=27 level=%d checkedSlots=%d maxError=%g rmse=%g decompositions=%d construction=%s", sum.Level, len(want), maximum, rmse, evaluator.profile.Decompositions, paperMKFCKKSConstructionID)
}

func TestPaperMKFCKKSCollectiveRefreshAdapterContract(t *testing.T) {
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for i := range sources {
		sources[i], public[i] = keyGenerator.GenKeyPair()
	}
	keys, _, err := newPaperMKFCKKSTestMaterial(params, sources, nil)
	if err != nil {
		t.Fatal(err)
	}
	online, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	adapter := newPaperMKFLattigoAdapter(online)
	encoder := ckks.NewEncoder(params)
	values := make([]complex128, params.Slots())
	for i := range values {
		values[i] = complex(-0.125+0.25*float64(i%127)/126, 0)
	}
	input, err := paperMKFOwnerCarrier(params, encoder, public, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), len(sources), params.Q())
	if !ok {
		t.Fatal("P2 does not expose a collective-refresh level")
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	refreshed, err := adapter.collectiveRefresh(input, sources, minLevel, logBound, crs)
	if err != nil {
		t.Fatal(err)
	}
	refreshedState, err := adapter.state(refreshed)
	if err != nil {
		t.Fatal(err)
	}
	assertRepeatedGamma(t, refreshedState)
	observed := decodePaperMKFState(t, params, encoder, refreshedState, sources)
	maximum := maximumComplexDifference(observed, values)
	if maximum >= 1e-5 {
		t.Fatalf("paper MKF collective-refresh adapter failed: max error=%g", maximum)
	}
	if refreshed.Level() != params.MaxLevel() {
		t.Fatalf("refresh output level=%d want %d", refreshed.Level(), params.MaxLevel())
	}
	t.Logf("paper MKF collective refresh PASS: folded n+1 level=%d->compact min=%d->materialized n+1 level=%d logBound=%d maxError=%g; no aggregate secret reconstructed online", input.Level(), minLevel, refreshed.Level(), logBound, maximum)
}

// TestPaperMKFCKKSPublicMaskAndStage2PackingPrefixContract guards the exact
// degree-n/public-plaintext path that Stage2 packing exercises. It is a
// lightweight prefix contract, not a full transition or performance test.
func TestPaperMKFCKKSPublicMaskAndStage2PackingPrefixContract(t *testing.T) {
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	sources, public := make([]*rlwe.SecretKey, 2), make([]*rlwe.PublicKey, 2)
	for party := range sources {
		sources[party], public[party] = keyGenerator.GenKeyPair()
	}
	keys, _, err := newPaperMKFCKKSTestMaterial(params, sources, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	online, err := newPaperMKFCKKSEvaluator(params, 2, keys)
	if err != nil {
		t.Fatal(err)
	}
	adapter := newPaperMKFLattigoAdapter(online)
	encoder := ckks.NewEncoder(params)
	values := make([]complex128, params.Slots())
	for index := range values {
		values[index] = complex(float64((index*17)%101-50)/512, 0)
	}
	input, err := paperMKFOwnerCarrier(params, encoder, public, realResNetTailDataOwner, values)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("degree2-carrier-times-packing-mask", func(t *testing.T) {
		mask := stage2AlternatingMask(params.Slots(), 2)
		plaintext := stage2EncodeUnitMask(params, encoder, input.Level(), mask)
		masked := adapter.MulNew(input, plaintext)
		if masked.Degree() != 2 || masked.Level() != input.Level() || masked.Scale != input.Scale*plaintext.Scale {
			t.Fatalf("public mask metadata mismatch: degree=%d level=%d scale=%g want 2/%d/%g", masked.Degree(), masked.Level(), masked.Scale, input.Level(), input.Scale*plaintext.Scale)
		}
		for component, polynomial := range masked.Value {
			if !polynomial.IsNTT {
				t.Fatalf("masked component %d lost its NTT flag", component)
			}
		}
		state, err := adapter.state(masked)
		if err != nil {
			t.Fatal(err)
		}
		observed := decodePaperMKFState(t, params, encoder, state, sources)
		want := stage2ApplyPlainMask(values, mask)
		if maximum := maximumComplexDifference(observed, want); maximum >= 2e-6 {
			t.Fatalf("degree-2 public packing mask mismatch: maxError=%g", maximum)
		}
	})

	t.Run("stage2-transition-packing-prefix", func(t *testing.T) {
		rotated := adapter.RotateNew(input, 1)
		summed := adapter.AddNew(input, rotated)
		mask := stage2AlternatingMask(params.Slots(), 2)
		masked := stage2MaskedCiphertext(params, encoder, adapter, summed, mask)
		state, err := adapter.state(masked)
		if err != nil {
			t.Fatal(err)
		}
		observed := decodePaperMKFState(t, params, encoder, state, sources)
		want := stage2ApplyPlainMask(stage2AddPlain(values, stage2RotatePlainLeft(values, 1)), mask)
		if maximum := maximumComplexDifference(observed, want); maximum >= 3e-6 {
			t.Fatalf("Stage2 packing prefix mismatch: maxError=%g", maximum)
		}
		if online.counts.MulRelinCalls != 0 || online.counts.RotationCalls != 1 {
			t.Fatalf("public packing prefix counts mismatch: mulRelin=%d rotations=%d", online.counts.MulRelinCalls, online.counts.RotationCalls)
		}
	})
	t.Logf("paper MKF P2 degree-2 public mask + Stage2 packing prefix PASS: mulRelin=%d rotations=%d decompositions=%d externalPolys=%d", online.counts.MulRelinCalls, online.counts.RotationCalls, online.profile.Decompositions, online.profile.ExternalProductPolys)
}
