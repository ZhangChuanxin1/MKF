package mkfcpu

import (
	"os"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// The upstream program calls these Layer2 blocks 2 and 3, while their frozen
// weight prefixes are layer5 and layer6 (main.cpp lines 377--411).
type realStage2RegularBlockConfig struct {
	name        string
	conv1Prefix string
	conv1Scale  float64
	conv2Prefix string
	conv2Scale  float64
}

var realStage2RegularBlocks = []realStage2RegularBlockConfig{
	{name: "block2", conv1Prefix: "layer5-conv1bn1", conv1Scale: 0.76, conv2Prefix: "layer5-conv2bn2", conv2Scale: 0.37},
	{name: "block3", conv1Prefix: "layer6-conv1bn1", conv1Scale: 0.63, conv2Prefix: "layer6-conv2bn2", conv2Scale: 0.25},
}

type realStage2RegularBlockTrace struct {
	config         realStage2RegularBlockConfig
	conv1          blockBoundaryMeasurement
	refresh1       blockRefreshMeasurement
	activation1    blockBoundaryMeasurement
	approximation1 float64
	conv2          blockBoundaryMeasurement
	residual       blockBoundaryMeasurement
	refresh2       blockRefreshMeasurement
	activation2    blockBoundaryMeasurement
	approximation2 float64
}

func requireStage2Boundary(t *testing.T, name string, measurement blockBoundaryMeasurement) {
	t.Helper()
	if measurement.maxError >= 1e-3 {
		t.Fatalf("complete Layer2 first failed boundary %s: level=%d maxError=%g rmse=%g", name, measurement.level, measurement.maxError, measurement.rmse)
	}
}

func TestEncryptedRealResNet20CompleteLayer2ChainP2TwoParty(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}

	transitionMain, err := loadRealStage2StrideWeights(weightsDirectory, realStage2MainScale, params.Slots())
	if err != nil {
		t.Fatalf("layer4-conv1bn1 audit failed: %v", err)
	}
	transitionSkip, err := loadRealStage2SkipWeights(weightsDirectory, realStage2SkipScale, params.Slots())
	if err != nil {
		t.Fatalf("layer4dx-conv1bn1 audit failed: %v", err)
	}
	transitionConv2, err := loadRealStage2ConvWeights(weightsDirectory, "layer4-conv2bn2", realStage2Conv2Scale, params.Slots())
	if err != nil {
		t.Fatalf("layer4-conv2bn2 audit failed: %v", err)
	}
	regularConv1 := make([]*realStage2ConvWeights, len(realStage2RegularBlocks))
	regularConv2 := make([]*realStage2ConvWeights, len(realStage2RegularBlocks))
	for index, config := range realStage2RegularBlocks {
		regularConv1[index], err = loadRealStage2ConvWeights(weightsDirectory, config.conv1Prefix, config.conv1Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv1 audit failed: %v", config.name, err)
		}
		regularConv2[index], err = loadRealStage2ConvWeights(weightsDirectory, config.conv2Prefix, config.conv2Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv2 audit failed: %v", config.name, err)
		}
	}

	packedInput, plainInput, err := deterministicRealResNetBlockInput(params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	plainTransitionMain := plainRealStage2StrideConv(plainInput, transitionMain)
	plainTransitionSkip := plainRealStage2Skip(plainInput, transitionSkip)
	mainSparse := [2][]complex128{plainSparseStage2Half(plainTransitionMain, 0), plainSparseStage2Half(plainTransitionMain, 1)}
	skipSparse := [2][]complex128{plainSparseStage2Half(plainTransitionSkip, 0), plainSparseStage2Half(plainTransitionSkip, 1)}
	if maximum, _, compareErr := stage2MaxDifference(plainPackStage2Halves(mainSparse[0], mainSparse[1]), plainTransitionMain); compareErr != nil || maximum != 0 {
		t.Fatalf("complete Layer2 scalar main packing audit failed: maxDifference=%g err=%v", maximum, compareErr)
	}
	if maximum, _, compareErr := stage2MaxDifference(plainPackStage2Halves(skipSparse[0], skipSparse[1]), plainTransitionSkip); compareErr != nil || maximum != 0 {
		t.Fatalf("complete Layer2 scalar skip packing audit failed: maxDifference=%g err=%v", maximum, compareErr)
	}

	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, 2)
	public := make([]*rlwe.PublicKey, 2)
	for party := range shares {
		shares[party], public[party] = kgen.GenKeyPair()
	}
	encoder := ckks.NewEncoder(params)
	inputMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, packedInput)
	if err != nil {
		t.Fatal(err)
	}
	input, err := interactiveOwnerFold(params, inputMulti, shares, 0)
	if err != nil {
		t.Fatal(err)
	}

	rotations := stage2RotationKeys()
	evaluationCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, rotations, evaluationCRS)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), len(shares), params.Q())
	if !ok {
		t.Fatal("P2 lacks modulus levels for two-party collective refresh")
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	aggregate := aggregateSecret(params.Parameters, shares)
	started := time.Now()

	inputMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, input, plainInput)
	requireStage2Boundary(t, "input", inputMeasurement)
	refreshedInput, inputRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, input, minLevel, logBound, refreshCRS, encoder, aggregate, plainInput)
	if err != nil {
		t.Fatalf("pre-transition refresh failed: %v", err)
	}
	requireStage2Boundary(t, "pre-transition-refresh", blockBoundaryMeasurement{level: inputRefresh.outputLevel, maxError: inputRefresh.maxError, rmse: inputRefresh.rmse})

	mainHalves, err := encryptedRealStage2StrideConv(params, encoder, evaluator, public, shares, refreshedInput, transitionMain)
	if err != nil {
		t.Fatalf("layer4-conv1bn1 failed: %v", err)
	}
	skipHalves, err := encryptedRealStage2Skip(params, encoder, evaluator, public, shares, refreshedInput, transitionSkip)
	if err != nil {
		t.Fatalf("layer4dx-conv1bn1 failed: %v", err)
	}
	mainHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[0], mainSparse[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[1], mainSparse[1]),
	}
	skipHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[0], skipSparse[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[1], skipSparse[1]),
	}
	requireStage2Boundary(t, "transition-main-half0", mainHalfMeasurements[0])
	requireStage2Boundary(t, "transition-main-half1", mainHalfMeasurements[1])
	requireStage2Boundary(t, "transition-skip-half0", skipHalfMeasurements[0])
	requireStage2Boundary(t, "transition-skip-half1", skipHalfMeasurements[1])

	packedMain := encryptedPackStage2Halves(params, encoder, evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage2Halves(params, encoder, evaluator, skipHalves[0], skipHalves[1])
	if err := evaluator.Rescale(packedMain, params.Scale(), packedMain); err != nil {
		t.Fatalf("packed-main scale normalization failed: %v", err)
	}
	if err := evaluator.Rescale(packedSkip, params.Scale(), packedSkip); err != nil {
		t.Fatalf("packed-skip scale normalization failed: %v", err)
	}
	packedMainMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedMain, plainTransitionMain)
	packedSkipMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedSkip, plainTransitionSkip)
	requireStage2Boundary(t, "transition-packed-main", packedMainMeasurement)
	requireStage2Boundary(t, "transition-packed-skip", packedSkipMeasurement)

	refreshedMain, transitionMainRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, packedMain, minLevel, logBound, refreshCRS, encoder, aggregate, plainTransitionMain)
	if err != nil {
		t.Fatalf("transition main refresh failed: %v", err)
	}
	requireStage2Boundary(t, "transition-main-refresh", blockBoundaryMeasurement{level: transitionMainRefresh.outputLevel, maxError: transitionMainRefresh.maxError, rmse: transitionMainRefresh.rmse})
	transitionPoly1 := referenceResNetBlockReLUPolynomial(realStage2MainScale)
	plainTransitionActivation1, transitionApproximation1, err := plainReferenceResNetBlockReLU(plainTransitionMain, transitionPoly1, realStage2MainScale)
	if err != nil {
		t.Fatalf("transition activation1 plaintext interval failed: %v", err)
	}
	transitionActivation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedMain, transitionPoly1, params.Scale())
	if err != nil {
		t.Fatalf("transition activation1 failed: %v", err)
	}
	transitionActivation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, transitionActivation1, plainTransitionActivation1)
	requireStage2Boundary(t, "transition-activation1", transitionActivation1Measurement)

	plainTransitionConv2 := plainRealStage2Conv(plainTransitionActivation1, transitionConv2)
	transitionConv2Ciphertext, err := encryptedRealStage2Conv(params, encoder, evaluator, public, shares, transitionActivation1, transitionConv2)
	if err != nil {
		t.Fatalf("layer4-conv2bn2 failed: %v", err)
	}
	transitionConv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, transitionConv2Ciphertext, plainTransitionConv2)
	requireStage2Boundary(t, "transition-conv2", transitionConv2Measurement)
	if packedSkip.Level() < transitionConv2Ciphertext.Level() {
		t.Fatalf("transition skip level %d below conv2 level %d", packedSkip.Level(), transitionConv2Ciphertext.Level())
	}
	evaluator.DropLevel(packedSkip, packedSkip.Level()-transitionConv2Ciphertext.Level())
	evaluator.Add(transitionConv2Ciphertext, packedSkip, transitionConv2Ciphertext)
	plainTransitionResidual := stage2AddPlainBranches(plainTransitionConv2, plainTransitionSkip)
	transitionResidualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, transitionConv2Ciphertext, plainTransitionResidual)
	requireStage2Boundary(t, "transition-residual", transitionResidualMeasurement)

	transitionResidualRefreshed, transitionResidualRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, transitionConv2Ciphertext, minLevel, logBound, refreshCRS, encoder, aggregate, plainTransitionResidual)
	if err != nil {
		t.Fatalf("transition residual refresh failed: %v", err)
	}
	requireStage2Boundary(t, "transition-residual-refresh", blockBoundaryMeasurement{level: transitionResidualRefresh.outputLevel, maxError: transitionResidualRefresh.maxError, rmse: transitionResidualRefresh.rmse})
	transitionPoly2 := referenceResNetBlockReLUPolynomial(realStage2SkipScale)
	plainTransitionActivation2, transitionApproximation2, err := plainReferenceResNetBlockReLU(plainTransitionResidual, transitionPoly2, realStage2SkipScale)
	if err != nil {
		t.Fatalf("transition activation2 plaintext interval failed: %v", err)
	}
	transitionActivation2, err := encryptedReferenceResNetBlockReLU(evaluator, transitionResidualRefreshed, transitionPoly2, params.Scale())
	if err != nil {
		t.Fatalf("transition activation2 failed: %v", err)
	}
	transitionActivation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, transitionActivation2, plainTransitionActivation2)
	requireStage2Boundary(t, "transition-activation2", transitionActivation2Measurement)
	t.Logf("progress transition block1 exact PASS: activation2 level=%d maxError=%g rmse=%g", transitionActivation2Measurement.level, transitionActivation2Measurement.maxError, transitionActivation2Measurement.rmse)

	currentCiphertext := transitionActivation2
	currentPlaintext := plainTransitionActivation2
	regularTraces := make([]realStage2RegularBlockTrace, 0, len(realStage2RegularBlocks))
	for index, config := range realStage2RegularBlocks {
		skipCiphertext := currentCiphertext
		skipPlaintext := currentPlaintext

		plainConv1 := plainRealStage2Conv(currentPlaintext, regularConv1[index])
		conv1, err := encryptedRealStage2Conv(params, encoder, evaluator, public, shares, currentCiphertext, regularConv1[index])
		if err != nil {
			t.Fatalf("%s conv1 failed: %v", config.name, err)
		}
		conv1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv1, plainConv1)
		requireStage2Boundary(t, config.name+"-conv1", conv1Measurement)
		refreshed1, refresh1, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv1, minLevel, logBound, refreshCRS, encoder, aggregate, plainConv1)
		if err != nil {
			t.Fatalf("%s refresh1 failed: %v", config.name, err)
		}
		requireStage2Boundary(t, config.name+"-refresh1", blockBoundaryMeasurement{level: refresh1.outputLevel, maxError: refresh1.maxError, rmse: refresh1.rmse})
		poly1 := referenceResNetBlockReLUPolynomial(config.conv1Scale)
		plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainConv1, poly1, config.conv1Scale)
		if err != nil {
			t.Fatalf("%s activation1 plaintext interval failed: %v", config.name, err)
		}
		activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed1, poly1, params.Scale())
		if err != nil {
			t.Fatalf("%s activation1 failed: %v", config.name, err)
		}
		activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)
		requireStage2Boundary(t, config.name+"-activation1", activation1Measurement)

		plainConv2 := plainRealStage2Conv(plainActivation1, regularConv2[index])
		conv2, err := encryptedRealStage2Conv(params, encoder, evaluator, public, shares, activation1, regularConv2[index])
		if err != nil {
			t.Fatalf("%s conv2 failed: %v", config.name, err)
		}
		conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
		requireStage2Boundary(t, config.name+"-conv2", conv2Measurement)
		plainResidual := addPlainRealResNetBlockSkip(plainConv2, skipPlaintext, config.conv2Scale)
		if err := addScaledRealResNetBlockSkip(params, evaluator, conv2, skipCiphertext, config.conv2Scale); err != nil {
			t.Fatalf("%s residual addition failed: %v", config.name, err)
		}
		residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)
		requireStage2Boundary(t, config.name+"-residual", residualMeasurement)
		refreshed2, refresh2, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
		if err != nil {
			t.Fatalf("%s refresh2 failed: %v", config.name, err)
		}
		requireStage2Boundary(t, config.name+"-refresh2", blockBoundaryMeasurement{level: refresh2.outputLevel, maxError: refresh2.maxError, rmse: refresh2.rmse})
		poly2 := referenceResNetBlockReLUPolynomial(config.conv2Scale)
		plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, config.conv2Scale)
		if err != nil {
			t.Fatalf("%s activation2 plaintext interval failed: %v", config.name, err)
		}
		activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed2, poly2, params.Scale())
		if err != nil {
			t.Fatalf("%s activation2 failed: %v", config.name, err)
		}
		activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation2, plainActivation2)
		requireStage2Boundary(t, config.name+"-activation2", activation2Measurement)
		t.Logf("progress %s exact PASS: activation2 level=%d maxError=%g rmse=%g", config.name, activation2Measurement.level, activation2Measurement.maxError, activation2Measurement.rmse)

		regularTraces = append(regularTraces, realStage2RegularBlockTrace{
			config: config, conv1: conv1Measurement, refresh1: refresh1,
			activation1: activation1Measurement, approximation1: approximation1,
			conv2: conv2Measurement, residual: residualMeasurement, refresh2: refresh2,
			activation2: activation2Measurement, approximation2: approximation2,
		})
		currentCiphertext = activation2
		currentPlaintext = plainActivation2
	}

	elapsed := time.Since(started)
	t.Logf("real ResNet-20 complete Layer2 P2: PASS; blocks=3 transition=16to32-stride2+projectedSkip+1024to256-pack dataOwner=0 modelOwner=1 encryptedWeightCiphertexts=1760 encryptedBiasCiphertexts=9 ct-ct-products=1760 collectiveRefreshes=7 degree59Activations=6 rotationKeys=%d rotationCalls=1891 publicPackingMaskMultiplications=140 packingRescaleLevels=3 logicalPacking=32x16x16x2copies checkedSlotsPerBoundary=%d", len(rotations), params.Slots())
	t.Logf("weight schedule audit: transition=(layer4-conv1bn1,%.2f;layer4dx-conv1bn1,%.2f;layer4-conv2bn2,%.2f) block2=(%s,%.2f;%s,%.2f) block3=(%s,%.2f;%s,%.2f)", realStage2MainScale, realStage2SkipScale, realStage2Conv2Scale, realStage2RegularBlocks[0].conv1Prefix, realStage2RegularBlocks[0].conv1Scale, realStage2RegularBlocks[0].conv2Prefix, realStage2RegularBlocks[0].conv2Scale, realStage2RegularBlocks[1].conv1Prefix, realStage2RegularBlocks[1].conv1Scale, realStage2RegularBlocks[1].conv2Prefix, realStage2RegularBlocks[1].conv2Scale)
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	t.Logf("boundary transition pre-refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", inputRefresh.preDropLevel, inputRefresh.inputLevel, inputRefresh.outputLevel, inputRefresh.logBound, inputRefresh.maxError, inputRefresh.rmse)
	t.Logf("boundary transition halves main=(g0 level=%d maxError=%g rmse=%g; g1 level=%d maxError=%g rmse=%g) skip=(g0 level=%d maxError=%g rmse=%g; g1 level=%d maxError=%g rmse=%g)", mainHalfMeasurements[0].level, mainHalfMeasurements[0].maxError, mainHalfMeasurements[0].rmse, mainHalfMeasurements[1].level, mainHalfMeasurements[1].maxError, mainHalfMeasurements[1].rmse, skipHalfMeasurements[0].level, skipHalfMeasurements[0].maxError, skipHalfMeasurements[0].rmse, skipHalfMeasurements[1].level, skipHalfMeasurements[1].maxError, skipHalfMeasurements[1].rmse)
	t.Logf("boundary transition packed main=(level=%d maxError=%g rmse=%g) skip=(level=%d maxError=%g rmse=%g)", packedMainMeasurement.level, packedMainMeasurement.maxError, packedMainMeasurement.rmse, packedSkipMeasurement.level, packedSkipMeasurement.maxError, packedSkipMeasurement.rmse)
	t.Logf("boundary transition refresh1: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", transitionMainRefresh.preDropLevel, transitionMainRefresh.inputLevel, transitionMainRefresh.outputLevel, transitionMainRefresh.logBound, transitionMainRefresh.maxError, transitionMainRefresh.rmse)
	t.Logf("boundary transition activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage2MainScale, transitionActivation1Measurement.level, transitionActivation1Measurement.maxError, transitionActivation1Measurement.rmse, transitionApproximation1)
	t.Logf("boundary transition conv2(scale=%.2f): level=%d maxError=%g rmse=%g; residual: level=%d maxError=%g rmse=%g", realStage2Conv2Scale, transitionConv2Measurement.level, transitionConv2Measurement.maxError, transitionConv2Measurement.rmse, transitionResidualMeasurement.level, transitionResidualMeasurement.maxError, transitionResidualMeasurement.rmse)
	t.Logf("boundary transition refresh2: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", transitionResidualRefresh.preDropLevel, transitionResidualRefresh.inputLevel, transitionResidualRefresh.outputLevel, transitionResidualRefresh.logBound, transitionResidualRefresh.maxError, transitionResidualRefresh.rmse)
	t.Logf("boundary transition activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage2SkipScale, transitionActivation2Measurement.level, transitionActivation2Measurement.maxError, transitionActivation2Measurement.rmse, transitionApproximation2)
	for _, trace := range regularTraces {
		t.Logf("boundary %s conv1(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv1Prefix, trace.config.conv1Scale, trace.conv1.level, trace.conv1.maxError, trace.conv1.rmse)
		t.Logf("boundary %s refresh1: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh1.preDropLevel, trace.refresh1.inputLevel, trace.refresh1.outputLevel, trace.refresh1.logBound, trace.refresh1.maxError, trace.refresh1.rmse)
		t.Logf("boundary %s activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv1Scale, trace.activation1.level, trace.activation1.maxError, trace.activation1.rmse, trace.approximation1)
		t.Logf("boundary %s conv2(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g; residual(+%.2f*skip): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv2Prefix, trace.config.conv2Scale, trace.conv2.level, trace.conv2.maxError, trace.conv2.rmse, trace.config.conv2Scale, trace.residual.level, trace.residual.maxError, trace.residual.rmse)
		t.Logf("boundary %s refresh2: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh2.preDropLevel, trace.refresh2.inputLevel, trace.refresh2.outputLevel, trace.refresh2.logBound, trace.refresh2.maxError, trace.refresh2.rmse)
		t.Logf("boundary %s activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv2Scale, trace.activation2.level, trace.activation2.maxError, trace.activation2.rmse, trace.approximation2)
	}
	t.Logf("functional-only elapsed=%s; this is a correctness-gate wall time, not a CPU or FPGA benchmark", elapsed)
}
