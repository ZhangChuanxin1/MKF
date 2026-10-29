package mkfcpu

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// The frozen main.cpp Layer3 schedule after the 32->64 transition is:
// layer8 (.57, .33), then layer9 (.69, .10). Layer9 has one additional
// bootstrap after its second activation before final_layer consumes it.
type realResNetStage3SameWidthConfig struct {
	name         string
	conv1Prefix  string
	conv1Scale   float64
	conv2Prefix  string
	conv2Scale   float64
	finalRefresh bool
}

var realResNetStage3SameWidthConfigs = []realResNetStage3SameWidthConfig{
	{name: "block2", conv1Prefix: "layer8-conv1bn1", conv1Scale: 0.57, conv2Prefix: "layer8-conv2bn2", conv2Scale: 0.33},
	{name: "block3", conv1Prefix: "layer9-conv1bn1", conv1Scale: 0.69, conv2Prefix: "layer9-conv2bn2", conv2Scale: 0.10, finalRefresh: true},
}

type realResNetStage3SameWidthTrace struct {
	config         realResNetStage3SameWidthConfig
	conv1          blockBoundaryMeasurement
	refresh1       blockRefreshMeasurement
	activation1    blockBoundaryMeasurement
	approximation1 float64
	conv2          blockBoundaryMeasurement
	residual       blockBoundaryMeasurement
	refresh2       blockRefreshMeasurement
	activation2    blockBoundaryMeasurement
	approximation2 float64
	finalRefresh   *blockRefreshMeasurement
}

func loadRealStage3SameWidthConvWeights(weightsDirectory, prefix string, scale float64, slots int) (*realStage3ConvWeights, error) {
	if slots%realStage3LogicalSlots != 0 {
		return nil, fmt.Errorf("stage3 logical layout needs a multiple of %d slots, got %d", realStage3LogicalSlots, slots)
	}
	loaded := new(realStage3ConvWeights)
	packingCopies := slots / realStage3LogicalSlots
	for diagonal := 0; diagonal < realStage3OutputChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				kernelIndex := kernelY*3 + kernelX + 1
				path := stage3FrozenFile(weightsDirectory, fmt.Sprintf("%s-ch%d-k%d.bin", prefix, diagonal, kernelIndex))
				values, err := readFrozenFloatVector(path)
				if err != nil {
					return nil, err
				}
				if err := stage3ExpectLength(path, values, realStage3LogicalSlots); err != nil {
					return nil, err
				}
				loaded.masks[diagonal][kernelY][kernelX] = make([]complex128, slots)
				deltaY, deltaX := kernelY-1, kernelX-1
				for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
					sourceBlock := (outputChannel + diagonal) % realStage3OutputChannels
					sourceBase := sourceBlock * realStage3OutputBlock
					coefficient := values[sourceBase+4*realStage3OutputWidth+4]
					loaded.kernel[outputChannel][sourceBlock][kernelY][kernelX] = scale * coefficient
					destinationBase := outputChannel * realStage3OutputBlock
					for y := 0; y < realStage3OutputWidth; y++ {
						for x := 0; x < realStage3OutputWidth; x++ {
							position := y*realStage3OutputWidth + x
							got := values[sourceBase+position]
							valid := y+deltaY >= 0 && y+deltaY < realStage3OutputWidth && x+deltaX >= 0 && x+deltaX < realStage3OutputWidth
							want := 0.0
							if valid {
								want = coefficient
							}
							if math.Abs(got-want) > stage3AuditTolerance {
								return nil, fmt.Errorf("%s: invalid 8x8 padding mask at output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, outputChannel, sourceBlock, y, x, got, want)
							}
							for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
								loaded.masks[diagonal][kernelY][kernelX][copyIndex*realStage3LogicalSlots+destinationBase+position] = complex(scale*got, 0)
							}
						}
					}
				}
			}
		}
	}

	biasPath := stage3FrozenFile(weightsDirectory, prefix+"-bias.bin")
	biasValues, err := readFrozenFloatVector(biasPath)
	if err != nil {
		return nil, err
	}
	if err := stage3ExpectLength(biasPath, biasValues, realStage3LogicalSlots); err != nil {
		return nil, err
	}
	loaded.packedBias = make([]complex128, slots)
	for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
		base := outputChannel * realStage3OutputBlock
		rawBias := biasValues[base]
		loaded.bias[outputChannel] = scale * rawBias
		for position := 0; position < realStage3OutputBlock; position++ {
			if math.Abs(biasValues[base+position]-rawBias) > stage3AuditTolerance {
				return nil, fmt.Errorf("%s: bias block %d is not constant at position %d", biasPath, outputChannel, position)
			}
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				loaded.packedBias[copyIndex*realStage3LogicalSlots+base+position] = complex(scale*rawBias, 0)
			}
		}
	}
	return loaded, nil
}

func requireRealStage3Boundary(t *testing.T, name string, measurement blockBoundaryMeasurement) {
	t.Helper()
	if measurement.maxError >= 1e-3 {
		t.Fatalf("real Layer3 boundary %s failed: level=%d maxError=%g rmse=%g", name, measurement.level, measurement.maxError, measurement.rmse)
	}
}

func TestEncryptedRealResNet20CompleteStage3ChainP2TwoParty(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}

	transitionMain, err := loadRealStage3StrideWeights(weightsDirectory, realStage3MainScale, params.Slots())
	if err != nil {
		t.Fatalf("layer7-conv1bn1 audit failed: %v", err)
	}
	transitionSkip, err := loadRealStage3SkipWeights(weightsDirectory, realStage3SkipScale, params.Slots())
	if err != nil {
		t.Fatalf("layer7dx-conv1bn1 audit failed: %v", err)
	}
	transitionConv2, err := loadRealStage3ConvWeights(weightsDirectory, realStage3Conv2Scale, params.Slots())
	if err != nil {
		t.Fatalf("layer7-conv2bn2 audit failed: %v", err)
	}
	conv1Weights := make([]*realStage3ConvWeights, len(realResNetStage3SameWidthConfigs))
	conv2Weights := make([]*realStage3ConvWeights, len(realResNetStage3SameWidthConfigs))
	for blockIndex, config := range realResNetStage3SameWidthConfigs {
		conv1Weights[blockIndex], err = loadRealStage3SameWidthConvWeights(weightsDirectory, config.conv1Prefix, config.conv1Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv1 audit failed: %v", config.name, err)
		}
		conv2Weights[blockIndex], err = loadRealStage3SameWidthConvWeights(weightsDirectory, config.conv2Prefix, config.conv2Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv2 audit failed: %v", config.name, err)
		}
	}

	packedInput, plainInput, err := deterministicRealStage3Input(params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	plainMain := plainRealStage3StrideConv(plainInput, transitionMain)
	plainSkip := plainRealStage3Skip(plainInput, transitionSkip)
	plainMainHalves := [2][]complex128{plainSparseStage3Half(plainMain, 0), plainSparseStage3Half(plainMain, 1)}
	plainSkipHalves := [2][]complex128{plainSparseStage3Half(plainSkip, 0), plainSparseStage3Half(plainSkip, 1)}
	if maximum, _, packErr := stage3MaxDifference(plainPackStage3Halves(plainMainHalves[0], plainMainHalves[1]), plainMain); packErr != nil || maximum != 0 {
		t.Fatalf("scalar main packing audit failed: maxDifference=%g err=%v", maximum, packErr)
	}
	if maximum, _, packErr := stage3MaxDifference(plainPackStage3Halves(plainSkipHalves[0], plainSkipHalves[1]), plainSkip); packErr != nil || maximum != 0 {
		t.Fatalf("scalar skip packing audit failed: maxDifference=%g err=%v", maximum, packErr)
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
	rotations := stage3RotationKeys()
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
	aggregate := aggregateSecret(params.Parameters, shares) // test verifier only
	started := time.Now()

	inputMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, input, plainInput)
	refreshedInput, inputRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, input, minLevel, logBound, refreshCRS, encoder, aggregate, plainInput)
	if err != nil {
		t.Fatalf("block1 pre-transition refresh failed: %v", err)
	}
	mainHalves, err := encryptedRealStage3StrideConv(params, encoder, evaluator, public, shares, refreshedInput, transitionMain)
	if err != nil {
		t.Fatalf("layer7-conv1bn1 failed: %v", err)
	}
	skipHalves, err := encryptedRealStage3Skip(params, encoder, evaluator, public, shares, refreshedInput, transitionSkip)
	if err != nil {
		t.Fatalf("layer7dx-conv1bn1 failed: %v", err)
	}
	mainHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[0], plainMainHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[1], plainMainHalves[1]),
	}
	skipHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[0], plainSkipHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[1], plainSkipHalves[1]),
	}
	packedMain := encryptedPackStage3Halves(params, encoder, evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage3Halves(params, encoder, evaluator, skipHalves[0], skipHalves[1])
	if err := evaluator.Rescale(packedMain, params.Scale(), packedMain); err != nil {
		t.Fatalf("block1 packed-main normalization failed: %v", err)
	}
	if err := evaluator.Rescale(packedSkip, params.Scale(), packedSkip); err != nil {
		t.Fatalf("block1 packed-skip normalization failed: %v", err)
	}
	packedMainMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedMain, plainMain)
	packedSkipMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedSkip, plainSkip)
	refreshedMain, mainRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, packedMain, minLevel, logBound, refreshCRS, encoder, aggregate, plainMain)
	if err != nil {
		t.Fatalf("block1 post-pack refresh failed: %v", err)
	}
	block1Poly1 := referenceResNetBlockReLUPolynomial(realStage3MainScale)
	plainBlock1Activation1, block1Approximation1, err := plainReferenceResNetBlockReLU(plainMain, block1Poly1, realStage3MainScale)
	if err != nil {
		t.Fatalf("block1 activation1 plaintext gate failed: %v", err)
	}
	block1Activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedMain, block1Poly1, params.Scale())
	if err != nil {
		t.Fatalf("block1 activation1 failed: %v", err)
	}
	block1Activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, block1Activation1, plainBlock1Activation1)
	plainBlock1Conv2 := plainRealStage3Conv(plainBlock1Activation1, transitionConv2)
	block1Conv2, err := encryptedRealStage3Conv(params, encoder, evaluator, public, shares, block1Activation1, transitionConv2)
	if err != nil {
		t.Fatalf("layer7-conv2bn2 failed: %v", err)
	}
	block1Conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, block1Conv2, plainBlock1Conv2)
	if packedSkip.Level() < block1Conv2.Level() {
		t.Fatalf("block1 skip level %d below conv2 level %d", packedSkip.Level(), block1Conv2.Level())
	}
	evaluator.DropLevel(packedSkip, packedSkip.Level()-block1Conv2.Level())
	evaluator.Add(block1Conv2, packedSkip, block1Conv2)
	plainBlock1Residual := stage3AddPlainBranches(plainBlock1Conv2, plainSkip)
	block1ResidualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, block1Conv2, plainBlock1Residual)
	refreshedBlock1Residual, block1ResidualRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, block1Conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainBlock1Residual)
	if err != nil {
		t.Fatalf("block1 post-residual refresh failed: %v", err)
	}
	block1Poly2 := referenceResNetBlockReLUPolynomial(realStage3SkipScale)
	plainBlock1Activation2, block1Approximation2, err := plainReferenceResNetBlockReLU(plainBlock1Residual, block1Poly2, realStage3SkipScale)
	if err != nil {
		t.Fatalf("block1 activation2 plaintext gate failed: %v", err)
	}
	block1Activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedBlock1Residual, block1Poly2, params.Scale())
	if err != nil {
		t.Fatalf("block1 activation2 failed: %v", err)
	}
	block1Activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, block1Activation2, plainBlock1Activation2)

	for name, measurement := range map[string]blockBoundaryMeasurement{
		"input":                inputMeasurement,
		"block1-input-refresh": {level: inputRefresh.outputLevel, maxError: inputRefresh.maxError, rmse: inputRefresh.rmse},
		"block1-main-half0":    mainHalfMeasurements[0], "block1-main-half1": mainHalfMeasurements[1],
		"block1-skip-half0": skipHalfMeasurements[0], "block1-skip-half1": skipHalfMeasurements[1],
		"block1-packed-main": packedMainMeasurement, "block1-packed-skip": packedSkipMeasurement,
		"block1-main-refresh": {level: mainRefresh.outputLevel, maxError: mainRefresh.maxError, rmse: mainRefresh.rmse},
		"block1-activation1":  block1Activation1Measurement, "block1-conv2": block1Conv2Measurement,
		"block1-residual":         block1ResidualMeasurement,
		"block1-residual-refresh": {level: block1ResidualRefresh.outputLevel, maxError: block1ResidualRefresh.maxError, rmse: block1ResidualRefresh.rmse},
		"block1-activation2":      block1Activation2Measurement,
	} {
		requireRealStage3Boundary(t, name, measurement)
	}

	currentCiphertext := block1Activation2
	currentPlaintext := plainBlock1Activation2
	traces := make([]realResNetStage3SameWidthTrace, 0, len(realResNetStage3SameWidthConfigs))
	for blockIndex, config := range realResNetStage3SameWidthConfigs {
		skipCiphertext := currentCiphertext
		skipPlaintext := currentPlaintext
		plainConv1 := plainRealStage3Conv(currentPlaintext, conv1Weights[blockIndex])
		conv1, err := encryptedRealStage3Conv(params, encoder, evaluator, public, shares, currentCiphertext, conv1Weights[blockIndex])
		if err != nil {
			t.Fatalf("%s conv1 failed: %v", config.name, err)
		}
		conv1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv1, plainConv1)
		refreshed1, refresh1, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv1, minLevel, logBound, refreshCRS, encoder, aggregate, plainConv1)
		if err != nil {
			t.Fatalf("%s refresh1 failed: %v", config.name, err)
		}
		poly1 := referenceResNetBlockReLUPolynomial(config.conv1Scale)
		plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainConv1, poly1, config.conv1Scale)
		if err != nil {
			t.Fatalf("%s activation1 plaintext gate failed: %v", config.name, err)
		}
		activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed1, poly1, params.Scale())
		if err != nil {
			t.Fatalf("%s activation1 failed: %v", config.name, err)
		}
		activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)
		plainConv2 := plainRealStage3Conv(plainActivation1, conv2Weights[blockIndex])
		conv2, err := encryptedRealStage3Conv(params, encoder, evaluator, public, shares, activation1, conv2Weights[blockIndex])
		if err != nil {
			t.Fatalf("%s conv2 failed: %v", config.name, err)
		}
		conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
		plainResidual := addPlainRealResNetBlockSkip(plainConv2, skipPlaintext, config.conv2Scale)
		if err := addScaledRealResNetBlockSkip(params, evaluator, conv2, skipCiphertext, config.conv2Scale); err != nil {
			t.Fatalf("%s residual addition failed: %v", config.name, err)
		}
		residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)
		refreshed2, refresh2, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
		if err != nil {
			t.Fatalf("%s refresh2 failed: %v", config.name, err)
		}
		poly2 := referenceResNetBlockReLUPolynomial(config.conv2Scale)
		plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, config.conv2Scale)
		if err != nil {
			t.Fatalf("%s activation2 plaintext gate failed: %v", config.name, err)
		}
		activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed2, poly2, params.Scale())
		if err != nil {
			t.Fatalf("%s activation2 failed: %v", config.name, err)
		}
		activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation2, plainActivation2)
		trace := realResNetStage3SameWidthTrace{config: config, conv1: conv1Measurement, refresh1: refresh1, activation1: activation1Measurement, approximation1: approximation1, conv2: conv2Measurement, residual: residualMeasurement, refresh2: refresh2, activation2: activation2Measurement, approximation2: approximation2}
		for name, measurement := range map[string]blockBoundaryMeasurement{
			config.name + "-conv1":       conv1Measurement,
			config.name + "-refresh1":    {level: refresh1.outputLevel, maxError: refresh1.maxError, rmse: refresh1.rmse},
			config.name + "-activation1": activation1Measurement,
			config.name + "-conv2":       conv2Measurement,
			config.name + "-residual":    residualMeasurement,
			config.name + "-refresh2":    {level: refresh2.outputLevel, maxError: refresh2.maxError, rmse: refresh2.rmse},
			config.name + "-activation2": activation2Measurement,
		} {
			requireRealStage3Boundary(t, name, measurement)
		}
		currentCiphertext = activation2
		currentPlaintext = plainActivation2
		if config.finalRefresh {
			finalCiphertext, finalRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, currentCiphertext, minLevel, logBound, refreshCRS, encoder, aggregate, currentPlaintext)
			if err != nil {
				t.Fatalf("%s final refresh failed: %v", config.name, err)
			}
			requireRealStage3Boundary(t, config.name+"-final-refresh", blockBoundaryMeasurement{level: finalRefresh.outputLevel, maxError: finalRefresh.maxError, rmse: finalRefresh.rmse})
			trace.finalRefresh = &finalRefresh
			currentCiphertext = finalCiphertext
		}
		traces = append(traces, trace)
	}

	finalMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, currentCiphertext, currentPlaintext)
	requireRealStage3Boundary(t, "final-output", finalMeasurement)
	elapsed := time.Since(started)
	t.Logf("real ResNet-20 complete Stage3/Layer3 chain P2: PASS; blocks=3 dataOwner=0 modelOwner=1 inputCiphertexts=1 encryptedWeightCiphertexts=3520 encryptedBiasCiphertexts=9 ct-ct-products=3520 scalarSkipCtPtProducts=2 collectiveRefreshes=8 degree59Activations=6 rotationKeys=%d rotationCalls=3709 publicPackingMaskMultiplications=200 packingRescaleLevels=3 logicalInputPacking=32x16x16x2copies logicalOutputPacking=64x8x8x4copies checkedSlotsPerBoundary=%d", len(rotations), params.Slots())
	t.Logf("weight schedule audit: block1=(layer7-conv1bn1,0.63;layer7dx-conv1bn1,0.40;layer7-conv2bn2,0.40) block2=(layer8-conv1bn1,0.57;layer8-conv2bn2,0.33) block3=(layer9-conv1bn1,0.69;layer9-conv2bn2,0.10;finalRefresh=true)")
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	t.Logf("boundary block1 pre-transition refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", inputRefresh.preDropLevel, inputRefresh.inputLevel, inputRefresh.outputLevel, inputRefresh.logBound, inputRefresh.maxError, inputRefresh.rmse)
	t.Logf("boundary block1 stride-main halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", mainHalfMeasurements[0].level, mainHalfMeasurements[0].maxError, mainHalfMeasurements[0].rmse, mainHalfMeasurements[1].level, mainHalfMeasurements[1].maxError, mainHalfMeasurements[1].rmse)
	t.Logf("boundary block1 stride-skip halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", skipHalfMeasurements[0].level, skipHalfMeasurements[0].maxError, skipHalfMeasurements[0].rmse, skipHalfMeasurements[1].level, skipHalfMeasurements[1].maxError, skipHalfMeasurements[1].rmse)
	t.Logf("boundary block1 packed-main: level=%d maxError=%g rmse=%g; packed-skip: level=%d maxError=%g rmse=%g", packedMainMeasurement.level, packedMainMeasurement.maxError, packedMainMeasurement.rmse, packedSkipMeasurement.level, packedSkipMeasurement.maxError, packedSkipMeasurement.rmse)
	t.Logf("boundary block1 main refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", mainRefresh.preDropLevel, mainRefresh.inputLevel, mainRefresh.outputLevel, mainRefresh.logBound, mainRefresh.maxError, mainRefresh.rmse)
	t.Logf("boundary block1 activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage3MainScale, block1Activation1Measurement.level, block1Activation1Measurement.maxError, block1Activation1Measurement.rmse, block1Approximation1)
	t.Logf("boundary block1 conv2(prefix=layer7-conv2bn2 scale=%.2f): level=%d maxError=%g rmse=%g", realStage3Conv2Scale, block1Conv2Measurement.level, block1Conv2Measurement.maxError, block1Conv2Measurement.rmse)
	t.Logf("boundary block1 residual: level=%d maxError=%g rmse=%g", block1ResidualMeasurement.level, block1ResidualMeasurement.maxError, block1ResidualMeasurement.rmse)
	t.Logf("boundary block1 residual refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", block1ResidualRefresh.preDropLevel, block1ResidualRefresh.inputLevel, block1ResidualRefresh.outputLevel, block1ResidualRefresh.logBound, block1ResidualRefresh.maxError, block1ResidualRefresh.rmse)
	t.Logf("boundary block1 activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage3SkipScale, block1Activation2Measurement.level, block1Activation2Measurement.maxError, block1Activation2Measurement.rmse, block1Approximation2)
	for _, trace := range traces {
		t.Logf("boundary %s conv1(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv1Prefix, trace.config.conv1Scale, trace.conv1.level, trace.conv1.maxError, trace.conv1.rmse)
		t.Logf("boundary %s refresh1: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh1.preDropLevel, trace.refresh1.inputLevel, trace.refresh1.outputLevel, trace.refresh1.logBound, trace.refresh1.maxError, trace.refresh1.rmse)
		t.Logf("boundary %s activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv1Scale, trace.activation1.level, trace.activation1.maxError, trace.activation1.rmse, trace.approximation1)
		t.Logf("boundary %s conv2(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv2Prefix, trace.config.conv2Scale, trace.conv2.level, trace.conv2.maxError, trace.conv2.rmse)
		t.Logf("boundary %s residual(+%.2f*skip): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv2Scale, trace.residual.level, trace.residual.maxError, trace.residual.rmse)
		t.Logf("boundary %s refresh2: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh2.preDropLevel, trace.refresh2.inputLevel, trace.refresh2.outputLevel, trace.refresh2.logBound, trace.refresh2.maxError, trace.refresh2.rmse)
		t.Logf("boundary %s activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv2Scale, trace.activation2.level, trace.activation2.maxError, trace.activation2.rmse, trace.approximation2)
		if trace.finalRefresh != nil {
			t.Logf("boundary %s final refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.finalRefresh.preDropLevel, trace.finalRefresh.inputLevel, trace.finalRefresh.outputLevel, trace.finalRefresh.logBound, trace.finalRefresh.maxError, trace.finalRefresh.rmse)
		}
	}
	t.Logf("boundary final output: level=%d maxError=%g rmse=%g", finalMeasurement.level, finalMeasurement.maxError, finalMeasurement.rmse)
	t.Logf("functional-only elapsed=%s; this is a correctness-gate wall time, not a CPU or FPGA benchmark", elapsed)
}
