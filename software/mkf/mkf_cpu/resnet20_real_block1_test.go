package mkfcpu

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const (
	realResNetBlockChannels   = 16
	realResNetBlockConv1Scale = 1.00
	realResNetBlockConv2Scale = 0.52
)

type realResNetBlockConvWeights struct {
	masks      [realResNetBlockChannels][3][3][]complex128
	kernel     [realResNetBlockChannels][realResNetBlockChannels][3][3]float64
	bias       [realResNetBlockChannels]float64
	packedBias []complex128
}

type blockBoundaryMeasurement struct {
	level    int
	maxError float64
	rmse     float64
}

type blockRefreshMeasurement struct {
	preDropLevel int
	inputLevel   int
	outputLevel  int
	logBound     int
	maxError     float64
	rmse         float64
}

func loadRealResNetBlockConvWeights(weightsDirectory, prefix string, referenceScale float64, slots int) (*realResNetBlockConvWeights, error) {
	if slots < realResNetStemActiveSlots || slots%realResNetStemActiveSlots != 0 {
		return nil, fmt.Errorf("parameter has %d slots; block packing requires a positive multiple of %d", slots, realResNetStemActiveSlots)
	}
	packingCopies := slots / realResNetStemActiveSlots
	loaded := new(realResNetBlockConvWeights)
	for diagonal := 0; diagonal < realResNetBlockChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				loaded.masks[diagonal][kernelY][kernelX] = make([]complex128, slots)
				kernelIndex := kernelY*3 + kernelX + 1
				path := filepath.Join(weightsDirectory, fmt.Sprintf("%s-ch%d-k%d.bin", prefix, diagonal, kernelIndex))
				values, err := readFrozenFloatVector(path)
				if err != nil {
					return nil, err
				}
				if len(values) != realResNetStemActiveSlots {
					return nil, fmt.Errorf("%s: got %d values, want %d", path, len(values), realResNetStemActiveSlots)
				}

				deltaY, deltaX := kernelY-1, kernelX-1
				for outputChannel := 0; outputChannel < realResNetBlockChannels; outputChannel++ {
					// The upstream notebook first stores output channel o in block o,
					// then applies np.roll(..., 1024*diagonal). Undo that block roll.
					sourceBlock := (outputChannel + diagonal) % realResNetBlockChannels
					inputChannel := sourceBlock
					sourceBase := sourceBlock * resnetChannelSlotSize
					coefficient := values[sourceBase+16*resnetImageWidth+16]
					loaded.kernel[outputChannel][inputChannel][kernelY][kernelX] = referenceScale * coefficient
					for y := 0; y < resnetImageWidth; y++ {
						for x := 0; x < resnetImageWidth; x++ {
							position := y*resnetImageWidth + x
							got := values[sourceBase+position]
							valid := y+deltaY >= 0 && y+deltaY < resnetImageWidth && x+deltaX >= 0 && x+deltaX < resnetImageWidth
							want := 0.0
							if valid {
								want = coefficient
							}
							if math.Abs(got-want) > 1e-15 {
								return nil, fmt.Errorf("%s: invalid rolled diagonal mask at output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, outputChannel, inputChannel, y, x, got, want)
							}
							for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
								destination := copyIndex*realResNetStemActiveSlots + outputChannel*resnetChannelSlotSize + position
								loaded.masks[diagonal][kernelY][kernelX][destination] = complex(referenceScale*got, 0)
							}
						}
					}
				}
			}
		}
	}

	biasPath := filepath.Join(weightsDirectory, prefix+"-bias.bin")
	biasValues, err := readFrozenFloatVector(biasPath)
	if err != nil {
		return nil, err
	}
	if len(biasValues) != realResNetStemActiveSlots {
		return nil, fmt.Errorf("%s: got %d values, want %d", biasPath, len(biasValues), realResNetStemActiveSlots)
	}
	loaded.packedBias = make([]complex128, slots)
	for outputChannel := 0; outputChannel < realResNetBlockChannels; outputChannel++ {
		base := outputChannel * resnetChannelSlotSize
		rawBias := biasValues[base]
		loaded.bias[outputChannel] = referenceScale * rawBias
		for position := 0; position < resnetChannelSlotSize; position++ {
			if math.Abs(biasValues[base+position]-rawBias) > 1e-15 {
				return nil, fmt.Errorf("%s: bias block %d is not constant at position %d", biasPath, outputChannel, position)
			}
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				destination := copyIndex*realResNetStemActiveSlots + base + position
				loaded.packedBias[destination] = complex(referenceScale*rawBias, 0)
			}
		}
	}
	return loaded, nil
}

func deterministicRealResNetBlockInput(slots int) ([]complex128, []complex128, error) {
	if slots < realResNetStemActiveSlots || slots%realResNetStemActiveSlots != 0 {
		return nil, nil, fmt.Errorf("parameter has %d slots; block input requires a positive multiple of %d", slots, realResNetStemActiveSlots)
	}
	active := make([]complex128, realResNetStemActiveSlots)
	for channel := 0; channel < realResNetBlockChannels; channel++ {
		base := channel * resnetChannelSlotSize
		for y := 0; y < resnetImageWidth; y++ {
			for x := 0; x < resnetImageWidth; x++ {
				active[base+y*resnetImageWidth+x] = complex(float64(((channel+1)*13+y*5+x*7)%31-15)/4096.0, 0)
			}
		}
	}
	packed := make([]complex128, slots)
	for copyIndex := 0; copyIndex < slots/realResNetStemActiveSlots; copyIndex++ {
		copy(packed[copyIndex*realResNetStemActiveSlots:], active)
	}
	return packed, active, nil
}

func plainRealResNetBlockConv(input []complex128, weights *realResNetBlockConvWeights) []complex128 {
	output := make([]complex128, realResNetStemActiveSlots)
	for outputChannel := 0; outputChannel < realResNetBlockChannels; outputChannel++ {
		base := outputChannel * resnetChannelSlotSize
		for y := 0; y < resnetImageWidth; y++ {
			for x := 0; x < resnetImageWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realResNetBlockChannels; inputChannel++ {
					inputBase := inputChannel * resnetChannelSlotSize
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY, sourceX := y+kernelY-1, x+kernelX-1
							if sourceY >= 0 && sourceY < resnetImageWidth && sourceX >= 0 && sourceX < resnetImageWidth {
								sum += input[inputBase+sourceY*resnetImageWidth+sourceX] * complex(weights.kernel[outputChannel][inputChannel][kernelY][kernelX], 0)
							}
						}
					}
				}
				output[base+y*resnetImageWidth+x] = sum
			}
		}
	}
	return output
}

func resNetBlockSpatialRotations(evaluator ckks.Evaluator, input *ckks.Ciphertext) [3][3]*ckks.Ciphertext {
	var shifted [3][3]*ckks.Ciphertext
	shifted[1][1] = input
	shifted[0][1] = evaluator.RotateNew(input, -resnetImageWidth)
	shifted[2][1] = evaluator.RotateNew(input, resnetImageWidth)
	shifted[1][0] = evaluator.RotateNew(input, -1)
	shifted[1][2] = evaluator.RotateNew(input, 1)
	shifted[0][0] = evaluator.RotateNew(shifted[0][1], -1)
	shifted[0][2] = evaluator.RotateNew(shifted[0][1], 1)
	shifted[2][0] = evaluator.RotateNew(shifted[2][1], -1)
	shifted[2][2] = evaluator.RotateNew(shifted[2][1], 1)
	return shifted
}

func encryptedRealResNetBlockConv(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realResNetBlockConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realResNetBlockChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*resnetChannelSlotSize)
		}
		spatial := resNetBlockSpatialRotations(evaluator, channelShifted)
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weightMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[diagonal][kernelY][kernelX])
				if err != nil {
					return nil, err
				}
				weightCompact, err := interactiveOwnerFold(params, weightMulti, shares, 1)
				if err != nil {
					return nil, err
				}
				term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weightCompact)
				if sum == nil {
					sum = term
				} else {
					evaluator.Add(sum, term, sum)
				}
			}
		}
	}
	if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
		return nil, err
	}

	biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.packedBias)
	if err != nil {
		return nil, err
	}
	biasCompact, err := interactiveOwnerFold(params, biasMulti, shares, 1)
	if err != nil {
		return nil, err
	}
	evaluator.DropLevel(biasCompact, biasCompact.Level()-sum.Level())
	evaluator.Add(sum, biasCompact, sum)
	return sum, nil
}

func referenceResNetBlockReLUPolynomial(scale float64) *ckks.Polynomial {
	return ckks.Approximate(func(x complex128) complex128 {
		if real(x) < 0 {
			return 0
		}
		return x / complex(scale, 0)
	}, -1, 1, resnetReferenceReLUDegree)
}

func encryptedReferenceResNetBlockReLU(evaluator ckks.Evaluator, input *ckks.Ciphertext, polynomial *ckks.Polynomial, targetScale float64) (*ckks.Ciphertext, error) {
	mapped := input.CopyNew()
	evaluator.MultByConst(mapped, 2/(polynomial.B-polynomial.A), mapped)
	evaluator.AddConst(mapped, (-polynomial.A-polynomial.B)/(polynomial.B-polynomial.A), mapped)
	if err := evaluator.Rescale(mapped, targetScale, mapped); err != nil {
		return nil, err
	}
	return evaluator.EvaluatePoly(mapped, polynomial, targetScale)
}

func plainReferenceResNetBlockReLU(input []complex128, polynomial *ckks.Polynomial, scale float64) ([]complex128, float64, error) {
	output := make([]complex128, len(input))
	var maxApproxError float64
	for i, value := range input {
		if real(value) < -1 || real(value) > 1 || math.Abs(imag(value)) > 1e-12 {
			return nil, 0, fmt.Errorf("activation input slot %d outside real reference interval [-1,1]: %v", i, value)
		}
		output[i] = evaluateChebyshevSeries(polynomial, value)
		exact := complex(0, 0)
		if real(value) >= 0 {
			exact = value / complex(scale, 0)
		}
		maxApproxError = math.Max(maxApproxError, cmplx.Abs(output[i]-exact))
	}
	return output, maxApproxError, nil
}

func measureRealResNetBlockBoundary(params ckks.Parameters, encoder ckks.Encoder, aggregate *rlwe.SecretKey, ciphertext *ckks.Ciphertext, want []complex128) blockBoundaryMeasurement {
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(ciphertext), params.LogSlots())
	var maxError, sumSquared float64
	for i := range want {
		error := cmplx.Abs(decoded[i] - want[i])
		maxError = math.Max(maxError, error)
		sumSquared += error * error
	}
	return blockBoundaryMeasurement{level: ciphertext.Level(), maxError: maxError, rmse: math.Sqrt(sumSquared / float64(len(want)))}
}

func refreshRealResNetBlockBoundary(params ckks.Parameters, evaluator ckks.Evaluator, shares []*rlwe.SecretKey, input *ckks.Ciphertext, minLevel, logBound int, crs utils.PRNG, encoder ckks.Encoder, aggregate *rlwe.SecretKey, want []complex128) (*ckks.Ciphertext, blockRefreshMeasurement, error) {
	preDropLevel := input.Level()
	if preDropLevel < minLevel {
		return nil, blockRefreshMeasurement{}, fmt.Errorf("ciphertext level %d below collective-refresh minimum %d", preDropLevel, minLevel)
	}
	evaluator.DropLevel(input, preDropLevel-minLevel)
	inputLevel := input.Level()
	refreshed, err := runCollectiveRefresh(params, shares, input, logBound, crs)
	if err != nil {
		return nil, blockRefreshMeasurement{}, err
	}
	measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, refreshed, want)
	return refreshed, blockRefreshMeasurement{
		preDropLevel: preDropLevel,
		inputLevel:   inputLevel,
		outputLevel:  refreshed.Level(),
		logBound:     logBound,
		maxError:     measurement.maxError,
		rmse:         measurement.rmse,
	}, nil
}

func addScaledRealResNetBlockSkip(params ckks.Parameters, evaluator ckks.Evaluator, main, skip *ckks.Ciphertext, scale float64) error {
	scaledSkip := evaluator.MultByConstNew(skip, scale)
	if err := evaluator.Rescale(scaledSkip, params.Scale(), scaledSkip); err != nil {
		return err
	}
	if scaledSkip.Level() < main.Level() {
		return fmt.Errorf("scaled skip level %d below main branch level %d", scaledSkip.Level(), main.Level())
	}
	evaluator.DropLevel(scaledSkip, scaledSkip.Level()-main.Level())
	evaluator.Add(main, scaledSkip, main)
	return nil
}

func addPlainRealResNetBlockSkip(main, skip []complex128, scale float64) []complex128 {
	output := make([]complex128, len(main))
	for i := range main {
		output[i] = main[i] + complex(scale, 0)*skip[i]
	}
	return output
}

func realResNetBlockRotationKeys() []int {
	rotations := []int{-resnetImageWidth, -1, 1, resnetImageWidth}
	for diagonal := 1; diagonal < realResNetBlockChannels; diagonal++ {
		rotations = append(rotations, diagonal*resnetChannelSlotSize)
	}
	return rotations
}

func runEncryptedRealResNet20Block1(t *testing.T, parameter string) {
	t.Helper()
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}

	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	conv1Weights, err := loadRealResNetBlockConvWeights(weightsDirectory, "layer1-conv1bn1", realResNetBlockConv1Scale, params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	conv2Weights, err := loadRealResNetBlockConvWeights(weightsDirectory, "layer1-conv2bn2", realResNetBlockConv2Scale, params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	packedInput, plainInput, err := deterministicRealResNetBlockInput(params.Slots())
	if err != nil {
		t.Fatal(err)
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

	rotations := realResNetBlockRotationKeys()
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
		t.Fatalf("%s lacks modulus levels for two-party collective refresh", parameter)
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	aggregate := aggregateSecret(params.Parameters, shares)

	inputMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, input, plainInput)
	plainConv1 := plainRealResNetBlockConv(plainInput, conv1Weights)
	conv1, err := encryptedRealResNetBlockConv(params, encoder, evaluator, public, shares, input, conv1Weights)
	if err != nil {
		t.Fatal(err)
	}
	conv1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv1, plainConv1)
	refreshed1, refresh1Measurement, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv1, minLevel, logBound, refreshCRS, encoder, aggregate, plainConv1)
	if err != nil {
		t.Fatal(err)
	}

	poly1 := referenceResNetBlockReLUPolynomial(realResNetBlockConv1Scale)
	if poly1.Depth() != 6 {
		t.Fatalf("unexpected first degree-%d Chebyshev depth %d", resnetReferenceReLUDegree, poly1.Depth())
	}
	plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainConv1, poly1, realResNetBlockConv1Scale)
	if err != nil {
		t.Fatal(err)
	}
	activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed1, poly1, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)

	plainConv2 := plainRealResNetBlockConv(plainActivation1, conv2Weights)
	conv2, err := encryptedRealResNetBlockConv(params, encoder, evaluator, public, shares, activation1, conv2Weights)
	if err != nil {
		t.Fatal(err)
	}
	conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
	plainResidual := addPlainRealResNetBlockSkip(plainConv2, plainInput, realResNetBlockConv2Scale)
	if err := addScaledRealResNetBlockSkip(params, evaluator, conv2, input, realResNetBlockConv2Scale); err != nil {
		t.Fatal(err)
	}
	residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)
	refreshed2, refresh2Measurement, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
	if err != nil {
		t.Fatal(err)
	}

	poly2 := referenceResNetBlockReLUPolynomial(realResNetBlockConv2Scale)
	if poly2.Depth() != 6 {
		t.Fatalf("unexpected second degree-%d Chebyshev depth %d", resnetReferenceReLUDegree, poly2.Depth())
	}
	plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, realResNetBlockConv2Scale)
	if err != nil {
		t.Fatal(err)
	}
	activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed2, poly2, params.Scale())
	if err != nil {
		t.Fatal(err)
	}
	activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation2, plainActivation2)

	measurements := []struct {
		name  string
		value blockBoundaryMeasurement
	}{
		{"input", inputMeasurement},
		{"conv1", conv1Measurement},
		{"refresh1", blockBoundaryMeasurement{level: refresh1Measurement.outputLevel, maxError: refresh1Measurement.maxError, rmse: refresh1Measurement.rmse}},
		{"activation1", activation1Measurement},
		{"conv2", conv2Measurement},
		{"residual", residualMeasurement},
		{"refresh2", blockBoundaryMeasurement{level: refresh2Measurement.outputLevel, maxError: refresh2Measurement.maxError, rmse: refresh2Measurement.rmse}},
		{"activation2", activation2Measurement},
	}
	for _, measurement := range measurements {
		if measurement.value.maxError >= 1e-3 {
			t.Fatalf("real Block1 boundary %s failed: level=%d maxError=%g rmse=%g", measurement.name, measurement.value.level, measurement.value.maxError, measurement.value.rmse)
		}
	}

	t.Logf("real folded ResNet-20 Layer1 Block1 %s: PASS; dataOwner=0 modelOwner=1 packedInputCiphertexts=1 encryptedWeightCiphertexts=288 encryptedBiasCiphertexts=2 channelDiagonalsPerConv=16 spatialOffsetsPerDiagonal=9 rotationKeys=%d rotationCallsPerConv=143 rotationCallsTotal=286 ct-ct-productsPerConv=144 ct-ct-productsTotal=288 refreshCount=2 packingCopies=%d checkedSlots=%d", parameter, len(rotations), params.Slots()/realResNetStemActiveSlots, len(plainInput))
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	t.Logf("boundary conv1(scale=%.2f): level=%d maxError=%g rmse=%g", realResNetBlockConv1Scale, conv1Measurement.level, conv1Measurement.maxError, conv1Measurement.rmse)
	t.Logf("boundary refresh1: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", refresh1Measurement.preDropLevel, refresh1Measurement.inputLevel, refresh1Measurement.outputLevel, refresh1Measurement.logBound, refresh1Measurement.maxError, refresh1Measurement.rmse)
	t.Logf("boundary activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realResNetBlockConv1Scale, activation1Measurement.level, activation1Measurement.maxError, activation1Measurement.rmse, approximation1)
	t.Logf("boundary conv2(scale=%.2f): level=%d maxError=%g rmse=%g", realResNetBlockConv2Scale, conv2Measurement.level, conv2Measurement.maxError, conv2Measurement.rmse)
	t.Logf("boundary residual(+%.2f*skip): level=%d maxError=%g rmse=%g", realResNetBlockConv2Scale, residualMeasurement.level, residualMeasurement.maxError, residualMeasurement.rmse)
	t.Logf("boundary refresh2: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", refresh2Measurement.preDropLevel, refresh2Measurement.inputLevel, refresh2Measurement.outputLevel, refresh2Measurement.logBound, refresh2Measurement.maxError, refresh2Measurement.rmse)
	t.Logf("boundary activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realResNetBlockConv2Scale, activation2Measurement.level, activation2Measurement.maxError, activation2Measurement.rmse, approximation2)
}

func TestEncryptedRealResNet20Layer1Block1P2TwoParty(t *testing.T) {
	runEncryptedRealResNet20Block1(t, "P2")
}

func TestEncryptedRealResNet20Layer1Block1P3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_REAL_BLOCK1") != "1" {
		t.Skip("set MKF_RUN_P3_REAL_BLOCK1=1 to run the real P3 ResNet-20 Layer1 Block1 functional gate")
	}
	runEncryptedRealResNet20Block1(t, "P3IRCandidate")
}
