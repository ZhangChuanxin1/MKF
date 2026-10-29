package mkfcpu

import (
	"bufio"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const (
	realResNetStemActiveSlots    = resnetStemChannels * resnetChannelSlotSize
	realResNetStemReferenceScale = 0.90
)

// realResNetStemWeights is the canonical 27-ciphertext view of the frozen
// LowMemoryFHEResNet20 stem. The upstream files are indexed by output channel
// and kernel position and store three 1024-slot RGB segments. loadRealResNetStemWeights
// validates that layout and transposes it into one 16384-slot mask for every
// (input channel, kernel row, kernel column).
type realResNetStemWeights struct {
	masks      [resnetImageChannels][3][3][]complex128
	kernel     [resnetStemChannels][resnetImageChannels][3][3]float64
	biasPacked []complex128
	bias       [resnetStemChannels]float64
}

func readFrozenFloatVector(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	values := make([]float64, 0, realResNetStemActiveSlots)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024), 1024*1024)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		for _, field := range strings.Split(scanner.Text(), ",") {
			field = strings.TrimSpace(field)
			if field == "" {
				continue
			}
			value, parseErr := strconv.ParseFloat(field, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("%s:%d: %w", path, lineNumber, parseErr)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%s:%d: non-finite value %g", path, lineNumber, value)
			}
			values = append(values, value)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	return values, nil
}

func loadRealResNetStemWeightsAtScale(weightsDirectory string, referenceScale float64, slots int) (*realResNetStemWeights, error) {
	if !finitePositive(referenceScale) {
		return nil, fmt.Errorf("real stem reference scale must be positive and finite, got %g", referenceScale)
	}
	if slots < realResNetStemActiveSlots || slots%realResNetStemActiveSlots != 0 {
		return nil, fmt.Errorf("parameter has %d slots; real stem requires a positive multiple of %d", slots, realResNetStemActiveSlots)
	}
	packingCopies := slots / realResNetStemActiveSlots

	loaded := new(realResNetStemWeights)
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				loaded.masks[inputChannel][kernelY][kernelX] = make([]complex128, slots)
			}
		}
	}

	for outputChannel := 0; outputChannel < resnetStemChannels; outputChannel++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				kernelIndex := kernelY*3 + kernelX + 1
				path := filepath.Join(weightsDirectory, fmt.Sprintf("conv1bn1-ch%d-k%d.bin", outputChannel, kernelIndex))
				values, err := readFrozenFloatVector(path)
				if err != nil {
					return nil, err
				}
				if len(values) != realResNetStemActiveSlots {
					return nil, fmt.Errorf("%s: got %d values, want %d", path, len(values), realResNetStemActiveSlots)
				}

				deltaY, deltaX := kernelY-1, kernelX-1
				for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
					segmentBase := inputChannel * resnetChannelSlotSize
					// (16,16) is valid for every 3x3 offset and therefore stores
					// the folded scalar kernel coefficient without a boundary mask.
					coefficient := values[segmentBase+16*resnetImageWidth+16]
					loaded.kernel[outputChannel][inputChannel][kernelY][kernelX] = referenceScale * coefficient
					outputBase := outputChannel * resnetChannelSlotSize
					for y := 0; y < resnetImageWidth; y++ {
						for x := 0; x < resnetImageWidth; x++ {
							position := y*resnetImageWidth + x
							got := values[segmentBase+position]
							valid := y+deltaY >= 0 && y+deltaY < resnetImageWidth && x+deltaX >= 0 && x+deltaX < resnetImageWidth
							want := 0.0
							if valid {
								want = coefficient
							}
							if math.Abs(got-want) > 1e-15 {
								return nil, fmt.Errorf("%s: invalid packed mask at inputChannel=%d y=%d x=%d: got %.17g want %.17g", path, inputChannel, y, x, got, want)
							}
							for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
								loaded.masks[inputChannel][kernelY][kernelX][copyIndex*realResNetStemActiveSlots+outputBase+position] = complex(referenceScale*got, 0)
							}
						}
					}
				}
				for i := resnetImageChannels * resnetChannelSlotSize; i < len(values); i++ {
					if values[i] != 0 {
						return nil, fmt.Errorf("%s: unexpected nonzero tail at slot %d: %.17g", path, i, values[i])
					}
				}
			}
		}
	}

	biasPath := filepath.Join(weightsDirectory, "conv1bn1-bias.bin")
	biasValues, err := readFrozenFloatVector(biasPath)
	if err != nil {
		return nil, err
	}
	if len(biasValues) != realResNetStemActiveSlots {
		return nil, fmt.Errorf("%s: got %d values, want %d", biasPath, len(biasValues), realResNetStemActiveSlots)
	}
	loaded.biasPacked = make([]complex128, slots)
	for outputChannel := 0; outputChannel < resnetStemChannels; outputChannel++ {
		base := outputChannel * resnetChannelSlotSize
		rawBias := biasValues[base]
		loaded.bias[outputChannel] = referenceScale * rawBias
		for position := 0; position < resnetChannelSlotSize; position++ {
			if math.Abs(biasValues[base+position]-rawBias) > 1e-15 {
				return nil, fmt.Errorf("%s: bias block %d is not constant at position %d", biasPath, outputChannel, position)
			}
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				loaded.biasPacked[copyIndex*realResNetStemActiveSlots+base+position] = complex(referenceScale*biasValues[base+position], 0)
			}
		}
	}
	return loaded, nil
}

func loadRealResNetStemWeights(weightsDirectory string, slots int) (*realResNetStemWeights, error) {
	return loadRealResNetStemWeightsAtScale(weightsDirectory, realResNetStemReferenceScale, slots)
}

func plainRealFoldedResNetStem(image [][][]float64, weights *realResNetStemWeights) []complex128 {
	out := make([]complex128, realResNetStemActiveSlots)
	for outputChannel := 0; outputChannel < resnetStemChannels; outputChannel++ {
		base := outputChannel * resnetChannelSlotSize
		for y := 0; y < resnetImageWidth; y++ {
			for x := 0; x < resnetImageWidth; x++ {
				sum := weights.bias[outputChannel]
				for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY, sourceX := y+kernelY-1, x+kernelX-1
							if sourceY >= 0 && sourceY < resnetImageWidth && sourceX >= 0 && sourceX < resnetImageWidth {
								sum += image[inputChannel][sourceY][sourceX] * weights.kernel[outputChannel][inputChannel][kernelY][kernelX]
							}
						}
					}
				}
				out[base+y*resnetImageWidth+x] = complex(sum, 0)
			}
		}
	}
	return out
}

func runEncryptedRealFoldedResNetStem(t *testing.T, parameter string) {
	t.Helper()
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}

	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := loadRealResNetStemWeights(weightsDirectory, params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	image := deterministicResNetStemFixture().image

	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, 2)
	public := make([]*rlwe.PublicKey, 2)
	for i := range shares {
		shares[i], public[i] = kgen.GenKeyPair()
	}
	encoder := ckks.NewEncoder(params)
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	rotations := []int{-resnetImageWidth, -1, 1, resnetImageWidth}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, rotations, crs)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)

	inputs := make([]*ckks.Ciphertext, resnetImageChannels)
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, packReplicatedInputChannel(params, image, inputChannel))
		if err != nil {
			t.Fatal(err)
		}
		inputs[inputChannel], err = interactiveOwnerFold(params, multi, shares, 0)
		if err != nil {
			t.Fatal(err)
		}
	}

	var sum *ckks.Ciphertext
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				deltaY, deltaX := kernelY-1, kernelX-1
				shifted := inputs[inputChannel].CopyNew()
				if deltaY != 0 {
					shifted = evaluator.RotateNew(shifted, deltaY*resnetImageWidth)
				}
				if deltaX != 0 {
					shifted = evaluator.RotateNew(shifted, deltaX)
				}

				weightMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[inputChannel][kernelY][kernelX])
				if err != nil {
					t.Fatal(err)
				}
				weightCompact, err := interactiveOwnerFold(params, weightMulti, shares, 1)
				if err != nil {
					t.Fatal(err)
				}
				term := evaluator.MulRelinNew(shifted, weightCompact)
				if sum == nil {
					sum = term
				} else {
					evaluator.Add(sum, term, sum)
				}
			}
		}
	}
	if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
		t.Fatal(err)
	}

	biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.biasPacked)
	if err != nil {
		t.Fatal(err)
	}
	biasCompact, err := interactiveOwnerFold(params, biasMulti, shares, 1)
	if err != nil {
		t.Fatal(err)
	}
	evaluator.DropLevel(biasCompact, biasCompact.Level()-sum.Level())
	evaluator.Add(sum, biasCompact, sum)

	aggregate := aggregateSecret(params.Parameters, shares)
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(sum), params.LogSlots())
	want := plainRealFoldedResNetStem(image, weights)
	var maxError, sumSquared float64
	for i := range want {
		error := cmplx.Abs(decoded[i] - want[i])
		if error > maxError {
			maxError = error
		}
		sumSquared += error * error
	}
	rmse := math.Sqrt(sumSquared / float64(len(want)))
	if maxError >= 1e-4 {
		t.Fatalf("real folded encrypted ResNet-20 stem failed: parameter=%s level=%d maxError=%g rmse=%g", parameter, sum.Level(), maxError, rmse)
	}
	t.Logf("real folded encrypted ResNet-20 stem %s: PASS; referenceScale=%.2f dataOwner=0 modelOwner=1 inputCiphertexts=3 encryptedWeightCiphertexts=27 encryptedBiasCiphertexts=1 rotationKeys=%d rotationCalls=24 ct-ct-products=27 outputLevel=%d checkedSlots=%d maxError=%g rmse=%g weightsDir=%s", parameter, realResNetStemReferenceScale, len(rotations), sum.Level(), len(want), maxError, rmse, weightsDirectory)
}

func TestEncryptedRealFoldedResNet20StemP2TwoParty(t *testing.T) {
	runEncryptedRealFoldedResNetStem(t, "P2")
}

func TestEncryptedRealFoldedResNet20StemP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_REAL_STEM") != "1" {
		t.Skip("set MKF_RUN_P3_REAL_STEM=1 to run the real folded P3 ResNet-20 stem functional gate")
	}
	runEncryptedRealFoldedResNetStem(t, "P3IRCandidate")
}
