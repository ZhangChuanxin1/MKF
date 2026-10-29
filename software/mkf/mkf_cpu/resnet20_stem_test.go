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
	resnetImageWidth      = 32
	resnetImageChannels   = 3
	resnetStemChannels    = 16
	resnetChannelSlotSize = resnetImageWidth * resnetImageWidth
)

type resnetStemFixture struct {
	image  [][][]float64
	weight [][][][]float64
}

func deterministicResNetStemFixture() resnetStemFixture {
	image := make([][][]float64, resnetImageChannels)
	for c := range image {
		image[c] = make([][]float64, resnetImageWidth)
		for y := range image[c] {
			image[c][y] = make([]float64, resnetImageWidth)
			for x := range image[c][y] {
				image[c][y][x] = float64(((c+1)*17+y*5+x*3)%31-15) / 64.0
			}
		}
	}
	weight := make([][][][]float64, resnetStemChannels)
	for oc := range weight {
		weight[oc] = make([][][]float64, resnetImageChannels)
		for ic := range weight[oc] {
			weight[oc][ic] = make([][]float64, 3)
			for ky := 0; ky < 3; ky++ {
				weight[oc][ic][ky] = make([]float64, 3)
				for kx := 0; kx < 3; kx++ {
					weight[oc][ic][ky][kx] = float64(((oc+1)*11+(ic+1)*7+ky*3+kx)%19-9) / 256.0
				}
			}
		}
	}
	return resnetStemFixture{image: image, weight: weight}
}

func packReplicatedInputChannel(params ckks.Parameters, image [][][]float64, inputChannel int) []complex128 {
	values := make([]complex128, params.Slots())
	logicalSlots := resnetStemChannels * resnetChannelSlotSize
	for copyIndex := 0; copyIndex < len(values)/logicalSlots; copyIndex++ {
		for outputChannel := 0; outputChannel < resnetStemChannels; outputChannel++ {
			base := copyIndex*logicalSlots + outputChannel*resnetChannelSlotSize
			for y := 0; y < resnetImageWidth; y++ {
				for x := 0; x < resnetImageWidth; x++ {
					values[base+y*resnetImageWidth+x] = complex(image[inputChannel][y][x], 0)
				}
			}
		}
	}
	return values
}

func packEncryptedKernelMask(params ckks.Parameters, weight [][][][]float64, inputChannel, kernelY, kernelX int) []complex128 {
	values := make([]complex128, params.Slots())
	deltaY, deltaX := kernelY-1, kernelX-1
	logicalSlots := resnetStemChannels * resnetChannelSlotSize
	for copyIndex := 0; copyIndex < len(values)/logicalSlots; copyIndex++ {
		for outputChannel := 0; outputChannel < resnetStemChannels; outputChannel++ {
			base := copyIndex*logicalSlots + outputChannel*resnetChannelSlotSize
			coefficient := complex(weight[outputChannel][inputChannel][kernelY][kernelX], 0)
			for y := 0; y < resnetImageWidth; y++ {
				for x := 0; x < resnetImageWidth; x++ {
					sourceY, sourceX := y+deltaY, x+deltaX
					if sourceY >= 0 && sourceY < resnetImageWidth && sourceX >= 0 && sourceX < resnetImageWidth {
						values[base+y*resnetImageWidth+x] = coefficient
					}
				}
			}
		}
	}
	return values
}

func plainResNetStem(fixture resnetStemFixture) []complex128 {
	out := make([]complex128, resnetStemChannels*resnetChannelSlotSize)
	for oc := 0; oc < resnetStemChannels; oc++ {
		base := oc * resnetChannelSlotSize
		for y := 0; y < resnetImageWidth; y++ {
			for x := 0; x < resnetImageWidth; x++ {
				var sum float64
				for ic := 0; ic < resnetImageChannels; ic++ {
					for ky := 0; ky < 3; ky++ {
						for kx := 0; kx < 3; kx++ {
							sourceY, sourceX := y+ky-1, x+kx-1
							if sourceY >= 0 && sourceY < resnetImageWidth && sourceX >= 0 && sourceX < resnetImageWidth {
								sum += fixture.image[ic][sourceY][sourceX] * fixture.weight[oc][ic][ky][kx]
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

func runEncryptedResNetStem(t *testing.T, parameter string) {
	t.Helper()
	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	if params.Slots() < resnetStemChannels*resnetChannelSlotSize {
		t.Fatalf("%s has %d slots; stem requires %d", parameter, params.Slots(), resnetStemChannels*resnetChannelSlotSize)
	}
	fixture := deterministicResNetStemFixture()
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
	// Four collective rotation keys are sufficient. Corner shifts compose one
	// vertical and one horizontal rotation, so the 3x3 stencil still performs
	// eight rotations per input channel without publishing diagonal keys.
	rotations := []int{-resnetImageWidth, -1, 1, resnetImageWidth}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, rotations, crs)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)

	inputs := make([]*ckks.Ciphertext, resnetImageChannels)
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, packReplicatedInputChannel(params, fixture.image, inputChannel))
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
				weightMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, packEncryptedKernelMask(params, fixture.weight, inputChannel, kernelY, kernelX))
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

	aggregate := aggregateSecret(params.Parameters, shares)
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(sum), params.LogSlots())
	want := plainResNetStem(fixture)
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
		t.Fatalf("encrypted ResNet-20 stem failed: parameter=%s level=%d maxError=%g rmse=%g", parameter, sum.Level(), maxError, rmse)
	}
	t.Logf("encrypted ResNet-20 stem %s: PASS; encrypted image owner=0 encrypted weights owner=1 rotationKeys=%d rotationCalls=24 ct-ct-products=27 outputLevel=%d maxError=%g rmse=%g", parameter, len(rotations), sum.Level(), maxError, rmse)
}

func TestEncryptedResNet20StemP2TwoParty(t *testing.T) {
	runEncryptedResNetStem(t, "P2")
}

func TestEncryptedResNet20StemP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_STEM") != "1" {
		t.Skip("set MKF_RUN_P3_STEM=1 to run the encrypted P3 ResNet-20 stem functional gate")
	}
	runEncryptedResNetStem(t, "P3IRCandidate")
}
