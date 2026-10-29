package mkfcpu

import (
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// These constants reproduce LowMemoryFHEResNet20's final_layer exactly:
// 64 channels, each containing an 8x8 feature map, packed channel-major in
// one 4096-slot logical copy. Full-network P2/P3 runs retain the surrounding
// 16384-slot layout and repeat this logical copy; the standalone tail gate may
// still use exactly 4096 slots. The frozen C++ reference applies no fc.bias.
const (
	realResNetTailChannels       = 64
	realResNetTailSpatialSlots   = 8 * 8
	realResNetTailClasses        = 10
	realResNetTailReferenceSlots = realResNetTailChannels * realResNetTailSpatialSlots
	realResNetTailRepeatWidth    = 16
	realResNetTailModelOwner     = 1
	realResNetTailDataOwner      = 0
)

type realResNetTailWeights struct {
	matrix [realResNetTailChannels][realResNetTailClasses]float64
	packed []complex128
	sha256 string
}

type realResNetTailBias struct {
	values [realResNetTailClasses]float64
	packed []complex128
	sha256 string
}

func loadRealResNetTailWeights(weightsDirectory string, slots int) (*realResNetTailWeights, error) {
	if slots < realResNetTailReferenceSlots || slots%realResNetTailReferenceSlots != 0 {
		return nil, fmt.Errorf("tail packing requires a positive multiple of %d slots, got %d", realResNetTailReferenceSlots, slots)
	}
	path := filepath.Join(weightsDirectory, "fc.bin")
	values, err := readFrozenFloatVector(path)
	if err != nil {
		return nil, err
	}
	if len(values) != realResNetTailChannels*realResNetTailClasses {
		return nil, fmt.Errorf("%s: got %d values, want %d channel-major FC coefficients", path, len(values), realResNetTailChannels*realResNetTailClasses)
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	loaded := &realResNetTailWeights{packed: make([]complex128, slots), sha256: fmt.Sprintf("%x", sha256.Sum256(data))}
	packingCopies := slots / realResNetTailReferenceSlots
	for channel := 0; channel < realResNetTailChannels; channel++ {
		for class := 0; class < realResNetTailClasses; class++ {
			value := values[channel*realResNetTailClasses+class]
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%s: non-finite coefficient at channel=%d class=%d", path, channel, class)
			}
			loaded.matrix[channel][class] = value
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				loaded.packed[copyIndex*realResNetTailReferenceSlots+channel*realResNetTailSpatialSlots+class] = complex(value, 0)
			}
		}
	}
	return loaded, nil
}

func loadCheckpointRealResNetTailBias(path string, slots int) (*realResNetTailBias, error) {
	if slots < realResNetTailReferenceSlots || slots%realResNetTailReferenceSlots != 0 {
		return nil, fmt.Errorf("tail bias packing requires a positive multiple of %d slots, got %d", realResNetTailReferenceSlots, slots)
	}
	values, err := readFrozenFloatVector(path)
	if err != nil {
		return nil, err
	}
	if len(values) != realResNetTailClasses {
		return nil, fmt.Errorf("%s: got %d values, want %d checkpoint FC biases", path, len(values), realResNetTailClasses)
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	loaded := &realResNetTailBias{packed: make([]complex128, slots), sha256: fmt.Sprintf("%x", sha256.Sum256(data))}
	packingCopies := slots / realResNetTailReferenceSlots
	for class, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("%s: non-finite bias at class=%d", path, class)
		}
		loaded.values[class] = value
		for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
			loaded.packed[copyIndex*realResNetTailReferenceSlots+class] = complex(value, 0)
		}
	}
	return loaded, nil
}

// deterministicRealResNetTailInput is nonnegative because final_layer consumes
// the output of the nineteenth reference ReLU. Values are deliberately small
// and bounded; this is a functional arithmetic gate, not an accuracy sample.
func deterministicRealResNetTailInput() []complex128 {
	values := make([]complex128, realResNetTailReferenceSlots)
	for channel := 0; channel < realResNetTailChannels; channel++ {
		base := channel * realResNetTailSpatialSlots
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				numerator := ((channel+1)*11 + y*7 + x*5) % 37
				values[base+y*8+x] = complex(float64(numerator)/4096.0, 0)
			}
		}
	}
	return values
}

// plainRealResNetTailLogits is independent of the packed FHE schedule. It
// computes the mathematical GAP followed by the 64x10 frozen classifier.
// It intentionally omits fc.bias because the frozen C++ FHE final_layer does.
func plainRealResNetTailLogits(input []complex128, weights *realResNetTailWeights, bias *realResNetTailBias) ([realResNetTailClasses]complex128, error) {
	var logits [realResNetTailClasses]complex128
	if len(input) != realResNetTailReferenceSlots {
		return logits, fmt.Errorf("tail input has %d values, want %d", len(input), realResNetTailReferenceSlots)
	}
	for channel := 0; channel < realResNetTailChannels; channel++ {
		var average complex128
		base := channel * realResNetTailSpatialSlots
		for position := 0; position < realResNetTailSpatialSlots; position++ {
			average += input[base+position]
		}
		average /= realResNetTailSpatialSlots
		for class := 0; class < realResNetTailClasses; class++ {
			logits[class] += average * complex(weights.matrix[channel][class], 0)
		}
	}
	if bias != nil {
		for class := 0; class < realResNetTailClasses; class++ {
			logits[class] += complex(bias.values[class], 0)
		}
	}
	return logits, nil
}

func realResNetTailRotationKeys() []int {
	// rotsum(...,64), repeat(...,16), and rotsum_padded(...,64), including
	// repeat's final -15 alignment. Duplicates from the 16-slot rotsum are
	// already present in the first six entries.
	return []int{1, 2, 4, 8, 16, 32, -15, 64, 128, 256, 512, 1024, 2048}
}

func encryptedRealResNetTailRotSum(evaluator ckks.Evaluator, input *ckks.Ciphertext, width int) *ckks.Ciphertext {
	result := input.CopyNew()
	for rotation := 1; rotation < width; rotation <<= 1 {
		shifted := evaluator.RotateNew(result, rotation)
		evaluator.Add(result, shifted, result)
	}
	return result
}

func encryptedRealResNetTailPaddedRotSum(evaluator ckks.Evaluator, input *ckks.Ciphertext) *ckks.Ciphertext {
	result := input.CopyNew()
	for channels := 1; channels < realResNetTailChannels; channels <<= 1 {
		shifted := evaluator.RotateNew(result, realResNetTailSpatialSlots*channels)
		evaluator.Add(result, shifted, result)
	}
	return result
}

func runEncryptedRealResNet20TailP2(t *testing.T, checkpointBias bool) {
	t.Helper()
	functionalStart := time.Now()
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}

	// P2's Q/P chain is retained, while LogSlots=12 matches the exact 4096-slot
	// final_layer packing used by the frozen OpenFHE implementation.
	literal := ckksP2
	literal.LogSlots = 12
	params, err := ckks.NewParametersFromLiteral(literal)
	if err != nil {
		t.Fatal(err)
	}
	if params.Slots() != realResNetTailReferenceSlots {
		t.Fatalf("unexpected P2-tail slot count %d", params.Slots())
	}
	weights, err := loadRealResNetTailWeights(weightsDirectory, params.Slots())
	if err != nil {
		t.Fatal(err)
	}
	var bias *realResNetTailBias
	semantics := "LowMemoryCPPNoFCBias"
	if checkpointBias {
		biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
		if biasPath == "" {
			t.Skip("set MKF_RESNET20_FC_BIAS_PATH to the audited cifar10_resnet20-4118986f FC-bias fixture")
		}
		bias, err = loadCheckpointRealResNetTailBias(biasPath, params.Slots())
		if err != nil {
			t.Fatal(err)
		}
		semantics = "CheckpointFaithfulWithEncryptedFCBias"
	}
	plainInput := deterministicRealResNetTailInput()
	wantLogits, err := plainRealResNetTailLogits(plainInput, weights, bias)
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
	inputMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, realResNetTailDataOwner, plainInput)
	if err != nil {
		t.Fatal(err)
	}
	input, err := interactiveOwnerFold(params, inputMulti, shares, realResNetTailDataOwner)
	if err != nil {
		t.Fatal(err)
	}

	rotations := realResNetTailRotationKeys()
	evaluationCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, rotations, evaluationCRS)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)

	// final_layer lines 219-220: 64-point spatial sum, then a public 1/64
	// mask at the first slot of every 64-slot channel block.
	gap := encryptedRealResNetTailRotSum(evaluator, input, realResNetTailSpatialSlots)
	gapMask := make([]complex128, params.Slots())
	for channel := 0; channel < realResNetTailChannels; channel++ {
		gapMask[channel*realResNetTailSpatialSlots] = complex(1.0/realResNetTailSpatialSlots, 0)
	}
	maskPlaintext := encoder.EncodeNTTAtLvlNew(gap.Level(), gapMask, params.LogSlots())
	gap = evaluator.MulNew(gap, maskPlaintext)
	if err := evaluator.Rescale(gap, params.Scale(), gap); err != nil {
		t.Fatal(err)
	}
	gapLevel := gap.Level()

	// final_layer lines 222-224: repeat each channel average into the first 16
	// slots, then multiply by the model owner's independently encrypted weights.
	repeated := encryptedRealResNetTailRotSum(evaluator, gap, realResNetTailRepeatWidth)
	repeated = evaluator.RotateNew(repeated, -realResNetTailRepeatWidth+1)
	weightMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, realResNetTailModelOwner, weights.packed)
	if err != nil {
		t.Fatal(err)
	}
	weight, err := interactiveOwnerFold(params, weightMulti, shares, realResNetTailModelOwner)
	if err != nil {
		t.Fatal(err)
	}
	if weight.Level() < repeated.Level() {
		t.Fatalf("encrypted FC weight level %d below GAP output level %d", weight.Level(), repeated.Level())
	}
	evaluator.DropLevel(weight, weight.Level()-repeated.Level())
	classified := evaluator.MulRelinNew(repeated, weight)
	if err := evaluator.Rescale(classified, params.Scale(), classified); err != nil {
		t.Fatal(err)
	}
	fcProductLevel := classified.Level()

	// final_layer line 225: add the 64 channel blocks at offsets 64*2^i.
	classified = encryptedRealResNetTailPaddedRotSum(evaluator, classified)
	encryptedBiasCiphertexts := 0
	if bias != nil {
		biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, realResNetTailModelOwner, bias.packed)
		if err != nil {
			t.Fatal(err)
		}
		biasCiphertext, err := interactiveOwnerFold(params, biasMulti, shares, realResNetTailModelOwner)
		if err != nil {
			t.Fatal(err)
		}
		if biasCiphertext.Level() < classified.Level() {
			t.Fatalf("encrypted FC bias level %d below classifier output level %d", biasCiphertext.Level(), classified.Level())
		}
		evaluator.DropLevel(biasCiphertext, biasCiphertext.Level()-classified.Level())
		evaluator.Add(classified, biasCiphertext, classified)
		encryptedBiasCiphertexts = 1
	}
	aggregate := aggregateSecret(params.Parameters, shares) // test verifier only
	decoded := encoder.Decode(ckks.NewDecryptor(params, aggregate).DecryptNew(classified), params.LogSlots())

	var maxError, sumSquared float64
	for class := 0; class < realResNetTailClasses; class++ {
		error := cmplx.Abs(decoded[class] - wantLogits[class])
		maxError = math.Max(maxError, error)
		sumSquared += error * error
		t.Logf("logit[%d]: expected=%.17g observed=%.17g absError=%.9g", class, real(wantLogits[class]), real(decoded[class]), error)
	}
	rmse := math.Sqrt(sumSquared / realResNetTailClasses)
	if maxError >= 1e-5 {
		t.Fatalf("real ResNet-20 tail failed: maxLogitError=%g rmse=%g", maxError, rmse)
	}

	t.Logf("real ResNet-20 GAP+FC tail P2: PASS; semantics=%s dataOwner=%d modelOwner=%d packing=64x8x8 slots=%d encryptedInputCiphertexts=1 encryptedFCWeightCiphertexts=1 encryptedBiasCiphertexts=%d publicMaskCiphertexts=1 rotationKeys=%d rotationCalls=17 ct-ct-products=1 ct-pt-products=1 gapLevel=%d fcProductLevel=%d checkedLogits=%d maxLogitError=%g rmse=%g", semantics, realResNetTailDataOwner, realResNetTailModelOwner, params.Slots(), encryptedBiasCiphertexts, len(rotations), gapLevel, fcProductLevel, realResNetTailClasses, maxError, rmse)
	t.Logf("frozen fc.bin SHA256=%s coefficients=%d ordering=input-channel-major_then_class packing=channelBlock64:first10", weights.sha256, realResNetTailChannels*realResNetTailClasses)
	if bias == nil {
		t.Logf("reference semantics: GAP normalization=1/64 repeatWidth=16 FC bias=OMITTED_BY_FROZEN_CPP_REFERENCE")
	} else {
		t.Logf("checkpoint-faithful extension: GAP normalization=1/64 repeatWidth=16 encrypted FC bias fixture SHA256=%s (kept separate from frozen C++ semantics)", bias.sha256)
	}
	t.Logf("functional-only elapsed=%s (includes key generation, encryption/folding, evaluation, decryption, and oracle checks; not a benchmark)", time.Since(functionalStart))
}

func TestEncryptedRealResNet20GapFCTailP2TwoParty(t *testing.T) {
	runEncryptedRealResNet20TailP2(t, false)
}

func TestEncryptedCheckpointFaithfulResNet20GapFCTailP2TwoParty(t *testing.T) {
	runEncryptedRealResNet20TailP2(t, true)
}
