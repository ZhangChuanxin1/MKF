package mkfcpu

// This file is an experiment harness, not part of the online implementation.
// It measures decoded CKKS precision and coefficient-wise BFV invariant noise
// after each complete aggregate-domain multiplication/relinearization.

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/big"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/bfv"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

var noiseHeader = []string{
	"implementation", "scheme", "parameter", "parties", "trial", "depth",
	"level", "correct", "max_decode_error", "precision_bits",
	"max_coeff_noise", "noise_budget_bits", "target_precision_bits",
	"timestamp_utc",
}

func TestConcreteNoiseMKF(t *testing.T) {
	outPath := os.Getenv("NOISE_OUT")
	if outPath == "" {
		t.Skip("set NOISE_OUT to run the concrete MKF noise experiment")
	}
	params := experimentStrings("NOISE_PARAMS", []string{"P1", "P2"})
	parties := experimentInts("NOISE_PARTIES", []int{2, 4, 8, 16, 32, 64})
	schemes := experimentStrings("NOISE_SCHEMES", []string{"CKKS", "BFV"})
	trials := experimentInt("NOISE_TRIALS", 1)
	maxDepth := experimentInt("NOISE_MAX_DEPTH", 16)
	targetPrecision := float64(experimentInt("NOISE_TARGET_PRECISION", 20))

	w, closeWriter, err := openNoiseCSV(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer closeWriter()

	for _, parameter := range params {
		for _, n := range parties {
			for trial := 1; trial <= trials; trial++ {
				for _, scheme := range schemes {
					switch strings.ToUpper(scheme) {
					case "CKKS":
						t.Run(fmt.Sprintf("CKKS/%s/n=%d/trial=%d", parameter, n, trial), func(t *testing.T) {
							runMKFCKKSNoise(t, w, parameter, n, trial, maxDepth, targetPrecision)
						})
					case "BFV":
						t.Run(fmt.Sprintf("BFV/%s/n=%d/trial=%d", parameter, n, trial), func(t *testing.T) {
							runMKFBFVNoise(t, w, parameter, n, trial, maxDepth, targetPrecision)
						})
					default:
						t.Fatalf("unknown scheme %q", scheme)
					}
				}
			}
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		t.Fatal(err)
	}
}

func runMKFCKKSNoise(t *testing.T, w *csv.Writer, parameter string, parties, trial, maxDepth int, targetPrecision float64) {
	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	secret := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	values := make([]float64, parties)
	for i := range secret {
		secret[i], public[i] = kgen.GenKeyPair()
		values[i] = 1.0 / float64(parties)
	}
	aggregate := aggregateSecret(params.Parameters, secret)
	keys := generateOnlineKeys(params.Parameters, kgen, secret, aggregate)
	encoder := ckks.NewEncoder(params)
	chainMode := strings.ToLower(strings.TrimSpace(os.Getenv("NOISE_CKKS_CHAIN_MODE")))
	guardedChain := chainMode == "guard"
	inputScale := params.Scale()
	if guardedChain {
		inputScale *= float64(params.Q()[params.MaxLevel()])
	}
	input := makeCKKSNoiseInput(params, encoder, public, values, inputScale)
	switcher := newSwitchWorkspace(params.Parameters)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})

	current := input
	var aggregateCurrent *outputCiphertext
	compactChain := chainMode == "compact" || guardedChain
	implementation := "MKF"
	if guardedChain {
		implementation = "MKF-GUARD"
		folded, foldErr := fold(params.Parameters, input, keys.Fold, true, switcher, &Profile{})
		if foldErr != nil {
			t.Fatal(foldErr)
		}
		guarded := ckks.NewCiphertextAtLevelFromPoly(input.Level, folded)
		guarded.Scale = input.Scale
		if rescaleErr := evaluator.Rescale(guarded, params.Scale(), guarded); rescaleErr != nil {
			t.Fatal(rescaleErr)
		}
		aggregateCurrent = &outputCiphertext{
			C0: guarded.Value[0], Gamma: guarded.Value[1],
			Level: guarded.Level(), Scale: guarded.Scale,
		}
		current.Level = guarded.Level()
	} else if compactChain {
		implementation = "MKF-AGG"
	}
	for depth := 1; depth <= maxDepth && current.Level > 0; depth++ {
		var out *outputCiphertext
		if compactChain && aggregateCurrent != nil {
			out, err = multiplyAggregateCKKS(params, keys, switcher, evaluator, aggregateCurrent, aggregateCurrent)
		} else {
			out, err = multiplyMKFCKKS(params, parties, keys, switcher, evaluator, current, current)
		}
		if err != nil {
			t.Fatalf("depth %d: %v", depth, err)
		}
		compact := ckks.NewCiphertextAtLevelFromPoly(out.Level, [2]*ring.Poly{out.C0, out.Gamma})
		compact.Scale = out.Scale
		plaintext := ckks.NewDecryptor(params, aggregate).DecryptNew(compact)
		decoded := encoder.Decode(plaintext, params.LogSlots())
		maxErr := 0.0
		for i, value := range decoded {
			expected := 0.0
			if i == 0 {
				expected = 1.0
			}
			err := math.Hypot(real(value)-expected, imag(value))
			if err > maxErr {
				maxErr = err
			}
		}
		precision := precisionBits(maxErr)
		correct := !math.IsNaN(maxErr) && !math.IsInf(maxErr, 0) && precision >= targetPrecision
		writeNoiseRow(t, w, []string{
			implementation, "CKKS", parameter, strconv.Itoa(parties), strconv.Itoa(trial), strconv.Itoa(depth),
			strconv.Itoa(out.Level), strconv.FormatBool(correct), formatFloat(maxErr), formatFloat(precision),
			"", "", formatFloat(targetPrecision), time.Now().UTC().Format(time.RFC3339Nano),
		})
		if compactChain {
			aggregateCurrent = out
			current.Level = out.Level
		} else {
			current = outputToMulti(out)
		}
	}
}

func makeCKKSNoiseInput(params ckks.Parameters, encoder ckks.Encoder, public []*rlwe.PublicKey, values []float64, scale float64) *multiCiphertext {
	level := params.MaxLevel()
	out := &multiCiphertext{
		C0: params.RingQ().NewPolyLvl(level), Components: make([]*ring.Poly, len(public)),
		Level: level, Scale: scale,
	}
	out.C0.IsNTT = true
	for i := range public {
		slots := []complex128{complex(values[i], 0)}
		plaintext := ckks.NewPlaintext(params, level, scale)
		encoder.EncodeNTT(plaintext, slots, params.LogSlots())
		ciphertext := ckks.NewEncryptor(params, public[i]).EncryptNew(plaintext)
		params.RingQ().AddLvl(level, out.C0, ciphertext.Value[0], out.C0)
		out.Components[i] = ciphertext.Value[1].CopyNew()
		out.Components[i].IsNTT = true
	}
	return out
}

// multiplyAggregateCKKS keeps an already folded ciphertext under the
// aggregate secret s_E. It avoids materializing n identical Gamma components
// and folding them again before every subsequent multiplication.
func multiplyAggregateCKKS(params ckks.Parameters, keys onlineKeys, switcher *switchWorkspace, evaluator ckks.Evaluator, left, right *outputCiphertext) (*outputCiphertext, error) {
	if left.Level != right.Level {
		return nil, fmt.Errorf("aggregate level mismatch: %d != %d", left.Level, right.Level)
	}
	level := left.Level
	ringQ := params.RingQ()
	mu0, mu1, mu2 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftM0, leftM1 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftSum, rightSum := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	ringQ.MFormLvl(level, left.C0, leftM0)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM0, right.C0, mu0)
	ringQ.MFormLvl(level, left.Gamma, leftM1)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM1, right.Gamma, mu2)
	ringQ.AddLvl(level, left.C0, left.Gamma, leftSum)
	ringQ.AddLvl(level, right.C0, right.Gamma, rightSum)
	ringQ.MFormLvl(level, leftSum, leftSum)
	ringQ.MulCoeffsMontgomeryLvl(level, leftSum, rightSum, mu1)
	ringQ.SubLvl(level, mu1, mu0, mu1)
	ringQ.SubLvl(level, mu1, mu2, mu1)
	mu0.IsNTT, mu1.IsNTT, mu2.IsNTT = true, true, true

	prof := Profile{}
	relin0, relin1 := switcher.switchOne(params.Parameters, level, mu2, keys.Relin.Keys[0], true, &prof)
	ringQ.AddLvl(level, mu0, relin0, mu0)
	ringQ.AddLvl(level, mu1, relin1, mu1)
	compact := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{mu0, mu1})
	compact.Scale = left.Scale * right.Scale
	if err := evaluator.Rescale(compact, params.Scale(), compact); err != nil {
		return nil, err
	}
	return &outputCiphertext{
		C0: compact.Value[0], Gamma: compact.Value[1],
		Level: compact.Level(), Scale: compact.Scale,
	}, nil
}

func multiplyMKFCKKS(params ckks.Parameters, parties int, keys onlineKeys, switcher *switchWorkspace, evaluator ckks.Evaluator, leftIn, rightIn *multiCiphertext) (*outputCiphertext, error) {
	if leftIn.Level != rightIn.Level {
		return nil, fmt.Errorf("level mismatch: %d != %d", leftIn.Level, rightIn.Level)
	}
	prof := Profile{}
	left, err := fold(params.Parameters, leftIn, keys.Fold, true, switcher, &prof)
	if err != nil {
		return nil, err
	}
	right, err := fold(params.Parameters, rightIn, keys.Fold, true, switcher, &prof)
	if err != nil {
		return nil, err
	}
	level := leftIn.Level
	ringQ := params.RingQ()
	mu0, mu1, mu2 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftM0, leftM1 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftSum, rightSum := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	ringQ.MFormLvl(level, left[0], leftM0)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM0, right[0], mu0)
	ringQ.MFormLvl(level, left[1], leftM1)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM1, right[1], mu2)
	ringQ.AddLvl(level, left[0], left[1], leftSum)
	ringQ.AddLvl(level, right[0], right[1], rightSum)
	ringQ.MFormLvl(level, leftSum, leftSum)
	ringQ.MulCoeffsMontgomeryLvl(level, leftSum, rightSum, mu1)
	ringQ.SubLvl(level, mu1, mu0, mu1)
	ringQ.SubLvl(level, mu1, mu2, mu1)
	mu0.IsNTT, mu1.IsNTT, mu2.IsNTT = true, true, true

	relin0, relin1 := switcher.switchOne(params.Parameters, level, mu2, keys.Relin.Keys[0], true, &prof)
	ringQ.AddLvl(level, mu0, relin0, mu0)
	ringQ.AddLvl(level, mu1, relin1, mu1)
	compact := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{mu0, mu1})
	compact.Scale = leftIn.Scale * rightIn.Scale
	if err := evaluator.Rescale(compact, params.Scale(), compact); err != nil {
		return nil, err
	}
	return materialize(params.Parameters, compact.Value[0], compact.Value[1], parties, compact.Level(), compact.Scale, &prof), nil
}

func runMKFBFVNoise(t *testing.T, w *csv.Writer, parameter string, parties, trial, maxDepth int, targetPrecision float64) {
	params, qMul, err := BFVParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	kgen := bfv.NewKeyGenerator(params)
	secret := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	values := make([]uint64, parties)
	for i := range secret {
		secret[i], public[i] = kgen.GenKeyPair()
	}
	values[0] = 1
	aggregate := aggregateSecret(params.Parameters, secret)
	keys := generateOnlineKeys(params.Parameters, kgen, secret, aggregate)
	encoder := bfv.NewEncoder(params)
	current := makeBFVMultiCiphertext(params, encoder, public, values)
	switcher := newSwitchWorkspace(params.Parameters)
	tensor, err := newBFVTensorWorkspace(params, qMul)
	if err != nil {
		t.Fatal(err)
	}

	for depth := 1; depth <= maxDepth; depth++ {
		out, err := multiplyMKFBFV(params, parties, keys, switcher, tensor, current, current)
		if err != nil {
			t.Fatalf("depth %d: %v", depth, err)
		}
		compact := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: []*ring.Poly{out.C0, out.Gamma}}}
		phase := bfv.NewDecryptor(params, aggregate).DecryptNew(compact)
		decoded := encoder.DecodeUintNew(phase)
		correct := decoded[0] == 1
		for i := 1; i < len(decoded) && correct; i++ {
			correct = decoded[i] == 0
		}
		maxNoise, budget := bfvNoiseMetrics(params, phase, encoder, 1)
		writeNoiseRow(t, w, []string{
			"MKF", "BFV", parameter, strconv.Itoa(parties), strconv.Itoa(trial), strconv.Itoa(depth),
			strconv.Itoa(out.Level), strconv.FormatBool(correct), "", "",
			maxNoise.String(), formatFloat(budget), formatFloat(targetPrecision), time.Now().UTC().Format(time.RFC3339Nano),
		})
		current = outputToMulti(out)
		if !correct {
			break
		}
	}
}

func multiplyMKFBFV(params bfv.Parameters, parties int, keys onlineKeys, switcher *switchWorkspace, tensor *bfvTensorWorkspace, leftIn, rightIn *multiCiphertext) (*outputCiphertext, error) {
	prof := Profile{}
	left, err := fold(params.Parameters, leftIn, keys.Fold, false, switcher, &prof)
	if err != nil {
		return nil, err
	}
	right, err := fold(params.Parameters, rightIn, keys.Fold, false, switcher, &prof)
	if err != nil {
		return nil, err
	}
	tensor.modUpLeft(left)
	tensor.rescaleRight(right)
	tensor.tensorKaratsuba()
	product := tensor.quantize()
	level := params.MaxLevel()
	relin0, relin1 := switcher.switchOne(params.Parameters, level, product[2], keys.Relin.Keys[0], false, &prof)
	params.RingQ().AddLvl(level, product[0], relin0, product[0])
	params.RingQ().AddLvl(level, product[1], relin1, product[1])
	return materialize(params.Parameters, product[0], product[1], parties, level, 0, &prof), nil
}

func outputToMulti(out *outputCiphertext) *multiCiphertext {
	return &multiCiphertext{C0: out.C0, Components: out.Components, Level: out.Level, Scale: out.Scale}
}

func bfvNoiseMetrics(params bfv.Parameters, phase *bfv.Plaintext, encoder bfv.Encoder, expected uint64) (*big.Int, float64) {
	expectedPT := bfv.NewPlaintext(params)
	encoder.EncodeUint([]uint64{expected}, expectedPT)
	errPoly := params.RingQ().NewPoly()
	params.RingQ().Sub(phase.Value, expectedPT.Value, errPoly)
	coeffs := make([]*big.Int, params.N())
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	params.RingQ().PolyToBigintCenteredLvl(params.MaxLevel(), errPoly, coeffs)
	maxAbs := new(big.Int)
	for _, coeff := range coeffs {
		if coeff.CmpAbs(maxAbs) > 0 {
			maxAbs.Abs(coeff)
		}
	}
	if maxAbs.Sign() == 0 {
		return maxAbs, math.Inf(1)
	}
	delta := new(big.Int).Quo(new(big.Int).Set(params.RingQ().ModulusBigint), new(big.Int).SetUint64(params.T()))
	budget := log2Big(delta) - 1 - log2Big(maxAbs)
	return maxAbs, budget
}

func log2Big(v *big.Int) float64 {
	if v.Sign() <= 0 {
		return math.Inf(-1)
	}
	bits := v.BitLen()
	shift := bits - 53
	top := new(big.Int).Set(v)
	if shift > 0 {
		top.Rsh(top, uint(shift))
	}
	mantissa := float64(top.Uint64())
	return math.Log2(mantissa) + float64(maxInt(shift, 0))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func precisionBits(maxErr float64) float64 {
	if maxErr == 0 {
		return math.Inf(1)
	}
	return -math.Log2(maxErr)
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', 12, 64)
}

func openNoiseCSV(path string) (*csv.Writer, func(), error) {
	info, statErr := os.Stat(path)
	needsHeader := os.IsNotExist(statErr) || (statErr == nil && info.Size() == 0)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, nil, err
	}
	w := csv.NewWriter(f)
	if needsHeader {
		if err := w.Write(noiseHeader); err != nil {
			f.Close()
			return nil, nil, err
		}
		w.Flush()
	}
	return w, func() { w.Flush(); _ = f.Close() }, nil
}

func writeNoiseRow(t *testing.T, w *csv.Writer, row []string) {
	t.Helper()
	if err := w.Write(row); err != nil {
		t.Fatal(err)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		t.Fatal(err)
	}
}

func experimentStrings(name string, defaults []string) []string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		parts := strings.Split(value, ",")
		out := make([]string, 0, len(parts))
		for _, part := range parts {
			out = append(out, strings.ToUpper(strings.TrimSpace(part)))
		}
		return out
	}
	return defaults
}

func experimentInts(name string, defaults []int) []int {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		parts := strings.Split(value, ",")
		out := make([]int, 0, len(parts))
		for _, part := range parts {
			parsed, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				panic(err)
			}
			out = append(out, parsed)
		}
		return out
	}
	return defaults
}

func experimentInt(name string, defaultValue int) int {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			panic(err)
		}
		return parsed
	}
	return defaultValue
}
