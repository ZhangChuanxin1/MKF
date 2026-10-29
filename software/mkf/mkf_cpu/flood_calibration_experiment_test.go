package mkfcpu

// This experiment-only test extracts the coefficient-domain phase error
// after one complete MKF multiplication/relinearization. It does not alter
// the online implementation and does not sample a flooding distribution.
// The exported maxima/RMS values are inputs to the separate flooding
// calibration and correctness simulation.

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

var floodCalibrationHeader = []string{
	"implementation", "scheme", "parameter", "parties", "trial",
	"level", "ring_degree", "correct", "decoded_precision_bits",
	"pre_max_coeff", "log2_pre_max_coeff", "log2_pre_rms_coeff",
	"log2_scale_or_delta", "log2_active_q", "timestamp_utc",
}

func TestFloodCalibrationMKF(t *testing.T) {
	outPath := strings.TrimSpace(os.Getenv("FLOOD_CALIB_OUT"))
	if outPath == "" {
		t.Skip("set FLOOD_CALIB_OUT to run the flooding calibration experiment")
	}
	parameters := experimentStrings("FLOOD_CALIB_PARAMS", []string{"P1G", "P2G"})
	parties := experimentInts("FLOOD_CALIB_PARTIES", []int{2, 4, 8, 16, 32, 64})
	schemes := experimentStrings("FLOOD_CALIB_SCHEMES", []string{"CKKS", "BFV"})
	trials := experimentInt("FLOOD_CALIB_TRIALS", 3)

	writer, closeWriter, err := openFloodCalibrationCSV(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer closeWriter()

	for _, scheme := range schemes {
		for _, parameter := range parameters {
			if scheme == "BFV" && strings.HasSuffix(parameter, "G") {
				parameter = strings.TrimSuffix(parameter, "G")
			}
			for _, n := range parties {
				for trial := 1; trial <= trials; trial++ {
					name := fmt.Sprintf("%s/%s/n=%d/trial=%d", scheme, parameter, n, trial)
					t.Run(name, func(t *testing.T) {
						switch scheme {
						case "CKKS":
							runFloodCalibrationCKKS(t, writer, parameter, n, trial)
						case "BFV":
							runFloodCalibrationBFV(t, writer, parameter, n, trial)
						default:
							t.Fatalf("unsupported scheme %q", scheme)
						}
					})
				}
			}
		}
	}
}

func runFloodCalibrationCKKS(t *testing.T, writer *csv.Writer, parameter string, parties, trial int) {
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
	switcher := newSwitchWorkspace(params.Parameters)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})

	guarded := strings.HasSuffix(parameter, "G")
	implementation := "MKF"
	if guarded {
		implementation = "MKF-GUARD"
	}
	inputScale := params.Scale()
	if guarded {
		inputScale *= float64(params.Q()[params.MaxLevel()])
	}
	input := makeCKKSNoiseInput(params, encoder, public, values, inputScale)

	var aggregateCurrent *outputCiphertext
	if guarded {
		folded, foldErr := fold(params.Parameters, input, keys.Fold, true, switcher, &Profile{})
		if foldErr != nil {
			t.Fatal(foldErr)
		}
		guardedCiphertext := ckks.NewCiphertextAtLevelFromPoly(input.Level, folded)
		guardedCiphertext.Scale = input.Scale
		if err := evaluator.Rescale(guardedCiphertext, params.Scale(), guardedCiphertext); err != nil {
			t.Fatal(err)
		}
		aggregateCurrent = &outputCiphertext{
			C0: guardedCiphertext.Value[0], Gamma: guardedCiphertext.Value[1],
			Level: guardedCiphertext.Level(), Scale: guardedCiphertext.Scale,
		}
	} else {
		aggregateCurrent, err = multiplyMKFCKKS(params, parties, keys, switcher, evaluator, input, input)
		if err != nil {
			t.Fatal(err)
		}
	}

	var out *outputCiphertext
	if guarded {
		out, err = multiplyAggregateCKKS(params, keys, switcher, evaluator, aggregateCurrent, aggregateCurrent)
	} else {
		out = aggregateCurrent
	}
	if err != nil {
		t.Fatal(err)
	}

	compact := ckks.NewCiphertextAtLevelFromPoly(out.Level, [2]*ring.Poly{out.C0, out.Gamma})
	compact.Scale = out.Scale
	phase := ckks.NewDecryptor(params, aggregate).DecryptNew(compact)
	decoded := encoder.Decode(phase, params.LogSlots())
	maxDecodedErr := 0.0
	for i, value := range decoded {
		expected := 0.0
		if i == 0 {
			expected = 1.0
		}
		err := math.Hypot(real(value)-expected, imag(value))
		if err > maxDecodedErr {
			maxDecodedErr = err
		}
	}

	expectedValues := make([]complex128, params.Slots())
	expectedValues[0] = 1
	expectedPlaintext := ckks.NewPlaintext(params, out.Level, out.Scale)
	encoder.EncodeNTT(expectedPlaintext, expectedValues, params.LogSlots())
	ringQ := params.RingQ()
	errNTT := ringQ.NewPolyLvl(out.Level)
	errCoeff := ringQ.NewPolyLvl(out.Level)
	ringQ.SubLvl(out.Level, phase.Value, expectedPlaintext.Value, errNTT)
	ringQ.InvNTTLvl(out.Level, errNTT, errCoeff)
	coeffs := centeredCoefficients(ringQ, out.Level, errCoeff)
	maxAbs, log2RMS := coefficientMetrics(coeffs)
	log2Max := math.Inf(-1)
	if maxAbs.Sign() > 0 {
		log2Max = log2Big(maxAbs)
	}
	activeQ := modulusProduct(params.Q()[:out.Level+1])

	writeFloodCalibrationRow(t, writer, []string{
		implementation, "CKKS", parameter, strconv.Itoa(parties), strconv.Itoa(trial),
		strconv.Itoa(out.Level), strconv.Itoa(params.N()),
		strconv.FormatBool(!math.IsNaN(maxDecodedErr) && !math.IsInf(maxDecodedErr, 0) && precisionBits(maxDecodedErr) >= 20.0),
		formatFloat(precisionBits(maxDecodedErr)), maxAbs.String(),
		formatFloat(log2Max), formatFloat(log2RMS), formatFloat(math.Log2(out.Scale)),
		formatFloat(log2Big(activeQ)), time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func runFloodCalibrationBFV(t *testing.T, writer *csv.Writer, parameter string, parties, trial int) {
	params, qMul, err := BFVParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}
	kgen := bfv.NewKeyGenerator(params)
	secret := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	values := make([]uint64, parties)
	values[0] = 1
	for i := range secret {
		secret[i], public[i] = kgen.GenKeyPair()
	}
	aggregate := aggregateSecret(params.Parameters, secret)
	keys := generateOnlineKeys(params.Parameters, kgen, secret, aggregate)
	encoder := bfv.NewEncoder(params)
	input := makeBFVMultiCiphertext(params, encoder, public, values)
	switcher := newSwitchWorkspace(params.Parameters)
	tensor, err := newBFVTensorWorkspace(params, qMul)
	if err != nil {
		t.Fatal(err)
	}
	out, err := multiplyMKFBFV(params, parties, keys, switcher, tensor, input, input)
	if err != nil {
		t.Fatal(err)
	}
	compact := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: []*ring.Poly{out.C0, out.Gamma}}}
	phase := bfv.NewDecryptor(params, aggregate).DecryptNew(compact)
	decoded := encoder.DecodeUintNew(phase)
	correct := decoded[0] == 1
	for i := 1; i < len(decoded) && correct; i++ {
		correct = decoded[i] == 0
	}

	expected := bfv.NewPlaintext(params)
	encoder.EncodeUint([]uint64{1}, expected)
	errPoly := params.RingQ().NewPoly()
	params.RingQ().Sub(phase.Value, expected.Value, errPoly)
	coeffs := centeredCoefficients(params.RingQ(), params.MaxLevel(), errPoly)
	maxAbs, log2RMS := coefficientMetrics(coeffs)
	log2Max := math.Inf(-1)
	if maxAbs.Sign() > 0 {
		log2Max = log2Big(maxAbs)
	}
	q := new(big.Int).Set(params.RingQ().ModulusBigint)
	delta := new(big.Int).Quo(new(big.Int).Set(q), new(big.Int).SetUint64(params.T()))

	writeFloodCalibrationRow(t, writer, []string{
		"MKF", "BFV", parameter, strconv.Itoa(parties), strconv.Itoa(trial),
		strconv.Itoa(params.MaxLevel()), strconv.Itoa(params.N()), strconv.FormatBool(correct), "",
		maxAbs.String(), formatFloat(log2Max), formatFloat(log2RMS),
		formatFloat(log2Big(delta)), formatFloat(log2Big(q)),
		time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func centeredCoefficients(ringQ *ring.Ring, level int, poly *ring.Poly) []*big.Int {
	coeffs := make([]*big.Int, ringQ.N)
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	ringQ.PolyToBigintCenteredLvl(level, poly, coeffs)
	return coeffs
}

func coefficientMetrics(coeffs []*big.Int) (*big.Int, float64) {
	maxAbs := new(big.Int)
	sumSquares := new(big.Int)
	tmp := new(big.Int)
	for _, coeff := range coeffs {
		if coeff.CmpAbs(maxAbs) > 0 {
			maxAbs.Abs(coeff)
		}
		tmp.Mul(coeff, coeff)
		sumSquares.Add(sumSquares, tmp)
	}
	if sumSquares.Sign() == 0 {
		return maxAbs, math.Inf(-1)
	}
	return maxAbs, 0.5 * (log2Big(sumSquares) - math.Log2(float64(len(coeffs))))
}

func modulusProduct(moduli []uint64) *big.Int {
	product := big.NewInt(1)
	for _, modulus := range moduli {
		product.Mul(product, new(big.Int).SetUint64(modulus))
	}
	return product
}

func openFloodCalibrationCSV(path string) (*csv.Writer, func(), error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return nil, nil, err
	}
	writer := csv.NewWriter(file)
	if err := writer.Write(floodCalibrationHeader); err != nil {
		_ = file.Close()
		return nil, nil, err
	}
	writer.Flush()
	return writer, func() { writer.Flush(); _ = file.Close() }, nil
}

func writeFloodCalibrationRow(t *testing.T, writer *csv.Writer, row []string) {
	t.Helper()
	if err := writer.Write(row); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
}
