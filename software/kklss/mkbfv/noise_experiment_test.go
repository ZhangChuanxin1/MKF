package mkbfv

// This experiment-only test records coefficient-wise invariant BFV noise
// after each complete KKLSS multiplication/relinearization.

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

	"mk-lattigo/mkrlwe"

	"github.com/ldsec/lattigo/v2/bfv"
)

var kklssBFVNoiseHeader = []string{
	"implementation", "scheme", "parameter", "parties", "trial", "depth",
	"level", "correct", "max_decode_error", "precision_bits",
	"max_coeff_noise", "noise_budget_bits", "target_precision_bits",
	"timestamp_utc",
}

func TestConcreteNoiseKKLSSBFV(t *testing.T) {
	outPath := os.Getenv("NOISE_OUT")
	if outPath == "" {
		t.Skip("set NOISE_OUT to run the concrete KKLSS BFV noise experiment")
	}
	parameterNames := kklssBFVExperimentStrings("NOISE_PARAMS", []string{"P1", "P2"})
	partyCounts := kklssBFVExperimentInts("NOISE_PARTIES", []int{2, 4, 8, 16, 32, 64})
	trials := kklssBFVExperimentInt("NOISE_TRIALS", 1)
	maxDepth := kklssBFVExperimentInt("NOISE_MAX_DEPTH", 16)
	targetPrecision := float64(kklssBFVExperimentInt("NOISE_TARGET_PRECISION", 20))
	w, closeWriter, err := kklssBFVOpenNoiseCSV(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer closeWriter()

	for _, parameterName := range parameterNames {
		var literal ParametersLiteral
		switch parameterName {
		case "P1":
			literal = PN14QP439
		case "P2":
			literal = PN15QP880
		default:
			t.Fatalf("unknown parameter %q", parameterName)
		}
		params := NewParametersFromLiteral(literal)
		for _, parties := range partyCounts {
			for trial := 1; trial <= trials; trial++ {
				t.Run(fmt.Sprintf("%s/n=%d/trial=%d", parameterName, parties, trial), func(t *testing.T) {
					runKKLSSBFVNoise(t, w, parameterName, params, parties, trial, maxDepth, targetPrecision)
				})
			}
		}
	}
}

func runKKLSSBFVNoise(t *testing.T, w *csv.Writer, parameterName string, params Parameters, parties, trial, maxDepth int, targetPrecision float64) {
	userList := make([]string, parties)
	idset := mkrlwe.NewIDSet()
	for i := range userList {
		userList[i] = fmt.Sprintf("user%03d", i)
		idset.Add(userList[i])
	}
	context, err := genTestParams(params, idset)
	if err != nil {
		t.Fatal(err)
	}
	var current *Ciphertext
	for i, user := range userList {
		msg := NewMessage(params)
		if i == 0 {
			msg.Value[0] = 1
		}
		ct := context.encryptor.EncryptMsgNew(msg, context.pkSet.GetPublicKey(user))
		if current == nil {
			current = ct
		} else {
			current = context.evaluator.AddNew(current, ct)
		}
	}
	baseParams, err := bfv.NewParameters(params.Parameters.Parameters, params.T())
	if err != nil {
		t.Fatal(err)
	}
	encoder := bfv.NewEncoder(baseParams)
	metricsDepth1Only := kklssBFVExperimentBool("NOISE_METRICS_DEPTH1_ONLY", false)

	for depth := 1; depth <= maxDepth; depth++ {
		current = context.evaluator.MulRelinNew(current, current, context.rlkSet)
		decoded := context.decryptor.Decrypt(current, context.skSet)
		correct := decoded.Value[0] == 1
		for i := 1; i < len(decoded.Value) && correct; i++ {
			correct = decoded.Value[i] == 0
		}
		maxNoiseString, budgetString := "", ""
		if !metricsDepth1Only || depth == 1 {
			phase := bfv.NewPlaintext(baseParams)
			context.decryptor.Decryptor.Decrypt(current.Ciphertext, context.skSet, phase.Plaintext)
			maxNoise, budget := kklssBFVNoiseMetrics(params, baseParams, phase, encoder)
			maxNoiseString = maxNoise.String()
			budgetString = kklssBFVFormatFloat(budget)
		}
		row := []string{
			"KKLSS", "BFV", parameterName, strconv.Itoa(parties), strconv.Itoa(trial), strconv.Itoa(depth),
			strconv.Itoa(current.Level()), strconv.FormatBool(correct), "", "",
			maxNoiseString, budgetString, kklssBFVFormatFloat(targetPrecision), time.Now().UTC().Format(time.RFC3339Nano),
		}
		if err := w.Write(row); err != nil {
			t.Fatal(err)
		}
		w.Flush()
		if err := w.Error(); err != nil {
			t.Fatal(err)
		}
		if !correct {
			break
		}
	}
}

func kklssBFVExperimentBool(name string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func kklssBFVNoiseMetrics(params Parameters, baseParams bfv.Parameters, phase *bfv.Plaintext, encoder bfv.Encoder) (*big.Int, float64) {
	expected := bfv.NewPlaintext(baseParams)
	encoder.EncodeUint([]uint64{1}, expected)
	errPoly := params.RingQ().NewPoly()
	params.RingQ().Sub(phase.Value, expected.Value, errPoly)
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
	budget := kklssBFVLog2Big(delta) - 1 - kklssBFVLog2Big(maxAbs)
	return maxAbs, budget
}

func kklssBFVLog2Big(v *big.Int) float64 {
	if v.Sign() <= 0 {
		return math.Inf(-1)
	}
	bits := v.BitLen()
	shift := bits - 53
	top := new(big.Int).Set(v)
	if shift > 0 {
		top.Rsh(top, uint(shift))
	}
	return math.Log2(float64(top.Uint64())) + float64(kklssBFVMaxInt(shift, 0))
}

func kklssBFVMaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func kklssBFVFormatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', 12, 64)
}

func kklssBFVOpenNoiseCSV(path string) (*csv.Writer, func(), error) {
	info, statErr := os.Stat(path)
	needsHeader := os.IsNotExist(statErr) || (statErr == nil && info.Size() == 0)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, nil, err
	}
	w := csv.NewWriter(f)
	if needsHeader {
		if err := w.Write(kklssBFVNoiseHeader); err != nil {
			_ = f.Close()
			return nil, nil, err
		}
		w.Flush()
	}
	return w, func() { w.Flush(); _ = f.Close() }, nil
}

func kklssBFVExperimentStrings(name string, defaults []string) []string {
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

func kklssBFVExperimentInts(name string, defaults []int) []int {
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

func kklssBFVExperimentInt(name string, defaultValue int) int {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			panic(err)
		}
		return parsed
	}
	return defaultValue
}
