package mkckks

// This experiment-only test records decoded precision after each complete
// KKLSS multiplication/relinearization under the exact P1/P2 literals.

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"mk-lattigo/mkrlwe"

	"github.com/ldsec/lattigo/v2/ckks"
)

var kklssNoiseHeader = []string{
	"implementation", "scheme", "parameter", "parties", "trial", "depth",
	"level", "correct", "max_decode_error", "precision_bits",
	"max_coeff_noise", "noise_budget_bits", "target_precision_bits",
	"timestamp_utc",
}

func TestConcreteNoiseKKLSSCKKS(t *testing.T) {
	outPath := os.Getenv("NOISE_OUT")
	if outPath == "" {
		t.Skip("set NOISE_OUT to run the concrete KKLSS CKKS noise experiment")
	}
	parameterNames := kklssExperimentStrings("NOISE_PARAMS", []string{"P1", "P2"})
	partyCounts := kklssExperimentInts("NOISE_PARTIES", []int{2, 4, 8, 16, 32, 64})
	trials := kklssExperimentInt("NOISE_TRIALS", 1)
	maxDepth := kklssExperimentInt("NOISE_MAX_DEPTH", 16)
	targetPrecision := float64(kklssExperimentInt("NOISE_TARGET_PRECISION", 20))
	w, closeWriter, err := kklssOpenNoiseCSV(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer closeWriter()

	for _, parameterName := range parameterNames {
		var literal ckks.ParametersLiteral
		switch parameterName {
		case "P1":
			literal = PN14QP439
		case "P2":
			literal = PN15QP880
		default:
			t.Fatalf("unknown parameter %q", parameterName)
		}
		baseParams, err := ckks.NewParametersFromLiteral(literal)
		if err != nil {
			t.Fatal(err)
		}
		params := NewParameters(baseParams)
		for _, parties := range partyCounts {
			for trial := 1; trial <= trials; trial++ {
				t.Run(fmt.Sprintf("%s/n=%d/trial=%d", parameterName, parties, trial), func(t *testing.T) {
					runKKLSSCKKSNoise(t, w, parameterName, params, parties, trial, maxDepth, targetPrecision)
				})
			}
		}
	}
}

func runKKLSSCKKSNoise(t *testing.T, w *csv.Writer, parameterName string, params Parameters, parties, trial, maxDepth int, targetPrecision float64) {
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
	for _, user := range userList {
		msg := NewMessage(params)
		msg.Value[0] = complex(1.0/float64(parties), 0)
		ct := context.encryptor.EncryptMsgNew(msg, context.pkSet.GetPublicKey(user))
		if current == nil {
			current = ct
		} else {
			current = context.evaluator.AddNew(current, ct)
		}
	}

	for depth := 1; depth <= maxDepth && current.Level() > 0; depth++ {
		current = context.evaluator.MulRelinNew(current, current, context.rlkSet)
		decoded := context.decryptor.Decrypt(current, context.skSet)
		maxErr := 0.0
		for i, value := range decoded.Value {
			expected := 0.0
			if i == 0 {
				expected = 1.0
			}
			err := math.Hypot(real(value)-expected, imag(value))
			if err > maxErr {
				maxErr = err
			}
		}
		precision := kklssPrecisionBits(maxErr)
		correct := !math.IsNaN(maxErr) && !math.IsInf(maxErr, 0) && precision >= targetPrecision
		row := []string{
			"KKLSS", "CKKS", parameterName, strconv.Itoa(parties), strconv.Itoa(trial), strconv.Itoa(depth),
			strconv.Itoa(current.Level()), strconv.FormatBool(correct), kklssFormatFloat(maxErr), kklssFormatFloat(precision),
			"", "", kklssFormatFloat(targetPrecision), time.Now().UTC().Format(time.RFC3339Nano),
		}
		if err := w.Write(row); err != nil {
			t.Fatal(err)
		}
		w.Flush()
		if err := w.Error(); err != nil {
			t.Fatal(err)
		}
	}
}

func kklssPrecisionBits(maxErr float64) float64 {
	if maxErr == 0 {
		return math.Inf(1)
	}
	return -math.Log2(maxErr)
}

func kklssFormatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', 12, 64)
}

func kklssOpenNoiseCSV(path string) (*csv.Writer, func(), error) {
	info, statErr := os.Stat(path)
	needsHeader := os.IsNotExist(statErr) || (statErr == nil && info.Size() == 0)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, nil, err
	}
	w := csv.NewWriter(f)
	if needsHeader {
		if err := w.Write(kklssNoiseHeader); err != nil {
			_ = f.Close()
			return nil, nil, err
		}
		w.Flush()
	}
	return w, func() { w.Flush(); _ = f.Close() }, nil
}

func kklssExperimentStrings(name string, defaults []string) []string {
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

func kklssExperimentInts(name string, defaults []int) []int {
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

func kklssExperimentInt(name string, defaultValue int) int {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			panic(err)
		}
		return parsed
	}
	return defaultValue
}
