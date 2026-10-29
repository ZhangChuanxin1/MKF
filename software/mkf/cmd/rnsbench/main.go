package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ldsec/lattigo/v2/bfv"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

type config struct {
	scheme     string
	param      string
	runs       int
	warmup     int
	inner      int
	output     string
	paramsJSON string
}

type operation struct {
	name string
	fn   func()
}

func main() {
	cfg := parseFlags()
	runtime.GOMAXPROCS(1)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	writer, closeOutput, err := csvWriter(cfg.output)
	if err != nil {
		fatalf("open output: %v", err)
	}
	defer closeOutput()

	if err := writer.Write([]string{
		"scheme", "parameter", "logN", "logQP", "q_count", "p_count",
		"operation", "sample", "inner", "elapsed_ns", "ns_per_op",
	}); err != nil {
		fatalf("write CSV header: %v", err)
	}

	for _, scheme := range expand(cfg.scheme, []string{"ckks", "bfv"}) {
		parameterLabels := expand(cfg.param, []string{"P1", "P2"})
		if cfg.paramsJSON != "" {
			parameterLabels = []string{"CUSTOM"}
		}
		for _, parameter := range parameterLabels {
			params, err := parameters(scheme, parameter, cfg.paramsJSON)
			if err != nil {
				fatalf("parameters: %v", err)
			}
			if err := run(writer, scheme, parameter, params, cfg); err != nil {
				fatalf("benchmark: %v", err)
			}
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		fatalf("flush CSV: %v", err)
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.scheme, "scheme", "all", "ckks, bfv, or all")
	flag.StringVar(&cfg.param, "param", "all", "P1, P2, or all")
	flag.IntVar(&cfg.runs, "runs", 30, "number of measured samples")
	flag.IntVar(&cfg.warmup, "warmup", 5, "number of unmeasured warm-up samples")
	flag.IntVar(&cfg.inner, "inner", 1, "operations per measured sample")
	flag.StringVar(&cfg.output, "out", "", "raw CSV path; empty writes to stdout")
	flag.StringVar(&cfg.paramsJSON, "params-json", "", "optional rlwe.ParametersLiteral JSON; overrides built-in P1/P2")
	flag.Parse()

	if cfg.runs < 1 || cfg.warmup < 0 || cfg.inner < 1 {
		fatalf("runs and inner must be positive; warmup must be non-negative")
	}
	cfg.scheme = strings.ToLower(cfg.scheme)
	cfg.param = strings.ToUpper(cfg.param)
	return cfg
}

func parameters(scheme, label, jsonPath string) (rlwe.Parameters, error) {
	if jsonPath != "" {
		file, err := os.Open(jsonPath)
		if err != nil {
			return rlwe.Parameters{}, err
		}
		defer file.Close()
		var literal rlwe.ParametersLiteral
		if err := json.NewDecoder(file).Decode(&literal); err != nil {
			return rlwe.Parameters{}, err
		}
		return rlwe.NewParametersFromLiteral(literal)
	}
	switch scheme {
	case "ckks":
		literal := ckks.PN14QP438
		if label == "P2" {
			literal = ckks.PN15QP880
		} else if label != "P1" {
			return rlwe.Parameters{}, fmt.Errorf("unknown parameter %q", label)
		}
		params, err := ckks.NewParametersFromLiteral(literal)
		return params.Parameters, err
	case "bfv":
		literal := bfv.PN14QP438
		if label == "P2" {
			literal = bfv.PN15QP880
		} else if label != "P1" {
			return rlwe.Parameters{}, fmt.Errorf("unknown parameter %q", label)
		}
		params, err := bfv.NewParametersFromLiteral(literal)
		return params.Parameters, err
	default:
		return rlwe.Parameters{}, fmt.Errorf("unknown scheme %q", scheme)
	}
}

func run(w *csv.Writer, scheme, label string, params rlwe.Parameters, cfg config) error {
	prng, err := utils.NewPRNG()
	if err != nil {
		return err
	}

	ringQ := params.RingQ()
	ringP := params.RingP()
	ringQP := params.RingQP()
	levelQ := params.QCount() - 1
	levelP := params.PCount() - 1
	alpha := params.PCount()

	coefficient := ringQ.NewPoly()
	nttInput := ringQ.NewPoly()
	nttOutput := ringQ.NewPoly()
	product := ringQ.NewPoly()
	ring.NewUniformSampler(prng, ringQ).Read(coefficient)
	ringQ.NTTLvl(levelQ, coefficient, nttInput)
	nttInput.IsNTT = true

	keygen := rlwe.NewKeyGenerator(params)
	skIn := keygen.GenSecretKey()
	skOut := keygen.GenSecretKey()
	switchingKey := keygen.GenSwitchingKey(skIn, skOut)
	keySwitcher := rlwe.NewKeySwitcher(params)

	decomposition := make([]rlwe.PolyQP, params.Beta())
	for i := range decomposition {
		decomposition[i] = ringQP.NewPoly()
	}
	keySwitcher.DecomposeNTT(levelQ, levelP, alpha, nttInput, decomposition)

	c0Q := ringQ.NewPoly()
	c1Q := ringQ.NewPoly()
	c0P := ringP.NewPoly()
	c1P := ringP.NewPoly()
	modDownOut := ringQ.NewPoly()
	fullOut0 := ringQ.NewPoly()
	fullOut1 := ringQ.NewPoly()

	// Seed the QP accumulators before timing ModDown in isolation.
	keySwitcher.KeyswitchHoistedNoModDown(
		levelQ, decomposition, switchingKey, c0Q, c1Q, c0P, c1P,
	)

	operations := []operation{
		{"ntt_q", func() { ringQ.NTTLvl(levelQ, coefficient, nttOutput) }},
		{"intt_q", func() { ringQ.InvNTTLvl(levelQ, nttInput, nttOutput) }},
		{"pointwise_mul_q", func() {
			ringQ.MulCoeffsMontgomeryLvl(levelQ, nttInput, nttInput, product)
		}},
		{"decompose_q_to_qp", func() {
			keySwitcher.DecomposeNTT(levelQ, levelP, alpha, nttInput, decomposition)
		}},
		{"external_product_hoisted_no_moddown", func() {
			keySwitcher.KeyswitchHoistedNoModDown(
				levelQ, decomposition, switchingKey, c0Q, c1Q, c0P, c1P,
			)
		}},
		{"moddown_qp_to_q_one_poly", func() {
			keySwitcher.Baseconverter.ModDownQPtoQNTT(
				levelQ, levelP, c0Q, c0P, modDownOut,
			)
		}},
		{"keyswitch_full_two_outputs", func() {
			keySwitcher.SwitchKeysInPlace(levelQ, nttInput, switchingKey, fullOut0, fullOut1)
		}},
	}

	for _, op := range operations {
		for i := 0; i < cfg.warmup; i++ {
			repeat(cfg.inner, op.fn)
		}
		for sample := 0; sample < cfg.runs; sample++ {
			start := time.Now()
			repeat(cfg.inner, op.fn)
			elapsed := time.Since(start).Nanoseconds()
			record := []string{
				scheme,
				label,
				strconv.Itoa(params.LogN()),
				strconv.Itoa(params.LogQP()),
				strconv.Itoa(params.QCount()),
				strconv.Itoa(params.PCount()),
				op.name,
				strconv.Itoa(sample),
				strconv.Itoa(cfg.inner),
				strconv.FormatInt(elapsed, 10),
				strconv.FormatFloat(float64(elapsed)/float64(cfg.inner), 'f', 3, 64),
			}
			if err := w.Write(record); err != nil {
				return err
			}
		}
	}

	return nil
}

func repeat(count int, fn func()) {
	for i := 0; i < count; i++ {
		fn()
	}
}

func expand(value string, all []string) []string {
	if strings.EqualFold(value, "all") {
		return all
	}
	return []string{value}
}

func csvWriter(path string) (*csv.Writer, func(), error) {
	if path == "" {
		return csv.NewWriter(os.Stdout), func() {}, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, nil, err
	}
	return csv.NewWriter(file), func() { _ = file.Close() }, nil
}

func fatalf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
