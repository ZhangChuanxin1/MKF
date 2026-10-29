package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	mkfcpu "mkf-cpu-experiments/mkf_cpu"
)

var header = []string{
	"scheme", "parameter", "parties", "run", "warmup", "gomaxprocs",
	"total_ns", "tensor_ns", "decomposition_ns", "external_product_ns",
	"modup_moddown_rescale_ns", "memory_copy_ns", "other_ns",
	"decomposition_count", "external_product_poly_count",
	"validation_passed", "expected", "observed", "max_error",
	"go_version", "goos", "goarch", "timestamp_utc",
}

func main() {
	scheme := flag.String("scheme", "all", "ckks, bfv, or all")
	parameter := flag.String("param", "P1", "P1 or P2")
	parties := flag.Int("parties", 16, "number of parties")
	partyList := flag.String("party-list", "", "comma-separated party counts; overrides -parties")
	warmup := flag.Int("warmup", 5, "warm-up iterations")
	runs := flag.Int("runs", 10, "measured iterations")
	gcBetween := flag.Bool("gc-between", true, "run GC before each untimed invocation")
	out := flag.String("out", "mkf_cpu_raw.csv", "output CSV")
	flag.Parse()

	if *parties < 1 || *warmup < 0 || *runs < 1 {
		fatalf("invalid parties/warmup/runs")
	}
	partyCounts, err := parsePartyCounts(*parties, *partyList)
	if err != nil {
		fatalf("party list: %v", err)
	}
	schemes := []string{strings.ToLower(*scheme)}
	if schemes[0] == "all" {
		schemes = []string{"ckks", "bfv"}
	}

	parameters := []string{strings.ToUpper(*parameter)}
	if strings.EqualFold(*parameter, "all") {
		parameters = []string{"P1", "P2"}
	}

	f, err := os.Create(*out)
	if err != nil {
		fatalf("create output: %v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write(header); err != nil {
		fatalf("write header: %v", err)
	}

	for _, parameterName := range parameters {
		for _, partyCount := range partyCounts {
			for _, name := range schemes {
				switch name {
				case "ckks":
					runCKKS(w, parameterName, partyCount, *warmup, *runs, *gcBetween)
				case "bfv":
					runBFV(w, parameterName, partyCount, *warmup, *runs, *gcBetween)
				default:
					fatalf("unknown scheme %q", name)
				}
				runtime.GC()
			}
		}
	}
	if err := w.Error(); err != nil {
		fatalf("write CSV: %v", err)
	}
}

func parsePartyCounts(single int, list string) ([]int, error) {
	if strings.TrimSpace(list) == "" {
		return []int{single}, nil
	}
	seen := make(map[int]bool)
	var result []int
	for _, token := range strings.Split(list, ",") {
		value, err := strconv.Atoi(strings.TrimSpace(token))
		if err != nil || value < 1 {
			return nil, fmt.Errorf("invalid value %q", token)
		}
		if !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	sort.Ints(result)
	return result, nil
}

func runCKKS(w *csv.Writer, parameter string, parties, warmup, runs int, gcBetween bool) {
	fmt.Printf("setup MKF-CKKS %s n=%d (offline keys/encryption excluded)...\n", parameter, parties)
	c, err := mkfcpu.NewCKKSCase(parameter, parties)
	if err != nil {
		fatalf("CKKS setup: %v", err)
	}
	runCase(w, "MKF-CKKS", parameter, parties, warmup, runs, gcBetween, c.Run)
}

func runBFV(w *csv.Writer, parameter string, parties, warmup, runs int, gcBetween bool) {
	fmt.Printf("setup MKF-BFV %s n=%d (offline keys/encryption excluded)...\n", parameter, parties)
	c, err := mkfcpu.NewBFVCase(parameter, parties)
	if err != nil {
		fatalf("BFV setup: %v", err)
	}
	runCase(w, "MKF-BFV", parameter, parties, warmup, runs, gcBetween, c.Run)
}

func runCase(w *csv.Writer, scheme, parameter string, parties, warmup, runs int, gcBetween bool, f func() (mkfcpu.Profile, mkfcpu.Validation, error)) {
	for i := 0; i < warmup; i++ {
		if gcBetween {
			runtime.GC()
		}
		profile, validation, err := f()
		if err != nil {
			fatalf("%s warmup %d: %v", scheme, i+1, err)
		}
		if !validation.Passed {
			fatalf("%s warmup %d validation failed: expected %.10g, observed %.10g, error %.4g", scheme, i+1, validation.Expected, validation.Observed, validation.MaxError)
		}
		fmt.Printf("  warmup %d/%d: %.3f ms\n", i+1, warmup, ms(profile.Total))
	}
	for i := 0; i < runs; i++ {
		if gcBetween {
			runtime.GC()
		}
		profile, validation, err := f()
		if err != nil {
			fatalf("%s run %d: %v", scheme, i+1, err)
		}
		row := []string{
			scheme, parameter, strconv.Itoa(parties), strconv.Itoa(i + 1), strconv.Itoa(warmup), strconv.Itoa(runtime.GOMAXPROCS(0)),
			strconv.FormatInt(profile.Total.Nanoseconds(), 10),
			strconv.FormatInt(profile.Tensor.Nanoseconds(), 10),
			strconv.FormatInt(profile.Decomposition.Nanoseconds(), 10),
			strconv.FormatInt(profile.External.Nanoseconds(), 10),
			strconv.FormatInt(profile.Modulus.Nanoseconds(), 10),
			strconv.FormatInt(profile.Memory.Nanoseconds(), 10),
			strconv.FormatInt(profile.Other.Nanoseconds(), 10),
			strconv.Itoa(profile.Decompositions), strconv.Itoa(profile.ExternalProductPolys),
			strconv.FormatBool(validation.Passed),
			strconv.FormatFloat(validation.Expected, 'g', 12, 64),
			strconv.FormatFloat(validation.Observed, 'g', 12, 64),
			strconv.FormatFloat(validation.MaxError, 'g', 12, 64),
			runtime.Version(), runtime.GOOS, runtime.GOARCH, time.Now().UTC().Format(time.RFC3339Nano),
		}
		if err := w.Write(row); err != nil {
			fatalf("write row: %v", err)
		}
		w.Flush()
		if err := w.Error(); err != nil {
			fatalf("flush row: %v", err)
		}
		fmt.Printf("  run %d/%d: %.3f ms, valid=%t, error=%.4g\n", i+1, runs, ms(profile.Total), validation.Passed, validation.MaxError)
		if !validation.Passed {
			fatalf("%s run %d validation failed", scheme, i+1)
		}
	}
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func fatalf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
