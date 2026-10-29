// Package stageprof provides low-overhead, exclusive stage timing for the
// existing KKLSS and MKF CPU implementations. It deliberately does not know
// anything about either implementation; callers place Measure calls around
// the real arithmetic boundaries used by the paper.
package stageprof

import (
	"encoding/csv"
	"fmt"
	"io"
	"sort"
	"strconv"
	"sync"
	"time"
)

const (
	TensorProduct     = "tensor_product"
	Decomposition     = "gadget_decomposition"
	ExternalProduct   = "external_product_accumulation"
	NTTINTT           = "ntt_intt"
	ModulusManagement = "modup_moddown_rescale"
	DataMovement      = "memory_data_movement"
	Other             = "other_synchronization"
)

// Sample is one exclusive timing observation.
type Sample struct {
	Scheme    string
	Parameter string
	Parties   int
	Trial     int
	Stage     string
	Elapsed   time.Duration
}

// Profiler is safe for concurrent recording, although the paper's CPU-ST
// experiment should execute the measured implementation with GOMAXPROCS=1.
type Profiler struct {
	mu      sync.Mutex
	samples []Sample
}

// Measure records the duration of fn. Do not nest Measure calls: the paper's
// breakdown requires mutually exclusive stages whose sum equals total time.
func (p *Profiler) Measure(meta Sample, fn func() error) error {
	start := time.Now()
	err := fn()
	meta.Elapsed = time.Since(start)
	p.mu.Lock()
	p.samples = append(p.samples, meta)
	p.mu.Unlock()
	return err
}

// Add records a duration obtained from an existing timer or hardware counter.
func (p *Profiler) Add(sample Sample) {
	p.mu.Lock()
	p.samples = append(p.samples, sample)
	p.mu.Unlock()
}

// Samples returns a stable copy sorted by trial and stage.
func (p *Profiler) Samples() []Sample {
	p.mu.Lock()
	result := append([]Sample(nil), p.samples...)
	p.mu.Unlock()
	sort.Slice(result, func(i, j int) bool {
		if result[i].Trial != result[j].Trial {
			return result[i].Trial < result[j].Trial
		}
		return result[i].Stage < result[j].Stage
	})
	return result
}

// WriteCSV writes raw samples. Aggregate medians only after all repetitions;
// retaining raw data makes the reported statistic auditable.
func (p *Profiler) WriteCSV(w io.Writer) error {
	csvWriter := csv.NewWriter(w)
	if err := csvWriter.Write([]string{
		"scheme", "parameter", "parties", "trial", "stage", "elapsed_ns",
	}); err != nil {
		return err
	}
	for _, sample := range p.Samples() {
		if err := csvWriter.Write([]string{
			sample.Scheme,
			sample.Parameter,
			strconv.Itoa(sample.Parties),
			strconv.Itoa(sample.Trial),
			sample.Stage,
			strconv.FormatInt(sample.Elapsed.Nanoseconds(), 10),
		}); err != nil {
			return err
		}
	}
	csvWriter.Flush()
	if err := csvWriter.Error(); err != nil {
		return fmt.Errorf("write stage CSV: %w", err)
	}
	return nil
}
