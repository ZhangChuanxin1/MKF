package mkfcpu

// Durable, fail-closed matched CPU runner for public KKLSS and canonical MKF
// under the frozen P2GN16 security/packing profile. The formal entry point is
// opt-in and is never exercised by ordinary go test.

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

const (
	p2gn16MatchedBenchmarkSchema = "mkf-resnet20-p2gn16-matched-cpu-v1"
	p2gn16MatchedWarmups         = 0
	p2gn16MatchedTrials          = 10
	p2gn16MatchedCore            = "0"
	p2gn16RequiredMemAvailableKB = uint64(110 * 1024 * 1024)
	p2gn16BalancedExecutionOrder = "paired-alternating-trial1-KKLSS-MKF-trial2-MKF-KKLSS-v1"
	matchedKKLSSName             = "KKLSS-P2GN16-CPU"
	matchedCanonicalMKFName      = "MKF-P2GN16-Canonical-CPU"
)

type p2gn16ProcSample struct {
	UserTicks uint64 `json:"user_ticks"`
	SysTicks  uint64 `json:"sys_ticks"`
	MinFlt    uint64 `json:"minflt"`
	MajFlt    uint64 `json:"majflt"`
	RSSKB     uint64 `json:"rss_kb"`
	HWMKB     uint64 `json:"hwm_kb"`
	SwapKB    uint64 `json:"swap_kb"`
}

type p2gn16StageMeasurement struct {
	Stage        resNet20CPUStage `json:"stage"`
	WallNS       int64            `json:"wall_ns"`
	UserNS       int64            `json:"user_ns"`
	SysNS        int64            `json:"sys_ns"`
	MinFltDelta  uint64           `json:"minflt_delta"`
	MajFltDelta  uint64           `json:"majflt_delta"`
	RSSBeforeKB  uint64           `json:"rss_before_kb"`
	RSSAfterKB   uint64           `json:"rss_after_kb"`
	MaxRSSKB     uint64           `json:"max_rss_kb"`
	SwapBeforeKB uint64           `json:"swap_before_kb"`
	SwapAfterKB  uint64           `json:"swap_after_kb"`
	MaxSwapKB    uint64           `json:"max_swap_kb"`
	Valid        bool             `json:"valid"`
}

type p2gn16TrialGate struct {
	CorrectnessPass    bool    `json:"correctness_pass"`
	ClassificationPass bool    `json:"classification_pass"`
	CountPass          bool    `json:"count_pass"`
	MaxError           float64 `json:"max_error"`
	EncryptedTop1      int     `json:"encrypted_top1"`
	ReferenceTop1      int     `json:"reference_top1"`
	Label              int     `json:"label"`
	OwnerIngresses     int     `json:"owner_ingresses"`
	GuardedFolds       int     `json:"guarded_folds"`
	GuardRescales      int     `json:"guard_rescales"`
	NetworkProducts    int     `json:"network_products"`
	PolynomialProducts int     `json:"polynomial_products"`
	Rotations          int     `json:"rotations"`
	Degree59           int     `json:"degree59_activations"`
	Refreshes          int     `json:"refreshes"`
}

type p2gn16TrialRecord struct {
	Schema              string                   `json:"schema"`
	Implementation      string                   `json:"implementation"`
	ConstructionID      string                   `json:"construction_id"`
	Parameter           string                   `json:"parameter"`
	Parties             int                      `json:"parties"`
	Fixture             string                   `json:"fixture"`
	Trial               int                      `json:"trial"`
	Warmups             int                      `json:"warmups"`
	GOMAXPROCS          int                      `json:"gomaxprocs"`
	AffinityCore        string                   `json:"affinity_core"`
	ClockTicksPerSecond uint64                   `json:"clock_ticks_per_second"`
	Boundary            string                   `json:"measurement_boundary"`
	StartedUTC          string                   `json:"started_utc"`
	CompletedUTC        string                   `json:"completed_utc"`
	Stages              []p2gn16StageMeasurement `json:"stages"`
	Gate                p2gn16TrialGate          `json:"gate"`
	Valid               bool                     `json:"valid"`
	InvalidReasons      []string                 `json:"invalid_reasons,omitempty"`
	RunID               string                   `json:"run_id"`
	Metadata            p2gn16RunMetadata        `json:"run_metadata"`
	Preflight           p2gn16TrialPreflight     `json:"pre_trial"`
}

type p2gn16Progress struct {
	Schema          string            `json:"schema"`
	CompletedTrials int               `json:"completed_trials"`
	ValidTrials     int               `json:"valid_trials"`
	ExpectedTrials  int               `json:"expected_trials"`
	Completed       bool              `json:"completed"`
	TrialFiles      []string          `json:"trial_files"`
	UpdatedUTC      string            `json:"updated_utc"`
	RunID           string            `json:"run_id"`
	Metadata        p2gn16RunMetadata `json:"run_metadata"`
}

type p2gn16RunMetadata struct {
	RunID                          string `json:"run_id"`
	CPUModel                       string `json:"cpu_model"`
	GoVersion                      string `json:"go_version"`
	LattigoModule                  string `json:"lattigo_module"`
	SourceManifestSHA256           string `json:"source_manifest_sha256"`
	AssetManifestSHA256            string `json:"asset_manifest_sha256"`
	ParameterSourceSHA256          string `json:"parameter_source_sha256"`
	TestBinarySHA256               string `json:"test_binary_sha256"`
	MatchedFunctionalGateSHA256    string `json:"matched_functional_gate_sha256"`
	PublicOperandAuditSHA256       string `json:"public_operand_semantic_audit_sha256"`
	PhysicalObserverSealSHA256     string `json:"same_run_physical_observer_seal_sha256"`
	FunctionalSourceManifestSHA256 string `json:"functional_source_manifest_sha256"`
	FunctionalAssetManifestSHA256  string `json:"functional_asset_manifest_sha256"`
	AffinityCore                   string `json:"affinity_core"`
	ExecutionOrder                 string `json:"execution_order"`
}

type p2gn16TrialPreflight struct {
	ProcessBeforeCleanup p2gn16ProcSample `json:"process_before_cleanup"`
	ProcessAfterCleanup  p2gn16ProcSample `json:"process_after_cleanup"`
	MemAvailableBeforeKB uint64           `json:"mem_available_before_kb"`
	MemAvailableAfterKB  uint64           `json:"mem_available_after_kb"`
	GlobalSwapUsedBefore uint64           `json:"global_swap_used_before_kb"`
	GlobalSwapUsedAfter  uint64           `json:"global_swap_used_after_kb"`
}

type p2gn16SystemMemory struct {
	MemAvailableKB uint64
	SwapUsedKB     uint64
}

type p2gn16SummaryRow struct {
	Implementation string           `json:"implementation"`
	Stage          resNet20CPUStage `json:"stage"`
	Trials         int              `json:"trials"`
	MeanNS         float64          `json:"mean_ns"`
	SampleStdDevNS float64          `json:"sample_stddev_ns"`
	MedianNS       float64          `json:"median_ns"`
	MinNS          int64            `json:"min_ns"`
	MaxNS          int64            `json:"max_ns"`
	CV             float64          `json:"coefficient_of_variation"`
}

type p2gn16MatchedFactory struct {
	Name           string
	ConstructionID string
	Boundary       string
	Fixture        string
	NewTrial       func() (resNet20CPUSegmentedTrial, error)
	Gate           func(resNet20CPUSegmentedTrial) (p2gn16TrialGate, error)
}

type p2gn16Sampler func() (p2gn16ProcSample, error)
type p2gn16Preflight func() (p2gn16TrialPreflight, error)

func p2gn16ReadProcSample() (p2gn16ProcSample, error) {
	if runtime.GOOS != "linux" {
		return p2gn16ProcSample{}, fmt.Errorf("formal resource accounting requires Linux /proc")
	}
	stat, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return p2gn16ProcSample{}, err
	}
	closeParen := strings.LastIndexByte(string(stat), ')')
	if closeParen < 0 {
		return p2gn16ProcSample{}, fmt.Errorf("malformed /proc/self/stat")
	}
	fields := strings.Fields(string(stat)[closeParen+1:])
	if len(fields) <= 12 {
		return p2gn16ProcSample{}, fmt.Errorf("truncated /proc/self/stat")
	}
	parse := func(index int, label string) (uint64, error) {
		value, parseErr := strconv.ParseUint(fields[index], 10, 64)
		if parseErr != nil {
			return 0, fmt.Errorf("parse %s: %w", label, parseErr)
		}
		return value, nil
	}
	result := p2gn16ProcSample{}
	if result.MinFlt, err = parse(7, "minflt"); err != nil {
		return result, err
	}
	if result.MajFlt, err = parse(9, "majflt"); err != nil {
		return result, err
	}
	if result.UserTicks, err = parse(11, "utime"); err != nil {
		return result, err
	}
	if result.SysTicks, err = parse(12, "stime"); err != nil {
		return result, err
	}
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return result, err
	}
	for _, line := range strings.Split(string(status), "\n") {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		value, parseErr := strconv.ParseUint(parts[1], 10, 64)
		if parseErr != nil {
			continue
		}
		switch strings.TrimSuffix(parts[0], ":") {
		case "VmRSS":
			result.RSSKB = value
		case "VmHWM":
			result.HWMKB = value
		case "VmSwap":
			result.SwapKB = value
		}
	}
	return result, nil
}

func p2gn16ReadSystemMemory() (p2gn16SystemMemory, error) {
	if runtime.GOOS != "linux" {
		return p2gn16SystemMemory{}, fmt.Errorf("formal memory preflight requires Linux /proc")
	}
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return p2gn16SystemMemory{}, err
	}
	values := make(map[string]uint64)
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		value, parseErr := strconv.ParseUint(fields[1], 10, 64)
		if parseErr == nil {
			values[strings.TrimSuffix(fields[0], ":")] = value
		}
	}
	available, availableOK := values["MemAvailable"]
	swapTotal, totalOK := values["SwapTotal"]
	swapFree, freeOK := values["SwapFree"]
	if !availableOK || !totalOK || !freeOK || swapFree > swapTotal {
		return p2gn16SystemMemory{}, fmt.Errorf("/proc/meminfo lacks consistent MemAvailable/SwapTotal/SwapFree")
	}
	return p2gn16SystemMemory{MemAvailableKB: available, SwapUsedKB: swapTotal - swapFree}, nil
}

func p2gn16FormalTrialPreflight() (p2gn16TrialPreflight, error) {
	result := p2gn16TrialPreflight{}
	beforeProcess, err := p2gn16ReadProcSample()
	if err != nil {
		return result, err
	}
	beforeMemory, err := p2gn16ReadSystemMemory()
	if err != nil {
		return result, err
	}
	result.ProcessBeforeCleanup = beforeProcess
	result.MemAvailableBeforeKB = beforeMemory.MemAvailableKB
	result.GlobalSwapUsedBefore = beforeMemory.SwapUsedKB

	// Strictly outside both NewTrial/key staging and the measured online
	// boundary: discard the previous trial's very large heap before deciding
	// whether this trial is eligible to start.
	runtime.GC()
	debug.FreeOSMemory()

	afterProcess, err := p2gn16ReadProcSample()
	if err != nil {
		return result, err
	}
	afterMemory, err := p2gn16ReadSystemMemory()
	if err != nil {
		return result, err
	}
	result.ProcessAfterCleanup = afterProcess
	result.MemAvailableAfterKB = afterMemory.MemAvailableKB
	result.GlobalSwapUsedAfter = afterMemory.SwapUsedKB
	return result, p2gn16ValidateTrialPreflight(result)
}

func p2gn16ValidateTrialPreflight(result p2gn16TrialPreflight) error {
	if result.ProcessAfterCleanup.SwapKB != 0 {
		return fmt.Errorf("pre-trial process VmSwap=%d KiB after heap cleanup", result.ProcessAfterCleanup.SwapKB)
	}
	if result.MemAvailableAfterKB < p2gn16RequiredMemAvailableKB {
		return fmt.Errorf("pre-trial MemAvailable=%d KiB < %d KiB", result.MemAvailableAfterKB, p2gn16RequiredMemAvailableKB)
	}
	// Existing global swap can contain unrelated cold pages. It is recorded
	// but does not invalidate the run unless it is actively growing here.
	if result.GlobalSwapUsedAfter > result.GlobalSwapUsedBefore {
		return fmt.Errorf("global swap grew during preflight: %d -> %d KiB", result.GlobalSwapUsedBefore, result.GlobalSwapUsedAfter)
	}
	return nil
}

func p2gn16Delta(after, before uint64) (uint64, error) {
	if after < before {
		return 0, fmt.Errorf("monotonic process counter decreased: %d -> %d", before, after)
	}
	return after - before, nil
}

func p2gn16TicksNS(delta, hz uint64) int64 {
	return int64((delta*1_000_000_000 + hz/2) / hz)
}

func p2gn16MeasureStage(trial resNet20CPUSegmentedTrial, stage resNet20CPUStage, hz uint64, sampler p2gn16Sampler) (p2gn16StageMeasurement, error) {
	measurement := p2gn16StageMeasurement{Stage: stage, Valid: true}
	segments := trial.SegmentCount(stage)
	if segments <= 0 {
		return measurement, fmt.Errorf("stage %s has %d segments", stage, segments)
	}
	for index := 0; index < segments; index++ {
		prepared, err := trial.PrepareSegment(stage, index)
		if err != nil {
			return measurement, fmt.Errorf("prepare segment %d outside timing: %w", index, err)
		}
		before, err := sampler()
		if err != nil {
			return measurement, err
		}
		start := time.Now()
		runErr := prepared.Run()
		wall := time.Since(start).Nanoseconds()
		after, sampleErr := sampler()
		releaseErr := prepared.Release()
		if runErr != nil {
			return measurement, fmt.Errorf("run segment %d: %w", index, runErr)
		}
		if sampleErr != nil {
			return measurement, sampleErr
		}
		if releaseErr != nil {
			return measurement, fmt.Errorf("release segment %d outside timing: %w", index, releaseErr)
		}
		userTicks, err := p2gn16Delta(after.UserTicks, before.UserTicks)
		if err != nil {
			return measurement, err
		}
		sysTicks, err := p2gn16Delta(after.SysTicks, before.SysTicks)
		if err != nil {
			return measurement, err
		}
		minFlt, err := p2gn16Delta(after.MinFlt, before.MinFlt)
		if err != nil {
			return measurement, err
		}
		majFlt, err := p2gn16Delta(after.MajFlt, before.MajFlt)
		if err != nil {
			return measurement, err
		}
		measurement.WallNS += wall
		measurement.UserNS += p2gn16TicksNS(userTicks, hz)
		measurement.SysNS += p2gn16TicksNS(sysTicks, hz)
		measurement.MinFltDelta += minFlt
		measurement.MajFltDelta += majFlt
		if index == 0 {
			measurement.RSSBeforeKB, measurement.SwapBeforeKB = before.RSSKB, before.SwapKB
		}
		measurement.RSSAfterKB, measurement.SwapAfterKB = after.RSSKB, after.SwapKB
		if before.HWMKB > measurement.MaxRSSKB {
			measurement.MaxRSSKB = before.HWMKB
		}
		if after.HWMKB > measurement.MaxRSSKB {
			measurement.MaxRSSKB = after.HWMKB
		}
		if before.SwapKB > measurement.MaxSwapKB {
			measurement.MaxSwapKB = before.SwapKB
		}
		if after.SwapKB > measurement.MaxSwapKB {
			measurement.MaxSwapKB = after.SwapKB
		}
	}
	measurement.Valid = measurement.WallNS > 0 && measurement.MajFltDelta == 0 && measurement.MaxSwapKB == 0
	return measurement, nil
}

func p2gn16Total(stages []p2gn16StageMeasurement) p2gn16StageMeasurement {
	total := p2gn16StageMeasurement{Stage: resNet20CPUTotal, Valid: true}
	for index, stage := range stages {
		total.WallNS += stage.WallNS
		total.UserNS += stage.UserNS
		total.SysNS += stage.SysNS
		total.MinFltDelta += stage.MinFltDelta
		total.MajFltDelta += stage.MajFltDelta
		if index == 0 {
			total.RSSBeforeKB, total.SwapBeforeKB = stage.RSSBeforeKB, stage.SwapBeforeKB
		}
		total.RSSAfterKB, total.SwapAfterKB = stage.RSSAfterKB, stage.SwapAfterKB
		if stage.MaxRSSKB > total.MaxRSSKB {
			total.MaxRSSKB = stage.MaxRSSKB
		}
		if stage.MaxSwapKB > total.MaxSwapKB {
			total.MaxSwapKB = stage.MaxSwapKB
		}
		total.Valid = total.Valid && stage.Valid
	}
	return total
}

var p2gn16DirectorySync = func(path string) error {
	if runtime.GOOS != "linux" {
		return nil
	}
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	syncErr := directory.Sync()
	closeErr := directory.Close()
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}

func p2gn16AtomicJSON(path string, value interface{}) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary := path + ".tmp"
	file, err := os.OpenFile(temporary, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Rename(temporary, path); err != nil {
		return err
	}
	return p2gn16DirectorySync(filepath.Dir(path))
}

func p2gn16AppendJSONL(path string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	written, writeErr := file.Write(data)
	if writeErr == nil && written != len(data) {
		writeErr = io.ErrShortWrite
	}
	if writeErr == nil {
		writeErr = file.Sync()
	}
	if closeErr := file.Close(); writeErr == nil {
		writeErr = closeErr
	}
	if writeErr != nil {
		return writeErr
	}
	return p2gn16DirectorySync(filepath.Dir(path))
}

func p2gn16RunTrial(factory p2gn16MatchedFactory, trialIndex int, hz uint64, sampler p2gn16Sampler, metadata p2gn16RunMetadata, preflight p2gn16Preflight) (p2gn16TrialRecord, error) {
	record := p2gn16TrialRecord{
		Schema: p2gn16MatchedBenchmarkSchema, Implementation: factory.Name, ConstructionID: factory.ConstructionID,
		Parameter: guardedCompactP2GN16ParameterID, Parties: fullResNet20Parties, Fixture: factory.Fixture,
		Trial: trialIndex, Warmups: p2gn16MatchedWarmups, GOMAXPROCS: runtime.GOMAXPROCS(0), AffinityCore: p2gn16MatchedCore,
		ClockTicksPerSecond: hz, Boundary: factory.Boundary, StartedUTC: time.Now().UTC().Format(time.RFC3339Nano),
		RunID: metadata.RunID, Metadata: metadata,
	}
	if preflight == nil {
		record.InvalidReasons = append(record.InvalidReasons, "pre-trial resource gate is required")
		record.CompletedUTC = time.Now().UTC().Format(time.RFC3339Nano)
		return record, fmt.Errorf("pre-trial resource gate is required")
	}
	var err error
	record.Preflight, err = preflight()
	if err != nil {
		record.InvalidReasons = append(record.InvalidReasons, "pre-trial resource gate: "+err.Error())
		record.CompletedUTC = time.Now().UTC().Format(time.RFC3339Nano)
		return record, fmt.Errorf("pre-trial resource gate: %w", err)
	}
	trial, err := factory.NewTrial()
	if err != nil {
		record.InvalidReasons = append(record.InvalidReasons, "prepare trial outside timing: "+err.Error())
		record.CompletedUTC = time.Now().UTC().Format(time.RFC3339Nano)
		return record, fmt.Errorf("prepare trial outside timing: %w", err)
	}
	for _, stage := range resNet20CPUStages {
		measurement, stageErr := p2gn16MeasureStage(trial, stage, hz, sampler)
		record.Stages = append(record.Stages, measurement)
		if stageErr != nil {
			record.InvalidReasons = append(record.InvalidReasons, stageErr.Error())
			break
		}
		if measurement.MajFltDelta != 0 {
			record.InvalidReasons = append(record.InvalidReasons, fmt.Sprintf("%s timed major faults=%d", stage, measurement.MajFltDelta))
		}
		if measurement.MaxSwapKB != 0 {
			record.InvalidReasons = append(record.InvalidReasons, fmt.Sprintf("%s timed VmSwap max=%d KiB", stage, measurement.MaxSwapKB))
		}
	}
	if len(record.Stages) == len(resNet20CPUStages) {
		record.Stages = append(record.Stages, p2gn16Total(record.Stages))
		if verifyErr := trial.Verify(); verifyErr != nil {
			record.InvalidReasons = append(record.InvalidReasons, "untimed verification: "+verifyErr.Error())
		} else if gate, gateErr := factory.Gate(trial); gateErr != nil {
			record.InvalidReasons = append(record.InvalidReasons, "gate extraction: "+gateErr.Error())
		} else {
			record.Gate = gate
			if !gate.CorrectnessPass || !gate.ClassificationPass || !gate.CountPass {
				record.InvalidReasons = append(record.InvalidReasons, "correctness/class/count gate failed")
			}
		}
	}
	record.CompletedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	record.Valid = len(record.InvalidReasons) == 0 && len(record.Stages) == 4 && record.Stages[3].Valid
	if !record.Valid {
		return record, fmt.Errorf("trial invalid: %s", strings.Join(record.InvalidReasons, "; "))
	}
	return record, nil
}

func p2gn16MatchedFactories(request fullResNet20Request) []p2gn16MatchedFactory {
	fixture := kklssResNet20Fixture
	return []p2gn16MatchedFactory{
		{
			Name: matchedKKLSSName, ConstructionID: kklssConstruction,
			Boundary: "online native public-KKLSS P2GN16-persistent evaluation and 22 refresh bridges; excludes keygen, encryption, owner staging/serialization, oracle and result I/O",
			Fixture:  fixture,
			NewTrial: func() (resNet20CPUSegmentedTrial, error) {
				backend := &kklssResNet20CPUBackend{state: new(kklssCPUBackendState)}
				return newKKLSSResNet20CPUTrial(backend, request)
			},
			Gate: func(prepared resNet20CPUSegmentedTrial) (p2gn16TrialGate, error) {
				trial, ok := prepared.(*kklssResNet20CPUTrial)
				if !ok {
					return p2gn16TrialGate{}, fmt.Errorf("unexpected KKLSS trial %T", prepared)
				}
				counts := trial.executor.kklssOperationSnapshot()
				return p2gn16TrialGate{
					CorrectnessPass: trial.verificationCorrectnessPass, ClassificationPass: trial.verificationClassPass, CountPass: trial.verificationCountPass,
					MaxError: trial.verificationMaxError, EncryptedTop1: trial.verificationEncryptedTop1, ReferenceTop1: trial.verificationReferenceTop1, Label: trial.verificationLabel,
					OwnerIngresses: trial.ownerCount, NetworkProducts: counts.mulRelin, PolynomialProducts: counts.polynomialMulRelin,
					Rotations: counts.rotations, Degree59: counts.polynomials, Refreshes: counts.refreshes,
				}, nil
			},
		},
		{
			Name: matchedCanonicalMKFName, ConstructionID: guardedCompactP2GN16CPUConstructionID,
			Boundary: "6201 public guarded FoldToAggregate+guard-prime rescales, full canonical aggregate-domain network, and 22 refreshes; excludes keygen, encryption, owner serialization, oracle and result I/O",
			Fixture:  fixture,
			NewTrial: func() (resNet20CPUSegmentedTrial, error) { return newGuardedCompactP2GN16CPUTrial(request) },
			Gate: func(prepared resNet20CPUSegmentedTrial) (p2gn16TrialGate, error) {
				trial, ok := prepared.(*paperMKFResNet20CPUTrial)
				if !ok {
					return p2gn16TrialGate{}, fmt.Errorf("unexpected canonical trial %T", prepared)
				}
				e := trial.guardedEvidence
				return p2gn16TrialGate{
					CorrectnessPass: e.CorrectnessPass, ClassificationPass: e.ClassificationPass, CountPass: e.CountPass,
					MaxError: e.MaxError, EncryptedTop1: e.EncryptedTop1, ReferenceTop1: e.ReferenceTop1, Label: e.Label,
					OwnerIngresses: e.Owners, GuardedFolds: e.Folds, GuardRescales: e.IngressRescales, NetworkProducts: e.NetworkProducts,
					PolynomialProducts: e.PolynomialProducts,
					Rotations:          e.Rotations, Degree59: e.Degree59Activations, Refreshes: e.Refreshes,
				}, nil
			},
		},
	}
}

func p2gn16EncodeRawCSV(destination io.Writer, trials []p2gn16TrialRecord) error {
	writer := csv.NewWriter(destination)
	if err := writer.Write([]string{"implementation", "construction_id", "parameter", "trial", "stage", "wall_ns", "user_ns", "sys_ns", "minflt_delta", "majflt_delta", "rss_before_kb", "rss_after_kb", "max_rss_kb", "swap_before_kb", "swap_after_kb", "max_swap_kb", "valid"}); err != nil {
		return err
	}
	for _, trial := range trials {
		for _, stage := range trial.Stages {
			if err := writer.Write([]string{trial.Implementation, trial.ConstructionID, trial.Parameter, strconv.Itoa(trial.Trial), string(stage.Stage), strconv.FormatInt(stage.WallNS, 10), strconv.FormatInt(stage.UserNS, 10), strconv.FormatInt(stage.SysNS, 10), strconv.FormatUint(stage.MinFltDelta, 10), strconv.FormatUint(stage.MajFltDelta, 10), strconv.FormatUint(stage.RSSBeforeKB, 10), strconv.FormatUint(stage.RSSAfterKB, 10), strconv.FormatUint(stage.MaxRSSKB, 10), strconv.FormatUint(stage.SwapBeforeKB, 10), strconv.FormatUint(stage.SwapAfterKB, 10), strconv.FormatUint(stage.MaxSwapKB, 10), strconv.FormatBool(stage.Valid)}); err != nil {
				return err
			}
		}
	}
	writer.Flush()
	return writer.Error()
}

func p2gn16WriteRawCSV(path string, trials []p2gn16TrialRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary := path + ".tmp"
	file, err := os.Create(temporary)
	if err != nil {
		return err
	}
	if err := p2gn16EncodeRawCSV(file, trials); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporary, path); err != nil {
		return err
	}
	return p2gn16DirectorySync(filepath.Dir(path))
}

func p2gn16Summaries(records []p2gn16TrialRecord) ([]p2gn16SummaryRow, error) {
	values := make(map[string][]int64)
	for _, record := range records {
		if !record.Valid || len(record.Stages) != 4 {
			return nil, fmt.Errorf("cannot summarize invalid trial %s/%d", record.Implementation, record.Trial)
		}
		for _, stage := range record.Stages {
			values[record.Implementation+"\x00"+string(stage.Stage)] = append(values[record.Implementation+"\x00"+string(stage.Stage)], stage.WallNS)
		}
	}
	rows := make([]p2gn16SummaryRow, 0, 8)
	for key, samples := range values {
		if len(samples) != p2gn16MatchedTrials {
			return nil, fmt.Errorf("summary group %q has %d trials", key, len(samples))
		}
		parts := strings.SplitN(key, "\x00", 2)
		sorted := append([]int64(nil), samples...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		var sum float64
		for _, value := range samples {
			sum += float64(value)
		}
		mean := sum / float64(len(samples))
		var squared float64
		for _, value := range samples {
			delta := float64(value) - mean
			squared += delta * delta
		}
		stddev := math.Sqrt(squared / float64(len(samples)-1))
		median := float64(sorted[4]+sorted[5]) / 2
		rows = append(rows, p2gn16SummaryRow{Implementation: parts[0], Stage: resNet20CPUStage(parts[1]), Trials: len(samples), MeanNS: mean, SampleStdDevNS: stddev, MedianNS: median, MinNS: sorted[0], MaxNS: sorted[len(sorted)-1], CV: stddev / mean})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Implementation != rows[j].Implementation {
			return rows[i].Implementation < rows[j].Implementation
		}
		return resNet20CPUStageOrder(rows[i].Stage) < resNet20CPUStageOrder(rows[j].Stage)
	})
	if len(rows) != 8 {
		return nil, fmt.Errorf("summary rows=%d, want 8", len(rows))
	}
	return rows, nil
}

func p2gn16FactoryOrder(trialIndex int) ([2]int, error) {
	if trialIndex < 1 || trialIndex > p2gn16MatchedTrials {
		return [2]int{}, fmt.Errorf("trial index %d outside 1..%d", trialIndex, p2gn16MatchedTrials)
	}
	if trialIndex%2 == 0 {
		return [2]int{1, 0}, nil
	}
	return [2]int{0, 1}, nil
}

func p2gn16RunMatched(resultDir string, request fullResNet20Request, hz uint64, sampler p2gn16Sampler, metadata p2gn16RunMetadata) error {
	progress := p2gn16Progress{Schema: p2gn16MatchedBenchmarkSchema, ExpectedTrials: 2 * p2gn16MatchedTrials, RunID: metadata.RunID, Metadata: metadata}
	var records []p2gn16TrialRecord
	factories := p2gn16MatchedFactories(request)
	for trialIndex := 1; trialIndex <= p2gn16MatchedTrials; trialIndex++ {
		order, orderErr := p2gn16FactoryOrder(trialIndex)
		if orderErr != nil {
			return orderErr
		}
		for _, factoryIndex := range order {
			factory := factories[factoryIndex]
			record, runErr := p2gn16RunTrial(factory, trialIndex, hz, sampler, metadata, p2gn16FormalTrialPreflight)
			relative := filepath.Join("trials", fmt.Sprintf("%s-trial-%02d.json", strings.ToLower(factory.Name), trialIndex))
			if writeErr := p2gn16AtomicJSON(filepath.Join(resultDir, relative), record); writeErr != nil {
				return writeErr
			}
			if record.Valid {
				if appendErr := p2gn16AppendJSONL(filepath.Join(resultDir, "trials.jsonl"), record); appendErr != nil {
					return appendErr
				}
			}
			progress.CompletedTrials++
			if record.Valid {
				progress.ValidTrials++
			}
			progress.TrialFiles = append(progress.TrialFiles, filepath.ToSlash(relative))
			progress.UpdatedUTC = time.Now().UTC().Format(time.RFC3339Nano)
			if writeErr := p2gn16AtomicJSON(filepath.Join(resultDir, "progress.json"), progress); writeErr != nil {
				return writeErr
			}
			records = append(records, record)
			if runErr != nil {
				return fmt.Errorf("%s trial %d persisted invalid: %w", factory.Name, trialIndex, runErr)
			}
		}
	}
	if len(records) != 20 {
		return fmt.Errorf("trial coverage=%d, want 20", len(records))
	}
	if err := p2gn16WriteRawCSV(filepath.Join(resultDir, "raw_80_rows.csv"), records); err != nil {
		return err
	}
	summaryRows, err := p2gn16Summaries(records)
	if err != nil {
		return err
	}
	progress.Completed = progress.CompletedTrials == progress.ExpectedTrials && progress.ValidTrials == progress.ExpectedTrials
	progress.UpdatedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	if err := p2gn16AtomicJSON(filepath.Join(resultDir, "progress.json"), progress); err != nil {
		return err
	}
	summary := struct {
		Schema    string             `json:"schema"`
		Parameter string             `json:"parameter"`
		Trials    int                `json:"trials_per_implementation"`
		Warmups   int                `json:"warmups"`
		Rows      int                `json:"raw_rows"`
		Complete  bool               `json:"complete"`
		Metadata  p2gn16RunMetadata  `json:"metadata"`
		Summary   []p2gn16SummaryRow `json:"summary"`
	}{p2gn16MatchedBenchmarkSchema, guardedCompactP2GN16ParameterID, 10, 0, 80, progress.Completed, metadata, summaryRows}
	return p2gn16AtomicJSON(filepath.Join(resultDir, "summary.json"), summary)
}

func TestP2GN16MatchedCPUFormalBenchmark(t *testing.T) {
	if os.Getenv("MKF_RUN_P2GN16_MATCHED_CPU_BENCHMARK") != "1" {
		t.Skip("formal matched P2GN16 benchmark is opt-in")
	}
	if runtime.GOMAXPROCS(0) != 1 {
		t.Fatalf("GOMAXPROCS=%d, want 1", runtime.GOMAXPROCS(0))
	}
	core, err := singletonAffinityCore()
	if err != nil {
		t.Fatal(err)
	}
	if core != p2gn16MatchedCore {
		t.Fatalf("affinity core=%s, want CPU0", core)
	}
	resultDir := strings.TrimSpace(os.Getenv("MKF_P2GN16_MATCHED_RESULT_DIR"))
	if resultDir == "" {
		t.Fatal("MKF_P2GN16_MATCHED_RESULT_DIR is required")
	}
	hz, err := strconv.ParseUint(os.Getenv("MKF_CLK_TCK"), 10, 64)
	if err != nil || hz == 0 {
		t.Fatal("MKF_CLK_TCK must be a positive integer from getconf CLK_TCK")
	}
	metadata := p2gn16RunMetadata{RunID: os.Getenv("MKF_P2GN16_RUN_ID"), CPUModel: os.Getenv("MKF_RESNET20_CPU_MODEL"), GoVersion: runtime.Version(), LattigoModule: os.Getenv("MKF_RESNET20_LATTIGO_MODULE"), AffinityCore: core, ExecutionOrder: p2gn16BalancedExecutionOrder}
	if metadata.RunID == "" || metadata.CPUModel == "" || metadata.LattigoModule == "" {
		t.Fatal("run ID, CPU model and Lattigo module identity are required")
	}
	shaTargets := map[string]*string{"MKF_RESNET20_SOURCE_MANIFEST_SHA256": &metadata.SourceManifestSHA256, "MKF_RESNET20_ASSET_MANIFEST_SHA256": &metadata.AssetManifestSHA256, "MKF_RESNET20_PARAMETER_SOURCE_SHA256": &metadata.ParameterSourceSHA256, "MKF_TEST_BINARY_SHA256": &metadata.TestBinarySHA256, "MKF_P2GN16_MATCHED_FUNCTIONAL_GATE_SHA256": &metadata.MatchedFunctionalGateSHA256, "MKF_P2GN16_PUBLIC_OPERAND_AUDIT_SHA256": &metadata.PublicOperandAuditSHA256, "MKF_P2GN16_PHYSICAL_OBSERVER_SEAL_SHA256": &metadata.PhysicalObserverSealSHA256, "MKF_P2GN16_FUNCTIONAL_SOURCE_MANIFEST_SHA256": &metadata.FunctionalSourceManifestSHA256, "MKF_P2GN16_FUNCTIONAL_ASSET_MANIFEST_SHA256": &metadata.FunctionalAssetManifestSHA256}
	for name, destination := range shaTargets {
		value := os.Getenv(name)
		decoded, decodeErr := hex.DecodeString(value)
		if decodeErr != nil || len(decoded) != sha256.Size {
			t.Fatalf("%s is not a SHA-256", name)
		}
		*destination = strings.ToLower(value)
	}
	if _, err := os.Stat(filepath.Join(resultDir, "progress.json")); err == nil || !os.IsNotExist(err) {
		t.Fatal("result directory already contains progress.json; refuse non-immutable restart")
	}
	request, err := kklssCPURequestFromEnvironmentFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	if err := p2gn16RunMatched(resultDir, request, hz, p2gn16ReadProcSample, metadata); err != nil {
		t.Fatal(err)
	}
}

func TestP2GN16MatchedCPUContractNoFullRun(t *testing.T) {
	if p2gn16MatchedWarmups != 0 || p2gn16MatchedTrials != 10 {
		t.Fatal("0-warmup/10-trial contract drifted")
	}
	if p2gn16BalancedExecutionOrder != "paired-alternating-trial1-KKLSS-MKF-trial2-MKF-KKLSS-v1" {
		t.Fatal("balanced AB/BA execution order drifted")
	}
	for trialIndex := 1; trialIndex <= p2gn16MatchedTrials; trialIndex++ {
		order, orderErr := p2gn16FactoryOrder(trialIndex)
		if orderErr != nil {
			t.Fatal(orderErr)
		}
		want := [2]int{0, 1}
		if trialIndex%2 == 0 {
			want = [2]int{1, 0}
		}
		if order != want {
			t.Fatalf("trial %d order=%v, want %v", trialIndex, order, want)
		}
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	kklss, err := kklssParameters(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	if kklss.LogN() != profile.base.LogN() || kklss.MaxLevel() != profile.base.MaxLevel() || kklss.Scale() != profile.base.Scale() {
		t.Fatal("KKLSS P2GN16 persistent parameter conversion drifted")
	}
	for index, modulus := range profile.base.Q() {
		if kklss.Q()[index] != modulus {
			t.Fatalf("KKLSS Q[%d] drifted", index)
		}
	}
	factories := p2gn16MatchedFactories(fullResNet20Request{})
	if len(factories) != 2 || factories[0].Name != matchedKKLSSName || factories[1].Name != matchedCanonicalMKFName {
		t.Fatalf("matched implementation set drifted: %+v", factories)
	}
	if guardedCompactSecurityGateFor(guardedCompactP2GN16ParameterID) != guardedCompactP2GN16SecurityGate {
		t.Fatal("P2GN16 ordinary-LWE gate drifted")
	}
	polynomial := referenceResNetBlockReLUPolynomial(1.0)
	polynomialProducts, countErr := kklssCarrierBSGSProductCount(polynomial)
	if polynomial.Degree() != resnetReferenceReLUDegree || polynomial.Basis != ckks.ChebyshevBasis || countErr != nil || polynomialProducts != 17 || 19*polynomialProducts != 323 {
		t.Fatalf("degree-59 internal product schedule drifted: degree=%d basis=%d per=%d total=%d err=%v", polynomial.Degree(), polynomial.Basis, polynomialProducts, 19*polynomialProducts, countErr)
	}
	if 2*p2gn16MatchedTrials*4 != 80 {
		t.Fatal("raw row coverage contract drifted")
	}
}

type p2gn16DryTrial struct {
	next     int
	active   bool
	verified bool
}
type p2gn16DrySegment struct {
	trial *p2gn16DryTrial
	ran   bool
}

func (*p2gn16DryTrial) RunStem() error                    { return fmt.Errorf("dry trial is segmented") }
func (*p2gn16DryTrial) RunResidualBlocks() error          { return fmt.Errorf("dry trial is segmented") }
func (*p2gn16DryTrial) RunFullyConnected() error          { return fmt.Errorf("dry trial is segmented") }
func (*p2gn16DryTrial) SegmentCount(resNet20CPUStage) int { return 1 }
func (trial *p2gn16DryTrial) PrepareSegment(stage resNet20CPUStage, _ int) (resNet20CPUPreparedSegment, error) {
	if trial.active || trial.next >= len(resNet20CPUStages) || resNet20CPUStages[trial.next] != stage {
		return nil, fmt.Errorf("dry stage order")
	}
	trial.active = true
	return &p2gn16DrySegment{trial: trial}, nil
}
func (segment *p2gn16DrySegment) Run() error {
	if segment.ran || !segment.trial.active {
		return fmt.Errorf("dry run lifecycle")
	}
	segment.ran = true
	time.Sleep(time.Microsecond)
	return nil
}
func (segment *p2gn16DrySegment) Release() error {
	if !segment.ran || !segment.trial.active {
		return fmt.Errorf("dry release lifecycle")
	}
	segment.trial.active = false
	segment.trial.next++
	return nil
}
func (trial *p2gn16DryTrial) Verify() error {
	if trial.active || trial.next != len(resNet20CPUStages) {
		return fmt.Errorf("dry verify lifecycle")
	}
	trial.verified = true
	return nil
}

func TestP2GN16MatchedCPUDurableDryRunNoCrypto(t *testing.T) {
	originalSync := p2gn16DirectorySync
	syncCalls := 0
	p2gn16DirectorySync = func(string) error { syncCalls++; return nil }
	defer func() { p2gn16DirectorySync = originalSync }()
	sequence := uint64(0)
	sampler := func() (p2gn16ProcSample, error) {
		sequence++
		return p2gn16ProcSample{UserTicks: sequence, MinFlt: sequence, RSSKB: 1024 + sequence, HWMKB: 2048 + sequence}, nil
	}
	factory := p2gn16MatchedFactory{
		Name: "DRY", ConstructionID: "no-crypto", Boundary: "dry", Fixture: "dry",
		NewTrial: func() (resNet20CPUSegmentedTrial, error) { return &p2gn16DryTrial{}, nil },
		Gate: func(trial resNet20CPUSegmentedTrial) (p2gn16TrialGate, error) {
			if dry, ok := trial.(*p2gn16DryTrial); !ok || !dry.verified {
				return p2gn16TrialGate{}, fmt.Errorf("dry gate before verify")
			}
			return p2gn16TrialGate{CorrectnessPass: true, ClassificationPass: true, CountPass: true}, nil
		},
	}
	metadata := p2gn16RunMetadata{RunID: "dry-run", AffinityCore: p2gn16MatchedCore}
	preflight := func() (p2gn16TrialPreflight, error) {
		return p2gn16TrialPreflight{MemAvailableAfterKB: p2gn16RequiredMemAvailableKB}, nil
	}
	record, err := p2gn16RunTrial(factory, 1, 100, sampler, metadata, preflight)
	if err != nil {
		t.Fatal(err)
	}
	if !record.Valid || len(record.Stages) != 4 {
		t.Fatalf("invalid dry record: %+v", record)
	}
	total := record.Stages[3]
	var wall, user, sys int64
	var minFlt, majFlt uint64
	for _, stage := range record.Stages[:3] {
		wall += stage.WallNS
		user += stage.UserNS
		sys += stage.SysNS
		minFlt += stage.MinFltDelta
		majFlt += stage.MajFltDelta
	}
	if total.WallNS != wall || total.UserNS != user || total.SysNS != sys || total.MinFltDelta != minFlt || total.MajFltDelta != majFlt {
		t.Fatal("dry total is not the exact integer stage sum")
	}
	directory := t.TempDir()
	trialPath := filepath.Join(directory, "trials", "dry-trial-01.json")
	if err := p2gn16AtomicJSON(trialPath, record); err != nil {
		t.Fatal(err)
	}
	if err := p2gn16AppendJSONL(filepath.Join(directory, "trials.jsonl"), record); err != nil {
		t.Fatal(err)
	}
	progress := p2gn16Progress{Schema: p2gn16MatchedBenchmarkSchema, CompletedTrials: 1, ValidTrials: 1, ExpectedTrials: 20, TrialFiles: []string{"trials/dry-trial-01.json"}}
	if err := p2gn16AtomicJSON(filepath.Join(directory, "progress.json"), progress); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(trialPath); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(directory, "progress.json")); err != nil {
		t.Fatal(err)
	}
	if syncCalls != 3 {
		t.Fatalf("durable writer invoked directory sync %d times, want 3", syncCalls)
	}
}

func TestP2GN16MatchedCPUMajorFaultAndSwapInvalidate(t *testing.T) {
	for _, swapped := range []bool{false, true} {
		calls := 0
		sampler := func() (p2gn16ProcSample, error) {
			calls++
			sample := p2gn16ProcSample{UserTicks: uint64(calls), RSSKB: 1024, HWMKB: 2048}
			if !swapped && calls == 2 {
				sample.MajFlt = 1
			}
			if swapped {
				sample.SwapKB = 4
			}
			return sample, nil
		}
		measurement, err := p2gn16MeasureStage(&p2gn16DryTrial{}, resNet20CPUStem, 100, sampler)
		if err != nil {
			t.Fatal(err)
		}
		if measurement.Valid {
			t.Fatalf("major-fault/swap case valid: %+v", measurement)
		}
	}
}

type p2gn16FailWriter struct{}

func (p2gn16FailWriter) Write([]byte) (int, error) {
	return 0, errors.New("injected CSV write failure")
}

func TestP2GN16MatchedCPUCSVWriterFailsClosed(t *testing.T) {
	if err := p2gn16EncodeRawCSV(p2gn16FailWriter{}, []p2gn16TrialRecord{{}}); err == nil || !strings.Contains(err.Error(), "injected CSV") {
		t.Fatalf("CSV writer did not propagate injected error: %v", err)
	}
}

func TestP2GN16MatchedCPUPreflightFailsClosed(t *testing.T) {
	base := p2gn16TrialPreflight{
		MemAvailableAfterKB:  p2gn16RequiredMemAvailableKB,
		GlobalSwapUsedBefore: 4096,
		GlobalSwapUsedAfter:  4096,
	}
	if err := p2gn16ValidateTrialPreflight(base); err != nil {
		t.Fatalf("stable unrelated global swap should be recorded, not rejected: %v", err)
	}
	tests := []struct {
		name string
		edit func(*p2gn16TrialPreflight)
	}{
		{"process-swap", func(value *p2gn16TrialPreflight) { value.ProcessAfterCleanup.SwapKB = 1 }},
		{"low-memory", func(value *p2gn16TrialPreflight) { value.MemAvailableAfterKB-- }},
		{"growing-global-swap", func(value *p2gn16TrialPreflight) { value.GlobalSwapUsedAfter++ }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			value := base
			test.edit(&value)
			if err := p2gn16ValidateTrialPreflight(value); err == nil {
				t.Fatal("contaminated preflight did not fail closed")
			}
		})
	}
}

func TestP2GN16MatchedCPUAtomicDirectorySyncFailsClosed(t *testing.T) {
	original := p2gn16DirectorySync
	p2gn16DirectorySync = func(string) error { return errors.New("injected directory sync failure") }
	defer func() { p2gn16DirectorySync = original }()
	err := p2gn16AtomicJSON(filepath.Join(t.TempDir(), "atomic.json"), map[string]bool{"valid": true})
	if err == nil || !strings.Contains(err.Error(), "injected directory sync failure") {
		t.Fatalf("atomic writer did not propagate directory-sync failure: %v", err)
	}
}
