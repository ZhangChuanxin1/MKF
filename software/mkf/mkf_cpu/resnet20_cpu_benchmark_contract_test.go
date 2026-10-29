package mkfcpu

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// This file defines the measurement contract for the ResNet-20 CPU table.
// It intentionally contains no cryptographic backend. A backend may register
// only after its construction identity and timing boundary have been audited.
// In particular, the oracle-instrumented functional executor in
// resnet20_full_executor_test.go is not a performance backend.

const (
	resNet20CPUWarmups = 0
	resNet20CPUTrials  = 10
)

type resNet20CPUStage string

const (
	resNet20CPUStem      resNet20CPUStage = "Stem Convolution"
	resNet20CPUResidual  resNet20CPUStage = "Residual Blocks"
	resNet20CPUFullyConn resNet20CPUStage = "Fully-Connected Layer"
	resNet20CPUTotal     resNet20CPUStage = "Total"
	resNet20CPUParameter                  = "P2"
	resNet20CPUParties                    = 2
)

var resNet20CPUStages = []resNet20CPUStage{
	resNet20CPUStem,
	resNet20CPUResidual,
	resNet20CPUFullyConn,
}

type resNet20CPUPreparedTrial interface {
	// The three Run methods contain only homomorphic evaluation and the
	// protocol operations explicitly included by the backend's frozen
	// measurement-boundary string. Key generation, plaintext encoding,
	// encryption, fixture loading, and correctness-oracle work belong in
	// NewTrial or Verify, never in these methods.
	RunStem() error
	RunResidualBlocks() error
	RunFullyConnected() error
	Verify() error
}

// resNet20CPUPreparedSegment permits a memory-bounded backend to stage one
// frozen network phase at a time. PrepareSegment and Release are invoked
// outside the timer; Run contains only the online homomorphic/protocol work.
// The runner still emits exactly one row for each of the three paper stages by
// summing the non-overlapping Run durations as integer nanoseconds.
type resNet20CPUPreparedSegment interface {
	Run() error
	Release() error
}

type resNet20CPUSegmentedTrial interface {
	resNet20CPUPreparedTrial
	SegmentCount(resNet20CPUStage) int
	PrepareSegment(resNet20CPUStage, int) (resNet20CPUPreparedSegment, error)
}

type resNet20CPUBackend interface {
	Implementation() string
	ConstructionID() string
	ParameterID() string
	Parties() int
	FixtureID() string
	MeasurementBoundary() string
	NewTrial() (resNet20CPUPreparedTrial, error)
}

// resNet20CPUCommunicationArtifactReporter supplies construction-specific,
// JSON-encoded byte-count rules and the corresponding exact serialized-byte
// evidence.  The benchmark refuses to run a real backend that omits this
// interface.  Byte counts are not network-latency measurements.
type resNet20CPUCommunicationArtifactReporter interface {
	CommunicationArtifacts() (policy, evidence json.RawMessage, err error)
}

type resNet20CPUBackendFactory func() (resNet20CPUBackend, error)

var resNet20CPUBackendFactories = map[string]resNet20CPUBackendFactory{}

func registerResNet20CPUBackend(name string, factory resNet20CPUBackendFactory) {
	if name == "" || factory == nil {
		panic("ResNet-20 CPU backend registration requires a name and factory")
	}
	if _, exists := resNet20CPUBackendFactories[name]; exists {
		panic("duplicate ResNet-20 CPU backend registration: " + name)
	}
	resNet20CPUBackendFactories[name] = factory
}

type resNet20CPURawRecord struct {
	Implementation string
	ConstructionID string
	Parameter      string
	Parties        int
	Fixture        string
	Trial          int
	Warmups        int
	Stage          resNet20CPUStage
	ElapsedNS      int64
	Valid          bool
	GOMAXPROCS     int
	AffinityCore   string
	GoVersion      string
	TimestampUTC   string
}

type resNet20CPUSummaryRecord struct {
	Implementation string           `json:"implementation"`
	ConstructionID string           `json:"construction_id"`
	Parameter      string           `json:"parameter"`
	Parties        int              `json:"parties"`
	Fixture        string           `json:"fixture"`
	Stage          resNet20CPUStage `json:"stage"`
	Trials         int              `json:"trials"`
	Warmups        int              `json:"warmups"`
	MeanNS         float64          `json:"mean_ns"`
	SampleStdDevNS float64          `json:"sample_stddev_ns"`
	MinNS          int64            `json:"min_ns"`
	MaxNS          int64            `json:"max_ns"`
}

type resNet20CPUManifest struct {
	Schema                string                     `json:"schema"`
	GeneratedUTC          string                     `json:"generated_utc"`
	GoVersion             string                     `json:"go_version"`
	GOOS                  string                     `json:"goos"`
	GOARCH                string                     `json:"goarch"`
	GOMAXPROCS            int                        `json:"gomaxprocs"`
	AffinityCore          string                     `json:"affinity_core"`
	CPUModel              string                     `json:"cpu_model"`
	LattigoModule         string                     `json:"lattigo_module"`
	SourceManifestSHA256  string                     `json:"source_manifest_sha256"`
	AssetManifestSHA256   string                     `json:"asset_manifest_sha256"`
	ParameterSourceSHA256 string                     `json:"parameter_source_sha256"`
	Warmups               int                        `json:"warmups"`
	Trials                int                        `json:"trials"`
	TotalDefinition       string                     `json:"total_definition"`
	MeasurementBoundary   map[string]string          `json:"measurement_boundary"`
	CommunicationPolicy   map[string]json.RawMessage `json:"communication_byte_policy"`
	CommunicationEvidence map[string]json.RawMessage `json:"communication_byte_evidence"`
	Summary               []resNet20CPUSummaryRecord `json:"summary"`
}

func requiredResNet20BenchmarkEnvironment(name string) (string, error) {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return "", fmt.Errorf("required benchmark evidence variable %s is empty", name)
	}
	return value, nil
}

func requiredResNet20SHA256Environment(name string) (string, error) {
	value, err := requiredResNet20BenchmarkEnvironment(name)
	if err != nil {
		return "", err
	}
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != sha256.Size {
		return "", fmt.Errorf("%s is not a 64-hex SHA-256 digest", name)
	}
	return strings.ToLower(value), nil
}

type resNet20CPURunOptions struct {
	ImplementationNames []string
	RawCSVPath          string
	SummaryJSONPath     string
	RequirePinnedCore   bool
}

func singletonAffinityCore() (string, error) {
	if runtime.GOOS != "linux" {
		return "", fmt.Errorf("CPU affinity audit is implemented only for Linux, got %s", runtime.GOOS)
	}
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "", fmt.Errorf("read /proc/self/status: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "Cpus_allowed_list:") {
			continue
		}
		value := strings.TrimSpace(strings.TrimPrefix(line, "Cpus_allowed_list:"))
		if value == "" || strings.ContainsAny(value, ",- ") {
			return "", fmt.Errorf("process is not pinned to one logical CPU: Cpus_allowed_list=%q", value)
		}
		if _, err := strconv.Atoi(value); err != nil {
			return "", fmt.Errorf("invalid singleton Cpus_allowed_list=%q", value)
		}
		return value, nil
	}
	return "", fmt.Errorf("Cpus_allowed_list is absent from /proc/self/status")
}

func runResNet20CPUBackend(backend resNet20CPUBackend, affinityCore string) ([]resNet20CPURawRecord, error) {
	if backend.ParameterID() != resNet20CPUParameter || backend.Parties() != resNet20CPUParties {
		return nil, fmt.Errorf("%s backend is not the frozen P2/n=2 experiment: parameter=%s parties=%d", backend.Implementation(), backend.ParameterID(), backend.Parties())
	}
	if backend.FixtureID() == "" || backend.ConstructionID() == "" || backend.MeasurementBoundary() == "" {
		return nil, fmt.Errorf("%s backend omitted fixture, construction, or boundary identity", backend.Implementation())
	}
	records := make([]resNet20CPURawRecord, 0, resNet20CPUTrials*4)
	for trialIndex := 1; trialIndex <= resNet20CPUTrials; trialIndex++ {
		// GC, trial cloning, key setup, input/model encryption, and other
		// backend preparation are deliberately outside every timed region.
		runtime.GC()
		prepared, err := backend.NewTrial()
		if err != nil {
			return nil, fmt.Errorf("%s prepare trial %d outside timing: %w", backend.Implementation(), trialIndex, err)
		}
		trialRecords := make([]resNet20CPURawRecord, 0, 4)
		var total int64
		for _, stage := range resNet20CPUStages {
			var elapsed int64
			if segmented, ok := prepared.(resNet20CPUSegmentedTrial); ok {
				segmentCount := segmented.SegmentCount(stage)
				if segmentCount <= 0 {
					return nil, fmt.Errorf("%s trial %d stage %s has invalid segment count %d", backend.Implementation(), trialIndex, stage, segmentCount)
				}
				for segmentIndex := 0; segmentIndex < segmentCount; segmentIndex++ {
					// Disk reads, ciphertext deserialization, and phase-local model
					// materialization belong here, before the timer starts.
					segment, prepareErr := segmented.PrepareSegment(stage, segmentIndex)
					if prepareErr != nil {
						return nil, fmt.Errorf("%s trial %d stage %s prepare segment %d outside timing: %w", backend.Implementation(), trialIndex, stage, segmentIndex, prepareErr)
					}
					if segment == nil {
						return nil, fmt.Errorf("%s trial %d stage %s prepared nil segment %d", backend.Implementation(), trialIndex, stage, segmentIndex)
					}
					start := time.Now()
					err = segment.Run()
					segmentElapsed := time.Since(start).Nanoseconds()
					releaseErr := segment.Release()
					if err != nil {
						return nil, fmt.Errorf("%s trial %d stage %s segment %d: %w", backend.Implementation(), trialIndex, stage, segmentIndex, err)
					}
					if releaseErr != nil {
						return nil, fmt.Errorf("%s trial %d stage %s release segment %d outside timing: %w", backend.Implementation(), trialIndex, stage, segmentIndex, releaseErr)
					}
					if segmentElapsed <= 0 {
						return nil, fmt.Errorf("%s trial %d stage %s segment %d produced non-positive elapsed time", backend.Implementation(), trialIndex, stage, segmentIndex)
					}
					elapsed += segmentElapsed
				}
			} else {
				start := time.Now()
				switch stage {
				case resNet20CPUStem:
					err = prepared.RunStem()
				case resNet20CPUResidual:
					err = prepared.RunResidualBlocks()
				case resNet20CPUFullyConn:
					err = prepared.RunFullyConnected()
				default:
					err = fmt.Errorf("unsupported stage %q", stage)
				}
				elapsed = time.Since(start).Nanoseconds()
			}
			if err != nil {
				return nil, fmt.Errorf("%s trial %d stage %s: %w", backend.Implementation(), trialIndex, stage, err)
			}
			if elapsed <= 0 {
				return nil, fmt.Errorf("%s trial %d stage %s produced non-positive elapsed time", backend.Implementation(), trialIndex, stage)
			}
			total += elapsed
			trialRecords = append(trialRecords, newResNet20CPURawRecord(backend, affinityCore, trialIndex, stage, elapsed))
		}
		// Decryption, decoding, classification comparison, and exact/oracle
		// checks are performed only after all timed stage regions have ended.
		if err := prepared.Verify(); err != nil {
			return nil, fmt.Errorf("%s trial %d untimed correctness verification: %w", backend.Implementation(), trialIndex, err)
		}
		for index := range trialRecords {
			trialRecords[index].Valid = true
		}
		totalRecord := newResNet20CPURawRecord(backend, affinityCore, trialIndex, resNet20CPUTotal, total)
		totalRecord.Valid = true
		trialRecords = append(trialRecords, totalRecord)
		records = append(records, trialRecords...)
	}
	return records, nil
}

func newResNet20CPURawRecord(backend resNet20CPUBackend, affinityCore string, trial int, stage resNet20CPUStage, elapsed int64) resNet20CPURawRecord {
	return resNet20CPURawRecord{
		Implementation: backend.Implementation(), ConstructionID: backend.ConstructionID(),
		Parameter: backend.ParameterID(), Parties: backend.Parties(), Fixture: backend.FixtureID(),
		Trial: trial, Warmups: resNet20CPUWarmups, Stage: stage, ElapsedNS: elapsed,
		GOMAXPROCS: runtime.GOMAXPROCS(0), AffinityCore: affinityCore,
		GoVersion: runtime.Version(), TimestampUTC: time.Now().UTC().Format(time.RFC3339Nano),
	}
}

func summarizeResNet20CPURecords(records []resNet20CPURawRecord) ([]resNet20CPUSummaryRecord, error) {
	type key struct {
		implementation, construction, parameter, fixture, stage string
		parties                                                 int
	}
	groups := make(map[key][]int64)
	seenTrials := make(map[key]map[int]struct{})
	for _, record := range records {
		if !record.Valid || record.Warmups != 0 || record.Trial < 1 || record.Trial > resNet20CPUTrials || record.ElapsedNS <= 0 {
			return nil, fmt.Errorf("invalid raw record: %+v", record)
		}
		group := key{record.Implementation, record.ConstructionID, record.Parameter, record.Fixture, string(record.Stage), record.Parties}
		if seenTrials[group] == nil {
			seenTrials[group] = make(map[int]struct{}, resNet20CPUTrials)
		}
		if _, duplicate := seenTrials[group][record.Trial]; duplicate {
			return nil, fmt.Errorf("duplicate raw record for %s/%s trial %d", group.implementation, group.stage, record.Trial)
		}
		seenTrials[group][record.Trial] = struct{}{}
		groups[group] = append(groups[group], record.ElapsedNS)
	}
	summary := make([]resNet20CPUSummaryRecord, 0, len(groups))
	for group, values := range groups {
		if len(values) != resNet20CPUTrials {
			return nil, fmt.Errorf("%s/%s has %d trials, want %d", group.implementation, group.stage, len(values), resNet20CPUTrials)
		}
		var sum float64
		minimum, maximum := values[0], values[0]
		for _, value := range values {
			sum += float64(value)
			if value < minimum {
				minimum = value
			}
			if value > maximum {
				maximum = value
			}
		}
		mean := sum / float64(len(values))
		var squared float64
		for _, value := range values {
			delta := float64(value) - mean
			squared += delta * delta
		}
		summary = append(summary, resNet20CPUSummaryRecord{
			Implementation: group.implementation, ConstructionID: group.construction,
			Parameter: group.parameter, Parties: group.parties, Fixture: group.fixture,
			Stage: resNet20CPUStage(group.stage), Trials: len(values), Warmups: 0,
			MeanNS: mean, SampleStdDevNS: math.Sqrt(squared / float64(len(values)-1)),
			MinNS: minimum, MaxNS: maximum,
		})
	}
	sort.Slice(summary, func(i, j int) bool {
		if summary[i].Implementation != summary[j].Implementation {
			return summary[i].Implementation < summary[j].Implementation
		}
		return resNet20CPUStageOrder(summary[i].Stage) < resNet20CPUStageOrder(summary[j].Stage)
	})
	return summary, nil
}

func resNet20CPUStageOrder(stage resNet20CPUStage) int {
	switch stage {
	case resNet20CPUStem:
		return 0
	case resNet20CPUResidual:
		return 1
	case resNet20CPUFullyConn:
		return 2
	case resNet20CPUTotal:
		return 3
	default:
		return 4
	}
}

func writeResNet20CPURawCSV(path string, records []resNet20CPURawRecord) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	header := []string{"implementation", "construction_id", "parameter", "parties", "fixture", "trial", "warmups", "stage", "elapsed_ns", "valid", "gomaxprocs", "affinity_core", "go_version", "timestamp_utc"}
	if err := writer.Write(header); err != nil {
		file.Close()
		return err
	}
	for _, record := range records {
		row := []string{
			record.Implementation, record.ConstructionID, record.Parameter, strconv.Itoa(record.Parties), record.Fixture,
			strconv.Itoa(record.Trial), strconv.Itoa(record.Warmups), string(record.Stage), strconv.FormatInt(record.ElapsedNS, 10),
			strconv.FormatBool(record.Valid), strconv.Itoa(record.GOMAXPROCS), record.AffinityCore, record.GoVersion, record.TimestampUTC,
		}
		if err := writer.Write(row); err != nil {
			file.Close()
			return err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func executeResNet20CPUBenchmark(options resNet20CPURunOptions) error {
	if resNet20CPUWarmups != 0 || resNet20CPUTrials != 10 {
		return fmt.Errorf("frozen protocol changed: warmups=%d trials=%d", resNet20CPUWarmups, resNet20CPUTrials)
	}
	if runtime.GOMAXPROCS(0) != 1 {
		return fmt.Errorf("GOMAXPROCS=%d, want exactly 1", runtime.GOMAXPROCS(0))
	}
	affinityCore := "unit-test-unpinned"
	cpuModel := "unit-test"
	lattigoModule := "unit-test"
	sourceManifestSHA256 := strings.Repeat("0", 64)
	assetManifestSHA256 := strings.Repeat("0", 64)
	parameterSourceSHA256 := strings.Repeat("0", 64)
	if options.RequirePinnedCore {
		var err error
		affinityCore, err = singletonAffinityCore()
		if err != nil {
			return err
		}
		if cpuModel, err = requiredResNet20BenchmarkEnvironment("MKF_RESNET20_CPU_MODEL"); err != nil {
			return err
		}
		if lattigoModule, err = requiredResNet20BenchmarkEnvironment("MKF_RESNET20_LATTIGO_MODULE"); err != nil {
			return err
		}
		if sourceManifestSHA256, err = requiredResNet20SHA256Environment("MKF_RESNET20_SOURCE_MANIFEST_SHA256"); err != nil {
			return err
		}
		if assetManifestSHA256, err = requiredResNet20SHA256Environment("MKF_RESNET20_ASSET_MANIFEST_SHA256"); err != nil {
			return err
		}
		if parameterSourceSHA256, err = requiredResNet20SHA256Environment("MKF_RESNET20_PARAMETER_SOURCE_SHA256"); err != nil {
			return err
		}
	}
	if len(options.ImplementationNames) != 2 || options.ImplementationNames[0] != "KKLSS-CPU" || options.ImplementationNames[1] != "MKF-CPU" {
		return fmt.Errorf("matched table requires implementation order [KKLSS-CPU MKF-CPU], got %v", options.ImplementationNames)
	}
	var records []resNet20CPURawRecord
	boundaries := make(map[string]string)
	communicationPolicies := make(map[string]json.RawMessage)
	communicationEvidence := make(map[string]json.RawMessage)
	var fixture string
	for _, name := range options.ImplementationNames {
		factory, ok := resNet20CPUBackendFactories[name]
		if !ok {
			return fmt.Errorf("required %s ResNet-20 backend is not registered; refusing to infer or substitute latency", name)
		}
		backend, err := factory()
		if err != nil {
			return fmt.Errorf("construct %s backend: %w", name, err)
		}
		if backend.Implementation() != name {
			return fmt.Errorf("backend registration %s reports implementation %s", name, backend.Implementation())
		}
		reporter, ok := backend.(resNet20CPUCommunicationArtifactReporter)
		if !ok {
			return fmt.Errorf("%s backend omitted exact communication byte policy/evidence reporting", name)
		}
		if fixture == "" {
			fixture = backend.FixtureID()
		} else if fixture != backend.FixtureID() {
			return fmt.Errorf("unmatched fixtures: first=%s %s=%s", fixture, name, backend.FixtureID())
		}
		backendRecords, err := runResNet20CPUBackend(backend, affinityCore)
		if err != nil {
			return err
		}
		records = append(records, backendRecords...)
		boundaries[name] = backend.MeasurementBoundary()
		policy, evidence, err := reporter.CommunicationArtifacts()
		if err != nil {
			return fmt.Errorf("collect %s communication artifacts: %w", name, err)
		}
		if len(policy) == 0 || !json.Valid(policy) || len(evidence) == 0 || !json.Valid(evidence) {
			return fmt.Errorf("%s returned invalid communication policy/evidence JSON", name)
		}
		communicationPolicies[name] = append(json.RawMessage(nil), policy...)
		communicationEvidence[name] = append(json.RawMessage(nil), evidence...)
	}
	if len(records) != 2*resNet20CPUTrials*4 {
		return fmt.Errorf("raw coverage is %d rows, want 80", len(records))
	}
	summary, err := summarizeResNet20CPURecords(records)
	if err != nil {
		return err
	}
	if len(summary) != 8 {
		return fmt.Errorf("summary coverage is %d rows, want 8", len(summary))
	}
	if err := writeResNet20CPURawCSV(options.RawCSVPath, records); err != nil {
		return fmt.Errorf("write raw CSV: %w", err)
	}
	manifest := resNet20CPUManifest{
		Schema: "mkf-resnet20-cpu-benchmark-v2", GeneratedUTC: time.Now().UTC().Format(time.RFC3339Nano),
		GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		GOMAXPROCS: 1, AffinityCore: affinityCore, CPUModel: cpuModel,
		LattigoModule: lattigoModule, SourceManifestSHA256: sourceManifestSHA256,
		AssetManifestSHA256: assetManifestSHA256, ParameterSourceSHA256: parameterSourceSHA256,
		Warmups: 0, Trials: 10,
		TotalDefinition:     "per trial, exact integer sum of the three non-overlapping measured stage durations",
		MeasurementBoundary: boundaries, CommunicationPolicy: communicationPolicies,
		CommunicationEvidence: communicationEvidence, Summary: summary,
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.WriteFile(options.SummaryJSONPath, data, 0o644); err != nil {
		return fmt.Errorf("write summary JSON: %w", err)
	}
	return nil
}

// The real entry point remains opt-in and fail-closed. Until both audited
// factories are registered, it exits before creating any paper-facing result.
func TestFullResNet20CPUBenchmark(t *testing.T) {
	if os.Getenv("MKF_RUN_FULL_RESNET20_CPU_BENCHMARK") != "1" {
		t.Skip("set MKF_RUN_FULL_RESNET20_CPU_BENCHMARK=1 only after both audited CPU backends are registered")
	}
	rawPath := os.Getenv("MKF_RESNET20_CPU_RAW_CSV")
	summaryPath := os.Getenv("MKF_RESNET20_CPU_SUMMARY_JSON")
	if rawPath == "" || summaryPath == "" {
		t.Fatal("MKF_RESNET20_CPU_RAW_CSV and MKF_RESNET20_CPU_SUMMARY_JSON are required")
	}
	if err := executeResNet20CPUBenchmark(resNet20CPURunOptions{
		ImplementationNames: []string{"KKLSS-CPU", "MKF-CPU"}, RawCSVPath: rawPath,
		SummaryJSONPath: summaryPath, RequirePinnedCore: true,
	}); err != nil {
		t.Fatal(err)
	}
}

type fakeResNet20CPUTrial struct {
	next       int
	verified   bool
	stageDelay time.Duration
}

func (trial *fakeResNet20CPUTrial) run(want int) error {
	if trial.next != want {
		return fmt.Errorf("stage order got %d want %d", trial.next, want)
	}
	trial.next++
	time.Sleep(trial.stageDelay)
	return nil
}

func (trial *fakeResNet20CPUTrial) RunStem() error           { return trial.run(0) }
func (trial *fakeResNet20CPUTrial) RunResidualBlocks() error { return trial.run(1) }
func (trial *fakeResNet20CPUTrial) RunFullyConnected() error { return trial.run(2) }
func (trial *fakeResNet20CPUTrial) Verify() error {
	if trial.next != 3 || trial.verified {
		return fmt.Errorf("verification was not called once after all timed stages")
	}
	trial.verified = true
	return nil
}

type fakeResNet20CPUBackend struct{ implementation string }

func (backend *fakeResNet20CPUBackend) Implementation() string { return backend.implementation }
func (backend *fakeResNet20CPUBackend) ConstructionID() string { return "unit-test-construction" }
func (backend *fakeResNet20CPUBackend) ParameterID() string    { return "P2" }
func (backend *fakeResNet20CPUBackend) Parties() int           { return 2 }
func (backend *fakeResNet20CPUBackend) FixtureID() string      { return "unit-test-fixture" }
func (backend *fakeResNet20CPUBackend) MeasurementBoundary() string {
	return "unit-test online evaluation only"
}
func (backend *fakeResNet20CPUBackend) NewTrial() (resNet20CPUPreparedTrial, error) {
	return &fakeResNet20CPUTrial{stageDelay: time.Microsecond}, nil
}
func (backend *fakeResNet20CPUBackend) CommunicationArtifacts() (json.RawMessage, json.RawMessage, error) {
	return json.RawMessage(`{"schema":"unit-test-byte-policy","not_measured":true}`),
		json.RawMessage(`{"schema":"unit-test-byte-evidence","not_measured":true}`), nil
}

type fakeSegmentedResNet20CPUTrial struct {
	nextPrepared int
	nextRun      int
	nextReleased int
}

type fakeResNet20CPUSegment struct {
	trial *fakeSegmentedResNet20CPUTrial
	index int
}

func (segment *fakeResNet20CPUSegment) Run() error {
	if segment.trial.nextRun != segment.index || segment.trial.nextPrepared <= segment.index {
		return fmt.Errorf("segment %d ran before ordered untimed preparation", segment.index)
	}
	segment.trial.nextRun++
	time.Sleep(time.Microsecond)
	return nil
}

func (segment *fakeResNet20CPUSegment) Release() error {
	if segment.trial.nextReleased != segment.index || segment.trial.nextRun <= segment.index {
		return fmt.Errorf("segment %d released before its timed run", segment.index)
	}
	segment.trial.nextReleased++
	return nil
}

func (trial *fakeSegmentedResNet20CPUTrial) RunStem() error {
	return fmt.Errorf("segmented trial must not call coarse RunStem")
}
func (trial *fakeSegmentedResNet20CPUTrial) RunResidualBlocks() error {
	return fmt.Errorf("segmented trial must not call coarse RunResidualBlocks")
}
func (trial *fakeSegmentedResNet20CPUTrial) RunFullyConnected() error {
	return fmt.Errorf("segmented trial must not call coarse RunFullyConnected")
}
func (trial *fakeSegmentedResNet20CPUTrial) SegmentCount(stage resNet20CPUStage) int {
	switch stage {
	case resNet20CPUStem, resNet20CPUFullyConn:
		return 1
	case resNet20CPUResidual:
		return 2
	default:
		return 0
	}
}
func (trial *fakeSegmentedResNet20CPUTrial) PrepareSegment(_ resNet20CPUStage, _ int) (resNet20CPUPreparedSegment, error) {
	index := trial.nextPrepared
	trial.nextPrepared++
	return &fakeResNet20CPUSegment{trial: trial, index: index}, nil
}
func (trial *fakeSegmentedResNet20CPUTrial) Verify() error {
	if trial.nextPrepared != 4 || trial.nextRun != 4 || trial.nextReleased != 4 {
		return fmt.Errorf("segmented lifecycle incomplete: prepared=%d run=%d released=%d", trial.nextPrepared, trial.nextRun, trial.nextReleased)
	}
	return nil
}

type fakeSegmentedResNet20CPUBackend struct{ fakeResNet20CPUBackend }

func (backend *fakeSegmentedResNet20CPUBackend) NewTrial() (resNet20CPUPreparedTrial, error) {
	return &fakeSegmentedResNet20CPUTrial{}, nil
}

func TestFullResNet20CPUSegmentedPreparationContract(t *testing.T) {
	backend := &fakeSegmentedResNet20CPUBackend{fakeResNet20CPUBackend{implementation: "KKLSS-CPU"}}
	records, err := runResNet20CPUBackend(backend, "unit-test-unpinned")
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != resNet20CPUTrials*4 {
		t.Fatalf("segmented runner emitted %d rows, want %d", len(records), resNet20CPUTrials*4)
	}
}

func TestFullResNet20CPUBenchmarkContract(t *testing.T) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	original := resNet20CPUBackendFactories
	resNet20CPUBackendFactories = map[string]resNet20CPUBackendFactory{
		"KKLSS-CPU": func() (resNet20CPUBackend, error) { return &fakeResNet20CPUBackend{implementation: "KKLSS-CPU"}, nil },
		"MKF-CPU":   func() (resNet20CPUBackend, error) { return &fakeResNet20CPUBackend{implementation: "MKF-CPU"}, nil },
	}
	defer func() { resNet20CPUBackendFactories = original }()
	directory := t.TempDir()
	rawPath, summaryPath := directory+string(os.PathSeparator)+"raw.csv", directory+string(os.PathSeparator)+"summary.json"
	if err := executeResNet20CPUBenchmark(resNet20CPURunOptions{
		ImplementationNames: []string{"KKLSS-CPU", "MKF-CPU"}, RawCSVPath: rawPath,
		SummaryJSONPath: summaryPath, RequirePinnedCore: false,
	}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(rawPath)
	if err != nil {
		t.Fatal(err)
	}
	if rows := len(strings.Split(strings.TrimSpace(string(raw)), "\n")) - 1; rows != 80 {
		t.Fatalf("raw CSV has %d rows, want 80", rows)
	}
	var manifest resNet20CPUManifest
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Warmups != 0 || manifest.Trials != 10 || len(manifest.Summary) != 8 || len(manifest.CommunicationPolicy) != 2 || len(manifest.CommunicationEvidence) != 2 {
		t.Fatalf("unexpected benchmark manifest: warmups=%d trials=%d summary=%d policies=%d evidence=%d", manifest.Warmups, manifest.Trials, len(manifest.Summary), len(manifest.CommunicationPolicy), len(manifest.CommunicationEvidence))
	}
	if manifest.CPUModel != "unit-test" || manifest.LattigoModule != "unit-test" || len(manifest.SourceManifestSHA256) != 64 || len(manifest.AssetManifestSHA256) != 64 || len(manifest.ParameterSourceSHA256) != 64 {
		t.Fatalf("benchmark provenance manifest is incomplete: cpu=%q lattigo=%q source=%q assets=%q params=%q", manifest.CPUModel, manifest.LattigoModule, manifest.SourceManifestSHA256, manifest.AssetManifestSHA256, manifest.ParameterSourceSHA256)
	}
}

func TestFullResNet20CPUBenchmarkFailsClosedWithoutRealBackends(t *testing.T) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	original := resNet20CPUBackendFactories
	resNet20CPUBackendFactories = map[string]resNet20CPUBackendFactory{}
	defer func() { resNet20CPUBackendFactories = original }()
	err := executeResNet20CPUBenchmark(resNet20CPURunOptions{
		ImplementationNames: []string{"KKLSS-CPU", "MKF-CPU"},
		RawCSVPath:          t.TempDir() + string(os.PathSeparator) + "raw.csv",
		SummaryJSONPath:     t.TempDir() + string(os.PathSeparator) + "summary.json",
	})
	if err == nil || !strings.Contains(err.Error(), "KKLSS-CPU ResNet-20 backend is not registered") {
		t.Fatalf("missing backend did not fail closed: %v", err)
	}
}
