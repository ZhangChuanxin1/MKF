package p2gn16staticexport

// Package p2gn16staticexport implements the production, fail-closed export of
// the 341 dense CKKS plaintext masks and seven fractional MultByConst
// operands used by the canonical P2GN16 ResNet-20 schedule.  It deliberately
// emits no board- or performance-eligibility claim.

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

const (
	ProductionSchema       = "mkf-p2gn16-static-public-operand-export-v2"
	ProductionStatus       = "PASS_EXACT_SEMANTIC_REENCODING_RELEASE_STILL_BLOCKED"
	ProductionEvidence     = "sha_bound_lattigo_v2.3_static_encoder_export"
	ProductionParameter    = "P2GN16"
	ProductionConstruction = "MKF-CKKS-P2GN16-guarded-fold-once+permanent-aggregate-rq2-v1"
	TranscriptSchema       = "mkf-p2gn16-static-encoding-transcript-v1"
	LattigoModule          = "github.com/ldsec/lattigo/v2"
	LattigoVersion         = "v2.3.0"
	LattigoModuleSum       = "h1:5bG7CqH0dzkdnCf4bDGLkl+G3HrF4obV6flN7LMfrNc="
	ParameterLiteralSHA    = "2ec6e579b7dd3fb53c771abadac2a95e522212d275b102732e11b1cf62b7f025"
	ParameterIdentitySHA   = "6cef45879aa5569bc860a887606682dd884cb49d5a3f4bd6265ce3b724bb2be6"
	SemanticCatalogSHA     = "386726a2bee3487e35a5bcbcb6cd40e7cf2286ce4ffe24e78713f36afeddfec4"
	CommandStreamSHA       = "d40d0a7ad0814fef4f224ae66bc194574c38b3fda3c6afaa358b29dd3ac865e5"
	ScaleOracleSHA         = "7dd2cf831feec964ecf543e0d7d459c2ba42a3dc301a2fb4adc22559b9a8d184"
	ExpectedReferences     = 348
	ExpectedDense          = 341
	ExpectedResidual       = 7
)

var coreSourceSHA256 = map[string]string{
	"ckks/ciphertext.go":        "4495668c451fe1a2545adef9e58b45aa211b93c1f3a7c24a0c71b7cbdf9d6d22",
	"ckks/encoder.go":           "fb11b2a2b188c1eb40b693c7d6d853a907a7c567d70c6ebf3052f7e7ec137f7e",
	"ckks/evaluator.go":         "6a8ba5ade1f4ffdc6e73b3f89ca798e362932625b543ab0abe0aae389948e57c",
	"ckks/params.go":            "0fd10b9a2891729d976830a5d352205d9db9bf16aa00ac08d3536b3f839af2be",
	"ckks/plaintext.go":         "42ac8b752b5a36d1978e59035974000d27928a73f5dd676d11dc926dcc95e19d",
	"ring/modular_reduction.go": "1cdb7acbba1720df9df10a023dbd8356553023140d3210f1c35c19229d14d36c",
	"ring/ring.go":              "0ce081d1ca23da28781adf3339a226879568bd131f9dc85a21d9385a630cd454",
	"ring/ring_ntt.go":          "f5ff4592c5744c1789b065f24b41cdeab403ebc8069bfca5042db77642b56529",
	"ring/ring_operations.go":   "dbcba0e5badd21d528ce2ec3a6e78d2efba021afa64edf4c12acf9996bf16272",
	"rlwe/params.go":            "b29366be12b53b3e800af072bc765e04bc3c737c57d96f7c2c8799a27ac86e9a",
}

type Authority struct {
	Label          string
	Path           string
	ExpectedSHA256 string
}

type Config struct {
	RepoRoot            string
	OutputRoot          string
	ParameterLiteral    Authority
	SemanticCatalog     Authority
	CommandStream       Authority
	ScaleOracle         Authority
	ExporterSourcePaths []string
	Schema              string
	Status              string
	EvidenceClass       string
	Parameter           string
	Construction        string
	ExpectedReferences  int
	ExpectedDense       int
	ExpectedResidual    int
	StrictP2GN16        bool
}

func ProductionConfig(repoRoot, outputRoot string) Config {
	return Config{
		RepoRoot:   repoRoot,
		OutputRoot: outputRoot,
		ParameterLiteral: Authority{
			Label:          "experiments/resnet20_p3/security/p2g_n16_candidate_literal_20260826.json",
			Path:           filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/security/p2g_n16_candidate_literal_20260826.json")),
			ExpectedSHA256: ParameterLiteralSHA,
		},
		SemanticCatalog: Authority{
			Label:          "experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_public_operand_semantic_catalog.json",
			Path:           filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_public_operand_semantic_catalog.json")),
			ExpectedSHA256: SemanticCatalogSHA,
		},
		CommandStream: Authority{
			Label:          "experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_resnet20_operation_command_stream.calibrated_margin070.json",
			Path:           filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_resnet20_operation_command_stream.calibrated_margin070.json")),
			ExpectedSHA256: CommandStreamSHA,
		},
		ScaleOracle: Authority{
			Label:          "tmp/cpu_experiments/experiments/resnet20_p3/fpga_scale_audit/p2gn16_lattigo_v23_scale_encoding_oracle.json",
			Path:           filepath.Join(repoRoot, filepath.FromSlash("tmp/cpu_experiments/experiments/resnet20_p3/fpga_scale_audit/p2gn16_lattigo_v23_scale_encoding_oracle.json")),
			ExpectedSHA256: ScaleOracleSHA,
		},
		ExporterSourcePaths: []string{
			"tmp/cpu_experiments/cmd/p2gn16_static_export/main.go",
			"tmp/cpu_experiments/internal/p2gn16staticexport/export.go",
			"tmp/cpu_experiments/go.mod",
			"tmp/cpu_experiments/go.sum",
		},
		Schema: ProductionSchema, Status: ProductionStatus,
		EvidenceClass: ProductionEvidence, Parameter: ProductionParameter,
		Construction:       ProductionConstruction,
		ExpectedReferences: ExpectedReferences, ExpectedDense: ExpectedDense,
		ExpectedResidual: ExpectedResidual, StrictP2GN16: true,
	}
}

type Binding struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type PayloadBinding struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type Entry struct {
	ReferenceSequence          int            `json:"reference_sequence"`
	CommandSequence            int            `json:"command_sequence"`
	Opcode                     string         `json:"opcode"`
	Purpose                    string         `json:"purpose"`
	Phase                      string         `json:"phase"`
	Boundary                   string         `json:"boundary"`
	Operation                  string         `json:"operation"`
	OperandClass               string         `json:"operand_class"`
	SemanticKind               string         `json:"semantic_kind"`
	SemanticPayloadID          string         `json:"semantic_payload_id"`
	SemanticPayload            PayloadBinding `json:"semantic_payload"`
	ActiveLevel                int            `json:"active_level"`
	OperandScale               float64        `json:"operand_scale"`
	OperandScaleFloat64BitsHex string         `json:"operand_scale_float64_bits_hex"`
	OperandScaleLog2Q3232      int64          `json:"operand_scale_log2_q32_32"`
	LattigoAPIBoundary         string         `json:"lattigo_api_boundary"`
	Standard                   PayloadBinding `json:"standard"`
	Montgomery                 PayloadBinding `json:"montgomery"`
}

type Manifest struct {
	Schema              string             `json:"schema"`
	Status              string             `json:"status"`
	Parameter           string             `json:"parameter"`
	Construction        string             `json:"construction"`
	EvidenceClass       string             `json:"evidence_class"`
	ObserverMode        string             `json:"observer_mode"`
	PerformanceMeasured bool               `json:"performance_measured"`
	SourceBindings      map[string]Binding `json:"source_bindings"`
	ParameterLiteral    struct {
		Binding         Binding  `json:"binding"`
		IdentitySHA256  string   `json:"identity_sha256"`
		LogN            int      `json:"log_n"`
		N               int      `json:"n"`
		LogSlots        int      `json:"log_slots"`
		Scale           float64  `json:"scale"`
		Sigma           float64  `json:"sigma"`
		Q               []uint64 `json:"q"`
		P               []uint64 `json:"p"`
		PersistentLevel int      `json:"persistent_max_level"`
		GuardedLevel    int      `json:"guarded_max_level"`
		Alpha           int      `json:"alpha"`
		Beta            int      `json:"beta"`
	} `json:"parameter_literal"`
	Toolchain struct {
		GoVersion                  string    `json:"go_version"`
		GOOS                       string    `json:"goos"`
		GOARCH                     string    `json:"goarch"`
		LattigoModule              string    `json:"lattigo_module"`
		LattigoVersion             string    `json:"lattigo_version"`
		LattigoModuleSum           string    `json:"lattigo_module_sum"`
		LattigoIdentityMethod      string    `json:"lattigo_identity_method"`
		CoreSourceBindings         []Binding `json:"core_source_bindings"`
		ExporterSourceBindings     []Binding `json:"exporter_source_bindings"`
		ExporterSourceAggregateSHA string    `json:"exporter_source_aggregate_sha256"`
	} `json:"toolchain"`
	Coverage struct {
		References               int `json:"references"`
		DenseMaskReferences      int `json:"dense_mask_references"`
		FractionalMultReferences int `json:"fractional_mult_references"`
		UniqueStandardObjects    int `json:"unique_standard_objects"`
		UniqueMontgomeryObjects  int `json:"unique_montgomery_objects"`
	} `json:"coverage"`
	EncodingTranscript struct {
		Schema    string `json:"schema"`
		Path      string `json:"path"`
		Bytes     int64  `json:"bytes"`
		SHA256    string `json:"sha256"`
		LineCount int    `json:"line_count"`
	} `json:"encoding_transcript"`
	Entries      []Entry `json:"entries"`
	AtomicCommit struct {
		Complete bool   `json:"complete"`
		Method   string `json:"method"`
	} `json:"atomic_commit"`
	ReleaseState struct {
		StaticExportComplete bool `json:"static_export_complete"`
		KernelIntegrated     bool `json:"kernel_integrated"`
		BoardEligible        bool `json:"board_eligible"`
		MeasurementEligible  bool `json:"measurement_eligible"`
		PerformanceMeasured  bool `json:"performance_measured"`
	} `json:"release_state"`
}

type parameterAuthority struct {
	Schema    string `json:"schema"`
	Generator struct {
		Dependency string `json:"dependency"`
		ModuleSum  string `json:"module_sum"`
	} `json:"generator"`
	Literal struct {
		IdentitySHA256    string   `json:"identity_sha256"`
		LogN              int      `json:"log_n"`
		N                 int      `json:"n"`
		LogSlots          int      `json:"log_slots"`
		SlotsUsed         int      `json:"slots_used"`
		Scale             float64  `json:"scale"`
		Sigma             float64  `json:"sigma"`
		QPost             []string `json:"q_post_conversion"`
		QGuard            string   `json:"q_guard"`
		P                 []string `json:"p"`
		PersistentMax     int      `json:"persistent_max_level"`
		GuardedMax        int      `json:"guarded_max_level"`
		Alpha             int      `json:"alpha"`
		Beta              int      `json:"beta"`
		AllPrimesPrime    bool     `json:"all_primes_prime"`
		AllPrimesNTTPrime bool     `json:"all_primes_congruent_one_mod_2n"`
	} `json:"literal"`
}

type semanticPayload struct {
	PayloadID     string `json:"payload_id"`
	Kind          string `json:"kind"`
	Encoding      string `json:"encoding"`
	Path          string `json:"path"`
	PayloadBytes  int64  `json:"payload_bytes"`
	ContentSHA256 string `json:"content_sha256"`
	AllZero       bool   `json:"all_zero"`
	LogicalSlots  int    `json:"logical_slots"`
}

type semanticReference struct {
	ReferenceSequence    int     `json:"reference_sequence"`
	CommandSequence      int     `json:"command_sequence"`
	Opcode               string  `json:"opcode"`
	Purpose              string  `json:"purpose"`
	Phase                string  `json:"phase"`
	Boundary             string  `json:"boundary"`
	Operation            string  `json:"operation"`
	ActiveLevel          int     `json:"active_level"`
	SemanticKind         string  `json:"semantic_kind"`
	SemanticPayloadID    string  `json:"semantic_payload_id"`
	SemanticPayloadExact bool    `json:"semantic_payload_exact"`
	OperandScale         float64 `json:"operand_scale"`
}

type semanticCatalog struct {
	Schema              string              `json:"schema"`
	Status              string              `json:"status"`
	Parameter           string              `json:"parameter"`
	Construction        string              `json:"construction"`
	PerformanceMeasured bool                `json:"performance_measured"`
	Payloads            []semanticPayload   `json:"payloads"`
	References          []semanticReference `json:"references"`
	ReleaseState        map[string]bool     `json:"release_state"`
}

type commandEvent struct {
	Sequence    int    `json:"sequence"`
	Phase       string `json:"phase"`
	Boundary    string `json:"boundary"`
	Opcode      string `json:"opcode"`
	Operation   string `json:"operation"`
	Purpose     string `json:"purpose"`
	InputLevels []int  `json:"input_levels"`
	OutputLevel int    `json:"output_level"`
}

type commandAuthority struct {
	Schema              string `json:"schema"`
	Status              string `json:"status"`
	Parameter           string `json:"parameter"`
	Construction        string `json:"construction"`
	PerformanceMeasured bool   `json:"performance_measured"`
	BoardEligible       bool   `json:"board_eligible"`
	CommandStream       struct {
		Commands []commandEvent `json:"commands"`
	} `json:"command_stream"`
}

type preparedReference struct {
	Reference semanticReference
	Payload   semanticPayload
	Data      []byte
	Class     string
	Key       string
}

func sha256Bytes(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func sha256File(path string) (string, int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()
	digest := sha256.New()
	count, err := io.Copy(digest, file)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(digest.Sum(nil)), count, nil
}

func readBound(authority Authority) ([]byte, Binding, error) {
	info, err := os.Lstat(authority.Path)
	if err != nil {
		return nil, Binding{}, fmt.Errorf("missing authority %s: %w", authority.Label, err)
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, Binding{}, fmt.Errorf("authority %s is not a regular non-symlink file", authority.Label)
	}
	data, err := os.ReadFile(authority.Path)
	if err != nil {
		return nil, Binding{}, err
	}
	digest := sha256Bytes(data)
	if authority.ExpectedSHA256 != "" && digest != strings.ToLower(authority.ExpectedSHA256) {
		return nil, Binding{}, fmt.Errorf("authority digest drift for %s: %s", authority.Label, digest)
	}
	return data, Binding{Path: authority.Label, Bytes: int64(len(data)), SHA256: digest}, nil
}

func safeRegular(root, relative string) (string, error) {
	if relative == "" || filepath.IsAbs(relative) {
		return "", fmt.Errorf("unsafe relative path %q", relative)
	}
	clean := filepath.Clean(filepath.FromSlash(relative))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe relative path %q", relative)
	}
	absolute := filepath.Join(root, clean)
	rel, err := filepath.Rel(root, absolute)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes root: %q", relative)
	}
	info, err := os.Lstat(absolute)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("path is not a regular non-symlink file: %q", relative)
	}
	return absolute, nil
}

func parseHexWords(values []string) ([]uint64, error) {
	result := make([]uint64, len(values))
	for index, value := range values {
		parsed, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(value), "0x"), 16, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid modulus %q: %w", value, err)
		}
		result[index] = parsed
	}
	return result, nil
}

func parameters(authority parameterAuthority, strict bool) (ckks.Parameters, []uint64, []uint64, error) {
	q, err := parseHexWords(authority.Literal.QPost)
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	guard, err := parseHexWords([]string{authority.Literal.QGuard})
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	q = append(q, guard[0])
	p, err := parseHexWords(authority.Literal.P)
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	if strict {
		if authority.Schema != "mkf-p2g-n16-security-candidate-v1" || authority.Generator.Dependency != LattigoModule+" "+LattigoVersion || authority.Generator.ModuleSum != LattigoModuleSum || authority.Literal.IdentitySHA256 != ParameterIdentitySHA {
			return ckks.Parameters{}, nil, nil, fmt.Errorf("P2GN16 parameter authority identity drift")
		}
		if authority.Literal.LogN != 16 || authority.Literal.N != 65536 || authority.Literal.LogSlots != 14 || authority.Literal.SlotsUsed != 16384 || len(q) != 15 || len(p) != 2 || authority.Literal.PersistentMax != 13 || authority.Literal.GuardedMax != 14 || authority.Literal.Alpha != 2 || authority.Literal.Beta != 8 || !authority.Literal.AllPrimesPrime || !authority.Literal.AllPrimesNTTPrime {
			return ckks.Parameters{}, nil, nil, fmt.Errorf("P2GN16 parameter literal shape drift")
		}
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: authority.Literal.LogN, LogSlots: authority.Literal.LogSlots,
		Q: q, P: p, Scale: authority.Literal.Scale, Sigma: authority.Literal.Sigma,
	})
	if err != nil {
		return ckks.Parameters{}, nil, nil, fmt.Errorf("initialize Lattigo parameters: %w", err)
	}
	if params.N() != authority.Literal.N || params.MaxLevel() != authority.Literal.GuardedMax || params.LogSlots() != authority.Literal.LogSlots || params.Scale() != authority.Literal.Scale || !reflect.DeepEqual(params.Q(), q) || !reflect.DeepEqual(params.P(), p) {
		return ckks.Parameters{}, nil, nil, fmt.Errorf("linked Lattigo parameter materialization drift")
	}
	return params, q, p, nil
}

func verifyGoSum(repoRoot string) error {
	path, err := safeRegular(repoRoot, "tmp/cpu_experiments/go.sum")
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	want := LattigoModule + " " + LattigoVersion + " " + LattigoModuleSum
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == want {
			return nil
		}
	}
	return fmt.Errorf("go.sum lacks exact Lattigo module sum")
}

func linkedLattigo(repoRoot string, exporterPaths []string) (goVersion, identityMethod string, core, exporter []Binding, aggregate string, err error) {
	identityMethod = "DEPENDENCY_ENTRY_UNAVAILABLE_GO_SUM_AND_CORE_SOURCE_VERIFIED"
	if info, ok := debug.ReadBuildInfo(); ok {
		found := false
		for _, dependency := range info.Deps {
			if dependency.Path == LattigoModule {
				found = true
				if dependency.Version != LattigoVersion || dependency.Sum != LattigoModuleSum {
					return "", "", nil, nil, "", fmt.Errorf("linked Lattigo identity=%s/%s", dependency.Version, dependency.Sum)
				}
			}
		}
		if found {
			identityMethod = "DEPENDENCY_VERSION_AND_SUM_VERIFIED"
		}
	} else {
		identityMethod = "BUILD_INFO_UNAVAILABLE_GO_SUM_AND_CORE_SOURCE_VERIFIED"
	}
	if err := verifyGoSum(repoRoot); err != nil {
		return "", "", nil, nil, "", err
	}
	function := runtime.FuncForPC(reflect.ValueOf(ckks.NewEncoder).Pointer())
	if function == nil {
		return "", "", nil, nil, "", fmt.Errorf("cannot locate linked Lattigo source")
	}
	encoderPath, _ := function.FileLine(function.Entry())
	moduleRoot := filepath.Dir(filepath.Dir(encoderPath))
	corePaths := make([]string, 0, len(coreSourceSHA256))
	for relative := range coreSourceSHA256 {
		corePaths = append(corePaths, relative)
	}
	sort.Strings(corePaths)
	for _, relative := range corePaths {
		path := filepath.Join(moduleRoot, filepath.FromSlash(relative))
		digest, bytes, hashErr := sha256File(path)
		if hashErr != nil {
			return "", "", nil, nil, "", hashErr
		}
		if digest != coreSourceSHA256[relative] {
			return "", "", nil, nil, "", fmt.Errorf("linked Lattigo source drift: %s", relative)
		}
		core = append(core, Binding{Path: relative, Bytes: bytes, SHA256: digest})
	}
	for _, relative := range exporterPaths {
		path, safeErr := safeRegular(repoRoot, relative)
		if safeErr != nil {
			return "", "", nil, nil, "", safeErr
		}
		digest, bytes, hashErr := sha256File(path)
		if hashErr != nil {
			return "", "", nil, nil, "", hashErr
		}
		exporter = append(exporter, Binding{Path: filepath.ToSlash(relative), Bytes: bytes, SHA256: digest})
	}
	sort.Slice(exporter, func(i, j int) bool { return exporter[i].Path < exporter[j].Path })
	encoded, marshalErr := json.Marshal(exporter)
	if marshalErr != nil {
		return "", "", nil, nil, "", marshalErr
	}
	return runtime.Version(), identityMethod, core, exporter, sha256Bytes(encoded), nil
}

func validateCommand(reference semanticReference, command commandEvent) error {
	if reference.CommandSequence != command.Sequence || reference.Opcode != command.Opcode || reference.Purpose != command.Purpose || reference.Phase != command.Phase || reference.Boundary != command.Boundary || reference.Operation != command.Operation || command.OutputLevel != reference.ActiveLevel {
		return fmt.Errorf("semantic/command binding drift at reference %d", reference.ReferenceSequence)
	}
	if len(command.InputLevels) == 0 || command.InputLevels[0] != reference.ActiveLevel {
		return fmt.Errorf("active input level drift at reference %d", reference.ReferenceSequence)
	}
	return nil
}

func prepare(cfg Config, catalogData, commandData []byte, params ckks.Parameters) (semanticCatalog, []preparedReference, error) {
	var catalog semanticCatalog
	if err := json.Unmarshal(catalogData, &catalog); err != nil {
		return catalog, nil, fmt.Errorf("decode semantic catalog: %w", err)
	}
	if catalog.Parameter != cfg.Parameter || catalog.Construction != cfg.Construction || catalog.PerformanceMeasured {
		return catalog, nil, fmt.Errorf("semantic catalog identity/release drift")
	}
	for flag, value := range catalog.ReleaseState {
		if value && (flag == "hardware_payload_complete" || flag == "board_eligible" || flag == "performance_measured") {
			return catalog, nil, fmt.Errorf("semantic catalog unexpectedly enables %s", flag)
		}
	}
	var commands commandAuthority
	if err := json.Unmarshal(commandData, &commands); err != nil {
		return catalog, nil, fmt.Errorf("decode command authority: %w", err)
	}
	if commands.Parameter != cfg.Parameter || commands.Construction != cfg.Construction || commands.PerformanceMeasured || commands.BoardEligible {
		return catalog, nil, fmt.Errorf("command authority identity/release drift")
	}
	byCommand := make(map[int]commandEvent, len(commands.CommandStream.Commands))
	for _, command := range commands.CommandStream.Commands {
		if _, exists := byCommand[command.Sequence]; exists {
			return catalog, nil, fmt.Errorf("duplicate command sequence %d", command.Sequence)
		}
		byCommand[command.Sequence] = command
	}
	payloads := make(map[string]semanticPayload, len(catalog.Payloads))
	for _, payload := range catalog.Payloads {
		if payload.PayloadID == "" {
			return catalog, nil, fmt.Errorf("empty semantic payload id")
		}
		if _, exists := payloads[payload.PayloadID]; exists {
			return catalog, nil, fmt.Errorf("duplicate semantic payload id %s", payload.PayloadID)
		}
		payloads[payload.PayloadID] = payload
	}
	seenReferences := make(map[int]bool)
	dense, residual := 0, 0
	var prepared []preparedReference
	semanticRoot := filepath.Dir(cfg.SemanticCatalog.Path)
	payloadData := make(map[string][]byte, len(payloads))
	for _, reference := range catalog.References {
		if seenReferences[reference.ReferenceSequence] {
			return catalog, nil, fmt.Errorf("duplicate semantic reference %d", reference.ReferenceSequence)
		}
		seenReferences[reference.ReferenceSequence] = true
		selectedDense := reference.SemanticKind != "semantic_complex_scalar"
		selectedResidual := reference.Opcode == "plaintext_mul" && reference.Purpose == "residual"
		if !selectedDense && !selectedResidual {
			continue
		}
		if !reference.SemanticPayloadExact || reference.ActiveLevel < 0 || reference.ActiveLevel > params.MaxLevel() || !isFinitePositive(reference.OperandScale) {
			return catalog, nil, fmt.Errorf("invalid selected semantic reference %d", reference.ReferenceSequence)
		}
		command, exists := byCommand[reference.CommandSequence]
		if !exists {
			return catalog, nil, fmt.Errorf("missing command %d", reference.CommandSequence)
		}
		if err := validateCommand(reference, command); err != nil {
			return catalog, nil, err
		}
		payload, exists := payloads[reference.SemanticPayloadID]
		if !exists || payload.Kind != reference.SemanticKind {
			return catalog, nil, fmt.Errorf("semantic payload binding drift at reference %d", reference.ReferenceSequence)
		}
		data, cached := payloadData[payload.PayloadID]
		if !cached {
			path, err := safeRegular(semanticRoot, payload.Path)
			if err != nil {
				return catalog, nil, fmt.Errorf("semantic payload %s: %w", payload.PayloadID, err)
			}
			data, err = os.ReadFile(path)
			if err != nil {
				return catalog, nil, err
			}
			if int64(len(data)) != payload.PayloadBytes || sha256Bytes(data) != payload.ContentSHA256 {
				return catalog, nil, fmt.Errorf("semantic payload content drift: %s", payload.PayloadID)
			}
			payloadData[payload.PayloadID] = data
		}
		class := "fractional_mult_by_const"
		if selectedDense {
			class = "dense_slot_mask"
			dense++
			if payload.Encoding != "complex128_slots_real_imag_float64_bits_little_endian_v1" || payload.LogicalSlots != 1<<params.LogSlots() || len(data) != payload.LogicalSlots*16 {
				return catalog, nil, fmt.Errorf("dense semantic payload shape drift: %s", payload.PayloadID)
			}
		} else {
			residual++
			if payload.Encoding != "complex128_real_imag_float64_bits_little_endian_v1" || len(data) != 16 {
				return catalog, nil, fmt.Errorf("residual semantic payload shape drift: %s", payload.PayloadID)
			}
		}
		key := strings.Join([]string{class, payload.ContentSHA256, strconv.Itoa(reference.ActiveLevel), fmt.Sprintf("%016x", math.Float64bits(reference.OperandScale)), strconv.FormatInt(scaleQ32(reference.OperandScale), 10)}, ":")
		prepared = append(prepared, preparedReference{Reference: reference, Payload: payload, Data: data, Class: class, Key: key})
	}
	if len(prepared) != cfg.ExpectedReferences || dense != cfg.ExpectedDense || residual != cfg.ExpectedResidual {
		return catalog, nil, fmt.Errorf("static semantic coverage=%d/%d/%d, want %d/%d/%d", len(prepared), dense, residual, cfg.ExpectedReferences, cfg.ExpectedDense, cfg.ExpectedResidual)
	}
	sort.Slice(prepared, func(i, j int) bool {
		return prepared[i].Reference.ReferenceSequence < prepared[j].Reference.ReferenceSequence
	})
	return catalog, prepared, nil
}

func isFinitePositive(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0
}

func scaleQ32(scale float64) int64 {
	return int64(math.Round(math.Log2(scale) * float64(uint64(1)<<32)))
}

func decodeComplexSlots(data []byte) ([]complex128, error) {
	if len(data)%16 != 0 {
		return nil, fmt.Errorf("complex slot payload length is not divisible by 16")
	}
	values := make([]complex128, len(data)/16)
	for index := range values {
		realBits := binary.LittleEndian.Uint64(data[index*16:])
		imagBits := binary.LittleEndian.Uint64(data[index*16+8:])
		values[index] = complex(math.Float64frombits(realBits), math.Float64frombits(imagBits))
		if math.IsNaN(real(values[index])) || math.IsInf(real(values[index]), 0) || math.IsNaN(imag(values[index])) || math.IsInf(imag(values[index]), 0) {
			return nil, fmt.Errorf("non-finite semantic complex value at slot %d", index)
		}
	}
	return values, nil
}

func serializeDense(poly *ring.Poly, level int) []byte {
	buffer := make([]byte, 8*(level+1)*len(poly.Coeffs[0]))
	offset := 0
	for limb := 0; limb <= level; limb++ {
		for _, coefficient := range poly.Coeffs[limb] {
			binary.LittleEndian.PutUint64(buffer[offset:], coefficient)
			offset += 8
		}
	}
	return buffer
}

func montgomeryPayload(standard []byte, ringQ *ring.Ring, level, wordsPerLimb int) ([]byte, error) {
	expected := 8 * (level + 1) * wordsPerLimb
	if len(standard) != expected {
		return nil, fmt.Errorf("standard payload bytes=%d, want %d", len(standard), expected)
	}
	montgomery := make([]byte, len(standard))
	for limb := 0; limb <= level; limb++ {
		q := ringQ.Modulus[limb]
		for word := 0; word < wordsPerLimb; word++ {
			offset := 8 * (limb*wordsPerLimb + word)
			value := binary.LittleEndian.Uint64(standard[offset:])
			if value >= q {
				return nil, fmt.Errorf("non-canonical standard residue limb=%d word=%d", limb, word)
			}
			binary.LittleEndian.PutUint64(montgomery[offset:], ring.MForm(value, q, ringQ.BredParams[limb]))
		}
	}
	return montgomery, nil
}

func encodeDense(params ckks.Parameters, encoder ckks.Encoder, reference preparedReference) ([]byte, []byte, error) {
	values, err := decodeComplexSlots(reference.Data)
	if err != nil {
		return nil, nil, err
	}
	plaintext := ckks.NewPlaintext(params, reference.Reference.ActiveLevel, reference.Reference.OperandScale)
	encoder.EncodeNTT(plaintext, values, params.LogSlots())
	if !plaintext.Value.IsNTT || plaintext.Level() != reference.Reference.ActiveLevel || plaintext.Scale != reference.Reference.OperandScale {
		return nil, nil, fmt.Errorf("Lattigo EncodeNTT metadata drift at reference %d", reference.Reference.ReferenceSequence)
	}
	standard := serializeDense(plaintext.Value, plaintext.Level())
	montgomery, err := montgomeryPayload(standard, params.RingQ(), plaintext.Level(), params.N())
	return standard, montgomery, err
}

func encodeResidual(params ckks.Parameters, evaluator ckks.Evaluator, reference preparedReference) ([]byte, []byte, error) {
	values, err := decodeComplexSlots(reference.Data)
	if err != nil || len(values) != 1 {
		return nil, nil, fmt.Errorf("decode residual scalar: %w", err)
	}
	level := reference.Reference.ActiveLevel
	input := ckks.NewCiphertext(params, 1, level, params.Scale())
	for component := range input.Value {
		for limb := 0; limb <= level; limb++ {
			for coefficient := range input.Value[component].Coeffs[limb] {
				input.Value[component].Coeffs[limb][coefficient] = 1
			}
		}
	}
	output := ckks.NewCiphertext(params, 1, level, params.Scale())
	evaluator.MultByConst(input, values[0], output)
	if output.Level() != level || output.Scale != input.Scale*reference.Reference.OperandScale {
		return nil, nil, fmt.Errorf("Lattigo MultByConst metadata drift at reference %d", reference.Reference.ReferenceSequence)
	}
	standard := make([]byte, 16*(level+1))
	for limb := 0; limb <= level; limb++ {
		positive := output.Value[0].Coeffs[limb][0]
		negative := output.Value[0].Coeffs[limb][params.N()/2]
		for component := range output.Value {
			for coefficient := 0; coefficient < params.N()/2; coefficient++ {
				if output.Value[component].Coeffs[limb][coefficient] != positive {
					return nil, nil, fmt.Errorf("MultByConst first-half residue drift")
				}
			}
			for coefficient := params.N() / 2; coefficient < params.N(); coefficient++ {
				if output.Value[component].Coeffs[limb][coefficient] != negative {
					return nil, nil, fmt.Errorf("MultByConst second-half residue drift")
				}
			}
		}
		binary.LittleEndian.PutUint64(standard[16*limb:], positive)
		binary.LittleEndian.PutUint64(standard[16*limb+8:], negative)
	}
	montgomery, err := montgomeryPayload(standard, params.RingQ(), level, 2)
	return standard, montgomery, err
}

func writeSynced(path string, data []byte) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
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
	closeErr := file.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

func syncDirectory(path string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func contentBinding(tempRoot, domain string, data []byte, seen map[string]PayloadBinding) (PayloadBinding, error) {
	digest := sha256Bytes(data)
	key := domain + ":" + digest
	if binding, exists := seen[key]; exists {
		return binding, nil
	}
	relative := filepath.ToSlash(filepath.Join("payloads", domain+"-"+digest+".bin"))
	path := filepath.Join(tempRoot, filepath.FromSlash(relative))
	if err := writeSynced(path, data); err != nil {
		return PayloadBinding{}, err
	}
	binding := PayloadBinding{Path: relative, Bytes: int64(len(data)), SHA256: digest}
	seen[key] = binding
	return binding, nil
}

func transcriptHeader(manifest *Manifest) string {
	fields := []string{
		TranscriptSchema, manifest.Parameter, manifest.Construction,
		manifest.ParameterLiteral.IdentitySHA256,
		manifest.ParameterLiteral.Binding.SHA256,
		manifest.Toolchain.LattigoModule, manifest.Toolchain.LattigoVersion,
		manifest.Toolchain.LattigoModuleSum,
	}
	paths := make([]string, 0, len(manifest.SourceBindings))
	for path := range manifest.SourceBindings {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		fields = append(fields, path+"="+manifest.SourceBindings[path].SHA256)
	}
	return strings.Join(fields, "\t") + "\n"
}

func transcriptEntry(entry Entry) (string, error) {
	fields := []string{
		strconv.Itoa(entry.ReferenceSequence), strconv.Itoa(entry.CommandSequence),
		entry.OperandClass, entry.Opcode, entry.Purpose, entry.Phase,
		entry.Boundary, entry.Operation, entry.SemanticKind,
		entry.SemanticPayloadID, entry.SemanticPayload.Path,
		strconv.FormatInt(entry.SemanticPayload.Bytes, 10), entry.SemanticPayload.SHA256,
		strconv.Itoa(entry.ActiveLevel), entry.OperandScaleFloat64BitsHex,
		strconv.FormatInt(entry.OperandScaleLog2Q3232, 10), entry.LattigoAPIBoundary,
		entry.Standard.Path, strconv.FormatInt(entry.Standard.Bytes, 10), entry.Standard.SHA256,
		entry.Montgomery.Path, strconv.FormatInt(entry.Montgomery.Bytes, 10), entry.Montgomery.SHA256,
	}
	for _, value := range fields {
		if strings.ContainsAny(value, "\t\r\n") {
			return "", fmt.Errorf("unsafe transcript field at reference %d", entry.ReferenceSequence)
		}
	}
	return strings.Join(fields, "\t") + "\n", nil
}

func validateConfig(cfg Config) error {
	if cfg.RepoRoot == "" || cfg.OutputRoot == "" || cfg.Schema == "" || cfg.Status == "" || cfg.EvidenceClass == "" || cfg.Parameter == "" || cfg.Construction == "" {
		return fmt.Errorf("incomplete exporter configuration")
	}
	if cfg.ExpectedReferences <= 0 || cfg.ExpectedDense <= 0 || cfg.ExpectedResidual <= 0 || cfg.ExpectedDense+cfg.ExpectedResidual != cfg.ExpectedReferences {
		return fmt.Errorf("invalid expected static coverage")
	}
	if cfg.StrictP2GN16 && (cfg.Schema != ProductionSchema || cfg.Status != ProductionStatus || cfg.EvidenceClass != ProductionEvidence || cfg.Parameter != ProductionParameter || cfg.Construction != ProductionConstruction || cfg.ExpectedReferences != ExpectedReferences || cfg.ExpectedDense != ExpectedDense || cfg.ExpectedResidual != ExpectedResidual) {
		return fmt.Errorf("production exporter configuration drift")
	}
	if _, err := os.Stat(cfg.OutputRoot); err == nil {
		return fmt.Errorf("output root already exists: %s", cfg.OutputRoot)
	} else if !os.IsNotExist(err) {
		return err
	}
	parent := filepath.Dir(cfg.OutputRoot)
	info, err := os.Stat(parent)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("output parent must already exist: %s", parent)
	}
	return nil
}

func Run(cfg Config) (*Manifest, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	literalData, literalBinding, err := readBound(cfg.ParameterLiteral)
	if err != nil {
		return nil, err
	}
	catalogData, catalogBinding, err := readBound(cfg.SemanticCatalog)
	if err != nil {
		return nil, err
	}
	commandData, commandBinding, err := readBound(cfg.CommandStream)
	if err != nil {
		return nil, err
	}
	_, oracleBinding, err := readBound(cfg.ScaleOracle)
	if err != nil {
		return nil, err
	}
	var literal parameterAuthority
	if err := json.Unmarshal(literalData, &literal); err != nil {
		return nil, fmt.Errorf("decode parameter literal: %w", err)
	}
	params, q, p, err := parameters(literal, cfg.StrictP2GN16)
	if err != nil {
		return nil, err
	}
	_, references, err := prepare(cfg, catalogData, commandData, params)
	if err != nil {
		return nil, err
	}
	goVersion, identityMethod, coreBindings, exporterBindings, exporterAggregate, err := linkedLattigo(cfg.RepoRoot, cfg.ExporterSourcePaths)
	if err != nil {
		return nil, err
	}

	manifest := &Manifest{
		Schema: cfg.Schema, Status: cfg.Status, Parameter: cfg.Parameter,
		Construction: cfg.Construction, EvidenceClass: cfg.EvidenceClass,
		ObserverMode:        "lattigo_v2.3_exact_public_operand_reencoding",
		PerformanceMeasured: false,
		SourceBindings: map[string]Binding{
			cfg.SemanticCatalog.Label: catalogBinding,
			cfg.CommandStream.Label:   commandBinding,
			cfg.ScaleOracle.Label:     oracleBinding,
		},
	}
	manifest.ParameterLiteral.Binding = literalBinding
	manifest.ParameterLiteral.IdentitySHA256 = literal.Literal.IdentitySHA256
	manifest.ParameterLiteral.LogN, manifest.ParameterLiteral.N = literal.Literal.LogN, literal.Literal.N
	manifest.ParameterLiteral.LogSlots, manifest.ParameterLiteral.Scale = literal.Literal.LogSlots, literal.Literal.Scale
	manifest.ParameterLiteral.Sigma = literal.Literal.Sigma
	manifest.ParameterLiteral.Q, manifest.ParameterLiteral.P = append([]uint64(nil), q...), append([]uint64(nil), p...)
	manifest.ParameterLiteral.PersistentLevel, manifest.ParameterLiteral.GuardedLevel = literal.Literal.PersistentMax, literal.Literal.GuardedMax
	manifest.ParameterLiteral.Alpha, manifest.ParameterLiteral.Beta = literal.Literal.Alpha, literal.Literal.Beta
	manifest.Toolchain.GoVersion, manifest.Toolchain.GOOS, manifest.Toolchain.GOARCH = goVersion, runtime.GOOS, runtime.GOARCH
	manifest.Toolchain.LattigoModule, manifest.Toolchain.LattigoVersion, manifest.Toolchain.LattigoModuleSum = LattigoModule, LattigoVersion, LattigoModuleSum
	manifest.Toolchain.LattigoIdentityMethod = identityMethod
	manifest.Toolchain.CoreSourceBindings, manifest.Toolchain.ExporterSourceBindings, manifest.Toolchain.ExporterSourceAggregateSHA = coreBindings, exporterBindings, exporterAggregate
	manifest.Coverage.References, manifest.Coverage.DenseMaskReferences, manifest.Coverage.FractionalMultReferences = cfg.ExpectedReferences, cfg.ExpectedDense, cfg.ExpectedResidual
	manifest.AtomicCommit.Complete = true
	manifest.AtomicCommit.Method = "sibling_temp_directory_fsync_then_atomic_rename"
	manifest.ReleaseState.StaticExportComplete = true
	manifest.ReleaseState.KernelIntegrated = false
	manifest.ReleaseState.BoardEligible = false
	manifest.ReleaseState.MeasurementEligible = false
	manifest.ReleaseState.PerformanceMeasured = false

	parent, base := filepath.Dir(cfg.OutputRoot), filepath.Base(cfg.OutputRoot)
	tempRoot, err := os.MkdirTemp(parent, "."+base+".tmp-")
	if err != nil {
		return nil, err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(tempRoot)
		}
	}()
	if err := os.Mkdir(filepath.Join(tempRoot, "payloads"), 0o755); err != nil {
		return nil, err
	}

	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})
	objectCache := make(map[string][2]PayloadBinding)
	contentSeen := make(map[string]PayloadBinding)
	transcript := []byte(transcriptHeader(manifest))
	for _, reference := range references {
		bindings, exists := objectCache[reference.Key]
		if !exists {
			var standard, montgomery []byte
			if reference.Class == "dense_slot_mask" {
				standard, montgomery, err = encodeDense(params, encoder, reference)
			} else {
				standard, montgomery, err = encodeResidual(params, evaluator, reference)
			}
			if err != nil {
				return nil, err
			}
			standardBinding, writeErr := contentBinding(tempRoot, "standard", standard, contentSeen)
			if writeErr != nil {
				return nil, writeErr
			}
			montgomeryBinding, writeErr := contentBinding(tempRoot, "montgomery", montgomery, contentSeen)
			if writeErr != nil {
				return nil, writeErr
			}
			bindings = [2]PayloadBinding{standardBinding, montgomeryBinding}
			objectCache[reference.Key] = bindings
		}
		api := "ckks.Encoder.EncodeNTT"
		if reference.Class == "fractional_mult_by_const" {
			api = "ckks.Evaluator.MultByConst(actual_output_capture)"
		}
		entry := Entry{
			ReferenceSequence: reference.Reference.ReferenceSequence,
			CommandSequence:   reference.Reference.CommandSequence,
			Opcode:            reference.Reference.Opcode, Purpose: reference.Reference.Purpose,
			Phase: reference.Reference.Phase, Boundary: reference.Reference.Boundary,
			Operation: reference.Reference.Operation, OperandClass: reference.Class,
			SemanticKind:               reference.Reference.SemanticKind,
			SemanticPayloadID:          reference.Reference.SemanticPayloadID,
			SemanticPayload:            PayloadBinding{Path: reference.Payload.Path, Bytes: reference.Payload.PayloadBytes, SHA256: reference.Payload.ContentSHA256},
			ActiveLevel:                reference.Reference.ActiveLevel,
			OperandScale:               reference.Reference.OperandScale,
			OperandScaleFloat64BitsHex: fmt.Sprintf("%016x", math.Float64bits(reference.Reference.OperandScale)),
			OperandScaleLog2Q3232:      scaleQ32(reference.Reference.OperandScale),
			LattigoAPIBoundary:         api, Standard: bindings[0], Montgomery: bindings[1],
		}
		line, lineErr := transcriptEntry(entry)
		if lineErr != nil {
			return nil, lineErr
		}
		transcript = append(transcript, []byte(line)...)
		manifest.Entries = append(manifest.Entries, entry)
	}
	for key := range contentSeen {
		if strings.HasPrefix(key, "standard:") {
			manifest.Coverage.UniqueStandardObjects++
		} else if strings.HasPrefix(key, "montgomery:") {
			manifest.Coverage.UniqueMontgomeryObjects++
		}
	}
	transcriptPath := "encoding_transcript.tsv"
	if err := writeSynced(filepath.Join(tempRoot, transcriptPath), transcript); err != nil {
		return nil, err
	}
	manifest.EncodingTranscript.Schema = TranscriptSchema
	manifest.EncodingTranscript.Path = transcriptPath
	manifest.EncodingTranscript.Bytes = int64(len(transcript))
	manifest.EncodingTranscript.SHA256 = sha256Bytes(transcript)
	manifest.EncodingTranscript.LineCount = len(manifest.Entries) + 1

	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, err
	}
	manifestData = append(manifestData, '\n')
	if err := writeSynced(filepath.Join(tempRoot, "manifest.json"), manifestData); err != nil {
		return nil, err
	}
	if err := syncDirectory(filepath.Join(tempRoot, "payloads")); err != nil {
		return nil, err
	}
	if err := syncDirectory(tempRoot); err != nil {
		return nil, err
	}
	if err := os.Rename(tempRoot, cfg.OutputRoot); err != nil {
		return nil, fmt.Errorf("atomic export commit: %w", err)
	}
	if err := syncDirectory(parent); err != nil {
		_ = os.RemoveAll(cfg.OutputRoot)
		_ = syncDirectory(parent)
		return nil, fmt.Errorf("sync committed export parent: %w", err)
	}
	committed = true
	return manifest, nil
}

// ReadManifest is intentionally small and is provided only for unit tests
// and launch wrappers; the independent validator does not import this package.
func ReadManifest(path string) (*Manifest, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	decoder := json.NewDecoder(bufio.NewReader(file))
	decoder.DisallowUnknownFields()
	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return nil, err
	}
	return &manifest, nil
}
