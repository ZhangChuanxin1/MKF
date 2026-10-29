package p2gn16staticvalidate

// Package p2gn16staticvalidate independently re-encodes every static P2GN16
// reference with Lattigo v2.3.0 and byte-compares it with the export. It
// intentionally does not import the exporter package.

import (
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
	ReceiptSchema          = "mkf-p2gn16-static-public-operand-reencoding-receipt-v1"
	ReceiptStatus          = "PASS_INDEPENDENT_LATTIGO_REENCODING_RELEASE_STILL_BLOCKED"
	ReceiptEvidence        = "independent_lattigo_v2.3_semantic_reencoding"
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
	RepoRoot             string
	ExportRoot           string
	ReceiptPath          string
	ParameterLiteral     Authority
	SemanticCatalog      Authority
	CommandStream        Authority
	ScaleOracle          Authority
	ExporterSourcePaths  []string
	ValidatorSourcePaths []string
	Schema               string
	Status               string
	EvidenceClass        string
	Parameter            string
	Construction         string
	ExpectedReferences   int
	ExpectedDense        int
	ExpectedResidual     int
	StrictP2GN16         bool
}

func ProductionConfig(repoRoot, exportRoot, receiptPath string) Config {
	return Config{
		RepoRoot: repoRoot, ExportRoot: exportRoot, ReceiptPath: receiptPath,
		ParameterLiteral: Authority{Label: "experiments/resnet20_p3/security/p2g_n16_candidate_literal_20260826.json", Path: filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/security/p2g_n16_candidate_literal_20260826.json")), ExpectedSHA256: ParameterLiteralSHA},
		SemanticCatalog:  Authority{Label: "experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_public_operand_semantic_catalog.json", Path: filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_public_operand_semantic_catalog.json")), ExpectedSHA256: SemanticCatalogSHA},
		CommandStream:    Authority{Label: "experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_resnet20_operation_command_stream.calibrated_margin070.json", Path: filepath.Join(repoRoot, filepath.FromSlash("experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_resnet20_operation_command_stream.calibrated_margin070.json")), ExpectedSHA256: CommandStreamSHA},
		ScaleOracle:      Authority{Label: "tmp/cpu_experiments/experiments/resnet20_p3/fpga_scale_audit/p2gn16_lattigo_v23_scale_encoding_oracle.json", Path: filepath.Join(repoRoot, filepath.FromSlash("tmp/cpu_experiments/experiments/resnet20_p3/fpga_scale_audit/p2gn16_lattigo_v23_scale_encoding_oracle.json")), ExpectedSHA256: ScaleOracleSHA},
		ExporterSourcePaths: []string{
			"tmp/cpu_experiments/cmd/p2gn16_static_export/main.go",
			"tmp/cpu_experiments/internal/p2gn16staticexport/export.go",
			"tmp/cpu_experiments/go.mod", "tmp/cpu_experiments/go.sum",
		},
		ValidatorSourcePaths: []string{
			"tmp/cpu_experiments/cmd/p2gn16_static_validate/main.go",
			"tmp/cpu_experiments/internal/p2gn16staticvalidate/validate.go",
			"tmp/cpu_experiments/go.mod", "tmp/cpu_experiments/go.sum",
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

type manifest struct {
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
	Parameter           string              `json:"parameter"`
	Construction        string              `json:"construction"`
	PerformanceMeasured bool                `json:"performance_measured"`
	Payloads            []semanticPayload   `json:"payloads"`
	References          []semanticReference `json:"references"`
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

type Receipt struct {
	Schema                      string             `json:"schema"`
	Status                      string             `json:"status"`
	Parameter                   string             `json:"parameter"`
	Construction                string             `json:"construction"`
	EvidenceClass               string             `json:"evidence_class"`
	Manifest                    Binding            `json:"manifest"`
	Transcript                  Binding            `json:"encoding_transcript"`
	Authorities                 map[string]Binding `json:"source_bindings"`
	ValidatorSourceBindings     []Binding          `json:"validator_source_bindings"`
	ValidatorSourceAggregateSHA string             `json:"validator_source_aggregate_sha256"`
	ValidatorToolchain          struct {
		GoVersion          string    `json:"go_version"`
		GOOS               string    `json:"goos"`
		GOARCH             string    `json:"goarch"`
		LattigoModule      string    `json:"lattigo_module"`
		LattigoVersion     string    `json:"lattigo_version"`
		LattigoModuleSum   string    `json:"lattigo_module_sum"`
		IdentityMethod     string    `json:"lattigo_identity_method"`
		CoreSourceBindings []Binding `json:"core_source_bindings"`
	} `json:"validator_toolchain"`
	Validation struct {
		ReferencesReencoded       int  `json:"references_reencoded"`
		UniqueObjectsReencoded    int  `json:"unique_objects_reencoded"`
		LattigoAPICalls           int  `json:"lattigo_api_calls"`
		DenseReferences           int  `json:"dense_references"`
		ResidualReferences        int  `json:"residual_references"`
		StandardByteComparisons   int  `json:"standard_byte_comparisons"`
		MontgomeryByteComparisons int  `json:"montgomery_byte_comparisons"`
		TranscriptExact           bool `json:"transcript_exact"`
		FileSetExact              bool `json:"file_set_exact"`
	} `json:"validation"`
	AtomicCommit struct {
		Complete bool   `json:"complete"`
		Method   string `json:"method"`
	} `json:"atomic_commit"`
	ReleaseState struct {
		StaticReencodingValidated bool `json:"static_reencoding_validated"`
		KernelIntegrated          bool `json:"kernel_integrated"`
		BoardEligible             bool `json:"board_eligible"`
		MeasurementEligible       bool `json:"measurement_eligible"`
		PerformanceMeasured       bool `json:"performance_measured"`
	} `json:"release_state"`
}

func shaBytes(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }

func fileSHA(path string) (string, int64, error) {
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

func strictJSON(path string, target interface{}) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("not a regular non-symlink JSON file: %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return nil, err
	}
	var extra interface{}
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("trailing JSON content in %s", path)
	}
	return data, nil
}

func readBound(authority Authority) ([]byte, Binding, error) {
	info, err := os.Lstat(authority.Path)
	if err != nil {
		return nil, Binding{}, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, Binding{}, fmt.Errorf("authority is not a regular file: %s", authority.Label)
	}
	data, err := os.ReadFile(authority.Path)
	if err != nil {
		return nil, Binding{}, err
	}
	digest := shaBytes(data)
	if authority.ExpectedSHA256 != "" && digest != strings.ToLower(authority.ExpectedSHA256) {
		return nil, Binding{}, fmt.Errorf("authority digest drift: %s", authority.Label)
	}
	return data, Binding{Path: authority.Label, Bytes: int64(len(data)), SHA256: digest}, nil
}

func safeRegular(root, relative string) (string, error) {
	if relative == "" || filepath.IsAbs(relative) {
		return "", fmt.Errorf("unsafe path %q", relative)
	}
	clean := filepath.Clean(filepath.FromSlash(relative))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe path %q", relative)
	}
	path := filepath.Join(root, clean)
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes root: %q", relative)
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("not a regular non-symlink file: %q", relative)
	}
	return path, nil
}

func parseWords(values []string) ([]uint64, error) {
	result := make([]uint64, len(values))
	for i, value := range values {
		parsed, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(value), "0x"), 16, 64)
		if err != nil {
			return nil, err
		}
		result[i] = parsed
	}
	return result, nil
}

func materializeParameters(authority parameterAuthority, strict bool) (ckks.Parameters, []uint64, []uint64, error) {
	q, err := parseWords(authority.Literal.QPost)
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	guard, err := parseWords([]string{authority.Literal.QGuard})
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	q = append(q, guard[0])
	p, err := parseWords(authority.Literal.P)
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	if strict {
		if authority.Schema != "mkf-p2g-n16-security-candidate-v1" || authority.Generator.Dependency != LattigoModule+" "+LattigoVersion || authority.Generator.ModuleSum != LattigoModuleSum || authority.Literal.IdentitySHA256 != ParameterIdentitySHA {
			return ckks.Parameters{}, nil, nil, fmt.Errorf("parameter authority identity drift")
		}
		if authority.Literal.LogN != 16 || authority.Literal.N != 65536 || authority.Literal.LogSlots != 14 || authority.Literal.SlotsUsed != 16384 || len(q) != 15 || len(p) != 2 || authority.Literal.PersistentMax != 13 || authority.Literal.GuardedMax != 14 || authority.Literal.Alpha != 2 || authority.Literal.Beta != 8 || !authority.Literal.AllPrimesPrime || !authority.Literal.AllPrimesNTTPrime {
			return ckks.Parameters{}, nil, nil, fmt.Errorf("parameter literal shape drift")
		}
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: authority.Literal.LogN, LogSlots: authority.Literal.LogSlots, Q: q, P: p, Scale: authority.Literal.Scale, Sigma: authority.Literal.Sigma})
	if err != nil {
		return ckks.Parameters{}, nil, nil, err
	}
	if params.N() != authority.Literal.N || params.MaxLevel() != authority.Literal.GuardedMax || params.Scale() != authority.Literal.Scale || !reflect.DeepEqual(params.Q(), q) || !reflect.DeepEqual(params.P(), p) {
		return ckks.Parameters{}, nil, nil, fmt.Errorf("Lattigo parameter materialization drift")
	}
	return params, q, p, nil
}

func sourceBindings(repoRoot string, paths []string) ([]Binding, string, error) {
	bindings := make([]Binding, 0, len(paths))
	for _, relative := range paths {
		path, err := safeRegular(repoRoot, relative)
		if err != nil {
			return nil, "", err
		}
		digest, bytes, err := fileSHA(path)
		if err != nil {
			return nil, "", err
		}
		bindings = append(bindings, Binding{Path: filepath.ToSlash(relative), Bytes: bytes, SHA256: digest})
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].Path < bindings[j].Path })
	encoded, _ := json.Marshal(bindings)
	return bindings, shaBytes(encoded), nil
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

func linkedCore(repoRoot string) ([]Binding, string, error) {
	info, ok := debug.ReadBuildInfo()
	identityMethod := "DEPENDENCY_ENTRY_UNAVAILABLE_GO_SUM_AND_CORE_SOURCE_VERIFIED"
	found := false
	if ok {
		for _, dep := range info.Deps {
			if dep.Path == LattigoModule {
				found = true
				if dep.Version != LattigoVersion || dep.Sum != LattigoModuleSum {
					return nil, "", fmt.Errorf("linked Lattigo identity drift")
				}
			}
		}
	} else {
		identityMethod = "BUILD_INFO_UNAVAILABLE_GO_SUM_AND_CORE_SOURCE_VERIFIED"
	}
	if found {
		identityMethod = "DEPENDENCY_VERSION_AND_SUM_VERIFIED"
	}
	if err := verifyGoSum(repoRoot); err != nil {
		return nil, "", err
	}
	fn := runtime.FuncForPC(reflect.ValueOf(ckks.NewEncoder).Pointer())
	if fn == nil {
		return nil, "", fmt.Errorf("cannot locate Lattigo source")
	}
	path, _ := fn.FileLine(fn.Entry())
	root := filepath.Dir(filepath.Dir(path))
	paths := make([]string, 0, len(coreSourceSHA256))
	for rel := range coreSourceSHA256 {
		paths = append(paths, rel)
	}
	sort.Strings(paths)
	bindings := make([]Binding, 0, len(paths))
	for _, rel := range paths {
		digest, bytes, err := fileSHA(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, "", err
		}
		if digest != coreSourceSHA256[rel] {
			return nil, "", fmt.Errorf("linked core source drift: %s", rel)
		}
		bindings = append(bindings, Binding{Path: rel, Bytes: bytes, SHA256: digest})
	}
	return bindings, identityMethod, nil
}

func equalBindings(left, right []Binding) bool { return reflect.DeepEqual(left, right) }

func scaleQ32(value float64) int64 {
	return int64(math.Round(math.Log2(value) * float64(uint64(1)<<32)))
}

func decodeSlots(data []byte) ([]complex128, error) {
	if len(data)%16 != 0 {
		return nil, fmt.Errorf("complex payload length")
	}
	result := make([]complex128, len(data)/16)
	for i := range result {
		r := math.Float64frombits(binary.LittleEndian.Uint64(data[16*i:]))
		im := math.Float64frombits(binary.LittleEndian.Uint64(data[16*i+8:]))
		if math.IsNaN(r) || math.IsInf(r, 0) || math.IsNaN(im) || math.IsInf(im, 0) {
			return nil, fmt.Errorf("non-finite semantic value")
		}
		result[i] = complex(r, im)
	}
	return result, nil
}

func denseBytes(poly *ring.Poly, level int) []byte {
	result := make([]byte, 8*(level+1)*len(poly.Coeffs[0]))
	offset := 0
	for limb := 0; limb <= level; limb++ {
		for _, value := range poly.Coeffs[limb] {
			binary.LittleEndian.PutUint64(result[offset:], value)
			offset += 8
		}
	}
	return result
}

func toMontgomery(standard []byte, ringQ *ring.Ring, level, words int) ([]byte, error) {
	if len(standard) != 8*(level+1)*words {
		return nil, fmt.Errorf("standard payload size mismatch")
	}
	result := make([]byte, len(standard))
	for limb := 0; limb <= level; limb++ {
		q := ringQ.Modulus[limb]
		for word := 0; word < words; word++ {
			offset := 8 * (limb*words + word)
			value := binary.LittleEndian.Uint64(standard[offset:])
			if value >= q {
				return nil, fmt.Errorf("non-canonical standard residue")
			}
			binary.LittleEndian.PutUint64(result[offset:], ring.MForm(value, q, ringQ.BredParams[limb]))
		}
	}
	return result, nil
}

func independentlyEncodeDense(params ckks.Parameters, encoder ckks.Encoder, ref preparedReference) ([]byte, []byte, error) {
	values, err := decodeSlots(ref.Data)
	if err != nil {
		return nil, nil, err
	}
	plaintext := ckks.NewPlaintext(params, ref.Reference.ActiveLevel, ref.Reference.OperandScale)
	encoder.EncodeNTT(plaintext, values, params.LogSlots())
	standard := denseBytes(plaintext.Value, plaintext.Level())
	mont, err := toMontgomery(standard, params.RingQ(), plaintext.Level(), params.N())
	return standard, mont, err
}

func independentlyCaptureResidual(params ckks.Parameters, evaluator ckks.Evaluator, ref preparedReference) ([]byte, []byte, error) {
	values, err := decodeSlots(ref.Data)
	if err != nil || len(values) != 1 {
		return nil, nil, fmt.Errorf("residual decode")
	}
	level := ref.Reference.ActiveLevel
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
	if output.Scale != input.Scale*ref.Reference.OperandScale {
		return nil, nil, fmt.Errorf("MultByConst scale drift")
	}
	standard := make([]byte, 16*(level+1))
	for limb := 0; limb <= level; limb++ {
		positive := output.Value[0].Coeffs[limb][0]
		negative := output.Value[0].Coeffs[limb][params.N()/2]
		for component := range output.Value {
			for i := 0; i < params.N()/2; i++ {
				if output.Value[component].Coeffs[limb][i] != positive {
					return nil, nil, fmt.Errorf("MultByConst first-half drift")
				}
			}
			for i := params.N() / 2; i < params.N(); i++ {
				if output.Value[component].Coeffs[limb][i] != negative {
					return nil, nil, fmt.Errorf("MultByConst second-half drift")
				}
			}
		}
		binary.LittleEndian.PutUint64(standard[16*limb:], positive)
		binary.LittleEndian.PutUint64(standard[16*limb+8:], negative)
	}
	mont, err := toMontgomery(standard, params.RingQ(), level, 2)
	return standard, mont, err
}

func prepare(cfg Config, catalogData, commandData []byte, params ckks.Parameters) ([]preparedReference, error) {
	var catalog semanticCatalog
	if err := json.Unmarshal(catalogData, &catalog); err != nil {
		return nil, err
	}
	if catalog.Parameter != cfg.Parameter || catalog.Construction != cfg.Construction || catalog.PerformanceMeasured {
		return nil, fmt.Errorf("semantic catalog identity drift")
	}
	var commands commandAuthority
	if err := json.Unmarshal(commandData, &commands); err != nil {
		return nil, err
	}
	if commands.Parameter != cfg.Parameter || commands.Construction != cfg.Construction || commands.PerformanceMeasured || commands.BoardEligible {
		return nil, fmt.Errorf("command authority identity drift")
	}
	byCommand := map[int]commandEvent{}
	for _, cmd := range commands.CommandStream.Commands {
		if _, ok := byCommand[cmd.Sequence]; ok {
			return nil, fmt.Errorf("duplicate command")
		}
		byCommand[cmd.Sequence] = cmd
	}
	payloads := map[string]semanticPayload{}
	for _, payload := range catalog.Payloads {
		if _, ok := payloads[payload.PayloadID]; ok {
			return nil, fmt.Errorf("duplicate payload")
		}
		payloads[payload.PayloadID] = payload
	}
	root := filepath.Dir(cfg.SemanticCatalog.Path)
	seen := map[int]bool{}
	result := []preparedReference{}
	dense, residual := 0, 0
	payloadData := make(map[string][]byte, len(payloads))
	for _, ref := range catalog.References {
		if seen[ref.ReferenceSequence] {
			return nil, fmt.Errorf("duplicate reference")
		}
		seen[ref.ReferenceSequence] = true
		isDense := ref.SemanticKind != "semantic_complex_scalar"
		isResidual := ref.Opcode == "plaintext_mul" && ref.Purpose == "residual"
		if !isDense && !isResidual {
			continue
		}
		if !ref.SemanticPayloadExact || ref.ActiveLevel < 0 || ref.ActiveLevel > params.MaxLevel() || math.IsNaN(ref.OperandScale) || math.IsInf(ref.OperandScale, 0) || ref.OperandScale <= 0 {
			return nil, fmt.Errorf("reference metadata drift")
		}
		cmd, ok := byCommand[ref.CommandSequence]
		if !ok || cmd.Sequence != ref.CommandSequence || cmd.Opcode != ref.Opcode || cmd.Purpose != ref.Purpose || cmd.Phase != ref.Phase || cmd.Boundary != ref.Boundary || cmd.Operation != ref.Operation || cmd.OutputLevel != ref.ActiveLevel || len(cmd.InputLevels) == 0 || cmd.InputLevels[0] != ref.ActiveLevel {
			return nil, fmt.Errorf("command/reference drift at %d", ref.ReferenceSequence)
		}
		payload, ok := payloads[ref.SemanticPayloadID]
		if !ok || payload.Kind != ref.SemanticKind {
			return nil, fmt.Errorf("semantic payload identity drift")
		}
		data, cached := payloadData[payload.PayloadID]
		if !cached {
			path, err := safeRegular(root, payload.Path)
			if err != nil {
				return nil, err
			}
			data, err = os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			if int64(len(data)) != payload.PayloadBytes || shaBytes(data) != payload.ContentSHA256 {
				return nil, fmt.Errorf("semantic payload content drift: %s", payload.PayloadID)
			}
			payloadData[payload.PayloadID] = data
		}
		class := "fractional_mult_by_const"
		if isDense {
			class = "dense_slot_mask"
			dense++
			if payload.Encoding != "complex128_slots_real_imag_float64_bits_little_endian_v1" || payload.LogicalSlots != 1<<params.LogSlots() || len(data) != payload.LogicalSlots*16 {
				return nil, fmt.Errorf("dense payload shape drift")
			}
		} else {
			residual++
			if payload.Encoding != "complex128_real_imag_float64_bits_little_endian_v1" || len(data) != 16 {
				return nil, fmt.Errorf("residual payload shape drift")
			}
		}
		key := strings.Join([]string{class, payload.ContentSHA256, strconv.Itoa(ref.ActiveLevel), fmt.Sprintf("%016x", math.Float64bits(ref.OperandScale)), strconv.FormatInt(scaleQ32(ref.OperandScale), 10)}, ":")
		result = append(result, preparedReference{Reference: ref, Payload: payload, Data: data, Class: class, Key: key})
	}
	if len(result) != cfg.ExpectedReferences || dense != cfg.ExpectedDense || residual != cfg.ExpectedResidual {
		return nil, fmt.Errorf("coverage drift %d/%d/%d", len(result), dense, residual)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Reference.ReferenceSequence < result[j].Reference.ReferenceSequence
	})
	return result, nil
}

func transcriptHeader(m *manifest) string {
	fields := []string{TranscriptSchema, m.Parameter, m.Construction, m.ParameterLiteral.IdentitySHA256, m.ParameterLiteral.Binding.SHA256, m.Toolchain.LattigoModule, m.Toolchain.LattigoVersion, m.Toolchain.LattigoModuleSum}
	paths := make([]string, 0, len(m.SourceBindings))
	for path := range m.SourceBindings {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		fields = append(fields, path+"="+m.SourceBindings[path].SHA256)
	}
	return strings.Join(fields, "\t") + "\n"
}

func transcriptLine(entry Entry) (string, error) {
	fields := []string{strconv.Itoa(entry.ReferenceSequence), strconv.Itoa(entry.CommandSequence), entry.OperandClass, entry.Opcode, entry.Purpose, entry.Phase, entry.Boundary, entry.Operation, entry.SemanticKind, entry.SemanticPayloadID, entry.SemanticPayload.Path, strconv.FormatInt(entry.SemanticPayload.Bytes, 10), entry.SemanticPayload.SHA256, strconv.Itoa(entry.ActiveLevel), entry.OperandScaleFloat64BitsHex, strconv.FormatInt(entry.OperandScaleLog2Q3232, 10), entry.LattigoAPIBoundary, entry.Standard.Path, strconv.FormatInt(entry.Standard.Bytes, 10), entry.Standard.SHA256, entry.Montgomery.Path, strconv.FormatInt(entry.Montgomery.Bytes, 10), entry.Montgomery.SHA256}
	for _, value := range fields {
		if strings.ContainsAny(value, "\t\r\n") {
			return "", fmt.Errorf("unsafe transcript field")
		}
	}
	return strings.Join(fields, "\t") + "\n", nil
}

func sameEntryBinding(entry Entry, ref preparedReference) error {
	api := "ckks.Encoder.EncodeNTT"
	if ref.Class == "fractional_mult_by_const" {
		api = "ckks.Evaluator.MultByConst(actual_output_capture)"
	}
	if entry.ReferenceSequence != ref.Reference.ReferenceSequence || entry.CommandSequence != ref.Reference.CommandSequence || entry.Opcode != ref.Reference.Opcode || entry.Purpose != ref.Reference.Purpose || entry.Phase != ref.Reference.Phase || entry.Boundary != ref.Reference.Boundary || entry.Operation != ref.Reference.Operation || entry.OperandClass != ref.Class || entry.SemanticKind != ref.Reference.SemanticKind || entry.SemanticPayloadID != ref.Reference.SemanticPayloadID || entry.SemanticPayload.Path != ref.Payload.Path || entry.SemanticPayload.Bytes != ref.Payload.PayloadBytes || entry.SemanticPayload.SHA256 != ref.Payload.ContentSHA256 || entry.ActiveLevel != ref.Reference.ActiveLevel || math.Float64bits(entry.OperandScale) != math.Float64bits(ref.Reference.OperandScale) || entry.OperandScaleFloat64BitsHex != fmt.Sprintf("%016x", math.Float64bits(ref.Reference.OperandScale)) || entry.OperandScaleLog2Q3232 != scaleQ32(ref.Reference.OperandScale) || entry.LattigoAPIBoundary != api {
		return fmt.Errorf("manifest semantic binding drift at reference %d", ref.Reference.ReferenceSequence)
	}
	return nil
}

func readPayload(root string, binding PayloadBinding, expectedBytes int64) ([]byte, error) {
	path, err := safeRegular(root, binding.Path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != expectedBytes || binding.Bytes != expectedBytes || shaBytes(data) != binding.SHA256 {
		return nil, fmt.Errorf("payload digest/length drift: %s", binding.Path)
	}
	return data, nil
}

func checkMontgomery(standard, mont []byte, ringQ *ring.Ring, level, words int) error {
	expected, err := toMontgomery(standard, ringQ, level, words)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(expected, mont) {
		return fmt.Errorf("standard/Montgomery relation mismatch")
	}
	return nil
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

func atomicReceipt(path string, value *Receipt) error {
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("receipt already exists")
	} else if !os.IsNotExist(err) {
		return err
	}
	parent := filepath.Dir(path)
	info, err := os.Stat(parent)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("receipt parent missing")
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	file, err := os.CreateTemp(parent, "."+filepath.Base(path)+".tmp-")
	if err != nil {
		return err
	}
	temp := file.Name()
	committed := false
	defer func() {
		if !committed {
			_ = os.Remove(temp)
		}
	}()
	if err = file.Chmod(0o644); err == nil {
		_, err = file.Write(data)
	}
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err = os.Rename(temp, path); err != nil {
		return err
	}
	if err = syncDirectory(parent); err != nil {
		_ = os.Remove(path)
		_ = syncDirectory(parent)
		return err
	}
	committed = true
	return nil
}

func validateConfig(cfg Config) error {
	if cfg.RepoRoot == "" || cfg.ExportRoot == "" || cfg.ReceiptPath == "" || cfg.ExpectedDense+cfg.ExpectedResidual != cfg.ExpectedReferences {
		return fmt.Errorf("invalid validator config")
	}
	if cfg.StrictP2GN16 && (cfg.Schema != ProductionSchema || cfg.Status != ProductionStatus || cfg.EvidenceClass != ProductionEvidence || cfg.Parameter != ProductionParameter || cfg.Construction != ProductionConstruction || cfg.ExpectedReferences != ExpectedReferences || cfg.ExpectedDense != ExpectedDense || cfg.ExpectedResidual != ExpectedResidual) {
		return fmt.Errorf("production validator configuration drift")
	}
	if _, err := os.Stat(cfg.ReceiptPath); err == nil {
		return fmt.Errorf("receipt already exists")
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func Validate(cfg Config) (*Receipt, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	var m manifest
	manifestPath := filepath.Join(cfg.ExportRoot, "manifest.json")
	manifestData, err := strictJSON(manifestPath, &m)
	if err != nil {
		return nil, err
	}
	if m.Schema != cfg.Schema || m.Status != cfg.Status || m.Parameter != cfg.Parameter || m.Construction != cfg.Construction || m.EvidenceClass != cfg.EvidenceClass || m.ObserverMode != "lattigo_v2.3_exact_public_operand_reencoding" || m.PerformanceMeasured {
		return nil, fmt.Errorf("manifest identity/status drift")
	}
	if !m.AtomicCommit.Complete || m.AtomicCommit.Method != "sibling_temp_directory_fsync_then_atomic_rename" || !m.ReleaseState.StaticExportComplete || m.ReleaseState.KernelIntegrated || m.ReleaseState.BoardEligible || m.ReleaseState.MeasurementEligible || m.ReleaseState.PerformanceMeasured {
		return nil, fmt.Errorf("atomic/release state drift")
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
	manifestAuthorities := map[string]Binding{cfg.SemanticCatalog.Label: catalogBinding, cfg.CommandStream.Label: commandBinding, cfg.ScaleOracle.Label: oracleBinding}
	if !reflect.DeepEqual(m.SourceBindings, manifestAuthorities) {
		return nil, fmt.Errorf("manifest source binding set drift")
	}
	var literal parameterAuthority
	if err = json.Unmarshal(literalData, &literal); err != nil {
		return nil, err
	}
	params, q, p, err := materializeParameters(literal, cfg.StrictP2GN16)
	if err != nil {
		return nil, err
	}
	if m.ParameterLiteral.Binding != literalBinding || m.ParameterLiteral.IdentitySHA256 != literal.Literal.IdentitySHA256 || m.ParameterLiteral.LogN != literal.Literal.LogN || m.ParameterLiteral.N != literal.Literal.N || m.ParameterLiteral.LogSlots != literal.Literal.LogSlots || math.Float64bits(m.ParameterLiteral.Scale) != math.Float64bits(literal.Literal.Scale) || math.Float64bits(m.ParameterLiteral.Sigma) != math.Float64bits(literal.Literal.Sigma) || !reflect.DeepEqual(m.ParameterLiteral.Q, q) || !reflect.DeepEqual(m.ParameterLiteral.P, p) || m.ParameterLiteral.PersistentLevel != literal.Literal.PersistentMax || m.ParameterLiteral.GuardedLevel != literal.Literal.GuardedMax || m.ParameterLiteral.Alpha != literal.Literal.Alpha || m.ParameterLiteral.Beta != literal.Literal.Beta {
		return nil, fmt.Errorf("manifest parameter literal binding drift")
	}
	core, identityMethod, err := linkedCore(cfg.RepoRoot)
	if err != nil {
		return nil, err
	}
	exporterSources, exporterAggregate, err := sourceBindings(cfg.RepoRoot, cfg.ExporterSourcePaths)
	if err != nil {
		return nil, err
	}
	if m.Toolchain.GoVersion != runtime.Version() || m.Toolchain.GOOS != runtime.GOOS || m.Toolchain.GOARCH != runtime.GOARCH || m.Toolchain.LattigoModule != LattigoModule || m.Toolchain.LattigoVersion != LattigoVersion || m.Toolchain.LattigoModuleSum != LattigoModuleSum || m.Toolchain.LattigoIdentityMethod != identityMethod || !equalBindings(m.Toolchain.CoreSourceBindings, core) || !equalBindings(m.Toolchain.ExporterSourceBindings, exporterSources) || m.Toolchain.ExporterSourceAggregateSHA != exporterAggregate {
		return nil, fmt.Errorf("manifest toolchain/source binding drift")
	}
	refs, err := prepare(cfg, catalogData, commandData, params)
	if err != nil {
		return nil, err
	}
	if m.Coverage.References != cfg.ExpectedReferences || m.Coverage.DenseMaskReferences != cfg.ExpectedDense || m.Coverage.FractionalMultReferences != cfg.ExpectedResidual || len(m.Entries) != cfg.ExpectedReferences {
		return nil, fmt.Errorf("manifest coverage drift")
	}
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})
	boundKeys := map[string][2]PayloadBinding{}
	standardObjects := map[string]bool{}
	montgomeryObjects := map[string]bool{}
	expectedFiles := map[string]bool{"manifest.json": true, m.EncodingTranscript.Path: true}
	transcript := []byte(transcriptHeader(&m))
	dense, residual := 0, 0
	for index, ref := range refs {
		entry := m.Entries[index]
		if err = sameEntryBinding(entry, ref); err != nil {
			return nil, err
		}
		words := 2
		if ref.Class == "dense_slot_mask" {
			words = params.N()
			dense++
		} else {
			residual++
		}
		expectedBytes := int64(8 * (ref.Reference.ActiveLevel + 1) * words)
		standard, err := readPayload(cfg.ExportRoot, entry.Standard, expectedBytes)
		if err != nil {
			return nil, err
		}
		mont, err := readPayload(cfg.ExportRoot, entry.Montgomery, expectedBytes)
		if err != nil {
			return nil, err
		}
		if err = checkMontgomery(standard, mont, params.RingQ(), ref.Reference.ActiveLevel, words); err != nil {
			return nil, fmt.Errorf("reference %d: %w", ref.Reference.ReferenceSequence, err)
		}
		expectedFiles[filepath.ToSlash(entry.Standard.Path)] = true
		expectedFiles[filepath.ToSlash(entry.Montgomery.Path)] = true
		standardObjects[entry.Standard.Path+":"+entry.Standard.SHA256] = true
		montgomeryObjects[entry.Montgomery.Path+":"+entry.Montgomery.SHA256] = true
		if prior, ok := boundKeys[ref.Key]; ok {
			if prior[0] != entry.Standard || prior[1] != entry.Montgomery {
				return nil, fmt.Errorf("deduplicated object binding drift")
			}
		} else {
			boundKeys[ref.Key] = [2]PayloadBinding{entry.Standard, entry.Montgomery}
		}
		var expectedStandard, expectedMont []byte
		if ref.Class == "dense_slot_mask" {
			expectedStandard, expectedMont, err = independentlyEncodeDense(params, encoder, ref)
		} else {
			expectedStandard, expectedMont, err = independentlyCaptureResidual(params, evaluator, ref)
		}
		if err != nil {
			return nil, err
		}
		if !reflect.DeepEqual(expectedStandard, standard) || !reflect.DeepEqual(expectedMont, mont) {
			return nil, fmt.Errorf("independent Lattigo byte comparison failed at reference %d", ref.Reference.ReferenceSequence)
		}
		line, lineErr := transcriptLine(entry)
		if lineErr != nil {
			return nil, lineErr
		}
		transcript = append(transcript, []byte(line)...)
	}
	if dense != cfg.ExpectedDense || residual != cfg.ExpectedResidual || m.Coverage.UniqueStandardObjects != len(standardObjects) || m.Coverage.UniqueMontgomeryObjects != len(montgomeryObjects) {
		return nil, fmt.Errorf("unique/declared coverage drift")
	}
	transcriptPath, err := safeRegular(cfg.ExportRoot, m.EncodingTranscript.Path)
	if err != nil {
		return nil, err
	}
	actualTranscript, err := os.ReadFile(transcriptPath)
	if err != nil {
		return nil, err
	}
	if m.EncodingTranscript.Schema != TranscriptSchema || m.EncodingTranscript.Bytes != int64(len(actualTranscript)) || m.EncodingTranscript.SHA256 != shaBytes(actualTranscript) || m.EncodingTranscript.LineCount != cfg.ExpectedReferences+1 || !reflect.DeepEqual(actualTranscript, transcript) {
		return nil, fmt.Errorf("encoding transcript mismatch")
	}
	actualFiles := map[string]bool{}
	err = filepath.Walk(cfg.ExportRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == cfg.ExportRoot {
			return nil
		}
		rel, _ := filepath.Rel(cfg.ExportRoot, path)
		rel = filepath.ToSlash(rel)
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("symlink in export: %s", rel)
		}
		if info.IsDir() {
			if rel != "payloads" {
				return fmt.Errorf("unexpected directory in export: %s", rel)
			}
			return nil
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("non-regular export file: %s", rel)
		}
		actualFiles[rel] = true
		return nil
	})
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(actualFiles, expectedFiles) {
		return nil, fmt.Errorf("export file set is incomplete or contains trailing files")
	}
	validatorSources, validatorAggregate, err := sourceBindings(cfg.RepoRoot, cfg.ValidatorSourcePaths)
	if err != nil {
		return nil, err
	}
	manifestSHA := shaBytes(manifestData)
	receiptAuthorities := map[string]Binding{
		cfg.ParameterLiteral.Label: literalBinding,
		cfg.SemanticCatalog.Label:  catalogBinding,
		cfg.CommandStream.Label:    commandBinding,
		cfg.ScaleOracle.Label:      oracleBinding,
	}
	receipt := &Receipt{Schema: ReceiptSchema, Status: ReceiptStatus, Parameter: cfg.Parameter, Construction: cfg.Construction, EvidenceClass: ReceiptEvidence, Manifest: Binding{Path: "manifest.json", Bytes: int64(len(manifestData)), SHA256: manifestSHA}, Transcript: Binding{Path: m.EncodingTranscript.Path, Bytes: int64(len(actualTranscript)), SHA256: shaBytes(actualTranscript)}, Authorities: receiptAuthorities, ValidatorSourceBindings: validatorSources, ValidatorSourceAggregateSHA: validatorAggregate}
	receipt.ValidatorToolchain.GoVersion = runtime.Version()
	receipt.ValidatorToolchain.GOOS = runtime.GOOS
	receipt.ValidatorToolchain.GOARCH = runtime.GOARCH
	receipt.ValidatorToolchain.LattigoModule = LattigoModule
	receipt.ValidatorToolchain.LattigoVersion = LattigoVersion
	receipt.ValidatorToolchain.LattigoModuleSum = LattigoModuleSum
	receipt.ValidatorToolchain.IdentityMethod = identityMethod
	receipt.ValidatorToolchain.CoreSourceBindings = core
	receipt.Validation.ReferencesReencoded = cfg.ExpectedReferences
	receipt.Validation.UniqueObjectsReencoded = len(boundKeys)
	receipt.Validation.LattigoAPICalls = cfg.ExpectedReferences
	receipt.Validation.DenseReferences = dense
	receipt.Validation.ResidualReferences = residual
	receipt.Validation.StandardByteComparisons = cfg.ExpectedReferences
	receipt.Validation.MontgomeryByteComparisons = cfg.ExpectedReferences
	receipt.Validation.TranscriptExact = true
	receipt.Validation.FileSetExact = true
	receipt.AtomicCommit.Complete = true
	receipt.AtomicCommit.Method = "same_directory_temp_file_fsync_then_atomic_rename"
	receipt.ReleaseState.StaticReencodingValidated = true
	receipt.ReleaseState.KernelIntegrated = false
	receipt.ReleaseState.BoardEligible = false
	receipt.ReleaseState.MeasurementEligible = false
	receipt.ReleaseState.PerformanceMeasured = false
	if err = atomicReceipt(cfg.ReceiptPath, receipt); err != nil {
		return nil, err
	}
	return receipt, nil
}
