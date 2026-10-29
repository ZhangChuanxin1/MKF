package p2gn16staticexport

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

func testRepoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "tmp", "cpu_experiments", "go.mod")); err != nil {
		t.Fatalf("cannot locate repository root: %v", err)
	}
	return root
}

func testDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func testWriteJSON(t *testing.T, path string, value interface{}) []byte {
	t.Helper()
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return data
}

func testHex(values []uint64) []string {
	result := make([]string, len(values))
	for i, value := range values {
		result[i] = fmt.Sprintf("0x%x", value)
	}
	return result
}

type exporterFixture struct {
	root        string
	config      Config
	catalogPath string
	densePath   string
	params      ckks.Parameters
}

func newExporterFixture(t *testing.T) exporterFixture {
	t.Helper()
	root := t.TempDir()
	authorityRoot := filepath.Join(root, "authority")
	if err := os.MkdirAll(filepath.Join(authorityRoot, "payloads"), 0o755); err != nil {
		t.Fatal(err)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.PN12QP109)
	if err != nil {
		t.Fatal(err)
	}
	literal := map[string]interface{}{
		"schema":    "test-parameter-v1",
		"generator": map[string]interface{}{"dependency": LattigoModule + " " + LattigoVersion, "module_sum": LattigoModuleSum},
		"literal": map[string]interface{}{
			"identity_sha256": "test-identity", "log_n": params.LogN(), "n": params.N(),
			"log_slots": params.LogSlots(), "slots_used": 1 << params.LogSlots(),
			"scale": params.Scale(), "sigma": ckks.PN12QP109.Sigma,
			"q_post_conversion": testHex(params.Q()[:params.MaxLevel()]),
			"q_guard":           fmt.Sprintf("0x%x", params.Q()[params.MaxLevel()]), "p": testHex(params.P()),
			"persistent_max_level": params.MaxLevel() - 1, "guarded_max_level": params.MaxLevel(),
			"alpha": params.Alpha(), "beta": params.Beta(), "all_primes_prime": true,
			"all_primes_congruent_one_mod_2n": true,
		},
	}
	literalPath := filepath.Join(root, "literal.json")
	literalData := testWriteJSON(t, literalPath, literal)

	dense := make([]byte, (1<<params.LogSlots())*16)
	for slot := 0; slot < 1<<params.LogSlots(); slot++ {
		binary.LittleEndian.PutUint64(dense[16*slot:], math.Float64bits(float64((slot%7)-3)/8))
		binary.LittleEndian.PutUint64(dense[16*slot+8:], math.Float64bits(float64(slot%3)/16))
	}
	densePath := filepath.Join(authorityRoot, "payloads", "dense.bin")
	if err := os.WriteFile(densePath, dense, 0o644); err != nil {
		t.Fatal(err)
	}
	payloads := []map[string]interface{}{{
		"payload_id": "dense", "kind": "test_dense_mask",
		"encoding": "complex128_slots_real_imag_float64_bits_little_endian_v1",
		"path":     "payloads/dense.bin", "payload_bytes": len(dense),
		"content_sha256": testDigest(dense), "all_zero": false,
		"logical_slots": 1 << params.LogSlots(),
	}}
	for index := 0; index < ExpectedResidual; index++ {
		data := make([]byte, 16)
		binary.LittleEndian.PutUint64(data, math.Float64bits(float64(index+1)/17))
		binary.LittleEndian.PutUint64(data[8:], math.Float64bits(float64(index-3)/31))
		name := fmt.Sprintf("residual-%d", index)
		if err := os.WriteFile(filepath.Join(authorityRoot, "payloads", name+".bin"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		payloads = append(payloads, map[string]interface{}{
			"payload_id": name, "kind": "semantic_complex_scalar",
			"encoding": "complex128_real_imag_float64_bits_little_endian_v1",
			"path":     "payloads/" + name + ".bin", "payload_bytes": 16,
			"content_sha256": testDigest(data), "all_zero": false,
		})
	}
	refs := make([]map[string]interface{}, 0, ExpectedReferences)
	commands := make([]map[string]interface{}, 0, ExpectedReferences)
	for sequence := 0; sequence < ExpectedDense; sequence++ {
		command := 1000 + sequence
		operation := fmt.Sprintf("test.dense.%03d", sequence)
		refs = append(refs, map[string]interface{}{
			"reference_sequence": sequence, "command_sequence": command,
			"opcode": "plaintext_mul", "purpose": "packing", "phase": "test",
			"boundary": "test.dense", "operation": operation, "active_level": params.MaxLevel(),
			"semantic_kind": "test_dense_mask", "semantic_payload_id": "dense",
			"semantic_payload_exact": true, "operand_scale": params.Scale(),
		})
		commands = append(commands, map[string]interface{}{
			"sequence": command, "phase": "test", "boundary": "test.dense",
			"opcode": "plaintext_mul", "purpose": "packing", "operation": operation,
			"input_levels": []int{params.MaxLevel()}, "output_level": params.MaxLevel(),
		})
	}
	for index := 0; index < ExpectedResidual; index++ {
		sequence := ExpectedDense + index
		command := 2000 + index
		operation := fmt.Sprintf("test.residual.%d", index)
		refs = append(refs, map[string]interface{}{
			"reference_sequence": sequence, "command_sequence": command,
			"opcode": "plaintext_mul", "purpose": "residual", "phase": "test",
			"boundary": "test.residual", "operation": operation, "active_level": params.MaxLevel(),
			"semantic_kind": "semantic_complex_scalar", "semantic_payload_id": fmt.Sprintf("residual-%d", index),
			"semantic_payload_exact": true, "operand_scale": float64(params.Q()[params.MaxLevel()]),
		})
		commands = append(commands, map[string]interface{}{
			"sequence": command, "phase": "test", "boundary": "test.residual",
			"opcode": "plaintext_mul", "purpose": "residual", "operation": operation,
			"input_levels": []int{params.MaxLevel()}, "output_level": params.MaxLevel(),
		})
	}
	catalog := map[string]interface{}{
		"schema": "test-semantic-v1", "status": "TEST_ONLY", "parameter": "PTEST",
		"construction": "TEST-CONSTRUCTION", "performance_measured": false,
		"payloads": payloads, "references": refs,
		"release_state": map[string]bool{"hardware_payload_complete": false, "board_eligible": false, "performance_measured": false},
	}
	catalogPath := filepath.Join(authorityRoot, "catalog.json")
	catalogData := testWriteJSON(t, catalogPath, catalog)
	stream := map[string]interface{}{
		"schema": "test-command-authority-v1", "status": "TEST_ONLY", "parameter": "PTEST",
		"construction": "TEST-CONSTRUCTION", "performance_measured": false, "board_eligible": false,
		"command_stream": map[string]interface{}{"commands": commands},
	}
	streamPath := filepath.Join(root, "stream.json")
	streamData := testWriteJSON(t, streamPath, stream)
	oraclePath := filepath.Join(root, "oracle.json")
	oracleData := testWriteJSON(t, oraclePath, map[string]interface{}{"schema": "test-oracle-v1", "performance_measured": false})

	cfg := Config{
		RepoRoot: testRepoRoot(t), OutputRoot: filepath.Join(root, "export"),
		ParameterLiteral:    Authority{Label: "fixture/literal.json", Path: literalPath, ExpectedSHA256: testDigest(literalData)},
		SemanticCatalog:     Authority{Label: "fixture/catalog.json", Path: catalogPath, ExpectedSHA256: testDigest(catalogData)},
		CommandStream:       Authority{Label: "fixture/stream.json", Path: streamPath, ExpectedSHA256: testDigest(streamData)},
		ScaleOracle:         Authority{Label: "fixture/oracle.json", Path: oraclePath, ExpectedSHA256: testDigest(oracleData)},
		ExporterSourcePaths: ProductionConfig(testRepoRoot(t), "unused").ExporterSourcePaths,
		Schema:              "test-static-export-v1", Status: "TEST_ONLY_PASS", EvidenceClass: "test_fixture_only",
		Parameter: "PTEST", Construction: "TEST-CONSTRUCTION",
		ExpectedReferences: ExpectedReferences, ExpectedDense: ExpectedDense, ExpectedResidual: ExpectedResidual,
		StrictP2GN16: false,
	}
	return exporterFixture{root: root, config: cfg, catalogPath: catalogPath, densePath: densePath, params: params}
}

func TestRunCoversAll348ReferencesAndCommitsAtomically(t *testing.T) {
	fixture := newExporterFixture(t)
	manifest, err := Run(fixture.config)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Coverage.References != 348 || manifest.Coverage.DenseMaskReferences != 341 || manifest.Coverage.FractionalMultReferences != 7 {
		t.Fatalf("coverage drift: %+v", manifest.Coverage)
	}
	if manifest.Coverage.UniqueStandardObjects != 8 || manifest.Coverage.UniqueMontgomeryObjects != 8 {
		t.Fatalf("content deduplication drift: %+v", manifest.Coverage)
	}
	if !manifest.AtomicCommit.Complete || manifest.ReleaseState.BoardEligible || manifest.ReleaseState.PerformanceMeasured {
		t.Fatal("atomic/release state drift")
	}
	if _, err := os.Stat(filepath.Join(fixture.config.OutputRoot, "manifest.json")); err != nil {
		t.Fatal(err)
	}
	if matches, _ := filepath.Glob(filepath.Join(fixture.root, ".export.tmp-*")); len(matches) != 0 {
		t.Fatalf("temporary export directories remain: %v", matches)
	}
}

func TestProductionAuthoritiesAndExactCoveragePreflight(t *testing.T) {
	workDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	repoRoot := filepath.Clean(filepath.Join(workDir, "..", "..", "..", ".."))
	cfg := ProductionConfig(repoRoot, filepath.Join(t.TempDir(), "must-not-be-created"))
	literalData, _, err := readBound(cfg.ParameterLiteral)
	if err != nil {
		t.Fatalf("parameter authority: %v", err)
	}
	catalogData, _, err := readBound(cfg.SemanticCatalog)
	if err != nil {
		t.Fatalf("semantic authority: %v", err)
	}
	commandData, _, err := readBound(cfg.CommandStream)
	if err != nil {
		t.Fatalf("command authority: %v", err)
	}
	if _, _, err = readBound(cfg.ScaleOracle); err != nil {
		t.Fatalf("scale-oracle authority: %v", err)
	}
	var literal parameterAuthority
	if err = json.Unmarshal(literalData, &literal); err != nil {
		t.Fatalf("decode production literal: %v", err)
	}
	params, _, _, err := parameters(literal, cfg.StrictP2GN16)
	if err != nil {
		t.Fatalf("production literal: %v", err)
	}
	_, refs, err := prepare(cfg, catalogData, commandData, params)
	if err != nil {
		t.Fatalf("production semantic preflight: %v", err)
	}
	if len(refs) != ExpectedReferences {
		t.Fatalf("production references=%d, want %d", len(refs), ExpectedReferences)
	}
	dense, residual := 0, 0
	encoder := ckks.NewEncoder(params)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})
	denseAPISampled, residualAPICalls := false, 0
	for _, ref := range refs {
		switch ref.Class {
		case "dense_slot_mask":
			dense++
			if !denseAPISampled {
				standard, montgomery, apiErr := encodeDense(params, encoder, ref)
				if apiErr != nil || len(standard) == 0 || len(montgomery) != len(standard) {
					t.Fatalf("production dense EncodeNTT sample: %v", apiErr)
				}
				denseAPISampled = true
			}
		case "fractional_mult_by_const":
			residual++
			standard, montgomery, apiErr := encodeResidual(params, evaluator, ref)
			if apiErr != nil || len(standard) == 0 || len(montgomery) != len(standard) {
				t.Fatalf("production residual MultByConst reference %d: %v", ref.Reference.ReferenceSequence, apiErr)
			}
			residualAPICalls++
		default:
			t.Fatalf("unexpected production class %q", ref.Class)
		}
	}
	if dense != ExpectedDense || residual != ExpectedResidual {
		t.Fatalf("production coverage=%d/%d, want %d/%d", dense, residual, ExpectedDense, ExpectedResidual)
	}
	if !denseAPISampled || residualAPICalls != ExpectedResidual {
		t.Fatalf("production API preflight dense=%v residual=%d", denseAPISampled, residualAPICalls)
	}
	if _, err = os.Stat(cfg.OutputRoot); !os.IsNotExist(err) {
		t.Fatalf("read-only preflight created output: %v", err)
	}
}

func TestRunFailsClosedBeforeOutput(t *testing.T) {
	tests := []struct {
		name string
		edit func(*testing.T, *exporterFixture)
	}{
		{"semantic-payload-tamper", func(t *testing.T, fixture *exporterFixture) {
			data, _ := os.ReadFile(fixture.densePath)
			data[0] ^= 1
			if err := os.WriteFile(fixture.densePath, data, 0o644); err != nil {
				t.Fatal(err)
			}
		}},
		{"coverage", func(t *testing.T, fixture *exporterFixture) {
			var catalog map[string]interface{}
			data, _ := os.ReadFile(fixture.catalogPath)
			_ = json.Unmarshal(data, &catalog)
			refs := catalog["references"].([]interface{})
			catalog["references"] = refs[:len(refs)-1]
			updated := testWriteJSON(t, fixture.catalogPath, catalog)
			fixture.config.SemanticCatalog.ExpectedSHA256 = testDigest(updated)
		}},
		{"level", func(t *testing.T, fixture *exporterFixture) {
			var catalog map[string]interface{}
			data, _ := os.ReadFile(fixture.catalogPath)
			_ = json.Unmarshal(data, &catalog)
			catalog["references"].([]interface{})[0].(map[string]interface{})["active_level"] = float64(fixture.params.MaxLevel() + 1)
			updated := testWriteJSON(t, fixture.catalogPath, catalog)
			fixture.config.SemanticCatalog.ExpectedSHA256 = testDigest(updated)
		}},
		{"scale", func(t *testing.T, fixture *exporterFixture) {
			var catalog map[string]interface{}
			data, _ := os.ReadFile(fixture.catalogPath)
			_ = json.Unmarshal(data, &catalog)
			catalog["references"].([]interface{})[0].(map[string]interface{})["operand_scale"] = 0.0
			updated := testWriteJSON(t, fixture.catalogPath, catalog)
			fixture.config.SemanticCatalog.ExpectedSHA256 = testDigest(updated)
		}},
		{"existing-output", func(t *testing.T, fixture *exporterFixture) {
			if err := os.Mkdir(fixture.config.OutputRoot, 0o755); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newExporterFixture(t)
			test.edit(t, &fixture)
			if _, err := Run(fixture.config); err == nil {
				t.Fatal("invalid export unexpectedly passed")
			}
			if test.name != "existing-output" {
				if _, err := os.Stat(fixture.config.OutputRoot); !os.IsNotExist(err) {
					t.Fatalf("failed invocation created output: %v", err)
				}
			}
		})
	}
}
