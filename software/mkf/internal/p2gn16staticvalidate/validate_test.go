package p2gn16staticvalidate

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"

	"mkf-cpu-experiments/internal/p2gn16staticexport"
)

type validationFixture struct {
	repoRoot      string
	root          string
	exportRoot    string
	receiptPath   string
	denseSemantic string
	exportConfig  p2gn16staticexport.Config
	config        Config
	params        ckks.Parameters
}

func validatorRepoRoot(t *testing.T) string {
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

func fixtureSHA(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func fixtureJSON(t *testing.T, path string, value interface{}) []byte {
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

func fixtureHex(values []uint64) []string {
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = fmt.Sprintf("0x%x", value)
	}
	return result
}

func newValidationFixture(t *testing.T) validationFixture {
	t.Helper()
	repoRoot := validatorRepoRoot(t)
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
			"q_post_conversion": fixtureHex(params.Q()[:params.MaxLevel()]),
			"q_guard":           fmt.Sprintf("0x%x", params.Q()[params.MaxLevel()]), "p": fixtureHex(params.P()),
			"persistent_max_level": params.MaxLevel() - 1, "guarded_max_level": params.MaxLevel(),
			"alpha": params.Alpha(), "beta": params.Beta(), "all_primes_prime": true,
			"all_primes_congruent_one_mod_2n": true,
		},
	}
	literalPath := filepath.Join(root, "literal.json")
	literalData := fixtureJSON(t, literalPath, literal)
	dense := make([]byte, (1<<params.LogSlots())*16)
	for slot := 0; slot < 1<<params.LogSlots(); slot++ {
		binary.LittleEndian.PutUint64(dense[16*slot:], math.Float64bits(float64((slot%11)-5)/16))
		binary.LittleEndian.PutUint64(dense[16*slot+8:], math.Float64bits(float64(slot%5)/32))
	}
	densePath := filepath.Join(authorityRoot, "payloads", "dense.bin")
	if err := os.WriteFile(densePath, dense, 0o644); err != nil {
		t.Fatal(err)
	}
	payloads := []map[string]interface{}{{
		"payload_id": "dense", "kind": "test_dense_mask",
		"encoding": "complex128_slots_real_imag_float64_bits_little_endian_v1",
		"path":     "payloads/dense.bin", "payload_bytes": len(dense),
		"content_sha256": fixtureSHA(dense), "logical_slots": 1 << params.LogSlots(),
	}}
	for index := 0; index < ExpectedResidual; index++ {
		data := make([]byte, 16)
		binary.LittleEndian.PutUint64(data, math.Float64bits(float64(index+1)/19))
		binary.LittleEndian.PutUint64(data[8:], math.Float64bits(float64(index-2)/37))
		name := fmt.Sprintf("residual-%d", index)
		if err := os.WriteFile(filepath.Join(authorityRoot, "payloads", name+".bin"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		payloads = append(payloads, map[string]interface{}{
			"payload_id": name, "kind": "semantic_complex_scalar",
			"encoding": "complex128_real_imag_float64_bits_little_endian_v1",
			"path":     "payloads/" + name + ".bin", "payload_bytes": 16,
			"content_sha256": fixtureSHA(data),
		})
	}
	refs := make([]map[string]interface{}, 0, ExpectedReferences)
	commands := make([]map[string]interface{}, 0, ExpectedReferences)
	for sequence := 0; sequence < ExpectedReferences; sequence++ {
		isDense := sequence < ExpectedDense
		command := 1000 + sequence
		purpose, boundary, kind, payloadID := "packing", "test.dense", "test_dense_mask", "dense"
		if !isDense {
			purpose, boundary, kind = "residual", "test.residual", "semantic_complex_scalar"
			payloadID = fmt.Sprintf("residual-%d", sequence-ExpectedDense)
		}
		operation := fmt.Sprintf("test.%s.%03d", purpose, sequence)
		scale := params.Scale()
		if !isDense {
			scale = float64(params.Q()[params.MaxLevel()])
		}
		refs = append(refs, map[string]interface{}{
			"reference_sequence": sequence, "command_sequence": command,
			"opcode": "plaintext_mul", "purpose": purpose, "phase": "test",
			"boundary": boundary, "operation": operation, "active_level": params.MaxLevel(),
			"semantic_kind": kind, "semantic_payload_id": payloadID,
			"semantic_payload_exact": true, "operand_scale": scale,
		})
		commands = append(commands, map[string]interface{}{
			"sequence": command, "phase": "test", "boundary": boundary,
			"opcode": "plaintext_mul", "purpose": purpose, "operation": operation,
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
	catalogData := fixtureJSON(t, catalogPath, catalog)
	stream := map[string]interface{}{
		"schema": "test-stream-v1", "status": "TEST_ONLY", "parameter": "PTEST",
		"construction": "TEST-CONSTRUCTION", "performance_measured": false, "board_eligible": false,
		"command_stream": map[string]interface{}{"commands": commands},
	}
	streamPath := filepath.Join(root, "stream.json")
	streamData := fixtureJSON(t, streamPath, stream)
	oraclePath := filepath.Join(root, "oracle.json")
	oracleData := fixtureJSON(t, oraclePath, map[string]interface{}{"schema": "test-oracle-v1", "performance_measured": false})
	exportRoot := filepath.Join(root, "export")
	exportCfg := p2gn16staticexport.Config{
		RepoRoot: repoRoot, OutputRoot: exportRoot,
		ParameterLiteral:    p2gn16staticexport.Authority{Label: "fixture/literal.json", Path: literalPath, ExpectedSHA256: fixtureSHA(literalData)},
		SemanticCatalog:     p2gn16staticexport.Authority{Label: "fixture/catalog.json", Path: catalogPath, ExpectedSHA256: fixtureSHA(catalogData)},
		CommandStream:       p2gn16staticexport.Authority{Label: "fixture/stream.json", Path: streamPath, ExpectedSHA256: fixtureSHA(streamData)},
		ScaleOracle:         p2gn16staticexport.Authority{Label: "fixture/oracle.json", Path: oraclePath, ExpectedSHA256: fixtureSHA(oracleData)},
		ExporterSourcePaths: p2gn16staticexport.ProductionConfig(repoRoot, "unused").ExporterSourcePaths,
		Schema:              "test-static-export-v2", Status: "TEST_EXACT_REENCODING_PASS", EvidenceClass: "test_fixture_only",
		Parameter: "PTEST", Construction: "TEST-CONSTRUCTION",
		ExpectedReferences: ExpectedReferences, ExpectedDense: ExpectedDense,
		ExpectedResidual: ExpectedResidual, StrictP2GN16: false,
	}
	if _, err := p2gn16staticexport.Run(exportCfg); err != nil {
		t.Fatal(err)
	}
	validatorProduction := ProductionConfig(repoRoot, "unused", "unused")
	receiptPath := filepath.Join(root, "validation-receipt.json")
	validateCfg := Config{
		RepoRoot: repoRoot, ExportRoot: exportRoot, ReceiptPath: receiptPath,
		ParameterLiteral:     Authority{Label: "fixture/literal.json", Path: literalPath, ExpectedSHA256: fixtureSHA(literalData)},
		SemanticCatalog:      Authority{Label: "fixture/catalog.json", Path: catalogPath, ExpectedSHA256: fixtureSHA(catalogData)},
		CommandStream:        Authority{Label: "fixture/stream.json", Path: streamPath, ExpectedSHA256: fixtureSHA(streamData)},
		ScaleOracle:          Authority{Label: "fixture/oracle.json", Path: oraclePath, ExpectedSHA256: fixtureSHA(oracleData)},
		ExporterSourcePaths:  exportCfg.ExporterSourcePaths,
		ValidatorSourcePaths: validatorProduction.ValidatorSourcePaths,
		Schema:               exportCfg.Schema, Status: exportCfg.Status, EvidenceClass: exportCfg.EvidenceClass,
		Parameter: exportCfg.Parameter, Construction: exportCfg.Construction,
		ExpectedReferences: ExpectedReferences, ExpectedDense: ExpectedDense,
		ExpectedResidual: ExpectedResidual, StrictP2GN16: false,
	}
	return validationFixture{repoRoot: repoRoot, root: root, exportRoot: exportRoot, receiptPath: receiptPath, denseSemantic: densePath, exportConfig: exportCfg, config: validateCfg, params: params}
}

func loadFixtureManifest(t *testing.T, fixture validationFixture) manifest {
	t.Helper()
	var value manifest
	data, err := os.ReadFile(filepath.Join(fixture.exportRoot, "manifest.json"))
	if err != nil || json.Unmarshal(data, &value) != nil {
		t.Fatalf("load manifest: %v", err)
	}
	return value
}

func resignFixtureManifest(t *testing.T, fixture validationFixture, value *manifest) {
	t.Helper()
	transcript := []byte(transcriptHeader(value))
	for _, entry := range value.Entries {
		line, err := transcriptLine(entry)
		if err != nil {
			t.Fatal(err)
		}
		transcript = append(transcript, []byte(line)...)
	}
	transcriptPath := filepath.Join(fixture.exportRoot, filepath.FromSlash(value.EncodingTranscript.Path))
	if err := os.WriteFile(transcriptPath, transcript, 0o644); err != nil {
		t.Fatal(err)
	}
	value.EncodingTranscript.Bytes = int64(len(transcript))
	value.EncodingTranscript.SHA256 = fixtureSHA(transcript)
	fixtureJSON(t, filepath.Join(fixture.exportRoot, "manifest.json"), value)
}

func mutateBoundFile(t *testing.T, fixture validationFixture, domain string, selfSign bool, mathematicallyMatch bool, trailing bool) {
	t.Helper()
	value := loadFixtureManifest(t, fixture)
	entry := value.Entries[0]
	binding := entry.Standard
	if domain == "montgomery" {
		binding = entry.Montgomery
	}
	path := filepath.Join(fixture.exportRoot, filepath.FromSlash(binding.Path))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if trailing {
		data = append(data, 0)
	} else {
		data[0] ^= 1
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if !selfSign {
		return
	}
	if domain == "standard" && mathematicallyMatch && !trailing {
		mont, err := toMontgomery(data, fixture.params.RingQ(), entry.ActiveLevel, fixture.params.N())
		if err != nil {
			t.Fatal(err)
		}
		montPath := filepath.Join(fixture.exportRoot, filepath.FromSlash(entry.Montgomery.Path))
		if err := os.WriteFile(montPath, mont, 0o644); err != nil {
			t.Fatal(err)
		}
		for index := range value.Entries {
			if value.Entries[index].Montgomery.Path == entry.Montgomery.Path {
				value.Entries[index].Montgomery.Bytes = int64(len(mont))
				value.Entries[index].Montgomery.SHA256 = fixtureSHA(mont)
			}
		}
	}
	for index := range value.Entries {
		var candidate *PayloadBinding
		if domain == "standard" {
			candidate = &value.Entries[index].Standard
		} else {
			candidate = &value.Entries[index].Montgomery
		}
		if candidate.Path == binding.Path {
			candidate.Bytes = int64(len(data))
			candidate.SHA256 = fixtureSHA(data)
		}
	}
	resignFixtureManifest(t, fixture, &value)
}

func TestValidateReencodesAll348AndWritesAtomicReceipt(t *testing.T) {
	fixture := newValidationFixture(t)
	receipt, err := Validate(fixture.config)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Validation.ReferencesReencoded != 348 || receipt.Validation.LattigoAPICalls != 348 || receipt.Validation.DenseReferences != 341 || receipt.Validation.ResidualReferences != 7 || receipt.Validation.UniqueObjectsReencoded != 8 || !receipt.Validation.TranscriptExact || !receipt.Validation.FileSetExact {
		t.Fatalf("receipt validation coverage drift: %+v", receipt.Validation)
	}
	if !receipt.AtomicCommit.Complete || !receipt.ReleaseState.StaticReencodingValidated || receipt.ReleaseState.BoardEligible || receipt.ReleaseState.PerformanceMeasured {
		t.Fatal("receipt atomic/release state drift")
	}
	if receipt.EvidenceClass != ReceiptEvidence || receipt.ValidatorToolchain.LattigoVersion != LattigoVersion || receipt.ValidatorToolchain.LattigoModuleSum != LattigoModuleSum || len(receipt.ValidatorToolchain.CoreSourceBindings) != len(coreSourceSHA256) {
		t.Fatal("receipt independent toolchain evidence drift")
	}
	if len(receipt.Authorities) != 4 {
		t.Fatalf("receipt authority bindings=%d, want 4", len(receipt.Authorities))
	}
	if _, err := os.Stat(fixture.receiptPath); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejectsTamperAndNeverWritesReceipt(t *testing.T) {
	tests := []struct {
		name string
		edit func(*testing.T, validationFixture)
	}{
		{"payload-tamper", func(t *testing.T, f validationFixture) { mutateBoundFile(t, f, "standard", false, false, false) }},
		{"self-signed-residue-and-montgomery", func(t *testing.T, f validationFixture) { mutateBoundFile(t, f, "standard", true, true, false) }},
		{"standard-montgomery-relation", func(t *testing.T, f validationFixture) { mutateBoundFile(t, f, "montgomery", true, false, false) }},
		{"trailing-payload", func(t *testing.T, f validationFixture) { mutateBoundFile(t, f, "standard", true, false, true) }},
		{"truncated-payload", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			binding := value.Entries[0].Standard
			path := filepath.Join(f.exportRoot, filepath.FromSlash(binding.Path))
			data, _ := os.ReadFile(path)
			data = data[:len(data)-1]
			_ = os.WriteFile(path, data, 0o644)
			for index := range value.Entries {
				if value.Entries[index].Standard.Path == binding.Path {
					value.Entries[index].Standard.Bytes = int64(len(data))
					value.Entries[index].Standard.SHA256 = fixtureSHA(data)
				}
			}
			resignFixtureManifest(t, f, &value)
		}},
		{"semantic-payload", func(t *testing.T, f validationFixture) {
			data, _ := os.ReadFile(f.denseSemantic)
			data[0] ^= 1
			_ = os.WriteFile(f.denseSemantic, data, 0o644)
		}},
		{"source-binding", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			binding := value.SourceBindings[f.config.SemanticCatalog.Label]
			binding.SHA256 = strings.Repeat("0", 64)
			value.SourceBindings[f.config.SemanticCatalog.Label] = binding
			resignFixtureManifest(t, f, &value)
		}},
		{"version", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			value.Toolchain.LattigoVersion = "v2.3.1"
			resignFixtureManifest(t, f, &value)
		}},
		{"level", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			value.Entries[0].ActiveLevel--
			resignFixtureManifest(t, f, &value)
		}},
		{"scale", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			value.Entries[0].OperandScaleLog2Q3232++
			resignFixtureManifest(t, f, &value)
		}},
		{"coverage", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			value.Coverage.References--
			resignFixtureManifest(t, f, &value)
		}},
		{"non-atomic", func(t *testing.T, f validationFixture) {
			value := loadFixtureManifest(t, f)
			value.AtomicCommit.Complete = false
			resignFixtureManifest(t, f, &value)
		}},
		{"incomplete-file-set", func(t *testing.T, f validationFixture) {
			_ = os.WriteFile(filepath.Join(f.exportRoot, ".incomplete"), []byte("partial\n"), 0o644)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newValidationFixture(t)
			test.edit(t, fixture)
			if _, err := Validate(fixture.config); err == nil {
				t.Fatal("tampered static export unexpectedly validated")
			}
			if _, err := os.Stat(fixture.receiptPath); !os.IsNotExist(err) {
				t.Fatalf("failed validation created receipt: %v", err)
			}
		})
	}
}
