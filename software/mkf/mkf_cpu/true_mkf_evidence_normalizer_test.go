package mkfcpu

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"
)

// trueMKFNormalizedEvidence is a read-only, construction-specific view of the
// TRUE_MKF_* records.  It deliberately does not feed functional wall time
// into the CPU benchmark schema and does not rewrite the source log.
type trueMKFNormalizedPhase struct {
	Phase                      string                      `json:"phase"`
	BoundaryCount              int                         `json:"boundary_count"`
	MaximumBoundaryError       float64                     `json:"maximum_boundary_error"`
	CumulativeOwnerCiphertexts int                         `json:"cumulative_owner_ciphertexts"`
	CumulativeMulRelin         int                         `json:"cumulative_multiply_relinearize"`
	CumulativeRotations        int                         `json:"cumulative_rotations"`
	CumulativeDegree59         int                         `json:"cumulative_degree59_polynomials"`
	CumulativeRefreshes        int                         `json:"cumulative_collective_refreshes"`
	CumulativeDecompositions   int                         `json:"cumulative_decompositions"`
	CumulativeExternalPolys    int                         `json:"cumulative_external_product_polynomials"`
	Boundaries                 []trueMKFNormalizedBoundary `json:"boundaries"`
}

type trueMKFNormalizedBoundary struct {
	Name          string  `json:"name"`
	Level         int     `json:"level"`
	CheckedSlots  int     `json:"checked_slots"`
	OracleChecked bool    `json:"oracle_checked"`
	MaxError      float64 `json:"max_error"`
	RMSE          float64 `json:"rmse"`
	Timed         bool    `json:"timed"`
	Cycles        uint64  `json:"cycles"`
	ElapsedNS     int64   `json:"elapsed_nanoseconds"`
}

type trueMKFNormalizedEvidence struct {
	Schema                  string                   `json:"schema"`
	SourceSHA256            string                   `json:"source_sha256"`
	Construction            string                   `json:"construction"`
	Parameter               string                   `json:"parameter"`
	Parties                 int                      `json:"parties"`
	Status                  string                   `json:"status"`
	FunctionalGate          float64                  `json:"functional_gate"`
	Phases                  []trueMKFNormalizedPhase `json:"phases"`
	BoundaryCount           int                      `json:"boundary_count"`
	MaximumBoundary         string                   `json:"maximum_boundary"`
	MaximumBoundaryError    float64                  `json:"maximum_boundary_error"`
	OwnerCiphertexts        int                      `json:"owner_ciphertexts"`
	MulRelin                int                      `json:"multiply_relinearize"`
	Rotations               int                      `json:"rotations"`
	Degree59                int                      `json:"degree59_polynomials"`
	Refreshes               int                      `json:"collective_refreshes"`
	Decompositions          int                      `json:"decompositions"`
	ExternalPolys           int                      `json:"external_product_polynomials"`
	EncryptedTop1           int                      `json:"encrypted_top1"`
	EncryptedMargin         float64                  `json:"encrypted_margin"`
	ReferenceTop1           int                      `json:"reference_polynomial_top1"`
	ReferenceMargin         float64                  `json:"reference_polynomial_margin"`
	GroundTruthLabel        int                      `json:"ground_truth_label"`
	MaximumLogitError       float64                  `json:"maximum_logit_error"`
	PerformanceMeasured     bool                     `json:"performance_measured"`
	FunctionalWallClockOnly float64                  `json:"functional_wall_seconds_not_performance"`
}

type trueMKFPhaseRecord struct {
	Schema     string `json:"schema"`
	Phase      string `json:"phase"`
	Boundaries []struct {
		Name          string  `json:"name"`
		Level         int     `json:"level"`
		CheckedSlots  int     `json:"checked_slots"`
		OracleChecked bool    `json:"oracle_checked"`
		MaxError      float64 `json:"max_error"`
		RMSE          float64 `json:"rmse"`
		Timed         bool    `json:"timed"`
		Cycles        uint64  `json:"cycles"`
		ElapsedNS     int64   `json:"elapsed_nanoseconds"`
	} `json:"boundaries"`
	CumulativeOwnerCiphertexts int `json:"cumulative_owner_ciphertexts"`
	CumulativeMulRelin         int `json:"cumulative_multiply_relinearize"`
	CumulativeRotations        int `json:"cumulative_rotations"`
	CumulativeDegree59         int `json:"cumulative_degree59_polynomials"`
	CumulativeRefreshes        int `json:"cumulative_collective_refreshes"`
	CumulativeDecompositions   int `json:"cumulative_decompositions"`
	CumulativeExternalPolys    int `json:"cumulative_external_product_polynomials"`
}

type trueMKFFullRecord struct {
	Schema               string  `json:"schema"`
	Status               string  `json:"status"`
	Construction         string  `json:"construction"`
	Parameter            string  `json:"parameter"`
	Parties              int     `json:"parties"`
	Phases               int     `json:"phases"`
	Boundaries           int     `json:"boundaries"`
	OwnerCiphertexts     int     `json:"owner_ciphertexts"`
	MulRelin             int     `json:"multiply_relinearize"`
	Rotations            int     `json:"rotations"`
	Degree59             int     `json:"degree59_polynomials"`
	Refreshes            int     `json:"collective_refreshes"`
	Decompositions       int     `json:"decompositions"`
	ExternalPolys        int     `json:"external_product_polynomials"`
	MaximumBoundary      string  `json:"maximum_boundary"`
	MaximumBoundaryError float64 `json:"maximum_boundary_error"`
	Classification       struct {
		GroundTruthLabel           int     `json:"ground_truth_label"`
		EncryptedTop1              int     `json:"encrypted_top1"`
		EncryptedMargin            float64 `json:"encrypted_margin"`
		EncryptedCorrect           bool    `json:"encrypted_correct"`
		ReferencePolynomialTop1    int     `json:"reference_polynomial_top1"`
		ReferencePolynomialMargin  float64 `json:"reference_polynomial_margin"`
		ReferencePolynomialCorrect bool    `json:"reference_polynomial_correct"`
		MaxLogitError              float64 `json:"max_logit_error"`
	} `json:"classification"`
	FunctionalWallClockOnly float64 `json:"functional_wall_seconds_not_performance"`
	PerformanceMeasured     bool    `json:"performance_measured"`
}

func trueMKFMarkerPayload(line, marker string) (string, bool) {
	index := strings.Index(line, marker+"=")
	if index < 0 {
		return "", false
	}
	payload := strings.TrimSpace(line[index+len(marker)+1:])
	return payload, payload != ""
}

func normalizeTrueMKFEvidenceReadOnly(reader io.Reader) (trueMKFNormalizedEvidence, error) {
	hasher := sha256.New()
	scanner := bufio.NewScanner(io.TeeReader(reader, hasher))
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	phaseRecords := make([]trueMKFPhaseRecord, 0, 12)
	var summary *trueMKFFullRecord
	for scanner.Scan() {
		line := scanner.Text()
		if payload, ok := trueMKFMarkerPayload(line, "TRUE_MKF_PHASE_JSON"); ok {
			var record trueMKFPhaseRecord
			if err := json.Unmarshal([]byte(payload), &record); err != nil {
				return trueMKFNormalizedEvidence{}, fmt.Errorf("decode TRUE_MKF_PHASE_JSON: %w", err)
			}
			phaseRecords = append(phaseRecords, record)
		}
		if payload, ok := trueMKFMarkerPayload(line, "TRUE_MKF_FULL_NETWORK_JSON"); ok {
			if summary != nil {
				return trueMKFNormalizedEvidence{}, fmt.Errorf("duplicate TRUE_MKF_FULL_NETWORK_JSON")
			}
			record := new(trueMKFFullRecord)
			if err := json.Unmarshal([]byte(payload), record); err != nil {
				return trueMKFNormalizedEvidence{}, fmt.Errorf("decode TRUE_MKF_FULL_NETWORK_JSON: %w", err)
			}
			summary = record
		}
	}
	if err := scanner.Err(); err != nil {
		return trueMKFNormalizedEvidence{}, err
	}
	if len(phaseRecords) != 12 || summary == nil {
		return trueMKFNormalizedEvidence{}, fmt.Errorf("true-MKF evidence coverage phases=%d summary=%t, want 12/true", len(phaseRecords), summary != nil)
	}
	wantNames := []string{"input", "stem", "layer1.block1", "layer1.block2", "layer1.block3", "layer2.block1", "layer2.block2", "layer2.block3", "layer3.block1", "layer3.block2", "layer3.block3", "tail"}
	normalized := trueMKFNormalizedEvidence{
		Schema: "true-mkf-resnet20-read-only-normalized-evidence-v1", SourceSHA256: fmt.Sprintf("%x", hasher.Sum(nil)),
		Construction: summary.Construction, Parameter: summary.Parameter, Parties: summary.Parties, Status: summary.Status,
		FunctionalGate: fullResNet20FunctionalGate, BoundaryCount: summary.Boundaries,
		MaximumBoundary: summary.MaximumBoundary, MaximumBoundaryError: summary.MaximumBoundaryError,
		OwnerCiphertexts: summary.OwnerCiphertexts, MulRelin: summary.MulRelin, Rotations: summary.Rotations,
		Degree59: summary.Degree59, Refreshes: summary.Refreshes, Decompositions: summary.Decompositions,
		ExternalPolys: summary.ExternalPolys, EncryptedTop1: summary.Classification.EncryptedTop1,
		EncryptedMargin:  summary.Classification.EncryptedMargin,
		ReferenceTop1:    summary.Classification.ReferencePolynomialTop1,
		ReferenceMargin:  summary.Classification.ReferencePolynomialMargin,
		GroundTruthLabel: summary.Classification.GroundTruthLabel, MaximumLogitError: summary.Classification.MaxLogitError,
		PerformanceMeasured: summary.PerformanceMeasured, FunctionalWallClockOnly: summary.FunctionalWallClockOnly,
	}
	boundaryCount := 0
	maximumBoundary, maximumBoundaryError := "", 0.0
	previous := trueMKFNormalizedPhase{}
	for index, phase := range phaseRecords {
		if phase.Schema != "true-mkf-resnet20-p2-functional-phase-v1" || phase.Phase != wantNames[index] || len(phase.Boundaries) == 0 {
			return trueMKFNormalizedEvidence{}, fmt.Errorf("invalid phase record %d schema=%q phase=%q boundaries=%d", index, phase.Schema, phase.Phase, len(phase.Boundaries))
		}
		maximum := 0.0
		normalizedBoundaries := make([]trueMKFNormalizedBoundary, 0, len(phase.Boundaries))
		for _, boundary := range phase.Boundaries {
			if boundary.Name == "" || boundary.CheckedSlots <= 0 || !boundary.OracleChecked || boundary.Timed || boundary.Cycles != 0 || boundary.ElapsedNS != 0 || math.IsNaN(boundary.MaxError) || math.IsInf(boundary.MaxError, 0) || math.IsNaN(boundary.RMSE) || math.IsInf(boundary.RMSE, 0) || boundary.MaxError < 0 || boundary.RMSE < 0 || boundary.MaxError >= fullResNet20FunctionalGate {
				return trueMKFNormalizedEvidence{}, fmt.Errorf("phase %s has invalid boundary %q error=%g", phase.Phase, boundary.Name, boundary.MaxError)
			}
			if index == 0 && boundary.Name == "input.normalized" {
				if boundary.Level != -1 {
					return trueMKFNormalizedEvidence{}, fmt.Errorf("input.normalized level=%d, want -1", boundary.Level)
				}
			} else if boundary.Level < 0 {
				return trueMKFNormalizedEvidence{}, fmt.Errorf("phase %s boundary %s has invalid level %d", phase.Phase, boundary.Name, boundary.Level)
			}
			maximum = math.Max(maximum, boundary.MaxError)
			if boundary.MaxError > maximumBoundaryError {
				maximumBoundary, maximumBoundaryError = boundary.Name, boundary.MaxError
			}
			normalizedBoundaries = append(normalizedBoundaries, trueMKFNormalizedBoundary{
				Name: boundary.Name, Level: boundary.Level, CheckedSlots: boundary.CheckedSlots,
				OracleChecked: boundary.OracleChecked, MaxError: boundary.MaxError, RMSE: boundary.RMSE,
				Timed: boundary.Timed, Cycles: boundary.Cycles, ElapsedNS: boundary.ElapsedNS,
			})
		}
		boundaryCount += len(phase.Boundaries)
		normalizedPhase := trueMKFNormalizedPhase{
			Phase: phase.Phase, BoundaryCount: len(phase.Boundaries), MaximumBoundaryError: maximum,
			CumulativeOwnerCiphertexts: phase.CumulativeOwnerCiphertexts, CumulativeMulRelin: phase.CumulativeMulRelin,
			CumulativeRotations: phase.CumulativeRotations, CumulativeDegree59: phase.CumulativeDegree59,
			CumulativeRefreshes: phase.CumulativeRefreshes, CumulativeDecompositions: phase.CumulativeDecompositions,
			CumulativeExternalPolys: phase.CumulativeExternalPolys, Boundaries: normalizedBoundaries,
		}
		if index > 0 && (normalizedPhase.CumulativeOwnerCiphertexts < previous.CumulativeOwnerCiphertexts || normalizedPhase.CumulativeMulRelin < previous.CumulativeMulRelin || normalizedPhase.CumulativeRotations < previous.CumulativeRotations || normalizedPhase.CumulativeDegree59 < previous.CumulativeDegree59 || normalizedPhase.CumulativeRefreshes < previous.CumulativeRefreshes || normalizedPhase.CumulativeDecompositions < previous.CumulativeDecompositions || normalizedPhase.CumulativeExternalPolys < previous.CumulativeExternalPolys) {
			return trueMKFNormalizedEvidence{}, fmt.Errorf("phase %s has regressing cumulative counters", phase.Phase)
		}
		normalized.Phases = append(normalized.Phases, normalizedPhase)
		previous = normalizedPhase
	}
	if summary.Schema != "true-mkf-resnet20-p2-functional-summary-v1" || summary.Status != "PASS" || summary.Construction != paperMKFCKKSConstructionID || summary.Parameter != "P2" || summary.Parties != 2 || summary.Phases != 12 || summary.Boundaries != 84 || boundaryCount != 84 || summary.PerformanceMeasured || summary.FunctionalWallClockOnly <= 0 {
		return trueMKFNormalizedEvidence{}, fmt.Errorf("invalid true-MKF final identity/status/coverage")
	}
	if summary.OwnerCiphertexts != 6201 || summary.MulRelin != 6495 || summary.Rotations != 6499 || summary.Degree59 != 19 || summary.Refreshes != 22 || summary.Decompositions != 45517 || summary.ExternalPolys != 91034 {
		return trueMKFNormalizedEvidence{}, fmt.Errorf("invalid true-MKF final operation counts")
	}
	if summary.Classification.EncryptedTop1 != summary.Classification.GroundTruthLabel || summary.Classification.ReferencePolynomialTop1 != summary.Classification.GroundTruthLabel || !summary.Classification.EncryptedCorrect || !summary.Classification.ReferencePolynomialCorrect || summary.Classification.EncryptedMargin <= 0 || summary.Classification.ReferencePolynomialMargin <= 0 || summary.MaximumBoundary != maximumBoundary || summary.MaximumBoundaryError != maximumBoundaryError || summary.MaximumBoundaryError >= fullResNet20FunctionalGate || math.IsNaN(summary.Classification.MaxLogitError) || math.IsInf(summary.Classification.MaxLogitError, 0) || summary.Classification.MaxLogitError < 0 || summary.Classification.MaxLogitError >= fullResNet20FunctionalGate {
		return trueMKFNormalizedEvidence{}, fmt.Errorf("invalid true-MKF final correctness evidence")
	}
	last := normalized.Phases[len(normalized.Phases)-1]
	if last.CumulativeOwnerCiphertexts != summary.OwnerCiphertexts || last.CumulativeMulRelin != summary.MulRelin || last.CumulativeRotations != summary.Rotations || last.CumulativeDegree59 != summary.Degree59 || last.CumulativeRefreshes != summary.Refreshes || last.CumulativeDecompositions != summary.Decompositions || last.CumulativeExternalPolys != summary.ExternalPolys {
		return trueMKFNormalizedEvidence{}, fmt.Errorf("phase totals disagree with final true-MKF summary")
	}
	return normalized, nil
}

func TestNormalizeTrueMKFEvidenceReadOnlyOptIn(t *testing.T) {
	path := os.Getenv("MKF_TRUE_MKF_EVIDENCE_LOG")
	if path == "" {
		t.Skip("set MKF_TRUE_MKF_EVIDENCE_LOG for read-only TRUE_MKF_* normalization")
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	normalized, err := normalizeTrueMKFEvidenceReadOnly(file)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("TRUE_MKF_NORMALIZED_JSON=%s", encoded)
}

func TestNormalizeTrueMKFEvidenceRejectsMissingCoverage(t *testing.T) {
	_, err := normalizeTrueMKFEvidenceReadOnly(bytes.NewBufferString("TRUE_MKF_PHASE_JSON={}\n"))
	if err == nil || !strings.Contains(err.Error(), "coverage") {
		t.Fatalf("incomplete evidence did not fail closed: %v", err)
	}
}
