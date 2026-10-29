package mkfcpu

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

func paperMKFFullResNet20RequestFromEnvironment(t *testing.T, parameter string, requireTail bool) fullResNet20Request {
	t.Helper()
	scales, err := loadFullResNet20Calibration(os.Getenv("MKF_RESNET20_ACTIVATION_SCALES_JSON"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	// Proposed MKF accepts sparse n+1 owner ciphertexts directly. Preserve
	// the owner-ingress count but remove the compatibility executor's CKS
	// ingress messages from this backend-specific contract.
	plan.Communication.CKSIngressShareMessages = 0
	plan.Communication.CKSIngressBytes = 0
	fixture, err := loadFullResNet20OfficialFixture(os.Getenv("MKF_RESNET20_CIFAR_FIXTURE_DIR"))
	if err != nil {
		t.Fatal(err)
	}
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Fatal("MKF_RESNET20_WEIGHTS_DIR is required")
	}
	biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
	if requireTail {
		if biasPath == "" {
			t.Fatal("MKF_RESNET20_FC_BIAS_PATH is required")
		}
		biasData, err := os.ReadFile(biasPath)
		if err != nil {
			t.Fatal(err)
		}
		if digest := fmt.Sprintf("%x", sha256.Sum256(biasData)); digest != fullResNet20CheckpointFCBiasSHA {
			t.Fatalf("checkpoint FC-bias SHA-256 mismatch: %s", digest)
		}
	}
	return fullResNet20Request{
		Parameter:          parameter,
		Mode:               fullResNet20FunctionalMode,
		Plan:               plan,
		Fixture:            fixture,
		WeightsDirectory:   weightsDirectory,
		CheckpointBiasPath: biasPath,
	}
}

// TestPaperMKFFullResNet20P2StemOptIn is the minimal end-to-end executor
// contract: sparse owner input -> true MKF stem products/rotations ->
// two-party collective refresh -> degree-59 true MKF polynomial evaluation.
// It is intentionally opt-in and its wall time is not a performance result.
func TestPaperMKFFullResNet20P2StemOptIn(t *testing.T) {
	if os.Getenv("MKF_RUN_TRUE_FULL_RESNET20_STEM") != "1" {
		t.Skip("set MKF_RUN_TRUE_FULL_RESNET20_STEM=1 for the proposed-MKF P2 input+stem functional contract")
	}
	request := paperMKFFullResNet20RequestFromEnvironment(t, "P2", false)
	executorInterface, err := newPaperMKFFullResNet20Executor(request)
	if err != nil {
		t.Fatal(err)
	}
	executor := executorInterface.(*encryptedFullResNet20Executor)
	for _, phase := range request.Plan.Phases[:2] {
		result, err := executor.ExecutePhase(phase)
		if err != nil {
			t.Fatalf("%s: %v", phase.Name, err)
		}
		if err := validateFullResNet20PhaseResult(request.Mode, phase, result); err != nil {
			t.Fatal(err)
		}
		if result.Communication.CKSIngressShareMessages != 0 {
			t.Fatalf("%s unexpectedly reported CKS ingress", phase.Name)
		}
		for _, boundary := range result.Boundaries {
			t.Logf("TRUE_MKF_BOUNDARY phase=%s name=%s level=%d checkedSlots=%d maxError=%.17g rmse=%.17g", phase.Name, boundary.Name, boundary.Level, boundary.CheckedSlots, boundary.MaxError, boundary.RMSE)
		}
	}
	counts := executor.paperOperationSnapshot()
	if counts.owners != 31 || counts.mulRelin != 44 || counts.rotations != 24 || counts.polynomials != 1 || counts.refreshes != 1 {
		t.Fatalf("input+stem structural counts mismatch: %+v", counts)
	}
	t.Logf("TRUE_MKF_STEM_PASS construction=%s owners=%d mulRelin=%d rotations=%d degree59=%d refresh=%d decompositions=%d externalPolys=%d", paperMKFCKKSConstructionID, counts.owners, counts.mulRelin, counts.rotations, counts.polynomials, counts.refreshes, executor.paperAdapter.online.profile.Decompositions, executor.paperAdapter.online.profile.ExternalProductPolys)
}

// TestPaperMKFFullResNet20P2FunctionalOptIn executes all 12 frozen phases.
// Central test-only EvalKeyGen and plaintext boundary oracles make this a
// functional validation only; the elapsed wall time must never be reported as
// CPU inference performance.
func TestPaperMKFFullResNet20P2FunctionalOptIn(t *testing.T) {
	if os.Getenv("MKF_RUN_TRUE_FULL_RESNET20_P2") != "1" {
		t.Skip("set MKF_RUN_TRUE_FULL_RESNET20_P2=1 only after the proposed-MKF stem gate passes")
	}
	functionalStarted := time.Now()
	request := paperMKFFullResNet20RequestFromEnvironment(t, "P2", true)
	var executor *encryptedFullResNet20Executor
	request.PhaseObserver = func(result fullResNet20PhaseResult) {
		snapshot := executor.paperOperationSnapshot()
		record := struct {
			Schema                   string                          `json:"schema"`
			Phase                    string                          `json:"phase"`
			Boundaries               []fullResNet20Boundary          `json:"boundaries"`
			CumulativeOwners         int                             `json:"cumulative_owner_ciphertexts"`
			CumulativeMulRelin       int                             `json:"cumulative_multiply_relinearize"`
			CumulativeRotations      int                             `json:"cumulative_rotations"`
			CumulativePolynomials    int                             `json:"cumulative_degree59_polynomials"`
			CumulativeRefreshes      int                             `json:"cumulative_collective_refreshes"`
			CumulativeDecompositions int                             `json:"cumulative_decompositions"`
			CumulativeExternalPolys  int                             `json:"cumulative_external_product_polynomials"`
			Communication            fullResNet20CommunicationCounts `json:"communication"`
		}{
			Schema:                   "true-mkf-resnet20-p2-functional-phase-v1",
			Phase:                    result.Name,
			Boundaries:               result.Boundaries,
			CumulativeOwners:         snapshot.owners,
			CumulativeMulRelin:       snapshot.mulRelin,
			CumulativeRotations:      snapshot.rotations,
			CumulativePolynomials:    snapshot.polynomials,
			CumulativeRefreshes:      snapshot.refreshes,
			CumulativeDecompositions: executor.paperAdapter.online.profile.Decompositions,
			CumulativeExternalPolys:  executor.paperAdapter.online.profile.ExternalProductPolys,
			Communication:            result.Communication,
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			t.Fatalf("marshal proposed-MKF phase evidence: %v", err)
		}
		t.Logf("TRUE_MKF_PHASE_JSON=%s", encoded)
	}
	executorInterface, err := newPaperMKFFullResNet20Executor(request)
	if err != nil {
		t.Fatal(err)
	}
	executor = executorInterface.(*encryptedFullResNet20Executor)
	results, err := executeFullResNet20(request, executor)
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := executor.finalClassificationEvidence(int(request.Fixture.Label))
	if err != nil {
		t.Fatal(err)
	}
	counts := executor.paperOperationSnapshot()
	if counts.owners != 6201 || counts.mulRelin != 6495 || counts.rotations != 6499 || counts.polynomials != 19 || counts.refreshes != 22 {
		t.Fatalf("full proposed-MKF structural counts mismatch: %+v", counts)
	}
	maximumBoundaryError := 0.0
	maximumBoundary := ""
	boundaryCount := 0
	for _, result := range results {
		if result.Communication.CKSIngressShareMessages != 0 {
			t.Fatalf("%s unexpectedly reported CKS ingress", result.Name)
		}
		for _, boundary := range result.Boundaries {
			boundaryCount++
			if boundary.MaxError > maximumBoundaryError {
				maximumBoundaryError = boundary.MaxError
				maximumBoundary = boundary.Name
			}
		}
	}
	if boundaryCount != 84 {
		t.Fatalf("full proposed-MKF boundary count=%d, want 84", boundaryCount)
	}
	summary := struct {
		Schema                              string                              `json:"schema"`
		Status                              string                              `json:"status"`
		Construction                        string                              `json:"construction"`
		Parameter                           string                              `json:"parameter"`
		Parties                             int                                 `json:"parties"`
		Phases                              int                                 `json:"phases"`
		Boundaries                          int                                 `json:"boundaries"`
		Owners                              int                                 `json:"owner_ciphertexts"`
		MulRelin                            int                                 `json:"multiply_relinearize"`
		Rotations                           int                                 `json:"rotations"`
		Degree59                            int                                 `json:"degree59_polynomials"`
		Refreshes                           int                                 `json:"collective_refreshes"`
		Decompositions                      int                                 `json:"decompositions"`
		ExternalPolys                       int                                 `json:"external_product_polynomials"`
		MaximumBoundary                     string                              `json:"maximum_boundary"`
		MaximumBoundaryError                float64                             `json:"maximum_boundary_error"`
		Classification                      *fullResNet20ClassificationEvidence `json:"classification"`
		FunctionalWallSecondsNotPerformance float64                             `json:"functional_wall_seconds_not_performance"`
		PerformanceMeasured                 bool                                `json:"performance_measured"`
	}{
		Schema:                              "true-mkf-resnet20-p2-functional-summary-v1",
		Status:                              "PASS",
		Construction:                        paperMKFCKKSConstructionID,
		Parameter:                           "P2",
		Parties:                             fullResNet20Parties,
		Phases:                              len(results),
		Boundaries:                          boundaryCount,
		Owners:                              counts.owners,
		MulRelin:                            counts.mulRelin,
		Rotations:                           counts.rotations,
		Degree59:                            counts.polynomials,
		Refreshes:                           counts.refreshes,
		Decompositions:                      executor.paperAdapter.online.profile.Decompositions,
		ExternalPolys:                       executor.paperAdapter.online.profile.ExternalProductPolys,
		MaximumBoundary:                     maximumBoundary,
		MaximumBoundaryError:                maximumBoundaryError,
		Classification:                      evidence,
		FunctionalWallSecondsNotPerformance: time.Since(functionalStarted).Seconds(),
		PerformanceMeasured:                 false,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("TRUE_MKF_FULL_NETWORK_JSON=%s", encoded)
}
