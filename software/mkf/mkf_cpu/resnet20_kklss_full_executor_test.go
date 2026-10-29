package mkfcpu

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

type kklssFunctionalCountEvidence struct {
	OwnerCiphertexts   int `json:"owner_ciphertexts"`
	NetworkProducts    int `json:"network_ciphertext_products"`
	PolynomialProducts int `json:"polynomial_ciphertext_products"`
	Rotations          int `json:"rotations"`
	Degree59           int `json:"degree59_polynomials"`
	Refreshes          int `json:"collective_refreshes"`
}

func kklssFunctionalCounts(snapshot kklssPhaseSnapshot) kklssFunctionalCountEvidence {
	return kklssFunctionalCountEvidence{
		OwnerCiphertexts:   snapshot.owners,
		NetworkProducts:    snapshot.mulRelin,
		PolynomialProducts: snapshot.polynomialMulRelin,
		Rotations:          snapshot.rotations,
		Degree59:           snapshot.polynomials,
		Refreshes:          snapshot.refreshes,
	}
}

func kklssFullResNet20RequestFromEnvironment(t *testing.T, requireTail bool) fullResNet20Request {
	return kklssFullResNet20RequestFromEnvironmentFor(t, "P2", requireTail)
}

func kklssFullResNet20RequestFromEnvironmentFor(t *testing.T, parameter string, requireTail bool) fullResNet20Request {
	t.Helper()
	if parameter != "P2" && parameter != guardedCompactP2GN16ParameterID {
		t.Fatalf("public-KKLSS functional request requires P2 or P2GN16, got %q", parameter)
	}
	scales, err := loadFullResNet20Calibration(os.Getenv("MKF_RESNET20_ACTIVATION_SCALES_JSON"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	// KKLSS ciphertexts already carry the native per-owner components, so
	// owner ingress has no CKS exchange. The currently validated refresh
	// bridge performs two source conversions with two party shares each.
	plan.Communication.CKSIngressShareMessages = fullResNet20Parties * fullResNet20Parties * plan.Expected.CollectiveRefreshes
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

// TestKKLSSFullResNet20P2StemOptIn is the first integrated gate for the
// public KKLSS backend: three data-owner ciphertexts, 28 model-owner
// ciphertexts, 27 native KKLSS products, the frozen rotations, one
// collective-refresh bridge, and one degree-59 activation. Its wall time
// includes setup/oracles and is never a performance result.
func TestKKLSSFullResNet20P2StemOptIn(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_FULL_RESNET20_STEM") != "1" {
		t.Skip("set MKF_RUN_KKLSS_FULL_RESNET20_STEM=1 for the public-KKLSS P2 input+stem functional gate")
	}
	runKKLSSFullResNet20Stem(t, "P2")
}

// TestKKLSSFullResNet20P2GN16StemOptIn is the bounded fail-fast gate for the
// LogN=16 public-KKLSS path. It must pass before the several-hour full-network
// gate is attempted. Its setup-inclusive wall time is not performance data.
func TestKKLSSFullResNet20P2GN16StemOptIn(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_FULL_RESNET20_P2GN16_STEM") != "1" {
		t.Skip("set MKF_RUN_KKLSS_FULL_RESNET20_P2GN16_STEM=1 for the public-KKLSS P2GN16 input+stem functional gate")
	}
	runKKLSSFullResNet20Stem(t, guardedCompactP2GN16ParameterID)
}

func runKKLSSFullResNet20Stem(t *testing.T, parameter string) {
	t.Helper()
	request := kklssFullResNet20RequestFromEnvironmentFor(t, parameter, false)
	stemRotations := []int{-resnetImageWidth, -1, 1, resnetImageWidth}
	executorInterface, err := newKKLSSFullResNet20ExecutorWithRotations(request, stemRotations)
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
		for _, boundary := range result.Boundaries {
			t.Logf("KKLSS_BOUNDARY phase=%s name=%s level=%d checkedSlots=%d maxError=%.17g rmse=%.17g", phase.Name, boundary.Name, boundary.Level, boundary.CheckedSlots, boundary.MaxError, boundary.RMSE)
		}
	}
	counts := executor.kklssOperationSnapshot()
	if counts.owners != 31 || counts.mulRelin != 27 || counts.polynomialMulRelin != 17 || counts.rotations != 24 || counts.polynomials != 1 || counts.refreshes != 1 {
		t.Fatalf("public-KKLSS input+stem structural counts mismatch: %+v", counts)
	}
	t.Logf("KKLSS_STEM_PASS construction=%s parameter=%s owners=%d networkProducts=%d polynomialProducts=%d rotations=%d degree59=%d refresh=%d functional_only=true", kklssConstruction, parameter, counts.owners, counts.mulRelin, counts.polynomialMulRelin, counts.rotations, counts.polynomials, counts.refreshes)
}

// TestKKLSSFullResNet20P2FunctionalOptIn executes the exact frozen 12-phase
// schedule. Owner encryption, key generation, plaintext reference work, and
// boundary decryption remain in this functional harness, so its wall time is
// explicitly non-performance evidence.
func TestKKLSSFullResNet20P2FunctionalOptIn(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_FULL_RESNET20_P2") != "1" {
		t.Skip("set MKF_RUN_KKLSS_FULL_RESNET20_P2=1 only after the public-KKLSS stem gate passes")
	}
	runKKLSSFullResNet20Functional(t, "P2", "public-kklss-resnet20-p2-functional-phase-v2", "public-kklss-resnet20-p2-functional-summary-v2")
}

// TestKKLSSFullResNet20P2GN16Functional is the fail-closed, full-network
// correctness gate required before any matched P2GN16 timing. It executes the
// frozen 12 phases and 84 oracle boundaries with the public KKLSS
// implementation at LogN=16. Its wall clock includes setup, owner encryption,
// and oracle work and is therefore explicitly functional-only.
func TestKKLSSFullResNet20P2GN16Functional(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_FULL_RESNET20_P2GN16") != "1" {
		t.Skip("set MKF_RUN_KKLSS_FULL_RESNET20_P2GN16=1 only for the full public-KKLSS P2GN16 functional gate")
	}
	runKKLSSFullResNet20Functional(t, guardedCompactP2GN16ParameterID, "public-kklss-resnet20-p2gn16-functional-phase-v1", "public-kklss-resnet20-p2gn16-functional-summary-v1")
}

func runKKLSSFullResNet20Functional(t *testing.T, parameter, phaseSchema, summarySchema string) {
	t.Helper()
	functionalStarted := time.Now()
	request := kklssFullResNet20RequestFromEnvironmentFor(t, parameter, true)
	var executor *encryptedFullResNet20Executor
	request.PhaseObserver = func(result fullResNet20PhaseResult) {
		record := struct {
			Schema        string                          `json:"schema"`
			Phase         string                          `json:"phase"`
			Boundaries    []fullResNet20Boundary          `json:"boundaries"`
			Counts        kklssFunctionalCountEvidence    `json:"cumulative_native_counts"`
			Communication fullResNet20CommunicationCounts `json:"communication"`
		}{
			Schema: phaseSchema, Phase: result.Name,
			Boundaries: result.Boundaries, Counts: kklssFunctionalCounts(executor.kklssOperationSnapshot()), Communication: result.Communication,
		}
		encoded, err := json.Marshal(record)
		if err != nil {
			t.Fatalf("marshal public-KKLSS phase evidence: %v", err)
		}
		t.Logf("KKLSS_PHASE_JSON=%s", encoded)
	}
	executorInterface, err := newKKLSSFullResNet20Executor(request)
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
	if evidence.EncryptedTop1 != evidence.ReferenceTop1 || evidence.EncryptedTop1 != int(request.Fixture.Label) || evidence.ReferenceTop1 != int(request.Fixture.Label) || evidence.MaxLogitError >= fullResNet20FunctionalGate {
		t.Fatalf("public-KKLSS final classification gate failed: encryptedTop1=%d referenceTop1=%d label=%d maxLogitError=%g", evidence.EncryptedTop1, evidence.ReferenceTop1, request.Fixture.Label, evidence.MaxLogitError)
	}
	counts := executor.kklssOperationSnapshot()
	if counts.owners != 6201 || counts.mulRelin != 6172 || counts.polynomialMulRelin != 323 || counts.rotations != 6499 || counts.polynomials != 19 || counts.refreshes != 22 {
		t.Fatalf("full public-KKLSS structural counts mismatch: %+v", counts)
	}
	boundaryCount := 0
	maximumBoundaryError := 0.0
	maximumBoundary := ""
	for _, result := range results {
		for _, boundary := range result.Boundaries {
			boundaryCount++
			if boundary.MaxError > maximumBoundaryError {
				maximumBoundaryError, maximumBoundary = boundary.MaxError, boundary.Name
			}
		}
	}
	if boundaryCount != 84 {
		t.Fatalf("full public-KKLSS boundary count=%d, want 84", boundaryCount)
	}
	summary := struct {
		Schema                              string                              `json:"schema"`
		Status                              string                              `json:"status"`
		Construction                        string                              `json:"construction"`
		Parameter                           string                              `json:"parameter"`
		Parties                             int                                 `json:"parties"`
		Phases                              int                                 `json:"phases"`
		Boundaries                          int                                 `json:"boundaries"`
		NativeCounts                        kklssFunctionalCountEvidence        `json:"native_counts"`
		MaximumBoundary                     string                              `json:"maximum_boundary"`
		MaximumBoundaryError                float64                             `json:"maximum_boundary_error"`
		Classification                      *fullResNet20ClassificationEvidence `json:"classification"`
		FunctionalWallSecondsNotPerformance float64                             `json:"functional_wall_seconds_not_performance"`
		PerformanceMeasured                 bool                                `json:"performance_measured"`
	}{
		Schema: summarySchema, Status: "PASS", Construction: kklssConstruction,
		Parameter: parameter, Parties: fullResNet20Parties, Phases: len(results), Boundaries: boundaryCount,
		NativeCounts: kklssFunctionalCounts(counts), MaximumBoundary: maximumBoundary, MaximumBoundaryError: maximumBoundaryError,
		Classification: evidence, FunctionalWallSecondsNotPerformance: time.Since(functionalStarted).Seconds(), PerformanceMeasured: false,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("KKLSS_FULL_NETWORK_JSON=%s", encoded)
}
