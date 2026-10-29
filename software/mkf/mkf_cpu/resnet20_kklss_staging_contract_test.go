package mkfcpu

import (
	"fmt"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
)

const (
	kklssStagingSchema                    = "public-kklss-resnet20-owner-staging-v1"
	kklssOwnerCiphertextPolynomialCount   = 2 // c0 plus the sparse owner's component
	kklssStagingStorageMode               = "phase-segmented serialized native KKLSS ciphertexts"
	kklssPerformanceOnlineEncryptionGuard = "all owner encoding/encryption is complete before a timed segment"
)

type kklssStagingPhase struct {
	Name              string `json:"name"`
	DataCiphertexts   int    `json:"data_ciphertexts"`
	WeightCiphertexts int    `json:"weight_ciphertexts"`
	BiasCiphertexts   int    `json:"bias_ciphertexts"`
	OwnerCiphertexts  int    `json:"owner_ciphertexts"`
	EstimatedBytes    uint64 `json:"estimated_native_ciphertext_bytes"`
}

type kklssStagingManifest struct {
	Schema                      string              `json:"schema"`
	Parameter                   string              `json:"parameter"`
	Parties                     int                 `json:"parties"`
	StorageMode                 string              `json:"storage_mode"`
	OnlineEncryptionGuard       string              `json:"online_encryption_guard"`
	Phases                      []kklssStagingPhase `json:"phases"`
	DataCiphertexts             int                 `json:"data_ciphertexts"`
	WeightCiphertexts           int                 `json:"weight_ciphertexts"`
	BiasCiphertexts             int                 `json:"bias_ciphertexts"`
	OwnerCiphertexts            int                 `json:"owner_ciphertexts"`
	EstimatedAllCiphertextBytes uint64              `json:"estimated_all_native_ciphertext_bytes"`
	EstimatedPeakPhaseBytes     uint64              `json:"estimated_peak_phase_native_ciphertext_bytes"`
	EstimatedPeakPhase          string              `json:"estimated_peak_phase"`
}

type kklssTimedStageSegments struct {
	Stage                resNet20CPUStage   `json:"stage"`
	UntimedPrerequisites []string           `json:"untimed_prerequisite_phases"`
	TimedPhases          []string           `json:"timed_phases"`
	ExpectedTimedCounts  fullResNet20Counts `json:"expected_timed_counts"`
}

func buildKKLSSTimedStageSegments(plan fullResNet20Plan) ([]kklssTimedStageSegments, error) {
	if len(plan.Phases) != 12 || plan.Phases[0].Name != "input" || plan.Phases[1].Name != "stem" || plan.Phases[len(plan.Phases)-1].Name != "tail" {
		return nil, fmt.Errorf("KKLSS timing segmentation requires the exact frozen 12-phase plan")
	}
	segments := []kklssTimedStageSegments{
		{Stage: resNet20CPUStem, UntimedPrerequisites: []string{"input"}, TimedPhases: []string{"stem"}, ExpectedTimedCounts: plan.Phases[1].Expected},
		{Stage: resNet20CPUResidual},
		{Stage: resNet20CPUFullyConn, TimedPhases: []string{"tail"}, ExpectedTimedCounts: plan.Phases[len(plan.Phases)-1].Expected},
	}
	for _, phase := range plan.Phases[2 : len(plan.Phases)-1] {
		segments[1].TimedPhases = append(segments[1].TimedPhases, phase.Name)
		segments[1].ExpectedTimedCounts.add(phase.Expected)
	}
	var timed fullResNet20Counts
	for _, segment := range segments {
		timed.add(segment.ExpectedTimedCounts)
	}
	want := plan.Expected
	want.DataInputCiphertexts = 0 // input staging/encryption is outside every timer
	if timed != want || len(segments[1].TimedPhases) != 9 {
		return nil, fmt.Errorf("KKLSS timed phase segmentation mismatch: got=%+v want=%+v residualPhases=%d", timed, want, len(segments[1].TimedPhases))
	}
	return segments, nil
}

// kklssNativeSparseOwnerCiphertextBytes is a serialization/storage estimate,
// not a measured traffic figure. It counts the exact native sparse object:
// c0 and one owner component, one uint64 coefficient per active RNS prime.
func kklssNativeSparseOwnerCiphertextBytes(params ckks.Parameters) uint64 {
	return uint64(kklssOwnerCiphertextPolynomialCount) * uint64(params.N()) * uint64(len(params.Q())) * 8
}

func buildKKLSSStagingManifest(plan fullResNet20Plan, params ckks.Parameters) (kklssStagingManifest, error) {
	manifest := kklssStagingManifest{
		Schema: kklssStagingSchema, Parameter: "P2", Parties: fullResNet20Parties,
		StorageMode: kklssStagingStorageMode, OnlineEncryptionGuard: kklssPerformanceOnlineEncryptionGuard,
		Phases: make([]kklssStagingPhase, 0, len(plan.Phases)),
	}
	bytesPerCiphertext := kklssNativeSparseOwnerCiphertextBytes(params)
	if bytesPerCiphertext == 0 {
		return kklssStagingManifest{}, fmt.Errorf("invalid zero-byte KKLSS ciphertext estimate")
	}
	for _, phase := range plan.Phases {
		owners := phase.Expected.DataInputCiphertexts + phase.Expected.ModelWeightCiphertexts + phase.Expected.ModelBiasCiphertexts
		entry := kklssStagingPhase{
			Name: phase.Name, DataCiphertexts: phase.Expected.DataInputCiphertexts,
			WeightCiphertexts: phase.Expected.ModelWeightCiphertexts,
			BiasCiphertexts:   phase.Expected.ModelBiasCiphertexts,
			OwnerCiphertexts:  owners, EstimatedBytes: uint64(owners) * bytesPerCiphertext,
		}
		manifest.Phases = append(manifest.Phases, entry)
		manifest.DataCiphertexts += entry.DataCiphertexts
		manifest.WeightCiphertexts += entry.WeightCiphertexts
		manifest.BiasCiphertexts += entry.BiasCiphertexts
		manifest.OwnerCiphertexts += entry.OwnerCiphertexts
		manifest.EstimatedAllCiphertextBytes += entry.EstimatedBytes
		if entry.EstimatedBytes > manifest.EstimatedPeakPhaseBytes {
			manifest.EstimatedPeakPhaseBytes, manifest.EstimatedPeakPhase = entry.EstimatedBytes, entry.Name
		}
	}
	if len(manifest.Phases) != len(plan.Phases) || manifest.DataCiphertexts != 3 || manifest.WeightCiphertexts != 6172 || manifest.BiasCiphertexts != 26 || manifest.OwnerCiphertexts != 6201 {
		return kklssStagingManifest{}, fmt.Errorf("KKLSS staging coverage mismatch: phases=%d data=%d weights=%d bias=%d owners=%d", len(manifest.Phases), manifest.DataCiphertexts, manifest.WeightCiphertexts, manifest.BiasCiphertexts, manifest.OwnerCiphertexts)
	}
	for index, phase := range plan.Phases {
		entry := manifest.Phases[index]
		if entry.Name != phase.Name || entry.OwnerCiphertexts != phase.Expected.DataInputCiphertexts+phase.Expected.ModelWeightCiphertexts+phase.Expected.ModelBiasCiphertexts {
			return kklssStagingManifest{}, fmt.Errorf("KKLSS staging phase %d does not match frozen phase %s", index, phase.Name)
		}
	}
	return manifest, nil
}

func TestKKLSSOfflineStagingManifestContract(t *testing.T) {
	var scales fullResNet20ScaleSet
	for index := range scales.Sites {
		scales.Sites[index] = 1
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := buildKKLSSStagingManifest(plan, params)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.EstimatedAllCiphertextBytes <= manifest.EstimatedPeakPhaseBytes || manifest.EstimatedPeakPhase != "layer3.block1" {
		t.Fatalf("phase segmentation did not reduce staged owner-ciphertext residency: all=%d peak=%d phase=%s", manifest.EstimatedAllCiphertextBytes, manifest.EstimatedPeakPhaseBytes, manifest.EstimatedPeakPhase)
	}
	segments, err := buildKKLSSTimedStageSegments(plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(segments) != 3 || len(segments[0].UntimedPrerequisites) != 1 || len(segments[0].TimedPhases) != 1 || len(segments[1].TimedPhases) != 9 || len(segments[2].TimedPhases) != 1 {
		t.Fatalf("unexpected KKLSS CPU stage segmentation: %+v", segments)
	}
	// The generic plan counts compatibility owner-ingress CKS messages. A
	// KKLSS constructor must reject it before key generation unless the plan
	// is explicitly rebound to the validated four-share-per-refresh bridge.
	if executor, err := newKKLSSFullResNet20Executor(fullResNet20Request{Parameter: "P2", Mode: fullResNet20FunctionalMode, Plan: plan}); executor != nil || err == nil {
		t.Fatalf("public-KKLSS constructor accepted an unadapted communication plan: executor=%T err=%v", executor, err)
	}
	t.Logf("KKLSS_STAGING_MANIFEST_PASS phases=%d data=%d weights=%d bias=%d owners=%d bytesPerSparseOwner=%d estimatedAllBytes=%d estimatedPeakPhase=%s estimatedPeakBytes=%d",
		len(manifest.Phases), manifest.DataCiphertexts, manifest.WeightCiphertexts, manifest.BiasCiphertexts, manifest.OwnerCiphertexts,
		kklssNativeSparseOwnerCiphertextBytes(params), manifest.EstimatedAllCiphertextBytes, manifest.EstimatedPeakPhase, manifest.EstimatedPeakPhaseBytes)
	// A performance adapter must fail before even validating slots or keys;
	// this makes accidental online encoding/encryption a reproducible error.
	adapter := &kklssLattigoAdapter{allowOnlineOwnerEncryption: false}
	if _, err := adapter.ownerCarrier(nil, nil, 0, nil); err == nil {
		t.Fatal("performance KKLSS adapter accepted an unstaged owner ciphertext")
	}
}
