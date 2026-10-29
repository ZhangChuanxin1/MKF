package mkfcpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// p2gn16FPGARotationInventory is a machine-readable bridge from the frozen
// Go scheduler to the future FPGA host runner. It is deliberately generated
// from fullResNet20RotationKeys instead of duplicating the rotation list in a
// Python script or a hardware header.
type p2gn16FPGARotationInventory struct {
	Schema              string   `json:"schema"`
	Parameter           string   `json:"parameter"`
	Construction        string   `json:"construction"`
	LogN                int      `json:"log_n"`
	N                   int      `json:"n"`
	LogSlots            int      `json:"log_slots"`
	PersistentMaxLevel  int      `json:"persistent_max_level"`
	RotationCount       int      `json:"rotation_count"`
	SignedRotations     []int    `json:"signed_rotations"`
	GaloisElements      []uint64 `json:"galois_elements"`
	OrderedPairSHA256   string   `json:"ordered_rotation_galois_pair_sha256"`
	PerformanceMeasured bool     `json:"performance_measured"`
}

func TestExportP2GN16FPGARotationInventoryOptIn(t *testing.T) {
	output := os.Getenv("MKF_P2GN16_FPGA_ROTATION_INVENTORY_JSON")
	if output == "" {
		t.Skip("set MKF_P2GN16_FPGA_ROTATION_INVENTORY_JSON to export the light P2GN16 rotation inventory")
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	rotations := fullResNet20RotationKeys()
	if len(rotations) != 164 {
		t.Fatalf("P2GN16 rotation inventory has %d entries, want 164", len(rotations))
	}
	galois := make([]uint64, len(rotations))
	seen := make(map[uint64]struct{}, len(rotations))
	hashInput := make([]byte, 0, len(rotations)*40)
	for index, rotation := range rotations {
		galois[index] = profile.base.GaloisElementForColumnRotationBy(rotation)
		if _, exists := seen[galois[index]]; exists {
			t.Fatalf("duplicate Galois element %d for signed rotation %d", galois[index], rotation)
		}
		seen[galois[index]] = struct{}{}
		hashInput = append(hashInput, []byte(fmt.Sprintf("%d:%d\n", rotation, galois[index]))...)
	}
	pairsDigest := sha256.Sum256(hashInput)
	record := p2gn16FPGARotationInventory{
		Schema:              "mkf-p2gn16-resnet20-rotation-inventory-v1",
		Parameter:           guardedCompactP2GN16ParameterID,
		Construction:        guardedCompactP2GN16CPUConstructionID,
		LogN:                profile.base.LogN(),
		N:                   profile.base.N(),
		LogSlots:            profile.base.LogSlots(),
		PersistentMaxLevel:  profile.base.MaxLevel(),
		RotationCount:       len(rotations),
		SignedRotations:     rotations,
		GaloisElements:      galois,
		OrderedPairSHA256:   hex.EncodeToString(pairsDigest[:]),
		PerformanceMeasured: false,
	}
	encoded, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	encoded = append(encoded, '\n')
	if err := os.WriteFile(output, encoded, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("P2GN16_FPGA_ROTATION_INVENTORY_PASS count=%d pair_sha256=%s output=%s; no FHE, FPGA, or timing executed", len(rotations), record.OrderedPairSHA256, output)
}
