package mkfcpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

const (
	fullResNet20CalibrationSchema = "mkf-resnet20-degree59-training-calibration-v1"
	fullResNet20TrainingPartition = "official CIFAR-10 training batches 1-5"
	fullResNet20ActivationDegree  = 59
	fullResNet20Parties           = 2
	fullResNet20FunctionalGate    = 1e-3

	fullResNet20FixtureMetadataSHA256 = "87fb314b4d886c6c99f20cfb52e7a80c37aacba624e13f373a7a2ef409fe8849"
	fullResNet20FixtureRecordSHA256   = "35ce668430e638d9f487715a81e2051fc1b8ca1263461ce72179c29bfdbc1ee8"
	fullResNet20CheckpointFCBiasSHA   = "2e9c7d4748008b155a97f091873cc438f4c705727a22c9d22c8c8db75af2827a"
)

var fullResNet20ActivationSiteNames = [...]string{
	"stem.conv1_bn1",
	"layer1.block1.conv1_bn1", "layer1.block1.residual_sum",
	"layer1.block2.conv1_bn1", "layer1.block2.residual_sum",
	"layer1.block3.conv1_bn1", "layer1.block3.residual_sum",
	"layer2.block1.conv1_bn1", "layer2.block1.residual_sum",
	"layer2.block2.conv1_bn1", "layer2.block2.residual_sum",
	"layer2.block3.conv1_bn1", "layer2.block3.residual_sum",
	"layer3.block1.conv1_bn1", "layer3.block1.residual_sum",
	"layer3.block2.conv1_bn1", "layer3.block2.residual_sum",
	"layer3.block3.conv1_bn1", "layer3.block3.residual_sum",
}

var (
	fullResNet20CheckpointMean = [...]float64{0.4914, 0.4822, 0.4465}
	fullResNet20CheckpointStd  = [...]float64{0.2023, 0.1994, 0.2010}
)

type fullResNet20CalibrationEnvelope struct {
	Schema  string `json:"schema"`
	Status  string `json:"status"`
	Dataset struct {
		Partition          string `json:"partition"`
		DataAndLabelSHA256 string `json:"data_and_label_sha256"`
	} `json:"dataset"`
	InputPreprocessing struct {
		Mode       string    `json:"mode"`
		Mean       []float64 `json:"mean"`
		Std        []float64 `json:"std"`
		ExecutedBy string    `json:"executed_by"`
	} `json:"input_preprocessing"`
	CalibrationAlgorithm struct {
		SafetyUtilization      float64 `json:"safety_utilization"`
		TestLabelsOrImagesUsed *bool   `json:"test_labels_or_test_images_used"`
	} `json:"calibration_algorithm"`
	Activation struct {
		Degree     int       `json:"degree"`
		SiteNames  []string  `json:"site_names"`
		SiteScales []float64 `json:"site_scales"`
	} `json:"activation"`
}

type fullResNet20ScaleSet struct {
	SourcePath                 string
	SourceSHA256               string
	TrainingDataAndLabelSHA256 string
	SafetyUtilization          float64
	Sites                      [len(fullResNet20ActivationSiteNames)]float64
}

func parseFullResNet20Calibration(data []byte, sourcePath string) (fullResNet20ScaleSet, error) {
	var envelope fullResNet20CalibrationEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		return fullResNet20ScaleSet{}, fmt.Errorf("decode activation-scale calibration: %w", err)
	}
	if envelope.Schema != fullResNet20CalibrationSchema {
		return fullResNet20ScaleSet{}, fmt.Errorf("unsupported activation-scale schema %q", envelope.Schema)
	}
	if envelope.Status != "PASS" {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation-scale calibration status is %q, not PASS", envelope.Status)
	}
	if envelope.Activation.Degree != fullResNet20ActivationDegree {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation degree is %d, want %d", envelope.Activation.Degree, fullResNet20ActivationDegree)
	}
	wantNames := fullResNet20ActivationSiteNames[:]
	if !reflect.DeepEqual(envelope.Activation.SiteNames, wantNames) {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation site names/order do not match the frozen A01--A19 contract")
	}
	if len(envelope.Activation.SiteScales) != len(wantNames) {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation-scale artifact has %d scales, want %d", len(envelope.Activation.SiteScales), len(wantNames))
	}
	if envelope.Dataset.Partition != fullResNet20TrainingPartition {
		return fullResNet20ScaleSet{}, fmt.Errorf("calibration partition is %q, want %q", envelope.Dataset.Partition, fullResNet20TrainingPartition)
	}
	if !validSHA256(envelope.Dataset.DataAndLabelSHA256) {
		return fullResNet20ScaleSet{}, fmt.Errorf("invalid training data-and-label SHA-256 %q", envelope.Dataset.DataAndLabelSHA256)
	}
	if envelope.CalibrationAlgorithm.TestLabelsOrImagesUsed == nil || *envelope.CalibrationAlgorithm.TestLabelsOrImagesUsed {
		return fullResNet20ScaleSet{}, fmt.Errorf("calibration must explicitly state that no test labels or images were used")
	}
	if !finitePositive(envelope.CalibrationAlgorithm.SafetyUtilization) || envelope.CalibrationAlgorithm.SafetyUtilization > 1 {
		return fullResNet20ScaleSet{}, fmt.Errorf("invalid safety utilization %g", envelope.CalibrationAlgorithm.SafetyUtilization)
	}
	if envelope.InputPreprocessing.Mode != "checkpoint_normalized" || envelope.InputPreprocessing.ExecutedBy != "data owner before CKKS encoding/encryption" {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation calibration does not use data-owner checkpoint normalization")
	}
	if !equalFloatSlices(envelope.InputPreprocessing.Mean, fullResNet20CheckpointMean[:]) || !equalFloatSlices(envelope.InputPreprocessing.Std, fullResNet20CheckpointStd[:]) {
		return fullResNet20ScaleSet{}, fmt.Errorf("activation calibration mean/std do not match the frozen checkpoint")
	}

	result := fullResNet20ScaleSet{
		SourcePath:                 sourcePath,
		SourceSHA256:               fmt.Sprintf("%x", sha256.Sum256(data)),
		TrainingDataAndLabelSHA256: envelope.Dataset.DataAndLabelSHA256,
		SafetyUtilization:          envelope.CalibrationAlgorithm.SafetyUtilization,
	}
	for index, scale := range envelope.Activation.SiteScales {
		if !finitePositive(scale) {
			return fullResNet20ScaleSet{}, fmt.Errorf("activation site A%02d (%s) has invalid scale %g", index+1, wantNames[index], scale)
		}
		result.Sites[index] = scale
	}
	return result, nil
}

func loadFullResNet20Calibration(path string) (fullResNet20ScaleSet, error) {
	if path == "" {
		return fullResNet20ScaleSet{}, fmt.Errorf("a frozen PASS activation-scale JSON is required; legacy embedded scales are rejected")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fullResNet20ScaleSet{}, fmt.Errorf("read activation-scale calibration %s: %w", path, err)
	}
	return parseFullResNet20Calibration(data, path)
}

func validSHA256(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func equalFloatSlices(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range want {
		if math.Abs(got[index]-want[index]) > 1e-15 {
			return false
		}
	}
	return true
}

type fullResNet20FixtureMetadata struct {
	ImageID string `json:"image_id"`
	Label   struct {
		ID int `json:"label_id"`
	} `json:"-"`
	LabelID        int `json:"label_id"`
	CombinedRecord struct {
		Bytes  int    `json:"bytes"`
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
	} `json:"combined_record"`
}

type fullResNet20Fixture struct {
	ImageID       string
	Label         byte
	NormalizedRGB [][][]float64
	RecordSHA256  string
}

func loadFullResNet20OfficialFixture(directory string) (fullResNet20Fixture, error) {
	metadataPath := filepath.Join(directory, "cifar10-test-00000.json")
	metadataData, err := os.ReadFile(metadataPath)
	if err != nil {
		return fullResNet20Fixture{}, fmt.Errorf("read official CIFAR fixture metadata: %w", err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(metadataData)); digest != fullResNet20FixtureMetadataSHA256 {
		return fullResNet20Fixture{}, fmt.Errorf("official CIFAR fixture metadata SHA-256 mismatch: %s", digest)
	}
	var metadata struct {
		ImageID        string `json:"image_id"`
		LabelID        int    `json:"label_id"`
		CombinedRecord struct {
			Bytes  int    `json:"bytes"`
			File   string `json:"file"`
			SHA256 string `json:"sha256"`
		} `json:"combined_record"`
	}
	if err := json.Unmarshal(metadataData, &metadata); err != nil {
		return fullResNet20Fixture{}, fmt.Errorf("decode official CIFAR fixture metadata: %w", err)
	}
	if metadata.ImageID != "cifar10-test-00000" || metadata.LabelID != 3 {
		return fullResNet20Fixture{}, fmt.Errorf("unexpected official fixture identity %q label=%d", metadata.ImageID, metadata.LabelID)
	}
	if metadata.CombinedRecord.Bytes != 3073 || filepath.Base(metadata.CombinedRecord.File) != metadata.CombinedRecord.File {
		return fullResNet20Fixture{}, fmt.Errorf("invalid combined fixture record declaration")
	}
	if metadata.CombinedRecord.SHA256 != fullResNet20FixtureRecordSHA256 {
		return fullResNet20Fixture{}, fmt.Errorf("combined fixture record identity mismatch")
	}
	record, err := os.ReadFile(filepath.Join(directory, metadata.CombinedRecord.File))
	if err != nil {
		return fullResNet20Fixture{}, fmt.Errorf("read combined official CIFAR fixture: %w", err)
	}
	if len(record) != metadata.CombinedRecord.Bytes {
		return fullResNet20Fixture{}, fmt.Errorf("combined fixture has %d bytes, want %d", len(record), metadata.CombinedRecord.Bytes)
	}
	digest := fmt.Sprintf("%x", sha256.Sum256(record))
	if digest != metadata.CombinedRecord.SHA256 || int(record[0]) != metadata.LabelID {
		return fullResNet20Fixture{}, fmt.Errorf("combined fixture payload identity or label mismatch")
	}
	return fullResNet20Fixture{
		ImageID:       metadata.ImageID,
		Label:         record[0],
		NormalizedRGB: normalizeFullResNet20CIFAR(record[1:]),
		RecordSHA256:  digest,
	}, nil
}

func normalizeFullResNet20CIFAR(planarRGB []byte) [][][]float64 {
	if len(planarRGB) != 3*32*32 {
		return nil
	}
	image := make([][][]float64, 3)
	for channel := 0; channel < 3; channel++ {
		image[channel] = make([][]float64, 32)
		for y := 0; y < 32; y++ {
			image[channel][y] = make([]float64, 32)
			for x := 0; x < 32; x++ {
				raw := float64(planarRGB[channel*1024+y*32+x]) / 255.0
				image[channel][y][x] = (raw - fullResNet20CheckpointMean[channel]) / fullResNet20CheckpointStd[channel]
			}
		}
	}
	return image
}

type fullResNet20PhaseKind string

const (
	fullResNet20InputPhase      fullResNet20PhaseKind = "input"
	fullResNet20StemPhase       fullResNet20PhaseKind = "stem"
	fullResNet20RegularPhase    fullResNet20PhaseKind = "regular-residual"
	fullResNet20TransitionPhase fullResNet20PhaseKind = "stride2-transition-residual"
	fullResNet20TailPhase       fullResNet20PhaseKind = "checkpoint-faithful-gap-fc-bias"
)

type fullResNet20Counts struct {
	DataInputCiphertexts       int `json:"data_input_ciphertexts"`
	ModelWeightCiphertexts     int `json:"model_weight_ciphertexts"`
	ModelBiasCiphertexts       int `json:"model_bias_ciphertexts"`
	CiphertextProducts         int `json:"ciphertext_products"`
	RotationCalls              int `json:"rotation_calls"`
	CollectiveRefreshes        int `json:"collective_refreshes"`
	Degree59Activations        int `json:"degree59_activations"`
	PublicMaskProducts         int `json:"public_mask_products"`
	PackingOrTailRescaleLevels int `json:"packing_or_tail_rescale_levels"`
}

func (counts *fullResNet20Counts) add(other fullResNet20Counts) {
	counts.DataInputCiphertexts += other.DataInputCiphertexts
	counts.ModelWeightCiphertexts += other.ModelWeightCiphertexts
	counts.ModelBiasCiphertexts += other.ModelBiasCiphertexts
	counts.CiphertextProducts += other.CiphertextProducts
	counts.RotationCalls += other.RotationCalls
	counts.CollectiveRefreshes += other.CollectiveRefreshes
	counts.Degree59Activations += other.Degree59Activations
	counts.PublicMaskProducts += other.PublicMaskProducts
	counts.PackingOrTailRescaleLevels += other.PackingOrTailRescaleLevels
}

type fullResNet20CommunicationCounts struct {
	OwnerIngressCiphertexts int
	CKSIngressShareMessages int
	CKSIngressBytes         uint64
	RefreshRounds           int
	RefreshShareMessages    int
	RefreshShareBytes       uint64
	EvaluationKeyBytes      uint64
}

func (counts fullResNet20Counts) communication(parties int) fullResNet20CommunicationCounts {
	ingress := counts.DataInputCiphertexts + counts.ModelWeightCiphertexts + counts.ModelBiasCiphertexts
	return fullResNet20CommunicationCounts{
		OwnerIngressCiphertexts: ingress,
		CKSIngressShareMessages: ingress * parties,
		RefreshRounds:           counts.CollectiveRefreshes,
		RefreshShareMessages:    counts.CollectiveRefreshes * parties,
	}
}

type fullResNet20ActivationSite struct {
	ID    string
	Name  string
	Scale float64
}

type fullResNet20Phase struct {
	Name               string
	Kind               fullResNet20PhaseKind
	Layer              int
	Block              int
	Channels           int
	Width              int
	Conv1Prefix        string
	SkipPrefix         string
	Conv2Prefix        string
	ActivationSites    []fullResNet20ActivationSite
	RequiredBoundaries []string
	Expected           fullResNet20Counts
}

type fullResNet20Plan struct {
	ScaleSource   fullResNet20ScaleSet
	Phases        []fullResNet20Phase
	Expected      fullResNet20Counts
	Communication fullResNet20CommunicationCounts
}

func fullResNet20Site(scales fullResNet20ScaleSet, index int) fullResNet20ActivationSite {
	return fullResNet20ActivationSite{ID: fmt.Sprintf("A%02d", index+1), Name: fullResNet20ActivationSiteNames[index], Scale: scales.Sites[index]}
}

func fullResNet20RegularBoundaries(name string) []string {
	return []string{name + ".conv1_bn1", name + ".refresh1", name + ".activation1", name + ".conv2_bn2", name + ".residual_sum", name + ".refresh2", name + ".activation2"}
}

func fullResNet20TransitionBoundaries(name string) []string {
	return []string{name + ".pre_transition_refresh", name + ".stride_main_half0", name + ".stride_main_half1", name + ".stride_skip_half0", name + ".stride_skip_half1", name + ".packed_main", name + ".packed_skip", name + ".refresh1", name + ".activation1", name + ".conv2_bn2", name + ".residual_sum", name + ".refresh2", name + ".activation2"}
}

func buildFullResNet20Plan(scales fullResNet20ScaleSet) (fullResNet20Plan, error) {
	for index, scale := range scales.Sites {
		if !finitePositive(scale) {
			return fullResNet20Plan{}, fmt.Errorf("cannot build plan: A%02d has invalid scale %g", index+1, scale)
		}
	}
	plan := fullResNet20Plan{ScaleSource: scales}
	plan.Phases = append(plan.Phases,
		fullResNet20Phase{Name: "input", Kind: fullResNet20InputPhase, RequiredBoundaries: []string{"input.normalized", "input.encrypted"}, Expected: fullResNet20Counts{DataInputCiphertexts: 3}},
		fullResNet20Phase{Name: "stem", Kind: fullResNet20StemPhase, Channels: 16, Width: 32, Conv1Prefix: "conv1bn1", ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 0)}, RequiredBoundaries: []string{"stem.conv1_bn1", "stem.refresh", "stem.activation"}, Expected: fullResNet20Counts{ModelWeightCiphertexts: 27, ModelBiasCiphertexts: 1, CiphertextProducts: 27, RotationCalls: 24, CollectiveRefreshes: 1, Degree59Activations: 1}},
	)

	regularPrefixes := [][2]string{{"layer1-conv1bn1", "layer1-conv2bn2"}, {"layer2-conv1bn1", "layer2-conv2bn2"}, {"layer3-conv1bn1", "layer3-conv2bn2"}}
	for block, prefixes := range regularPrefixes {
		name := fmt.Sprintf("layer1.block%d", block+1)
		plan.Phases = append(plan.Phases, fullResNet20Phase{Name: name, Kind: fullResNet20RegularPhase, Layer: 1, Block: block + 1, Channels: 16, Width: 32, Conv1Prefix: prefixes[0], Conv2Prefix: prefixes[1], ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 1+2*block), fullResNet20Site(scales, 2+2*block)}, RequiredBoundaries: fullResNet20RegularBoundaries(name), Expected: fullResNet20Counts{ModelWeightCiphertexts: 288, ModelBiasCiphertexts: 2, CiphertextProducts: 288, RotationCalls: 286, CollectiveRefreshes: 2, Degree59Activations: 2}})
	}

	plan.Phases = append(plan.Phases, fullResNet20Phase{Name: "layer2.block1", Kind: fullResNet20TransitionPhase, Layer: 2, Block: 1, Channels: 32, Width: 16, Conv1Prefix: "layer4-conv1bn1", SkipPrefix: "layer4dx-conv1bn1", Conv2Prefix: "layer4-conv2bn2", ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 7), fullResNet20Site(scales, 8)}, RequiredBoundaries: fullResNet20TransitionBoundaries("layer2.block1"), Expected: fullResNet20Counts{ModelWeightCiphertexts: 608, ModelBiasCiphertexts: 5, CiphertextProducts: 608, RotationCalls: 743, CollectiveRefreshes: 3, Degree59Activations: 2, PublicMaskProducts: 140, PackingOrTailRescaleLevels: 3}})
	for block, prefixes := range [][2]string{{"layer5-conv1bn1", "layer5-conv2bn2"}, {"layer6-conv1bn1", "layer6-conv2bn2"}} {
		name := fmt.Sprintf("layer2.block%d", block+2)
		plan.Phases = append(plan.Phases, fullResNet20Phase{Name: name, Kind: fullResNet20RegularPhase, Layer: 2, Block: block + 2, Channels: 32, Width: 16, Conv1Prefix: prefixes[0], Conv2Prefix: prefixes[1], ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 9+2*block), fullResNet20Site(scales, 10+2*block)}, RequiredBoundaries: fullResNet20RegularBoundaries(name), Expected: fullResNet20Counts{ModelWeightCiphertexts: 576, ModelBiasCiphertexts: 2, CiphertextProducts: 576, RotationCalls: 574, CollectiveRefreshes: 2, Degree59Activations: 2}})
	}

	plan.Phases = append(plan.Phases, fullResNet20Phase{Name: "layer3.block1", Kind: fullResNet20TransitionPhase, Layer: 3, Block: 1, Channels: 64, Width: 8, Conv1Prefix: "layer7-conv1bn1", SkipPrefix: "layer7dx-conv1bn1", Conv2Prefix: "layer7-conv2bn2", ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 13), fullResNet20Site(scales, 14)}, RequiredBoundaries: fullResNet20TransitionBoundaries("layer3.block1"), Expected: fullResNet20Counts{ModelWeightCiphertexts: 1216, ModelBiasCiphertexts: 5, CiphertextProducts: 1216, RotationCalls: 1409, CollectiveRefreshes: 3, Degree59Activations: 2, PublicMaskProducts: 200, PackingOrTailRescaleLevels: 3}})
	for block, prefixes := range [][2]string{{"layer8-conv1bn1", "layer8-conv2bn2"}, {"layer9-conv1bn1", "layer9-conv2bn2"}} {
		name := fmt.Sprintf("layer3.block%d", block+2)
		boundaries := fullResNet20RegularBoundaries(name)
		refreshes := 2
		if block == 1 {
			// The frozen reference refreshes once more after A19 before the
			// GAP+FC tail. Omitting this round undercounts both the interactive
			// protocol and the modulus reset required by the tail.
			boundaries = append(boundaries, name+".final_refresh")
			refreshes = 3
		}
		plan.Phases = append(plan.Phases, fullResNet20Phase{Name: name, Kind: fullResNet20RegularPhase, Layer: 3, Block: block + 2, Channels: 64, Width: 8, Conv1Prefix: prefixes[0], Conv2Prefix: prefixes[1], ActivationSites: []fullResNet20ActivationSite{fullResNet20Site(scales, 15+2*block), fullResNet20Site(scales, 16+2*block)}, RequiredBoundaries: boundaries, Expected: fullResNet20Counts{ModelWeightCiphertexts: 1152, ModelBiasCiphertexts: 2, CiphertextProducts: 1152, RotationCalls: 1150, CollectiveRefreshes: refreshes, Degree59Activations: 2}})
	}
	plan.Phases = append(plan.Phases, fullResNet20Phase{Name: "tail", Kind: fullResNet20TailPhase, Channels: 64, Width: 8, Conv1Prefix: "fc", RequiredBoundaries: []string{"tail.gap", "tail.fc", "tail.logits"}, Expected: fullResNet20Counts{ModelWeightCiphertexts: 1, ModelBiasCiphertexts: 1, CiphertextProducts: 1, RotationCalls: 17, PublicMaskProducts: 1, PackingOrTailRescaleLevels: 2}})

	seen := make([]string, 0, len(fullResNet20ActivationSiteNames))
	for _, phase := range plan.Phases {
		plan.Expected.add(phase.Expected)
		for _, site := range phase.ActivationSites {
			seen = append(seen, site.Name)
		}
	}
	if !reflect.DeepEqual(seen, fullResNet20ActivationSiteNames[:]) {
		return fullResNet20Plan{}, fmt.Errorf("orchestration consumed activation sites in the wrong order: %v", seen)
	}
	plan.Communication = plan.Expected.communication(fullResNet20Parties)
	return plan, nil
}

type fullResNet20FrozenInventory struct {
	MaskOrFCWeightFiles int
	FoldedBiasFiles     int
	CheckpointBiasFiles int
}

func auditFullResNet20FrozenInventory(plan fullResNet20Plan, weightsDirectory, checkpointBiasPath string) (fullResNet20FrozenInventory, error) {
	if weightsDirectory == "" || checkpointBiasPath == "" {
		return fullResNet20FrozenInventory{}, fmt.Errorf("weights directory and checkpoint FC-bias path are required")
	}
	var inventory fullResNet20FrozenInventory
	require := func(name string, bias bool) error {
		path := filepath.Join(weightsDirectory, name)
		info, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("missing frozen file %s: %w", path, err)
		}
		if !info.Mode().IsRegular() || info.Size() == 0 {
			return fmt.Errorf("frozen file %s is empty or not regular", path)
		}
		if bias {
			inventory.FoldedBiasFiles++
		} else {
			inventory.MaskOrFCWeightFiles++
		}
		return nil
	}
	requireConv := func(prefix string, channels, kernels int, biasNames ...string) error {
		for channel := 0; channel < channels; channel++ {
			for kernel := 1; kernel <= kernels; kernel++ {
				if err := require(fmt.Sprintf("%s-ch%d-k%d.bin", prefix, channel, kernel), false); err != nil {
					return err
				}
			}
		}
		for _, name := range biasNames {
			if err := require(name, true); err != nil {
				return err
			}
		}
		return nil
	}
	if err := requireConv("conv1bn1", 16, 9, "conv1bn1-bias.bin"); err != nil {
		return inventory, err
	}
	for _, phase := range plan.Phases {
		switch phase.Kind {
		case fullResNet20RegularPhase:
			if err := requireConv(phase.Conv1Prefix, phase.Channels, 9, phase.Conv1Prefix+"-bias.bin"); err != nil {
				return inventory, err
			}
			if err := requireConv(phase.Conv2Prefix, phase.Channels, 9, phase.Conv2Prefix+"-bias.bin"); err != nil {
				return inventory, err
			}
		case fullResNet20TransitionPhase:
			if err := requireConv(phase.Conv1Prefix, phase.Channels, 9, phase.Conv1Prefix+"-bias1.bin", phase.Conv1Prefix+"-bias2.bin"); err != nil {
				return inventory, err
			}
			if err := requireConv(phase.SkipPrefix, phase.Channels, 1, phase.SkipPrefix+"-bias1.bin", phase.SkipPrefix+"-bias2.bin"); err != nil {
				return inventory, err
			}
			if err := requireConv(phase.Conv2Prefix, phase.Channels, 9, phase.Conv2Prefix+"-bias.bin"); err != nil {
				return inventory, err
			}
		}
	}
	if err := require("fc.bin", false); err != nil {
		return inventory, err
	}
	biasData, err := os.ReadFile(checkpointBiasPath)
	if err != nil {
		return inventory, fmt.Errorf("read checkpoint FC bias: %w", err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(biasData)); digest != fullResNet20CheckpointFCBiasSHA {
		return inventory, fmt.Errorf("checkpoint FC-bias SHA-256 mismatch: %s", digest)
	}
	inventory.CheckpointBiasFiles = 1
	if inventory != (fullResNet20FrozenInventory{MaskOrFCWeightFiles: 6289, FoldedBiasFiles: 25, CheckpointBiasFiles: 1}) {
		return inventory, fmt.Errorf("unexpected frozen inventory totals: %+v", inventory)
	}
	return inventory, nil
}

type fullResNet20RunMode string

const (
	fullResNet20FunctionalMode  fullResNet20RunMode = "functional"
	fullResNet20PerformanceMode fullResNet20RunMode = "performance"
)

type fullResNet20Boundary struct {
	Name          string        `json:"name"`
	Level         int           `json:"level"`
	CheckedSlots  int           `json:"checked_slots"`
	OracleChecked bool          `json:"oracle_checked"`
	MaxError      float64       `json:"max_error"`
	RMSE          float64       `json:"rmse"`
	Timed         bool          `json:"timed"`
	Cycles        uint64        `json:"cycles"`
	Elapsed       time.Duration `json:"elapsed_nanoseconds"`
}

type fullResNet20PhaseResult struct {
	Name          string
	Counts        fullResNet20Counts
	Communication fullResNet20CommunicationCounts
	Boundaries    []fullResNet20Boundary
}

type fullResNet20Request struct {
	Parameter          string
	Mode               fullResNet20RunMode
	Plan               fullResNet20Plan
	Fixture            fullResNet20Fixture
	WeightsDirectory   string
	CheckpointBiasPath string
	// PhaseObserver is evidence plumbing only. It runs after the phase result
	// has passed the frozen functional/performance contract and must not alter
	// the ciphertext dataflow.
	PhaseObserver func(fullResNet20PhaseResult)
}

const fullResNet20EvidenceSchema = "mkf-resnet20-full-network-evidence-v1"

type fullResNet20LogicalCommunicationEvidence struct {
	OwnerIngressCiphertexts int  `json:"owner_ingress_ciphertexts"`
	CKSIngressShareMessages int  `json:"cks_ingress_share_messages"`
	RefreshRounds           int  `json:"refresh_rounds"`
	RefreshShareMessages    int  `json:"refresh_share_messages"`
	ByteCountsMeasured      bool `json:"byte_counts_measured"`
}

func fullResNet20LogicalCommunication(counts fullResNet20CommunicationCounts) fullResNet20LogicalCommunicationEvidence {
	return fullResNet20LogicalCommunicationEvidence{
		OwnerIngressCiphertexts: counts.OwnerIngressCiphertexts,
		CKSIngressShareMessages: counts.CKSIngressShareMessages,
		RefreshRounds:           counts.RefreshRounds,
		RefreshShareMessages:    counts.RefreshShareMessages,
		// The current functional harness counts logical messages but does not
		// serialize them on a measured transport boundary.
		ByteCountsMeasured: false,
	}
}

type fullResNet20RunStartEvidence struct {
	Schema         string              `json:"schema"`
	RecordType     string              `json:"record_type"`
	Parameter      string              `json:"parameter"`
	Mode           fullResNet20RunMode `json:"mode"`
	FunctionalGate float64             `json:"functional_gate"`
	Fixture        struct {
		ImageID      string `json:"image_id"`
		Label        int    `json:"label"`
		RecordSHA256 string `json:"record_sha256"`
	} `json:"fixture"`
	Calibration struct {
		ScaleJSONSHA256           string  `json:"scale_json_sha256"`
		TrainingDataLabelSHA256   string  `json:"training_data_and_label_sha256"`
		SafetyUtilization         float64 `json:"safety_utilization"`
		ActivationSites           int     `json:"activation_sites"`
		CalibrationIsTrainingOnly bool    `json:"calibration_is_training_only"`
	} `json:"calibration"`
	ExpectedCounts        fullResNet20Counts                       `json:"expected_counts"`
	ExpectedCommunication fullResNet20LogicalCommunicationEvidence `json:"expected_communication"`
	PhaseNames            []string                                 `json:"phase_names"`
}

type fullResNet20PhaseEvidence struct {
	Schema        string                                   `json:"schema"`
	RecordType    string                                   `json:"record_type"`
	Parameter     string                                   `json:"parameter"`
	Mode          fullResNet20RunMode                      `json:"mode"`
	Phase         string                                   `json:"phase"`
	Counts        fullResNet20Counts                       `json:"counts"`
	Communication fullResNet20LogicalCommunicationEvidence `json:"communication"`
	Boundaries    []fullResNet20Boundary                   `json:"boundaries"`
}

type fullResNet20ClassificationEvidence struct {
	GroundTruthLabel int       `json:"ground_truth_label"`
	EncryptedTop1    int       `json:"encrypted_top1"`
	EncryptedMargin  float64   `json:"encrypted_margin"`
	EncryptedCorrect bool      `json:"encrypted_correct"`
	ReferenceTop1    int       `json:"reference_polynomial_top1"`
	ReferenceMargin  float64   `json:"reference_polynomial_margin"`
	ReferenceCorrect bool      `json:"reference_polynomial_correct"`
	EncryptedLogits  []float64 `json:"encrypted_logits"`
	ReferenceLogits  []float64 `json:"reference_polynomial_logits"`
	MaxLogitError    float64   `json:"max_logit_error"`
}

type fullResNet20RunSummaryEvidence struct {
	Schema               string              `json:"schema"`
	RecordType           string              `json:"record_type"`
	Status               string              `json:"status"`
	Parameter            string              `json:"parameter"`
	Mode                 fullResNet20RunMode `json:"mode"`
	PhaseCount           int                 `json:"phase_count"`
	BoundaryCount        int                 `json:"boundary_count"`
	MaximumBoundaryError struct {
		Name  string  `json:"name"`
		Value float64 `json:"value"`
	} `json:"maximum_boundary_error"`
	FinalLevel                               int                                      `json:"final_level"`
	Counts                                   fullResNet20Counts                       `json:"counts"`
	Communication                            fullResNet20LogicalCommunicationEvidence `json:"communication"`
	Classification                           *fullResNet20ClassificationEvidence      `json:"classification,omitempty"`
	FunctionalWallClockSecondsNotPerformance float64                                  `json:"functional_wall_clock_seconds_not_performance"`
	PerformanceMeasured                      bool                                     `json:"performance_measured"`
}

type fullResNet20Executor interface {
	ExecutePhase(fullResNet20Phase) (fullResNet20PhaseResult, error)
}

// The factory is intentionally nil until the already-running Stage3 chain and
// 16384-slot checkpoint-faithful tail adapters pass their own functional gates.
// A later integration file can wire the verified adapters without changing the
// plan, scale loader, fixture semantics, or functional/performance separation.
var fullResNet20ExecutorFactory func(fullResNet20Request) (fullResNet20Executor, error)

func validateFullResNet20PhaseResult(mode fullResNet20RunMode, phase fullResNet20Phase, result fullResNet20PhaseResult) error {
	if result.Name != phase.Name || result.Counts != phase.Expected {
		return fmt.Errorf("phase %s result identity or operation counts do not match the frozen plan", phase.Name)
	}
	if len(result.Boundaries) != len(phase.RequiredBoundaries) {
		return fmt.Errorf("phase %s returned %d boundaries, want %d", phase.Name, len(result.Boundaries), len(phase.RequiredBoundaries))
	}
	for index, boundary := range result.Boundaries {
		if boundary.Name != phase.RequiredBoundaries[index] {
			return fmt.Errorf("phase %s boundary %d is %q, want %q", phase.Name, index, boundary.Name, phase.RequiredBoundaries[index])
		}
		switch mode {
		case fullResNet20FunctionalMode:
			if !boundary.OracleChecked || boundary.CheckedSlots <= 0 || boundary.Timed || boundary.Cycles != 0 || boundary.Elapsed != 0 {
				return fmt.Errorf("functional boundary %s must be oracle-checked and untimed", boundary.Name)
			}
			if math.IsNaN(boundary.MaxError) || math.IsInf(boundary.MaxError, 0) || boundary.MaxError >= fullResNet20FunctionalGate {
				return fmt.Errorf("functional boundary %s failed: maxError=%g", boundary.Name, boundary.MaxError)
			}
		case fullResNet20PerformanceMode:
			if boundary.OracleChecked || boundary.CheckedSlots != 0 || !boundary.Timed || (boundary.Cycles == 0 && boundary.Elapsed <= 0) {
				return fmt.Errorf("performance boundary %s must be timed without an in-boundary plaintext oracle", boundary.Name)
			}
		default:
			return fmt.Errorf("unsupported full-network run mode %q", mode)
		}
	}
	return nil
}

func executeFullResNet20(request fullResNet20Request, executor fullResNet20Executor) ([]fullResNet20PhaseResult, error) {
	results := make([]fullResNet20PhaseResult, 0, len(request.Plan.Phases))
	var counts fullResNet20Counts
	var communication fullResNet20CommunicationCounts
	for _, phase := range request.Plan.Phases {
		result, err := executor.ExecutePhase(phase)
		if err != nil {
			return nil, fmt.Errorf("execute %s: %w", phase.Name, err)
		}
		if err := validateFullResNet20PhaseResult(request.Mode, phase, result); err != nil {
			return nil, err
		}
		counts.add(result.Counts)
		communication.OwnerIngressCiphertexts += result.Communication.OwnerIngressCiphertexts
		communication.CKSIngressShareMessages += result.Communication.CKSIngressShareMessages
		communication.CKSIngressBytes += result.Communication.CKSIngressBytes
		communication.RefreshRounds += result.Communication.RefreshRounds
		communication.RefreshShareMessages += result.Communication.RefreshShareMessages
		communication.RefreshShareBytes += result.Communication.RefreshShareBytes
		communication.EvaluationKeyBytes += result.Communication.EvaluationKeyBytes
		results = append(results, result)
		if request.PhaseObserver != nil {
			request.PhaseObserver(result)
		}
	}
	if counts != request.Plan.Expected {
		return nil, fmt.Errorf("full-network operation totals do not match the plan: got=%+v want=%+v", counts, request.Plan.Expected)
	}
	if communication.OwnerIngressCiphertexts != request.Plan.Communication.OwnerIngressCiphertexts || communication.CKSIngressShareMessages != request.Plan.Communication.CKSIngressShareMessages || communication.RefreshRounds != request.Plan.Communication.RefreshRounds || communication.RefreshShareMessages != request.Plan.Communication.RefreshShareMessages {
		return nil, fmt.Errorf("full-network logical communication totals do not match the plan")
	}
	return results, nil
}

func logFullResNet20EvidenceJSON(t *testing.T, marker string, value interface{}) error {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode %s evidence: %w", marker, err)
	}
	t.Logf("%s=%s", marker, encoded)
	return nil
}

func runEncryptedFullResNet20Gate(t *testing.T, parameter string, mode fullResNet20RunMode, gateEnvironment string) {
	t.Helper()
	if os.Getenv(gateEnvironment) != "1" {
		t.Skipf("set %s=1 only after the Stage3 chain, frozen scale artifact, and full-width tail adapter pass", gateEnvironment)
	}
	if fullResNet20ExecutorFactory == nil {
		t.Skip("full-network executor is intentionally unwired pending the independently verified Stage3-chain and 16384-slot checkpoint-faithful tail adapters")
	}
	scales, err := loadFullResNet20Calibration(os.Getenv("MKF_RESNET20_ACTIVATION_SCALES_JSON"))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := loadFullResNet20OfficialFixture(os.Getenv("MKF_RESNET20_CIFAR_FIXTURE_DIR"))
	if err != nil {
		t.Fatal(err)
	}
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
	if weightsDirectory == "" || biasPath == "" {
		t.Fatal("MKF_RESNET20_WEIGHTS_DIR and MKF_RESNET20_FC_BIAS_PATH are required")
	}
	biasData, err := os.ReadFile(biasPath)
	if err != nil {
		t.Fatal(err)
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(biasData)); digest != fullResNet20CheckpointFCBiasSHA {
		t.Fatalf("checkpoint FC-bias SHA-256 mismatch: %s", digest)
	}
	request := fullResNet20Request{Parameter: parameter, Mode: mode, Plan: plan, Fixture: fixture, WeightsDirectory: weightsDirectory, CheckpointBiasPath: biasPath}
	startEvidence := fullResNet20RunStartEvidence{
		Schema:                fullResNet20EvidenceSchema,
		RecordType:            "run_start",
		Parameter:             parameter,
		Mode:                  mode,
		FunctionalGate:        fullResNet20FunctionalGate,
		ExpectedCounts:        plan.Expected,
		ExpectedCommunication: fullResNet20LogicalCommunication(plan.Communication),
	}
	startEvidence.Fixture.ImageID = fixture.ImageID
	startEvidence.Fixture.Label = int(fixture.Label)
	startEvidence.Fixture.RecordSHA256 = fixture.RecordSHA256
	startEvidence.Calibration.ScaleJSONSHA256 = scales.SourceSHA256
	startEvidence.Calibration.TrainingDataLabelSHA256 = scales.TrainingDataAndLabelSHA256
	startEvidence.Calibration.SafetyUtilization = scales.SafetyUtilization
	startEvidence.Calibration.ActivationSites = len(scales.Sites)
	startEvidence.Calibration.CalibrationIsTrainingOnly = true
	for _, phase := range plan.Phases {
		startEvidence.PhaseNames = append(startEvidence.PhaseNames, phase.Name)
	}
	if err := logFullResNet20EvidenceJSON(t, "FULL_RESNET20_RUN_JSON", startEvidence); err != nil {
		t.Fatal(err)
	}
	var phaseEvidenceError error
	request.PhaseObserver = func(result fullResNet20PhaseResult) {
		if phaseEvidenceError != nil {
			return
		}
		phaseEvidenceError = logFullResNet20EvidenceJSON(t, "FULL_RESNET20_PHASE_JSON", fullResNet20PhaseEvidence{
			Schema:        fullResNet20EvidenceSchema,
			RecordType:    "phase",
			Parameter:     parameter,
			Mode:          mode,
			Phase:         result.Name,
			Counts:        result.Counts,
			Communication: fullResNet20LogicalCommunication(result.Communication),
			Boundaries:    result.Boundaries,
		})
	}
	functionalStarted := time.Now()
	executor, err := fullResNet20ExecutorFactory(request)
	if err != nil {
		t.Fatal(err)
	}
	results, err := executeFullResNet20(request, executor)
	if err != nil {
		t.Fatal(err)
	}
	if phaseEvidenceError != nil {
		t.Fatal(phaseEvidenceError)
	}
	summary := fullResNet20RunSummaryEvidence{
		Schema:              fullResNet20EvidenceSchema,
		RecordType:          "run_summary",
		Status:              "PASS",
		Parameter:           parameter,
		Mode:                mode,
		PhaseCount:          len(results),
		Counts:              plan.Expected,
		Communication:       fullResNet20LogicalCommunication(plan.Communication),
		PerformanceMeasured: mode == fullResNet20PerformanceMode,
	}
	if mode == fullResNet20FunctionalMode {
		summary.FunctionalWallClockSecondsNotPerformance = time.Since(functionalStarted).Seconds()
	}
	for _, result := range results {
		for _, boundary := range result.Boundaries {
			summary.BoundaryCount++
			if summary.MaximumBoundaryError.Name == "" || boundary.MaxError > summary.MaximumBoundaryError.Value {
				summary.MaximumBoundaryError.Name = boundary.Name
				summary.MaximumBoundaryError.Value = boundary.MaxError
			}
			summary.FinalLevel = boundary.Level
		}
	}
	if mode == fullResNet20FunctionalMode {
		concrete, ok := executor.(*encryptedFullResNet20Executor)
		if !ok {
			t.Fatal("functional full-network executor does not expose final classification evidence")
		}
		summary.Classification, err = concrete.finalClassificationEvidence(int(fixture.Label))
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := logFullResNet20EvidenceJSON(t, "FULL_RESNET20_SUMMARY_JSON", summary); err != nil {
		t.Fatal(err)
	}
}

func TestFullResNet20OrchestrationContract(t *testing.T) {
	var scales fullResNet20ScaleSet
	for index := range scales.Sites {
		scales.Sites[index] = 0.05 + float64(index+1)/100
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	want := fullResNet20Counts{DataInputCiphertexts: 3, ModelWeightCiphertexts: 6172, ModelBiasCiphertexts: 26, CiphertextProducts: 6172, RotationCalls: 6499, CollectiveRefreshes: 22, Degree59Activations: 19, PublicMaskProducts: 341, PackingOrTailRescaleLevels: 8}
	if plan.Expected != want {
		t.Fatalf("unexpected full-network counts: got=%+v want=%+v", plan.Expected, want)
	}
	wantCommunication := fullResNet20CommunicationCounts{OwnerIngressCiphertexts: 6201, CKSIngressShareMessages: 12402, RefreshRounds: 22, RefreshShareMessages: 44}
	if plan.Communication != wantCommunication {
		t.Fatalf("unexpected communication counts: got=%+v want=%+v", plan.Communication, wantCommunication)
	}
	if len(plan.Phases) != 12 {
		t.Fatalf("got %d orchestration phases, want 12", len(plan.Phases))
	}
}

func TestFullResNet20CalibrationJSONContract(t *testing.T) {
	testPartition := false
	var envelope fullResNet20CalibrationEnvelope
	envelope.Schema, envelope.Status = fullResNet20CalibrationSchema, "PASS"
	envelope.Dataset.Partition = fullResNet20TrainingPartition
	envelope.Dataset.DataAndLabelSHA256 = "0a49508339d8cc746630fbfae9836a0c916e54754909ff73786b95b3b00f59b9"
	envelope.InputPreprocessing.Mode = "checkpoint_normalized"
	envelope.InputPreprocessing.Mean = append([]float64(nil), fullResNet20CheckpointMean[:]...)
	envelope.InputPreprocessing.Std = append([]float64(nil), fullResNet20CheckpointStd[:]...)
	envelope.InputPreprocessing.ExecutedBy = "data owner before CKKS encoding/encryption"
	envelope.CalibrationAlgorithm.SafetyUtilization = 0.9
	envelope.CalibrationAlgorithm.TestLabelsOrImagesUsed = &testPartition
	envelope.Activation.Degree = fullResNet20ActivationDegree
	envelope.Activation.SiteNames = append([]string(nil), fullResNet20ActivationSiteNames[:]...)
	for index := range fullResNet20ActivationSiteNames {
		envelope.Activation.SiteScales = append(envelope.Activation.SiteScales, 0.1+float64(index)/100)
	}
	data, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseFullResNet20Calibration(data, "unit-test.json")
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Sites[18] != envelope.Activation.SiteScales[18] || parsed.SourceSHA256 == "" {
		t.Fatal("calibration scales were not consumed from JSON")
	}
	envelope.Activation.SiteNames[18] = "legacy.wrong.site"
	badData, _ := json.Marshal(envelope)
	if _, err := parseFullResNet20Calibration(badData, "bad.json"); err == nil {
		t.Fatal("site-order mismatch was not rejected")
	}
}

func TestFullResNet20CheckpointNormalizationContract(t *testing.T) {
	raw := make([]byte, 3*32*32)
	raw[0], raw[1024], raw[2048] = 0, 127, 255
	image := normalizeFullResNet20CIFAR(raw)
	if image == nil || len(image) != 3 || len(image[0]) != 32 || len(image[0][0]) != 32 {
		t.Fatal("checkpoint normalization returned an invalid planar shape")
	}
	want := []float64{
		(0 - fullResNet20CheckpointMean[0]) / fullResNet20CheckpointStd[0],
		(float64(127)/255 - fullResNet20CheckpointMean[1]) / fullResNet20CheckpointStd[1],
		(1 - fullResNet20CheckpointMean[2]) / fullResNet20CheckpointStd[2],
	}
	got := []float64{image[0][0][0], image[1][0][0], image[2][0][0]}
	if !equalFloatSlices(got, want) {
		t.Fatalf("data-owner checkpoint normalization mismatch: got=%v want=%v", got, want)
	}
}

func TestFullResNet20ModeSeparationContract(t *testing.T) {
	var scales fullResNet20ScaleSet
	for index := range scales.Sites {
		scales.Sites[index] = 0.2
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	phase := plan.Phases[0]
	functional := fullResNet20PhaseResult{Name: phase.Name, Counts: phase.Expected}
	for _, name := range phase.RequiredBoundaries {
		functional.Boundaries = append(functional.Boundaries, fullResNet20Boundary{Name: name, Level: -1, CheckedSlots: 3072, OracleChecked: true, MaxError: 0})
	}
	if err := validateFullResNet20PhaseResult(fullResNet20FunctionalMode, phase, functional); err != nil {
		t.Fatalf("valid functional result rejected: %v", err)
	}
	if err := validateFullResNet20PhaseResult(fullResNet20PerformanceMode, phase, functional); err == nil {
		t.Fatal("functional result was incorrectly accepted as performance evidence")
	}
	performance := fullResNet20PhaseResult{Name: phase.Name, Counts: phase.Expected}
	for _, name := range phase.RequiredBoundaries {
		performance.Boundaries = append(performance.Boundaries, fullResNet20Boundary{Name: name, Level: -1, Timed: true, Cycles: 1})
	}
	if err := validateFullResNet20PhaseResult(fullResNet20PerformanceMode, phase, performance); err != nil {
		t.Fatalf("valid performance result rejected: %v", err)
	}
	if err := validateFullResNet20PhaseResult(fullResNet20FunctionalMode, phase, performance); err == nil {
		t.Fatal("performance result was incorrectly accepted as functional evidence")
	}
}

func TestFullResNet20FrozenInputsAudit(t *testing.T) {
	scalePath := os.Getenv("MKF_RESNET20_ACTIVATION_SCALES_JSON")
	fixtureDirectory := os.Getenv("MKF_RESNET20_CIFAR_FIXTURE_DIR")
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
	if scalePath == "" || fixtureDirectory == "" || weightsDirectory == "" || biasPath == "" {
		t.Skip("set the four MKF_RESNET20_* frozen-input paths to run the light source/fixture inventory audit")
	}
	scales, err := loadFullResNet20Calibration(scalePath)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := loadFullResNet20OfficialFixture(fixtureDirectory)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := auditFullResNet20FrozenInventory(plan, weightsDirectory, biasPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("full-network frozen-input audit PASS: scaleJSONSHA256=%s trainingDataSHA256=%s safetyUtilization=%g fixture=%s label=%d recordSHA256=%s inventory=%+v; no FHE executed", scales.SourceSHA256, scales.TrainingDataAndLabelSHA256, scales.SafetyUtilization, fixture.ImageID, fixture.Label, fixture.RecordSHA256, inventory)
}

func TestEncryptedRealResNet20FullNetworkP2Functional(t *testing.T) {
	runEncryptedFullResNet20Gate(t, "P2", fullResNet20FunctionalMode, "MKF_RUN_FULL_RESNET20_P2_FUNCTIONAL")
}

func TestEncryptedRealResNet20FullNetworkP3Functional(t *testing.T) {
	runEncryptedFullResNet20Gate(t, "P3IRCandidate", fullResNet20FunctionalMode, "MKF_RUN_FULL_RESNET20_P3_FUNCTIONAL")
}

func TestEncryptedRealResNet20FullNetworkP2Performance(t *testing.T) {
	runEncryptedFullResNet20Gate(t, "P2", fullResNet20PerformanceMode, "MKF_RUN_FULL_RESNET20_P2_PERFORMANCE")
}

func TestEncryptedRealResNet20FullNetworkP3Performance(t *testing.T) {
	runEncryptedFullResNet20Gate(t, "P3IRCandidate", fullResNet20PerformanceMode, "MKF_RUN_FULL_RESNET20_P3_PERFORMANCE")
}
