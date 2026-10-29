//go:build p2gn16_physobs
// +build p2gn16_physobs

package mkfcpu

// This file is compiled only with the p2gn16_physobs tag and a hash-gated Go
// overlay for Lattigo v2.3.0.  It closes the observation gap at the actual
// AddConst and MultByGaussianIntegerAndAdd call boundaries without changing
// the module cache or any production source.  The default test binary cannot
// see or enable these hooks.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

const (
	p2gn16PhysicalObserverSchema                = "mkf-p2gn16-same-run-physical-operand-observer-v1"
	p2gn16PhysicalObserverExportEnv             = "MKF_P2GN16_PHYSICAL_OBSERVER_ROOT"
	p2gn16PhysicalObserverAckEnv                = "MKF_P2GN16_PHYSICAL_OBSERVER_ACK"
	p2gn16PhysicalObserverAckValue              = "I_ACCEPT_ONE_HEAVY_FUNCTIONAL_RUN_NO_PERFORMANCE_CLAIM"
	p2gn16PhysicalObserverOraclePath            = "experiments/resnet20_p3/fpga_scale_audit/p2gn16_lattigo_v23_scale_encoding_oracle.json"
	p2gn16PhysicalObserverOracleSHA             = "7dd2cf831feec964ecf543e0d7d459c2ba42a3dc301a2fb4adc22559b9a8d184"
	p2gn16PhysicalObserverStreamPath            = "experiments/resnet20_p3/fpga_contract/p2gn16_public_operands_margin070_current_source_20260826/p2gn16_resnet20_operation_command_stream.calibrated_margin070.json"
	p2gn16PhysicalObserverStreamSHA             = "d40d0a7ad0814fef4f224ae66bc194574c38b3fda3c6afaa358b29dd3ac865e5"
	p2gn16PhysicalObserverGaussianSHA           = "9d6f955554b31e23f4fc077cfa0fc0ef98bb73e64f524af0a3022fd2b020c030"
	p2gn16PhysicalObserverAddConstFullOracleSHA = "954b885111cdf57330d74bac26171b7623c6813c6bef371b52e1ae39efebabcb"
	p2gn16PhysicalObserverAddConstNonzeroSHA    = "38bf31e8e4d687ece17bed62c658ff7ebb0b56d3ceb020591e5b1dc0d439624b"
)

type p2gn16PhysicalCall struct {
	CallIndex                   int      `json:"call_index"`
	AuthoritySequence           int      `json:"authority_sequence"`
	AuthorityOperation          string   `json:"authority_operation"`
	Kind                        string   `json:"kind"`
	Context                     string   `json:"context"`
	InputLevel                  int      `json:"input_level"`
	OutputLevel                 int      `json:"output_level"`
	InputScale                  float64  `json:"input_scale"`
	OutputScaleBefore           float64  `json:"output_scale_before"`
	OutputScaleAfter            float64  `json:"output_scale_after"`
	InputAliasesOutput          bool     `json:"input_aliases_output"`
	ConstantReal                string   `json:"constant_real"`
	ConstantImag                string   `json:"constant_imag"`
	PositiveStandardRNS         []uint64 `json:"positive_standard_rns"`
	NegativeStandardRNS         []uint64 `json:"negative_standard_rns"`
	PositiveMontgomeryRNS       []uint64 `json:"positive_montgomery_rns,omitempty"`
	NegativeMontgomeryRNS       []uint64 `json:"negative_montgomery_rns,omitempty"`
	LineExactAgainstScaleOracle bool     `json:"line_exact_against_scale_oracle"`
}

type p2gn16PhysicalExpected struct {
	event             p2gn16CommandEvent
	inputLevel        int
	outputLevel       int
	inputScale        float64
	outputScaleBefore float64
	outputScaleAfter  float64
	alias             bool
	real              string
	imag              string
	positiveStandard  []uint64
	negativeStandard  []uint64
	positiveMont      []uint64
	negativeMont      []uint64
}

type p2gn16PhysicalReport struct {
	Schema              string `json:"schema"`
	Status              string `json:"status"`
	Parameter           string `json:"parameter"`
	Construction        string `json:"construction"`
	ObserverMode        string `json:"observer_mode"`
	PerformanceMeasured bool   `json:"performance_measured"`
	SourceBindings      []struct {
		RootScope string `json:"root_scope"`
		Path      string `json:"path"`
		Bytes     int64  `json:"bytes"`
		SHA256    string `json:"sha256"`
	} `json:"source_bindings"`
	Coverage struct {
		Calls                       int `json:"calls"`
		GaussianCalls               int `json:"gaussian_calls"`
		AddConstCalls               int `json:"add_const_calls"`
		AddConstNonzeroCalls        int `json:"add_const_nonzero_calls"`
		AddConstZeroCalls           int `json:"add_const_zero_calls"`
		SemanticZeroAddConstElided  int `json:"semantic_zero_add_const_elided"`
		GaussianOutOfPlaceCalls     int `json:"gaussian_out_of_place_calls"`
		AddConstInPlaceCalls        int `json:"add_const_in_place_calls"`
		LineExactCalls              int `json:"line_exact_calls"`
		FirstMismatchCall           int `json:"first_mismatch_call"`
		FirstMismatchAuthorityEvent int `json:"first_mismatch_authority_event"`
	} `json:"coverage"`
	Transcript struct {
		GaussianSHA256     string `json:"gaussian_sha256"`
		GaussianExpected   string `json:"gaussian_expected_sha256"`
		GaussianExact      bool   `json:"gaussian_exact"`
		AddConstSHA256     string `json:"add_const_sha256"`
		AddConstExpected   string `json:"add_const_expected_sha256"`
		AddConstFullOracle string `json:"add_const_full_304_oracle_sha256"`
		AddConstExact      bool   `json:"add_const_exact"`
	} `json:"transcript"`
	Calls        []p2gn16PhysicalCall `json:"calls"`
	ReleaseState struct {
		ProductionPopulated bool `json:"production_populated"`
		BoardEligible       bool `json:"board_eligible"`
		MeasurementEligible bool `json:"measurement_eligible"`
		PerformanceMeasured bool `json:"performance_measured"`
	} `json:"release_state"`
	Blockers []string `json:"blockers"`
}

var p2gn16PhysicalCapture struct {
	sync.Mutex
	records []ckks.P2GN16PhysicalOperandRecord
}

func p2gn16PhysicalClone(record ckks.P2GN16PhysicalOperandRecord) ckks.P2GN16PhysicalOperandRecord {
	record.PositiveStandardRNS = append([]uint64(nil), record.PositiveStandardRNS...)
	record.NegativeStandardRNS = append([]uint64(nil), record.NegativeStandardRNS...)
	record.PositiveMontgomeryRNS = append([]uint64(nil), record.PositiveMontgomeryRNS...)
	record.NegativeMontgomeryRNS = append([]uint64(nil), record.NegativeMontgomeryRNS...)
	return record
}

func p2gn16PhysicalCaptureRecord(record ckks.P2GN16PhysicalOperandRecord) {
	p2gn16PhysicalCapture.Lock()
	p2gn16PhysicalCapture.records = append(p2gn16PhysicalCapture.records, p2gn16PhysicalClone(record))
	p2gn16PhysicalCapture.Unlock()
}

func p2gn16PhysicalDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func p2gn16PhysicalReadBound(root, relative, expectedSHA string) ([]byte, error) {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(relative)))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", relative, err)
	}
	if got := p2gn16PhysicalDigest(data); got != expectedSHA {
		return nil, fmt.Errorf("source identity drift for %s: %s != %s", relative, got, expectedSHA)
	}
	return data, nil
}

func p2gn16PhysicalRepoRoot() (string, error) {
	current, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if info, statErr := os.Stat(filepath.Join(current, "go.mod")); statErr == nil && !info.IsDir() {
			return current, nil
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", fmt.Errorf("cannot locate module root from %s", current)
		}
		current = parent
	}
}

func p2gn16PhysicalInterleave(positive, negative []uint64) []byte {
	if len(positive) != len(negative) {
		panic("physical residue limbs are unbalanced")
	}
	buffer := make([]byte, 0, len(positive)*16)
	for limb := range positive {
		buffer = p2gn16AppendU64(buffer, positive[limb], negative[limb])
	}
	return buffer
}

func p2gn16PhysicalStandardGaussian(ringQ *ring.Ring, level int, realInteger, imaginaryInteger *big.Int) (positive, negative []uint64) {
	positive = make([]uint64, level+1)
	negative = make([]uint64, level+1)
	for limb := 0; limb <= level; limb++ {
		qi := ringQ.Modulus[limb]
		realResidue := p2gn16InterfaceMod(realInteger, qi)
		imagResidue := p2gn16InterfaceMod(imaginaryInteger, qi)
		imagTwisted := ring.MRed(imagResidue, ringQ.NttPsi[limb][1], qi, ringQ.MredParams[limb])
		positive[limb] = ring.CRed(realResidue+imagTwisted, qi)
		negative[limb] = ring.CRed(realResidue+(qi-imagTwisted), qi)
	}
	return
}

func p2gn16PhysicalSplitPayload(payload []byte) (positive, negative []uint64, err error) {
	if len(payload)%16 != 0 {
		return nil, nil, fmt.Errorf("compact payload bytes=%d is not limb-pair aligned", len(payload))
	}
	limbs := len(payload) / 16
	positive = make([]uint64, limbs)
	negative = make([]uint64, limbs)
	for limb := 0; limb < limbs; limb++ {
		positive[limb] = binary.LittleEndian.Uint64(payload[limb*16 : limb*16+8])
		negative[limb] = binary.LittleEndian.Uint64(payload[limb*16+8 : limb*16+16])
	}
	return
}

func p2gn16PhysicalExpectedCalls(stream p2gn16CommandStream, params ckks.Parameters) ([]p2gn16PhysicalExpected, []p2gn16PhysicalExpected, error) {
	producer := make(map[string]p2gn16CommandEvent, len(stream.Commands))
	lastActivation := map[string]int{}
	for _, event := range stream.Commands {
		producer[event.Output] = event
		if event.ActivationID != "" {
			lastActivation[event.ActivationID] = event.Sequence
		}
	}
	termAccumulator := map[string]string{}
	for _, event := range stream.Commands {
		if event.Opcode != "public_add" || event.Purpose != "degree59_bsgs" || len(event.Inputs) != 2 {
			continue
		}
		for index, input := range event.Inputs {
			if source, ok := producer[input]; ok && source.Opcode == "plaintext_mul" && source.Purpose == "degree59_bsgs" {
				termAccumulator[input] = event.Inputs[1-index]
			}
		}
	}

	states := map[string]p2gn16ScaleReplayValue{}
	ringQ := params.RingQ()
	gaussian := make([]p2gn16PhysicalExpected, 0, 828)
	addConst := make([]p2gn16PhysicalExpected, 0, 304)
	for _, event := range stream.Commands {
		inputs := make([]p2gn16ScaleReplayValue, len(event.Inputs))
		for index, id := range event.Inputs {
			inputs[index] = states[id]
			if event.Opcode != "host_interactive_refresh_import" && !inputs[index].known {
				return nil, nil, fmt.Errorf("event %d consumes unknown replay input %s", event.Sequence, id)
			}
		}
		output := p2gn16ScaleReplayValue{level: event.OutputLevel, known: true}
		switch event.Opcode {
		case "host_boundary_gate", "host_interactive_refresh_export":
			output.known = false
		case "owner_ciphertext_input":
			output.scale = params.Scale() * float64(params.Q()[params.MaxLevel()])
		case "host_interactive_refresh_import":
			output.scale = params.Scale()
		case "fold_to_aggregate", "rotate", "level_drop":
			output.scale = inputs[0].scale
		case "compact_mul_relin":
			output.scale = inputs[0].scale * inputs[1].scale
		case "rescale_drop":
			output.scale = inputs[0].scale
			for drop := 0; drop < event.LevelsDropped; drop++ {
				output.scale /= float64(params.Q()[inputs[0].level-drop])
			}
		case "rescale_noop":
			output.scale = inputs[0].scale
		case "ciphertext_zero_init":
			if strings.HasSuffix(event.Operation, ".bsgs.leaf.zero") {
				output.scale = params.Scale()
			} else if strings.Contains(event.Operation, ".bsgs.leaf") {
				output.scale = params.Scale() * float64(params.Q()[event.OutputLevel])
			} else {
				return nil, nil, fmt.Errorf("event %d has unclassified zero-init", event.Sequence)
			}
		case "plaintext_mul":
			switch event.Purpose {
			case "degree59_map":
				output.scale = inputs[0].scale
			case "residual":
				output.scale = inputs[0].scale * float64(params.Q()[inputs[0].level])
			case "packing":
				level := inputs[0].level
				output.scale = inputs[0].scale * math.Pow(float64(params.Q()[level])*float64(params.Q()[level-1])*float64(params.Q()[level-2]), 1.0/5.0)
			case "tail":
				output.scale = inputs[0].scale * params.Scale()
			case "degree59_bsgs":
				accumulatorID, ok := termAccumulator[event.Output]
				if !ok || !states[accumulatorID].known {
					return nil, nil, fmt.Errorf("event %d lacks a known BSGS accumulator", event.Sequence)
				}
				accumulator := states[accumulatorID]
				output.scale = accumulator.scale
				ratio := accumulator.scale / inputs[0].scale
				realInteger := p2gn16GaussianInteger(*event.ConstantReal, ratio)
				imagInteger := p2gn16GaussianInteger(*event.ConstantImag, ratio)
				standardPositive, standardNegative := p2gn16PhysicalStandardGaussian(ringQ, output.level, realInteger, imagInteger)
				montPositive, montNegative, err := p2gn16PhysicalSplitPayload(p2gn16GaussianCompactPayload(ringQ, output.level, realInteger, imagInteger))
				if err != nil {
					return nil, nil, err
				}
				gaussian = append(gaussian, p2gn16PhysicalExpected{
					event: event, inputLevel: inputs[0].level, outputLevel: output.level,
					inputScale: inputs[0].scale, outputScaleBefore: accumulator.scale, outputScaleAfter: accumulator.scale,
					alias: false, real: realInteger.String(), imag: imagInteger.String(),
					positiveStandard: standardPositive, negativeStandard: standardNegative,
					positiveMont: montPositive, negativeMont: montNegative,
				})
			default:
				return nil, nil, fmt.Errorf("event %d unclassified plaintext purpose %s", event.Sequence, event.Purpose)
			}
		case "public_add_const":
			output.scale = inputs[0].scale
			if *event.ConstantReal == 0 && *event.ConstantImag == 0 {
				break
			}
			payload := p2gn16AddConstCompactPayload(ringQ, output.level, output.scale, complex(*event.ConstantReal, *event.ConstantImag))
			positive, negative, err := p2gn16PhysicalSplitPayload(payload)
			if err != nil {
				return nil, nil, err
			}
			addConst = append(addConst, p2gn16PhysicalExpected{
				event: event, inputLevel: inputs[0].level, outputLevel: output.level,
				inputScale: inputs[0].scale, outputScaleBefore: inputs[0].scale, outputScaleAfter: output.scale,
				alias: true, real: strconv.FormatFloat(*event.ConstantReal, 'g', 17, 64), imag: strconv.FormatFloat(*event.ConstantImag, 'g', 17, 64),
				positiveStandard: positive, negativeStandard: negative,
			})
		case "public_add", "public_sub":
			output.scale = math.Max(inputs[0].scale, inputs[1].scale)
		default:
			return nil, nil, fmt.Errorf("event %d unsupported opcode %s", event.Sequence, event.Opcode)
		}
		if event.ActivationID != "" && lastActivation[event.ActivationID] == event.Sequence {
			output.scale = params.Scale()
		}
		states[event.Output] = output
	}
	if len(gaussian) != 828 || len(addConst) != 285 {
		return nil, nil, fmt.Errorf("expected-call coverage drift Gaussian/AddConst=%d/%d", len(gaussian), len(addConst))
	}
	return gaussian, addConst, nil
}

func p2gn16PhysicalSameBits(left, right float64) bool {
	return math.Float64bits(left) == math.Float64bits(right)
}

func p2gn16PhysicalCompareRecord(record ckks.P2GN16PhysicalOperandRecord, expected p2gn16PhysicalExpected) error {
	if record.InputLevel != expected.inputLevel || record.OutputLevel != expected.outputLevel {
		return fmt.Errorf("level actual=%d/%d expected=%d/%d", record.InputLevel, record.OutputLevel, expected.inputLevel, expected.outputLevel)
	}
	if !p2gn16PhysicalSameBits(record.InputScale, expected.inputScale) || !p2gn16PhysicalSameBits(record.OutputScaleBefore, expected.outputScaleBefore) || !p2gn16PhysicalSameBits(record.OutputScaleAfter, expected.outputScaleAfter) {
		return fmt.Errorf("scale actual=%016x/%016x/%016x expected=%016x/%016x/%016x", math.Float64bits(record.InputScale), math.Float64bits(record.OutputScaleBefore), math.Float64bits(record.OutputScaleAfter), math.Float64bits(expected.inputScale), math.Float64bits(expected.outputScaleBefore), math.Float64bits(expected.outputScaleAfter))
	}
	if record.InputAliasesOutput != expected.alias {
		return fmt.Errorf("alias actual=%t expected=%t", record.InputAliasesOutput, expected.alias)
	}
	if record.ConstantReal != expected.real || record.ConstantImag != expected.imag {
		return fmt.Errorf("constant actual=(%s,%s) expected=(%s,%s)", record.ConstantReal, record.ConstantImag, expected.real, expected.imag)
	}
	if !equalUint64s(record.PositiveStandardRNS, expected.positiveStandard) || !equalUint64s(record.NegativeStandardRNS, expected.negativeStandard) || !equalUint64s(record.PositiveMontgomeryRNS, expected.positiveMont) || !equalUint64s(record.NegativeMontgomeryRNS, expected.negativeMont) {
		return fmt.Errorf("physical residues differ")
	}
	return nil
}

func equalUint64s(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func p2gn16PhysicalMakeCall(index int, record ckks.P2GN16PhysicalOperandRecord, expected p2gn16PhysicalExpected, exact bool) p2gn16PhysicalCall {
	return p2gn16PhysicalCall{
		CallIndex: index, AuthoritySequence: expected.event.Sequence, AuthorityOperation: expected.event.Operation,
		Kind: record.Kind, Context: record.Context, InputLevel: record.InputLevel, OutputLevel: record.OutputLevel,
		InputScale: record.InputScale, OutputScaleBefore: record.OutputScaleBefore, OutputScaleAfter: record.OutputScaleAfter,
		InputAliasesOutput: record.InputAliasesOutput, ConstantReal: record.ConstantReal, ConstantImag: record.ConstantImag,
		PositiveStandardRNS: record.PositiveStandardRNS, NegativeStandardRNS: record.NegativeStandardRNS,
		PositiveMontgomeryRNS: record.PositiveMontgomeryRNS, NegativeMontgomeryRNS: record.NegativeMontgomeryRNS,
		LineExactAgainstScaleOracle: exact,
	}
}

func p2gn16PhysicalBuildReport(repoRoot string, raw []ckks.P2GN16PhysicalOperandRecord, testExit int) (p2gn16PhysicalReport, error) {
	report := p2gn16PhysicalReport{
		Schema: p2gn16PhysicalObserverSchema, Status: "SAME_RUN_CAPTURE_FAIL_CLOSED",
		Parameter: guardedCompactP2GN16ParameterID, Construction: guardedCompactP2GN16CPUConstructionID,
		ObserverMode: "lattigo_v2.3_hash_gated_source_overlay_actual_api_boundary", PerformanceMeasured: false,
	}
	report.Coverage.FirstMismatchCall = -1
	report.Coverage.FirstMismatchAuthorityEvent = -1
	report.Transcript.GaussianExpected = p2gn16PhysicalObserverGaussianSHA
	report.Transcript.AddConstExpected = p2gn16PhysicalObserverAddConstNonzeroSHA
	report.Transcript.AddConstFullOracle = p2gn16PhysicalObserverAddConstFullOracleSHA
	report.Coverage.SemanticZeroAddConstElided = 19
	report.Blockers = []string{"physical capture does not itself release descriptor-v2, HBM payloads, a PDI, board eligibility, or performance data"}
	workspaceRoot := filepath.Clean(filepath.Join(repoRoot, "..", ".."))
	for _, binding := range []struct{ rootScope, root, path, sha string }{
		{"workspace", workspaceRoot, p2gn16PhysicalObserverStreamPath, p2gn16PhysicalObserverStreamSHA},
		{"module", repoRoot, p2gn16PhysicalObserverOraclePath, p2gn16PhysicalObserverOracleSHA},
		{"module", repoRoot, "mkf_cpu/p2gn16_same_run_physical_operand_observer_overlay_test.go", ""},
	} {
		data, err := os.ReadFile(filepath.Join(binding.root, filepath.FromSlash(binding.path)))
		if err != nil {
			return report, err
		}
		digest := p2gn16PhysicalDigest(data)
		if binding.sha != "" && digest != binding.sha {
			return report, fmt.Errorf("source identity drift for %s", binding.path)
		}
		report.SourceBindings = append(report.SourceBindings, struct {
			RootScope string `json:"root_scope"`
			Path      string `json:"path"`
			Bytes     int64  `json:"bytes"`
			SHA256    string `json:"sha256"`
		}{binding.rootScope, binding.path, int64(len(data)), digest})
	}
	streamData, err := p2gn16PhysicalReadBound(workspaceRoot, p2gn16PhysicalObserverStreamPath, p2gn16PhysicalObserverStreamSHA)
	if err != nil {
		return report, err
	}
	var authority struct {
		Schema              string              `json:"schema"`
		Status              string              `json:"status"`
		Parameter           string              `json:"parameter"`
		Construction        string              `json:"construction"`
		PerformanceMeasured bool                `json:"performance_measured"`
		BoardEligible       bool                `json:"board_eligible"`
		CommandStream       p2gn16CommandStream `json:"command_stream"`
	}
	if err := json.Unmarshal(streamData, &authority); err != nil {
		return report, err
	}
	if authority.Schema != "mkf-p2gn16-calibrated-semantic-command-authority-v1" || authority.Status != "CALIBRATED_SEMANTIC_SCHEDULE_ONLY_HARDWARE_LOWERING_INCOMPLETE" || authority.Parameter != guardedCompactP2GN16ParameterID || authority.Construction != guardedCompactP2GN16CPUConstructionID || authority.PerformanceMeasured || authority.BoardEligible {
		return report, fmt.Errorf("current-source authority envelope identity drift")
	}
	stream := authority.CommandStream
	if stream.Schema != p2gn16CommandStreamSchema || stream.Target.Parameter != guardedCompactP2GN16ParameterID || stream.Target.Construction != guardedCompactP2GN16CPUConstructionID || stream.BoardEligible || stream.PerformanceMeasured {
		return report, fmt.Errorf("current-source command stream identity drift")
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		return report, err
	}
	expectedGaussian, expectedAddConst, err := p2gn16PhysicalExpectedCalls(stream, profile.guarded)
	if err != nil {
		return report, err
	}
	actualGaussian := make([]ckks.P2GN16PhysicalOperandRecord, 0, 828)
	actualAddConst := make([]ckks.P2GN16PhysicalOperandRecord, 0, 285)
	for _, record := range raw {
		switch record.Kind {
		case "mult_by_gaussian_integer_and_add":
			actualGaussian = append(actualGaussian, record)
		case "add_const":
			actualAddConst = append(actualAddConst, record)
		default:
			// Preserved below as an unmatched raw call in the fail-closed report.
		}
	}
	report.Coverage.Calls = len(raw)
	report.Coverage.GaussianCalls = len(actualGaussian)
	report.Coverage.AddConstCalls = len(actualAddConst)
	gaussianTranscript := sha256.New()
	addConstTranscript := sha256.New()
	firstErr := error(nil)
	gaussianIndex, addConstIndex := 0, 0
	for callIndex, record := range raw {
		var expected p2gn16PhysicalExpected
		switch record.Kind {
		case "mult_by_gaussian_integer_and_add":
			if gaussianIndex >= len(expectedGaussian) {
				if firstErr == nil {
					firstErr = fmt.Errorf("extra Gaussian call %d", gaussianIndex)
				}
				report.Calls = append(report.Calls, p2gn16PhysicalMakeCall(callIndex, record, p2gn16PhysicalExpected{}, false))
				continue
			}
			expected = expectedGaussian[gaussianIndex]
			ratio := record.OutputScaleBefore / record.InputScale
			binary.Write(gaussianTranscript, binary.LittleEndian, uint64(expected.event.Sequence))
			binary.Write(gaussianTranscript, binary.LittleEndian, math.Float64bits(ratio))
			gaussianTranscript.Write([]byte(record.ConstantReal))
			gaussianTranscript.Write([]byte{0})
			gaussianTranscript.Write([]byte(record.ConstantImag))
			gaussianTranscript.Write([]byte{0})
			gaussianTranscript.Write(p2gn16PhysicalInterleave(record.PositiveMontgomeryRNS, record.NegativeMontgomeryRNS))
			if !record.InputAliasesOutput {
				report.Coverage.GaussianOutOfPlaceCalls++
			}
			gaussianIndex++
		case "add_const":
			if addConstIndex >= len(expectedAddConst) {
				if firstErr == nil {
					firstErr = fmt.Errorf("extra AddConst call %d", addConstIndex)
				}
				report.Calls = append(report.Calls, p2gn16PhysicalMakeCall(callIndex, record, p2gn16PhysicalExpected{}, false))
				continue
			}
			expected = expectedAddConst[addConstIndex]
			binary.Write(addConstTranscript, binary.LittleEndian, uint64(expected.event.Sequence))
			binary.Write(addConstTranscript, binary.LittleEndian, math.Float64bits(record.OutputScaleAfter))
			addConstTranscript.Write(p2gn16PhysicalInterleave(record.PositiveStandardRNS, record.NegativeStandardRNS))
			if record.InputAliasesOutput {
				report.Coverage.AddConstInPlaceCalls++
			}
			realValue, _ := strconv.ParseFloat(record.ConstantReal, 64)
			imagValue, _ := strconv.ParseFloat(record.ConstantImag, 64)
			if realValue == 0 && imagValue == 0 {
				report.Coverage.AddConstZeroCalls++
			} else {
				report.Coverage.AddConstNonzeroCalls++
			}
			addConstIndex++
		default:
			if firstErr == nil {
				firstErr = fmt.Errorf("unknown physical observer kind %q", record.Kind)
			}
			report.Calls = append(report.Calls, p2gn16PhysicalMakeCall(callIndex, record, p2gn16PhysicalExpected{}, false))
			continue
		}
		compareErr := p2gn16PhysicalCompareRecord(record, expected)
		exact := compareErr == nil
		if exact {
			report.Coverage.LineExactCalls++
		} else if firstErr == nil {
			firstErr = fmt.Errorf("physical call %d kind %s event %d: %w", callIndex, record.Kind, expected.event.Sequence, compareErr)
			report.Coverage.FirstMismatchCall = callIndex
			report.Coverage.FirstMismatchAuthorityEvent = expected.event.Sequence
		}
		report.Calls = append(report.Calls, p2gn16PhysicalMakeCall(callIndex, record, expected, exact))
	}
	report.Transcript.GaussianSHA256 = hex.EncodeToString(gaussianTranscript.Sum(nil))
	report.Transcript.AddConstSHA256 = hex.EncodeToString(addConstTranscript.Sum(nil))
	report.Transcript.GaussianExact = report.Transcript.GaussianSHA256 == p2gn16PhysicalObserverGaussianSHA
	report.Transcript.AddConstExact = report.Transcript.AddConstSHA256 == p2gn16PhysicalObserverAddConstNonzeroSHA
	if testExit != 0 && firstErr == nil {
		firstErr = fmt.Errorf("canonical functional test exit=%d", testExit)
	}
	if len(actualGaussian) != 828 || len(actualAddConst) != 285 || report.Coverage.AddConstNonzeroCalls != 285 || report.Coverage.AddConstZeroCalls != 0 || report.Coverage.SemanticZeroAddConstElided != 19 || report.Coverage.GaussianOutOfPlaceCalls != 828 || report.Coverage.AddConstInPlaceCalls != 285 || report.Coverage.LineExactCalls != 1113 || !report.Transcript.GaussianExact || !report.Transcript.AddConstExact {
		if firstErr == nil {
			firstErr = fmt.Errorf("coverage/transcript gate failed: %+v %+v", report.Coverage, report.Transcript)
		}
	}
	if firstErr != nil {
		report.Blockers = append(report.Blockers, firstErr.Error())
		return report, firstErr
	}
	report.Status = "SAME_RUN_PHYSICAL_OPERANDS_EXACT_PASS_RELEASE_STILL_BLOCKED"
	return report, nil
}

func p2gn16PhysicalWriteReport(root string, report p2gn16PhysicalReport) (string, error) {
	if err := os.MkdirAll(root, 0755); err != nil {
		return "", err
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", err
	}
	encoded = append(encoded, '\n')
	path := filepath.Join(root, "p2gn16_same_run_physical_operand_observer.json")
	if err := os.WriteFile(path, encoded, 0644); err != nil {
		return "", err
	}
	return p2gn16PhysicalDigest(encoded), nil
}

func p2gn16PhysicalExactRunSelected() bool {
	for index, argument := range os.Args {
		if argument == "-test.run=^TestGuardedCompactP2GResNet20FullFunctionalOptIn$" {
			return true
		}
		if argument == "-test.run" && index+1 < len(os.Args) && os.Args[index+1] == "^TestGuardedCompactP2GResNet20FullFunctionalOptIn$" {
			return true
		}
	}
	return false
}

func TestMain(m *testing.M) {
	exportRoot := strings.TrimSpace(os.Getenv(p2gn16PhysicalObserverExportEnv))
	if exportRoot == "" {
		os.Exit(m.Run())
	}
	if os.Getenv(p2gn16PhysicalObserverAckEnv) != p2gn16PhysicalObserverAckValue || os.Getenv("MKF_RUN_GUARDED_COMPACT_RESNET20_P2G_FULL") != "1" || os.Getenv("MKF_GUARDED_COMPACT_PARAMETER") != guardedCompactP2GN16ParameterID || runtime.GOMAXPROCS(0) != 1 || !p2gn16PhysicalExactRunSelected() {
		fmt.Fprintln(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_GATE_FAIL: require explicit acknowledgement, full-functional gate, P2GN16, GOMAXPROCS=1, and the exact single test selector")
		os.Exit(2)
	}
	for _, forbidden := range []string{"MKF_RUN_P2GN16_MATCHED_CPU_BENCHMARK", "MKF_CPU_BENCHMARK", "MKF_FPGA_BENCHMARK", "MKF_BOARD_ELIGIBLE"} {
		if os.Getenv(forbidden) != "" {
			fmt.Fprintf(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_GATE_FAIL: performance/board environment %s is forbidden\n", forbidden)
			os.Exit(2)
		}
	}
	ckks.SetP2GN16PhysicalOperandContext("canonical_p2gn16_guarded_compact_margin070_full_functional")
	restore := ckks.SetP2GN16PhysicalOperandObserver("canonical_p2gn16_guarded_compact_margin070_full_functional", p2gn16PhysicalCaptureRecord)
	exitCode := m.Run()
	restore()
	p2gn16PhysicalCapture.Lock()
	records := append([]ckks.P2GN16PhysicalOperandRecord(nil), p2gn16PhysicalCapture.records...)
	p2gn16PhysicalCapture.Unlock()
	repoRoot, rootErr := p2gn16PhysicalRepoRoot()
	if rootErr != nil {
		fmt.Fprintf(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_EXPORT_FAIL: %v\n", rootErr)
		os.Exit(1)
	}
	report, validationErr := p2gn16PhysicalBuildReport(repoRoot, records, exitCode)
	digest, writeErr := p2gn16PhysicalWriteReport(exportRoot, report)
	if writeErr != nil {
		fmt.Fprintf(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_EXPORT_FAIL: %v\n", writeErr)
		os.Exit(1)
	}
	if validationErr != nil {
		fmt.Fprintf(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_EXACT_FAIL: %v report_sha256=%s\n", validationErr, digest)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "P2GN16_PHYSICAL_OBSERVER_EXACT_PASS calls=1113 gaussian=828 add_const=285(nonzero=285 physical_zero=0 semantic_zero_elided=19) report_sha256=%s board_eligible=false performance_measured=false\n", digest)
	os.Exit(exitCode)
}

func TestP2GN16PhysicalObserverOverlayContractNoCrypto(t *testing.T) {
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	params := profile.guarded
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{})
	level := 2
	scale := params.Scale()
	input := ckks.NewCiphertext(params, 1, level, scale)
	output := ckks.NewCiphertext(params, 1, level, scale*8)
	var records []ckks.P2GN16PhysicalOperandRecord
	restore := ckks.SetP2GN16PhysicalOperandObserver("no_crypto_contract", func(record ckks.P2GN16PhysicalOperandRecord) {
		records = append(records, p2gn16PhysicalClone(record))
	})
	evaluator.AddConst(input, complex(-1.25, 0.5), input)
	evaluator.MultByGaussianIntegerAndAdd(input, big.NewInt(-17), big.NewInt(9), output)
	restore()
	if len(records) != 2 {
		t.Fatalf("observer records=%d, want 2", len(records))
	}
	add := records[0]
	if add.Kind != "add_const" || !add.InputAliasesOutput || add.Context != "no_crypto_contract" || !p2gn16PhysicalSameBits(add.InputScale, scale) || !p2gn16PhysicalSameBits(add.OutputScaleAfter, scale) {
		t.Fatalf("AddConst observer metadata drift: %+v", add)
	}
	addPositive, addNegative, err := p2gn16PhysicalSplitPayload(p2gn16AddConstCompactPayload(params.RingQ(), level, scale, complex(-1.25, 0.5)))
	if err != nil {
		t.Fatal(err)
	}
	if !equalUint64s(add.PositiveStandardRNS, addPositive) || !equalUint64s(add.NegativeStandardRNS, addNegative) {
		t.Fatal("AddConst physical residues differ from oracle")
	}
	gaussian := records[1]
	if gaussian.Kind != "mult_by_gaussian_integer_and_add" || gaussian.InputAliasesOutput || gaussian.Context != "no_crypto_contract" || gaussian.ConstantReal != "-17" || gaussian.ConstantImag != "9" {
		t.Fatalf("Gaussian observer metadata drift: %+v", gaussian)
	}
	standardPositive, standardNegative := p2gn16PhysicalStandardGaussian(params.RingQ(), level, big.NewInt(-17), big.NewInt(9))
	montPositive, montNegative, err := p2gn16PhysicalSplitPayload(p2gn16GaussianCompactPayload(params.RingQ(), level, big.NewInt(-17), big.NewInt(9)))
	if err != nil {
		t.Fatal(err)
	}
	if !equalUint64s(gaussian.PositiveStandardRNS, standardPositive) || !equalUint64s(gaussian.NegativeStandardRNS, standardNegative) || !equalUint64s(gaussian.PositiveMontgomeryRNS, montPositive) || !equalUint64s(gaussian.NegativeMontgomeryRNS, montNegative) {
		t.Fatal("Gaussian physical residues differ from oracle")
	}
	t.Log("P2GN16_PHYSICAL_OBSERVER_OVERLAY_NO_CRYPTO_PASS calls=2 add_const_alias=in_place gaussian_alias=out_of_place standard_and_montgomery_residues=exact production_populated=false board_eligible=false performance_measured=false")
}

func TestP2GN16PhysicalObserverExpectedTranscriptNoCrypto(t *testing.T) {
	moduleRoot, err := p2gn16PhysicalRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	workspaceRoot := filepath.Clean(filepath.Join(moduleRoot, "..", ".."))
	data, err := p2gn16PhysicalReadBound(workspaceRoot, p2gn16PhysicalObserverStreamPath, p2gn16PhysicalObserverStreamSHA)
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		CommandStream p2gn16CommandStream `json:"command_stream"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		t.Fatal(err)
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	gaussian, addConst, err := p2gn16PhysicalExpectedCalls(envelope.CommandStream, profile.guarded)
	if err != nil {
		t.Fatal(err)
	}
	gaussianTranscript := sha256.New()
	for _, expected := range gaussian {
		ratio := expected.outputScaleBefore / expected.inputScale
		binary.Write(gaussianTranscript, binary.LittleEndian, uint64(expected.event.Sequence))
		binary.Write(gaussianTranscript, binary.LittleEndian, math.Float64bits(ratio))
		gaussianTranscript.Write([]byte(expected.real))
		gaussianTranscript.Write([]byte{0})
		gaussianTranscript.Write([]byte(expected.imag))
		gaussianTranscript.Write([]byte{0})
		gaussianTranscript.Write(p2gn16PhysicalInterleave(expected.positiveMont, expected.negativeMont))
	}
	addConstTranscript := sha256.New()
	for _, expected := range addConst {
		binary.Write(addConstTranscript, binary.LittleEndian, uint64(expected.event.Sequence))
		binary.Write(addConstTranscript, binary.LittleEndian, math.Float64bits(expected.outputScaleAfter))
		addConstTranscript.Write(p2gn16PhysicalInterleave(expected.positiveStandard, expected.negativeStandard))
	}
	gaussianSHA := hex.EncodeToString(gaussianTranscript.Sum(nil))
	addConstSHA := hex.EncodeToString(addConstTranscript.Sum(nil))
	if gaussianSHA != p2gn16PhysicalObserverGaussianSHA {
		t.Fatalf("Gaussian expected transcript=%s, want %s", gaussianSHA, p2gn16PhysicalObserverGaussianSHA)
	}
	if addConstSHA != p2gn16PhysicalObserverAddConstNonzeroSHA {
		t.Fatalf("nonzero AddConst expected transcript=%s, want %s", addConstSHA, p2gn16PhysicalObserverAddConstNonzeroSHA)
	}
	t.Logf("P2GN16_PHYSICAL_EXPECTED_TRANSCRIPT_NO_CRYPTO_PASS gaussian_calls=%d gaussian_sha256=%s nonzero_add_const_calls=%d nonzero_add_const_sha256=%s semantic_zero_add_const_elided=19", len(gaussian), gaussianSHA, len(addConst), addConstSHA)
}

func TestP2GN16PhysicalObserverExporterSyntheticTranscriptNoCrypto(t *testing.T) {
	moduleRoot, err := p2gn16PhysicalRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	workspaceRoot := filepath.Clean(filepath.Join(moduleRoot, "..", ".."))
	data, err := p2gn16PhysicalReadBound(workspaceRoot, p2gn16PhysicalObserverStreamPath, p2gn16PhysicalObserverStreamSHA)
	if err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		CommandStream p2gn16CommandStream `json:"command_stream"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		t.Fatal(err)
	}
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	gaussian, addConst, err := p2gn16PhysicalExpectedCalls(envelope.CommandStream, profile.guarded)
	if err != nil {
		t.Fatal(err)
	}
	type sequenced struct {
		sequence int
		record   ckks.P2GN16PhysicalOperandRecord
	}
	merged := make([]sequenced, 0, 1113)
	for _, expected := range gaussian {
		merged = append(merged, sequenced{expected.event.Sequence, ckks.P2GN16PhysicalOperandRecord{
			Kind: "mult_by_gaussian_integer_and_add", Context: "synthetic_exporter_contract",
			InputLevel: expected.inputLevel, OutputLevel: expected.outputLevel,
			InputScale: expected.inputScale, OutputScaleBefore: expected.outputScaleBefore, OutputScaleAfter: expected.outputScaleAfter,
			InputAliasesOutput: expected.alias, ConstantReal: expected.real, ConstantImag: expected.imag,
			PositiveStandardRNS: expected.positiveStandard, NegativeStandardRNS: expected.negativeStandard,
			PositiveMontgomeryRNS: expected.positiveMont, NegativeMontgomeryRNS: expected.negativeMont,
		}})
	}
	for _, expected := range addConst {
		merged = append(merged, sequenced{expected.event.Sequence, ckks.P2GN16PhysicalOperandRecord{
			Kind: "add_const", Context: "synthetic_exporter_contract",
			InputLevel: expected.inputLevel, OutputLevel: expected.outputLevel,
			InputScale: expected.inputScale, OutputScaleBefore: expected.outputScaleBefore, OutputScaleAfter: expected.outputScaleAfter,
			InputAliasesOutput: expected.alias, ConstantReal: expected.real, ConstantImag: expected.imag,
			PositiveStandardRNS: expected.positiveStandard, NegativeStandardRNS: expected.negativeStandard,
		}})
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].sequence < merged[j].sequence })
	raw := make([]ckks.P2GN16PhysicalOperandRecord, len(merged))
	for index := range merged {
		raw[index] = merged[index].record
	}
	report, err := p2gn16PhysicalBuildReport(moduleRoot, raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	if report.Status != "SAME_RUN_PHYSICAL_OPERANDS_EXACT_PASS_RELEASE_STILL_BLOCKED" || report.Coverage.Calls != 1113 || report.Coverage.LineExactCalls != 1113 || !report.Transcript.GaussianExact || !report.Transcript.AddConstExact || report.ReleaseState.BoardEligible || report.PerformanceMeasured {
		t.Fatalf("synthetic exporter report drift: %+v %+v", report.Coverage, report.Transcript)
	}
	t.Log("P2GN16_PHYSICAL_EXPORTER_SYNTHETIC_NO_CRYPTO_PASS calls=1113 gaussian=828 nonzero_add_const=285 semantic_zero_elided=19 transcripts=exact board_eligible=false performance_measured=false")
}
