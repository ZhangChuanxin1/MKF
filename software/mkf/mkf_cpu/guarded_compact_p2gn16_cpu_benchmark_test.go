package mkfcpu

// This file adapts the audited phase-segmented ResNet-20 schedule to the
// security-gated P2GN16 canonical MKF construction. It deliberately contains
// no default benchmark entry point: the durable matched runner in
// p2gn16_matched_cpu_benchmark_test.go is the only formal launcher.

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
)

const guardedCompactP2GN16CPUConstructionID = "MKF-CKKS-P2GN16-guarded-fold-once+permanent-aggregate-rq2-v1"

type guardedCompactP2GN16TrialEvidence struct {
	Parameter           string                  `json:"parameter"`
	SecurityGate        string                  `json:"security_gate"`
	Owners              int                     `json:"guarded_owner_ingresses"`
	Folds               int                     `json:"guarded_folds"`
	IngressRescales     int                     `json:"guard_prime_rescales"`
	NetworkProducts     int                     `json:"network_products"`
	PolynomialProducts  int                     `json:"polynomial_products"`
	Rotations           int                     `json:"rotations"`
	Degree59Activations int                     `json:"degree59_activations"`
	Refreshes           int                     `json:"collective_refreshes"`
	MaxError            float64                 `json:"max_error"`
	EncryptedTop1       int                     `json:"encrypted_top1"`
	ReferenceTop1       int                     `json:"reference_top1"`
	Label               int                     `json:"label"`
	CorrectnessPass     bool                    `json:"correctness_pass"`
	ClassificationPass  bool                    `json:"classification_pass"`
	CountPass           bool                    `json:"count_pass"`
	Counts              guardedCompactMKFCounts `json:"-"`
}

func newGuardedCompactP2GN16CPUTrial(request fullResNet20Request) (*paperMKFResNet20CPUTrial, error) {
	if request.Parameter != guardedCompactP2GN16ParameterID || request.Mode != fullResNet20PerformanceMode {
		return nil, fmt.Errorf("canonical CPU trial requires P2GN16 performance mode")
	}
	executorInterface, err := newGuardedCompactFullResNet20ExecutorWithRotations(request, fullResNet20RotationKeys())
	if err != nil {
		return nil, err
	}
	executor, ok := executorInterface.(*encryptedFullResNet20Executor)
	if !ok || executor.guardedCompactAdapter == nil {
		return nil, fmt.Errorf("canonical constructor returned %T without guarded adapter", executorInterface)
	}
	trial := &paperMKFResNet20CPUTrial{
		request: request, executor: executor, nextPhase: 1, guardedCanonical: true,
		timedEvaluatorOverride: executor.guardedCompactAdapter,
		communication: map[resNet20CPUStage]paperMKFCPUStageCommunication{
			resNet20CPUStem: {Stage: resNet20CPUStem}, resNet20CPUResidual: {Stage: resNet20CPUResidual}, resNet20CPUFullyConn: {Stage: resNet20CPUFullyConn},
		},
		recordCommunication: func(paperMKFCPUTrialCommunication) {},
	}

	protocol := dckks.NewRefreshProtocol(executor.params, executor.logBound, collectiveRefreshSigmaSmudging)
	share := protocol.AllocateShare(executor.minLevel, executor.guardedCompactAdapter.baseMaxLevel)
	wire, err := share.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("measure canonical refresh-share encoding outside timing: %w", err)
	}
	trial.refreshShareSize = uint64(len(wire))

	trial.stageOwnerOverride = func(owner int, label string, values []complex128) (paperMKFStagedOwnerCiphertext, error) {
		return trial.stageGuardedOwnerOutsideTiming(owner, label, values)
	}
	trial.timedRefreshOverride = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
		adapter := executor.guardedCompactAdapter
		adapter.requireCompact("timed collective refresh input", input)
		if input.Level() < executor.minLevel {
			return nil, fmt.Errorf("timed refresh input level=%d below minimum=%d", input.Level(), executor.minLevel)
		}
		if input.Level() > executor.minLevel {
			adapter.Evaluator.DropLevel(input, input.Level()-executor.minLevel)
		}
		output, refreshErr := runGuardedCompactCollectiveRefresh(executor.params, executor.shares, input, executor.logBound, adapter.baseMaxLevel, executor.refreshCRS)
		if refreshErr != nil {
			return nil, refreshErr
		}
		adapter.requireCompact("timed collective refresh output", output)
		adapter.counts.refreshes++
		return output, nil
	}

	trial.inputCiphertexts = make([]*ckks.Ciphertext, resnetImageChannels)
	trial.inputMaterializers = make([]func() (*ckks.Ciphertext, error), resnetImageChannels)
	stemCommunication := trial.communication[resNet20CPUStem]
	for channel := 0; channel < resnetImageChannels; channel++ {
		packed := packReplicatedInputChannel(executor.params, request.Fixture.NormalizedRGB, channel)
		entry, stageErr := trial.stageGuardedOwnerOutsideTiming(realResNetTailDataOwner, fmt.Sprintf("Stem/input/channel=%d", channel), packed)
		if stageErr != nil {
			return nil, stageErr
		}
		trial.inputMaterializers[channel] = entry.materialize
		trial.ownerCount++
		stemCommunication.OwnerIngressCiphertexts++
		stemCommunication.OwnerIngressBytes += entry.wireBytes
	}
	trial.communication[resNet20CPUStem] = stemCommunication
	return trial, nil
}

func (trial *paperMKFResNet20CPUTrial) stageGuardedOwnerOutsideTiming(owner int, label string, values []complex128) (paperMKFStagedOwnerCiphertext, error) {
	adapter := trial.executor.guardedCompactAdapter
	if adapter == nil || owner < 0 || owner >= len(trial.executor.public) || len(values) != adapter.params.Slots() {
		return paperMKFStagedOwnerCiphertext{}, fmt.Errorf("invalid staged P2GN16 owner=%d label=%s slots=%d", owner, label, len(values))
	}
	level := adapter.params.MaxLevel()
	guardedScale := adapter.params.Scale() * float64(adapter.guardPrime)
	plaintext := ckks.NewPlaintext(adapter.params, level, guardedScale)
	trial.executor.encoder.EncodeNTT(plaintext, values, adapter.params.LogSlots())
	raw := ckks.NewEncryptor(adapter.params, trial.executor.public[owner]).EncryptNew(plaintext)
	wire, err := raw.MarshalBinary()
	if err != nil {
		return paperMKFStagedOwnerCiphertext{}, fmt.Errorf("marshal staged P2GN16 owner %s: %w", label, err)
	}
	used := false
	return paperMKFStagedOwnerCiphertext{
		label: label, wireBytes: uint64(len(wire)),
		materialize: func() (*ckks.Ciphertext, error) {
			if used {
				return nil, fmt.Errorf("staged P2GN16 owner %s materialized more than once", label)
			}
			used = true
			return adapter.foldGuardedOwnerCiphertext(raw, owner)
		},
	}, nil
}

func (trial *paperMKFResNet20CPUTrial) verifyGuardedCanonical() error {
	if trial.verified || trial.active != nil || trial.nextPhase != len(trial.request.Plan.Phases) || trial.current == nil || len(trial.reference) != trial.executor.params.Slots() {
		return fmt.Errorf("canonical trial is incomplete at untimed verification: nextPhase=%d active=%t", trial.nextPhase, trial.active != nil)
	}
	trial.verified = true
	counts := trial.executor.guardedCompactAdapter.snapshot()
	countPass := counts.owners == 6201 && counts.guardedFolds == 6201 && counts.guardRescales == 6201 && counts.networkProducts == 6172 && counts.polynomialProducts == 323 && trial.expectedPolynomialMulRelin == 323 && counts.rotations == 6499 && counts.polynomials == 19 && counts.refreshes == 22
	if !countPass {
		return fmt.Errorf("P2GN16 canonical final count mismatch: owners/folds/rescales/network-products/polynomial-products(actual/expected)/rotations/poly/refresh=%d/%d/%d/%d/%d/%d/%d/%d/%d", counts.owners, counts.guardedFolds, counts.guardRescales, counts.networkProducts, counts.polynomialProducts, trial.expectedPolynomialMulRelin, counts.rotations, counts.polynomials, counts.refreshes)
	}
	decoded := trial.executor.encoder.Decode(ckks.NewDecryptor(trial.executor.params, trial.executor.aggregate).DecryptNew(trial.current), trial.executor.params.LogSlots())
	if len(decoded) < len(trial.reference) {
		return fmt.Errorf("canonical decoded %d slots, want %d", len(decoded), len(trial.reference))
	}
	maxError := 0.0
	for index, want := range trial.reference {
		delta := cmplx.Abs(decoded[index] - want)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			return fmt.Errorf("canonical non-finite final error at slot %d", index)
		}
		if delta > maxError {
			maxError = delta
		}
	}
	correctnessPass := maxError < fullResNet20FunctionalGate
	if !correctnessPass {
		return fmt.Errorf("canonical final max error=%g exceeds %g", maxError, fullResNet20FunctionalGate)
	}
	encryptedLogits := make([]float64, realResNetTailClasses)
	referenceLogits := make([]float64, realResNetTailClasses)
	for class := 0; class < realResNetTailClasses; class++ {
		encryptedLogits[class] = real(decoded[class])
		referenceLogits[class] = real(trial.reference[class])
	}
	encryptedTop1, _ := fullResNet20Top1AndMargin(encryptedLogits)
	referenceTop1, _ := fullResNet20Top1AndMargin(referenceLogits)
	label := int(trial.request.Fixture.Label)
	classificationPass := encryptedTop1 == label && referenceTop1 == label && encryptedTop1 == referenceTop1
	if !classificationPass {
		return fmt.Errorf("canonical classification mismatch: encrypted=%d reference=%d label=%d", encryptedTop1, referenceTop1, label)
	}
	trial.guardedEvidence = guardedCompactP2GN16TrialEvidence{
		Parameter: guardedCompactP2GN16ParameterID, SecurityGate: guardedCompactP2GN16SecurityGate,
		Owners: counts.owners, Folds: counts.guardedFolds, IngressRescales: counts.guardRescales,
		NetworkProducts: counts.networkProducts, PolynomialProducts: counts.polynomialProducts, Rotations: counts.rotations, Degree59Activations: counts.polynomials, Refreshes: counts.refreshes,
		MaxError: maxError, EncryptedTop1: encryptedTop1, ReferenceTop1: referenceTop1, Label: label,
		CorrectnessPass: correctnessPass, ClassificationPass: classificationPass, CountPass: countPass, Counts: counts,
	}
	return nil
}
