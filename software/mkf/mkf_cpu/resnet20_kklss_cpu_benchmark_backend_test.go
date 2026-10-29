package mkfcpu

// This file is the fail-closed, phase-prepared CPU measurement adapter for
// the public SNUCP/MKHE-KKLSS implementation. It intentionally does not call
// encryptedFullResNet20Executor.ExecutePhase: that functional path loads,
// packs and encrypts model material and decrypts intermediate boundaries.
// Here key generation, fixture/model loading, plaintext reference work,
// owner encryption, staging serialization and final verification are outside
// every timer. Only the prepared segment's Run method is timed.

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

type kklssCPUMessageBytePolicy struct {
	Schema                  string `json:"schema"`
	OwnerIngressEncoding    string `json:"owner_ingress_encoding"`
	CKSShareEncoding        string `json:"cks_share_encoding"`
	RefreshShareEncoding    string `json:"refresh_share_encoding"`
	EvaluationKeyEncoding   string `json:"evaluation_key_encoding"`
	EvaluationKeyScope      string `json:"evaluation_key_scope"`
	OnlineTimingScope       string `json:"online_timing_scope"`
	TransportScope          string `json:"transport_scope"`
	CKSIngressShareMessages int    `json:"cks_ingress_share_messages"`
}

type kklssCPUStageCommunication struct {
	Stage                   resNet20CPUStage `json:"stage"`
	OwnerIngressCiphertexts int              `json:"owner_ingress_ciphertexts"`
	OwnerIngressBytes       uint64           `json:"owner_ingress_bytes"`
	RefreshRounds           int              `json:"refresh_rounds"`
	CKSShareMessages        int              `json:"cks_share_messages"`
	CKSShareBytes           uint64           `json:"cks_share_bytes"`
	RefreshShareMessages    int              `json:"refresh_share_messages"`
	RefreshShareBytes       uint64           `json:"refresh_share_bytes"`
}

type kklssCPUTrialCommunication struct {
	Trial                 int                                             `json:"trial"`
	EvaluationKeyPayloads int                                             `json:"evaluation_key_switching_key_payloads"`
	EvaluationKeyBytes    uint64                                          `json:"evaluation_key_polyqp_bytes"`
	Stages                map[resNet20CPUStage]kklssCPUStageCommunication `json:"stages"`
}

type kklssCPUBackendState struct {
	mu             sync.Mutex
	communications []kklssCPUTrialCommunication
	requestOnce    sync.Once
	request        fullResNet20Request
	requestErr     error
}

func (backend *kklssResNet20CPUBackend) CommunicationBytePolicy() kklssCPUMessageBytePolicy {
	return kklssCPUMessageBytePolicy{
		Schema:                  "public-kklss-resnet20-cpu-message-byte-policy-v1",
		OwnerIngressEncoding:    "sum len(marshalKKLSSStagedCiphertext) over 6201 native sparse single-owner KKLSS ciphertexts, checksum-verified by untimed unmarshal before materialization",
		CKSShareEncoding:        "for each refresh input level, four logical source-by-party shares times len(drlwe.CKSShare.MarshalBinary()); level is recorded during Run and zero-share encoding length is measured during untimed Release",
		RefreshShareEncoding:    "22 rounds times two parties times len(dckks.RefreshShare.MarshalBinary()) for fixed minLevel-to-MaxLevel shape, measured outside timing",
		EvaluationKeyEncoding:   "sum exact rlwe.PolyQP.GetDataLen(true), verified by PolyQP.WriteTo, across every public-KKLSS relinearization and required rotation switching-key polynomial",
		EvaluationKeyScope:      "one-time offline public-KKLSS P2/n=2 relinearization and frozen rotation material; excludes secret/public encryption keys and CRS transport",
		OnlineTimingScope:       "Run includes native network and polynomial MulRelinNew, rotations, rescaling/modulus management, CKS share generation/aggregation, collective refresh, and aggregate-domain rematerialization; all byte serialization is excluded",
		TransportScope:          "current uncompressed Lattigo v2.3.0 encodings only; excludes framing, copies, sockets, and network latency",
		CKSIngressShareMessages: fullResNet20Parties * fullResNet20Parties * 22,
	}
}

func kklssExpectedStageCommunication(stage resNet20CPUStage) (owners, refreshes int, ok bool) {
	switch stage {
	case resNet20CPUStem:
		return 31, 1, true
	case resNet20CPUResidual:
		return 6168, 21, true
	case resNet20CPUFullyConn:
		return 2, 0, true
	default:
		return 0, 0, false
	}
}

func (backend *kklssResNet20CPUBackend) recordCommunication(evidence kklssCPUTrialCommunication) {
	backend.state.mu.Lock()
	defer backend.state.mu.Unlock()
	evidence.Trial = len(backend.state.communications) + 1
	backend.state.communications = append(backend.state.communications, evidence)
}

func (backend *kklssResNet20CPUBackend) communicationEvidence() []kklssCPUTrialCommunication {
	backend.state.mu.Lock()
	defer backend.state.mu.Unlock()
	result := make([]kklssCPUTrialCommunication, len(backend.state.communications))
	for index, trial := range backend.state.communications {
		result[index] = kklssCPUTrialCommunication{
			Trial: trial.Trial, EvaluationKeyPayloads: trial.EvaluationKeyPayloads,
			EvaluationKeyBytes: trial.EvaluationKeyBytes,
			Stages:             make(map[resNet20CPUStage]kklssCPUStageCommunication, len(trial.Stages)),
		}
		for stage, evidence := range trial.Stages {
			result[index].Stages[stage] = evidence
		}
	}
	return result
}

func (backend *kklssResNet20CPUBackend) CommunicationArtifacts() (json.RawMessage, json.RawMessage, error) {
	if backend == nil || backend.state == nil {
		return nil, nil, errKKLSSResNet20BackendUninitialized
	}
	evidence := backend.communicationEvidence()
	if len(evidence) != resNet20CPUTrials {
		return nil, nil, fmt.Errorf("KKLSS communication evidence has %d trials, want %d", len(evidence), resNet20CPUTrials)
	}
	for index, trial := range evidence {
		if trial.Trial != index+1 || trial.EvaluationKeyPayloads <= 0 || trial.EvaluationKeyBytes == 0 || len(trial.Stages) != len(resNet20CPUStages) {
			return nil, nil, fmt.Errorf("KKLSS communication trial %d is incomplete: %+v", index+1, trial)
		}
		if index > 0 {
			if trial.EvaluationKeyPayloads != evidence[0].EvaluationKeyPayloads || trial.EvaluationKeyBytes != evidence[0].EvaluationKeyBytes {
				return nil, nil, fmt.Errorf("KKLSS evaluation-key evidence changed between trials 1 and %d", index+1)
			}
			for _, stage := range resNet20CPUStages {
				if trial.Stages[stage] != evidence[0].Stages[stage] {
					return nil, nil, fmt.Errorf("KKLSS %s communication evidence changed between trials 1 and %d", stage, index+1)
				}
			}
		}
	}
	policyJSON, err := json.Marshal(backend.CommunicationBytePolicy())
	if err != nil {
		return nil, nil, err
	}
	evidenceJSON, err := json.Marshal(struct {
		Schema string                       `json:"schema"`
		Trials []kklssCPUTrialCommunication `json:"trials"`
	}{Schema: "public-kklss-resnet20-cpu-message-byte-evidence-v1", Trials: evidence})
	if err != nil {
		return nil, nil, err
	}
	return policyJSON, evidenceJSON, nil
}

func kklssCPURequestFromEnvironment() (fullResNet20Request, error) {
	return kklssCPURequestFromEnvironmentFor("P2")
}

func kklssCPURequestFromEnvironmentFor(parameter string) (fullResNet20Request, error) {
	scales, err := loadFullResNet20Calibration(os.Getenv("MKF_RESNET20_ACTIVATION_SCALES_JSON"))
	if err != nil {
		return fullResNet20Request{}, fmt.Errorf("load frozen activation scales: %w", err)
	}
	plan, err := buildFullResNet20Plan(scales)
	if err != nil {
		return fullResNet20Request{}, err
	}
	plan.Communication.CKSIngressShareMessages = fullResNet20Parties * fullResNet20Parties * plan.Expected.CollectiveRefreshes
	plan.Communication.CKSIngressBytes = 0
	fixture, err := loadFullResNet20OfficialFixture(os.Getenv("MKF_RESNET20_CIFAR_FIXTURE_DIR"))
	if err != nil {
		return fullResNet20Request{}, fmt.Errorf("load frozen CIFAR fixture: %w", err)
	}
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
	if weightsDirectory == "" || biasPath == "" {
		return fullResNet20Request{}, fmt.Errorf("MKF_RESNET20_WEIGHTS_DIR and MKF_RESNET20_FC_BIAS_PATH are required")
	}
	if _, err := auditFullResNet20FrozenInventory(plan, weightsDirectory, biasPath); err != nil {
		return fullResNet20Request{}, fmt.Errorf("audit frozen checkpoint inventory: %w", err)
	}
	biasData, err := os.ReadFile(biasPath)
	if err != nil {
		return fullResNet20Request{}, err
	}
	if digest := fmt.Sprintf("%x", sha256.Sum256(biasData)); digest != fullResNet20CheckpointFCBiasSHA {
		return fullResNet20Request{}, fmt.Errorf("checkpoint FC-bias SHA-256 mismatch: %s", digest)
	}
	if scales.SourceSHA256 != "ea9e6bad9179fbf1bff4209b741ca7a199def0c1b56ccd8b5d80e1d4b4bfae86" {
		return fullResNet20Request{}, fmt.Errorf("activation-scale artifact mismatch: %s", scales.SourceSHA256)
	}
	return fullResNet20Request{
		Parameter: parameter, Mode: fullResNet20PerformanceMode, Plan: plan, Fixture: fixture,
		WeightsDirectory: weightsDirectory, CheckpointBiasPath: biasPath,
	}, nil
}

func newKKLSSPerformanceExecutor(request fullResNet20Request) (*encryptedFullResNet20Executor, error) {
	if request.Mode != fullResNet20PerformanceMode || (request.Parameter != "P2" && request.Parameter != guardedCompactP2GN16ParameterID) {
		return nil, fmt.Errorf("KKLSS CPU backend requires frozen P2 or P2GN16 performance mode")
	}
	wantCounts := fullResNet20Counts{DataInputCiphertexts: 3, ModelWeightCiphertexts: 6172, ModelBiasCiphertexts: 26, CiphertextProducts: 6172, RotationCalls: 6499, CollectiveRefreshes: 22, Degree59Activations: 19, PublicMaskProducts: 341, PackingOrTailRescaleLevels: 8}
	if len(request.Plan.Phases) != 12 || request.Plan.Expected != wantCounts || request.Plan.Communication.CKSIngressShareMessages != 88 {
		return nil, fmt.Errorf("KKLSS CPU backend requires the exact frozen 12-phase operation and 88-message CKS plan")
	}
	var params ckks.Parameters
	var err error
	if request.Parameter == guardedCompactP2GN16ParameterID {
		profile, profileErr := guardedCompactParameterProfileFor(request.Parameter)
		if profileErr != nil {
			return nil, profileErr
		}
		params = profile.base
	} else {
		params, err = fullResNet20Parameters("P2")
		if err != nil {
			return nil, err
		}
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), fullResNet20Parties, params.Q())
	if !ok {
		return nil, fmt.Errorf("P2 lacks levels for the two-party KKLSS refresh bridge")
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create KKLSS refresh CRS: %w", err)
	}
	context, err := newKKLSSN2Context(request.Parameter, fullResNet20RotationKeys())
	if err != nil {
		return nil, fmt.Errorf("generate public-KKLSS P2/n=2 evaluation material: %w", err)
	}
	shares, err := kklssIndividualRLWEShares(context.partySecrets)
	if err != nil {
		return nil, err
	}
	public := make([]*rlwe.PublicKey, len(context.partyIDs))
	for index, id := range context.partyIDs {
		public[index] = &context.public.GetPublicKey(id).PublicKey
	}
	adapter, err := newKKLSSLattigoAdapter(params, context, false)
	if err != nil {
		return nil, err
	}
	return &encryptedFullResNet20Executor{
		request: request, params: params, encoder: ckks.NewEncoder(params),
		shares: shares, public: public, aggregate: aggregateSecret(params.Parameters, shares),
		evaluator: adapter, kklssAdapter: adapter,
		minLevel: minLevel, logBound: logBound, refreshCRS: refreshCRS,
	}, nil
}

func kklssSwitchingKeyWireSize(label string, key *mkrlwe.SwitchingKey) (uint64, error) {
	if key == nil || len(key.Value) == 0 {
		return 0, fmt.Errorf("%s switching key is absent", label)
	}
	var total uint64
	for index := range key.Value {
		length := key.Value[index].GetDataLen(true)
		if length <= 0 {
			return 0, fmt.Errorf("%s PolyQP[%d] has invalid encoded length %d", label, index, length)
		}
		wire := make([]byte, length)
		written, err := key.Value[index].WriteTo(wire)
		if err != nil || written != length {
			return 0, fmt.Errorf("serialize %s PolyQP[%d]: wrote %d/%d: %w", label, index, written, length, err)
		}
		total += uint64(length)
	}
	return total, nil
}

func kklssEvaluationKeyWireSize(context *kklssP2N2PrimitiveContext) (payloads int, bytes uint64, err error) {
	if context == nil || context.relin == nil || context.rotation == nil {
		return 0, 0, fmt.Errorf("KKLSS evaluation material is absent")
	}
	add := func(label string, key *mkrlwe.SwitchingKey) error {
		length, sizeErr := kklssSwitchingKeyWireSize(label, key)
		if sizeErr != nil {
			return sizeErr
		}
		payloads++
		bytes += length
		return nil
	}
	for id, key := range context.relin.Value {
		if key == nil {
			return 0, 0, fmt.Errorf("relinearization key %q is nil", id)
		}
		for component, switchingKey := range key.Value {
			if err := add(fmt.Sprintf("relin[%s][%d]", id, component), switchingKey); err != nil {
				return 0, 0, err
			}
		}
	}
	for id, rotations := range context.rotation.Value {
		for rotation, key := range rotations {
			if key == nil {
				return 0, 0, fmt.Errorf("rotation key %q/%d is nil", id, rotation)
			}
			if err := add(fmt.Sprintf("rotation[%s][%d]", id, rotation), key.Value); err != nil {
				return 0, 0, err
			}
		}
	}
	if payloads == 0 || bytes == 0 {
		return 0, 0, fmt.Errorf("KKLSS evaluation-key byte coverage is empty")
	}
	return payloads, bytes, nil
}

func (backend *kklssResNet20CPUBackend) newPreparedTrial() (resNet20CPUPreparedTrial, error) {
	backend.state.requestOnce.Do(func() {
		backend.state.request, backend.state.requestErr = kklssCPURequestFromEnvironment()
	})
	if backend.state.requestErr != nil {
		return nil, backend.state.requestErr
	}
	return newKKLSSResNet20CPUTrial(backend, backend.state.request)
}

type kklssResNet20CPUTrial struct {
	backend                     *kklssResNet20CPUBackend
	request                     fullResNet20Request
	executor                    *encryptedFullResNet20Executor
	inputCiphertexts            []*ckks.Ciphertext
	current                     *ckks.Ciphertext
	reference                   []complex128
	nextPhase                   int
	active                      *kklssResNet20CPUSegment
	verified                    bool
	ownerCount                  int
	expectedPolynomialMulRelin  int
	communication               map[resNet20CPUStage]kklssCPUStageCommunication
	refreshShareSize            uint64
	evaluationKeyPayloads       int
	evaluationKeyBytes          uint64
	verificationMaxError        float64
	verificationEncryptedTop1   int
	verificationReferenceTop1   int
	verificationLabel           int
	verificationCountPass       bool
	verificationCorrectnessPass bool
	verificationClassPass       bool
}

func newKKLSSResNet20CPUTrial(backend *kklssResNet20CPUBackend, request fullResNet20Request) (*kklssResNet20CPUTrial, error) {
	executor, err := newKKLSSPerformanceExecutor(request)
	if err != nil {
		return nil, err
	}
	trial := &kklssResNet20CPUTrial{
		backend: backend, request: request, executor: executor, nextPhase: 1,
		communication: map[resNet20CPUStage]kklssCPUStageCommunication{
			resNet20CPUStem: {Stage: resNet20CPUStem}, resNet20CPUResidual: {Stage: resNet20CPUResidual}, resNet20CPUFullyConn: {Stage: resNet20CPUFullyConn},
		},
	}
	trial.evaluationKeyPayloads, trial.evaluationKeyBytes, err = kklssEvaluationKeyWireSize(executor.kklssAdapter.context)
	if err != nil {
		return nil, fmt.Errorf("measure offline KKLSS evaluation-key encodings: %w", err)
	}
	protocol := dckks.NewRefreshProtocol(executor.params, executor.logBound, collectiveRefreshSigmaSmudging)
	share := protocol.AllocateShare(executor.minLevel, executor.params.MaxLevel())
	wire, err := share.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("measure fixed refresh-share encoding outside timing: %w", err)
	}
	trial.refreshShareSize = uint64(len(wire))
	trial.inputCiphertexts = make([]*ckks.Ciphertext, resnetImageChannels)
	stemCommunication := trial.communication[resNet20CPUStem]
	for channel := 0; channel < resnetImageChannels; channel++ {
		packed := packReplicatedInputChannel(executor.params, request.Fixture.NormalizedRGB, channel)
		ciphertext, bytes, err := trial.encryptOwnerOutsideTiming(realResNetTailDataOwner, packed)
		if err != nil {
			return nil, fmt.Errorf("stage KKLSS input channel %d: %w", channel, err)
		}
		trial.inputCiphertexts[channel] = ciphertext
		stemCommunication.OwnerIngressCiphertexts++
		stemCommunication.OwnerIngressBytes += bytes
	}
	trial.communication[resNet20CPUStem] = stemCommunication
	return trial, nil
}

func (trial *kklssResNet20CPUTrial) encryptOwnerOutsideTiming(owner int, values []complex128) (*ckks.Ciphertext, uint64, error) {
	adapter := trial.executor.kklssAdapter
	if adapter == nil || owner < 0 || owner >= len(adapter.context.partyIDs) || len(values) != adapter.params.Slots() {
		return nil, 0, fmt.Errorf("invalid KKLSS staged owner=%d or slots=%d", owner, len(values))
	}
	message := mkckks.NewMessage(adapter.context.params)
	copy(message.Value, values)
	native := adapter.context.encryptor.EncryptMsgNew(message, adapter.context.public.GetPublicKey(adapter.context.partyIDs[owner]))
	wire, err := marshalKKLSSStagedCiphertext(adapter.context.params, native)
	if err != nil {
		return nil, 0, err
	}
	restored, err := unmarshalKKLSSStagedCiphertext(adapter.context.params, wire)
	if err != nil {
		return nil, 0, fmt.Errorf("verify KKLSS staged ciphertext: %w", err)
	}
	carrier, err := adapter.carrier(restored)
	if err != nil {
		return nil, 0, err
	}
	trial.ownerCount++
	return carrier, uint64(len(wire)), nil
}

func (trial *kklssResNet20CPUTrial) RunStem() error {
	return fmt.Errorf("KKLSS CPU trial is segmented; coarse RunStem is disabled")
}
func (trial *kklssResNet20CPUTrial) RunResidualBlocks() error {
	return fmt.Errorf("KKLSS CPU trial is segmented; coarse RunResidualBlocks is disabled")
}
func (trial *kklssResNet20CPUTrial) RunFullyConnected() error {
	return fmt.Errorf("KKLSS CPU trial is segmented; coarse RunFullyConnected is disabled")
}

func (trial *kklssResNet20CPUTrial) SegmentCount(stage resNet20CPUStage) int {
	switch stage {
	case resNet20CPUStem, resNet20CPUFullyConn:
		return 1
	case resNet20CPUResidual:
		return 9
	default:
		return 0
	}
}

func (trial *kklssResNet20CPUTrial) PrepareSegment(stage resNet20CPUStage, segmentIndex int) (resNet20CPUPreparedSegment, error) {
	if trial.verified || trial.active != nil {
		return nil, fmt.Errorf("KKLSS segment preparation is out of lifecycle order")
	}
	phaseIndex, err := paperMKFSegmentPhaseIndex(stage, segmentIndex)
	if err != nil {
		return nil, err
	}
	if phaseIndex != trial.nextPhase {
		return nil, fmt.Errorf("prepared phase index %d, want %d", phaseIndex, trial.nextPhase)
	}
	phase := trial.request.Plan.Phases[phaseIndex]
	segment := &kklssResNet20CPUSegment{trial: trial, stage: stage, phase: phase, refreshInputLevels: make([]int, 0, phase.Expected.CollectiveRefreshes)}
	segment.staged = &paperMKFStagedOwnerEvaluator{Evaluator: trial.executor.kklssAdapter}
	switch phase.Kind {
	case fullResNet20StemPhase:
		err = trial.prepareStemSegment(segment)
	case fullResNet20RegularPhase:
		err = trial.prepareRegularSegment(segment)
	case fullResNet20TransitionPhase:
		err = trial.prepareTransitionSegment(segment)
	case fullResNet20TailPhase:
		err = trial.prepareTailSegment(segment)
	default:
		err = fmt.Errorf("unsupported timed phase kind %q", phase.Kind)
	}
	if err != nil {
		return nil, fmt.Errorf("prepare %s outside timing: %w", phase.Name, err)
	}
	wantOwners := phase.Expected.ModelWeightCiphertexts + phase.Expected.ModelBiasCiphertexts
	if len(segment.staged.entries) != wantOwners {
		return nil, fmt.Errorf("%s staged %d KKLSS model ciphertexts, want %d", phase.Name, len(segment.staged.entries), wantOwners)
	}
	for _, site := range phase.ActivationSites {
		polynomial := referenceResNetBlockReLUPolynomial(site.Scale)
		if polynomial.Degree() != resnetReferenceReLUDegree || polynomial.Basis != ckks.ChebyshevBasis {
			return nil, fmt.Errorf("audit %s polynomial identity drifted", site.ID)
		}
		count, countErr := kklssCarrierBSGSProductCount(polynomial)
		if countErr != nil || count != p2gn16Degree59ProductsPerActivation {
			return nil, fmt.Errorf("audit %s BSGS product schedule=%d: %w", site.ID, count, countErr)
		}
		segment.expectedPolynomialMulRelin += count
	}
	trial.expectedPolynomialMulRelin += segment.expectedPolynomialMulRelin
	communication := trial.communication[stage]
	communication.OwnerIngressCiphertexts += len(segment.staged.entries)
	for _, entry := range segment.staged.entries {
		communication.OwnerIngressBytes += entry.wireBytes
	}
	communication.RefreshRounds += phase.Expected.CollectiveRefreshes
	communication.RefreshShareMessages += phase.Expected.CollectiveRefreshes * fullResNet20Parties
	communication.RefreshShareBytes += uint64(phase.Expected.CollectiveRefreshes*fullResNet20Parties) * trial.refreshShareSize
	trial.communication[stage] = communication
	segment.before = trial.executor.kklssOperationSnapshot()
	trial.active = segment
	return segment, nil
}

type kklssResNet20CPUSegment struct {
	trial                      *kklssResNet20CPUTrial
	stage                      resNet20CPUStage
	phase                      fullResNet20Phase
	staged                     *paperMKFStagedOwnerEvaluator
	referenceOutput            []complex128
	run                        func() (*ckks.Ciphertext, error)
	before                     kklssPhaseSnapshot
	expectedPolynomialMulRelin int
	refreshInputLevels         []int
	output                     *ckks.Ciphertext
	ran                        bool
	runOK                      bool
	released                   bool
}

func (segment *kklssResNet20CPUSegment) Run() error {
	if segment == nil || segment.trial == nil || segment.trial.active != segment || segment.ran || segment.released || segment.run == nil {
		return fmt.Errorf("invalid KKLSS timed segment lifecycle")
	}
	segment.ran = true
	output, err := segment.run()
	if err != nil {
		return err
	}
	// Keep structural/correctness auditing out of the measured interval.
	// Release validates the frozen operation counts immediately after the
	// runner has stopped the timer and before any next phase is prepared.
	segment.output = output
	segment.runOK = true
	return nil
}

func kklssCKSShareWireSize(params ckks.Parameters, level int) (uint64, error) {
	if level < 0 || level > params.MaxLevel() {
		return 0, fmt.Errorf("invalid CKS share level %d", level)
	}
	protocol := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
	share := protocol.AllocateShare(level)
	wire, err := share.MarshalBinary()
	if err != nil {
		return 0, err
	}
	return uint64(len(wire)), nil
}

func (segment *kklssResNet20CPUSegment) Release() error {
	if segment == nil || segment.trial == nil || segment.trial.active != segment || !segment.ran || segment.released {
		return fmt.Errorf("invalid KKLSS segment release order")
	}
	segment.released = true
	if segment.runOK {
		if err := segment.staged.requireConsumed(); err != nil {
			return err
		}
		if len(segment.refreshInputLevels) != segment.phase.Expected.CollectiveRefreshes {
			return fmt.Errorf("%s observed %d refresh levels, want %d", segment.phase.Name, len(segment.refreshInputLevels), segment.phase.Expected.CollectiveRefreshes)
		}
		after := segment.trial.executor.kklssOperationSnapshot()
		if after.mulRelin-segment.before.mulRelin != segment.phase.Expected.CiphertextProducts ||
			after.polynomialMulRelin-segment.before.polynomialMulRelin != segment.expectedPolynomialMulRelin ||
			after.rotations-segment.before.rotations != segment.phase.Expected.RotationCalls ||
			after.polynomials-segment.before.polynomials != segment.phase.Expected.Degree59Activations ||
			after.refreshes-segment.before.refreshes != segment.phase.Expected.CollectiveRefreshes {
			return fmt.Errorf("%s KKLSS online count mismatch: networkMul=%d/%d polyMul=%d/%d rotations=%d/%d poly=%d/%d refresh=%d/%d", segment.phase.Name,
				after.mulRelin-segment.before.mulRelin, segment.phase.Expected.CiphertextProducts,
				after.polynomialMulRelin-segment.before.polynomialMulRelin, segment.expectedPolynomialMulRelin,
				after.rotations-segment.before.rotations, segment.phase.Expected.RotationCalls,
				after.polynomials-segment.before.polynomials, segment.phase.Expected.Degree59Activations,
				after.refreshes-segment.before.refreshes, segment.phase.Expected.CollectiveRefreshes)
		}
		segment.trial.current = segment.output
	}
	communication := segment.trial.communication[segment.stage]
	for _, level := range segment.refreshInputLevels {
		shareBytes, err := kklssCKSShareWireSize(segment.trial.executor.params, level)
		if err != nil {
			return fmt.Errorf("measure %s CKS share bytes outside timing: %w", segment.phase.Name, err)
		}
		messages := fullResNet20Parties * fullResNet20Parties
		communication.CKSShareMessages += messages
		communication.CKSShareBytes += uint64(messages) * shareBytes
	}
	segment.trial.communication[segment.stage] = communication
	if segment.runOK {
		segment.trial.reference = segment.referenceOutput
		segment.trial.nextPhase++
	}
	if segment.phase.Kind == fullResNet20StemPhase {
		segment.trial.inputCiphertexts = nil
	}
	for index := range segment.staged.entries {
		segment.staged.entries[index].ciphertext = nil
	}
	segment.staged.entries = nil
	segment.referenceOutput = nil
	segment.output = nil
	segment.run = nil
	segment.refreshInputLevels = nil
	segment.trial.active = nil
	runtime.GC()
	return nil
}

func (trial *kklssResNet20CPUTrial) stageOwner(segment *kklssResNet20CPUSegment, label string, values []complex128) error {
	ciphertext, bytes, err := trial.encryptOwnerOutsideTiming(realResNetTailModelOwner, values)
	if err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	segment.staged.entries = append(segment.staged.entries, paperMKFStagedOwnerCiphertext{label: label, ciphertext: ciphertext, wireBytes: bytes})
	return nil
}

// timedRefresh contains the real KKLSS two-source/four-CKS-share bridge and
// dckks refresh. It records only the already-public ciphertext level into a
// preallocated slice; serialization happens later in untimed Release.
func (trial *kklssResNet20CPUTrial) timedRefresh(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
	if trial.active == nil || len(trial.active.refreshInputLevels) >= cap(trial.active.refreshInputLevels) {
		return nil, fmt.Errorf("KKLSS refresh is outside a prepared segment or exceeds the frozen count")
	}
	trial.active.refreshInputLevels = append(trial.active.refreshInputLevels, input.Level())
	return trial.executor.kklssAdapter.collectiveRefresh(input, trial.executor.shares, trial.executor.minLevel, trial.executor.logBound, trial.executor.refreshCRS)
}

func kklssChebyshevMulRelinCount(polynomial *ckks.Polynomial) (int, error) {
	if polynomial == nil || polynomial.Degree() != resnetReferenceReLUDegree || polynomial.Basis != ckks.ChebyshevBasis {
		return 0, fmt.Errorf("expected frozen degree-%d Chebyshev polynomial", resnetReferenceReLUDegree)
	}
	seen := map[int]bool{1: true}
	var countPower func(int) (int, error)
	countPower = func(n int) (int, error) {
		if n < 1 {
			return 0, fmt.Errorf("invalid Chebyshev power %d", n)
		}
		if seen[n] {
			return 0, nil
		}
		var a, b, c int
		if n&(n-1) == 0 {
			a, b = n/2, n/2
		} else {
			k := int(math.Ceil(math.Log2(float64(n)))) - 1
			a = (1 << k) - 1
			b = n + 1 - (1 << k)
			c = int(math.Abs(float64(a - b)))
		}
		left, err := countPower(a)
		if err != nil {
			return 0, err
		}
		right, err := countPower(b)
		if err != nil {
			return 0, err
		}
		correction := 0
		if c != 0 {
			correction, err = countPower(c)
			if err != nil {
				return 0, err
			}
		}
		seen[n] = true
		return left + right + correction + 1, nil
	}
	total := 0
	for degree := 1; degree < len(polynomial.Coeffs); degree++ {
		coefficient := polynomial.Coeffs[degree]
		if math.Abs(real(coefficient)) <= ckks.IsNegligbleThreshold && math.Abs(imag(coefficient)) <= ckks.IsNegligbleThreshold {
			continue
		}
		count, err := countPower(degree)
		if err != nil {
			return 0, err
		}
		total += count
	}
	if total <= 0 {
		return 0, fmt.Errorf("KKLSS polynomial schedule has no MulRelin nodes")
	}
	return total, nil
}

func (trial *kklssResNet20CPUTrial) Verify() error {
	if trial.verified || trial.active != nil || trial.nextPhase != len(trial.request.Plan.Phases) || trial.current == nil || len(trial.reference) != trial.executor.params.Slots() {
		return fmt.Errorf("KKLSS trial is incomplete at untimed verification: nextPhase=%d active=%t", trial.nextPhase, trial.active != nil)
	}
	trial.verified = true
	counts := trial.executor.kklssOperationSnapshot()
	trial.verificationCountPass = trial.ownerCount == 6201 && counts.mulRelin == 6172 && trial.expectedPolynomialMulRelin == 323 && counts.polynomialMulRelin == 323 && counts.rotations == 6499 && counts.polynomials == 19 && counts.refreshes == 22
	if !trial.verificationCountPass {
		return fmt.Errorf("KKLSS final structural counts mismatch: owners=%d networkMul=%d polyMul=%d/%d rotations=%d degree59=%d refresh=%d", trial.ownerCount, counts.mulRelin, counts.polynomialMulRelin, trial.expectedPolynomialMulRelin, counts.rotations, counts.polynomials, counts.refreshes)
	}
	oracle, err := trial.executor.kklssAdapter.oracleCarrier(trial.current)
	if err != nil {
		return err
	}
	decoded := trial.executor.encoder.Decode(ckks.NewDecryptor(trial.executor.params, trial.executor.aggregate).DecryptNew(oracle), trial.executor.params.LogSlots())
	if len(decoded) < len(trial.reference) {
		return fmt.Errorf("decoded %d slots, want %d", len(decoded), len(trial.reference))
	}
	maxError := 0.0
	for index, want := range trial.reference {
		delta := cmplx.Abs(decoded[index] - want)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			return fmt.Errorf("non-finite final error at slot %d", index)
		}
		if delta > maxError {
			maxError = delta
		}
	}
	trial.verificationMaxError = maxError
	trial.verificationCorrectnessPass = maxError < fullResNet20FunctionalGate
	if !trial.verificationCorrectnessPass {
		return fmt.Errorf("untimed KKLSS final oracle max error=%g exceeds %g", maxError, fullResNet20FunctionalGate)
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
	trial.verificationEncryptedTop1 = encryptedTop1
	trial.verificationReferenceTop1 = referenceTop1
	trial.verificationLabel = label
	trial.verificationClassPass = encryptedTop1 == label && referenceTop1 == label && encryptedTop1 == referenceTop1
	if !trial.verificationClassPass {
		return fmt.Errorf("untimed KKLSS classification mismatch: encrypted=%d reference=%d label=%d", encryptedTop1, referenceTop1, label)
	}
	ownerTotal, refreshRounds, cksMessages, refreshMessages := 0, 0, 0, 0
	var ownerBytes, cksBytes, refreshBytes uint64
	for _, stage := range resNet20CPUStages {
		evidence := trial.communication[stage]
		wantOwners, wantRefreshes, ok := kklssExpectedStageCommunication(stage)
		wantCKSMessages := wantRefreshes * fullResNet20Parties * fullResNet20Parties
		wantRefreshMessages := wantRefreshes * fullResNet20Parties
		if !ok || evidence.OwnerIngressCiphertexts != wantOwners || evidence.RefreshRounds != wantRefreshes || evidence.CKSShareMessages != wantCKSMessages || evidence.RefreshShareMessages != wantRefreshMessages {
			return fmt.Errorf("KKLSS %s communication mismatch: owners=%d/%d refresh=%d/%d CKS=%d/%d refreshMsgs=%d/%d", stage, evidence.OwnerIngressCiphertexts, wantOwners, evidence.RefreshRounds, wantRefreshes, evidence.CKSShareMessages, wantCKSMessages, evidence.RefreshShareMessages, wantRefreshMessages)
		}
		if evidence.OwnerIngressCiphertexts > 0 && evidence.OwnerIngressBytes == 0 {
			return fmt.Errorf("KKLSS %s lacks owner-ciphertext byte evidence", stage)
		}
		if evidence.CKSShareMessages > 0 && evidence.CKSShareBytes == 0 {
			return fmt.Errorf("KKLSS %s lacks CKS-share byte evidence", stage)
		}
		if evidence.RefreshShareMessages > 0 && evidence.RefreshShareBytes != uint64(evidence.RefreshShareMessages)*trial.refreshShareSize {
			return fmt.Errorf("KKLSS %s refresh-share bytes=%d, want %d", stage, evidence.RefreshShareBytes, uint64(evidence.RefreshShareMessages)*trial.refreshShareSize)
		}
		ownerTotal += evidence.OwnerIngressCiphertexts
		ownerBytes += evidence.OwnerIngressBytes
		refreshRounds += evidence.RefreshRounds
		cksMessages += evidence.CKSShareMessages
		cksBytes += evidence.CKSShareBytes
		refreshMessages += evidence.RefreshShareMessages
		refreshBytes += evidence.RefreshShareBytes
	}
	if ownerTotal != 6201 || ownerBytes == 0 || refreshRounds != 22 || cksMessages != 88 || cksBytes == 0 || refreshMessages != 44 || refreshBytes != uint64(44)*trial.refreshShareSize {
		return fmt.Errorf("KKLSS communication totals mismatch: owners=%d bytes=%d refresh=%d CKS=%d/%dB refreshMsgs=%d/%dB", ownerTotal, ownerBytes, refreshRounds, cksMessages, cksBytes, refreshMessages, refreshBytes)
	}
	if trial.evaluationKeyPayloads <= 0 || trial.evaluationKeyBytes == 0 {
		return fmt.Errorf("KKLSS evaluation-key byte evidence is empty")
	}
	trial.backend.recordCommunication(kklssCPUTrialCommunication{
		EvaluationKeyPayloads: trial.evaluationKeyPayloads,
		EvaluationKeyBytes:    trial.evaluationKeyBytes,
		Stages:                trial.communication,
	})
	return nil
}

func (trial *kklssResNet20CPUTrial) prepareStemSegment(segment *kklssResNet20CPUSegment) error {
	phase := segment.phase
	if len(trial.inputCiphertexts) != resnetImageChannels || len(phase.ActivationSites) != 1 {
		return fmt.Errorf("stem input or activation contract is incomplete")
	}
	scale := phase.ActivationSites[0].Scale
	weights, err := loadRealResNetStemWeightsAtScale(trial.request.WeightsDirectory, scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				label := fmt.Sprintf("%s/weight/ic=%d/y=%d/x=%d", phase.Name, inputChannel, kernelY, kernelX)
				if err := trial.stageOwner(segment, label, weights.masks[inputChannel][kernelY][kernelX]); err != nil {
					return err
				}
			}
		}
	}
	if err := trial.stageOwner(segment, phase.Name+"/bias", weights.biasPacked); err != nil {
		return err
	}
	plain := plainRealFoldedResNetStem(trial.request.Fixture.NormalizedRGB, weights)
	polynomial := referenceResNetBlockReLUPolynomial(scale)
	referenceOutput, _, err := plainReferenceResNetBlockReLU(plain, polynomial, scale)
	if err != nil {
		return err
	}
	segment.referenceOutput = referenceOutput
	segment.run = func() (*ckks.Ciphertext, error) {
		conv, err := paperMKFTimedStemConv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, trial.inputCiphertexts, phase.Name)
		if err != nil {
			return nil, err
		}
		refreshed, err := trial.timedRefresh(conv)
		if err != nil {
			return nil, err
		}
		return encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshed, polynomial, trial.executor.params.Scale())
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) prepareRegularSegment(segment *kklssResNet20CPUSegment) error {
	phase := segment.phase
	if trial.current == nil || len(trial.reference) != trial.executor.params.Slots() || len(phase.ActivationSites) != 2 {
		return fmt.Errorf("regular block %s has no staged input/reference", phase.Name)
	}
	skipReference := append([]complex128(nil), trial.reference...)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	var runConv1, runConv2 func(*ckks.Ciphertext) (*ckks.Ciphertext, error)
	var plainConv1, plainConv2 []complex128
	switch phase.Layer {
	case 1:
		conv1, err := loadRealResNetBlockConvWeights(trial.request.WeightsDirectory, phase.Conv1Prefix, phase.ActivationSites[0].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		conv2, err := loadRealResNetBlockConvWeights(trial.request.WeightsDirectory, phase.Conv2Prefix, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		if err := trial.stageLayer1Weights(segment, phase.Name+"/conv1", conv1); err != nil {
			return err
		}
		plainConv1 = plainRealResNetBlockConv(trial.reference, conv1)
		activation1, _, err := plainReferenceResNetBlockReLU(plainConv1, poly1, phase.ActivationSites[0].Scale)
		if err != nil {
			return err
		}
		if err := trial.stageLayer1Weights(segment, phase.Name+"/conv2", conv2); err != nil {
			return err
		}
		plainConv2 = plainRealResNetBlockConv(activation1, conv2)
		runConv1 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer1Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv1")
		}
		runConv2 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer1Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv2")
		}
	case 2:
		conv1, err := loadRealStage2ConvWeights(trial.request.WeightsDirectory, phase.Conv1Prefix, phase.ActivationSites[0].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		conv2, err := loadRealStage2ConvWeights(trial.request.WeightsDirectory, phase.Conv2Prefix, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		if err := trial.stageLayer2ConvWeights(segment, phase.Name+"/conv1", conv1); err != nil {
			return err
		}
		plainConv1 = plainRealStage2Conv(trial.reference, conv1)
		activation1, _, err := plainReferenceResNetBlockReLU(plainConv1, poly1, phase.ActivationSites[0].Scale)
		if err != nil {
			return err
		}
		if err := trial.stageLayer2ConvWeights(segment, phase.Name+"/conv2", conv2); err != nil {
			return err
		}
		plainConv2 = plainRealStage2Conv(activation1, conv2)
		runConv1 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer2Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv1")
		}
		runConv2 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer2Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv2")
		}
	case 3:
		conv1, err := loadRealStage3SameWidthConvWeights(trial.request.WeightsDirectory, phase.Conv1Prefix, phase.ActivationSites[0].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		conv2, err := loadRealStage3SameWidthConvWeights(trial.request.WeightsDirectory, phase.Conv2Prefix, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
		if err != nil {
			return err
		}
		if err := trial.stageLayer3ConvWeights(segment, phase.Name+"/conv1", conv1); err != nil {
			return err
		}
		plainConv1 = plainRealStage3Conv(trial.reference, conv1)
		activation1, _, err := plainReferenceResNetBlockReLU(plainConv1, poly1, phase.ActivationSites[0].Scale)
		if err != nil {
			return err
		}
		if err := trial.stageLayer3ConvWeights(segment, phase.Name+"/conv2", conv2); err != nil {
			return err
		}
		plainConv2 = plainRealStage3Conv(activation1, conv2)
		runConv1 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer3Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv1")
		}
		runConv2 = func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
			return paperMKFTimedLayer3Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, input, phase.Name+"/conv2")
		}
	default:
		return fmt.Errorf("unsupported regular layer %d", phase.Layer)
	}
	plainResidual := addPlainRealResNetBlockSkip(plainConv2, skipReference, phase.ActivationSites[1].Scale)
	referenceOutput, _, err := plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return err
	}
	segment.referenceOutput = referenceOutput
	input := trial.current
	segment.run = func() (*ckks.Ciphertext, error) {
		skipCiphertext := input
		conv1, err := runConv1(input)
		if err != nil {
			return nil, err
		}
		refreshed1, err := trial.timedRefresh(conv1)
		if err != nil {
			return nil, err
		}
		activation1, err := encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshed1, poly1, trial.executor.params.Scale())
		if err != nil {
			return nil, err
		}
		conv2, err := runConv2(activation1)
		if err != nil {
			return nil, err
		}
		if err := addScaledRealResNetBlockSkip(trial.executor.params, trial.executor.kklssAdapter, conv2, skipCiphertext, phase.ActivationSites[1].Scale); err != nil {
			return nil, err
		}
		refreshed2, err := trial.timedRefresh(conv2)
		if err != nil {
			return nil, err
		}
		output, err := encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshed2, poly2, trial.executor.params.Scale())
		if err != nil {
			return nil, err
		}
		if len(phase.RequiredBoundaries) == 8 {
			return trial.timedRefresh(output)
		}
		return output, nil
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) stageLayer1Weights(segment *kklssResNet20CPUSegment, prefix string, weights *realResNetBlockConvWeights) error {
	for diagonal := 0; diagonal < realResNetBlockChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				if err := trial.stageOwner(segment, fmt.Sprintf("%s/d=%d/y=%d/x=%d", prefix, diagonal, kernelY, kernelX), weights.masks[diagonal][kernelY][kernelX]); err != nil {
					return err
				}
			}
		}
	}
	return trial.stageOwner(segment, prefix+"/bias", weights.packedBias)
}

func (trial *kklssResNet20CPUTrial) stageLayer2ConvWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage2ConvWeights) error {
	for diagonal := 0; diagonal < realStage2OutputChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				if err := trial.stageOwner(segment, fmt.Sprintf("%s/d=%d/y=%d/x=%d", prefix, diagonal, kernelY, kernelX), weights.masks[diagonal][kernelY][kernelX]); err != nil {
					return err
				}
			}
		}
	}
	return trial.stageOwner(segment, prefix+"/bias", weights.packedBias)
}

func (trial *kklssResNet20CPUTrial) stageLayer3ConvWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage3ConvWeights) error {
	for diagonal := 0; diagonal < realStage3OutputChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				if err := trial.stageOwner(segment, fmt.Sprintf("%s/d=%d/y=%d/x=%d", prefix, diagonal, kernelY, kernelX), weights.masks[diagonal][kernelY][kernelX]); err != nil {
					return err
				}
			}
		}
	}
	return trial.stageOwner(segment, prefix+"/bias", weights.packedBias)
}

func (trial *kklssResNet20CPUTrial) prepareTransitionSegment(segment *kklssResNet20CPUSegment) error {
	if trial.current == nil || len(trial.reference) != trial.executor.params.Slots() || len(segment.phase.ActivationSites) != 2 {
		return fmt.Errorf("transition block %s has no staged input/reference", segment.phase.Name)
	}
	if segment.phase.Layer == 2 {
		return trial.prepareStage2TransitionSegment(segment)
	}
	if segment.phase.Layer == 3 {
		return trial.prepareStage3TransitionSegment(segment)
	}
	return fmt.Errorf("unsupported transition layer %d", segment.phase.Layer)
}

func (trial *kklssResNet20CPUTrial) prepareStage2TransitionSegment(segment *kklssResNet20CPUSegment) error {
	phase := segment.phase
	mainWeights, err := loadRealStage2StrideWeights(trial.request.WeightsDirectory, phase.ActivationSites[0].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	skipWeights, err := loadRealStage2SkipWeights(trial.request.WeightsDirectory, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	conv2Weights, err := loadRealStage2ConvWeights(trial.request.WeightsDirectory, phase.Conv2Prefix, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	if err := trial.stageStage2StrideWeights(segment, phase.Name+"/main", mainWeights); err != nil {
		return err
	}
	if err := trial.stageStage2SkipWeights(segment, phase.Name+"/skip", skipWeights); err != nil {
		return err
	}
	if err := trial.stageLayer2ConvWeights(segment, phase.Name+"/conv2", conv2Weights); err != nil {
		return err
	}
	plainMain := plainRealStage2StrideConv(trial.reference, mainWeights)
	plainSkip := plainRealStage2Skip(trial.reference, skipWeights)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	plainActivation1, _, err := plainReferenceResNetBlockReLU(plainMain, poly1, phase.ActivationSites[0].Scale)
	if err != nil {
		return err
	}
	plainConv2 := plainRealStage2Conv(plainActivation1, conv2Weights)
	plainResidual := stage2AddPlainBranches(plainConv2, plainSkip)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	segment.referenceOutput, _, err = plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return err
	}
	packing := newPaperMKFStage2PackingMaterial(trial.executor.params, trial.executor.encoder, trial.executor.params.MaxLevel()-1)
	input := trial.current
	segment.run = func() (*ckks.Ciphertext, error) {
		refreshedInput, err := trial.timedRefresh(input)
		if err != nil {
			return nil, err
		}
		mainHalves, err := paperMKFTimedStage2StrideConv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, refreshedInput, phase.Name+"/main")
		if err != nil {
			return nil, err
		}
		skipHalves, err := paperMKFTimedStage2Skip(trial.executor.params, trial.executor.kklssAdapter, segment.staged, refreshedInput, phase.Name+"/skip")
		if err != nil {
			return nil, err
		}
		packedMain, err := paperMKFTimedPackStage2Halves(trial.executor.kklssAdapter, mainHalves[0], mainHalves[1], packing)
		if err != nil {
			return nil, err
		}
		packedSkip, err := paperMKFTimedPackStage2Halves(trial.executor.kklssAdapter, skipHalves[0], skipHalves[1], packing)
		if err != nil {
			return nil, err
		}
		if err := trial.executor.kklssAdapter.Rescale(packedMain, trial.executor.params.Scale(), packedMain); err != nil {
			return nil, err
		}
		if err := trial.executor.kklssAdapter.Rescale(packedSkip, trial.executor.params.Scale(), packedSkip); err != nil {
			return nil, err
		}
		refreshedMain, err := trial.timedRefresh(packedMain)
		if err != nil {
			return nil, err
		}
		activation1, err := encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshedMain, poly1, trial.executor.params.Scale())
		if err != nil {
			return nil, err
		}
		conv2, err := paperMKFTimedLayer2Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, activation1, phase.Name+"/conv2")
		if err != nil {
			return nil, err
		}
		if packedSkip.Level() < conv2.Level() {
			return nil, fmt.Errorf("stage2 skip level %d below conv2 %d", packedSkip.Level(), conv2.Level())
		}
		trial.executor.kklssAdapter.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
		trial.executor.kklssAdapter.Add(conv2, packedSkip, conv2)
		refreshedResidual, err := trial.timedRefresh(conv2)
		if err != nil {
			return nil, err
		}
		return encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshedResidual, poly2, trial.executor.params.Scale())
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) stageStage2StrideWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage2StrideWeights) error {
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			for kernelY := 0; kernelY < 3; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/d=%d/y=%d/x=%d", prefix, group, diagonal, kernelY, kernelX), weights.masks[group][diagonal][kernelY][kernelX]); err != nil {
						return err
					}
				}
			}
		}
		if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/bias", prefix, group), weights.packedBias[group]); err != nil {
			return err
		}
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) stageStage2SkipWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage2SkipWeights) error {
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/d=%d", prefix, group, diagonal), weights.masks[group][diagonal]); err != nil {
				return err
			}
		}
		if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/bias", prefix, group), weights.packedBias[group]); err != nil {
			return err
		}
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) prepareStage3TransitionSegment(segment *kklssResNet20CPUSegment) error {
	phase := segment.phase
	mainWeights, err := loadRealStage3StrideWeights(trial.request.WeightsDirectory, phase.ActivationSites[0].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	skipWeights, err := loadRealStage3SkipWeights(trial.request.WeightsDirectory, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	conv2Weights, err := loadRealStage3ConvWeights(trial.request.WeightsDirectory, phase.ActivationSites[1].Scale, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	if err := trial.stageStage3StrideWeights(segment, phase.Name+"/main", mainWeights); err != nil {
		return err
	}
	if err := trial.stageStage3SkipWeights(segment, phase.Name+"/skip", skipWeights); err != nil {
		return err
	}
	if err := trial.stageLayer3ConvWeights(segment, phase.Name+"/conv2", conv2Weights); err != nil {
		return err
	}
	plainMain := plainRealStage3StrideConv(trial.reference, mainWeights)
	plainSkip := plainRealStage3Skip(trial.reference, skipWeights)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	plainActivation1, _, err := plainReferenceResNetBlockReLU(plainMain, poly1, phase.ActivationSites[0].Scale)
	if err != nil {
		return err
	}
	plainConv2 := plainRealStage3Conv(plainActivation1, conv2Weights)
	plainResidual := stage3AddPlainBranches(plainConv2, plainSkip)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	segment.referenceOutput, _, err = plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return err
	}
	packing := newPaperMKFStage3PackingMaterial(trial.executor.params, trial.executor.encoder, trial.executor.params.MaxLevel()-1)
	input := trial.current
	segment.run = func() (*ckks.Ciphertext, error) {
		refreshedInput, err := trial.timedRefresh(input)
		if err != nil {
			return nil, err
		}
		mainHalves, err := paperMKFTimedStage3StrideConv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, refreshedInput, phase.Name+"/main")
		if err != nil {
			return nil, err
		}
		skipHalves, err := paperMKFTimedStage3Skip(trial.executor.params, trial.executor.kklssAdapter, segment.staged, refreshedInput, phase.Name+"/skip")
		if err != nil {
			return nil, err
		}
		packedMain, err := paperMKFTimedPackStage3Halves(trial.executor.kklssAdapter, mainHalves[0], mainHalves[1], packing)
		if err != nil {
			return nil, err
		}
		packedSkip, err := paperMKFTimedPackStage3Halves(trial.executor.kklssAdapter, skipHalves[0], skipHalves[1], packing)
		if err != nil {
			return nil, err
		}
		if err := trial.executor.kklssAdapter.Rescale(packedMain, trial.executor.params.Scale(), packedMain); err != nil {
			return nil, err
		}
		if err := trial.executor.kklssAdapter.Rescale(packedSkip, trial.executor.params.Scale(), packedSkip); err != nil {
			return nil, err
		}
		refreshedMain, err := trial.timedRefresh(packedMain)
		if err != nil {
			return nil, err
		}
		activation1, err := encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshedMain, poly1, trial.executor.params.Scale())
		if err != nil {
			return nil, err
		}
		conv2, err := paperMKFTimedLayer3Conv(trial.executor.params, trial.executor.kklssAdapter, segment.staged, activation1, phase.Name+"/conv2")
		if err != nil {
			return nil, err
		}
		if packedSkip.Level() < conv2.Level() {
			return nil, fmt.Errorf("stage3 skip level %d below conv2 %d", packedSkip.Level(), conv2.Level())
		}
		trial.executor.kklssAdapter.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
		trial.executor.kklssAdapter.Add(conv2, packedSkip, conv2)
		refreshedResidual, err := trial.timedRefresh(conv2)
		if err != nil {
			return nil, err
		}
		return encryptedReferenceResNetBlockReLU(trial.executor.kklssAdapter, refreshedResidual, poly2, trial.executor.params.Scale())
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) stageStage3StrideWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage3StrideWeights) error {
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			for kernelY := 0; kernelY < 3; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/d=%d/y=%d/x=%d", prefix, group, diagonal, kernelY, kernelX), weights.masks[group][diagonal][kernelY][kernelX]); err != nil {
						return err
					}
				}
			}
		}
		if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/bias", prefix, group), weights.packedBias[group]); err != nil {
			return err
		}
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) stageStage3SkipWeights(segment *kklssResNet20CPUSegment, prefix string, weights *realStage3SkipWeights) error {
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/d=%d", prefix, group, diagonal), weights.masks[group][diagonal]); err != nil {
				return err
			}
		}
		if err := trial.stageOwner(segment, fmt.Sprintf("%s/g=%d/bias", prefix, group), weights.packedBias[group]); err != nil {
			return err
		}
	}
	return nil
}

func (trial *kklssResNet20CPUTrial) prepareTailSegment(segment *kklssResNet20CPUSegment) error {
	if trial.current == nil || len(trial.reference) != trial.executor.params.Slots() {
		return fmt.Errorf("tail has no staged input/reference")
	}
	weights, err := loadRealResNetTailWeights(trial.request.WeightsDirectory, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	bias, err := loadCheckpointRealResNetTailBias(trial.request.CheckpointBiasPath, trial.executor.params.Slots())
	if err != nil {
		return err
	}
	if err := trial.stageOwner(segment, segment.phase.Name+"/weight", weights.packed); err != nil {
		return err
	}
	if err := trial.stageOwner(segment, segment.phase.Name+"/bias", bias.packed); err != nil {
		return err
	}
	_, _, segment.referenceOutput = fullResNet20PlainTailSchedule(trial.reference, weights, bias)
	gapMask := make([]complex128, trial.executor.params.Slots())
	for copyIndex := 0; copyIndex < trial.executor.params.Slots()/realResNetTailReferenceSlots; copyIndex++ {
		for channel := 0; channel < realResNetTailChannels; channel++ {
			gapMask[copyIndex*realResNetTailReferenceSlots+channel*realResNetTailSpatialSlots] = complex(1.0/realResNetTailSpatialSlots, 0)
		}
	}
	gapPlaintext := trial.executor.encoder.EncodeNTTAtLvlNew(trial.current.Level(), gapMask, trial.executor.params.LogSlots())
	input := trial.current
	phaseName := segment.phase.Name
	segment.run = func() (*ckks.Ciphertext, error) {
		gap := encryptedRealResNetTailRotSum(trial.executor.kklssAdapter, input, realResNetTailSpatialSlots)
		if gap.Level() != gapPlaintext.Level() {
			return nil, fmt.Errorf("tail pre-encoded GAP mask level=%d, input=%d", gapPlaintext.Level(), gap.Level())
		}
		gap = trial.executor.kklssAdapter.MulNew(gap, gapPlaintext)
		if err := trial.executor.kklssAdapter.Rescale(gap, trial.executor.params.Scale(), gap); err != nil {
			return nil, err
		}
		repeated := encryptedRealResNetTailRotSum(trial.executor.kklssAdapter, gap, realResNetTailRepeatWidth)
		repeated = trial.executor.kklssAdapter.RotateNew(repeated, -realResNetTailRepeatWidth+1)
		weight, err := segment.staged.take(phaseName + "/weight")
		if err != nil {
			return nil, err
		}
		trial.executor.kklssAdapter.DropLevel(weight, weight.Level()-repeated.Level())
		classified := trial.executor.kklssAdapter.MulRelinNew(repeated, weight)
		if err := trial.executor.kklssAdapter.Rescale(classified, trial.executor.params.Scale(), classified); err != nil {
			return nil, err
		}
		classified = encryptedRealResNetTailPaddedRotSum(trial.executor.kklssAdapter, classified)
		biasCiphertext, err := segment.staged.take(phaseName + "/bias")
		if err != nil {
			return nil, err
		}
		trial.executor.kklssAdapter.DropLevel(biasCiphertext, biasCiphertext.Level()-classified.Level())
		trial.executor.kklssAdapter.Add(classified, biasCiphertext, classified)
		return classified, nil
	}
	return nil
}

var (
	_ resNet20CPUBackend                       = (*kklssResNet20CPUBackend)(nil)
	_ resNet20CPUCommunicationArtifactReporter = (*kklssResNet20CPUBackend)(nil)
	_ resNet20CPUSegmentedTrial                = (*kklssResNet20CPUTrial)(nil)
	_ resNet20CPUPreparedSegment               = (*kklssResNet20CPUSegment)(nil)
	_ *drlwe.CKSShare                          = nil
)

func TestKKLSSCPUPreparedBackendStaticContract(t *testing.T) {
	backendInterface, err := newKKLSSResNet20CPUBackend()
	if err != nil {
		t.Fatal(err)
	}
	backend := backendInterface.(*kklssResNet20CPUBackend)
	if backend.state == nil || backend.Implementation() != "KKLSS-CPU" || backend.ParameterID() != "P2" || backend.Parties() != 2 {
		t.Fatalf("invalid prepared KKLSS backend identity/state: %+v", backend)
	}
	policy := backend.CommunicationBytePolicy()
	if policy.CKSIngressShareMessages != 88 || policy.OwnerIngressEncoding == "" || policy.CKSShareEncoding == "" || policy.RefreshShareEncoding == "" || policy.EvaluationKeyEncoding == "" || policy.OnlineTimingScope == "" {
		t.Fatalf("incomplete KKLSS communication policy: %+v", policy)
	}
	owners, refreshes := 0, 0
	trial := new(kklssResNet20CPUTrial)
	for _, stage := range resNet20CPUStages {
		stageOwners, stageRefreshes, ok := kklssExpectedStageCommunication(stage)
		if !ok || trial.SegmentCount(stage) <= 0 {
			t.Fatalf("missing KKLSS stage/segment contract for %s", stage)
		}
		owners += stageOwners
		refreshes += stageRefreshes
	}
	if owners != 6201 || refreshes != 22 {
		t.Fatalf("KKLSS static communication totals=%d owners/%d refreshes, want 6201/22", owners, refreshes)
	}
	polynomial := referenceResNetBlockReLUPolynomial(1)
	if polynomial.Degree() != resnetReferenceReLUDegree || polynomial.Basis != ckks.ChebyshevBasis {
		t.Fatal("frozen degree-59 polynomial identity drifted")
	}
	t.Logf("matched public-KKLSS BSGS degree-59 schedule uses %d internal MulRelinNew calls per activation; kept separate from 6172 network products", p2gn16Degree59ProductsPerActivation)
}
