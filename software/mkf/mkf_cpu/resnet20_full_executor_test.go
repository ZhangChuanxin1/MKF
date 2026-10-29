package mkfcpu

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"sort"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// The frozen LowMemoryFHEResNet20 schedule uses 16384 slots end to end. P2
// and P2G already have LogSlots=14. P2G is the guarded-ingress profile: its
// final Q prime is consumed exactly once while converting a fresh sparse
// n+1 ciphertext to the compact aggregate domain. PN16QP1761 has a larger
// native slot capacity, so
// the P3 interactive-refresh candidate deliberately lowers only LogSlots to
// 14; its ring and Q/P chains remain unchanged. This is a packing profile,
// not a new security parameter claim.
func fullResNet20Parameters(name string) (ckks.Parameters, error) {
	switch name {
	case "P2":
		return CKKSParameters(name)
	case guardedCompactP2GParameterID, guardedCompactP2GN16ParameterID:
		profile, err := guardedCompactParameterProfileFor(name)
		if err != nil {
			return ckks.Parameters{}, err
		}
		return profile.guarded, nil
	case "P3IRCandidate":
		literal := ckks.PN16QP1761
		literal.LogSlots = 14
		return ckks.NewParametersFromLiteral(literal)
	default:
		return ckks.Parameters{}, fmt.Errorf("full ResNet-20 executor does not support parameter %q", name)
	}
}

func fullResNet20RotationKeys() []int {
	set := make(map[int]struct{})
	add := func(values ...int) {
		for _, value := range values {
			if value != 0 {
				set[value] = struct{}{}
			}
		}
	}
	add(-resnetImageWidth, -1, 1, resnetImageWidth)
	add(realResNetBlockRotationKeys()...)
	add(stage2RotationKeys()...)
	add(stage3RotationKeys()...)
	add(realResNetTailRotationKeys()...)
	rotations := make([]int, 0, len(set))
	for rotation := range set {
		rotations = append(rotations, rotation)
	}
	sort.Ints(rotations)
	return rotations
}

type encryptedFullResNet20Executor struct {
	request   fullResNet20Request
	params    ckks.Parameters
	encoder   ckks.Encoder
	shares    []*rlwe.SecretKey
	public    []*rlwe.PublicKey
	aggregate *rlwe.SecretKey
	evaluator ckks.Evaluator
	// paperAdapter is non-nil only for the opt-in proposed-MKF functional
	// executor. Its presence keeps every online ciphertext in n+1
	// vector-secret form and disables CKS owner ingress.
	paperAdapter *paperMKFLattigoAdapter
	// kklssAdapter is non-nil only for the opt-in public-KKLSS functional
	// executor. Its degree-two carrier is an exact encoding of the native
	// KKLSS (c0,c_data,c_model) tuple; aggregate is used only by the
	// post-boundary correctness oracle.
	kklssAdapter *kklssLattigoAdapter
	// guardedCompactAdapter is non-nil only for the canonical P2G functional
	// backend. Every fresh owner ciphertext is folded exactly once at scale
	// Delta*q_g, rescaled once to remove q_g, and remains a degree-one
	// aggregate-domain ciphertext for all subsequent online evaluation.
	guardedCompactAdapter *guardedCompactMKFLattigoAdapter
	minLevel              int
	logBound              int
	refreshCRS            utils.PRNG
	nextPhase             int
	// boundaryFixtureObserver is an opt-in, synchronous diagnostic hook. It
	// receives the live compact ciphertext before the scheduler can reuse it.
	// It is never installed by CPU timing runs. A write/identity failure is
	// latched and makes the current phase fail before the next phase starts.
	boundaryFixtureObserver  func(name string, ciphertexts []*ckks.Ciphertext, normalized []float64) error
	boundaryFixtureError     error
	boundaryFixtureFinalizer func() error
	boundaryFixtureOutput    string

	inputCiphertexts  []*ckks.Ciphertext
	currentCiphertext *ckks.Ciphertext
	currentPlaintext  []complex128
}

// newKKLSSFullResNet20Executor wires the public SNUCP/MKHE-KKLSS P2/n=2
// implementation into the same frozen 12-phase scheduler used by the other
// functional executors. Key generation, CRS/rotation-key generation, and the
// aggregate correctness oracle are constructed here, before ExecutePhase.
// This constructor is deliberately functional-only: owner ciphertexts are
// encrypted lazily by the opt-in functional gate and therefore cannot be
// reused as a performance measurement.
func newKKLSSFullResNet20Executor(request fullResNet20Request) (fullResNet20Executor, error) {
	return newKKLSSFullResNet20ExecutorWithRotations(request, fullResNet20RotationKeys())
}

func newKKLSSFullResNet20ExecutorWithRotations(request fullResNet20Request, rotations []int) (fullResNet20Executor, error) {
	if request.Mode != fullResNet20FunctionalMode {
		return nil, fmt.Errorf("the public-KKLSS integrated executor is functional-only; performance requires offline ciphertext staging")
	}
	if request.Parameter != "P2" && request.Parameter != guardedCompactP2GN16ParameterID {
		return nil, fmt.Errorf("the public-KKLSS full-network functional contract requires P2 or P2GN16, got %q", request.Parameter)
	}
	if len(request.Plan.Phases) != 12 || request.Plan.Expected.DataInputCiphertexts != 3 || request.Plan.Expected.ModelWeightCiphertexts != 6172 || request.Plan.Expected.ModelBiasCiphertexts != 26 || request.Plan.Expected.CiphertextProducts != 6172 || request.Plan.Expected.RotationCalls != 6499 || request.Plan.Expected.Degree59Activations != 19 || request.Plan.Expected.CollectiveRefreshes != 22 {
		return nil, fmt.Errorf("the public-KKLSS executor requires the exact frozen 3/6172/26 owner, 6172-product, 6499-rotation, 19-activation, 22-refresh plan")
	}
	wantCKSBridgeMessages := fullResNet20Parties * fullResNet20Parties * request.Plan.Expected.CollectiveRefreshes
	if request.Plan.Communication.CKSIngressShareMessages != wantCKSBridgeMessages {
		return nil, fmt.Errorf("public-KKLSS plan reports %d CKS bridge messages, want %d", request.Plan.Communication.CKSIngressShareMessages, wantCKSBridgeMessages)
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
		params, err = fullResNet20Parameters(request.Parameter)
		if err != nil {
			return nil, err
		}
	}
	if params.Slots() != realResNetStemActiveSlots {
		return nil, fmt.Errorf("full-network packing requires %d slots, got %d", realResNetStemActiveSlots, params.Slots())
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), fullResNet20Parties, params.Q())
	if !ok {
		return nil, fmt.Errorf("%s lacks levels for the two-party KKLSS collective-refresh bridge", request.Parameter)
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create KKLSS refresh CRS: %w", err)
	}
	context, err := newKKLSSN2Context(request.Parameter, rotations)
	if err != nil {
		return nil, fmt.Errorf("generate public-KKLSS %s/n=2 evaluation material: %w", request.Parameter, err)
	}
	shares, err := kklssIndividualRLWEShares(context.partySecrets)
	if err != nil {
		return nil, err
	}
	public := make([]*rlwe.PublicKey, len(context.partyIDs))
	for index, id := range context.partyIDs {
		public[index] = &context.public.GetPublicKey(id).PublicKey
	}
	adapter, err := newKKLSSLattigoAdapter(params, context, true)
	if err != nil {
		return nil, err
	}
	return &encryptedFullResNet20Executor{
		request:      request,
		params:       params,
		encoder:      ckks.NewEncoder(params),
		shares:       shares,
		public:       public,
		aggregate:    aggregateSecret(params.Parameters, shares), // post-boundary oracle only
		evaluator:    adapter,
		kklssAdapter: adapter,
		minLevel:     minLevel,
		logBound:     logBound,
		refreshCRS:   refreshCRS,
	}, nil
}

// newPaperMKFFullResNet20Executor is a functional-contract constructor for
// the paper's proposed n+1 online path.  newPaperMKFCKKSTestMaterial forms
// s_E centrally and is therefore deliberately isolated as test-only
// EvalKeyGen; neither this constructor nor its wall time is performance
// evidence for the distributed protocol claimed by the manuscript.
func newPaperMKFFullResNet20Executor(request fullResNet20Request) (fullResNet20Executor, error) {
	if request.Mode != fullResNet20FunctionalMode {
		return nil, fmt.Errorf("the proposed-MKF integrated executor is functional-only")
	}
	params, err := fullResNet20Parameters(request.Parameter)
	if err != nil {
		return nil, err
	}
	if params.Slots() != realResNetStemActiveSlots {
		return nil, fmt.Errorf("full-network packing requires %d slots, got %d", realResNetStemActiveSlots, params.Slots())
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), fullResNet20Parties, params.Q())
	if !ok {
		return nil, fmt.Errorf("%s lacks levels for the two-party collective refresh", request.Parameter)
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create refresh CRS: %w", err)
	}
	keyGenerator := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, fullResNet20Parties)
	public := make([]*rlwe.PublicKey, fullResNet20Parties)
	for party := range shares {
		shares[party], public[party] = keyGenerator.GenKeyPair()
	}
	keys, aggregate, err := newPaperMKFCKKSTestMaterial(params, shares, fullResNet20RotationKeys())
	if err != nil {
		return nil, fmt.Errorf("generate test-only proposed-MKF evaluation material: %w", err)
	}
	online, err := newPaperMKFCKKSEvaluator(params, fullResNet20Parties, keys)
	if err != nil {
		return nil, err
	}
	adapter := newPaperMKFLattigoAdapter(online)
	return &encryptedFullResNet20Executor{
		request:      request,
		params:       params,
		encoder:      ckks.NewEncoder(params),
		shares:       shares,
		public:       public,
		aggregate:    aggregate, // out-of-boundary correctness oracle only
		evaluator:    adapter,
		paperAdapter: adapter,
		minLevel:     minLevel,
		logBound:     logBound,
		refreshCRS:   refreshCRS,
	}, nil
}

func newEncryptedFullResNet20Executor(request fullResNet20Request) (fullResNet20Executor, error) {
	if request.Mode != fullResNet20FunctionalMode {
		return nil, fmt.Errorf("the integrated executor is functional-only; performance mode requires a separately timed harness and cannot reuse oracle-instrumented execution")
	}
	params, err := fullResNet20Parameters(request.Parameter)
	if err != nil {
		return nil, err
	}
	if params.Slots() != realResNetStemActiveSlots {
		return nil, fmt.Errorf("full-network packing requires %d slots, got %d", realResNetStemActiveSlots, params.Slots())
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), fullResNet20Parties, params.Q())
	if !ok {
		return nil, fmt.Errorf("%s lacks levels for the two-party collective refresh", request.Parameter)
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create refresh CRS: %w", err)
	}
	evaluationCRS, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create evaluation-key CRS: %w", err)
	}
	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, fullResNet20Parties)
	public := make([]*rlwe.PublicKey, fullResNet20Parties)
	for party := range shares {
		shares[party], public[party] = kgen.GenKeyPair()
	}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, fullResNet20RotationKeys(), evaluationCRS)
	if err != nil {
		return nil, fmt.Errorf("generate collective evaluation keys: %w", err)
	}
	return &encryptedFullResNet20Executor{
		request:    request,
		params:     params,
		encoder:    ckks.NewEncoder(params),
		shares:     shares,
		public:     public,
		aggregate:  aggregateSecret(params.Parameters, shares), // functional verifier only
		evaluator:  ckks.NewEvaluator(params, evaluationKey),
		minLevel:   minLevel,
		logBound:   logBound,
		refreshCRS: refreshCRS,
	}, nil
}

func init() {
	fullResNet20ExecutorFactory = newEncryptedFullResNet20Executor
}

func (executor *encryptedFullResNet20Executor) ExecutePhase(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	if executor.nextPhase >= len(executor.request.Plan.Phases) || executor.request.Plan.Phases[executor.nextPhase].Name != phase.Name {
		return fullResNet20PhaseResult{}, fmt.Errorf("phase %q is out of order at position %d", phase.Name, executor.nextPhase)
	}
	executor.nextPhase++
	beforePaper := executor.paperOperationSnapshot()
	beforeKKLSS := executor.kklssOperationSnapshot()
	beforeGuarded := executor.guardedCompactOperationSnapshot()
	var result fullResNet20PhaseResult
	var err error
	switch phase.Kind {
	case fullResNet20InputPhase:
		result, err = executor.executeInput(phase)
	case fullResNet20StemPhase:
		result, err = executor.executeStem(phase)
	case fullResNet20RegularPhase:
		result, err = executor.executeRegularBlock(phase)
	case fullResNet20TransitionPhase:
		result, err = executor.executeTransitionBlock(phase)
	case fullResNet20TailPhase:
		result, err = executor.executeTail(phase)
	default:
		err = fmt.Errorf("unsupported phase kind %q", phase.Kind)
	}
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	if executor.boundaryFixtureError != nil {
		return fullResNet20PhaseResult{}, fmt.Errorf("export P2GN16 boundary fixture in phase %s: %w", phase.Name, executor.boundaryFixtureError)
	}
	if err := executor.validatePaperPhaseCounts(phase, beforePaper); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	if err := executor.validateKKLSSPhaseCounts(phase, beforeKKLSS); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	if err := executor.validateGuardedCompactPhaseCounts(phase, beforeGuarded); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	result.Name = phase.Name
	result.Counts = phase.Expected
	result.Communication = phase.Expected.communication(fullResNet20Parties)
	if executor.paperAdapter != nil || executor.guardedCompactAdapter != nil {
		// Owner ciphertexts enter as sparse n+1 objects. There is no CKS
		// share exchange before evaluation in the proposed-MKF path.
		result.Communication.CKSIngressShareMessages = 0
		result.Communication.CKSIngressBytes = 0
	} else if executor.kklssAdapter != nil {
		// Native KKLSS owner ciphertexts require no ingress key switch. The
		// collective-refresh compatibility bridge converts each of the two
		// source components to s_E with one CKS share from each of two parties:
		// four logical CKS messages per refresh. No byte count is claimed until
		// the transcript is serialized on a measured transport boundary.
		result.Communication.CKSIngressShareMessages = fullResNet20Parties * fullResNet20Parties * phase.Expected.CollectiveRefreshes
		result.Communication.CKSIngressBytes = 0
	}
	return result, nil
}

func (executor *encryptedFullResNet20Executor) observeBoundaryFixture(name string, ciphertexts []*ckks.Ciphertext, normalized []float64) {
	if executor.boundaryFixtureObserver == nil || executor.boundaryFixtureError != nil {
		return
	}
	executor.boundaryFixtureError = executor.boundaryFixtureObserver(name, ciphertexts, normalized)
}

func (executor *encryptedFullResNet20Executor) kklssOperationSnapshot() kklssPhaseSnapshot {
	if executor.kklssAdapter == nil {
		return kklssPhaseSnapshot{}
	}
	return executor.kklssAdapter.operationSnapshot()
}

func (executor *encryptedFullResNet20Executor) validateKKLSSPhaseCounts(phase fullResNet20Phase, before kklssPhaseSnapshot) error {
	if executor.kklssAdapter == nil {
		return nil
	}
	after := executor.kklssOperationSnapshot()
	wantOwners := phase.Expected.DataInputCiphertexts + phase.Expected.ModelWeightCiphertexts + phase.Expected.ModelBiasCiphertexts
	actual := kklssPhaseSnapshot{
		owners:             after.owners - before.owners,
		mulRelin:           after.mulRelin - before.mulRelin,
		polynomialMulRelin: after.polynomialMulRelin - before.polynomialMulRelin,
		rotations:          after.rotations - before.rotations,
		polynomials:        after.polynomials - before.polynomials,
		refreshes:          after.refreshes - before.refreshes,
	}
	// mulRelin counts the frozen network products, while polynomialMulRelin
	// separately counts the exact 17-product degree-59 BSGS DAG. Keeping both
	// counters makes every phase fail closed on the same polynomial schedule
	// used by the canonical MKF backend and by the matched timing harness.
	wantPolynomialProducts := phase.Expected.Degree59Activations * p2gn16Degree59ProductsPerActivation
	if actual.owners != wantOwners || actual.mulRelin != phase.Expected.CiphertextProducts || actual.polynomialMulRelin != wantPolynomialProducts || actual.rotations != phase.Expected.RotationCalls || actual.polynomials != phase.Expected.Degree59Activations || actual.refreshes != phase.Expected.CollectiveRefreshes {
		return fmt.Errorf("%s public-KKLSS structural count mismatch: actual owners=%d network-products=%d polynomial-products=%d rotations=%d poly=%d refresh=%d; want %d/%d/%d/%d/%d/%d", phase.Name, actual.owners, actual.mulRelin, actual.polynomialMulRelin, actual.rotations, actual.polynomials, actual.refreshes, wantOwners, phase.Expected.CiphertextProducts, wantPolynomialProducts, phase.Expected.RotationCalls, phase.Expected.Degree59Activations, phase.Expected.CollectiveRefreshes)
	}
	return nil
}

type paperMKFPhaseSnapshot struct {
	owners      int
	mulRelin    int
	rotations   int
	polynomials int
	refreshes   int
}

func (executor *encryptedFullResNet20Executor) paperOperationSnapshot() paperMKFPhaseSnapshot {
	if executor.paperAdapter == nil {
		return paperMKFPhaseSnapshot{}
	}
	counts := executor.paperAdapter.online.counts
	return paperMKFPhaseSnapshot{
		owners:      executor.paperAdapter.ownerCiphertexts,
		mulRelin:    counts.MulRelinCalls,
		rotations:   counts.RotationCalls,
		polynomials: counts.PolynomialCalls,
		refreshes:   executor.paperAdapter.collectiveRefreshes,
	}
}

func (executor *encryptedFullResNet20Executor) validatePaperPhaseCounts(phase fullResNet20Phase, before paperMKFPhaseSnapshot) error {
	if executor.paperAdapter == nil {
		return nil
	}
	after := executor.paperOperationSnapshot()
	wantOwners := phase.Expected.DataInputCiphertexts + phase.Expected.ModelWeightCiphertexts + phase.Expected.ModelBiasCiphertexts
	// The frozen degree-59 balanced BSGS schedule has 17 ciphertext
	// multiplications. This was independently established by the P2
	// degree-59 primitive contract and is checked at every activation site.
	wantMulRelin := phase.Expected.CiphertextProducts + 17*phase.Expected.Degree59Activations
	actual := paperMKFPhaseSnapshot{
		owners:      after.owners - before.owners,
		mulRelin:    after.mulRelin - before.mulRelin,
		rotations:   after.rotations - before.rotations,
		polynomials: after.polynomials - before.polynomials,
		refreshes:   after.refreshes - before.refreshes,
	}
	if actual.owners != wantOwners || actual.mulRelin != wantMulRelin || actual.rotations != phase.Expected.RotationCalls || actual.polynomials != phase.Expected.Degree59Activations || actual.refreshes != phase.Expected.CollectiveRefreshes {
		return fmt.Errorf("%s proposed-MKF structural count mismatch: actual owners=%d mul=%d rotations=%d poly=%d refresh=%d; want %d/%d/%d/%d/%d", phase.Name, actual.owners, actual.mulRelin, actual.rotations, actual.polynomials, actual.refreshes, wantOwners, wantMulRelin, phase.Expected.RotationCalls, phase.Expected.Degree59Activations, phase.Expected.CollectiveRefreshes)
	}
	return nil
}

func (executor *encryptedFullResNet20Executor) functionalBoundary(name string, ciphertext *ckks.Ciphertext, want []complex128) fullResNet20Boundary {
	executor.observeBoundaryFixture(name, []*ckks.Ciphertext{ciphertext}, nil)
	if executor.paperAdapter != nil || executor.kklssAdapter != nil {
		oracle, err := executor.paperOracleCiphertext(ciphertext)
		if err != nil {
			return fullResNet20Boundary{Name: name, Level: ciphertext.Level(), CheckedSlots: len(want), OracleChecked: false, MaxError: math.Inf(1), RMSE: math.Inf(1)}
		}
		measurement := measureRealResNetBlockBoundary(executor.params, executor.encoder, executor.aggregate, oracle, want)
		return fullResNet20Boundary{Name: name, Level: measurement.level, CheckedSlots: len(want), OracleChecked: true, MaxError: measurement.maxError, RMSE: measurement.rmse}
	}
	measurement := measureRealResNetBlockBoundary(executor.params, executor.encoder, executor.aggregate, ciphertext, want)
	return fullResNet20Boundary{Name: name, Level: measurement.level, CheckedSlots: len(want), OracleChecked: true, MaxError: measurement.maxError, RMSE: measurement.rmse}
}

// paperOracleCiphertext converts either supported multi-key phase to a
// degree-one ciphertext with a zero nonconstant component. It is called only
// after an online boundary has completed and cannot influence evaluation.
// The historical name is retained to avoid changing the already-validated
// proposed-MKF executor call sites.
func (executor *encryptedFullResNet20Executor) paperOracleCiphertext(ciphertext *ckks.Ciphertext) (*ckks.Ciphertext, error) {
	if executor.kklssAdapter != nil {
		return executor.kklssAdapter.oracleCarrier(ciphertext)
	}
	if executor.paperAdapter == nil {
		return ciphertext, nil
	}
	state, err := executor.paperAdapter.state(ciphertext)
	if err != nil {
		return nil, err
	}
	phase, err := paperMKFPhase(executor.params, state, executor.shares)
	if err != nil {
		return nil, err
	}
	zero := executor.params.RingQ().NewPolyLvl(state.Level)
	zero.IsNTT = true
	oracle := ckks.NewCiphertextAtLevelFromPoly(state.Level, [2]*ring.Poly{phase, zero})
	oracle.Scale = state.Scale
	return oracle, nil
}

func (executor *encryptedFullResNet20Executor) ownerCiphertext(owner int, values []complex128) (*ckks.Ciphertext, error) {
	if executor.guardedCompactAdapter != nil {
		return executor.guardedCompactAdapter.ownerCarrier(executor.encoder, executor.public, owner, values)
	}
	if executor.paperAdapter != nil {
		return executor.paperAdapter.ownerCarrier(executor.encoder, executor.public, owner, values)
	}
	if executor.kklssAdapter != nil {
		return executor.kklssAdapter.ownerCarrier(executor.encoder, executor.public, owner, values)
	}
	multi, err := makeOwnerCKKSMultiCiphertext(executor.params, executor.encoder, executor.public, owner, values)
	if err != nil {
		return nil, err
	}
	return interactiveOwnerFold(executor.params, multi, executor.shares, owner)
}

func (executor *encryptedFullResNet20Executor) refresh(name string, ciphertext *ckks.Ciphertext, want []complex128) (*ckks.Ciphertext, fullResNet20Boundary, error) {
	if executor.guardedCompactAdapter != nil {
		return executor.guardedCompactRefresh(name, ciphertext, want)
	}
	if executor.paperAdapter != nil {
		refreshed, err := executor.paperAdapter.collectiveRefresh(ciphertext, executor.shares, executor.minLevel, executor.logBound, executor.refreshCRS)
		if err != nil {
			return nil, fullResNet20Boundary{}, err
		}
		boundary := executor.functionalBoundary(name, refreshed, want)
		if !boundary.OracleChecked {
			return nil, fullResNet20Boundary{}, fmt.Errorf("%s proposed-MKF refresh oracle conversion failed", name)
		}
		return refreshed, boundary, nil
	}
	if executor.kklssAdapter != nil {
		refreshed, err := executor.kklssAdapter.collectiveRefresh(ciphertext, executor.shares, executor.minLevel, executor.logBound, executor.refreshCRS)
		if err != nil {
			return nil, fullResNet20Boundary{}, err
		}
		boundary := executor.functionalBoundary(name, refreshed, want)
		if !boundary.OracleChecked {
			return nil, fullResNet20Boundary{}, fmt.Errorf("%s public-KKLSS refresh oracle conversion failed", name)
		}
		return refreshed, boundary, nil
	}
	refreshed, measurement, err := refreshRealResNetBlockBoundary(executor.params, executor.evaluator, executor.shares, ciphertext, executor.minLevel, executor.logBound, executor.refreshCRS, executor.encoder, executor.aggregate, want)
	if err != nil {
		return nil, fullResNet20Boundary{}, err
	}
	return refreshed, fullResNet20Boundary{Name: name, Level: measurement.outputLevel, CheckedSlots: len(want), OracleChecked: true, MaxError: measurement.maxError, RMSE: measurement.rmse}, nil
}

func (executor *encryptedFullResNet20Executor) executeInput(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	image := executor.request.Fixture.NormalizedRGB
	if len(image) != resnetImageChannels {
		return fullResNet20PhaseResult{}, fmt.Errorf("normalized fixture has %d channels", len(image))
	}
	executor.inputCiphertexts = make([]*ckks.Ciphertext, resnetImageChannels)
	var maxError, sumSquared float64
	checked := 0
	normalized := make([]float64, 0, resnetImageChannels*resnetImageWidth*resnetImageWidth)
	for channel := 0; channel < resnetImageChannels; channel++ {
		for y := 0; y < resnetImageWidth; y++ {
			normalized = append(normalized, image[channel][y]...)
		}
	}
	executor.observeBoundaryFixture(phase.RequiredBoundaries[0], nil, normalized)
	for channel := 0; channel < resnetImageChannels; channel++ {
		packed := packReplicatedInputChannel(executor.params, image, channel)
		var err error
		executor.inputCiphertexts[channel], err = executor.ownerCiphertext(realResNetTailDataOwner, packed)
		if err != nil {
			return fullResNet20PhaseResult{}, err
		}
		oracle, err := executor.paperOracleCiphertext(executor.inputCiphertexts[channel])
		if err != nil {
			return fullResNet20PhaseResult{}, err
		}
		decoded := executor.encoder.Decode(ckks.NewDecryptor(executor.params, executor.aggregate).DecryptNew(oracle), executor.params.LogSlots())
		for index, want := range packed {
			delta := cmplx.Abs(decoded[index] - want)
			maxError = math.Max(maxError, delta)
			sumSquared += delta * delta
			checked++
		}
	}
	executor.observeBoundaryFixture(phase.RequiredBoundaries[1], executor.inputCiphertexts, nil)
	return fullResNet20PhaseResult{Boundaries: []fullResNet20Boundary{
		{Name: phase.RequiredBoundaries[0], Level: -1, CheckedSlots: resnetImageChannels * resnetImageWidth * resnetImageWidth, OracleChecked: true},
		{Name: phase.RequiredBoundaries[1], Level: executor.inputCiphertexts[0].Level(), CheckedSlots: checked, OracleChecked: true, MaxError: maxError, RMSE: math.Sqrt(sumSquared / float64(checked))},
	}}, nil
}

func (executor *encryptedFullResNet20Executor) executeStem(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	if len(executor.inputCiphertexts) != resnetImageChannels || len(phase.ActivationSites) != 1 {
		return fullResNet20PhaseResult{}, fmt.Errorf("stem input or activation-site contract is incomplete")
	}
	scale := phase.ActivationSites[0].Scale
	weights, err := loadRealResNetStemWeightsAtScale(executor.request.WeightsDirectory, scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	var sum *ckks.Ciphertext
	for inputChannel := 0; inputChannel < resnetImageChannels; inputChannel++ {
		spatial := resNetBlockSpatialRotations(executor.evaluator, executor.inputCiphertexts[inputChannel])
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weight, err := executor.ownerCiphertext(realResNetTailModelOwner, weights.masks[inputChannel][kernelY][kernelX])
				if err != nil {
					return fullResNet20PhaseResult{}, err
				}
				term := executor.evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
				if sum == nil {
					sum = term
				} else {
					executor.evaluator.Add(sum, term, sum)
				}
			}
		}
	}
	if err := executor.evaluator.Rescale(sum, executor.params.Scale(), sum); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	bias, err := executor.ownerCiphertext(realResNetTailModelOwner, weights.biasPacked)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	executor.evaluator.DropLevel(bias, bias.Level()-sum.Level())
	executor.evaluator.Add(sum, bias, sum)
	plain := plainRealFoldedResNetStem(executor.request.Fixture.NormalizedRGB, weights)
	convBoundary := executor.functionalBoundary(phase.RequiredBoundaries[0], sum, plain)
	refreshed, refreshBoundary, err := executor.refresh(phase.RequiredBoundaries[1], sum, plain)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	polynomial := referenceResNetBlockReLUPolynomial(scale)
	plainActivation, _, err := plainReferenceResNetBlockReLU(plain, polynomial, scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshed, polynomial, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activationBoundary := executor.functionalBoundary(phase.RequiredBoundaries[2], activation, plainActivation)
	executor.currentCiphertext, executor.currentPlaintext = activation, plainActivation
	return fullResNet20PhaseResult{Boundaries: []fullResNet20Boundary{convBoundary, refreshBoundary, activationBoundary}}, nil
}

type fullResNet20RegularOperators struct {
	plainConv1     func([]complex128) []complex128
	plainConv2     func([]complex128) []complex128
	encryptedConv1 func(*ckks.Ciphertext) (*ckks.Ciphertext, error)
	encryptedConv2 func(*ckks.Ciphertext) (*ckks.Ciphertext, error)
}

func (executor *encryptedFullResNet20Executor) multiKeyOwnerEvaluator() fullResNet20OwnerEvaluator {
	if executor.guardedCompactAdapter != nil {
		return executor.guardedCompactAdapter
	}
	if executor.paperAdapter != nil {
		return executor.paperAdapter
	}
	if executor.kklssAdapter != nil {
		return executor.kklssAdapter
	}
	return nil
}

func (executor *encryptedFullResNet20Executor) encryptedLayer1Conv(input *ckks.Ciphertext, weights *realResNetBlockConvWeights) (*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealResNetBlockConv(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealResNetBlockConv(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) encryptedLayer2Conv(input *ckks.Ciphertext, weights *realStage2ConvWeights) (*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage2Conv(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage2Conv(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) encryptedLayer3Conv(input *ckks.Ciphertext, weights *realStage3ConvWeights) (*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage3Conv(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage3Conv(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) regularOperators(phase fullResNet20Phase) (fullResNet20RegularOperators, error) {
	conv1Scale, conv2Scale := phase.ActivationSites[0].Scale, phase.ActivationSites[1].Scale
	switch phase.Layer {
	case 1:
		conv1, err := loadRealResNetBlockConvWeights(executor.request.WeightsDirectory, phase.Conv1Prefix, conv1Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		conv2, err := loadRealResNetBlockConvWeights(executor.request.WeightsDirectory, phase.Conv2Prefix, conv2Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		return fullResNet20RegularOperators{
			plainConv1: func(input []complex128) []complex128 { return plainRealResNetBlockConv(input, conv1) },
			plainConv2: func(input []complex128) []complex128 { return plainRealResNetBlockConv(input, conv2) },
			encryptedConv1: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer1Conv(input, conv1)
			},
			encryptedConv2: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer1Conv(input, conv2)
			},
		}, nil
	case 2:
		conv1, err := loadRealStage2ConvWeights(executor.request.WeightsDirectory, phase.Conv1Prefix, conv1Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		conv2, err := loadRealStage2ConvWeights(executor.request.WeightsDirectory, phase.Conv2Prefix, conv2Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		return fullResNet20RegularOperators{
			plainConv1: func(input []complex128) []complex128 { return plainRealStage2Conv(input, conv1) },
			plainConv2: func(input []complex128) []complex128 { return plainRealStage2Conv(input, conv2) },
			encryptedConv1: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer2Conv(input, conv1)
			},
			encryptedConv2: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer2Conv(input, conv2)
			},
		}, nil
	case 3:
		conv1, err := loadRealStage3SameWidthConvWeights(executor.request.WeightsDirectory, phase.Conv1Prefix, conv1Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		conv2, err := loadRealStage3SameWidthConvWeights(executor.request.WeightsDirectory, phase.Conv2Prefix, conv2Scale, executor.params.Slots())
		if err != nil {
			return fullResNet20RegularOperators{}, err
		}
		return fullResNet20RegularOperators{
			plainConv1: func(input []complex128) []complex128 { return plainRealStage3Conv(input, conv1) },
			plainConv2: func(input []complex128) []complex128 { return plainRealStage3Conv(input, conv2) },
			encryptedConv1: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer3Conv(input, conv1)
			},
			encryptedConv2: func(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
				return executor.encryptedLayer3Conv(input, conv2)
			},
		}, nil
	default:
		return fullResNet20RegularOperators{}, fmt.Errorf("regular block %s has unsupported layer %d", phase.Name, phase.Layer)
	}
}

func (executor *encryptedFullResNet20Executor) executeRegularBlock(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	if executor.currentCiphertext == nil || len(phase.ActivationSites) != 2 {
		return fullResNet20PhaseResult{}, fmt.Errorf("regular block %s has no input or scale pair", phase.Name)
	}
	operators, err := executor.regularOperators(phase)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	skipCiphertext := executor.currentCiphertext
	skipPlaintext := append([]complex128(nil), executor.currentPlaintext...)
	plainConv1 := operators.plainConv1(executor.currentPlaintext)
	conv1, err := operators.encryptedConv1(executor.currentCiphertext)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries := []fullResNet20Boundary{executor.functionalBoundary(phase.RequiredBoundaries[0], conv1, plainConv1)}
	refreshed1, refresh1, err := executor.refresh(phase.RequiredBoundaries[1], conv1, plainConv1)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh1)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	plainActivation1, _, err := plainReferenceResNetBlockReLU(plainConv1, poly1, phase.ActivationSites[0].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation1, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshed1, poly1, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[2], activation1, plainActivation1))
	plainConv2 := operators.plainConv2(plainActivation1)
	conv2, err := operators.encryptedConv2(activation1)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[3], conv2, plainConv2))
	plainResidual := addPlainRealResNetBlockSkip(plainConv2, skipPlaintext, phase.ActivationSites[1].Scale)
	if err := addScaledRealResNetBlockSkip(executor.params, executor.evaluator, conv2, skipCiphertext, phase.ActivationSites[1].Scale); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[4], conv2, plainResidual))
	refreshed2, refresh2, err := executor.refresh(phase.RequiredBoundaries[5], conv2, plainResidual)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh2)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	plainActivation2, _, err := plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation2, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshed2, poly2, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[6], activation2, plainActivation2))
	executor.currentCiphertext, executor.currentPlaintext = activation2, plainActivation2
	if len(phase.RequiredBoundaries) == 8 {
		finalCiphertext, finalRefresh, err := executor.refresh(phase.RequiredBoundaries[7], executor.currentCiphertext, executor.currentPlaintext)
		if err != nil {
			return fullResNet20PhaseResult{}, err
		}
		executor.currentCiphertext = finalCiphertext
		boundaries = append(boundaries, finalRefresh)
	}
	return fullResNet20PhaseResult{Boundaries: boundaries}, nil
}

func (executor *encryptedFullResNet20Executor) executeTransitionBlock(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	if executor.currentCiphertext == nil || len(phase.ActivationSites) != 2 {
		return fullResNet20PhaseResult{}, fmt.Errorf("transition block %s has no input or scale pair", phase.Name)
	}
	if phase.Layer == 2 {
		return executor.executeStage2Transition(phase)
	}
	if phase.Layer == 3 {
		return executor.executeStage3Transition(phase)
	}
	return fullResNet20PhaseResult{}, fmt.Errorf("transition block %s has unsupported layer %d", phase.Name, phase.Layer)
}

func (executor *encryptedFullResNet20Executor) encryptedStage2Stride(input *ckks.Ciphertext, weights *realStage2StrideWeights) ([2]*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage2StrideConv(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage2StrideConv(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) encryptedStage2Skip(input *ckks.Ciphertext, weights *realStage2SkipWeights) ([2]*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage2Skip(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage2Skip(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) encryptedStage3Stride(input *ckks.Ciphertext, weights *realStage3StrideWeights) ([2]*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage3StrideConv(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage3StrideConv(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) encryptedStage3Skip(input *ckks.Ciphertext, weights *realStage3SkipWeights) ([2]*ckks.Ciphertext, error) {
	if adapter := executor.multiKeyOwnerEvaluator(); adapter != nil {
		return paperMKFEncryptedRealStage3Skip(executor.params, executor.encoder, adapter, executor.public, input, weights)
	}
	return encryptedRealStage3Skip(executor.params, executor.encoder, executor.evaluator, executor.public, executor.shares, input, weights)
}

func (executor *encryptedFullResNet20Executor) executeStage2Transition(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	mainWeights, err := loadRealStage2StrideWeights(executor.request.WeightsDirectory, phase.ActivationSites[0].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	skipWeights, err := loadRealStage2SkipWeights(executor.request.WeightsDirectory, phase.ActivationSites[1].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	conv2Weights, err := loadRealStage2ConvWeights(executor.request.WeightsDirectory, phase.Conv2Prefix, phase.ActivationSites[1].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	refreshedInput, preRefresh, err := executor.refresh(phase.RequiredBoundaries[0], executor.currentCiphertext, executor.currentPlaintext)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	plainMain := plainRealStage2StrideConv(executor.currentPlaintext, mainWeights)
	plainSkip := plainRealStage2Skip(executor.currentPlaintext, skipWeights)
	plainMainHalves := [2][]complex128{plainSparseStage2Half(plainMain, 0), plainSparseStage2Half(plainMain, 1)}
	plainSkipHalves := [2][]complex128{plainSparseStage2Half(plainSkip, 0), plainSparseStage2Half(plainSkip, 1)}
	mainHalves, err := executor.encryptedStage2Stride(refreshedInput, mainWeights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	skipHalves, err := executor.encryptedStage2Skip(refreshedInput, skipWeights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries := []fullResNet20Boundary{preRefresh,
		executor.functionalBoundary(phase.RequiredBoundaries[1], mainHalves[0], plainMainHalves[0]),
		executor.functionalBoundary(phase.RequiredBoundaries[2], mainHalves[1], plainMainHalves[1]),
		executor.functionalBoundary(phase.RequiredBoundaries[3], skipHalves[0], plainSkipHalves[0]),
		executor.functionalBoundary(phase.RequiredBoundaries[4], skipHalves[1], plainSkipHalves[1]),
	}
	packedMain := encryptedPackStage2Halves(executor.params, executor.encoder, executor.evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage2Halves(executor.params, executor.encoder, executor.evaluator, skipHalves[0], skipHalves[1])
	if err := executor.evaluator.Rescale(packedMain, executor.params.Scale(), packedMain); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	if err := executor.evaluator.Rescale(packedSkip, executor.params.Scale(), packedSkip); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[5], packedMain, plainMain), executor.functionalBoundary(phase.RequiredBoundaries[6], packedSkip, plainSkip))
	refreshedMain, refresh1, err := executor.refresh(phase.RequiredBoundaries[7], packedMain, plainMain)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh1)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	plainActivation1, _, err := plainReferenceResNetBlockReLU(plainMain, poly1, phase.ActivationSites[0].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation1, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshedMain, poly1, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[8], activation1, plainActivation1))
	plainConv2 := plainRealStage2Conv(plainActivation1, conv2Weights)
	conv2, err := executor.encryptedLayer2Conv(activation1, conv2Weights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[9], conv2, plainConv2))
	if packedSkip.Level() < conv2.Level() {
		return fullResNet20PhaseResult{}, fmt.Errorf("stage2 skip level %d below conv2 level %d", packedSkip.Level(), conv2.Level())
	}
	executor.evaluator.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
	executor.evaluator.Add(conv2, packedSkip, conv2)
	plainResidual := stage2AddPlainBranches(plainConv2, plainSkip)
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[10], conv2, plainResidual))
	refreshedResidual, refresh2, err := executor.refresh(phase.RequiredBoundaries[11], conv2, plainResidual)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh2)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	plainActivation2, _, err := plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation2, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshedResidual, poly2, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[12], activation2, plainActivation2))
	executor.currentCiphertext, executor.currentPlaintext = activation2, plainActivation2
	return fullResNet20PhaseResult{Boundaries: boundaries}, nil
}

func (executor *encryptedFullResNet20Executor) executeStage3Transition(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	mainWeights, err := loadRealStage3StrideWeights(executor.request.WeightsDirectory, phase.ActivationSites[0].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	skipWeights, err := loadRealStage3SkipWeights(executor.request.WeightsDirectory, phase.ActivationSites[1].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	conv2Weights, err := loadRealStage3ConvWeights(executor.request.WeightsDirectory, phase.ActivationSites[1].Scale, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	refreshedInput, preRefresh, err := executor.refresh(phase.RequiredBoundaries[0], executor.currentCiphertext, executor.currentPlaintext)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	plainMain := plainRealStage3StrideConv(executor.currentPlaintext, mainWeights)
	plainSkip := plainRealStage3Skip(executor.currentPlaintext, skipWeights)
	plainMainHalves := [2][]complex128{plainSparseStage3Half(plainMain, 0), plainSparseStage3Half(plainMain, 1)}
	plainSkipHalves := [2][]complex128{plainSparseStage3Half(plainSkip, 0), plainSparseStage3Half(plainSkip, 1)}
	mainHalves, err := executor.encryptedStage3Stride(refreshedInput, mainWeights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	skipHalves, err := executor.encryptedStage3Skip(refreshedInput, skipWeights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries := []fullResNet20Boundary{preRefresh,
		executor.functionalBoundary(phase.RequiredBoundaries[1], mainHalves[0], plainMainHalves[0]),
		executor.functionalBoundary(phase.RequiredBoundaries[2], mainHalves[1], plainMainHalves[1]),
		executor.functionalBoundary(phase.RequiredBoundaries[3], skipHalves[0], plainSkipHalves[0]),
		executor.functionalBoundary(phase.RequiredBoundaries[4], skipHalves[1], plainSkipHalves[1]),
	}
	packedMain := encryptedPackStage3Halves(executor.params, executor.encoder, executor.evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage3Halves(executor.params, executor.encoder, executor.evaluator, skipHalves[0], skipHalves[1])
	if err := executor.evaluator.Rescale(packedMain, executor.params.Scale(), packedMain); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	if err := executor.evaluator.Rescale(packedSkip, executor.params.Scale(), packedSkip); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[5], packedMain, plainMain), executor.functionalBoundary(phase.RequiredBoundaries[6], packedSkip, plainSkip))
	refreshedMain, refresh1, err := executor.refresh(phase.RequiredBoundaries[7], packedMain, plainMain)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh1)
	poly1 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[0].Scale)
	plainActivation1, _, err := plainReferenceResNetBlockReLU(plainMain, poly1, phase.ActivationSites[0].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation1, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshedMain, poly1, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[8], activation1, plainActivation1))
	plainConv2 := plainRealStage3Conv(plainActivation1, conv2Weights)
	conv2, err := executor.encryptedLayer3Conv(activation1, conv2Weights)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[9], conv2, plainConv2))
	if packedSkip.Level() < conv2.Level() {
		return fullResNet20PhaseResult{}, fmt.Errorf("stage3 skip level %d below conv2 level %d", packedSkip.Level(), conv2.Level())
	}
	executor.evaluator.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
	executor.evaluator.Add(conv2, packedSkip, conv2)
	plainResidual := stage3AddPlainBranches(plainConv2, plainSkip)
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[10], conv2, plainResidual))
	refreshedResidual, refresh2, err := executor.refresh(phase.RequiredBoundaries[11], conv2, plainResidual)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, refresh2)
	poly2 := referenceResNetBlockReLUPolynomial(phase.ActivationSites[1].Scale)
	plainActivation2, _, err := plainReferenceResNetBlockReLU(plainResidual, poly2, phase.ActivationSites[1].Scale)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	activation2, err := encryptedReferenceResNetBlockReLU(executor.evaluator, refreshedResidual, poly2, executor.params.Scale())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[12], activation2, plainActivation2))
	executor.currentCiphertext, executor.currentPlaintext = activation2, plainActivation2
	return fullResNet20PhaseResult{Boundaries: boundaries}, nil
}

func fullResNet20PlainRotateLeft(input []complex128, rotation int) []complex128 {
	output := make([]complex128, len(input))
	rotation %= len(input)
	if rotation < 0 {
		rotation += len(input)
	}
	for destination := range output {
		output[destination] = input[(destination+rotation)%len(input)]
	}
	return output
}

func fullResNet20PlainRotSum(input []complex128, width int) []complex128 {
	result := append([]complex128(nil), input...)
	for rotation := 1; rotation < width; rotation <<= 1 {
		shifted := fullResNet20PlainRotateLeft(result, rotation)
		for index := range result {
			result[index] += shifted[index]
		}
	}
	return result
}

func fullResNet20PlainTailSchedule(input []complex128, weights *realResNetTailWeights, bias *realResNetTailBias) (gap, classified, logits []complex128) {
	gap = fullResNet20PlainRotSum(input, realResNetTailSpatialSlots)
	for index := range gap {
		if index%realResNetTailSpatialSlots == 0 {
			gap[index] /= realResNetTailSpatialSlots
		} else {
			gap[index] = 0
		}
	}
	repeated := fullResNet20PlainRotSum(gap, realResNetTailRepeatWidth)
	repeated = fullResNet20PlainRotateLeft(repeated, -realResNetTailRepeatWidth+1)
	classified = make([]complex128, len(input))
	for index := range classified {
		classified[index] = repeated[index] * weights.packed[index]
	}
	for channels := 1; channels < realResNetTailChannels; channels <<= 1 {
		shifted := fullResNet20PlainRotateLeft(classified, realResNetTailSpatialSlots*channels)
		for index := range classified {
			classified[index] += shifted[index]
		}
	}
	logits = append([]complex128(nil), classified...)
	if bias != nil {
		for index := range logits {
			logits[index] += bias.packed[index]
		}
	}
	return
}

func (executor *encryptedFullResNet20Executor) executeTail(phase fullResNet20Phase) (fullResNet20PhaseResult, error) {
	if executor.currentCiphertext == nil || len(executor.currentPlaintext) != executor.params.Slots() {
		return fullResNet20PhaseResult{}, fmt.Errorf("tail requires one full-width Stage3 ciphertext and plaintext oracle")
	}
	weights, err := loadRealResNetTailWeights(executor.request.WeightsDirectory, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	bias, err := loadCheckpointRealResNetTailBias(executor.request.CheckpointBiasPath, executor.params.Slots())
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	plainGap, plainClassified, plainLogits := fullResNet20PlainTailSchedule(executor.currentPlaintext, weights, bias)
	gap := encryptedRealResNetTailRotSum(executor.evaluator, executor.currentCiphertext, realResNetTailSpatialSlots)
	gapMask := make([]complex128, executor.params.Slots())
	for copyIndex := 0; copyIndex < executor.params.Slots()/realResNetTailReferenceSlots; copyIndex++ {
		for channel := 0; channel < realResNetTailChannels; channel++ {
			gapMask[copyIndex*realResNetTailReferenceSlots+channel*realResNetTailSpatialSlots] = complex(1.0/realResNetTailSpatialSlots, 0)
		}
	}
	gap = executor.evaluator.MulNew(gap, executor.encoder.EncodeNTTAtLvlNew(gap.Level(), gapMask, executor.params.LogSlots()))
	if err := executor.evaluator.Rescale(gap, executor.params.Scale(), gap); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	boundaries := []fullResNet20Boundary{executor.functionalBoundary(phase.RequiredBoundaries[0], gap, plainGap)}
	repeated := encryptedRealResNetTailRotSum(executor.evaluator, gap, realResNetTailRepeatWidth)
	repeated = executor.evaluator.RotateNew(repeated, -realResNetTailRepeatWidth+1)
	weight, err := executor.ownerCiphertext(realResNetTailModelOwner, weights.packed)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	executor.evaluator.DropLevel(weight, weight.Level()-repeated.Level())
	classified := executor.evaluator.MulRelinNew(repeated, weight)
	if err := executor.evaluator.Rescale(classified, executor.params.Scale(), classified); err != nil {
		return fullResNet20PhaseResult{}, err
	}
	classified = encryptedRealResNetTailPaddedRotSum(executor.evaluator, classified)
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[1], classified, plainClassified))
	biasCiphertext, err := executor.ownerCiphertext(realResNetTailModelOwner, bias.packed)
	if err != nil {
		return fullResNet20PhaseResult{}, err
	}
	executor.evaluator.DropLevel(biasCiphertext, biasCiphertext.Level()-classified.Level())
	executor.evaluator.Add(classified, biasCiphertext, classified)
	boundaries = append(boundaries, executor.functionalBoundary(phase.RequiredBoundaries[2], classified, plainLogits))
	executor.currentCiphertext, executor.currentPlaintext = classified, plainLogits
	return fullResNet20PhaseResult{Boundaries: boundaries}, nil
}

func fullResNet20Top1AndMargin(logits []float64) (top1 int, margin float64) {
	best, second := math.Inf(-1), math.Inf(-1)
	top1 = -1
	for index, value := range logits {
		if value > best {
			second, best, top1 = best, value, index
		} else if value > second {
			second = value
		}
	}
	if top1 < 0 || math.IsInf(second, -1) {
		return top1, math.NaN()
	}
	return top1, best - second
}

// finalClassificationEvidence decrypts only after the functional dataflow is
// complete. It is an out-of-boundary correctness oracle and is never used by
// the separately gated performance mode.
func (executor *encryptedFullResNet20Executor) finalClassificationEvidence(label int) (*fullResNet20ClassificationEvidence, error) {
	if executor.currentCiphertext == nil || len(executor.currentPlaintext) < realResNetTailClasses {
		return nil, fmt.Errorf("full-network final logits are unavailable")
	}
	oracle, err := executor.paperOracleCiphertext(executor.currentCiphertext)
	if err != nil {
		return nil, err
	}
	decoded := executor.encoder.Decode(
		ckks.NewDecryptor(executor.params, executor.aggregate).DecryptNew(oracle),
		executor.params.LogSlots(),
	)
	if len(decoded) < realResNetTailClasses {
		return nil, fmt.Errorf("decoded final output has %d slots, want at least %d", len(decoded), realResNetTailClasses)
	}
	evidence := &fullResNet20ClassificationEvidence{
		GroundTruthLabel: label,
		EncryptedLogits:  make([]float64, realResNetTailClasses),
		ReferenceLogits:  make([]float64, realResNetTailClasses),
	}
	for class := 0; class < realResNetTailClasses; class++ {
		evidence.EncryptedLogits[class] = real(decoded[class])
		evidence.ReferenceLogits[class] = real(executor.currentPlaintext[class])
		evidence.MaxLogitError = math.Max(evidence.MaxLogitError, cmplx.Abs(decoded[class]-executor.currentPlaintext[class]))
	}
	evidence.EncryptedTop1, evidence.EncryptedMargin = fullResNet20Top1AndMargin(evidence.EncryptedLogits)
	evidence.ReferenceTop1, evidence.ReferenceMargin = fullResNet20Top1AndMargin(evidence.ReferenceLogits)
	evidence.EncryptedCorrect = evidence.EncryptedTop1 == label
	evidence.ReferenceCorrect = evidence.ReferenceTop1 == label
	if evidence.EncryptedTop1 < 0 || evidence.ReferenceTop1 < 0 || math.IsNaN(evidence.EncryptedMargin) || math.IsNaN(evidence.ReferenceMargin) {
		return nil, fmt.Errorf("invalid full-network top-1 classification evidence")
	}
	return evidence, nil
}

func TestFullResNet20ParameterPackingContract(t *testing.T) {
	for _, name := range []string{"P2", guardedCompactP2GParameterID, guardedCompactP2GN16ParameterID, "P3IRCandidate"} {
		params, err := fullResNet20Parameters(name)
		if err != nil {
			t.Fatal(err)
		}
		if params.Slots() != realResNetStemActiveSlots || params.LogSlots() != 14 {
			t.Fatalf("%s full-network packing is LogSlots=%d slots=%d, want 14/%d", name, params.LogSlots(), params.Slots(), realResNetStemActiveSlots)
		}
	}
}

func TestFullResNet20FullWidthTailAdapterContract(t *testing.T) {
	const slots = realResNetStemActiveSlots
	weights := &realResNetTailWeights{packed: make([]complex128, slots)}
	bias := &realResNetTailBias{packed: make([]complex128, slots)}
	for channel := 0; channel < realResNetTailChannels; channel++ {
		for class := 0; class < realResNetTailClasses; class++ {
			value := float64((channel+1)*(class+2)%23-11) / 128
			weights.matrix[channel][class] = value
			for copyIndex := 0; copyIndex < slots/realResNetTailReferenceSlots; copyIndex++ {
				weights.packed[copyIndex*realResNetTailReferenceSlots+channel*realResNetTailSpatialSlots+class] = complex(value, 0)
			}
		}
	}
	for class := 0; class < realResNetTailClasses; class++ {
		bias.values[class] = float64(class-5) / 256
		for copyIndex := 0; copyIndex < slots/realResNetTailReferenceSlots; copyIndex++ {
			bias.packed[copyIndex*realResNetTailReferenceSlots+class] = complex(bias.values[class], 0)
		}
	}
	logical := deterministicRealResNetTailInput()
	input := make([]complex128, slots)
	for copyIndex := 0; copyIndex < slots/realResNetTailReferenceSlots; copyIndex++ {
		copy(input[copyIndex*realResNetTailReferenceSlots:], logical)
	}
	want, err := plainRealResNetTailLogits(logical, weights, bias)
	if err != nil {
		t.Fatal(err)
	}
	_, _, got := fullResNet20PlainTailSchedule(input, weights, bias)
	for copyIndex := 0; copyIndex < slots/realResNetTailReferenceSlots; copyIndex++ {
		for class := 0; class < realResNetTailClasses; class++ {
			index := copyIndex*realResNetTailReferenceSlots + class
			if delta := cmplx.Abs(got[index] - want[class]); delta > 1e-12 {
				t.Fatalf("full-width tail copy=%d class=%d mismatch=%g", copyIndex, class, delta)
			}
		}
	}
}

func TestFullResNet20FullWidthTailFrozenPackingAudit(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	biasPath := os.Getenv("MKF_RESNET20_FC_BIAS_PATH")
	if weightsDirectory == "" || biasPath == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR and MKF_RESNET20_FC_BIAS_PATH for the light frozen-tail packing audit")
	}
	weights, err := loadRealResNetTailWeights(weightsDirectory, realResNetStemActiveSlots)
	if err != nil {
		t.Fatal(err)
	}
	bias, err := loadCheckpointRealResNetTailBias(biasPath, realResNetStemActiveSlots)
	if err != nil {
		t.Fatal(err)
	}
	for copyIndex := 1; copyIndex < realResNetStemActiveSlots/realResNetTailReferenceSlots; copyIndex++ {
		base := copyIndex * realResNetTailReferenceSlots
		for index := 0; index < realResNetTailReferenceSlots; index++ {
			if weights.packed[base+index] != weights.packed[index] || bias.packed[base+index] != bias.packed[index] {
				t.Fatalf("frozen tail logical copy %d differs at position %d", copyIndex, index)
			}
		}
	}
	t.Logf("full-width frozen tail packing PASS: slots=%d logicalCopies=%d fcSHA256=%s biasSHA256=%s; no FHE executed", realResNetStemActiveSlots, realResNetStemActiveSlots/realResNetTailReferenceSlots, weights.sha256, bias.sha256)
}

func TestFullResNet20ActivationScaleAdaptationContract(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR for the light scale-adaptation audit")
	}
	const firstScale, secondScale = 0.9, 0.7
	first, err := loadRealResNetStemWeightsAtScale(weightsDirectory, firstScale, realResNetStemActiveSlots)
	if err != nil {
		t.Fatal(err)
	}
	second, err := loadRealResNetStemWeightsAtScale(weightsDirectory, secondScale, realResNetStemActiveSlots)
	if err != nil {
		t.Fatal(err)
	}
	wantRatio := secondScale / firstScale
	checkedKernel, checkedMask, checkedBias := false, false, false
	for outputChannel := 0; outputChannel < resnetStemChannels && !checkedKernel; outputChannel++ {
		for inputChannel := 0; inputChannel < resnetImageChannels && !checkedKernel; inputChannel++ {
			for kernelY := 0; kernelY < 3 && !checkedKernel; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					left := first.kernel[outputChannel][inputChannel][kernelY][kernelX]
					right := second.kernel[outputChannel][inputChannel][kernelY][kernelX]
					if left != 0 {
						if math.Abs(right/left-wantRatio) > 1e-14 {
							t.Fatalf("stem kernel did not follow the supplied activation scale")
						}
						checkedKernel = true
						break
					}
				}
			}
		}
	}
	for inputChannel := 0; inputChannel < resnetImageChannels && !checkedMask; inputChannel++ {
		for kernelY := 0; kernelY < 3 && !checkedMask; kernelY++ {
			for kernelX := 0; kernelX < 3 && !checkedMask; kernelX++ {
				for index, left := range first.masks[inputChannel][kernelY][kernelX] {
					if left != 0 {
						if delta := cmplx.Abs(second.masks[inputChannel][kernelY][kernelX][index]/left - complex(wantRatio, 0)); delta > 1e-14 {
							t.Fatalf("stem packed mask did not follow the supplied activation scale: delta=%g", delta)
						}
						checkedMask = true
						break
					}
				}
			}
		}
	}
	for outputChannel, left := range first.bias {
		if left != 0 {
			if math.Abs(second.bias[outputChannel]/left-wantRatio) > 1e-14 {
				t.Fatalf("stem folded bias did not follow the supplied activation scale")
			}
			checkedBias = true
			break
		}
	}
	if !checkedKernel || !checkedMask || !checkedBias {
		t.Fatal("frozen stem did not expose nonzero kernel/mask/bias samples for the scale audit")
	}

	for _, scale := range []float64{firstScale, secondScale} {
		polynomial := referenceResNetBlockReLUPolynomial(scale)
		unscaled := []complex128{-0.75, -0.25, 0, 0.25, 0.75}
		scaled := make([]complex128, len(unscaled))
		for index := range unscaled {
			scaled[index] = complex(scale, 0) * unscaled[index]
		}
		got, _, err := plainReferenceResNetBlockReLU(scaled, polynomial, scale)
		if err != nil {
			t.Fatal(err)
		}
		for index, value := range unscaled {
			want := math.Max(0, real(value))
			if delta := math.Abs(real(got[index]) - want); delta > 0.05 {
				t.Fatalf("degree-59 activation failed 1/scale restoration: scale=%g input=%g got=%g want=%g delta=%g", scale, real(value), real(got[index]), want, delta)
			}
		}
	}
	t.Logf("activation scale adaptation PASS: folded kernels/masks/biases scale by %.17g and degree-59 output restores the unscaled ReLU domain", wantRatio)
}
