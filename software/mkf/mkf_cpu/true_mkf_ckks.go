package mkfcpu

// This file contains the online algebra needed by the paper's proposed
// aggregate-domain MKF-CKKS construction.  It deliberately does not reuse
// Lattigo's ordinary aggregate-key MulRelinNew or RotateNew paths.
//
// A ciphertext remains an (n+1)-component object decrypting as
//
//     c0 + sum_i c_i s_i.
//
// Multiplication folds both operands with the party-specific s_i -> s_E
// switching keys, multiplies the two aggregate-domain pairs, relinearizes the
// s_E^2 term once, and materializes Gamma in every nonconstant component.
// Rotation applies the automorphism to every component and accumulates the
// party-specific sigma_kappa(s_i) -> s_E switching-key outputs before the same
// repeated-Gamma materialization.  Standard CKKS code is used only for public
// linear operations and modulus rescaling, neither of which consumes an
// evaluation key.

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

const paperMKFCKKSConstructionID = "MKF-aggregate-fold+nplus1-repeated-gamma-v1"

// paperMKFCKKSCiphertext is intentionally distinct from ckks.Ciphertext.
// Lattigo interprets a degree-n ciphertext as powers of one secret, whereas
// this object has the vector-secret phase c0 + sum_i c_i*s_i.
type paperMKFCKKSCiphertext struct {
	C0         *ring.Poly
	Components []*ring.Poly
	Level      int
	Scale      float64
}

type paperMKFCKKSOnlineKeys struct {
	Fold      []*rlwe.SwitchingKey
	Relin     *rlwe.RelinearizationKey
	Rotations map[uint64][]*rlwe.SwitchingKey
}

type paperMKFCKKSEvaluator struct {
	params   ckks.Parameters
	parties  int
	keys     paperMKFCKKSOnlineKeys
	switcher *switchWorkspace
	linear   ckks.Evaluator
	profile  Profile
	counts   paperMKFCKKSOperationCounts
}

// paperMKFCKKSOperationCounts is an algebraic audit trail, not a timing
// result.  It records only successfully completed online primitives.
type paperMKFCKKSOperationCounts struct {
	MulRelinCalls   int
	RotationCalls   int
	PolynomialCalls int
}

func newPaperMKFCKKSEvaluator(params ckks.Parameters, parties int, keys paperMKFCKKSOnlineKeys) (*paperMKFCKKSEvaluator, error) {
	if parties < 2 {
		return nil, fmt.Errorf("paper MKF evaluator requires at least two parties")
	}
	if len(keys.Fold) != parties {
		return nil, fmt.Errorf("fold-key count mismatch: got %d want %d", len(keys.Fold), parties)
	}
	if keys.Relin == nil || len(keys.Relin.Keys) == 0 || keys.Relin.Keys[0] == nil {
		return nil, fmt.Errorf("aggregate relinearization key is missing")
	}
	return &paperMKFCKKSEvaluator{
		params:   params,
		parties:  parties,
		keys:     keys,
		switcher: newSwitchWorkspace(params.Parameters),
		linear:   ckks.NewEvaluator(params, rlwe.EvaluationKey{}),
	}, nil
}

func copyPolyAtLevel(level int, source *ring.Poly) *ring.Poly {
	result := ring.NewPoly(len(source.Coeffs[0]), level+1)
	ring.CopyLvl(level, source, result)
	result.IsNTT = source.IsNTT
	return result
}

func newPaperMKFCKKSStateFromMulti(params ckks.Parameters, input *multiCiphertext) (*paperMKFCKKSCiphertext, error) {
	if input == nil || input.C0 == nil || input.Level < 0 || input.Scale <= 0 {
		return nil, fmt.Errorf("invalid multi-key ciphertext")
	}
	state := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(input.Level, input.C0),
		Components: make([]*ring.Poly, len(input.Components)),
		Level:      input.Level,
		Scale:      input.Scale,
	}
	for i, component := range input.Components {
		if component == nil {
			return nil, fmt.Errorf("nil component %d", i)
		}
		state.Components[i] = copyPolyAtLevel(input.Level, component)
	}
	return state, nil
}

func (state *paperMKFCKKSCiphertext) copyNew() *paperMKFCKKSCiphertext {
	result := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(state.Level, state.C0),
		Components: make([]*ring.Poly, len(state.Components)),
		Level:      state.Level,
		Scale:      state.Scale,
	}
	for i := range state.Components {
		result.Components[i] = copyPolyAtLevel(state.Level, state.Components[i])
	}
	return result
}

func (state *paperMKFCKKSCiphertext) asMulti() *multiCiphertext {
	return &multiCiphertext{C0: state.C0, Components: state.Components, Level: state.Level, Scale: state.Scale}
}

// asLinearCiphertext is used only for public linear operations and rescaling.
// It must never be passed to Lattigo's decryptor, relinearizer, key switcher,
// polynomial evaluator, or rotation evaluator because its components have
// vector-secret rather than single-secret-power semantics.
func (state *paperMKFCKKSCiphertext) asLinearCiphertext(params ckks.Parameters) *ckks.Ciphertext {
	result := ckks.NewCiphertext(params, len(state.Components), state.Level, state.Scale)
	ring.CopyLvl(state.Level, state.C0, result.Value[0])
	result.Value[0].IsNTT = state.C0.IsNTT
	for i := range state.Components {
		ring.CopyLvl(state.Level, state.Components[i], result.Value[i+1])
		result.Value[i+1].IsNTT = state.Components[i].IsNTT
	}
	return result
}

func paperMKFStateFromLinear(params ckks.Parameters, ciphertext *ckks.Ciphertext, parties int) (*paperMKFCKKSCiphertext, error) {
	if ciphertext == nil || ciphertext.Degree() != parties {
		degree := -1
		if ciphertext != nil {
			degree = ciphertext.Degree()
		}
		return nil, fmt.Errorf("linear ciphertext degree mismatch: got %d want %d", degree, parties)
	}
	state := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(ciphertext.Level(), ciphertext.Value[0]),
		Components: make([]*ring.Poly, parties),
		Level:      ciphertext.Level(),
		Scale:      ciphertext.Scale,
	}
	for i := range state.Components {
		state.Components[i] = copyPolyAtLevel(ciphertext.Level(), ciphertext.Value[i+1])
	}
	return state, nil
}

func (evaluator *paperMKFCKKSEvaluator) validateState(state *paperMKFCKKSCiphertext) error {
	if state == nil || state.C0 == nil || state.Level < 0 || state.Scale <= 0 {
		return fmt.Errorf("invalid paper MKF ciphertext")
	}
	if len(state.Components) != evaluator.parties {
		return fmt.Errorf("component count mismatch: got %d want %d", len(state.Components), evaluator.parties)
	}
	for i, component := range state.Components {
		if component == nil {
			return fmt.Errorf("nil component %d", i)
		}
	}
	return nil
}

func (evaluator *paperMKFCKKSEvaluator) newZero(level int, scale float64) *paperMKFCKKSCiphertext {
	ringQ := evaluator.params.RingQ()
	state := &paperMKFCKKSCiphertext{C0: ringQ.NewPolyLvl(level), Components: make([]*ring.Poly, evaluator.parties), Level: level, Scale: scale}
	state.C0.IsNTT = true
	for i := range state.Components {
		state.Components[i] = ringQ.NewPolyLvl(level)
		state.Components[i].IsNTT = true
	}
	return state
}

func (evaluator *paperMKFCKKSEvaluator) AddNew(left, right *paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(left); err != nil {
		return nil, err
	}
	if err := evaluator.validateState(right); err != nil {
		return nil, err
	}
	leftLinear, rightLinear := left.asLinearCiphertext(evaluator.params), right.asLinearCiphertext(evaluator.params)
	return paperMKFStateFromLinear(evaluator.params, evaluator.linear.AddNew(leftLinear, rightLinear), evaluator.parties)
}

func (evaluator *paperMKFCKKSEvaluator) SubNew(left, right *paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(left); err != nil {
		return nil, err
	}
	if err := evaluator.validateState(right); err != nil {
		return nil, err
	}
	leftLinear, rightLinear := left.asLinearCiphertext(evaluator.params), right.asLinearCiphertext(evaluator.params)
	return paperMKFStateFromLinear(evaluator.params, evaluator.linear.SubNew(leftLinear, rightLinear), evaluator.parties)
}

func (evaluator *paperMKFCKKSEvaluator) AddConstNew(input *paperMKFCKKSCiphertext, constant interface{}) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	linear := input.asLinearCiphertext(evaluator.params)
	evaluator.linear.AddConst(linear, constant, linear)
	return paperMKFStateFromLinear(evaluator.params, linear, evaluator.parties)
}

func (evaluator *paperMKFCKKSEvaluator) MultByConstNew(input *paperMKFCKKSCiphertext, constant interface{}) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	linear := input.asLinearCiphertext(evaluator.params)
	evaluator.linear.MultByConst(linear, constant, linear)
	return paperMKFStateFromLinear(evaluator.params, linear, evaluator.parties)
}

func (evaluator *paperMKFCKKSEvaluator) RescaleNew(input *paperMKFCKKSCiphertext, targetScale float64) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	linear := input.asLinearCiphertext(evaluator.params)
	result := ckks.NewCiphertext(evaluator.params, linear.Degree(), linear.Level(), linear.Scale)
	if err := evaluator.linear.Rescale(linear, targetScale, result); err != nil {
		return nil, err
	}
	return paperMKFStateFromLinear(evaluator.params, result, evaluator.parties)
}

func (evaluator *paperMKFCKKSEvaluator) DropLevelNew(input *paperMKFCKKSCiphertext, levels int) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	if levels < 0 || input.Level-levels < 0 {
		return nil, fmt.Errorf("invalid level drop %d from level %d", levels, input.Level)
	}
	linear := input.asLinearCiphertext(evaluator.params)
	evaluator.linear.DropLevel(linear, levels)
	return paperMKFStateFromLinear(evaluator.params, linear, evaluator.parties)
}

// MulRelinNew is the only ciphertext-ciphertext multiplication primitive used
// by the paper-faithful runtime and by its polynomial evaluator.
func (evaluator *paperMKFCKKSEvaluator) MulRelinNew(leftInput, rightInput *paperMKFCKKSCiphertext) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(leftInput); err != nil {
		return nil, err
	}
	if err := evaluator.validateState(rightInput); err != nil {
		return nil, err
	}
	left, err := fold(evaluator.params.Parameters, leftInput.asMulti(), evaluator.keys.Fold, true, evaluator.switcher, &evaluator.profile)
	if err != nil {
		return nil, err
	}
	right, err := fold(evaluator.params.Parameters, rightInput.asMulti(), evaluator.keys.Fold, true, evaluator.switcher, &evaluator.profile)
	if err != nil {
		return nil, err
	}
	level := leftInput.Level
	if rightInput.Level < level {
		level = rightInput.Level
	}
	ringQ := evaluator.params.RingQ()
	mu0, mu1, mu2 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftM0, leftM1 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftSum, rightSum := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	measure(&evaluator.profile.Tensor, func() {
		ringQ.MFormLvl(level, left[0], leftM0)
		ringQ.MulCoeffsMontgomeryLvl(level, leftM0, right[0], mu0)
		ringQ.MFormLvl(level, left[1], leftM1)
		ringQ.MulCoeffsMontgomeryLvl(level, leftM1, right[1], mu2)
		ringQ.AddLvl(level, left[0], left[1], leftSum)
		ringQ.AddLvl(level, right[0], right[1], rightSum)
		ringQ.MFormLvl(level, leftSum, leftSum)
		ringQ.MulCoeffsMontgomeryLvl(level, leftSum, rightSum, mu1)
		ringQ.SubLvl(level, mu1, mu0, mu1)
		ringQ.SubLvl(level, mu1, mu2, mu1)
	})
	mu0.IsNTT, mu1.IsNTT, mu2.IsNTT = true, true, true
	relin0, relin1 := evaluator.switcher.switchOne(evaluator.params.Parameters, level, mu2, evaluator.keys.Relin.Keys[0], true, &evaluator.profile)
	ringQ.AddLvl(level, mu0, relin0, mu0)
	ringQ.AddLvl(level, mu1, relin1, mu1)

	result := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(level, mu0),
		Components: make([]*ring.Poly, evaluator.parties),
		Level:      level,
		Scale:      leftInput.Scale * rightInput.Scale,
	}
	for i := range result.Components {
		result.Components[i] = copyPolyAtLevel(level, mu1)
	}
	evaluator.counts.MulRelinCalls++
	return result, nil
}

func mkfAutomorphismNTT(params rlwe.Parameters, level int, input *ring.Poly, galoisElement uint64) *ring.Poly {
	ringQ := params.RingQ()
	// ring.Ring.Permute in Lattigo v2.3 iterates over the ring's complete
	// modulus chain and has no level-restricted variant.  Allocate full-chain
	// scratch polynomials even when the active ciphertext level is lower.
	coefficientInput := ringQ.NewPoly()
	coefficientOutput := ringQ.NewPoly()
	nttOutput := ringQ.NewPoly()
	ringQ.InvNTTLvl(level, input, coefficientInput)
	ringQ.Permute(coefficientInput, galoisElement, coefficientOutput)
	ringQ.NTTLvl(level, coefficientOutput, nttOutput)
	nttOutput.IsNTT = true
	return nttOutput
}

// RotateNew uses one sigma_kappa(s_i)->s_E key for each party and never calls
// ckks.Evaluator.RotateNew.
func (evaluator *paperMKFCKKSEvaluator) RotateNew(input *paperMKFCKKSCiphertext, rotation int) (*paperMKFCKKSCiphertext, error) {
	if err := evaluator.validateState(input); err != nil {
		return nil, err
	}
	galoisElement := evaluator.params.GaloisElementForColumnRotationBy(rotation)
	rotationKeys, exists := evaluator.keys.Rotations[galoisElement]
	if !exists || len(rotationKeys) != evaluator.parties {
		return nil, fmt.Errorf("party-specific Galois keys missing for rotation %d (element %d)", rotation, galoisElement)
	}
	ringQ := evaluator.params.RingQ()
	accumulated0 := mkfAutomorphismNTT(evaluator.params.Parameters, input.Level, input.C0, galoisElement)
	accumulated1 := ringQ.NewPolyLvl(input.Level)
	accumulated1.IsNTT = true
	for i, component := range input.Components {
		permuted := mkfAutomorphismNTT(evaluator.params.Parameters, input.Level, component, galoisElement)
		u0, u1 := evaluator.switcher.switchOne(evaluator.params.Parameters, input.Level, permuted, rotationKeys[i], true, &evaluator.profile)
		ringQ.AddLvl(input.Level, accumulated0, u0, accumulated0)
		ringQ.AddLvl(input.Level, accumulated1, u1, accumulated1)
	}
	result := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(input.Level, accumulated0),
		Components: make([]*ring.Poly, evaluator.parties),
		Level:      input.Level,
		Scale:      input.Scale,
	}
	for i := range result.Components {
		result.Components[i] = copyPolyAtLevel(input.Level, accumulated1)
	}
	evaluator.counts.RotationCalls++
	return result, nil
}

func mkfPermutedSecret(params rlwe.Parameters, secret *rlwe.SecretKey, galoisElement uint64) *rlwe.SecretKey {
	result := rlwe.NewSecretKey(params)
	index := ring.PermuteNTTIndex(galoisElement, uint64(params.N()))
	ring.PermuteNTTWithIndexLvl(params.QCount()-1, secret.Value.Q, index, result.Value.Q)
	ring.PermuteNTTWithIndexLvl(params.PCount()-1, secret.Value.P, index, result.Value.P)
	return result
}

// newPaperMKFCKKSTestMaterial centrally forms s_E and therefore is test-only.
// It validates the online algebra, but it is not evidence for the manuscript's
// distributed EvalKeyGen transcript or its circular/KDM-security assumption.
func newPaperMKFCKKSTestMaterial(params ckks.Parameters, sources []*rlwe.SecretKey, rotations []int) (paperMKFCKKSOnlineKeys, *rlwe.SecretKey, error) {
	if len(sources) < 2 {
		return paperMKFCKKSOnlineKeys{}, nil, fmt.Errorf("test material requires at least two source secrets")
	}
	aggregate := aggregateSecret(params.Parameters, sources)
	keyGenerator := ckks.NewKeyGenerator(params)
	ordinary := generateOnlineKeys(params.Parameters, keyGenerator, sources, aggregate)
	keys := paperMKFCKKSOnlineKeys{
		Fold:      ordinary.Fold,
		Relin:     ordinary.Relin,
		Rotations: make(map[uint64][]*rlwe.SwitchingKey),
	}
	for _, rotation := range rotations {
		if rotation == 0 {
			continue
		}
		galoisElement := params.GaloisElementForColumnRotationBy(rotation)
		if _, exists := keys.Rotations[galoisElement]; exists {
			continue
		}
		partyKeys := make([]*rlwe.SwitchingKey, len(sources))
		for i := range sources {
			partyKeys[i] = keyGenerator.GenSwitchingKey(mkfPermutedSecret(params.Parameters, sources[i], galoisElement), aggregate)
		}
		keys.Rotations[galoisElement] = partyKeys
	}
	return keys, aggregate, nil
}

func paperMKFPhase(params ckks.Parameters, state *paperMKFCKKSCiphertext, sourceSecrets []*rlwe.SecretKey) (*ring.Poly, error) {
	if len(state.Components) != len(sourceSecrets) {
		return nil, fmt.Errorf("component/secret count mismatch")
	}
	ringQ := params.RingQ()
	phase := copyPolyAtLevel(state.Level, state.C0)
	for i, component := range state.Components {
		product := ringQ.NewPolyLvl(state.Level)
		// Ciphertext coefficients are in the ordinary NTT representation and
		// secret-key coefficients are already in Montgomery form, exactly as
		// expected by rlwe.Decryptor.
		ringQ.MulCoeffsMontgomeryLvl(state.Level, component, sourceSecrets[i].Value.Q, product)
		ringQ.AddLvl(state.Level, phase, product, phase)
	}
	phase.IsNTT = true
	return phase, nil
}
