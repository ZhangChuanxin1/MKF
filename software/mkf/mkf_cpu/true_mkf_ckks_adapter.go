package mkfcpu

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// paperMKFLattigoAdapter lets the existing ResNet scheduling/packing helpers
// operate on degree-n carrier objects while preserving vector-secret MKF
// semantics. The embedded evaluator is used for public linear operations
// only. Ciphertext multiplication, polynomial evaluation, and rotation are
// overridden below and routed to paperMKFCKKSEvaluator.
type paperMKFLattigoAdapter struct {
	ckks.Evaluator
	online              *paperMKFCKKSEvaluator
	ownerCiphertexts    int
	collectiveRefreshes int
}

func newPaperMKFLattigoAdapter(online *paperMKFCKKSEvaluator) *paperMKFLattigoAdapter {
	return &paperMKFLattigoAdapter{
		Evaluator: ckks.NewEvaluator(online.params, rlwe.EvaluationKey{}),
		online:    online,
	}
}

func (adapter *paperMKFLattigoAdapter) state(ciphertext *ckks.Ciphertext) (*paperMKFCKKSCiphertext, error) {
	return paperMKFStateFromLinear(adapter.online.params, ciphertext, adapter.online.parties)
}

func (adapter *paperMKFLattigoAdapter) carrier(state *paperMKFCKKSCiphertext) *ckks.Ciphertext {
	return state.asLinearCiphertext(adapter.online.params)
}

func (adapter *paperMKFLattigoAdapter) copyCarrier(output, input *ckks.Ciphertext) {
	output.Ciphertext.Resize(adapter.online.params.Parameters, input.Degree())
	output.Copy(input)
	output.Scale = input.Scale
}

func paperMKFCiphertextOperand(operand ckks.Operand) (*ckks.Ciphertext, error) {
	ciphertext, ok := operand.(*ckks.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("proposed-MKF MulRelin requires two ciphertext operands, got %T", operand)
	}
	return ciphertext, nil
}

func (adapter *paperMKFLattigoAdapter) MulRelinNew(leftOperand, rightOperand ckks.Operand) *ckks.Ciphertext {
	leftCarrier, err := paperMKFCiphertextOperand(leftOperand)
	if err != nil {
		panic(err)
	}
	rightCarrier, err := paperMKFCiphertextOperand(rightOperand)
	if err != nil {
		panic(err)
	}
	left, err := adapter.state(leftCarrier)
	if err != nil {
		panic(err)
	}
	right, err := adapter.state(rightCarrier)
	if err != nil {
		panic(err)
	}
	result, err := adapter.online.MulRelinNew(left, right)
	if err != nil {
		panic(err)
	}
	return adapter.carrier(result)
}

func (adapter *paperMKFLattigoAdapter) MulRelin(left, right ckks.Operand, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.MulRelinNew(left, right))
}

// publicProductOperands identifies the n+1 carrier and the public degree-zero
// operand. Public packing masks must be applied component-wise; treating the
// carrier's degree as a single-secret polynomial degree would be incorrect
// and is rejected by Lattigo's ordinary evaluator.
func (adapter *paperMKFLattigoAdapter) publicProductOperands(left, right ckks.Operand) (*ckks.Ciphertext, ckks.Operand, error) {
	if carrier, ok := left.(*ckks.Ciphertext); ok && carrier.Degree() == adapter.online.parties && right.Degree() == 0 {
		return carrier, right, nil
	}
	if carrier, ok := right.(*ckks.Ciphertext); ok && carrier.Degree() == adapter.online.parties && left.Degree() == 0 {
		return carrier, left, nil
	}
	return nil, nil, fmt.Errorf("proposed-MKF public Mul requires one degree-%d carrier and one degree-0 operand, got %T/degree-%d and %T/degree-%d", adapter.online.parties, left, left.Degree(), right, right.Degree())
}

// MulNew applies a public plaintext/mask independently to c0,...,cn. It does
// not invoke an evaluation key and therefore does not increment the proposed-
// MKF ciphertext-product counters.
func (adapter *paperMKFLattigoAdapter) MulNew(left, right ckks.Operand) *ckks.Ciphertext {
	carrier, public, err := adapter.publicProductOperands(left, right)
	if err != nil {
		panic(err)
	}
	level := carrier.Level()
	if public.Level() < level {
		level = public.Level()
	}
	if !public.El().Value[0].IsNTT {
		panic("proposed-MKF public Mul requires an NTT-domain public operand")
	}
	for component, polynomial := range carrier.Value {
		if !polynomial.IsNTT {
			panic(fmt.Sprintf("proposed-MKF carrier component %d is not in the NTT domain", component))
		}
	}
	result := ckks.NewCiphertext(adapter.online.params, adapter.online.parties, level, carrier.Scale*public.ScalingFactor())
	ringQ := adapter.online.params.RingQ()
	publicMontgomery := ringQ.NewPolyLvl(level)
	ringQ.MFormLvl(level, public.El().Value[0], publicMontgomery)
	publicMontgomery.IsNTT = true
	for component := range result.Value {
		ringQ.MulCoeffsMontgomeryLvl(level, publicMontgomery, carrier.Value[component], result.Value[component])
		result.Value[component].IsNTT = true
	}
	return result
}

func (adapter *paperMKFLattigoAdapter) Mul(left, right ckks.Operand, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.MulNew(left, right))
}

func (adapter *paperMKFLattigoAdapter) RotateNew(input *ckks.Ciphertext, rotation int) *ckks.Ciphertext {
	state, err := adapter.state(input)
	if err != nil {
		panic(err)
	}
	result, err := adapter.online.RotateNew(state, rotation)
	if err != nil {
		panic(err)
	}
	return adapter.carrier(result)
}

func (adapter *paperMKFLattigoAdapter) Rotate(input *ckks.Ciphertext, rotation int, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.RotateNew(input, rotation))
}

func (adapter *paperMKFLattigoAdapter) RotateHoistedNew(input *ckks.Ciphertext, rotations []int) map[int]*ckks.Ciphertext {
	outputs := make(map[int]*ckks.Ciphertext, len(rotations))
	for _, rotation := range rotations {
		outputs[rotation] = adapter.RotateNew(input, rotation)
	}
	return outputs
}

func (adapter *paperMKFLattigoAdapter) RotateHoisted(input *ckks.Ciphertext, rotations []int, outputs map[int]*ckks.Ciphertext) {
	for _, rotation := range rotations {
		adapter.copyCarrier(outputs[rotation], adapter.RotateNew(input, rotation))
	}
}

func (adapter *paperMKFLattigoAdapter) EvaluatePoly(input *ckks.Ciphertext, polynomial *ckks.Polynomial, targetScale float64) (*ckks.Ciphertext, error) {
	state, err := adapter.state(input)
	if err != nil {
		return nil, err
	}
	result, err := adapter.online.EvaluatePoly(state, polynomial, targetScale)
	if err != nil {
		return nil, err
	}
	return adapter.carrier(result), nil
}

func paperMKFOwnerCarrier(params ckks.Parameters, encoder ckks.Encoder, public []*rlwe.PublicKey, owner int, values []complex128) (*ckks.Ciphertext, error) {
	multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, owner, values)
	if err != nil {
		return nil, err
	}
	state, err := newPaperMKFCKKSStateFromMulti(params, multi)
	if err != nil {
		return nil, err
	}
	return state.asLinearCiphertext(params), nil
}

// ownerCarrier creates a sparse n+1 owner ciphertext and records the logical
// encrypted-model/input object.  The counter is used only for structural
// coverage checks; it is not a performance metric.
func (adapter *paperMKFLattigoAdapter) ownerCarrier(encoder ckks.Encoder, public []*rlwe.PublicKey, owner int, values []complex128) (*ckks.Ciphertext, error) {
	result, err := paperMKFOwnerCarrier(adapter.online.params, encoder, public, owner, values)
	if err == nil {
		adapter.ownerCiphertexts++
	}
	return result, err
}

// foldCarrierToAggregate applies only the public party-specific folding keys;
// it neither receives nor reconstructs the aggregate secret.
func (adapter *paperMKFLattigoAdapter) foldCarrierToAggregate(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
	state, err := adapter.state(input)
	if err != nil {
		return nil, err
	}
	folded, err := fold(adapter.online.params.Parameters, state.asMulti(), adapter.online.keys.Fold, true, adapter.online.switcher, &adapter.online.profile)
	if err != nil {
		return nil, err
	}
	compact := ckks.NewCiphertextAtLevelFromPoly(state.Level, folded)
	compact.Scale = state.Scale
	return compact, nil
}

func (adapter *paperMKFLattigoAdapter) materializeAggregate(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
	if input == nil || input.Degree() != 1 {
		return nil, fmt.Errorf("aggregate materialization requires degree-1 ciphertext")
	}
	state := &paperMKFCKKSCiphertext{
		C0:         copyPolyAtLevel(input.Level(), input.Value[0]),
		Components: make([]*ring.Poly, adapter.online.parties),
		Level:      input.Level(),
		Scale:      input.Scale,
	}
	for i := range state.Components {
		state.Components[i] = copyPolyAtLevel(input.Level(), input.Value[1])
	}
	return adapter.carrier(state), nil
}

// collectiveRefresh performs public folding, the standard two-party dCKKS
// masked decrypt--recrypt protocol, and repeated-Gamma materialization. The
// online function never forms s_E. Tests may reconstruct it only after this
// method returns as an out-of-boundary correctness oracle.
func (adapter *paperMKFLattigoAdapter) collectiveRefresh(input *ckks.Ciphertext, shares []*rlwe.SecretKey, minLevel, logBound int, crs utils.PRNG) (*ckks.Ciphertext, error) {
	compact, err := adapter.foldCarrierToAggregate(input)
	if err != nil {
		return nil, err
	}
	if compact.Level() < minLevel {
		return nil, fmt.Errorf("refresh input level %d is below minimum %d", compact.Level(), minLevel)
	}
	if compact.Level() > minLevel {
		adapter.Evaluator.DropLevel(compact, compact.Level()-minLevel)
	}
	refreshed, err := runCollectiveRefresh(adapter.online.params, shares, compact, logBound, crs)
	if err != nil {
		return nil, err
	}
	result, err := adapter.materializeAggregate(refreshed)
	if err == nil {
		adapter.collectiveRefreshes++
	}
	return result, err
}
