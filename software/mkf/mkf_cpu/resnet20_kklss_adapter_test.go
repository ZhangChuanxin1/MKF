package mkfcpu

import (
	"fmt"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

// kklssLattigoAdapter lets the already frozen ResNet-20 packing/scheduling
// code operate on a degree-two carrier while every secret-dependent online
// operation remains a real public-KKLSS operation. Carrier polynomials are
// kept in NTT form for Lattigo's public linear operations. Conversion to the
// coefficient-domain representation expected by public KKLSS is explicit at
// the adapter boundary and changes no ciphertext/key semantics.
type kklssLattigoAdapter struct {
	ckks.Evaluator
	params                     ckks.Parameters
	context                    *kklssP2N2PrimitiveContext
	allowOnlineOwnerEncryption bool
	ownerCiphertexts           int
	mulRelinCalls              int
	polynomialMulRelinCalls    int
	inPolynomialEvaluation     bool
	rotationCalls              int
	polynomialCalls            int
	collectiveRefreshes        int
}

func newKKLSSLattigoAdapter(params ckks.Parameters, context *kklssP2N2PrimitiveContext, allowOnlineOwnerEncryption bool) (*kklssLattigoAdapter, error) {
	if context == nil || len(context.partyIDs) != fullResNet20Parties || len(context.partySecrets) != fullResNet20Parties {
		return nil, fmt.Errorf("KKLSS full-network adapter requires the exact two-party context")
	}
	if params.LogN() != context.params.LogN() || params.LogSlots() != context.params.LogSlots() || params.MaxLevel() != context.params.MaxLevel() || params.Scale() != context.params.Scale() {
		return nil, fmt.Errorf("ordinary and KKLSS parameter identities differ")
	}
	return &kklssLattigoAdapter{
		Evaluator:                  ckks.NewEvaluator(params, rlwe.EvaluationKey{}),
		params:                     params,
		context:                    context,
		allowOnlineOwnerEncryption: allowOnlineOwnerEncryption,
	}, nil
}

func (adapter *kklssLattigoAdapter) state(carrier *ckks.Ciphertext) (*mkckks.Ciphertext, error) {
	if carrier == nil || carrier.Degree() != len(adapter.context.partyIDs) {
		return nil, fmt.Errorf("KKLSS carrier degree=%d, want %d", func() int {
			if carrier == nil {
				return -1
			}
			return carrier.Degree()
		}(), len(adapter.context.partyIDs))
	}
	ids := mkrlwe.NewIDSet()
	for _, id := range adapter.context.partyIDs {
		ids.Add(id)
	}
	level := carrier.Level()
	state := mkckks.NewCiphertext(adapter.context.params, ids, level, carrier.Scale)
	ringQ := adapter.params.RingQ()
	copyComponent := func(source, destination *ring.Poly) {
		if source.IsNTT {
			ringQ.InvNTTLvl(level, source, destination)
		} else {
			destination.Copy(source)
		}
		destination.IsNTT = false
	}
	copyComponent(carrier.Value[0], state.Value["0"])
	for index, id := range adapter.context.partyIDs {
		copyComponent(carrier.Value[index+1], state.Value[id])
	}
	return state, nil
}

func (adapter *kklssLattigoAdapter) carrier(state *mkckks.Ciphertext) (*ckks.Ciphertext, error) {
	if state == nil || len(state.IDSet().Value) == 0 || len(state.IDSet().Value) > len(adapter.context.partyIDs) {
		return nil, fmt.Errorf("invalid KKLSS state for carrier conversion")
	}
	known := make(map[string]struct{}, len(adapter.context.partyIDs))
	for _, id := range adapter.context.partyIDs {
		known[id] = struct{}{}
	}
	for id := range state.IDSet().Value {
		if _, exists := known[id]; !exists {
			return nil, fmt.Errorf("KKLSS state contains unknown party %q", id)
		}
	}
	level := state.Level()
	carrier := ckks.NewCiphertext(adapter.params, len(adapter.context.partyIDs), level, state.Scale)
	ringQ := adapter.params.RingQ()
	copyComponent := func(source, destination *ring.Poly) {
		if source.IsNTT {
			destination.Copy(source)
		} else {
			ringQ.NTTLvl(level, source, destination)
		}
		destination.IsNTT = true
	}
	copyComponent(state.Value["0"], carrier.Value[0])
	for index, id := range adapter.context.partyIDs {
		component, exists := state.Value[id]
		if !exists {
			carrier.Value[index+1].Zero()
			carrier.Value[index+1].IsNTT = true
			continue
		}
		copyComponent(component, carrier.Value[index+1])
	}
	return carrier, nil
}

func (adapter *kklssLattigoAdapter) copyCarrier(output, input *ckks.Ciphertext) {
	output.Ciphertext.Resize(adapter.params.Parameters, input.Degree())
	output.Copy(input)
	output.Scale = input.Scale
}

func kklssCiphertextOperand(operand ckks.Operand) (*ckks.Ciphertext, error) {
	ciphertext, ok := operand.(*ckks.Ciphertext)
	if !ok {
		return nil, fmt.Errorf("KKLSS MulRelin requires two ciphertext operands, got %T", operand)
	}
	return ciphertext, nil
}

func (adapter *kklssLattigoAdapter) MulRelinNew(leftOperand, rightOperand ckks.Operand) *ckks.Ciphertext {
	leftCarrier, err := kklssCiphertextOperand(leftOperand)
	if err != nil {
		panic(err)
	}
	rightCarrier, err := kklssCiphertextOperand(rightOperand)
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
	result, err := adapter.carrier(adapter.context.evaluator.MulRelinNew(left, right, adapter.context.relin))
	if err != nil {
		panic(err)
	}
	if adapter.inPolynomialEvaluation {
		adapter.polynomialMulRelinCalls++
	} else {
		adapter.mulRelinCalls++
	}
	return result
}

func (adapter *kklssLattigoAdapter) MulRelin(left, right ckks.Operand, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.MulRelinNew(left, right))
}

func (adapter *kklssLattigoAdapter) publicProductOperands(left, right ckks.Operand) (*ckks.Ciphertext, ckks.Operand, error) {
	parties := len(adapter.context.partyIDs)
	if carrier, ok := left.(*ckks.Ciphertext); ok && carrier.Degree() == parties && right.Degree() == 0 {
		return carrier, right, nil
	}
	if carrier, ok := right.(*ckks.Ciphertext); ok && carrier.Degree() == parties && left.Degree() == 0 {
		return carrier, left, nil
	}
	return nil, nil, fmt.Errorf("public-KKLSS public Mul requires one degree-%d fixed-key-set carrier and one degree-0 operand, got %T/degree-%d and %T/degree-%d", parties, left, left.Degree(), right, right.Degree())
}

// MulNew applies a public plaintext/mask independently to c0,c_data,c_model.
// The ordinary evaluator cannot be used here because it interprets carrier
// degree two as a single-secret quadratic ciphertext. This component-wise
// product requires no KKLSS evaluation key and is not a network ciphertext
// product in the frozen 6,172 count.
func (adapter *kklssLattigoAdapter) MulNew(left, right ckks.Operand) *ckks.Ciphertext {
	carrier, public, err := adapter.publicProductOperands(left, right)
	if err != nil {
		panic(err)
	}
	level := carrier.Level()
	if public.Level() < level {
		level = public.Level()
	}
	if !public.El().Value[0].IsNTT {
		panic("public-KKLSS public Mul requires an NTT-domain public operand")
	}
	for component, polynomial := range carrier.Value {
		if !polynomial.IsNTT {
			panic(fmt.Sprintf("public-KKLSS carrier component %d is not in the NTT domain", component))
		}
	}
	result := ckks.NewCiphertext(adapter.params, len(adapter.context.partyIDs), level, carrier.Scale*public.ScalingFactor())
	ringQ := adapter.params.RingQ()
	publicMontgomery := ringQ.NewPolyLvl(level)
	ringQ.MFormLvl(level, public.El().Value[0], publicMontgomery)
	publicMontgomery.IsNTT = true
	for component := range result.Value {
		ringQ.MulCoeffsMontgomeryLvl(level, publicMontgomery, carrier.Value[component], result.Value[component])
		result.Value[component].IsNTT = true
	}
	return result
}

func (adapter *kklssLattigoAdapter) Mul(left, right ckks.Operand, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.MulNew(left, right))
}

func (adapter *kklssLattigoAdapter) RotateNew(input *ckks.Ciphertext, rotation int) *ckks.Ciphertext {
	state, err := adapter.state(input)
	if err != nil {
		panic(err)
	}
	result, err := adapter.carrier(adapter.context.evaluator.RotateNew(state, rotation, adapter.context.rotation))
	if err != nil {
		panic(err)
	}
	adapter.rotationCalls++
	return result
}

func (adapter *kklssLattigoAdapter) Rotate(input *ckks.Ciphertext, rotation int, output *ckks.Ciphertext) {
	adapter.copyCarrier(output, adapter.RotateNew(input, rotation))
}

func (adapter *kklssLattigoAdapter) RotateHoistedNew(input *ckks.Ciphertext, rotations []int) map[int]*ckks.Ciphertext {
	outputs := make(map[int]*ckks.Ciphertext, len(rotations))
	for _, rotation := range rotations {
		outputs[rotation] = adapter.RotateNew(input, rotation)
	}
	return outputs
}

func (adapter *kklssLattigoAdapter) RotateHoisted(input *ckks.Ciphertext, rotations []int, outputs map[int]*ckks.Ciphertext) {
	for _, rotation := range rotations {
		adapter.copyCarrier(outputs[rotation], adapter.RotateNew(input, rotation))
	}
}

func (adapter *kklssLattigoAdapter) EvaluatePoly(input *ckks.Ciphertext, polynomial *ckks.Polynomial, targetScale float64) (*ckks.Ciphertext, error) {
	if adapter.inPolynomialEvaluation {
		return nil, fmt.Errorf("nested KKLSS polynomial evaluation is forbidden")
	}
	adapter.inPolynomialEvaluation = true
	before := adapter.polynomialMulRelinCalls
	result, products, err := kklssCarrierEvaluatePolyBSGS(adapter, input, polynomial, targetScale)
	adapter.inPolynomialEvaluation = false
	if err != nil {
		return nil, err
	}
	observed := adapter.polynomialMulRelinCalls - before
	if products != p2gn16Degree59ProductsPerActivation || observed != products {
		return nil, fmt.Errorf("KKLSS BSGS product schedule=%d observed=%d, want %d", products, observed, p2gn16Degree59ProductsPerActivation)
	}
	adapter.polynomialCalls++
	return result, nil
}

// ownerCarrier currently permits direct encryption only for opt-in
// functional gates. A performance-mode prepared trial must provide an
// offline-staged ciphertext store and leaves this flag false, causing a hard
// failure on any accidental in-boundary encoding/encryption.
func (adapter *kklssLattigoAdapter) ownerCarrier(_ ckks.Encoder, _ []*rlwe.PublicKey, owner int, values []complex128) (*ckks.Ciphertext, error) {
	if !adapter.allowOnlineOwnerEncryption {
		return nil, fmt.Errorf("KKLSS owner ciphertext cache miss: encoding/encryption is forbidden in the online timing boundary")
	}
	if owner < 0 || owner >= len(adapter.context.partyIDs) || len(values) != adapter.params.Slots() {
		return nil, fmt.Errorf("invalid KKLSS owner=%d or slots=%d", owner, len(values))
	}
	message := mkckks.NewMessage(adapter.context.params)
	copy(message.Value, values)
	ciphertext := adapter.context.encryptor.EncryptMsgNew(message, adapter.context.public.GetPublicKey(adapter.context.partyIDs[owner]))
	adapter.ownerCiphertexts++
	return adapter.carrier(ciphertext)
}

func (adapter *kklssLattigoAdapter) collectiveRefresh(input *ckks.Ciphertext, _ []*rlwe.SecretKey, _ int, _ int, crs utils.PRNG) (*ckks.Ciphertext, error) {
	state, err := adapter.state(input)
	if err != nil {
		return nil, err
	}
	refreshed, _, err := refreshKKLSSMultiCiphertext(adapter.params, adapter.context.params, adapter.context.partyIDs, adapter.context.partySecrets, state, crs)
	if err != nil {
		return nil, err
	}
	carrier, err := adapter.carrier(refreshed)
	if err == nil {
		adapter.collectiveRefreshes++
	}
	return carrier, err
}

// oracleCarrier performs the final collective phase evaluation only after an
// online boundary. It never feeds plaintext or an aggregate secret back into
// evaluation. The returned degree-one ciphertext has a zero nonconstant
// component so the existing out-of-boundary numerical checker can decode it.
func (adapter *kklssLattigoAdapter) oracleCarrier(input *ckks.Ciphertext) (*ckks.Ciphertext, error) {
	state, err := adapter.state(input)
	if err != nil {
		return nil, err
	}
	level := state.Level()
	ringQ := adapter.params.RingQ()
	phase := state.Value["0"].CopyNew()
	if phase.IsNTT {
		ringQ.InvNTTLvl(level, phase, phase)
	}
	phase.IsNTT = false
	for _, id := range adapter.context.partyIDs {
		component := state.Value[id].CopyNew()
		if !component.IsNTT {
			ringQ.NTTLvl(level, component, component)
		}
		secret := adapter.context.secret.GetSecretKey(id)
		ringQ.MulCoeffsMontgomeryLvl(level, component, secret.Value.Q, component)
		ringQ.InvNTTLvl(level, component, component)
		ringQ.AddLvl(level, phase, component, phase)
	}
	ringQ.NTTLvl(level, phase, phase)
	phase.IsNTT = true
	zero := ringQ.NewPolyLvl(level)
	zero.Zero()
	zero.IsNTT = true
	oracle := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{phase, zero})
	oracle.Scale = state.Scale
	return oracle, nil
}

type kklssPhaseSnapshot struct {
	owners             int
	mulRelin           int
	polynomialMulRelin int
	rotations          int
	polynomials        int
	refreshes          int
}

func (adapter *kklssLattigoAdapter) operationSnapshot() kklssPhaseSnapshot {
	return kklssPhaseSnapshot{
		owners: adapter.ownerCiphertexts, mulRelin: adapter.mulRelinCalls,
		polynomialMulRelin: adapter.polynomialMulRelinCalls,
		rotations:          adapter.rotationCalls, polynomials: adapter.polynomialCalls,
		refreshes: adapter.collectiveRefreshes,
	}
}

func TestKKLSSSparseOwnerCarrierContract(t *testing.T) {
	ordinaryParams, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	kklssParams, err := kklssP2Parameters()
	if err != nil {
		t.Fatal(err)
	}
	context := &kklssP2N2PrimitiveContext{
		params: kklssParams, partyIDs: []string{"data-owner", "model-owner"},
		partySecrets: make([]*mkrlwe.SecretKey, fullResNet20Parties),
	}
	adapter, err := newKKLSSLattigoAdapter(ordinaryParams, context, false)
	if err != nil {
		t.Fatal(err)
	}
	idset := mkrlwe.NewIDSet()
	idset.Add("model-owner")
	sparse := mkckks.NewCiphertext(kklssParams, idset, 0, ordinaryParams.Scale())
	for coefficientIndex := 0; coefficientIndex < ordinaryParams.N(); coefficientIndex++ {
		sparse.Value["0"].Coeffs[0][coefficientIndex] = uint64(coefficientIndex + 1)
		sparse.Value["model-owner"].Coeffs[0][coefficientIndex] = uint64(3*coefficientIndex + 7)
	}
	carrier, err := adapter.carrier(sparse)
	if err != nil {
		t.Fatal(err)
	}
	if carrier.Degree() != fullResNet20Parties {
		t.Fatalf("sparse KKLSS carrier degree=%d, want %d", carrier.Degree(), fullResNet20Parties)
	}
	for _, coefficient := range carrier.Value[1].Coeffs[0] {
		if coefficient != 0 {
			t.Fatal("sparse model-owner carrier acquired a nonzero data-owner component")
		}
	}
	dense, err := adapter.state(carrier)
	if err != nil {
		t.Fatal(err)
	}
	if !dense.IDSet().Has("data-owner") || !dense.IDSet().Has("model-owner") || len(dense.IDSet().Value) != fullResNet20Parties {
		t.Fatalf("fixed-key-set carrier did not materialize the two-party ID set")
	}
	for coefficientIndex := 0; coefficientIndex < ordinaryParams.N(); coefficientIndex++ {
		if dense.Value["data-owner"].Coeffs[0][coefficientIndex] != 0 || dense.Value["model-owner"].Coeffs[0][coefficientIndex] != sparse.Value["model-owner"].Coeffs[0][coefficientIndex] {
			t.Fatalf("sparse owner semantics changed at coefficient %d", coefficientIndex)
		}
	}

	mask := make([]complex128, ordinaryParams.Slots())
	for index := range mask {
		if index%3 != 0 {
			mask[index] = 1
		}
	}
	plaintext := ckks.NewEncoder(ordinaryParams).EncodeNTTAtLvlNew(0, mask, ordinaryParams.LogSlots())
	masked := adapter.MulNew(carrier, plaintext)
	if masked.Degree() != fullResNet20Parties || masked.Level() != 0 || masked.Scale != carrier.Scale*plaintext.Scale {
		t.Fatalf("public-KKLSS mask metadata mismatch: degree=%d level=%d scale=%g", masked.Degree(), masked.Level(), masked.Scale)
	}
	ringQ := ordinaryParams.RingQ()
	publicMontgomery := ringQ.NewPolyLvl(0)
	ringQ.MFormLvl(0, plaintext.Value, publicMontgomery)
	for component := range carrier.Value {
		expected := ringQ.NewPolyLvl(0)
		ringQ.MulCoeffsMontgomeryLvl(0, publicMontgomery, carrier.Value[component], expected)
		for coefficientIndex := 0; coefficientIndex < ordinaryParams.N(); coefficientIndex++ {
			if masked.Value[component].Coeffs[0][coefficientIndex] != expected.Coeffs[0][coefficientIndex] {
				t.Fatalf("public-KKLSS mask component=%d coefficient=%d mismatch", component, coefficientIndex)
			}
		}
	}
}
