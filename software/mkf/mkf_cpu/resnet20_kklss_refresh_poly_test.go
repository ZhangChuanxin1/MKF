package mkfcpu

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

// kklssCollectiveRefreshEvidence records protocol shape and numerical output
// for a functional gate. The CKS and Refresh transcript sizes are not yet
// serialized, so these counts must not be presented as measured bytes.
type kklssCollectiveRefreshEvidence struct {
	InputLevel           int
	OutputLevel          int
	LogBound             int
	CKSSourceConversions int
	CKSShareMessages     int
	RefreshShareMessages int
}

func kklssIndividualRLWEShares(secrets []*mkrlwe.SecretKey) ([]*rlwe.SecretKey, error) {
	if len(secrets) < 2 {
		return nil, fmt.Errorf("KKLSS collective refresh requires at least two individual secrets")
	}
	shares := make([]*rlwe.SecretKey, len(secrets))
	for index, secret := range secrets {
		if secret == nil {
			return nil, fmt.Errorf("KKLSS party %d has a nil individual secret", index)
		}
		// SecretKey embeds rlwe.SecretKey by value. Taking this address exposes
		// only the same party's share to its protocol instance; no aggregate
		// secret is constructed by the bridge.
		shares[index] = &secret.SecretKey
	}
	return shares, nil
}

func kklssCiphertextToMulti(input *mkckks.Ciphertext, partyIDs []string) (*multiCiphertext, error) {
	if input == nil || len(partyIDs) < 2 {
		return nil, fmt.Errorf("invalid KKLSS ciphertext or party list")
	}
	ids := input.IDSet()
	if len(ids.Value) != len(partyIDs) {
		return nil, fmt.Errorf("KKLSS ciphertext ID count=%d, want %d", len(ids.Value), len(partyIDs))
	}
	out := &multiCiphertext{
		C0: input.Value["0"].CopyNew(), Components: make([]*ring.Poly, len(partyIDs)),
		Level: input.Level(), Scale: input.Scale,
	}
	for index, id := range partyIDs {
		component, exists := input.Value[id]
		if !exists || !ids.Has(id) {
			return nil, fmt.Errorf("KKLSS ciphertext is missing party component %q", id)
		}
		out.Components[index] = component.CopyNew()
	}
	return out, nil
}

// kklssMaterializeAggregateCompact maps a compact ciphertext under
// s_E=sum_i s_i back to the KKLSS n+1 tuple by assigning Gamma to every party
// component. Its phase is therefore c0+sum_i Gamma*s_i=c0+Gamma*s_E. KKLSS
// multiplication and rotation expect coefficient-domain polynomials, so the
// dckks NTT output is converted back before publication.
func kklssMaterializeAggregateCompact(kklssParams mkckks.Parameters, compact *ckks.Ciphertext, partyIDs []string) (*mkckks.Ciphertext, error) {
	if compact == nil || compact.Degree() != 1 {
		return nil, fmt.Errorf("materialization requires a compact degree-1 ciphertext")
	}
	idset := mkrlwe.NewIDSet()
	for _, id := range partyIDs {
		if id == "" || id == "0" || idset.Has(id) {
			return nil, fmt.Errorf("invalid or duplicate KKLSS party ID %q", id)
		}
		idset.Add(id)
	}
	level := compact.Level()
	ringQ := kklssParams.RingQ()
	c0, gamma := compact.Value[0].CopyNew(), compact.Value[1].CopyNew()
	if c0.IsNTT {
		ringQ.InvNTTLvl(level, c0, c0)
	}
	if gamma.IsNTT {
		ringQ.InvNTTLvl(level, gamma, gamma)
	}
	c0.IsNTT, gamma.IsNTT = false, false
	out := mkckks.NewCiphertext(kklssParams, idset, level, compact.Scale)
	out.Value["0"].Copy(c0)
	out.Value["0"].IsNTT = false
	for _, id := range partyIDs {
		out.Value[id].Copy(gamma)
		out.Value[id].IsNTT = false
	}
	return out, nil
}

// refreshKKLSSMultiCiphertext is a protocol composition, not a centralized
// decrypt--reencrypt shortcut:
//
//  1. For each KKLSS term c_i*s_i, all parties run CKS from the sparse input
//     sharing (only party i owns s_i) to the additive output sharing s_E.
//  2. The converted terms are summed to a compact ciphertext under s_E.
//  3. The same individual shares run dckks Refresh.
//  4. The compact result is materialized as (c0,Gamma,...,Gamma), retaining
//     the original KKLSS ID set and phase.
//
// No function on this path calls aggregateSecret or accepts an aggregate
// secret key. The individual KKLSS secrets are passed only to their own CKS
// and Refresh protocol instances.
func foldKKLSSMultiCiphertext(ordinaryParams ckks.Parameters, partyIDs []string, partySecrets []*mkrlwe.SecretKey, input *mkckks.Ciphertext) (*ckks.Ciphertext, error) {
	if len(partyIDs) != len(partySecrets) {
		return nil, fmt.Errorf("KKLSS party ID/secret count mismatch: %d != %d", len(partyIDs), len(partySecrets))
	}
	shares, err := kklssIndividualRLWEShares(partySecrets)
	if err != nil {
		return nil, err
	}
	multi, err := kklssCiphertextToMulti(input, partyIDs)
	if err != nil {
		return nil, err
	}
	// The public KKLSS evaluator stores ciphertexts in coefficient form,
	// whereas ckks.NewCiphertext initializes representation flags to NTT.
	// dckks.CKS supports either form but relies on those flags. Assert one
	// common representation and restore the copied output metadata below;
	// this changes no coefficient and performs no hidden transform.
	inputIsNTT := multi.C0.IsNTT
	for index, component := range multi.Components {
		if component.IsNTT != inputIsNTT {
			return nil, fmt.Errorf("KKLSS component %d representation differs from c0", index)
		}
	}
	compact, err := interactiveAggregateFold(ordinaryParams, multi, shares)
	if err != nil {
		return nil, fmt.Errorf("KKLSS multi-component to aggregate-domain CKS: %w", err)
	}
	compact.Value[0].IsNTT, compact.Value[1].IsNTT = inputIsNTT, inputIsNTT
	return compact, nil
}

func refreshKKLSSMultiCiphertext(ordinaryParams ckks.Parameters, kklssParams mkckks.Parameters, partyIDs []string, partySecrets []*mkrlwe.SecretKey, input *mkckks.Ciphertext, crs utils.PRNG) (*mkckks.Ciphertext, kklssCollectiveRefreshEvidence, error) {
	shares, err := kklssIndividualRLWEShares(partySecrets)
	if err != nil {
		return nil, kklssCollectiveRefreshEvidence{}, err
	}
	compact, err := foldKKLSSMultiCiphertext(ordinaryParams, partyIDs, partySecrets, input)
	if err != nil {
		return nil, kklssCollectiveRefreshEvidence{}, err
	}
	// dckks.Refresh is defined on the ordinary CKKS NTT representation. The
	// public KKLSS evaluator and the CKS bridge above intentionally retain
	// coefficient form, so perform the explicit public representation change
	// before generating refresh shares. This is not a key switch and does not
	// alter the encrypted phase.
	if !compact.Value[0].IsNTT {
		ringQ := ordinaryParams.RingQ()
		ringQ.NTTLvl(compact.Level(), compact.Value[0], compact.Value[0])
		ringQ.NTTLvl(compact.Level(), compact.Value[1], compact.Value[1])
		compact.Value[0].IsNTT, compact.Value[1].IsNTT = true, true
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, ordinaryParams.Scale(), len(shares), ordinaryParams.Q())
	if !ok {
		return nil, kklssCollectiveRefreshEvidence{}, fmt.Errorf("P2 lacks levels for the KKLSS two-party collective refresh")
	}
	inputLevel := compact.Level()
	if compact.Level() > minLevel {
		ckks.NewEvaluator(ordinaryParams, rlwe.EvaluationKey{}).DropLevel(compact, compact.Level()-minLevel)
	}
	refreshed, err := runCollectiveRefresh(ordinaryParams, shares, compact, logBound, crs)
	if err != nil {
		return nil, kklssCollectiveRefreshEvidence{}, fmt.Errorf("KKLSS aggregate-domain dckks refresh: %w", err)
	}
	out, err := kklssMaterializeAggregateCompact(kklssParams, refreshed, partyIDs)
	if err != nil {
		return nil, kklssCollectiveRefreshEvidence{}, err
	}
	return out, kklssCollectiveRefreshEvidence{
		InputLevel: inputLevel, OutputLevel: out.Level(), LogBound: logBound,
		CKSSourceConversions: len(partyIDs), CKSShareMessages: len(partyIDs) * len(partyIDs),
		RefreshShareMessages: len(partyIDs),
	}, nil
}

func kklssAddConstantAllSlots(ordinaryParams ckks.Parameters, ciphertext *mkckks.Ciphertext, constant complex128) error {
	if ciphertext == nil {
		return fmt.Errorf("cannot add a constant to a nil KKLSS ciphertext")
	}
	values := make([]complex128, ordinaryParams.Slots())
	for index := range values {
		values[index] = constant
	}
	plaintext := ckks.NewPlaintext(ordinaryParams, ciphertext.Level(), ciphertext.Scale)
	encoder := ckks.NewEncoder(ordinaryParams)
	if ciphertext.Value["0"].IsNTT {
		encoder.EncodeNTT(plaintext, values, ordinaryParams.LogSlots())
	} else {
		encoder.Encode(plaintext, values, ordinaryParams.LogSlots())
	}
	ordinaryParams.RingQ().AddLvl(ciphertext.Level(), ciphertext.Value["0"], plaintext.Value, ciphertext.Value["0"])
	return nil
}

func computeKKLSSChebyshevPower(n int, powers map[int]*mkckks.Ciphertext, evaluator *mkckks.Evaluator, relin *mkrlwe.RelinearizationKeySet, ordinaryParams ckks.Parameters) (*mkckks.Ciphertext, error) {
	return computeKKLSSChebyshevPowerCounted(n, powers, evaluator, relin, ordinaryParams, nil)
}

// computeKKLSSChebyshevPowerCounted is the native public-KKLSS recurrence
// with optional operation accounting. The counter is incremented exactly at
// each public mkckks.MulRelinNew invocation; cache hits do not increment it.
// This keeps the polynomial-internal multiplications separate from the
// frozen network's 6,172 ciphertext products.
func computeKKLSSChebyshevPowerCounted(n int, powers map[int]*mkckks.Ciphertext, evaluator *mkckks.Evaluator, relin *mkrlwe.RelinearizationKeySet, ordinaryParams ckks.Parameters, mulRelinCalls *int) (*mkckks.Ciphertext, error) {
	if n < 1 {
		return nil, fmt.Errorf("Chebyshev power index must be positive, got %d", n)
	}
	if power := powers[n]; power != nil {
		return power, nil
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
	left, err := computeKKLSSChebyshevPowerCounted(a, powers, evaluator, relin, ordinaryParams, mulRelinCalls)
	if err != nil {
		return nil, err
	}
	right, err := computeKKLSSChebyshevPowerCounted(b, powers, evaluator, relin, ordinaryParams, mulRelinCalls)
	if err != nil {
		return nil, err
	}
	// The public KKLSS MulRelinNew performs its own one-prime CKKS rescale.
	// Therefore this call is the sole level-consuming multiplication for the
	// current recurrence node.
	power := evaluator.MulRelinNew(left, right, relin)
	if mulRelinCalls != nil {
		*mulRelinCalls++
	}
	evaluator.MultByConst(power, int64(2), power)
	if c == 0 {
		if err := kklssAddConstantAllSlots(ordinaryParams, power, complex(-1, 0)); err != nil {
			return nil, err
		}
	} else {
		correction, err := computeKKLSSChebyshevPowerCounted(c, powers, evaluator, relin, ordinaryParams, mulRelinCalls)
		if err != nil {
			return nil, err
		}
		power = evaluator.SubNew(power, correction)
	}
	powers[n] = power
	return power, nil
}

// evaluateKKLSSChebyshevDegree59 evaluates the same coefficient vector and
// Chebyshev convention as the frozen network. Power-basis nodes use only the
// public KKLSS MulRelinNew path; rational coefficient products are followed
// by the package's CKKS Rescale. This is a functional evaluator, not yet a
// stage-optimized implementation.
func evaluateKKLSSChebyshevDegree59(ordinaryParams ckks.Parameters, input *mkckks.Ciphertext, polynomial *ckks.Polynomial, evaluator *mkckks.Evaluator, relin *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, error) {
	result, _, err := evaluateKKLSSChebyshevDegree59Counted(ordinaryParams, input, polynomial, evaluator, relin)
	return result, err
}

func evaluateKKLSSChebyshevDegree59Counted(ordinaryParams ckks.Parameters, input *mkckks.Ciphertext, polynomial *ckks.Polynomial, evaluator *mkckks.Evaluator, relin *mkrlwe.RelinearizationKeySet) (*mkckks.Ciphertext, int, error) {
	if input == nil || polynomial == nil || polynomial.Degree() != resnetReferenceReLUDegree || polynomial.Basis != ckks.ChebyshevBasis {
		return nil, 0, fmt.Errorf("KKLSS evaluator requires the frozen degree-%d Chebyshev polynomial", resnetReferenceReLUDegree)
	}
	if polynomial.A != complex(-1, 0) || polynomial.B != complex(1, 0) {
		return nil, 0, fmt.Errorf("KKLSS evaluator requires the frozen interval [-1,1], got [%v,%v]", polynomial.A, polynomial.B)
	}
	if input.Level() < bits.Len(uint(polynomial.Degree()))+1 {
		return nil, 0, fmt.Errorf("KKLSS input level=%d is insufficient for the degree-%d balanced Chebyshev evaluator", input.Level(), polynomial.Degree())
	}
	powers := map[int]*mkckks.Ciphertext{1: input.CopyNew()}
	terms := make([]*mkckks.Ciphertext, 0, len(polynomial.Coeffs)-1)
	mulRelinCalls := 0
	for degree := 1; degree < len(polynomial.Coeffs); degree++ {
		coefficient := polynomial.Coeffs[degree]
		if math.Abs(real(coefficient)) <= ckks.IsNegligbleThreshold && math.Abs(imag(coefficient)) <= ckks.IsNegligbleThreshold {
			continue
		}
		power, err := computeKKLSSChebyshevPowerCounted(degree, powers, evaluator, relin, ordinaryParams, &mulRelinCalls)
		if err != nil {
			return nil, 0, err
		}
		term := power.CopyNew()
		evaluator.MultByConst(term, coefficient, term)
		if err := evaluator.Rescale(term, ordinaryParams.Scale(), term); err != nil {
			return nil, 0, fmt.Errorf("rescale Chebyshev coefficient %d: %w", degree, err)
		}
		terms = append(terms, term)
	}
	if len(terms) == 0 {
		return nil, 0, fmt.Errorf("degree-%d polynomial has no nonconstant coefficients", polynomial.Degree())
	}
	minimumLevel := terms[0].Level()
	for _, term := range terms[1:] {
		if term.Level() < minimumLevel {
			minimumLevel = term.Level()
		}
	}
	for _, term := range terms {
		if term.Level() > minimumLevel {
			evaluator.DropLevel(term, term.Level()-minimumLevel)
		}
	}
	result := terms[0]
	for _, term := range terms[1:] {
		result = evaluator.AddNew(result, term)
	}
	if constant := polynomial.Coeffs[0]; math.Abs(real(constant)) > ckks.IsNegligbleThreshold || math.Abs(imag(constant)) > ckks.IsNegligbleThreshold {
		if err := kklssAddConstantAllSlots(ordinaryParams, result, constant); err != nil {
			return nil, 0, err
		}
	}
	result.Scale = ordinaryParams.Scale()
	return result, mulRelinCalls, nil
}

func TestKKLSSP2N2MulRefreshDegree59RotateCollectiveDecrypt(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_P2_N2_REFRESH_POLY") != "1" {
		t.Skip("set MKF_RUN_KKLSS_P2_N2_REFRESH_POLY=1 for the real KKLSS MulRelin->refresh->degree59->Rotate functional gate")
	}
	context, err := newKKLSSP2N2PrimitiveContext()
	if err != nil {
		t.Fatal(err)
	}
	ordinaryParams, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	leftValue, rightValue := complex(0.25, 0), complex(0.5, 0)
	left := context.encryptor.EncryptMsgNew(kklssConstantMessage(context.params, leftValue), context.public.GetPublicKey("data-owner"))
	right := context.encryptor.EncryptMsgNew(kklssConstantMessage(context.params, rightValue), context.public.GetPublicKey("model-owner"))
	product := context.evaluator.MulRelinNew(left, right, context.relin)
	foldedCompact, err := foldKKLSSMultiCiphertext(ordinaryParams, context.partyIDs, context.partySecrets, product)
	if err != nil {
		t.Fatal(err)
	}
	foldedTuple, err := kklssMaterializeAggregateCompact(context.params, foldedCompact, context.partyIDs)
	if err != nil {
		t.Fatal(err)
	}
	foldedDecoded := context.decryptor.Decrypt(foldedTuple, context.secret)
	foldedMaximum := 0.0
	for _, got := range foldedDecoded.Value {
		delta := cmplxAbs(got - leftValue*rightValue)
		if delta > foldedMaximum {
			foldedMaximum = delta
		}
	}
	if foldedMaximum > 5e-3 {
		t.Fatalf("KKLSS P2/n=2 CKS fold bridge max error=%g exceeds functional guard", foldedMaximum)
	}
	t.Logf("KKLSS CKS fold bridge PASS: max_error=%g level=%d scale=%g", foldedMaximum, foldedTuple.Level(), foldedTuple.Scale)
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	refreshed, refreshEvidence, err := refreshKKLSSMultiCiphertext(ordinaryParams, context.params, context.partyIDs, context.partySecrets, product, crs)
	if err != nil {
		t.Fatal(err)
	}
	refreshedDecoded := context.decryptor.Decrypt(refreshed, context.secret)
	refreshedMaximum := 0.0
	for _, got := range refreshedDecoded.Value {
		delta := cmplxAbs(got - leftValue*rightValue)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			t.Fatal("KKLSS collective refresh produced a non-finite error")
		}
		if delta > refreshedMaximum {
			refreshedMaximum = delta
		}
	}
	if refreshedMaximum > 5e-3 {
		t.Fatalf("KKLSS P2/n=2 collective-refresh bridge max error=%g exceeds functional guard; evidence=%+v", refreshedMaximum, refreshEvidence)
	}
	t.Logf("KKLSS collective-refresh bridge PASS: max_error=%g level=%d scale=%g", refreshedMaximum, refreshed.Level(), refreshed.Scale)
	activationScale := 0.7
	polynomial := referenceResNetBlockReLUPolynomial(activationScale)
	activated, err := evaluateKKLSSChebyshevDegree59(ordinaryParams, refreshed, polynomial, context.evaluator, context.relin)
	if err != nil {
		t.Fatal(err)
	}
	activatedDecoded := context.decryptor.Decrypt(activated, context.secret)
	activationWant := evaluateChebyshevSeries(polynomial, leftValue*rightValue)
	activationMaximum := 0.0
	for _, got := range activatedDecoded.Value {
		delta := cmplxAbs(got - activationWant)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			t.Fatal("KKLSS degree-59 evaluator produced a non-finite error")
		}
		if delta > activationMaximum {
			activationMaximum = delta
		}
	}
	if activationMaximum > 5e-3 {
		t.Fatalf("KKLSS P2/n=2 degree-59 evaluator max error=%g exceeds functional guard at level=%d scale=%g", activationMaximum, activated.Level(), activated.Scale)
	}
	rotated := context.evaluator.RotateNew(activated, 1, context.rotation)
	decoded := context.decryptor.Decrypt(rotated, context.secret)
	want := activationWant
	maximum := 0.0
	for _, got := range decoded.Value {
		delta := cmplxAbs(got - want)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			t.Fatal("KKLSS refresh/polynomial chain produced a non-finite error")
		}
		if delta > maximum {
			maximum = delta
		}
	}
	if maximum > 5e-3 {
		t.Fatalf("KKLSS P2/n=2 refresh/polynomial chain max error=%g exceeds functional guard", maximum)
	}
	if refreshed.Level() != ordinaryParams.MaxLevel() || activated.Level() >= refreshed.Level() || len(rotated.IDSet().Value) != 2 {
		t.Fatalf("unexpected KKLSS state: refreshLevel=%d activationLevel=%d outputIDs=%v", refreshed.Level(), activated.Level(), rotated.IDSet().Value)
	}
	t.Logf("KKLSS P2/n=2 MulRelin->CKS/Refresh->degree59 KKLSS MulRelin chain->Rotate->collective Decrypt PASS: max_error=%g refresh=%+v activation_level=%d; no latency measured", maximum, refreshEvidence, activated.Level())
}
