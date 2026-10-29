package mkfcpu

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

// The KKLSS source tree used by this adapter is the public
// SNUCP/MKHE-KKLSS implementation at the commit below. The mkckks package is
// imported through the local module replacement in ../go.mod; it is not an
// operation-count model or an MKF substitute.
const (
	kklssSourceCommit = "13fed55fd28270fb4345dd532da24b323ebc5ce3"
	kklssConstruction = "SNUCP/MKHE-KKLSS@" + kklssSourceCommit + ":mkckks.MulRelinNew/RotateNew"

	// Both CPU columns must use this exact frozen input and scale profile.
	// The fixture digest is checked by loadFullResNet20OfficialFixture. The
	// scale-profile digest is the current training-only margin-0.70 artifact.
	kklssResNet20Fixture = "cifar10-test-00000:record-sha256=35ce668430e638d9f487715a81e2051fc1b8ca1263461ce72179c29bfdbc1ee8:activation-scales-sha256=ea9e6bad9179fbf1bff4209b741ca7a199def0c1b56ccd8b5d80e1d4b4bfae86"

	kklssResNet20Boundary = "online-only KKLSS evaluation: encrypted Stem, nine Residual Blocks, and checkpoint-faithful GAP+FC; includes native public-KKLSS rotations, network ct-ct MulRelinNew, degree-59 internal MulRelinNew, rescaling/modulus management, the coefficient/NTT carrier conversions used by the shared network scheduler, and 22 two-party refresh rounds with the source-to-aggregate CKS bridge; excludes offline key generation, fixture/model loading, plaintext packing, phase-local owner encryption/staging, ciphertext serialization accounting, decryption, and correctness verification"
)

var errKKLSSResNet20BackendUninitialized = errors.New("KKLSS-CPU ResNet-20 backend state is uninitialized")

// kklssResNet20CPUBackend exposes only the independently prepared KKLSS
// performance trial. The oracle-instrumented functional executor is never
// reachable through this factory.
type kklssResNet20CPUBackend struct {
	state *kklssCPUBackendState
}

func (backend *kklssResNet20CPUBackend) Implementation() string { return "KKLSS-CPU" }
func (backend *kklssResNet20CPUBackend) ConstructionID() string { return kklssConstruction }
func (backend *kklssResNet20CPUBackend) ParameterID() string    { return "P2" }
func (backend *kklssResNet20CPUBackend) Parties() int           { return 2 }
func (backend *kklssResNet20CPUBackend) FixtureID() string      { return kklssResNet20Fixture }
func (backend *kklssResNet20CPUBackend) MeasurementBoundary() string {
	return kklssResNet20Boundary
}
func (backend *kklssResNet20CPUBackend) NewTrial() (resNet20CPUPreparedTrial, error) {
	if backend == nil || backend.state == nil {
		return nil, errKKLSSResNet20BackendUninitialized
	}
	return backend.newPreparedTrial()
}

func newKKLSSResNet20CPUBackend() (resNet20CPUBackend, error) {
	return &kklssResNet20CPUBackend{state: new(kklssCPUBackendState)}, nil
}

func init() {
	registerResNet20CPUBackend("KKLSS-CPU", newKKLSSResNet20CPUBackend)
}

// kklssP2Literal duplicates neither parameters nor a theoretical proxy: it
// obtains the exact P2 literal already frozen for the matched MKF experiment
// and converts it to the public KKLSS parameter wrapper.
func kklssP2Parameters() (mkckks.Parameters, error) {
	return kklssParameters("P2")
}

// kklssParameters converts an already frozen ordinary CKKS literal into the
// public KKLSS wrapper. P2GN16 uses the persistent Q prefix: the extra guard
// prime is an MKF ingress mechanism and is not exposed to native KKLSS owner
// ciphertexts or online network evaluation.
func kklssParameters(parameter string) (mkckks.Parameters, error) {
	var (
		params ckks.Parameters
		err    error
	)
	if parameter == guardedCompactP2GN16ParameterID {
		profile, profileErr := guardedCompactParameterProfileFor(parameter)
		if profileErr != nil {
			return mkckks.Parameters{}, profileErr
		}
		params = profile.base
	} else {
		params, err = CKKSParameters(parameter)
	}
	if err != nil {
		return mkckks.Parameters{}, err
	}
	return mkckks.NewParameters(params), nil
}

type kklssP2N2PrimitiveContext struct {
	params       mkckks.Parameters
	partyIDs     []string
	partySecrets []*mkrlwe.SecretKey
	secret       *mkrlwe.SecretKeySet
	public       *mkrlwe.PublicKeySet
	relin        *mkrlwe.RelinearizationKeySet
	rotation     *mkrlwe.RotationKeySet
	encryptor    *mkckks.Encryptor
	decryptor    *mkckks.Decryptor
	evaluator    *mkckks.Evaluator
}

// newKKLSSP2N2PrimitiveContext performs key generation only. Callers must
// keep it outside all timed regions. Only rotation by one is generated for
// the primitive contract; the full network will require its audited frozen
// rotation-key inventory before it can become measurement-ready.
func newKKLSSP2N2PrimitiveContext() (*kklssP2N2PrimitiveContext, error) {
	return newKKLSSP2N2Context([]int{1})
}

// newKKLSSP2N2Context creates the exact public-KKLSS P2/two-party key
// material for a caller-supplied rotation inventory. Negative scheduler
// rotations are normalized before CRS/key generation because the public
// KKLSS Rotate path performs the same normalization before key lookup.
// Callers must keep this constructor outside every timed region.
func newKKLSSP2N2Context(rotations []int) (*kklssP2N2PrimitiveContext, error) {
	return newKKLSSN2Context("P2", rotations)
}

func newKKLSSN2Context(parameter string, rotations []int) (*kklssP2N2PrimitiveContext, error) {
	params, err := kklssParameters(parameter)
	if err != nil {
		return nil, err
	}
	normalizedRotations := make([]int, 0, len(rotations))
	seenRotations := make(map[int]struct{}, len(rotations))
	for _, rotation := range rotations {
		rotation %= params.N() / 2
		if rotation < 0 {
			rotation += params.N() / 2
		}
		if rotation == 0 {
			continue
		}
		if _, exists := seenRotations[rotation]; exists {
			continue
		}
		seenRotations[rotation] = struct{}{}
		normalizedRotations = append(normalizedRotations, rotation)
		if _, exists := params.CRS[rotation]; !exists {
			params.AddCRS(rotation)
		}
	}
	context := &kklssP2N2PrimitiveContext{
		params: params, partyIDs: []string{"data-owner", "model-owner"}, secret: mkrlwe.NewSecretKeySet(), public: mkrlwe.NewPublicKeyKeySet(),
		relin: mkrlwe.NewRelinearizationKeyKeySet(params.Parameters), rotation: mkrlwe.NewRotationKeySet(),
		encryptor: mkckks.NewEncryptor(params), decryptor: mkckks.NewDecryptor(params), evaluator: mkckks.NewEvaluator(params),
	}
	keygen := mkckks.NewKeyGenerator(params)
	for _, id := range context.partyIDs {
		secret, public := keygen.GenKeyPair(id)
		// The public KKLSS implementation requires an auxiliary r_i with the
		// same party identifier when producing its three-part relin key.
		auxiliary := keygen.GenSecretKey(id)
		context.secret.AddSecretKey(secret)
		context.partySecrets = append(context.partySecrets, secret)
		context.public.AddPublicKey(public)
		context.relin.AddRelinearizationKey(keygen.GenRelinearizationKey(secret, auxiliary))
		for _, rotation := range normalizedRotations {
			context.rotation.AddRotationKey(keygen.GenRotationKey(rotation, secret))
		}
	}
	return context, nil
}

func kklssConstantMessage(params mkckks.Parameters, value complex128) *mkckks.Message {
	message := mkckks.NewMessage(params)
	for index := range message.Value {
		message.Value[index] = value
	}
	return message
}

// verifyKKLSSP2N2CrossOwnerPrimitive exercises the actual KKLSS
// multiplication/relinearization and rotation path at the exact P2/n=2
// contract. It is functional evidence only and returns no latency.
func verifyKKLSSP2N2CrossOwnerPrimitive() (float64, error) {
	context, err := newKKLSSP2N2PrimitiveContext()
	if err != nil {
		return 0, err
	}
	leftValue := complex(0.125, -0.03125)
	rightValue := complex(-0.25, 0.0625)
	left := context.encryptor.EncryptMsgNew(kklssConstantMessage(context.params, leftValue), context.public.GetPublicKey("data-owner"))
	right := context.encryptor.EncryptMsgNew(kklssConstantMessage(context.params, rightValue), context.public.GetPublicKey("model-owner"))
	product := context.evaluator.MulRelinNew(left, right, context.relin)
	rotated := context.evaluator.RotateNew(product, 1, context.rotation)
	if rotated.Level() != context.params.MaxLevel()-1 {
		return 0, fmt.Errorf("KKLSS P2 product level=%d, want %d after one rescale", rotated.Level(), context.params.MaxLevel()-1)
	}
	if ids := rotated.IDSet(); !ids.Has("data-owner") || !ids.Has("model-owner") || len(ids.Value) != 2 {
		return 0, fmt.Errorf("KKLSS cross-owner output ID set is not {data-owner,model-owner}: %+v", ids.Value)
	}
	decoded := context.decryptor.Decrypt(rotated, context.secret)
	want := leftValue * rightValue
	maximum := 0.0
	for _, got := range decoded.Value {
		delta := cmplxAbs(got - want)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			return 0, fmt.Errorf("KKLSS P2 primitive produced a non-finite decoding error")
		}
		if delta > maximum {
			maximum = delta
		}
	}
	return maximum, nil
}

func cmplxAbs(value complex128) float64 {
	return math.Hypot(real(value), imag(value))
}

func TestKKLSSResNet20CPUBackendRegistrationIsAuditedAndFailsClosed(t *testing.T) {
	factory, exists := resNet20CPUBackendFactories["KKLSS-CPU"]
	if !exists {
		t.Fatal("real KKLSS-CPU factory is not registered")
	}
	backend, err := factory()
	if err != nil {
		t.Fatal(err)
	}
	if backend.ParameterID() != "P2" || backend.Parties() != 2 || backend.FixtureID() != kklssResNet20Fixture {
		t.Fatalf("unmatched KKLSS backend identity: parameter=%s parties=%d fixture=%s", backend.ParameterID(), backend.Parties(), backend.FixtureID())
	}
	if !strings.Contains(backend.ConstructionID(), kklssSourceCommit) || !strings.Contains(backend.MeasurementBoundary(), "22 two-party refresh rounds") {
		t.Fatalf("incomplete KKLSS construction/boundary identity: construction=%q boundary=%q", backend.ConstructionID(), backend.MeasurementBoundary())
	}
	reporter, ok := backend.(resNet20CPUCommunicationArtifactReporter)
	if !ok || reporter == nil {
		t.Fatal("KKLSS backend omitted the mandatory communication-artifact reporter")
	}
	prepared, err := (&kklssResNet20CPUBackend{}).NewTrial()
	if prepared != nil || !errors.Is(err, errKKLSSResNet20BackendUninitialized) {
		t.Fatalf("uninitialized backend did not fail closed: prepared=%T err=%v", prepared, err)
	}
}

func TestKKLSSP2N2CrossOwnerMulRelinPrimitiveContract(t *testing.T) {
	if os.Getenv("MKF_RUN_KKLSS_P2_N2_PRIMITIVE") != "1" {
		t.Skip("set MKF_RUN_KKLSS_P2_N2_PRIMITIVE=1 for the real P2/n=2 KKLSS functional primitive gate")
	}
	maximum, err := verifyKKLSSP2N2CrossOwnerPrimitive()
	if err != nil {
		t.Fatal(err)
	}
	// This is a loose functional guard, not a claimed precision result.
	if maximum > 1e-3 {
		t.Fatalf("KKLSS P2/n=2 cross-owner primitive max error=%g exceeds functional guard", maximum)
	}
	t.Logf("KKLSS P2/n=2 cross-owner MulRelinNew+RotateNew functional PASS: max_error=%g; no latency measured", maximum)
}

// Compile-time anchors make accidental replacement with the ordinary
// single-key CKKS API visible during review.
var (
	_ *mkckks.Ciphertext = nil
	_ ckks.Parameters    = ckks.Parameters{}
)
