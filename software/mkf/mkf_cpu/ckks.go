package mkfcpu

import (
	"fmt"
	"math"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type Validation struct {
	Passed   bool
	MaxError float64
	Expected float64
	Observed float64
}

type CKKSCase struct {
	params  ckks.Parameters
	parties int
	// secret retains the per-party source keys for non-timed functional
	// diagnostics (e.g., the party-specific Galois-key rotation fixture).
	// The multiplication benchmark never reads this field inside its timed
	// boundary.
	secret        []*rlwe.SecretKey
	keys          onlineKeys
	aggregate     *rlwe.SecretKey
	left          *multiCiphertext
	right         *multiCiphertext
	leftExpected  float64
	rightExpected float64
	expected      float64
	encoder       ckks.Encoder
	evaluator     ckks.Evaluator
	switcher      *switchWorkspace
	// lastOutput is populated after a successful run for fixture export and
	// correctness inspection. It is not read inside the timed boundary.
	lastOutput *outputCiphertext
}

func NewCKKSCase(parameter string, parties int) (*CKKSCase, error) {
	if parties < 1 {
		return nil, fmt.Errorf("parties must be positive")
	}
	params, err := CKKSParameters(parameter)
	if err != nil {
		return nil, err
	}
	kgen := ckks.NewKeyGenerator(params)
	secret := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	for i := range secret {
		secret[i], public[i] = kgen.GenKeyPair()
	}
	aggregate := aggregateSecret(params.Parameters, secret)
	keys := generateOnlineKeys(params.Parameters, kgen, secret, aggregate)
	encoder := ckks.NewEncoder(params)

	leftValues := make([]float64, parties)
	rightValues := make([]float64, parties)
	var leftSum, rightSum float64
	for i := 0; i < parties; i++ {
		leftValues[i] = float64((i%7)+1) / float64(16*parties)
		rightValues[i] = float64((i%5)+1) / float64(12*parties)
		leftSum += leftValues[i]
		rightSum += rightValues[i]
	}
	left := makeCKKSMultiCiphertext(params, encoder, public, leftValues)
	right := makeCKKSMultiCiphertext(params, encoder, public, rightValues)
	return &CKKSCase{
		params: params, parties: parties, secret: secret, keys: keys, aggregate: aggregate,
		left: left, right: right, leftExpected: leftSum, rightExpected: rightSum,
		expected: leftSum * rightSum, encoder: encoder,
		evaluator: ckks.NewEvaluator(params, rlwe.EvaluationKey{}),
		switcher:  newSwitchWorkspace(params.Parameters),
	}, nil
}

func makeCKKSMultiCiphertext(params ckks.Parameters, encoder ckks.Encoder, public []*rlwe.PublicKey, values []float64) *multiCiphertext {
	level := params.MaxLevel()
	out := &multiCiphertext{
		C0:         params.RingQ().NewPolyLvl(level),
		Components: make([]*ring.Poly, len(public)),
		Level:      level,
		Scale:      params.Scale(),
	}
	out.C0.IsNTT = true
	for i := range public {
		slots := make([]complex128, 1)
		slots[0] = complex(values[i], 0)
		plaintext := encoder.EncodeNTTAtLvlNew(level, slots, params.LogSlots())
		ciphertext := ckks.NewEncryptor(params, public[i]).EncryptNew(plaintext)
		params.RingQ().AddLvl(level, out.C0, ciphertext.Value[0], out.C0)
		out.Components[i] = ciphertext.Value[1].CopyNew()
		out.Components[i].IsNTT = true
	}
	return out
}

func (c *CKKSCase) Run() (Profile, Validation, error) {
	prof := Profile{}
	start := time.Now()

	left, err := fold(c.params.Parameters, c.left, c.keys.Fold, true, c.switcher, &prof)
	if err != nil {
		return prof, Validation{}, err
	}
	right, err := fold(c.params.Parameters, c.right, c.keys.Fold, true, c.switcher, &prof)
	if err != nil {
		return prof, Validation{}, err
	}

	level := c.params.MaxLevel()
	ringQ := c.params.RingQ()
	mu0, mu1, mu2 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftM0, leftM1 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftSum, rightSum := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	measure(&prof.Tensor, func() {
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

	relin0, relin1 := c.switcher.switchOne(c.params.Parameters, level, mu2, c.keys.Relin.Keys[0], true, &prof)
	measure(&prof.External, func() {
		ringQ.AddLvl(level, mu0, relin0, mu0)
		ringQ.AddLvl(level, mu1, relin1, mu1)
	})

	compact := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{mu0, mu1})
	compact.Scale = c.params.Scale() * c.params.Scale()
	measure(&prof.Modulus, func() {
		if rescaleErr := c.evaluator.Rescale(compact, c.params.Scale(), compact); rescaleErr != nil {
			err = rescaleErr
		}
	})
	if err != nil {
		return prof, Validation{}, err
	}

	out := materialize(c.params.Parameters, compact.Value[0], compact.Value[1], c.parties, compact.Level(), compact.Scale, &prof)
	prof.finish(start)
	c.lastOutput = out

	verificationCT := ckks.NewCiphertextAtLevelFromPoly(out.Level, [2]*ring.Poly{out.C0, out.Gamma})
	verificationCT.Scale = out.Scale
	plaintext := ckks.NewDecryptor(c.params, c.aggregate).DecryptNew(verificationCT)
	decoded := c.encoder.Decode(plaintext, c.params.LogSlots())
	observed := real(decoded[0])
	maxError := math.Abs(observed - c.expected)
	validation := Validation{
		Passed: maxError < 1e-6, MaxError: maxError,
		Expected: c.expected, Observed: observed,
	}
	return prof, validation, nil
}
