package mkfcpu

import (
	"fmt"
	"math/big"
	"time"

	"github.com/ldsec/lattigo/v2/bfv"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type BFVCase struct {
	params    bfv.Parameters
	parties   int
	keys      onlineKeys
	aggregate *rlwe.SecretKey
	left      *multiCiphertext
	right     *multiCiphertext
	expected  uint64
	encoder   bfv.Encoder
	switcher  *switchWorkspace
	tensor    *bfvTensorWorkspace
	// lastOutput is populated after a successful run for fixture export and
	// correctness inspection. It is not read inside the timed boundary.
	lastOutput *outputCiphertext
}

func NewBFVCase(parameter string, parties int) (*BFVCase, error) {
	if parties < 1 {
		return nil, fmt.Errorf("parties must be positive")
	}
	params, qMul, err := BFVParameters(parameter)
	if err != nil {
		return nil, err
	}
	kgen := bfv.NewKeyGenerator(params)
	secret := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	for i := range secret {
		secret[i], public[i] = kgen.GenKeyPair()
	}
	aggregate := aggregateSecret(params.Parameters, secret)
	keys := generateOnlineKeys(params.Parameters, kgen, secret, aggregate)
	encoder := bfv.NewEncoder(params)

	leftValues := make([]uint64, parties)
	rightValues := make([]uint64, parties)
	var leftSum, rightSum uint64
	for i := 0; i < parties; i++ {
		leftValues[i] = uint64((i % 7) + 1)
		rightValues[i] = uint64((i % 5) + 2)
		leftSum = (leftSum + leftValues[i]) % params.T()
		rightSum = (rightSum + rightValues[i]) % params.T()
	}
	left := makeBFVMultiCiphertext(params, encoder, public, leftValues)
	right := makeBFVMultiCiphertext(params, encoder, public, rightValues)
	tensor, err := newBFVTensorWorkspace(params, qMul)
	if err != nil {
		return nil, err
	}
	return &BFVCase{
		params: params, parties: parties, keys: keys, aggregate: aggregate,
		left: left, right: right, expected: (leftSum * rightSum) % params.T(),
		encoder: encoder, switcher: newSwitchWorkspace(params.Parameters), tensor: tensor,
	}, nil
}

func makeBFVMultiCiphertext(params bfv.Parameters, encoder bfv.Encoder, public []*rlwe.PublicKey, values []uint64) *multiCiphertext {
	level := params.MaxLevel()
	out := &multiCiphertext{
		C0:         params.RingQ().NewPolyLvl(level),
		Components: make([]*ring.Poly, len(public)),
		Level:      level,
	}
	for i := range public {
		plaintext := bfv.NewPlaintext(params)
		encoder.EncodeUint([]uint64{values[i]}, plaintext)
		ciphertext := bfv.NewEncryptor(params, public[i]).EncryptNew(plaintext)
		params.RingQ().AddLvl(level, out.C0, ciphertext.Value[0], out.C0)
		out.Components[i] = ciphertext.Value[1].CopyNew()
		out.Components[i].IsNTT = false
	}
	return out
}

type bfvTensorWorkspace struct {
	params    bfv.Parameters
	ringQ     *ring.Ring
	ringQMul  *ring.Ring
	ringR     *ring.Ring
	basis     *ring.FastBasisExtender
	qMulBig   *big.Int
	mFormQMul *ring.Poly
	poolQ     *ring.Poly
	poolQMul  *ring.Poly
	poolR     *ring.Poly
	inR       [2][2]*ring.Poly
	productR  [3]*ring.Poly
	mformR    [3]*ring.Poly
	sumR      [2]*ring.Poly
	quantized [3]*ring.Poly
}

func newBFVTensorWorkspace(params bfv.Parameters, qMul []uint64) (*bfvTensorWorkspace, error) {
	ringQMul, err := ring.NewRing(params.N(), qMul)
	if err != nil {
		return nil, fmt.Errorf("create exact KKLSS QMul ring: %w", err)
	}
	ringR, err := ring.NewRing(params.N(), append(params.Q(), qMul...))
	if err != nil {
		return nil, fmt.Errorf("create exact KKLSS R=Q*QMul ring: %w", err)
	}
	w := &bfvTensorWorkspace{
		params: params, ringQ: params.RingQ(), ringQMul: ringQMul, ringR: ringR,
		basis:   ring.NewFastBasisExtender(params.RingQ(), ringQMul),
		qMulBig: new(big.Int).Set(ringQMul.ModulusBigint),
		poolQ:   params.RingQ().NewPoly(), poolQMul: ringQMul.NewPoly(), poolR: ringR.NewPoly(),
	}
	w.mFormQMul = w.ringQ.NewPoly()
	w.ringQ.AddScalarBigint(w.mFormQMul, w.qMulBig, w.mFormQMul)
	w.ringQ.MForm(w.mFormQMul, w.mFormQMul)
	for side := range w.inR {
		for j := range w.inR[side] {
			w.inR[side][j] = w.ringR.NewPoly()
		}
	}
	for i := 0; i < 3; i++ {
		w.productR[i] = w.ringR.NewPoly()
		w.mformR[i] = w.ringR.NewPoly()
		w.quantized[i] = w.ringQ.NewPoly()
	}
	for i := 0; i < 2; i++ {
		w.sumR[i] = w.ringR.NewPoly()
	}
	return w, nil
}

func (w *bfvTensorWorkspace) copyBasesToR(polyQ, polyQMul, polyR *ring.Poly) {
	levelQ := w.params.MaxLevel()
	levelQMul := len(w.ringQMul.Modulus) - 1
	for i := 0; i <= levelQ; i++ {
		copy(polyR.Coeffs[i], polyQ.Coeffs[i])
	}
	for i := 0; i <= levelQMul; i++ {
		copy(polyR.Coeffs[levelQ+1+i], polyQMul.Coeffs[i])
	}
}

// modUpLeft embeds Q into R=Q*QMul. rescaleRight represents
// round(x*QMul/Q) in R. Their product can therefore be divided by QMul
// after multiplication to implement the BFV t/Q scale-and-round. This is
// the exact lifted-basis strategy used by the KKLSS P1/P2 parameter sets.
func (w *bfvTensorWorkspace) modUpLeft(in [2]*ring.Poly) {
	levelQ := w.params.MaxLevel()
	levelQMul := len(w.ringQMul.Modulus) - 1
	for j := 0; j < 2; j++ {
		w.basis.ModUpQtoP(levelQ, levelQMul, in[j], w.poolQMul)
		w.copyBasesToR(in[j], w.poolQMul, w.inR[0][j])
		w.ringR.NTTLazy(w.inR[0][j], w.inR[0][j])
	}
}

func (w *bfvTensorWorkspace) rescaleRight(in [2]*ring.Poly) {
	levelQ := w.params.MaxLevel()
	levelQMul := len(w.ringQMul.Modulus) - 1
	for j := 0; j < 2; j++ {
		w.ringQ.MulCoeffsMontgomery(in[j], w.mFormQMul, w.poolQ)
		w.poolQMul.Zero()
		w.basis.ModDownQPtoP(levelQ, levelQMul, w.poolQ, w.poolQMul, w.poolQMul)
		w.basis.ModUpPtoQ(levelQMul, levelQ, w.poolQMul, w.poolQ)
		w.copyBasesToR(w.poolQ, w.poolQMul, w.inR[1][j])
		w.ringR.NTTLazy(w.inR[1][j], w.inR[1][j])
	}
}

// tensorKaratsuba computes (AA', AB'+BA', BB') with three polynomial
// products in each RNS basis instead of four.
func (w *bfvTensorWorkspace) tensorKaratsuba() {
	levelR := len(w.ringR.Modulus) - 1
	left, right := w.inR[0], w.inR[1]
	w.ringR.MFormLvl(levelR, left[0], w.mformR[0])
	w.ringR.MulCoeffsMontgomeryLvl(levelR, w.mformR[0], right[0], w.productR[0])
	w.ringR.MFormLvl(levelR, left[1], w.mformR[1])
	w.ringR.MulCoeffsMontgomeryLvl(levelR, w.mformR[1], right[1], w.productR[2])
	w.ringR.AddLvl(levelR, left[0], left[1], w.sumR[0])
	w.ringR.AddLvl(levelR, right[0], right[1], w.sumR[1])
	w.ringR.MFormLvl(levelR, w.sumR[0], w.mformR[2])
	w.ringR.MulCoeffsMontgomeryLvl(levelR, w.mformR[2], w.sumR[1], w.productR[1])
	w.ringR.SubLvl(levelR, w.productR[1], w.productR[0], w.productR[1])
	w.ringR.SubLvl(levelR, w.productR[1], w.productR[2], w.productR[1])
}

func (w *bfvTensorWorkspace) quantize() [3]*ring.Poly {
	levelQ := w.params.MaxLevel()
	levelQMul := len(w.ringQMul.Modulus) - 1
	for i := 0; i < 3; i++ {
		w.ringR.MulScalar(w.productR[i], w.params.T(), w.poolR)
		w.ringR.InvNTTLazy(w.poolR, w.poolR)
		for j := 0; j <= levelQ; j++ {
			copy(w.poolQ.Coeffs[j], w.poolR.Coeffs[j])
		}
		for j := 0; j <= levelQMul; j++ {
			copy(w.poolQMul.Coeffs[j], w.poolR.Coeffs[levelQ+1+j])
		}
		w.basis.ModDownQPtoQ(levelQ, levelQMul, w.poolQ, w.poolQMul, w.quantized[i])
		w.quantized[i].IsNTT = false
	}
	return w.quantized
}

func (c *BFVCase) Run() (Profile, Validation, error) {
	prof := Profile{}
	start := time.Now()
	left, err := fold(c.params.Parameters, c.left, c.keys.Fold, false, c.switcher, &prof)
	if err != nil {
		return prof, Validation{}, err
	}
	right, err := fold(c.params.Parameters, c.right, c.keys.Fold, false, c.switcher, &prof)
	if err != nil {
		return prof, Validation{}, err
	}

	measure(&prof.Modulus, func() {
		c.tensor.modUpLeft(left)
		c.tensor.rescaleRight(right)
	})
	measure(&prof.Tensor, c.tensor.tensorKaratsuba)
	var product [3]*ring.Poly
	measure(&prof.Modulus, func() { product = c.tensor.quantize() })

	level := c.params.MaxLevel()
	relin0, relin1 := c.switcher.switchOne(c.params.Parameters, level, product[2], c.keys.Relin.Keys[0], false, &prof)
	measure(&prof.External, func() {
		c.params.RingQ().AddLvl(level, product[0], relin0, product[0])
		c.params.RingQ().AddLvl(level, product[1], relin1, product[1])
	})
	out := materialize(c.params.Parameters, product[0], product[1], c.parties, level, 0, &prof)
	prof.finish(start)
	c.lastOutput = out

	verificationCT := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: []*ring.Poly{out.C0, out.Gamma}}}
	plaintext := bfv.NewDecryptor(c.params, c.aggregate).DecryptNew(verificationCT)
	decoded := c.encoder.DecodeUintNew(plaintext)
	observed := decoded[0]
	validation := Validation{
		Passed:   observed == c.expected,
		MaxError: float64(absUintMod(observed, c.expected, c.params.T())),
		Expected: float64(c.expected), Observed: float64(observed),
	}
	return prof, validation, nil
}

func absUintMod(a, b, modulus uint64) uint64 {
	if a >= b {
		if a-b <= modulus-(a-b) {
			return a - b
		}
		return modulus - (a - b)
	}
	return absUintMod(b, a, modulus)
}
