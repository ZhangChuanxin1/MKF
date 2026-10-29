package mkfcpu

import (
	"testing"

	"github.com/ldsec/lattigo/v2/bfv"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

func TestBFVCorrectnessP1(t *testing.T) {
	c, err := NewBFVCase("P1", 2)
	if err != nil {
		t.Fatal(err)
	}
	profile, validation, err := c.Run()
	if err != nil {
		t.Fatal(err)
	}
	if !validation.Passed {
		t.Fatalf("validation failed: expected=%g observed=%g error=%g", validation.Expected, validation.Observed, validation.MaxError)
	}
	if profile.Decompositions != 5 || profile.ExternalProductPolys != 10 {
		t.Fatalf("operation counts: got decomp=%d ext-polys=%d, want 5 and 10", profile.Decompositions, profile.ExternalProductPolys)
	}
}

func TestBFVStagesP1(t *testing.T) {
	c, err := NewBFVCase("P1", 2)
	if err != nil {
		t.Fatal(err)
	}
	profile := Profile{}
	left, err := fold(c.params.Parameters, c.left, c.keys.Fold, false, c.switcher, &profile)
	if err != nil {
		t.Fatal(err)
	}
	right, err := fold(c.params.Parameters, c.right, c.keys.Fold, false, c.switcher, &profile)
	if err != nil {
		t.Fatal(err)
	}
	decode := func(value []*ring.Poly) uint64 {
		ct := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: value}}
		return c.encoder.DecodeUintNew(bfv.NewDecryptor(c.params, c.aggregate).DecryptNew(ct))[0]
	}
	if got := decode([]*ring.Poly{left[0], left[1]}); got != 3 {
		t.Fatalf("left fold: got %d want 3", got)
	}
	if got := decode([]*ring.Poly{right[0], right[1]}); got != 5 {
		t.Fatalf("right fold: got %d want 5", got)
	}

	leftCT := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: []*ring.Poly{left[0], left[1]}}}
	rightCT := &bfv.Ciphertext{Ciphertext: &rlwe.Ciphertext{Value: []*ring.Poly{right[0], right[1]}}}
	evaluator := bfv.NewEvaluator(c.params, rlwe.EvaluationKey{Rlk: c.keys.Relin})
	standardProduct := evaluator.MulNew(leftCT, rightCT)
	if got := decode(standardProduct.Value); got != 15 {
		t.Fatalf("standard tensor: got %d want 15", got)
	}
	standardRelin := evaluator.RelinearizeNew(standardProduct)
	if got := decode(standardRelin.Value); got != 15 {
		t.Fatalf("standard relin: got %d want 15", got)
	}

	c.tensor.modUpLeft(left)
	c.tensor.rescaleRight(right)
	c.tensor.tensorKaratsuba()
	ours := c.tensor.quantize()
	if got := decode([]*ring.Poly{ours[0], ours[1], ours[2]}); got != 15 {
		t.Fatalf("custom tensor: got %d want 15", got)
	}
}
