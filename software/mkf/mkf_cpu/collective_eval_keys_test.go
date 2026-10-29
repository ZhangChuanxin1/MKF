package mkfcpu

import (
	"math/cmplx"
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

func testCollectiveEvalAfterRefresh(t *testing.T, parameter string) {
	t.Helper()
	c, err := newCollectiveRefreshCase(parameter, 2)
	if err != nil {
		t.Fatal(err)
	}
	refreshed, err := c.refresh()
	if err != nil {
		t.Fatal(err)
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	key, err := generateCollectiveEvaluationKey(c.params, c.shares, []int{1}, crs)
	if err != nil {
		t.Fatal(err)
	}

	evaluator := ckks.NewEvaluator(c.params, key)
	product := evaluator.MulRelinNew(refreshed, refreshed)
	if err := evaluator.Rescale(product, c.params.Scale(), product); err != nil {
		t.Fatal(err)
	}
	rotated := evaluator.RotateNew(product, 1)

	aggregate := aggregateSecret(c.params.Parameters, c.shares)
	plaintext := ckks.NewDecryptor(c.params, aggregate).DecryptNew(rotated)
	decoded := c.encoder.Decode(plaintext, c.params.LogSlots())
	want := make([]complex128, c.params.Slots())
	for i, value := range c.expected {
		want[i] = value * value
	}
	want = utils.RotateComplex128Slice(want, 1)
	var maxError float64
	for i := range want {
		if err := cmplx.Abs(decoded[i] - want[i]); err > maxError {
			maxError = err
		}
	}
	if maxError >= 1e-4 {
		t.Fatalf("collective evaluation after refresh failed: parameter=%s maxError=%g", parameter, maxError)
	}
	t.Logf("collective evaluation after refresh %s: multiply/relinearize/rescale/rotate PASS; outputLevel=%d maxError=%g", parameter, rotated.Level(), maxError)
}

func TestCollectiveEvaluationAfterRefreshP2TwoParty(t *testing.T) {
	testCollectiveEvalAfterRefresh(t, "P2")
}

func TestCollectiveEvaluationAfterRefreshP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR_EVAL") != "1" {
		t.Skip("set MKF_RUN_P3_IR_EVAL=1 to run P3 collective RKG/RTG and post-refresh evaluation")
	}
	testCollectiveEvalAfterRefresh(t, "P3IRCandidate")
}

// testInteractiveIngressRefreshThenEval composes the protocol-facing
// interactive CKS ingress conversion with collective Refresh and collective
// aggregate-domain evaluation. The aggregate secret is reconstructed only
// after all protocol operations as a test decryption oracle.
func testInteractiveIngressRefreshThenEval(t *testing.T, parameter string) {
	t.Helper()
	c, err := newInteractiveIngressCase(parameter, 2)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := interactiveAggregateFold(c.params, c.input, c.shares)
	if err != nil {
		t.Fatal(err)
	}
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, compact.Scale, c.parties, c.params.Q())
	if !ok {
		t.Fatalf("%s lacks levels for two-party collective refresh", parameter)
	}
	ckks.NewEvaluator(c.params, rlwe.EvaluationKey{}).DropLevel(compact, compact.Level()-minLevel)
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	refreshed, err := runCollectiveRefresh(c.params, c.shares, compact, logBound, crs)
	if err != nil {
		t.Fatal(err)
	}
	key, err := generateCollectiveEvaluationKey(c.params, c.shares, []int{1}, crs)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(c.params, key)
	product := evaluator.MulRelinNew(refreshed, refreshed)
	if err := evaluator.Rescale(product, c.params.Scale(), product); err != nil {
		t.Fatal(err)
	}
	rotated := evaluator.RotateNew(product, 1)

	aggregate := aggregateSecret(c.params.Parameters, c.shares)
	plaintext := ckks.NewDecryptor(c.params, aggregate).DecryptNew(rotated)
	decoded := c.encoder.Decode(plaintext, c.params.LogSlots())
	want := make([]complex128, c.params.Slots())
	want[0] = complex(c.expected*c.expected, 0)
	want = utils.RotateComplex128Slice(want, 1)
	var maxError float64
	for i := range want {
		if err := cmplx.Abs(decoded[i] - want[i]); err > maxError {
			maxError = err
		}
	}
	if maxError >= 1e-4 {
		t.Fatalf("interactive ingress->refresh->collective evaluation failed: parameter=%s maxError=%g", parameter, maxError)
	}
	t.Logf("interactive ingress->refresh->collective evaluation %s: PASS; refreshLevel=%d outputLevel=%d maxError=%g", parameter, minLevel, rotated.Level(), maxError)
}

func TestInteractiveIngressRefreshThenCollectiveEvaluationP2TwoParty(t *testing.T) {
	testInteractiveIngressRefreshThenEval(t, "P2")
}

func TestInteractiveIngressRefreshThenCollectiveEvaluationP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR_E2E") != "1" {
		t.Skip("set MKF_RUN_P3_IR_E2E=1 to run P3 interactive-ingress-to-evaluation functional test")
	}
	testInteractiveIngressRefreshThenEval(t, "P3IRCandidate")
}
