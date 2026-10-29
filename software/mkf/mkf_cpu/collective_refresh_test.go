package mkfcpu

import (
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

func TestCollectiveRefreshP2TwoParty(t *testing.T) {
	c, err := newCollectiveRefreshCase("P2", 2)
	if err != nil {
		t.Fatal(err)
	}
	out, err := c.refresh()
	if err != nil {
		t.Fatal(err)
	}
	validation := c.validateWithTestOnlyAggregate(out)
	if !validation.Passed {
		t.Fatalf("collective refresh failed: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
	}
	t.Logf("collective refresh P2: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
}

func TestFoldedAggregateBridgeP2TwoParty(t *testing.T) {
	b, err := newFoldedRefreshBridge("P2", 2)
	if err != nil {
		t.Fatal(err)
	}
	_, validation, err := b.foldThenRefresh()
	if err != nil {
		t.Fatal(err)
	}
	if !validation.Passed {
		t.Fatalf("folded aggregate bridge failed: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
	}
	t.Logf("folded aggregate bridge P2: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
}

// P3IRCandidate is intentionally opt-in because a LogN=16, 34-Q-prime
// functional run is resource-intensive. It produces no timing result and is
// still not a validated P3 security parameter set. Run it explicitly with
// MKF_RUN_P3_IR=1.
func TestCollectiveRefreshP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR") != "1" {
		t.Skip("set MKF_RUN_P3_IR=1 to run the LogN=16 P3 interactive-refresh functional check")
	}
	c, err := newCollectiveRefreshCase("P3IRCandidate", 2)
	if err != nil {
		t.Fatal(err)
	}
	out, err := c.refresh()
	if err != nil {
		t.Fatal(err)
	}
	validation := c.validateWithTestOnlyAggregate(out)
	if !validation.Passed {
		t.Fatalf("P3 candidate collective refresh failed: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
	}
	t.Logf("collective refresh P3 candidate: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
}

func TestFoldedAggregateBridgeP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR_BRIDGE") != "1" {
		t.Skip("set MKF_RUN_P3_IR_BRIDGE=1 to run the resource-intensive P3 folded bridge check")
	}
	b, err := newFoldedRefreshBridge("P3IRCandidate", 2)
	if err != nil {
		t.Fatal(err)
	}
	_, validation, err := b.foldThenRefresh()
	if err != nil {
		t.Fatal(err)
	}
	if !validation.Passed {
		t.Fatalf("P3 candidate folded bridge failed: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
	}
	t.Logf("folded aggregate bridge P3 candidate: level=%d->%d logBound=%d maxError=%g", validation.LevelIn, validation.LevelOut, validation.LogBound, validation.MaxError)
}

func TestInteractiveIngressFoldThenRefreshP2TwoParty(t *testing.T) {
	c, err := newInteractiveIngressCase("P2", 2)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := interactiveAggregateFold(c.params, c.input, c.shares)
	if err != nil {
		t.Fatal(err)
	}
	pre := c.validateInteractiveIngressWithTestOnlyAggregate(compact)
	if !pre.Passed {
		t.Fatalf("interactive ingress fold failed: maxError=%g", pre.MaxError)
	}

	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, c.params.Scale(), c.parties, c.params.Q())
	if !ok {
		t.Fatal("P2 does not satisfy the dckks collective-refresh level gate")
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
	post := c.validateInteractiveIngressWithTestOnlyAggregate(refreshed)
	post.LevelIn, post.LevelOut, post.LogBound = compact.Level(), refreshed.Level(), logBound
	if !post.Passed {
		t.Fatalf("interactive ingress fold then refresh failed: level=%d->%d logBound=%d maxError=%g", post.LevelIn, post.LevelOut, post.LogBound, post.MaxError)
	}
	t.Logf("interactive ingress P2: preMaxError=%g; refresh level=%d->%d logBound=%d postMaxError=%g", pre.MaxError, post.LevelIn, post.LevelOut, post.LogBound, post.MaxError)
}

func TestInteractiveIngressFoldThenRefreshP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR_INGRESS") != "1" {
		t.Skip("set MKF_RUN_P3_IR_INGRESS=1 to run the P3 interactive-ingress and refresh check")
	}
	c, err := newInteractiveIngressCase("P3IRCandidate", 2)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := interactiveAggregateFold(c.params, c.input, c.shares)
	if err != nil {
		t.Fatal(err)
	}
	pre := c.validateInteractiveIngressWithTestOnlyAggregate(compact)
	if !pre.Passed {
		t.Fatalf("P3 interactive ingress fold failed: maxError=%g", pre.MaxError)
	}

	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, c.params.Scale(), c.parties, c.params.Q())
	if !ok {
		t.Fatal("P3 candidate does not satisfy the dckks collective-refresh level gate")
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
	post := c.validateInteractiveIngressWithTestOnlyAggregate(refreshed)
	post.LevelIn, post.LevelOut, post.LogBound = compact.Level(), refreshed.Level(), logBound
	if !post.Passed {
		t.Fatalf("P3 interactive ingress then refresh failed: level=%d->%d logBound=%d maxError=%g", post.LevelIn, post.LevelOut, post.LogBound, post.MaxError)
	}
	t.Logf("interactive ingress P3 candidate: preMaxError=%g; refresh level=%d->%d logBound=%d postMaxError=%g", pre.MaxError, post.LevelIn, post.LevelOut, post.LogBound, post.MaxError)
}
