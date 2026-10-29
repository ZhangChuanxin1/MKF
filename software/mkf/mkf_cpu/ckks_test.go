package mkfcpu

import "testing"

func TestCKKSCorrectnessP1(t *testing.T) {
	c, err := NewCKKSCase("P1", 2)
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
