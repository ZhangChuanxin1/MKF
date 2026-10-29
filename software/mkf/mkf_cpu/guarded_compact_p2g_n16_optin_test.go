package mkfcpu

// This file wires the independently frozen P2G-N16 candidate into the
// guarded-compact ResNet-20 test backend behind an explicit parameter ID. It
// does not change CKKSParameters("P2G") or the legacy P2/P2G defaults.

import (
	"fmt"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
)

const (
	guardedCompactP2GN16ParameterID     = "P2GN16"
	guardedCompactP2GN16SecurityGate    = "PASS_FIXED_ORDINARY_LWE_ONLY_KDM_PROTOCOL_CONDITIONAL"
	guardedCompactP2GLegacySecurityGate = "FAIL_BLOCKED_P2G_N32768_HE_V2_TABLE_GATE"
)

type guardedCompactParameterProfile struct {
	id           string
	guardPrime   uint64
	guarded      ckks.Parameters
	base         ckks.Parameters
	securityGate string
}

func guardedCompactParameterProfileFor(name string) (guardedCompactParameterProfile, error) {
	var (
		guarded      ckks.Parameters
		base         ckks.Parameters
		guardPrime   uint64
		securityGate string
		err          error
	)
	switch name {
	case guardedCompactP2GParameterID:
		guarded, err = CKKSParameters(guardedCompactP2GParameterID)
		if err != nil {
			return guardedCompactParameterProfile{}, err
		}
		base, err = CKKSParameters("P2")
		if err != nil {
			return guardedCompactParameterProfile{}, err
		}
		guardPrime = ckksGuardPrime
		securityGate = guardedCompactP2GLegacySecurityGate

	case guardedCompactP2GN16ParameterID:
		guardedLiteral := p2gN16SecurityCandidate
		guardedLiteral.Q = append([]uint64(nil), p2gN16SecurityCandidate.Q...)
		guardedLiteral.P = append([]uint64(nil), p2gN16SecurityCandidate.P...)
		guarded, err = ckks.NewParametersFromLiteral(guardedLiteral)
		if err != nil {
			return guardedCompactParameterProfile{}, fmt.Errorf("initialize frozen P2GN16 guarded literal: %w", err)
		}
		baseLiteral := guardedLiteral
		baseLiteral.Q = append([]uint64(nil), guardedLiteral.Q[:len(guardedLiteral.Q)-1]...)
		base, err = ckks.NewParametersFromLiteral(baseLiteral)
		if err != nil {
			return guardedCompactParameterProfile{}, fmt.Errorf("initialize frozen P2GN16 persistent literal: %w", err)
		}
		guardPrime = p2gN16GuardPrime
		securityGate = guardedCompactP2GN16SecurityGate

	default:
		return guardedCompactParameterProfile{}, fmt.Errorf("unknown guarded-compact parameter %q", name)
	}

	if guarded.MaxLevel() != base.MaxLevel()+1 || guarded.Q()[guarded.MaxLevel()] != guardPrime {
		return guardedCompactParameterProfile{}, fmt.Errorf("%s guard layout mismatch: guarded level/top=%d/0x%x base level=%d expected guard=0x%x", name, guarded.MaxLevel(), guarded.Q()[guarded.MaxLevel()], base.MaxLevel(), guardPrime)
	}
	if guarded.LogN() != base.LogN() || guarded.LogSlots() != base.LogSlots() || guarded.Scale() != base.Scale() || len(guarded.P()) != len(base.P()) {
		return guardedCompactParameterProfile{}, fmt.Errorf("%s guarded/persistent parameter-shape mismatch", name)
	}
	for level, modulus := range base.Q() {
		if guarded.Q()[level] != modulus {
			return guardedCompactParameterProfile{}, fmt.Errorf("%s persistent Q prefix mismatch at level %d", name, level)
		}
	}
	for index, modulus := range base.P() {
		if guarded.P()[index] != modulus {
			return guardedCompactParameterProfile{}, fmt.Errorf("%s P mismatch at index %d", name, index)
		}
	}
	return guardedCompactParameterProfile{
		id: name, guardPrime: guardPrime, guarded: guarded, base: base,
		securityGate: securityGate,
	}, nil
}

func guardedCompactSecurityGateFor(name string) string {
	profile, err := guardedCompactParameterProfileFor(name)
	if err != nil {
		return "FAIL_UNKNOWN_GUARDED_COMPACT_PARAMETER"
	}
	return profile.securityGate
}

func TestGuardedCompactP2GN16ParameterSelectionContract(t *testing.T) {
	profile, err := guardedCompactParameterProfileFor(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	if profile.id != guardedCompactP2GN16ParameterID || profile.guarded.LogN() != 16 || profile.guarded.LogSlots() != 14 || profile.guarded.MaxLevel() != 14 || profile.base.MaxLevel() != 13 || profile.guardPrime != p2gN16GuardPrime || profile.guarded.Alpha() != 2 || profile.guarded.Beta() != 8 {
		t.Fatalf("P2GN16 profile drifted: id=%s logN/logSlots=%d/%d levels=%d/%d guard=0x%x alpha/beta=%d/%d", profile.id, profile.guarded.LogN(), profile.guarded.LogSlots(), profile.guarded.MaxLevel(), profile.base.MaxLevel(), profile.guardPrime, profile.guarded.Alpha(), profile.guarded.Beta())
	}
	if profile.securityGate != guardedCompactP2GN16SecurityGate {
		t.Fatalf("P2GN16 security-gate label=%q", profile.securityGate)
	}
	if construction := guardedCompactResNet20ConstructionFor(profile.id); construction != guardedCompactP2GN16CPUConstructionID {
		t.Fatalf("P2GN16 construction identity=%q want=%q", construction, guardedCompactP2GN16CPUConstructionID)
	}
	wired, err := fullResNet20Parameters(guardedCompactP2GN16ParameterID)
	if err != nil {
		t.Fatal(err)
	}
	if wired.LogN() != profile.guarded.LogN() || wired.MaxLevel() != profile.guarded.MaxLevel() || wired.Q()[wired.MaxLevel()] != profile.guardPrime {
		t.Fatalf("full ResNet-20 P2GN16 selector did not return the frozen guarded profile")
	}
	legacy, err := guardedCompactParameterProfileFor(guardedCompactP2GParameterID)
	if err != nil {
		t.Fatal(err)
	}
	if legacy.guarded.LogN() != 15 || legacy.securityGate != guardedCompactP2GLegacySecurityGate {
		t.Fatalf("legacy P2G default changed: logN=%d gate=%q", legacy.guarded.LogN(), legacy.securityGate)
	}
	t.Logf("P2GN16_GUARDED_BACKEND_PARAMETER_PASS parameter=%s logN=%d logSlots=%d guarded_level=%d persistent_level=%d guard=0x%x rotation_key_count=%d security_gate=%s", profile.id, profile.guarded.LogN(), profile.guarded.LogSlots(), profile.guarded.MaxLevel(), profile.base.MaxLevel(), profile.guardPrime, len(fullResNet20RotationKeys()), profile.securityGate)
}
