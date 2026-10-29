package mkfcpu

import (
	"fmt"
	"math/cmplx"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// interactiveIngressCase creates independently encrypted components but does
// not construct an aggregate secret or any aggregate-domain switching key.
// It is the appropriate starting point for validating an interactive ingress
// conversion: the protocol uses only individual secret shares, while the
// aggregate secret is reconstructed solely by the test verifier.
type interactiveIngressCase struct {
	params   ckks.Parameters
	parties  int
	shares   []*rlwe.SecretKey
	input    *multiCiphertext
	expected float64
	encoder  ckks.Encoder
}

func newInteractiveIngressCase(parameter string, parties int) (*interactiveIngressCase, error) {
	if parties < 2 {
		return nil, fmt.Errorf("interactive ingress requires at least two parties")
	}
	params, err := CKKSParameters(parameter)
	if err != nil {
		return nil, err
	}

	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, parties)
	public := make([]*rlwe.PublicKey, parties)
	for i := range shares {
		shares[i], public[i] = kgen.GenKeyPair()
	}

	values := make([]float64, parties)
	var expected float64
	for i := range values {
		values[i] = float64((i%7)+1) / float64(16*parties)
		expected += values[i]
	}
	encoder := ckks.NewEncoder(params)
	input := makeCKKSMultiCiphertext(params, encoder, public, values)

	return &interactiveIngressCase{
		params: params, parties: parties, shares: shares, input: input,
		expected: expected, encoder: encoder,
	}, nil
}

// makeOwnerCKKSMultiCiphertext embeds one ordinary degree-1 CKKS ciphertext
// into the n+1-component multi-key interface. Only ownerComponent is nonzero;
// this models independently encrypted input or model material before the
// interactive aggregate-domain ingress conversion.
func makeOwnerCKKSMultiCiphertext(params ckks.Parameters, encoder ckks.Encoder, public []*rlwe.PublicKey, ownerComponent int, values []complex128) (*multiCiphertext, error) {
	if ownerComponent < 0 || ownerComponent >= len(public) {
		return nil, fmt.Errorf("owner component %d outside [0,%d)", ownerComponent, len(public))
	}
	level := params.MaxLevel()
	plaintext := encoder.EncodeNTTAtLvlNew(level, values, params.LogSlots())
	ciphertext := ckks.NewEncryptor(params, public[ownerComponent]).EncryptNew(plaintext)
	out := &multiCiphertext{
		C0:         ciphertext.Value[0].CopyNew(),
		Components: make([]*ring.Poly, len(public)),
		Level:      level,
		Scale:      ciphertext.Scale,
	}
	for i := range out.Components {
		out.Components[i] = params.RingQ().NewPolyLvl(level)
		out.Components[i].Zero()
		out.Components[i].IsNTT = true
	}
	out.Components[ownerComponent].CopyValues(ciphertext.Value[1])
	out.Components[ownerComponent].IsNTT = true
	out.C0.IsNTT = true
	return out, nil
}

// interactiveAggregateFold transforms each source component (0, c_i) from
// its individual key s_i to the collective key s_E=sum_j s_j using dckks CKS,
// then adds all transformed components to c_0. For source i, the CKS input
// shares are (0,...,s_i,...,0), while the output shares are
// (s_1,...,s_n). Thus no party needs to reconstruct s_E.
//
// This deliberately changes the ingress protocol relative to MKF's current
// public fold-key construction: all parties participate once per source
// component. The function is a functional prototype; its smudging/noise and
// transcript-security parameters require a separate proof before publication.
func interactiveAggregateFold(params ckks.Parameters, in *multiCiphertext, sourceShares []*rlwe.SecretKey) (*ckks.Ciphertext, error) {
	if len(in.Components) != len(sourceShares) {
		return nil, fmt.Errorf("component/share count mismatch: %d != %d", len(in.Components), len(sourceShares))
	}
	if len(sourceShares) < 2 {
		return nil, fmt.Errorf("interactive aggregate fold requires at least two shares")
	}

	level := in.Level
	ringQ := params.RingQ()
	compact := ckks.NewCiphertext(params, 1, level, in.Scale)
	compact.Value[0].CopyValues(in.C0)
	compact.Value[0].IsNTT = in.C0.IsNTT
	compact.Value[1].Zero()
	compact.Value[1].IsNTT = true

	zeroShare := rlwe.NewSecretKey(params.Parameters)
	for sourceIndex, component := range in.Components {
		sourceCT := ckks.NewCiphertext(params, 1, level, in.Scale)
		sourceCT.Value[0].Zero()
		sourceCT.Value[1].CopyValues(component)
		sourceCT.Value[0].IsNTT = component.IsNTT
		sourceCT.Value[1].IsNTT = component.IsNTT

		coordinator := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
		var aggregateShare *drlwe.CKSShare
		for partyIndex := range sourceShares {
			party := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
			share := party.AllocateShare(level)
			inputShare := zeroShare
			if partyIndex == sourceIndex {
				inputShare = sourceShares[partyIndex]
			}
			party.GenShare(inputShare, sourceShares[partyIndex], sourceCT.Ciphertext, share)
			if partyIndex == 0 {
				aggregateShare = share
			} else {
				coordinator.AggregateShares(aggregateShare, share, aggregateShare)
			}
		}

		converted := ckks.NewCiphertext(params, 1, level, in.Scale)
		coordinator.KeySwitchCKKS(aggregateShare, sourceCT, converted)
		ringQ.AddLvl(level, compact.Value[0], converted.Value[0], compact.Value[0])
		ringQ.AddLvl(level, compact.Value[1], converted.Value[1], compact.Value[1])
	}

	return compact, nil
}

// interactiveOwnerFold is the sparse-owner specialization used at CNN
// ingress: one independently encrypted object has exactly one nonzero secret
// component. Publishing the owner index is part of the experiment's ownership
// map, so zero components need no CKS transcript, noise, or communication.
func interactiveOwnerFold(params ckks.Parameters, in *multiCiphertext, sourceShares []*rlwe.SecretKey, ownerComponent int) (*ckks.Ciphertext, error) {
	if len(in.Components) != len(sourceShares) {
		return nil, fmt.Errorf("component/share count mismatch: %d != %d", len(in.Components), len(sourceShares))
	}
	if ownerComponent < 0 || ownerComponent >= len(sourceShares) {
		return nil, fmt.Errorf("owner component %d outside [0,%d)", ownerComponent, len(sourceShares))
	}
	if len(sourceShares) < 2 {
		return nil, fmt.Errorf("interactive owner fold requires at least two shares")
	}

	level := in.Level
	sourceCT := ckks.NewCiphertext(params, 1, level, in.Scale)
	sourceCT.Value[0].Zero()
	sourceCT.Value[1].CopyValues(in.Components[ownerComponent])
	sourceCT.Value[0].IsNTT = in.Components[ownerComponent].IsNTT
	sourceCT.Value[1].IsNTT = in.Components[ownerComponent].IsNTT

	zeroShare := rlwe.NewSecretKey(params.Parameters)
	coordinator := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
	var aggregateShare *drlwe.CKSShare
	for partyIndex := range sourceShares {
		party := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
		share := party.AllocateShare(level)
		inputShare := zeroShare
		if partyIndex == ownerComponent {
			inputShare = sourceShares[partyIndex]
		}
		party.GenShare(inputShare, sourceShares[partyIndex], sourceCT.Ciphertext, share)
		if partyIndex == 0 {
			aggregateShare = share
		} else {
			coordinator.AggregateShares(aggregateShare, share, aggregateShare)
		}
	}

	converted := ckks.NewCiphertext(params, 1, level, in.Scale)
	coordinator.KeySwitchCKKS(aggregateShare, sourceCT, converted)
	params.RingQ().AddLvl(level, in.C0, converted.Value[0], converted.Value[0])
	converted.Value[0].IsNTT = in.C0.IsNTT
	converted.Value[1].IsNTT = in.Components[ownerComponent].IsNTT
	return converted, nil
}

// validateInteractiveIngressWithTestOnlyAggregate is deliberately a test
// oracle. Unlike interactiveAggregateFold, it reconstructs the aggregate key
// and therefore must not be incorporated into the protocol implementation.
func (c *interactiveIngressCase) validateInteractiveIngressWithTestOnlyAggregate(compact *ckks.Ciphertext) RefreshValidation {
	aggregate := aggregateSecret(c.params.Parameters, c.shares)
	plaintext := ckks.NewDecryptor(c.params, aggregate).DecryptNew(compact)
	decoded := c.encoder.Decode(plaintext, c.params.LogSlots())
	maxError := cmplx.Abs(decoded[0] - complex(c.expected, 0))
	return RefreshValidation{
		Passed: maxError < 1e-5, MaxError: maxError,
		LevelIn: compact.Level(), LevelOut: compact.Level(),
	}
}
