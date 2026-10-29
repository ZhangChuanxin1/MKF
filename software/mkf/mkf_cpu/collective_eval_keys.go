package mkfcpu

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// generateCollectiveEvaluationKey produces a collective relinearization key
// and the requested collective column-rotation keys under the additive secret
// shares. The protocol never forms the aggregate secret. It is intended for
// the compact aggregate-domain ciphertext that is produced by the interactive
// ingress conversion and refreshed by collective Refresh.
//
// This helper records only a functional construction. Its public transcript,
// communication size, and security assumptions must be measured and proved
// separately before it is used to support a manuscript claim.
func generateCollectiveEvaluationKey(params ckks.Parameters, shares []*rlwe.SecretKey, rotations []int, crs utils.PRNG) (rlwe.EvaluationKey, error) {
	if len(shares) < 2 {
		return rlwe.EvaluationKey{}, fmt.Errorf("collective evaluation-key generation requires at least two shares")
	}

	type rkgParty struct {
		protocol *dckks.RKGProtocol
		sk       *rlwe.SecretKey
		eph      *rlwe.SecretKey
		round1   *drlwe.RKGShare
		round2   *drlwe.RKGShare
	}

	coordinator := dckks.NewRKGProtocol(params)
	rkgCRP := coordinator.SampleCRP(crs)
	parties := make([]rkgParty, len(shares))
	var aggregateRound1, aggregateRound2 *drlwe.RKGShare
	for i, sk := range shares {
		party := dckks.NewRKGProtocol(params)
		eph, round1, round2 := party.AllocateShares()
		parties[i] = rkgParty{protocol: party, sk: sk, eph: eph, round1: round1, round2: round2}
		party.GenShareRoundOne(sk, rkgCRP, eph, round1)
		if i == 0 {
			aggregateRound1 = round1
		} else {
			coordinator.AggregateShares(aggregateRound1, round1, aggregateRound1)
		}
	}
	for i := range parties {
		party := &parties[i]
		party.protocol.GenShareRoundTwo(party.eph, party.sk, aggregateRound1, party.round2)
		if i == 0 {
			aggregateRound2 = party.round2
		} else {
			coordinator.AggregateShares(aggregateRound2, party.round2, aggregateRound2)
		}
	}
	rlin := ckks.NewRelinearizationKey(params)
	coordinator.GenRelinearizationKey(aggregateRound1, aggregateRound2, rlin)

	galEls := make([]uint64, 0, len(rotations))
	seen := make(map[uint64]struct{}, len(rotations))
	for _, rotation := range rotations {
		if rotation == 0 {
			continue
		}
		galEl := params.GaloisElementForColumnRotationBy(rotation)
		if _, exists := seen[galEl]; !exists {
			seen[galEl] = struct{}{}
			galEls = append(galEls, galEl)
		}
	}
	rotationKeys := ckks.NewRotationKeySet(params, galEls)
	for _, galEl := range galEls {
		rotCoordinator := dckks.NewRotKGProtocol(params)
		rotCRP := rotCoordinator.SampleCRP(crs)
		var aggregateRotationShare *drlwe.RTGShare
		for i, sk := range shares {
			party := dckks.NewRotKGProtocol(params)
			share := party.AllocateShares()
			party.GenShare(sk, galEl, rotCRP, share)
			if i == 0 {
				aggregateRotationShare = share
			} else {
				rotCoordinator.Aggregate(aggregateRotationShare, share, aggregateRotationShare)
			}
		}
		rotCoordinator.GenRotationKey(aggregateRotationShare, rotCRP, rotationKeys.Keys[galEl])
	}

	evaluationKey := rlwe.EvaluationKey{Rlk: rlin, Rtks: rotationKeys}
	if err := maybeExportP2GN16ProductionKeysSameRun(params, rotations, evaluationKey); err != nil {
		return rlwe.EvaluationKey{}, fmt.Errorf("export same-run P2GN16 production evaluation keys: %w", err)
	}
	return evaluationKey, nil
}
