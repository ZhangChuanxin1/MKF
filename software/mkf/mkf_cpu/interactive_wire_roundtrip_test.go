package mkfcpu

import (
	"os"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/drlwe"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const projectedStemIngressCiphertexts = 30

// wireRoundTripInteractiveAggregateFold is the protocol-equivalent of
// interactiveAggregateFold with one additional requirement: every party's CKS
// share is marshaled and unmarshaled before it reaches the aggregator. The
// returned sizes describe Lattigo's current, uncompressed in-process binary
// encoding. They exclude transport framing and are not network-latency data.
func wireRoundTripInteractiveAggregateFold(t *testing.T, params ckks.Parameters, in *multiCiphertext, sourceShares []*rlwe.SecretKey) (*ckks.Ciphertext, [][]int) {
	t.Helper()

	if len(in.Components) != len(sourceShares) {
		t.Fatalf("component/share count mismatch: %d != %d", len(in.Components), len(sourceShares))
	}

	level := in.Level
	ringQ := params.RingQ()
	compact := ckks.NewCiphertext(params, 1, level, in.Scale)
	compact.Value[0].CopyValues(in.C0)
	compact.Value[0].IsNTT = in.C0.IsNTT
	compact.Value[1].Zero()
	compact.Value[1].IsNTT = true

	zeroShare := rlwe.NewSecretKey(params.Parameters)
	shareBytes := make([][]int, len(in.Components))
	for sourceIndex, component := range in.Components {
		sourceCT := ckks.NewCiphertext(params, 1, level, in.Scale)
		sourceCT.Value[0].Zero()
		sourceCT.Value[1].CopyValues(component)
		sourceCT.Value[0].IsNTT = component.IsNTT
		sourceCT.Value[1].IsNTT = component.IsNTT

		coordinator := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
		shareBytes[sourceIndex] = make([]int, len(sourceShares))
		var aggregateShare *drlwe.CKSShare
		for partyIndex := range sourceShares {
			party := dckks.NewCKSProtocol(params, collectiveRefreshSigmaSmudging)
			share := party.AllocateShare(level)
			inputShare := zeroShare
			if partyIndex == sourceIndex {
				inputShare = sourceShares[partyIndex]
			}
			party.GenShare(inputShare, sourceShares[partyIndex], sourceCT.Ciphertext, share)

			wire, err := share.MarshalBinary()
			if err != nil {
				t.Fatalf("marshal CKS share source=%d party=%d: %v", sourceIndex, partyIndex, err)
			}
			decoded := new(drlwe.CKSShare)
			if err = decoded.UnmarshalBinary(wire); err != nil {
				t.Fatalf("unmarshal CKS share source=%d party=%d: %v", sourceIndex, partyIndex, err)
			}
			if decoded.Value.Level() != level {
				t.Fatalf("CKS share level changed after wire round-trip: source=%d party=%d got=%d want=%d", sourceIndex, partyIndex, decoded.Value.Level(), level)
			}
			if decoded.Value.IsNTT != share.Value.IsNTT {
				t.Fatalf("CKS share NTT flag changed after wire round-trip: source=%d party=%d", sourceIndex, partyIndex)
			}
			shareBytes[sourceIndex][partyIndex] = len(wire)
			t.Logf("CKS_SHARE_BYTES source=%d party=%d bytes=%d", sourceIndex, partyIndex, len(wire))

			if partyIndex == 0 {
				aggregateShare = decoded
			} else {
				coordinator.AggregateShares(aggregateShare, decoded, aggregateShare)
			}
		}

		converted := ckks.NewCiphertext(params, 1, level, in.Scale)
		coordinator.KeySwitchCKKS(aggregateShare, sourceCT, converted)
		ringQ.AddLvl(level, compact.Value[0], converted.Value[0], compact.Value[0])
		ringQ.AddLvl(level, compact.Value[1], converted.Value[1], compact.Value[1])
	}

	return compact, shareBytes
}

// wireRoundTripCollectiveRefresh likewise forces every RefreshShare through
// MarshalBinary/UnmarshalBinary before aggregation. The byte counts are the
// current uncompressed in-process encoding, not a measured network transfer.
func wireRoundTripCollectiveRefresh(t *testing.T, params ckks.Parameters, shares []*rlwe.SecretKey, input *ckks.Ciphertext, logBound int, crs utils.PRNG) (*ckks.Ciphertext, []int) {
	t.Helper()

	levelIn, levelOut := input.Level(), params.MaxLevel()
	coordinator := dckks.NewRefreshProtocol(params, logBound, collectiveRefreshSigmaSmudging)
	refreshCRP := coordinator.SampleCRP(levelOut, crs)
	shareBytes := make([]int, len(shares))

	var aggregateShare *dckks.RefreshShare
	for partyIndex, sk := range shares {
		party := dckks.NewRefreshProtocol(params, logBound, collectiveRefreshSigmaSmudging)
		share := party.AllocateShare(levelIn, levelOut)
		party.GenShares(sk, logBound, params.LogSlots(), input, refreshCRP, share)

		wire, err := share.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal RefreshShare party=%d: %v", partyIndex, err)
		}
		decoded := new(dckks.RefreshShare)
		if err = decoded.UnmarshalBinary(wire); err != nil {
			t.Fatalf("unmarshal RefreshShare party=%d: %v", partyIndex, err)
		}
		shareBytes[partyIndex] = len(wire)
		t.Logf("REFRESH_SHARE_BYTES party=%d bytes=%d", partyIndex, len(wire))

		if partyIndex == 0 {
			aggregateShare = decoded
		} else {
			coordinator.Aggregate(decoded, aggregateShare, aggregateShare)
		}
	}

	output := ckks.NewCiphertext(params, 1, levelOut, params.Scale())
	coordinator.Finalize(input, params.LogSlots(), refreshCRP, aggregateShare, output)
	// Lattigo v2.3.0 does not preserve these representation flags when it
	// copies the refresh CRS into the output. The values themselves are NTT.
	output.Value[0].IsNTT, output.Value[1].IsNTT = true, true
	return output, shareBytes
}

func sumWireBytes(sizes []int) int64 {
	var total int64
	for _, size := range sizes {
		total += int64(size)
	}
	return total
}

func testInteractiveWireRoundTrip(t *testing.T, parameter string) {
	t.Helper()
	t.Log("WIRE_ENCODING_SCOPE=current uncompressed Lattigo v2.3.0 in-process MarshalBinary encoding; excludes transport framing; bytes are not network latency")

	c, err := newInteractiveIngressCase(parameter, 2)
	if err != nil {
		t.Fatal(err)
	}
	compact, ingressSizes := wireRoundTripInteractiveAggregateFold(t, c.params, c.input, c.shares)
	pre := c.validateInteractiveIngressWithTestOnlyAggregate(compact)
	if !pre.Passed {
		t.Fatalf("wire-round-tripped interactive ingress failed: parameter=%s maxError=%g", parameter, pre.MaxError)
	}

	var ingressTestTotal int64
	var ingressPerCipher int64 = -1
	for sourceIndex, sizes := range ingressSizes {
		sourceTotal := sumWireBytes(sizes)
		ingressTestTotal += sourceTotal
		t.Logf("CKS_INGRESS_BYTES_PER_CIPHERTEXT source=%d parties=%d total_bytes=%d", sourceIndex, len(sizes), sourceTotal)
		if ingressPerCipher == -1 {
			ingressPerCipher = sourceTotal
		} else if ingressPerCipher != sourceTotal {
			t.Fatalf("cannot project fixed-size stem ingress: source totals differ: first=%d source%d=%d", ingressPerCipher, sourceIndex, sourceTotal)
		}
	}
	projectedStemIngress := ingressPerCipher * projectedStemIngressCiphertexts
	t.Logf("CKS_INGRESS_TEST_TOTAL_BYTES source_ciphertexts=%d total_bytes=%d", len(ingressSizes), ingressTestTotal)
	t.Logf("PROJECTED_STEM_INGRESS_BYTES source_ciphertexts=%d parties=%d total_bytes=%d scope=current_uncompressed_in_process_wire_encoding_not_network_latency", projectedStemIngressCiphertexts, len(c.shares), projectedStemIngress)

	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, compact.Scale, c.parties, c.params.Q())
	if !ok {
		t.Fatalf("%s lacks levels for two-party collective refresh", parameter)
	}
	ckks.NewEvaluator(c.params, rlwe.EvaluationKey{}).DropLevel(compact, compact.Level()-minLevel)
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	refreshed, refreshSizes := wireRoundTripCollectiveRefresh(t, c.params, c.shares, compact, logBound, crs)
	post := c.validateInteractiveIngressWithTestOnlyAggregate(refreshed)
	if !post.Passed {
		t.Fatalf("wire-round-tripped collective refresh failed: parameter=%s maxError=%g", parameter, post.MaxError)
	}

	refreshTotal := sumWireBytes(refreshSizes)
	t.Logf("REFRESH_TOTAL_BYTES parties=%d total_bytes=%d", len(refreshSizes), refreshTotal)
	t.Logf("WIRE_ROUNDTRIP_PASS parameter=%s ingress_max_error=%g refresh_max_error=%g refresh_level=%d->%d log_bound=%d", parameter, pre.MaxError, post.MaxError, minLevel, refreshed.Level(), logBound)
}

func TestInteractiveWireRoundTripP2TwoParty(t *testing.T) {
	testInteractiveWireRoundTrip(t, "P2")
}

func TestInteractiveWireRoundTripP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_IR_WIRE") != "1" {
		t.Skip("set MKF_RUN_P3_IR_WIRE=1 to run the P3 wire-format functional test")
	}
	testInteractiveWireRoundTrip(t, "P3IRCandidate")
}
