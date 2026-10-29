package mkfcpu

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"math/big"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const distributedPublicFoldKeyConstructionID = "MKF-P2G-public-fold-keygen-ckg+pke-per-party-v1"

// distributedPublicFoldKeyTranscript records byte-exact public evidence from
// the functional protocol. CRPBytes is the encoded common random polynomial;
// it is reproducible from a CRS seed and is therefore not counted again in
// TransmittedBytes. Each CKG share and each party fold key is an outgoing
// public message. CollectivePublicKeyBytes records the derived public output.
//
// This is not a security proof. In particular, encrypting P*g_j*s_i under the
// collective key is secret-dependent evaluation material and requires the
// aggregate-key circular/KDM assumption stated by the manuscript.
type distributedPublicFoldKeyTranscript struct {
	ConstructionID           string
	Parties                  int
	Digits                   int
	CRPBytes                 int
	CKGRounds                int
	CKGShareMessages         int
	CKGShareBytes            []int
	CollectivePublicKeyBytes int
	PublishedFoldKeyMessages int
	FoldKeyBytes             []int
	PublishedFoldKeyBytes    int
	TotalPublicMaterialBytes int
	TransmittedBytes         int
	SHA256                   string
}

type distributedPublicFoldKeyMaterial struct {
	CollectivePublicKey *rlwe.PublicKey
	FoldKeys            []*rlwe.SwitchingKey
	Transcript          distributedPublicFoldKeyTranscript
}

func hashTaggedBytes(digest hash.Hash, tag string, data []byte) {
	digest.Write([]byte(fmt.Sprintf("%s:%d:", tag, len(data))))
	digest.Write(data)
}

func marshalPolyQP(poly rlwe.PolyQP) ([]byte, error) {
	data := make([]byte, poly.GetDataLen(true))
	if _, err := poly.WriteTo(data); err != nil {
		return nil, err
	}
	return data, nil
}

// generateDistributedPublicFoldKeys runs dRLWE CKG for pk_E and then lets
// each party independently encrypt every RNS-decomposition digit target
// P*g_j*s_i under pk_E. The coordinator sees only the public CKG shares,
// pk_E, and the resulting public fold keys; it never receives or reconstructs
// s_E. A test may reconstruct s_E later, outside this function, solely as a
// correctness oracle.
func generateDistributedPublicFoldKeys(params ckks.Parameters, shares []*rlwe.SecretKey, crs utils.PRNG) (*distributedPublicFoldKeyMaterial, error) {
	if len(shares) < 2 {
		return nil, fmt.Errorf("distributed public FoldKeyGen requires at least two parties")
	}
	if params.PCount() == 0 {
		return nil, fmt.Errorf("distributed public FoldKeyGen requires a non-empty special-modulus basis")
	}

	digest := sha256.New()
	coordinator := dckks.NewCKGProtocol(params)
	ckgCRP := coordinator.SampleCRP(crs)
	crpBytes, err := marshalPolyQP(rlwe.PolyQP(ckgCRP))
	if err != nil {
		return nil, fmt.Errorf("marshal CKG CRP: %w", err)
	}
	hashTaggedBytes(digest, "ckg-crp", crpBytes)

	transcript := distributedPublicFoldKeyTranscript{
		ConstructionID:           distributedPublicFoldKeyConstructionID,
		Parties:                  len(shares),
		Digits:                   params.Beta(),
		CRPBytes:                 len(crpBytes),
		CKGRounds:                1,
		CKGShareMessages:         len(shares),
		CKGShareBytes:            make([]int, len(shares)),
		PublishedFoldKeyMessages: len(shares),
		FoldKeyBytes:             make([]int, len(shares)),
	}

	aggregated := coordinator.AllocateShares()
	for party, secretShare := range shares {
		participant := dckks.NewCKGProtocol(params)
		share := participant.AllocateShares()
		participant.GenShare(secretShare, ckgCRP, share)
		encoded, marshalErr := share.MarshalBinary()
		if marshalErr != nil {
			return nil, fmt.Errorf("marshal CKG share %d: %w", party, marshalErr)
		}
		transcript.CKGShareBytes[party] = len(encoded)
		transcript.TransmittedBytes += len(encoded)
		hashTaggedBytes(digest, fmt.Sprintf("ckg-share-%d", party), encoded)
		if party == 0 {
			aggregated = share
		} else {
			coordinator.AggregateShares(aggregated, share, aggregated)
		}
	}

	collectivePublicKey := rlwe.NewPublicKey(params.Parameters)
	coordinator.GenPublicKey(aggregated, ckgCRP, collectivePublicKey)
	encodedPublicKey, err := collectivePublicKey.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("marshal collective public key: %w", err)
	}
	transcript.CollectivePublicKeyBytes = len(encodedPublicKey)
	hashTaggedBytes(digest, "collective-public-key", encodedPublicKey)

	foldKeys := make([]*rlwe.SwitchingKey, len(shares))
	for party, secretShare := range shares {
		foldKey, keyErr := generatePartyPublicFoldKey(params.Parameters, secretShare, collectivePublicKey)
		if keyErr != nil {
			return nil, fmt.Errorf("generate party %d fold key: %w", party, keyErr)
		}
		foldKeys[party] = foldKey
		encoded, marshalErr := foldKey.MarshalBinary()
		if marshalErr != nil {
			return nil, fmt.Errorf("marshal party %d fold key: %w", party, marshalErr)
		}
		transcript.FoldKeyBytes[party] = len(encoded)
		transcript.PublishedFoldKeyBytes += len(encoded)
		transcript.TransmittedBytes += len(encoded)
		hashTaggedBytes(digest, fmt.Sprintf("fold-key-%d", party), encoded)
	}
	transcript.TotalPublicMaterialBytes = transcript.CollectivePublicKeyBytes + transcript.PublishedFoldKeyBytes
	transcript.SHA256 = hex.EncodeToString(digest.Sum(nil))
	if err := stageP2GN16ProductionFoldKeysSameRun(params, foldKeys); err != nil {
		return nil, fmt.Errorf("stage same-run P2GN16 production fold keys: %w", err)
	}

	return &distributedPublicFoldKeyMaterial{
		CollectivePublicKey: collectivePublicKey,
		FoldKeys:            foldKeys,
		Transcript:          transcript,
	}, nil
}

// generatePartyPublicFoldKey is the local, non-interactive party step. Its
// output has the exact [beta][2] PolyQP layout consumed by Lattigo v2.3.0's
// KeyswitchHoistedNoModDown. For decomposition digit j it samples a small u
// and errors e0,e1 and publishes
//
//	d_j = u*b_E + e0 + P*g_j*s_i,    k_j = u*a_E + e1.
//
// g_j is Lattigo's RNS digit selector: the P*s_i target is present only in
// the alpha Q limbs belonging to digit j and is zero in every P limb.
func generatePartyPublicFoldKey(params rlwe.Parameters, secretShare *rlwe.SecretKey, collectivePublicKey *rlwe.PublicKey) (*rlwe.SwitchingKey, error) {
	if secretShare == nil || collectivePublicKey == nil {
		return nil, fmt.Errorf("nil secret share or collective public key")
	}
	levelQ, levelP := params.QCount()-1, params.PCount()-1
	if levelP < 0 {
		return nil, fmt.Errorf("empty P basis")
	}

	prng, err := utils.NewPRNG()
	if err != nil {
		return nil, fmt.Errorf("create local FoldKeyGen PRNG: %w", err)
	}
	ringQ, ringQP := params.RingQ(), params.RingQP()
	ternary := ring.NewTernarySampler(prng, ringQ, 0.5, false)
	gaussian := ring.NewGaussianSampler(prng, ringQ, params.Sigma(), int(6*params.Sigma()))

	// Lattigo switching keys are stored in NTT+Montgomery form. The public-key
	// products are converted to Montgomery form below before adding the target.
	pTimesSecretMontgomery := ringQ.NewPolyLvl(levelQ)
	ringQ.MulScalarBigintLvl(levelQ, secretShare.Value.Q, specialModulusBigint(params, levelP), pTimesSecretMontgomery)
	pTimesSecretMontgomery.IsNTT = true

	foldKey := rlwe.NewSwitchingKey(params, levelQ, levelP)
	alpha := levelP + 1
	for digit := range foldKey.Value {
		u := ringQP.NewPolyLvl(levelQ, levelP)
		ternary.ReadLvl(levelQ, u.Q)
		ringQP.ExtendBasisSmallNormAndCenter(u.Q, levelP, nil, u.P)
		ringQP.NTTLvl(levelQ, levelP, u, u)
		ringQP.MFormLvl(levelQ, levelP, u, u)

		digitPair := &foldKey.Value[digit]
		ringQP.MulCoeffsMontgomeryLvl(levelQ, levelP, u, collectivePublicKey.Value[0], digitPair[0])
		ringQP.MulCoeffsMontgomeryLvl(levelQ, levelP, u, collectivePublicKey.Value[1], digitPair[1])
		ringQP.MFormLvl(levelQ, levelP, digitPair[0], digitPair[0])
		ringQP.MFormLvl(levelQ, levelP, digitPair[1], digitPair[1])

		addPublicEncryptionError(params, levelQ, levelP, gaussian, digitPair[0])
		addPublicEncryptionError(params, levelQ, levelP, gaussian, digitPair[1])

		for offset := 0; offset < alpha; offset++ {
			limb := digit*alpha + offset
			if limb >= levelQ+1 {
				break
			}
			qi := ringQ.Modulus[limb]
			for coefficient, target := range pTimesSecretMontgomery.Coeffs[limb] {
				digitPair[0].Q.Coeffs[limb][coefficient] = ring.CRed(digitPair[0].Q.Coeffs[limb][coefficient]+target, qi)
			}
		}
		digitPair[0].Q.IsNTT, digitPair[0].P.IsNTT = true, true
		digitPair[1].Q.IsNTT, digitPair[1].P.IsNTT = true, true
	}
	return foldKey, nil
}

func specialModulusBigint(params rlwe.Parameters, levelP int) *big.Int {
	modulus := new(big.Int).SetUint64(params.P()[0])
	for index := 1; index <= levelP; index++ {
		modulus.Mul(modulus, new(big.Int).SetUint64(params.P()[index]))
	}
	return modulus
}

func addPublicEncryptionError(params rlwe.Parameters, levelQ, levelP int, sampler *ring.GaussianSampler, output rlwe.PolyQP) {
	errorPoly := params.RingQP().NewPolyLvl(levelQ, levelP)
	sampler.ReadLvl(levelQ, errorPoly.Q)
	params.RingQP().ExtendBasisSmallNormAndCenter(errorPoly.Q, levelP, nil, errorPoly.P)
	params.RingQP().NTTLvl(levelQ, levelP, errorPoly, errorPoly)
	params.RingQP().MFormLvl(levelQ, levelP, errorPoly, errorPoly)
	params.RingQP().AddLvl(levelQ, levelP, output, errorPoly, output)
}

// switchingKeyRelationMaxCentered verifies the algebraic public-key relation
// digit by digit. aggregate is reconstructed only by the test and never by
// generateDistributedPublicFoldKeys.
func switchingKeyRelationMaxCentered(params rlwe.Parameters, foldKey *rlwe.SwitchingKey, secretShare, aggregate *rlwe.SecretKey) uint64 {
	levelQ, levelP := params.QCount()-1, params.PCount()-1
	ringQ, ringP, ringQP := params.RingQ(), params.RingP(), params.RingQP()
	pTimesSecretMontgomery := ringQ.NewPolyLvl(levelQ)
	ringQ.MulScalarBigintLvl(levelQ, secretShare.Value.Q, specialModulusBigint(params, levelP), pTimesSecretMontgomery)
	alpha := levelP + 1
	var maximum uint64
	for digit := range foldKey.Value {
		phase := foldKey.Value[digit][0].CopyNew()
		product := ringQP.NewPolyLvl(levelQ, levelP)
		ringQP.MulCoeffsMontgomeryLvl(levelQ, levelP, foldKey.Value[digit][1], aggregate.Value, product)
		ringQP.AddLvl(levelQ, levelP, phase, product, phase)
		for offset := 0; offset < alpha; offset++ {
			limb := digit*alpha + offset
			if limb >= levelQ+1 {
				break
			}
			qi := ringQ.Modulus[limb]
			for coefficient, target := range pTimesSecretMontgomery.Coeffs[limb] {
				phase.Q.Coeffs[limb][coefficient] = ring.CRed(phase.Q.Coeffs[limb][coefficient]+qi-target, qi)
			}
		}
		ringQP.InvMFormLvl(levelQ, levelP, phase, phase)
		ringQP.InvNTTLvl(levelQ, levelP, phase, phase)
		ringQ.ReduceLvl(levelQ, phase.Q, phase.Q)
		ringP.ReduceLvl(levelP, phase.P, phase.P)
		maximum = maxCenteredPolyLimbDistance(maximum, phase.Q, ringQ.Modulus[:levelQ+1])
		maximum = maxCenteredPolyLimbDistance(maximum, phase.P, ringP.Modulus[:levelP+1])
	}
	return maximum
}

func maxCenteredPolyLimbDistance(current uint64, polynomial *ring.Poly, moduli []uint64) uint64 {
	for limb, modulus := range moduli {
		half := modulus >> 1
		for _, coefficient := range polynomial.Coeffs[limb] {
			coefficient %= modulus
			if coefficient > half {
				coefficient = modulus - coefficient
			}
			if coefficient > current {
				current = coefficient
			}
		}
	}
	return current
}

func TestDistributedPublicFoldKeyGenRelationAndTranscript(t *testing.T) {
	params, err := CKKSParameters("P1")
	if err != nil {
		t.Fatal(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	shares := []*rlwe.SecretKey{kgen.GenSecretKey(), kgen.GenSecretKey()}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	material, err := generateDistributedPublicFoldKeys(params, shares, crs)
	if err != nil {
		t.Fatal(err)
	}
	aggregate := aggregateSecret(params.Parameters, shares)
	for party := range shares {
		maximum := switchingKeyRelationMaxCentered(params.Parameters, material.FoldKeys[party], shares[party], aggregate)
		if maximum == 0 || maximum > 1<<32 {
			t.Fatalf("party %d public fold-key relation noise is implausible: %d", party, maximum)
		}
		t.Logf("party=%d digit_relation_max_centered=%d", party, maximum)
	}
	transcript := material.Transcript
	if transcript.ConstructionID != distributedPublicFoldKeyConstructionID || transcript.Parties != 2 || transcript.Digits != params.Beta() || transcript.CRPBytes == 0 || transcript.CKGRounds != 1 || transcript.CKGShareMessages != 2 || transcript.PublishedFoldKeyMessages != 2 || transcript.CollectivePublicKeyBytes == 0 || transcript.PublishedFoldKeyBytes == 0 || transcript.TotalPublicMaterialBytes != transcript.CollectivePublicKeyBytes+transcript.PublishedFoldKeyBytes || transcript.TransmittedBytes == 0 || len(transcript.CKGShareBytes) != 2 || len(transcript.FoldKeyBytes) != 2 || len(transcript.SHA256) != sha256.Size*2 {
		t.Fatalf("incomplete public transcript evidence: %+v", transcript)
	}
	for party := range shares {
		if transcript.CKGShareBytes[party] == 0 || transcript.FoldKeyBytes[party] == 0 {
			t.Fatalf("party %d missing transcript bytes: %+v", party, transcript)
		}
	}
	t.Logf("PUBLIC_FOLD_KEYGEN_TRANSCRIPT=%+v", transcript)
}

func TestDistributedPublicFoldKeyGenEndToEndAgainstCentralReference(t *testing.T) {
	params, err := CKKSParameters(guardedCompactP2GParameterID)
	if err != nil {
		t.Fatal(err)
	}
	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, fullResNet20Parties)
	public := make([]*rlwe.PublicKey, fullResNet20Parties)
	for party := range shares {
		shares[party], public[party] = kgen.GenKeyPair()
	}
	aggregate := aggregateSecret(params.Parameters, shares)
	central := make([]*rlwe.SwitchingKey, len(shares))
	for party := range shares {
		central[party] = kgen.GenSwitchingKey(shares[party], aggregate)
	}
	crs, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	material, err := generateDistributedPublicFoldKeys(params, shares, crs)
	if err != nil {
		t.Fatal(err)
	}

	encoder := ckks.NewEncoder(params)
	want := []complex128{0.125, -0.25, 0.375, -0.0625}
	centralAdapter, err := newGuardedCompactMKFLattigoAdapter(params, rlwe.EvaluationKey{}, central, len(shares))
	if err != nil {
		t.Fatal(err)
	}
	distributedAdapter, err := newGuardedCompactMKFLattigoAdapter(params, rlwe.EvaluationKey{}, material.FoldKeys, len(shares))
	if err != nil {
		t.Fatal(err)
	}
	centralCiphertext, err := centralAdapter.ownerCarrier(encoder, public, 1, want)
	if err != nil {
		t.Fatal(err)
	}
	distributedCiphertext, err := distributedAdapter.ownerCarrier(encoder, public, 1, want)
	if err != nil {
		t.Fatal(err)
	}
	decryptor := ckks.NewDecryptor(params, aggregate)
	centralDecoded := encoder.Decode(decryptor.DecryptNew(centralCiphertext), params.LogSlots())
	distributedDecoded := encoder.Decode(decryptor.DecryptNew(distributedCiphertext), params.LogSlots())
	var centralError, distributedError, crossError float64
	for index := range want {
		centralError = math.Max(centralError, math.Abs(real(centralDecoded[index]-want[index])))
		distributedError = math.Max(distributedError, math.Abs(real(distributedDecoded[index]-want[index])))
		crossError = math.Max(crossError, math.Abs(real(distributedDecoded[index]-centralDecoded[index])))
	}
	if centralError > 1e-6 || distributedError > 1e-6 || crossError > 1e-6 {
		t.Fatalf("fold-key end-to-end mismatch: central=%g distributed=%g cross=%g", centralError, distributedError, crossError)
	}
	t.Logf("central_max_error=%g distributed_max_error=%g cross_max_error=%g transcript_sha256=%s", centralError, distributedError, crossError, material.Transcript.SHA256)
}
