package mkfcpu

// This opt-in test exports a P1/n=16 fixture for the paper's
// party-specific-Galois-key MKF-CKKS rotation. It is intentionally separate
// from the multiplication fixture: the Galois-key target relation and the
// automorphism boundary are both independently materialized and checked.

import (
	"encoding/json"
	"fmt"
	"io"
	"math/cmplx"
	"os"
	"path/filepath"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type rotationFixtureManifest struct {
	Scheme                 string            `json:"scheme"`
	Parameter              string            `json:"parameter"`
	Parties                int               `json:"parties"`
	PolynomialDegree       int               `json:"polynomial_degree"`
	Q                      []uint64          `json:"q"`
	P                      []uint64          `json:"p"`
	InputLevelQ            int               `json:"input_level_q"`
	Beta                   int               `json:"beta"`
	RotationSlots          int               `json:"rotation_slots"`
	GaloisElement          uint64            `json:"galois_element"`
	InputOrder             string            `json:"input_order"`
	GaloisKeyOrder         string            `json:"galois_key_order"`
	OutputOrder            string            `json:"output_order"`
	AutomorphismCoeffFile  string            `json:"automorphism_coeff_file"`
	AutomorphismNTTFile    string            `json:"automorphism_ntt_file"`
	InputFile              string            `json:"input_file"`
	GaloisKeyFile          string            `json:"galois_key_file"`
	Switch0DecompFile      string            `json:"switch0_decomposition_qp_file"`
	Switch0ExternalFile    string            `json:"switch0_external_qp_file"`
	Switch0ModDownFile     string            `json:"switch0_moddown_q_file"`
	GaloisAccumFile        string            `json:"galois_accum_q_file"`
	OutputFile             string            `json:"output_file"`
	SHA256                 map[string]string `json:"sha256"`
}

type rotationTrace struct {
	coefficients  []*ring.Poly
	ntt           []*ring.Poly
	galoisKeys    []*rlwe.SwitchingKey
	accumulated0  *ring.Poly
	accumulated1  *ring.Poly
	output        *outputCiphertext
	expectedSlots []complex128
	observedSlots []complex128
}

func permuteNTTThroughCoefficientDomain(params rlwe.Parameters, level int, in *ring.Poly, galEl uint64) (*ring.Poly, *ring.Poly) {
	ringQ := params.RingQ()
	coeffIn := ringQ.NewPolyLvl(level)
	coeffOut := ringQ.NewPolyLvl(level)
	nttOut := ringQ.NewPolyLvl(level)
	ringQ.InvNTTLvl(level, in, coeffIn)
	ringQ.Permute(coeffIn, galEl, coeffOut)
	ringQ.NTTLvl(level, coeffOut, nttOut)
	coeffOut.IsNTT = false
	nttOut.IsNTT = true
	return coeffOut, nttOut
}

func permutedSecret(params rlwe.Parameters, sk *rlwe.SecretKey, galEl uint64) *rlwe.SecretKey {
	out := rlwe.NewSecretKey(params)
	index := ring.PermuteNTTIndex(galEl, uint64(params.N()))
	ring.PermuteNTTWithIndexLvl(params.QCount()-1, sk.Value.Q, index, out.Value.Q)
	ring.PermuteNTTWithIndexLvl(params.PCount()-1, sk.Value.P, index, out.Value.P)
	return out
}

func aggregatePhase(params rlwe.Parameters, level int, c *multiCiphertext, source []*rlwe.SecretKey) (*ring.Poly, error) {
	if len(c.Components) != len(source) {
		return nil, fmt.Errorf("component/key count mismatch: %d != %d", len(c.Components), len(source))
	}
	ringQ := params.RingQ()
	phase := ringQ.NewPolyLvl(level)
	phase.CopyValues(c.C0)
	for i, component := range c.Components {
		mform := ringQ.NewPolyLvl(level)
		product := ringQ.NewPolyLvl(level)
		ringQ.MFormLvl(level, component, mform)
		ringQ.MulCoeffsMontgomeryLvl(level, mform, source[i].Value.Q, product)
		ringQ.AddLvl(level, phase, product, phase)
	}
	phase.IsNTT = true
	return phase, nil
}

// runPartySpecificGaloisRotation implements the exact interface specified in
// the manuscript: after sigma_kappa, each party-specific Galois key switches
// sigma_kappa(s_i) to the aggregate secret s_E, and the resulting Gamma is
// repeated in the n nonconstant output components.
func runPartySpecificGaloisRotation(c *CKKSCase, rotationSlots int) (*rotationTrace, error) {
	level := c.left.Level
	params := c.params.Parameters
	ringQ := params.RingQ()
	galEl := c.params.GaloisElementForColumnRotationBy(rotationSlots)

	inputs := make([]*ring.Poly, c.parties+1)
	inputs[0] = c.left.C0
	copy(inputs[1:], c.left.Components)

	trace := &rotationTrace{
		coefficients: make([]*ring.Poly, len(inputs)),
		ntt:          make([]*ring.Poly, len(inputs)),
		galoisKeys:   make([]*rlwe.SwitchingKey, c.parties),
	}
	for i, input := range inputs {
		trace.coefficients[i], trace.ntt[i] = permuteNTTThroughCoefficientDomain(params, level, input, galEl)
	}

	kgen := ckks.NewKeyGenerator(c.params)
	for i := 0; i < c.parties; i++ {
		// This is the paper's key relation: d_i^rot+s_E*k_i^rot
		// encrypts sigma_kappa(s_i) under s_E.
		trace.galoisKeys[i] = kgen.GenSwitchingKey(permutedSecret(params, c.secret[i], galEl), c.aggregate)
	}

	trace.accumulated0 = trace.ntt[0].CopyNew()
	trace.accumulated0.IsNTT = true
	trace.accumulated1 = ringQ.NewPolyLvl(level)
	trace.accumulated1.IsNTT = true
	workspace := newSwitchWorkspace(params)
	profile := Profile{}
	for i := 0; i < c.parties; i++ {
		u0, u1 := workspace.switchOne(params, level, trace.ntt[i+1], trace.galoisKeys[i], true, &profile)
		ringQ.AddLvl(level, trace.accumulated0, u0, trace.accumulated0)
		ringQ.AddLvl(level, trace.accumulated1, u1, trace.accumulated1)
	}
	trace.output = materialize(params, trace.accumulated0, trace.accumulated1, c.parties, level, c.left.Scale, &profile)

	// Independently validate slot semantics against Lattigo's standard
	// aggregate-secret rotation path. It intentionally has different
	// switching-key noise from the party-specific construction above, so this
	// is a plaintext/slot oracle rather than a word-for-word output oracle.
	folded, err := fold(params, c.left, c.keys.Fold, true, newSwitchWorkspace(params), &Profile{})
	if err != nil {
		return nil, err
	}
	compact := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{folded[0], folded[1]})
	compact.Scale = c.left.Scale
	standardRotationKeys := kgen.GenRotationKeysForRotations([]int{rotationSlots}, false, c.aggregate)
	standardRotation := ckks.NewEvaluator(c.params, rlwe.EvaluationKey{Rtks: standardRotationKeys}).RotateNew(compact, rotationSlots)
	trace.expectedSlots = c.encoder.Decode(ckks.NewDecryptor(c.params, c.aggregate).DecryptNew(standardRotation), c.params.LogSlots())

	actualCT := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{trace.output.C0, trace.output.Gamma})
	actualCT.Scale = trace.output.Scale
	trace.observedSlots = c.encoder.Decode(ckks.NewDecryptor(c.params, c.aggregate).DecryptNew(actualCT), c.params.LogSlots())
	return trace, nil
}

func maxComplexDifference(left, right []complex128) (float64, error) {
	if len(left) != len(right) {
		return 0, fmt.Errorf("slot count mismatch: %d != %d", len(left), len(right))
	}
	var max float64
	for i := range left {
		if d := cmplx.Abs(left[i] - right[i]); d > max {
			max = d
		}
	}
	return max, nil
}

func equalPolyCoefficients(left, right *ring.Poly) bool {
	if len(left.Coeffs) != len(right.Coeffs) {
		return false
	}
	for m := range left.Coeffs {
		if len(left.Coeffs[m]) != len(right.Coeffs[m]) {
			return false
		}
		for i := range left.Coeffs[m] {
			if left.Coeffs[m][i] != right.Coeffs[m][i] {
				return false
			}
		}
	}
	return true
}

func writeRotationInput(path string, c *CKKSCase) error {
	return withMKFWriter(path, func(w io.Writer) error {
		if err := writeMKFPoly(w, c.left.C0); err != nil {
			return err
		}
		for _, component := range c.left.Components {
			if err := writeMKFPoly(w, component); err != nil {
				return err
			}
		}
		return nil
	})
}

func writeRotationKeys(path string, keys []*rlwe.SwitchingKey) error {
	return withMKFWriter(path, func(w io.Writer) error {
		for _, key := range keys {
			if err := writeMKFSwitchingKey(w, key); err != nil {
				return err
			}
		}
		return nil
	})
}

// The deployed rotation ciphertext uses the paper's n+1-component interface:
// (c0, Gamma, ..., Gamma).  Unlike the generic multiplication diagnostic
// serializer, it deliberately does not emit an extra compact Gamma copy.
func writeRotationOutput(path string, output *outputCiphertext) error {
	return withMKFWriter(path, func(w io.Writer) error {
		if err := writeMKFPoly(w, output.C0); err != nil {
			return err
		}
		for _, component := range output.Components {
			if err := writeMKFPoly(w, component); err != nil {
				return err
			}
		}
		return nil
	})
}

func writeFirstGaloisSwitchStages(pathDecomp, pathExternal, pathModDown string, c *CKKSCase, trace *rotationTrace) error {
	level := c.left.Level
	levelP := c.params.PCount() - 1
	w := newSwitchWorkspace(c.params.Parameters)
	w.ks.DecomposeNTT(level, levelP, c.params.Alpha(), trace.ntt[1], w.decomp)
	if err := withMKFWriter(pathDecomp, func(writer io.Writer) error {
		for _, poly := range w.decomp {
			if err := writeMKFPoly(writer, poly.Q); err != nil {
				return err
			}
			if err := writeMKFPoly(writer, poly.P); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return err
	}
	w.ks.KeyswitchHoistedNoModDown(level, w.decomp, trace.galoisKeys[0], w.c0Q, w.c1Q, w.c0P, w.c1P)
	if err := writeMKFPolys(pathExternal, w.c0Q, w.c0P, w.c1Q, w.c1P); err != nil {
		return err
	}
	w.ks.Baseconverter.ModDownQPtoQNTT(level, levelP, w.c0Q, w.c0P, w.c0Q)
	w.ks.Baseconverter.ModDownQPtoQNTT(level, levelP, w.c1Q, w.c1P, w.c1Q)
	return writeMKFPolys(pathModDown, w.c0Q, w.c1Q)
}

func TestExportP1N16PartySpecificRotationFixture(t *testing.T) {
	root := os.Getenv("FPGA_ROTATION_FIXTURE_ROOT")
	if root == "" {
		t.Skip("set FPGA_ROTATION_FIXTURE_ROOT to export the P1/n=16 MKF-CKKS rotation fixture")
	}
	const rotationSlots = 1
	c, err := NewCKKSCase("P1", 16)
	if err != nil {
		t.Fatal(err)
	}
	trace, err := runPartySpecificGaloisRotation(c, rotationSlots)
	if err != nil {
		t.Fatal(err)
	}
	maxError, err := maxComplexDifference(trace.expectedSlots, trace.observedSlots)
	if err != nil {
		t.Fatal(err)
	}
	if maxError >= 1e-6 {
		t.Fatalf("party-specific Galois-key rotation mismatch: max slot error %.4e expected[0]=%v observed[0]=%v expected[1]=%v observed[1]=%v", maxError, trace.expectedSlots[0], trace.observedSlots[0], trace.expectedSlots[1], trace.observedSlots[1])
	}
	galEl := c.params.GaloisElementForColumnRotationBy(rotationSlots)
	index := ring.PermuteNTTIndex(galEl, uint64(c.params.N()))
	for component, nttFromCoeffPath := range trace.ntt {
		directNTT := c.params.RingQ().NewPolyLvl(c.left.Level)
		input := c.left.C0
		if component > 0 {
			input = c.left.Components[component-1]
		}
		ring.PermuteNTTWithIndexLvl(c.left.Level, input, index, directNTT)
		if !equalPolyCoefficients(nttFromCoeffPath, directNTT) {
			t.Fatalf("coefficient-domain automorphism and direct NTT automorphism differ for component %d", component)
		}
	}

	dir := filepath.Join(root, "mkf_ckks_rotation_p1_n16")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	inputPath := filepath.Join(dir, "input_ciphertext_q.bin")
	keyPath := filepath.Join(dir, "galois_keys_qp.bin")
	autoCoeffPath := filepath.Join(dir, "stage_automorphism_coeff_q.bin")
	autoNTTPath := filepath.Join(dir, "stage_automorphism_ntt_q.bin")
	decompPath := filepath.Join(dir, "detail_galois_switch0_decomposition_qp.bin")
	externalPath := filepath.Join(dir, "detail_galois_switch0_external_qp.bin")
	modDownPath := filepath.Join(dir, "detail_galois_switch0_moddown_q.bin")
	accumPath := filepath.Join(dir, "stage_galois_accum_q.bin")
	outputPath := filepath.Join(dir, "expected_rotation_output_q.bin")

	if err := writeRotationInput(inputPath, c); err != nil {
		t.Fatal(err)
	}
	if err := writeRotationKeys(keyPath, trace.galoisKeys); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(autoCoeffPath, trace.coefficients...); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(autoNTTPath, trace.ntt...); err != nil {
		t.Fatal(err)
	}
	if err := writeFirstGaloisSwitchStages(decompPath, externalPath, modDownPath, c, trace); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(accumPath, trace.accumulated0, trace.accumulated1); err != nil {
		t.Fatal(err)
	}
	if err := writeRotationOutput(outputPath, trace.output); err != nil {
		t.Fatal(err)
	}

	paths := []string{inputPath, keyPath, autoCoeffPath, autoNTTPath, decompPath, externalPath, modDownPath, accumPath, outputPath}
	hashes := make(map[string]string, len(paths))
	for _, path := range paths {
		hash, err := hashMKFFile(path)
		if err != nil {
			t.Fatal(err)
		}
		hashes[filepath.Base(path)] = hash
	}
	manifest := rotationFixtureManifest{
		Scheme:                "MKF-CKKS party-specific-Galois-key rotation",
		Parameter:             "P1",
		Parties:               c.parties,
		PolynomialDegree:      c.params.N(),
		Q:                     append([]uint64(nil), c.params.Q()...),
		P:                     append([]uint64(nil), c.params.P()...),
		InputLevelQ:           c.left.Level,
		Beta:                  c.params.Beta(),
		RotationSlots:         rotationSlots,
		GaloisElement:         galEl,
		InputOrder:            "component[c0,party0..party15], modulus, coefficient; NTT-domain",
	GaloisKeyOrder:        "party0..party15; each key is sigma_g(s_i)->s_E; decomposition, output_poly[0..1], Q_then_P, modulus, coefficient",
	OutputOrder:            "component[c0,gamma_x16], modulus, coefficient; NTT-domain; 17 components",
		InputFile:             filepath.Base(inputPath),
		GaloisKeyFile:         filepath.Base(keyPath),
		AutomorphismCoeffFile: filepath.Base(autoCoeffPath),
		AutomorphismNTTFile:   filepath.Base(autoNTTPath),
		Switch0DecompFile:     filepath.Base(decompPath),
		Switch0ExternalFile:   filepath.Base(externalPath),
		Switch0ModDownFile:    filepath.Base(modDownPath),
		GaloisAccumFile:       filepath.Base(accumPath),
		OutputFile:            filepath.Base(outputPath),
		SHA256:                hashes,
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("MKF-CKKS party-specific Galois rotation fixture: slots=%d galEl=%d max_error=%.4e dir=%s", rotationSlots, galEl, maxError, dir)
}
