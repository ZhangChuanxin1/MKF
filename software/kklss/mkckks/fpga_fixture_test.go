package mkckks

// This test is an explicit bridge from the bit-exact KKLSS reference to the
// U55C HLS implementation. It is deliberately opt-in: set FPGA_FIXTURE_DIR and run
// only TestExportP1N16Fixture. The output is a stable-layout little-endian
// uint64 RNS dump plus a JSON manifest consumed by the FPGA test harness.

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"mk-lattigo/mkrlwe"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
)

type fpgaFixtureManifest struct {
	Scheme               string            `json:"scheme"`
	Parameter            string            `json:"parameter"`
	Parties              int               `json:"parties"`
	PolynomialDegree     int               `json:"polynomial_degree"`
	Q                    []uint64          `json:"q"`
	P                    []uint64          `json:"p"`
	InputLevelQ          int               `json:"input_level_q"`
	OutputLevelQ         int               `json:"output_level_q"`
	Beta                 int               `json:"beta"`
	CiphertextOrder      []string          `json:"ciphertext_order"`
	SwitchingKeyOrder    string            `json:"switching_key_order"`
	InputCiphertextFile  string            `json:"input_ciphertexts_file"`
	InputOrder           string            `json:"input_order"`
	RelinKeyFile         string            `json:"relinearization_key_file"`
	CRSUFile             string            `json:"crs_u_file"`
	OutputCiphertextFile string            `json:"output_ciphertext_file"`
	SHA256               map[string]string `json:"sha256"`
}

func writePoly(w io.Writer, poly *ring.Poly) error {
	for _, rns := range poly.Coeffs {
		for _, coefficient := range rns {
			if err := binary.Write(w, binary.LittleEndian, coefficient); err != nil {
				return err
			}
		}
	}
	return nil
}

func writePolyQP(w io.Writer, polyQP mkrlwePolyQP) error {
	if err := writePoly(w, polyQP.Q); err != nil {
		return err
	}
	return writePoly(w, polyQP.P)
}

// Keep this local alias explicit so the file layout does not depend on
// reflection over the Lattigo representation.
type mkrlwePolyQP struct {
	Q *ring.Poly
	P *ring.Poly
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func writeCiphertextFile(path string, ct *Ciphertext, order []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, id := range order {
		poly, ok := ct.Value[id]
		if !ok {
			return fmt.Errorf("ciphertext is missing component %q", id)
		}
		if err := writePoly(w, poly); err != nil {
			return err
		}
	}
	return w.Flush()
}

func writeCiphertextPairFile(path string, left, right *Ciphertext, order []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, ct := range []*Ciphertext{left, right} {
		for _, id := range order {
			poly, ok := ct.Value[id]
			if !ok {
				return fmt.Errorf("ciphertext is missing component %q", id)
			}
			if err := writePoly(w, poly); err != nil {
				return err
			}
		}
	}
	return w.Flush()
}

func writeRelinKeyFile(path string, rlkSet *mkrlwe.RelinearizationKeySet, order []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, id := range order {
		rlk, ok := rlkSet.Value[id]
		if !ok {
			return fmt.Errorf("relinearization key is missing user %q", id)
		}
		for component := 0; component < 3; component++ {
			swk := rlk.Value[component]
			for decomposition := range swk.Value {
				polyQP := swk.Value[decomposition]
				if err := writePolyQP(w, mkrlwePolyQP{Q: polyQP.Q, P: polyQP.P}); err != nil {
					return err
				}
			}
		}
	}
	return w.Flush()
}

func writeSwitchingKeyFile(path string, swk *mkrlwe.SwitchingKey) error {
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, polyQP := range swk.Value {
		if err := writePolyQP(w, mkrlwePolyQP{Q: polyQP.Q, P: polyQP.P}); err != nil { return err }
	}
	return w.Flush()
}

func TestExportP1N16Fixture(t *testing.T) {
	dir := os.Getenv("FPGA_FIXTURE_DIR")
	if dir == "" {
		t.Skip("set FPGA_FIXTURE_DIR to export the P1/n=16 KKLSS-CKKS fixture")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	paramsCKKS, err := ckks.NewParametersFromLiteral(PN14QP439)
	if err != nil {
		t.Fatal(err)
	}
	params := NewParameters(paramsCKKS)
	ids := make([]string, 16)
	idset := mkrlwe.NewIDSet()
	for i := range ids {
		ids[i] = fmt.Sprintf("user%d", i)
		idset.Add(ids[i])
	}
	ctx, err := genTestParams(params, idset)
	if err != nil {
		t.Fatal(err)
	}

	ctList := make([]*Ciphertext, len(ids))
	for i, id := range ids {
		_, ctList[i] = newTestVectors(ctx, id,
			complex(0.1/float64(len(ids)), 1.0/float64(len(ids))),
			complex(0.1/float64(len(ids)), 1.0/float64(len(ids))))
	}
	// Preserve the exact operand construction used by
	// TestProfileStagesExactP1N16, including the first component appearing in
	// both the accumulator initialization and the loop.
	left, right := ctList[0], ctList[0]
	for i := range ids {
		left = ctx.evaluator.AddNew(left, ctList[i])
		right = ctx.evaluator.SubNew(right, ctList[i])
	}
	output := ctx.evaluator.MulRelinNew(left, right, ctx.rlkSet)

	order := append([]string{"0"}, ids...)
	inputPath := filepath.Join(dir, "input_ciphertexts_q.bin")
	keyPath := filepath.Join(dir, "relinearization_keys_qp.bin")
	crsPath := filepath.Join(dir, "crs_u_qp.bin")
	outputPath := filepath.Join(dir, "expected_ciphertext_q.bin")
	if err := writeCiphertextPairFile(inputPath, left, right, order); err != nil {
		t.Fatal(err)
	}
	if err := writeRelinKeyFile(keyPath, ctx.rlkSet, ids); err != nil {
		t.Fatal(err)
	}
	if err := writeSwitchingKeyFile(crsPath, params.CRS[-1]); err != nil { t.Fatal(err) }
	if err := writeCiphertextFile(outputPath, output, order); err != nil {
		t.Fatal(err)
	}

	hashes := make(map[string]string)
	for _, path := range []string{inputPath, keyPath, crsPath, outputPath} {
		hash, err := sha256File(path)
		if err != nil {
			t.Fatal(err)
		}
		hashes[filepath.Base(path)] = hash
	}
	manifest := fpgaFixtureManifest{
		Scheme: "KKLSS-CKKS", Parameter: "P1", Parties: len(ids),
		PolynomialDegree: params.N(), Q: append([]uint64(nil), params.Q()...), P: append([]uint64(nil), params.P()...),
		InputLevelQ: left.Level(), OutputLevelQ: output.Level(),
		Beta: params.Parameters.Beta(left.Level()), CiphertextOrder: order,
		SwitchingKeyOrder:   "user, relin_component[0..2], decomposition[0..beta-1], Q_then_P, modulus_major, coefficient_major",
		InputCiphertextFile: filepath.Base(inputPath), RelinKeyFile: filepath.Base(keyPath),
		CRSUFile: filepath.Base(crsPath),
		InputOrder:           "side[left,right], component[0,user0..user15], modulus, coefficient",
		OutputCiphertextFile: filepath.Base(outputPath), SHA256: hashes,
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}

	// The test vector must remain generated by the same code path used for the
	// software correctness test; avoid an unused import changing that property.
	_ = utils.RandUint64
}
