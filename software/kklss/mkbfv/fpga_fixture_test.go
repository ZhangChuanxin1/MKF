package mkbfv

// Opt-in P1/n=16 fixture export for the U55C HLS implementation. The binary layout
// is little-endian uint64, modulus-major and then coefficient-major.

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

	"github.com/ldsec/lattigo/v2/ring"
)

type fpgaBFVFixtureManifest struct {
	Scheme               string            `json:"scheme"`
	Parameter            string            `json:"parameter"`
	Parties              int               `json:"parties"`
	PolynomialDegree     int               `json:"polynomial_degree"`
	Q                    []uint64          `json:"q"`
	QMul                 []uint64          `json:"q_mul"`
	P                    []uint64          `json:"p"`
	PlaintextModulus     uint64            `json:"plaintext_modulus"`
	InputLevelQ          int               `json:"input_level_q"`
	OutputLevelQ         int               `json:"output_level_q"`
	Beta                 int               `json:"beta"`
	CiphertextOrder      []string          `json:"ciphertext_order"`
	SwitchingKeyOrder    string            `json:"switching_key_order"`
	InputCiphertextFile  string            `json:"input_ciphertext_file"`
	RelinKeyFile         string            `json:"relinearization_key_file"`
	CRSUFile             string            `json:"crs_u_file"`
	OutputCiphertextFile string            `json:"output_ciphertext_file"`
	SHA256               map[string]string `json:"sha256"`
}

func writeBFVPoly(w io.Writer, poly *ring.Poly) error {
	for _, rns := range poly.Coeffs {
		for _, coefficient := range rns {
			if err := binary.Write(w, binary.LittleEndian, coefficient); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeBFVCiphertext(path string, ct *Ciphertext, order []string) error {
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
		if err := writeBFVPoly(w, poly); err != nil {
			return err
		}
	}
	return w.Flush()
}

func writeBFVRelinKeys(path string, keys *RelinearizationKeySet, order []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, id := range order {
		key, ok := keys.Value[id]
		if !ok {
			return fmt.Errorf("relinearization key is missing user %q", id)
		}
		for branch := 0; branch < 2; branch++ {
			for component := 0; component < 3; component++ {
				swk := key.Value[branch].Value[component]
				for decomposition := range swk.Value {
					polyQP := swk.Value[decomposition]
					if err := writeBFVPoly(w, polyQP.Q); err != nil {
						return err
					}
					if err := writeBFVPoly(w, polyQP.P); err != nil {
						return err
					}
				}
			}
		}
	}
	return w.Flush()
}

func writeBFVSwitchingKey(path string, key *mkrlwe.SwitchingKey) error {
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, polyQP := range key.Value {
		if err := writeBFVPoly(w, polyQP.Q); err != nil { return err }
		if err := writeBFVPoly(w, polyQP.P); err != nil { return err }
	}
	return w.Flush()
}

func hashBFVFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func TestExportP1N16Fixture(t *testing.T) {
	dir := os.Getenv("FPGA_FIXTURE_DIR")
	if dir == "" {
		t.Skip("set FPGA_FIXTURE_DIR to export the P1/n=16 KKLSS-BFV fixture")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	params := NewParametersFromLiteral(PN14QP439)
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

	msgList := make([]*Message, len(ids))
	ctList := make([]*Ciphertext, len(ids))
	for i, id := range ids {
		msgList[i], ctList[i] = newTestVectors(ctx, id, 0, 2)
	}
	input := ctList[0]
	expectedMessage := msgList[0]
	for i := range ids {
		input = ctx.evaluator.AddNew(input, ctList[i])
		for j := range expectedMessage.Value {
			expectedMessage.Value[j] += msgList[i].Value[j]
		}
	}
	for i := range expectedMessage.Value {
		expectedMessage.Value[i] *= expectedMessage.Value[i]
	}
	leftR := params.RingR().NewPoly()
	rightR := params.RingR().NewPoly()
	ctx.evaluator.conv.ModUpQtoR(input.Value["0"], leftR)
	ctx.evaluator.conv.Rescale(input.Value["0"], rightR)
	func() {
		f, err := os.Create(filepath.Join(dir, "stage_lift_left0_r.bin")); if err != nil { t.Fatal(err) }; defer f.Close()
		w := bufio.NewWriterSize(f, 4<<20); if err = writeBFVPoly(w, leftR); err != nil { t.Fatal(err) }; if err = w.Flush(); err != nil { t.Fatal(err) }
	}()
	func() {
		f, err := os.Create(filepath.Join(dir, "stage_lift_right0_r.bin")); if err != nil { t.Fatal(err) }; defer f.Close()
		w := bufio.NewWriterSize(f, 4<<20); if err = writeBFVPoly(w, rightR); err != nil { t.Fatal(err) }; if err = w.Flush(); err != nil { t.Fatal(err) }
	}()
	order := append([]string{"0"}, ids...)
	fpgaMulRelinStageHook = func(stage string, ct *mkrlwe.Ciphertext) {
		wrapped := &Ciphertext{Ciphertext: ct}
		if err := writeBFVCiphertext(filepath.Join(dir, "stage_"+stage+"_q.bin"), wrapped, order); err != nil { t.Fatal(err) }
	}
	fpgaBFVSwitchHook = func(stage string, keys ...*mkrlwe.SwitchingKey) {
		for i, key := range keys { if err := writeBFVSwitchingKey(filepath.Join(dir, fmt.Sprintf("stage_%s_switch%d_qp.bin", stage, i)), key); err != nil { t.Fatal(err) } }
	}
	fpgaBFVPolyHook = func(stage string, poly *ring.Poly) {
		f, err := os.Create(filepath.Join(dir, "stage_"+stage+"_q.bin")); if err != nil { t.Fatal(err) }; defer f.Close()
		w := bufio.NewWriterSize(f, 4<<20); if err = writeBFVPoly(w, poly); err != nil { t.Fatal(err) }; if err = w.Flush(); err != nil { t.Fatal(err) }
	}
	output := ctx.evaluator.MulRelinNew(input, input, ctx.rlkSet)
	fpgaMulRelinStageHook = nil
	fpgaBFVSwitchHook = nil
	fpgaBFVPolyHook = nil
	decoded := ctx.decryptor.Decrypt(output, ctx.skSet)
	for i := range decoded.Value {
		if decoded.Value[i] != expectedMessage.Value[i] {
			t.Fatalf("software reference mismatch at coefficient %d: got %d, want %d", i, decoded.Value[i], expectedMessage.Value[i])
		}
	}

	inputPath := filepath.Join(dir, "input_ciphertext_q.bin")
	keyPath := filepath.Join(dir, "relinearization_keys_qp.bin")
	crsPath := filepath.Join(dir, "crs_u_qp.bin")
	outputPath := filepath.Join(dir, "expected_ciphertext_q.bin")
	if err := writeBFVCiphertext(inputPath, input, order); err != nil {
		t.Fatal(err)
	}
	if err := writeBFVRelinKeys(keyPath, ctx.rlkSet, ids); err != nil {
		t.Fatal(err)
	}
	if err := writeBFVSwitchingKey(crsPath, params.CRS[-1]); err != nil { t.Fatal(err) }
	if err := writeBFVCiphertext(outputPath, output, order); err != nil {
		t.Fatal(err)
	}

	hashes := make(map[string]string)
	for _, path := range []string{inputPath, keyPath, crsPath, outputPath} {
		hash, err := hashBFVFile(path)
		if err != nil {
			t.Fatal(err)
		}
		hashes[filepath.Base(path)] = hash
	}
	manifest := fpgaBFVFixtureManifest{
		Scheme: "KKLSS-BFV", Parameter: "P1", Parties: len(ids), PolynomialDegree: params.N(),
		Q: append([]uint64(nil), PN14QP439.Q...), QMul: append([]uint64(nil), PN14QP439.QMul...),
		P: append([]uint64(nil), PN14QP439.P...), PlaintextModulus: params.T(),
		InputLevelQ: input.Level(), OutputLevelQ: output.Level(), Beta: params.Beta(input.Level()),
		CiphertextOrder:     order,
		SwitchingKeyOrder:   "user, BFV_branch[0..1], relin_component[0..2], decomposition[0..beta-1], Q_then_P, modulus_major, coefficient_major",
		InputCiphertextFile: filepath.Base(inputPath), RelinKeyFile: filepath.Base(keyPath),
		CRSUFile: filepath.Base(crsPath),
		OutputCiphertextFile: filepath.Base(outputPath), SHA256: hashes,
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
