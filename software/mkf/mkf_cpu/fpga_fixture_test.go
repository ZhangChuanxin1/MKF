package mkfcpu

// Opt-in bit-exact fixture export for the U55C HLS implementation. Set
// FPGA_FIXTURE_ROOT and run TestExportP1N16Fixtures. All arrays are little-
// endian uint64, modulus-major and then coefficient-major.

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

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type mkfFixtureManifest struct {
	Scheme            string            `json:"scheme"`
	Parameter         string            `json:"parameter"`
	Parties           int               `json:"parties"`
	PolynomialDegree  int               `json:"polynomial_degree"`
	Q                 []uint64          `json:"q"`
	QMul              []uint64          `json:"q_mul,omitempty"`
	P                 []uint64          `json:"p"`
	PlaintextModulus  uint64            `json:"plaintext_modulus,omitempty"`
	InputLevelQ       int               `json:"input_level_q"`
	OutputLevelQ      int               `json:"output_level_q"`
	Beta              int               `json:"beta"`
	InputOrder        string            `json:"input_order"`
	SwitchingKeyOrder string            `json:"switching_key_order"`
	OutputOrder       string            `json:"output_order"`
	InputFile         string            `json:"input_file"`
	SwitchingKeyFile  string            `json:"switching_key_file"`
	OutputFile        string            `json:"output_file"`
	SHA256            map[string]string `json:"sha256"`
}

func writeMKFPoly(w io.Writer, poly *ring.Poly) error {
	for _, rns := range poly.Coeffs {
		for _, coefficient := range rns {
			if err := binary.Write(w, binary.LittleEndian, coefficient); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeMKFSwitchingKey(w io.Writer, key *rlwe.SwitchingKey) error {
	for decomposition := range key.Value {
		for outputPoly := 0; outputPoly < 2; outputPoly++ {
			polyQP := key.Value[decomposition][outputPoly]
			if err := writeMKFPoly(w, polyQP.Q); err != nil {
				return err
			}
			if err := writeMKFPoly(w, polyQP.P); err != nil {
				return err
			}
		}
	}
	return nil
}

func withMKFWriter(path string, fn func(io.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	if err := fn(w); err != nil {
		return err
	}
	return w.Flush()
}

func writeMKFInputs(path string, left, right *multiCiphertext) error {
	return withMKFWriter(path, func(w io.Writer) error {
		for _, input := range []*multiCiphertext{left, right} {
			if err := writeMKFPoly(w, input.C0); err != nil {
				return err
			}
			for _, component := range input.Components {
				if err := writeMKFPoly(w, component); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

func writeMKFKeys(path string, keys onlineKeys) error {
	return withMKFWriter(path, func(w io.Writer) error {
		for _, key := range keys.Fold {
			if err := writeMKFSwitchingKey(w, key); err != nil {
				return err
			}
		}
		if len(keys.Relin.Keys) == 0 {
			return fmt.Errorf("relinearization key has no switching keys")
		}
		return writeMKFSwitchingKey(w, keys.Relin.Keys[0])
	})
}

func writeMKFOutput(path string, output *outputCiphertext) error {
	return withMKFWriter(path, func(w io.Writer) error {
		if err := writeMKFPoly(w, output.C0); err != nil {
			return err
		}
		if err := writeMKFPoly(w, output.Gamma); err != nil {
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

func writeMKFPolys(path string, polys ...*ring.Poly) error {
	return withMKFWriter(path, func(w io.Writer) error {
		for _, poly := range polys {
			if err := writeMKFPoly(w, poly); err != nil {
				return err
			}
		}
		return nil
	})
}

func exportMKFCKKSStages(t *testing.T, dir string, c *CKKSCase) {
	t.Helper()
	prof := Profile{}
	w := newSwitchWorkspace(c.params.Parameters)
	level := c.params.MaxLevel()
	levelP := c.params.PCount() - 1
	w.ks.DecomposeNTT(level, levelP, c.params.Alpha(), c.left.Components[0], w.decomp)
	if err := withMKFWriter(filepath.Join(dir, "debug_switch0_decomposition_qp.bin"), func(writer io.Writer) error {
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
		t.Fatal(err)
	}
	w.ks.KeyswitchHoistedNoModDown(level, w.decomp, c.keys.Fold[0], w.c0Q, w.c1Q, w.c0P, w.c1P)
	if err := writeMKFPolys(filepath.Join(dir, "debug_switch0_external_qp.bin"), w.c0Q, w.c0P, w.c1Q, w.c1P); err != nil {
		t.Fatal(err)
	}
	w.ks.Baseconverter.ModDownQPtoQNTT(level, levelP, w.c0Q, w.c0P, w.c0Q)
	w.ks.Baseconverter.ModDownQPtoQNTT(level, levelP, w.c1Q, w.c1P, w.c1Q)
	if err := writeMKFPolys(filepath.Join(dir, "debug_switch0_moddown_q.bin"), w.c0Q, w.c1Q); err != nil {
		t.Fatal(err)
	}

	w = newSwitchWorkspace(c.params.Parameters)
	left, err := fold(c.params.Parameters, c.left, c.keys.Fold, true, w, &prof)
	if err != nil {
		t.Fatal(err)
	}
	right, err := fold(c.params.Parameters, c.right, c.keys.Fold, true, w, &prof)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(filepath.Join(dir, "stage_fold_q.bin"), left[0], left[1], right[0], right[1]); err != nil {
		t.Fatal(err)
	}
	ringQ := c.params.RingQ()
	mu0, mu1, mu2 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftM0, leftM1 := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	leftSum, rightSum := ringQ.NewPolyLvl(level), ringQ.NewPolyLvl(level)
	ringQ.MFormLvl(level, left[0], leftM0)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM0, right[0], mu0)
	ringQ.MFormLvl(level, left[1], leftM1)
	ringQ.MulCoeffsMontgomeryLvl(level, leftM1, right[1], mu2)
	ringQ.AddLvl(level, left[0], left[1], leftSum)
	ringQ.AddLvl(level, right[0], right[1], rightSum)
	ringQ.MFormLvl(level, leftSum, leftSum)
	ringQ.MulCoeffsMontgomeryLvl(level, leftSum, rightSum, mu1)
	ringQ.SubLvl(level, mu1, mu0, mu1)
	ringQ.SubLvl(level, mu1, mu2, mu1)
	mu0.IsNTT, mu1.IsNTT, mu2.IsNTT = true, true, true
	if err := writeMKFPolys(filepath.Join(dir, "stage_tensor_q.bin"), mu0, mu1, mu2); err != nil {
		t.Fatal(err)
	}
	relin0, relin1 := w.switchOne(c.params.Parameters, level, mu2, c.keys.Relin.Keys[0], true, &prof)
	ringQ.AddLvl(level, mu0, relin0, mu0)
	ringQ.AddLvl(level, mu1, relin1, mu1)
	if err := writeMKFPolys(filepath.Join(dir, "stage_relinearized_q.bin"), mu0, mu1); err != nil {
		t.Fatal(err)
	}
	compact := ckks.NewCiphertextAtLevelFromPoly(level, [2]*ring.Poly{mu0, mu1})
	compact.Scale = c.params.Scale() * c.params.Scale()
	if err := c.evaluator.Rescale(compact, c.params.Scale(), compact); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(filepath.Join(dir, "stage_rescaled_q.bin"), compact.Value[0], compact.Value[1]); err != nil {
		t.Fatal(err)
	}
}

func exportMKFBFVStages(t *testing.T, dir string, c *BFVCase) {
	t.Helper()
	prof := Profile{}
	w := newSwitchWorkspace(c.params.Parameters)
	level := c.params.MaxLevel()
	levelP := c.params.PCount() - 1
	w.ks.DecomposeNTT(level, levelP, c.params.Alpha(), c.left.Components[0], w.decomp)
	if err := withMKFWriter(filepath.Join(dir, "debug_switch0_decomposition_qp.bin"), func(writer io.Writer) error {
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
		t.Fatal(err)
	}
	w.ks.KeyswitchHoistedNoModDown(level, w.decomp, c.keys.Fold[0], w.c0Q, w.c1Q, w.c0P, w.c1P)
	if err := writeMKFPolys(filepath.Join(dir, "debug_switch0_external_qp.bin"), w.c0Q, w.c0P, w.c1Q, w.c1P); err != nil {
		t.Fatal(err)
	}
	c.params.RingQ().InvNTTLazyLvl(level, w.c0Q, w.c0Q)
	c.params.RingQ().InvNTTLazyLvl(level, w.c1Q, w.c1Q)
	c.params.RingP().InvNTTLazyLvl(levelP, w.c0P, w.c0P)
	c.params.RingP().InvNTTLazyLvl(levelP, w.c1P, w.c1P)
	w.ks.Baseconverter.ModDownQPtoQ(level, levelP, w.c0Q, w.c0P, w.c0Q)
	w.ks.Baseconverter.ModDownQPtoQ(level, levelP, w.c1Q, w.c1P, w.c1Q)
	if err := writeMKFPolys(filepath.Join(dir, "debug_switch0_moddown_q.bin"), w.c0Q, w.c1Q); err != nil {
		t.Fatal(err)
	}

	w = newSwitchWorkspace(c.params.Parameters)
	left, err := fold(c.params.Parameters, c.left, c.keys.Fold, false, w, &prof)
	if err != nil {
		t.Fatal(err)
	}
	right, err := fold(c.params.Parameters, c.right, c.keys.Fold, false, w, &prof)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeMKFPolys(filepath.Join(dir, "stage_fold_q.bin"), left[0], left[1], right[0], right[1]); err != nil {
		t.Fatal(err)
	}
	c.tensor.modUpLeft(left)
	c.tensor.rescaleRight(right)
	if err := writeMKFPolys(filepath.Join(dir, "stage_lifted_r.bin"),
		c.tensor.inR[0][0], c.tensor.inR[0][1], c.tensor.inR[1][0], c.tensor.inR[1][1]); err != nil {
		t.Fatal(err)
	}
	c.tensor.tensorKaratsuba()
	if err := writeMKFPolys(filepath.Join(dir, "stage_tensor_r.bin"), c.tensor.productR[0], c.tensor.productR[1], c.tensor.productR[2]); err != nil {
		t.Fatal(err)
	}
	product := c.tensor.quantize()
	if err := writeMKFPolys(filepath.Join(dir, "stage_quantized_q.bin"), product[0], product[1], product[2]); err != nil {
		t.Fatal(err)
	}
	relin0, relin1 := w.switchOne(c.params.Parameters, level, product[2], c.keys.Relin.Keys[0], false, &prof)
	c.params.RingQ().AddLvl(level, product[0], relin0, product[0])
	c.params.RingQ().AddLvl(level, product[1], relin1, product[1])
	if err := writeMKFPolys(filepath.Join(dir, "stage_relinearized_q.bin"), product[0], product[1]); err != nil {
		t.Fatal(err)
	}
}

func hashMKFFile(path string) (string, error) {
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

func exportMKFFixture(t *testing.T, dir string, manifest mkfFixtureManifest,
	left, right *multiCiphertext, keys onlineKeys, output *outputCiphertext) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	inputPath := filepath.Join(dir, "input_ciphertexts_q.bin")
	keyPath := filepath.Join(dir, "fold_and_relin_keys_qp.bin")
	outputPath := filepath.Join(dir, "expected_output_q.bin")
	if err := writeMKFInputs(inputPath, left, right); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFKeys(keyPath, keys); err != nil {
		t.Fatal(err)
	}
	if err := writeMKFOutput(outputPath, output); err != nil {
		t.Fatal(err)
	}
	manifest.InputFile = filepath.Base(inputPath)
	manifest.SwitchingKeyFile = filepath.Base(keyPath)
	manifest.OutputFile = filepath.Base(outputPath)
	manifest.SHA256 = make(map[string]string)
	for _, path := range []string{inputPath, keyPath, outputPath} {
		hash, err := hashMKFFile(path)
		if err != nil {
			t.Fatal(err)
		}
		manifest.SHA256[filepath.Base(path)] = hash
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestExportP1N16Fixtures(t *testing.T) {
	root := os.Getenv("FPGA_FIXTURE_ROOT")
	if root == "" {
		t.Skip("set FPGA_FIXTURE_ROOT to export MKF P1/n=16 fixtures")
	}

	t.Run("CKKS", func(t *testing.T) {
		c, err := NewCKKSCase("P1", 16)
		if err != nil {
			t.Fatal(err)
		}
		_, validation, err := c.Run()
		if err != nil {
			t.Fatal(err)
		}
		if !validation.Passed || c.lastOutput == nil {
			t.Fatalf("software reference failed: %+v", validation)
		}
		exportMKFFixture(t, filepath.Join(root, "mkf_ckks_p1_n16"), mkfFixtureManifest{
			Scheme: "MKF-CKKS", Parameter: "P1", Parties: c.parties,
			PolynomialDegree: c.params.N(), Q: append([]uint64(nil), c.params.Q()...), P: append([]uint64(nil), c.params.P()...),
			InputLevelQ: c.left.Level, OutputLevelQ: c.lastOutput.Level, Beta: c.params.Beta(),
			InputOrder:        "side[left,right], component[c0,party0..party15], modulus, coefficient",
			SwitchingKeyOrder: "fold_key[party0..party15], then aggregate_relin_key; decomposition, output_poly[0..1], Q_then_P, modulus, coefficient",
			OutputOrder:       "c0, gamma, repeated_gamma[party0..party15], modulus, coefficient",
		}, c.left, c.right, c.keys, c.lastOutput)
		exportMKFCKKSStages(t, filepath.Join(root, "mkf_ckks_p1_n16"), c)
	})

	t.Run("BFV", func(t *testing.T) {
		c, err := NewBFVCase("P1", 16)
		if err != nil {
			t.Fatal(err)
		}
		_, validation, err := c.Run()
		if err != nil {
			t.Fatal(err)
		}
		if !validation.Passed || c.lastOutput == nil {
			t.Fatalf("software reference failed: %+v", validation)
		}
		exportMKFFixture(t, filepath.Join(root, "mkf_bfv_p1_n16"), mkfFixtureManifest{
			Scheme: "MKF-BFV", Parameter: "P1", Parties: c.parties,
			PolynomialDegree: c.params.N(), Q: append([]uint64(nil), c.params.Q()...),
			QMul: append([]uint64(nil), bfvP1QMul...), P: append([]uint64(nil), c.params.P()...),
			PlaintextModulus: c.params.T(), InputLevelQ: c.left.Level,
			OutputLevelQ: c.lastOutput.Level, Beta: c.params.Beta(),
			InputOrder:        "side[left,right], component[c0,party0..party15], modulus, coefficient",
			SwitchingKeyOrder: "fold_key[party0..party15], then aggregate_relin_key; decomposition, output_poly[0..1], Q_then_P, modulus, coefficient",
			OutputOrder:       "c0, gamma, repeated_gamma[party0..party15], modulus, coefficient",
		}, c.left, c.right, c.keys, c.lastOutput)
		exportMKFBFVStages(t, filepath.Join(root, "mkf_bfv_p1_n16"), c)
	})
}
