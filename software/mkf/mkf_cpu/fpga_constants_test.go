package mkfcpu

// This opt-in exporter serializes the exact Lattigo v2.3.0 NTT tables and
// scalar RNS constants consumed by the U55C HLS kernel. Keeping the tables derived
// from the same ring objects as the CPU reference avoids primitive-root or
// Montgomery-convention drift between software and hardware.

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	"github.com/ldsec/lattigo/v2/ring"
)

type modUpConstants struct {
	Source                 []uint64   `json:"source"`
	Targets                []uint64   `json:"targets"`
	QOverQiInvQiMontgomery []uint64   `json:"q_over_qi_inv_qi_montgomery"`
	QOverQiModTargetMont   [][]uint64 `json:"q_over_qi_mod_target_montgomery"`
	MinusQModTarget        []uint64   `json:"minus_q_mod_target"`
}

type ringConstants struct {
	N                    int              `json:"n"`
	Q                    []uint64         `json:"q"`
	P                    []uint64         `json:"p"`
	MRedQ                []uint64         `json:"mred_q"`
	MRedP                []uint64         `json:"mred_p"`
	NInvQMontgomery      []uint64         `json:"n_inv_q_montgomery"`
	NInvPMontgomery      []uint64         `json:"n_inv_p_montgomery"`
	DecompositionModUp   []modUpConstants `json:"decomposition_mod_up"`
	PToQModUp            modUpConstants   `json:"p_to_q_mod_up"`
	PInvQMontgomery      []uint64         `json:"p_inv_q_montgomery"`
	RescaleLevel5To4Mont []uint64         `json:"rescale_level5_to4_montgomery"`
}

type bfvExtraConstants struct {
	QMul               []uint64       `json:"q_mul"`
	MRedQMul           []uint64       `json:"mred_q_mul"`
	NInvQMulMontgomery []uint64       `json:"n_inv_q_mul_montgomery"`
	QToQMulModUp       modUpConstants `json:"q_to_q_mul_mod_up"`
	QMulToQModUp       modUpConstants `json:"q_mul_to_q_mod_up"`
	QInvQMulMontgomery []uint64       `json:"q_inv_q_mul_montgomery"`
	QMulInvQMontgomery []uint64       `json:"q_mul_inv_q_montgomery"`
	QMulMFormQ         []uint64       `json:"q_mul_mform_q"`
	PlaintextMFormR    []uint64       `json:"plaintext_mform_r"`
	Reciprocal96Q      []uint64       `json:"reciprocal_2pow96_q"`
	Reciprocal96QMul   []uint64       `json:"reciprocal_2pow96_q_mul"`
}

func mFormBig(x uint64, modulus uint64) uint64 {
	m := new(big.Int).SetUint64(modulus)
	r := new(big.Int).Lsh(big.NewInt(1), 64)
	v := new(big.Int).SetUint64(x)
	v.Mul(v, r).Mod(v, m)
	return v.Uint64()
}

func modUpFor(source, targets []uint64) modUpConstants {
	product := big.NewInt(1)
	for _, q := range source {
		product.Mul(product, new(big.Int).SetUint64(q))
	}
	out := modUpConstants{
		Source: append([]uint64(nil), source...), Targets: append([]uint64(nil), targets...),
		QOverQiInvQiMontgomery: make([]uint64, len(source)),
		QOverQiModTargetMont:   make([][]uint64, len(targets)),
		MinusQModTarget:        make([]uint64, len(targets)),
	}
	for i, qi := range source {
		qiBig := new(big.Int).SetUint64(qi)
		qStar := new(big.Int).Quo(new(big.Int).Set(product), qiBig)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(new(big.Int).Set(qStar), qiBig), qiBig)
		out.QOverQiInvQiMontgomery[i] = mFormBig(inv.Uint64(), qi)
		for j, pj := range targets {
			if out.QOverQiModTargetMont[j] == nil {
				out.QOverQiModTargetMont[j] = make([]uint64, len(source))
			}
			residue := new(big.Int).Mod(new(big.Int).Set(qStar), new(big.Int).SetUint64(pj)).Uint64()
			out.QOverQiModTargetMont[j][i] = mFormBig(residue, pj)
		}
	}
	for j, pj := range targets {
		qmodp := new(big.Int).Mod(new(big.Int).Set(product), new(big.Int).SetUint64(pj)).Uint64()
		if qmodp != 0 {
			out.MinusQModTarget[j] = pj - qmodp
		}
	}
	return out
}

func writeUint64Slices(path string, values [][]uint64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 4<<20)
	for _, valuesForModulus := range values {
		for _, value := range valuesForModulus {
			if err := binary.Write(w, binary.LittleEndian, value); err != nil {
				return err
			}
		}
	}
	return w.Flush()
}

func exportRingConstants(t *testing.T, dir string, ringQ, ringP *ring.Ring) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, values := range map[string][][]uint64{
		"ntt_forward_q.bin": ringQ.NttPsi, "ntt_inverse_q.bin": ringQ.NttPsiInv,
		"ntt_forward_p.bin": ringP.NttPsi, "ntt_inverse_p.bin": ringP.NttPsiInv,
	} {
		if err := writeUint64Slices(filepath.Join(dir, name), values); err != nil {
			t.Fatal(err)
		}
	}
	constants := ringConstants{
		N: ringQ.N, Q: append([]uint64(nil), ringQ.Modulus...), P: append([]uint64(nil), ringP.Modulus...),
		MRedQ: append([]uint64(nil), ringQ.MredParams...), MRedP: append([]uint64(nil), ringP.MredParams...),
		NInvQMontgomery: append([]uint64(nil), ringQ.NttNInv...), NInvPMontgomery: append([]uint64(nil), ringP.NttNInv...),
	}
	for start := 0; start < len(ringQ.Modulus); start += len(ringP.Modulus) {
		end := start + len(ringP.Modulus)
		constants.DecompositionModUp = append(constants.DecompositionModUp,
			modUpFor(ringQ.Modulus[start:end], append(append([]uint64(nil), ringQ.Modulus...), ringP.Modulus...)))
	}
	constants.PToQModUp = modUpFor(ringP.Modulus, ringQ.Modulus)
	pProduct := big.NewInt(1)
	for _, p := range ringP.Modulus {
		pProduct.Mul(pProduct, new(big.Int).SetUint64(p))
	}
	constants.PInvQMontgomery = make([]uint64, len(ringQ.Modulus))
	for i, q := range ringQ.Modulus {
		qBig := new(big.Int).SetUint64(q)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(new(big.Int).Set(pProduct), qBig), qBig)
		constants.PInvQMontgomery[i] = mFormBig(inv.Uint64(), q)
	}
	// DivRoundByLastModulusNTTLvl(5, ...) consumes this row.
	constants.RescaleLevel5To4Mont = append([]uint64(nil), ringQ.RescaleParams[4]...)
	encoded, err := json.MarshalIndent(constants, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ring_constants.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func productInverseMontgomery(source, targets []uint64) []uint64 {
	product := big.NewInt(1)
	for _, value := range source {
		product.Mul(product, new(big.Int).SetUint64(value))
	}
	result := make([]uint64, len(targets))
	for i, target := range targets {
		modulus := new(big.Int).SetUint64(target)
		inverse := new(big.Int).ModInverse(new(big.Int).Mod(new(big.Int).Set(product), modulus), modulus)
		result[i] = mFormBig(inverse.Uint64(), target)
	}
	return result
}

func reciprocal96(moduli []uint64) []uint64 {
	scale := new(big.Int).Lsh(big.NewInt(1), 96)
	result := make([]uint64, len(moduli))
	for i, modulus := range moduli {
		result[i] = new(big.Int).Quo(new(big.Int).Set(scale), new(big.Int).SetUint64(modulus)).Uint64()
	}
	return result
}

func exportBFVExtraConstants(t *testing.T, dir string, ringQ, ringQMul *ring.Ring, plaintext uint64) {
	t.Helper()
	if err := writeUint64Slices(filepath.Join(dir, "ntt_forward_q_mul.bin"), ringQMul.NttPsi); err != nil {
		t.Fatal(err)
	}
	if err := writeUint64Slices(filepath.Join(dir, "ntt_inverse_q_mul.bin"), ringQMul.NttPsiInv); err != nil {
		t.Fatal(err)
	}
	qMulProduct := big.NewInt(1)
	for _, modulus := range ringQMul.Modulus {
		qMulProduct.Mul(qMulProduct, new(big.Int).SetUint64(modulus))
	}
	extra := bfvExtraConstants{
		QMul: append([]uint64(nil), ringQMul.Modulus...), MRedQMul: append([]uint64(nil), ringQMul.MredParams...),
		NInvQMulMontgomery: append([]uint64(nil), ringQMul.NttNInv...),
		QToQMulModUp:       modUpFor(ringQ.Modulus, ringQMul.Modulus), QMulToQModUp: modUpFor(ringQMul.Modulus, ringQ.Modulus),
		QInvQMulMontgomery: productInverseMontgomery(ringQ.Modulus, ringQMul.Modulus),
		QMulInvQMontgomery: productInverseMontgomery(ringQMul.Modulus, ringQ.Modulus),
		QMulMFormQ:         make([]uint64, len(ringQ.Modulus)), PlaintextMFormR: make([]uint64, len(ringQ.Modulus)+len(ringQMul.Modulus)),
		Reciprocal96Q: reciprocal96(ringQ.Modulus), Reciprocal96QMul: reciprocal96(ringQMul.Modulus),
	}
	for i, modulus := range ringQ.Modulus {
		residue := new(big.Int).Mod(new(big.Int).Set(qMulProduct), new(big.Int).SetUint64(modulus)).Uint64()
		extra.QMulMFormQ[i] = mFormBig(residue, modulus)
		extra.PlaintextMFormR[i] = mFormBig(plaintext, modulus)
	}
	for i, modulus := range ringQMul.Modulus {
		extra.PlaintextMFormR[len(ringQ.Modulus)+i] = mFormBig(plaintext, modulus)
	}
	encoded, err := json.MarshalIndent(extra, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bfv_ring_constants.json"), append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestExportP1N16RingConstants(t *testing.T) {
	root := os.Getenv("FPGA_FIXTURE_ROOT")
	if root == "" {
		t.Skip("set FPGA_FIXTURE_ROOT to export P1 ring constants")
	}
	ckksParams, err := CKKSParameters("P1")
	if err != nil {
		t.Fatal(err)
	}
	exportRingConstants(t, filepath.Join(root, "mkf_ckks_p1_n16", "constants"), ckksParams.RingQ(), ckksParams.RingP())
	bfvParams, _, err := BFVParameters("P1")
	if err != nil {
		t.Fatal(err)
	}
	exportRingConstants(t, filepath.Join(root, "mkf_bfv_p1_n16", "constants"), bfvParams.RingQ(), bfvParams.RingP())
	ringQMul, err := ring.NewRing(bfvParams.N(), bfvP1QMul)
	if err != nil {
		t.Fatal(err)
	}
	exportBFVExtraConstants(t, filepath.Join(root, "mkf_bfv_p1_n16", "constants"), bfvParams.RingQ(), ringQMul, bfvParams.T())
}
