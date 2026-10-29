package mkfcpu

import (
	"fmt"

	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

type multiCiphertext struct {
	C0         *ring.Poly
	Components []*ring.Poly
	Level      int
	Scale      float64
}

// outputCiphertext stores the compact (c0, Gamma) result as well as the n
// repeated Gamma components required by the paper's n+1-component interface.
type outputCiphertext struct {
	C0         *ring.Poly
	Gamma      *ring.Poly
	Components []*ring.Poly
	Level      int
	Scale      float64
}

type onlineKeys struct {
	Fold  []*rlwe.SwitchingKey
	Relin *rlwe.RelinearizationKey
}

func aggregateSecret(params rlwe.Parameters, keys []*rlwe.SecretKey) *rlwe.SecretKey {
	levelQ, levelP := params.QCount()-1, params.PCount()-1
	out := rlwe.NewSecretKey(params)
	out.Value.Q.Zero()
	out.Value.P.Zero()
	ringQP := params.RingQP()
	for _, sk := range keys {
		ringQP.AddLvl(levelQ, levelP, out.Value, sk.Value, out.Value)
	}
	return out
}

func generateOnlineKeys(params rlwe.Parameters, kgen rlwe.KeyGenerator, src []*rlwe.SecretKey, aggregate *rlwe.SecretKey) onlineKeys {
	keys := onlineKeys{Fold: make([]*rlwe.SwitchingKey, len(src))}
	for i := range src {
		keys.Fold[i] = kgen.GenSwitchingKey(src[i], aggregate)
	}
	keys.Relin = kgen.GenRelinearizationKey(aggregate, 1)
	return keys
}

type switchWorkspace struct {
	ks     *rlwe.KeySwitcher
	decomp []rlwe.PolyQP
	c0Q    *ring.Poly
	c1Q    *ring.Poly
	c0P    *ring.Poly
	c1P    *ring.Poly
}

func newSwitchWorkspace(params rlwe.Parameters) *switchWorkspace {
	decomp := make([]rlwe.PolyQP, params.Beta())
	for i := range decomp {
		decomp[i] = params.RingQP().NewPoly()
	}
	return &switchWorkspace{
		ks:     rlwe.NewKeySwitcher(params),
		decomp: decomp,
		c0Q:    params.RingQ().NewPoly(),
		c1Q:    params.RingQ().NewPoly(),
		c0P:    params.RingP().NewPoly(),
		c1P:    params.RingP().NewPoly(),
	}
}

// switchOne performs one hoisted QP key switch. One decomposition produces
// two evaluation-key polynomial outputs, matching d_i and k_i in the paper.
func (w *switchWorkspace) switchOne(params rlwe.Parameters, levelQ int, in *ring.Poly, swk *rlwe.SwitchingKey, nttOutput bool, prof *Profile) (*ring.Poly, *ring.Poly) {
	levelP := params.PCount() - 1
	measure(&prof.Decomposition, func() {
		w.ks.DecomposeNTT(levelQ, levelP, params.Alpha(), in, w.decomp)
	})
	prof.Decompositions++

	measure(&prof.External, func() {
		w.ks.KeyswitchHoistedNoModDown(levelQ, w.decomp, swk, w.c0Q, w.c1Q, w.c0P, w.c1P)
	})
	prof.ExternalProductPolys += 2

	measure(&prof.Modulus, func() {
		if nttOutput {
			w.ks.Baseconverter.ModDownQPtoQNTT(levelQ, levelP, w.c0Q, w.c0P, w.c0Q)
			w.ks.Baseconverter.ModDownQPtoQNTT(levelQ, levelP, w.c1Q, w.c1P, w.c1Q)
			w.c0Q.IsNTT, w.c1Q.IsNTT = true, true
		} else {
			params.RingQ().InvNTTLazyLvl(levelQ, w.c0Q, w.c0Q)
			params.RingQ().InvNTTLazyLvl(levelQ, w.c1Q, w.c1Q)
			params.RingP().InvNTTLazyLvl(levelP, w.c0P, w.c0P)
			params.RingP().InvNTTLazyLvl(levelP, w.c1P, w.c1P)
			w.ks.Baseconverter.ModDownQPtoQ(levelQ, levelP, w.c0Q, w.c0P, w.c0Q)
			w.ks.Baseconverter.ModDownQPtoQ(levelQ, levelP, w.c1Q, w.c1P, w.c1Q)
			w.c0Q.IsNTT, w.c1Q.IsNTT = false, false
		}
	})
	return w.c0Q, w.c1Q
}

func fold(params rlwe.Parameters, in *multiCiphertext, foldKeys []*rlwe.SwitchingKey, nttOutput bool, w *switchWorkspace, prof *Profile) ([2]*ring.Poly, error) {
	if len(in.Components) != len(foldKeys) {
		return [2]*ring.Poly{}, fmt.Errorf("component/key count mismatch: %d != %d", len(in.Components), len(foldKeys))
	}
	ringQ := params.RingQ()
	out := [2]*ring.Poly{ringQ.NewPoly(), ringQ.NewPoly()}
	out[0].CopyValues(in.C0)
	out[0].IsNTT = nttOutput
	out[1].IsNTT = nttOutput
	for i, component := range in.Components {
		p0, p1 := w.switchOne(params, in.Level, component, foldKeys[i], nttOutput, prof)
		measure(&prof.External, func() {
			ringQ.AddLvl(in.Level, out[0], p0, out[0])
			ringQ.AddLvl(in.Level, out[1], p1, out[1])
		})
	}
	return out, nil
}

func materialize(params rlwe.Parameters, c0, gamma *ring.Poly, parties, level int, scale float64, prof *Profile) *outputCiphertext {
	out := &outputCiphertext{C0: c0, Gamma: gamma, Level: level, Scale: scale}
	measure(&prof.Memory, func() {
		out.Components = make([]*ring.Poly, parties)
		for i := range out.Components {
			out.Components[i] = params.RingQ().NewPolyLvl(level)
			out.Components[i].CopyValues(gamma)
			out.Components[i].IsNTT = gamma.IsNTT
		}
	})
	return out
}
