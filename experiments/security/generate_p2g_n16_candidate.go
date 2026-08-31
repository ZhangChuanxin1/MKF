// Command generate_p2g_n16_candidate deterministically freezes the RNS
// literals for the guarded compact ResNet-20 parameter candidate.
//
// It intentionally lives outside mkf_cpu/params.go.  The command uses the
// exact prime search in Lattigo v2.3.0 and emits only NTT primes p satisfying
// p = 1 (mod 2N) for N=65536.  The first 13 54-bit primes are the persistent
// post-conversion Q levels; the fourteenth is the one-shot ingress guard.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/ldsec/lattigo/v2/ring"
)

const (
	logN       = 16
	logSlots   = 14
	defaultLog = 54
	primeRoot  = 2 << logN
)

type candidate struct {
	Schema        string   `json:"schema"`
	Generator     string   `json:"generator"`
	LogN          int      `json:"log_n"`
	LogSlots      int      `json:"log_slots"`
	Scale         uint64   `json:"scale"`
	Q             []uint64 `json:"q_post_conversion"`
	Guard         uint64   `json:"q_guard"`
	P             []uint64 `json:"p"`
	NthRoot       int      `json:"ntt_nth_root"`
	Log2QPGuarded string   `json:"log2_q_guard_p_high_precision"`
	LiteralSHA256 string   `json:"literal_sha256"`
}

func literalBytes(q []uint64, guard uint64, p []uint64) []byte {
	payload := fmt.Sprintf("LogN=%d;LogSlots=%d;Scale=%d;Q=%#x;Guard=%#x;P=%#x;NthRoot=%d", logN, logSlots, uint64(1)<<defaultLog, q, guard, p, primeRoot)
	return []byte(payload)
}

func exactLog2(values ...[]uint64) string {
	product := big.NewInt(1)
	for _, group := range values {
		for _, value := range group {
			product.Mul(product, new(big.Int).SetUint64(value))
		}
	}
	precision := uint(512)
	f := new(big.Float).SetPrec(precision).SetInt(product)
	// MantExp gives f = mantissa * 2^exponent with 0.5 <= mantissa < 1.
	mantissa := new(big.Float).SetPrec(precision)
	exponent := f.MantExp(mantissa)
	// big.Float has no logarithm.  The emitted decimal is filled by the
	// independent audit script; retain the exact product bit length here.
	return fmt.Sprintf("bit_length=%d;mantissa=%s;exponent=%d", product.BitLen(), mantissa.Text('g', 80), exponent)
}

func main() {
	q60 := ring.GenerateNTTPrimesP(60, primeRoot, 1)
	q54 := ring.GenerateNTTPrimesP(54, primeRoot, 14)
	p59 := ring.GenerateNTTPrimesP(59, primeRoot, 2)
	q := append(append([]uint64(nil), q60...), q54[:13]...)
	guard := q54[13]
	for _, value := range append(append(append([]uint64(nil), q...), guard), p59...) {
		if (value-1)%uint64(primeRoot) != 0 || !ring.IsPrime(value) {
			panic(fmt.Sprintf("invalid 2N-friendly prime %#x", value))
		}
	}
	digest := sha256.Sum256(literalBytes(q, guard, p59))
	out := candidate{
		Schema:        "mkf-p2g-n16-candidate-v1",
		Generator:     "github.com/ldsec/lattigo/v2/ring.GenerateNTTPrimesP@v2.3.0",
		LogN:          logN,
		LogSlots:      logSlots,
		Scale:         uint64(1) << defaultLog,
		Q:             q,
		Guard:         guard,
		P:             p59,
		NthRoot:       primeRoot,
		Log2QPGuarded: exactLog2(q, []uint64{guard}, p59),
		LiteralSHA256: hex.EncodeToString(digest[:]),
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(out); err != nil {
		panic(err)
	}
}
