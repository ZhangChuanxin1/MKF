package mkfcpu

// This file defines the sidecar-free switching-key payload consumed by the
// FPGA host path. The raw payload intentionally contains no dimensions or
// version tag: those are frozen in a separately hashed contract manifest.
// Keeping the framing outside the payload lets the host page one fixed-size
// key at a time without copying or decoding a variable-length container.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/ldsec/lattigo/v2/rlwe"
)

// fpgaRawSwitchingKeyShape is the complete interpretation required for one
// raw rlwe.SwitchingKey payload. QModuli and PModuli are ordered; their values
// are part of the contract even though the raw file stores only coefficients.
type fpgaRawSwitchingKeyShape struct {
	PolynomialDegree   int
	QModuli            []uint64
	PModuli            []uint64
	DecompositionCount int
	OutputPolynomials  int
}

type fpgaAggregateKeyRecord struct {
	Kind          string
	Party         int
	GaloisElement uint64
	Key           *rlwe.SwitchingKey
}

// orderedFPGAAggregateKeys maps the canonical compact construction's public
// material into the only allowed host order: party fold keys, one collective
// degree-2 relinearization key, then collective Galois keys in the externally
// frozen inventory order. Map iteration is never used for Galois keys.
func orderedFPGAAggregateKeys(foldKeys []*rlwe.SwitchingKey, evaluationKey rlwe.EvaluationKey, orderedGaloisElements []uint64) ([]fpgaAggregateKeyRecord, error) {
	if len(foldKeys) < 2 {
		return nil, fmt.Errorf("need at least two distributed public fold keys")
	}
	if evaluationKey.Rlk == nil || len(evaluationKey.Rlk.Keys) != 1 || evaluationKey.Rlk.Keys[0] == nil {
		return nil, fmt.Errorf("need exactly one collective degree-2 relinearization switching key")
	}
	if evaluationKey.Rtks == nil || len(evaluationKey.Rtks.Keys) != len(orderedGaloisElements) {
		rotationCount := 0
		if evaluationKey.Rtks != nil {
			rotationCount = len(evaluationKey.Rtks.Keys)
		}
		return nil, fmt.Errorf("collective rotation-key count=%d, want %d", rotationCount, len(orderedGaloisElements))
	}
	records := make([]fpgaAggregateKeyRecord, 0, len(foldKeys)+1+len(orderedGaloisElements))
	for party, key := range foldKeys {
		if key == nil {
			return nil, fmt.Errorf("nil fold key for party %d", party)
		}
		records = append(records, fpgaAggregateKeyRecord{Kind: "distributed_public_fold", Party: party, Key: key})
	}
	records = append(records, fpgaAggregateKeyRecord{Kind: "collective_relinearization", Party: -1, Key: evaluationKey.Rlk.Keys[0]})
	seen := make(map[uint64]struct{}, len(orderedGaloisElements))
	for index, galoisElement := range orderedGaloisElements {
		if _, duplicate := seen[galoisElement]; duplicate {
			return nil, fmt.Errorf("duplicate Galois element %d at ordered index %d", galoisElement, index)
		}
		seen[galoisElement] = struct{}{}
		key, exists := evaluationKey.Rtks.Keys[galoisElement]
		if !exists || key == nil {
			return nil, fmt.Errorf("missing collective Galois key for element %d at ordered index %d", galoisElement, index)
		}
		records = append(records, fpgaAggregateKeyRecord{Kind: "collective_galois", Party: -1, GaloisElement: galoisElement, Key: key})
	}
	return records, nil
}

func (shape fpgaRawSwitchingKeyShape) expectedBytes() (uint64, error) {
	if shape.PolynomialDegree <= 0 || len(shape.QModuli) == 0 || len(shape.PModuli) == 0 || shape.DecompositionCount <= 0 || shape.OutputPolynomials != 2 {
		return 0, fmt.Errorf("invalid switching-key shape N/Q/P/beta/outputs=%d/%d/%d/%d/%d", shape.PolynomialDegree, len(shape.QModuli), len(shape.PModuli), shape.DecompositionCount, shape.OutputPolynomials)
	}
	factors := []uint64{uint64(shape.PolynomialDegree), uint64(len(shape.QModuli) + len(shape.PModuli)), uint64(shape.DecompositionCount), uint64(shape.OutputPolynomials), 8}
	result := uint64(1)
	for _, factor := range factors {
		if factor != 0 && result > math.MaxUint64/factor {
			return 0, fmt.Errorf("switching-key byte count overflows uint64")
		}
		result *= factor
	}
	return result, nil
}

func validateFPGARawSwitchingKeyShape(key *rlwe.SwitchingKey, shape fpgaRawSwitchingKeyShape) error {
	if key == nil {
		return fmt.Errorf("nil switching key")
	}
	if _, err := shape.expectedBytes(); err != nil {
		return err
	}
	if len(key.Value) != shape.DecompositionCount {
		return fmt.Errorf("decomposition count=%d, want %d", len(key.Value), shape.DecompositionCount)
	}
	for decomposition := range key.Value {
		for output := 0; output < shape.OutputPolynomials; output++ {
			poly := key.Value[decomposition][output]
			if poly.Q == nil || poly.P == nil {
				return fmt.Errorf("nil PolyQP at decomposition=%d output=%d", decomposition, output)
			}
			for basis, coefficients := range []struct {
				name string
				rows [][]uint64
				want int
			}{{"Q", poly.Q.Coeffs, len(shape.QModuli)}, {"P", poly.P.Coeffs, len(shape.PModuli)}} {
				_ = basis
				if len(coefficients.rows) != coefficients.want {
					return fmt.Errorf("%s row count at decomposition=%d output=%d is %d, want %d", coefficients.name, decomposition, output, len(coefficients.rows), coefficients.want)
				}
				for modulus, row := range coefficients.rows {
					if len(row) != shape.PolynomialDegree {
						return fmt.Errorf("%s[%d] coefficient count at decomposition=%d output=%d is %d, want %d", coefficients.name, modulus, decomposition, output, len(row), shape.PolynomialDegree)
					}
				}
			}
		}
	}
	return nil
}

func writeFPGARawUint64Rows(writer *bufio.Writer, rows [][]uint64) error {
	var encoded [8]byte
	for _, row := range rows {
		for _, coefficient := range row {
			binary.LittleEndian.PutUint64(encoded[:], coefficient)
			if _, err := writer.Write(encoded[:]); err != nil {
				return err
			}
		}
	}
	return nil
}

// writeFPGARawSwitchingKey emits exactly:
// decomposition -> output polynomial [0,1] -> Q rows -> P rows ->
// modulus-major -> coefficient-major -> little-endian uint64.
func writeFPGARawSwitchingKey(dst io.Writer, key *rlwe.SwitchingKey, shape fpgaRawSwitchingKeyShape) error {
	if err := validateFPGARawSwitchingKeyShape(key, shape); err != nil {
		return err
	}
	writer := bufio.NewWriterSize(dst, 4<<20)
	for decomposition := range key.Value {
		for output := 0; output < shape.OutputPolynomials; output++ {
			poly := key.Value[decomposition][output]
			if err := writeFPGARawUint64Rows(writer, poly.Q.Coeffs); err != nil {
				return err
			}
			if err := writeFPGARawUint64Rows(writer, poly.P.Coeffs); err != nil {
				return err
			}
		}
	}
	return writer.Flush()
}

func readFPGARawUint64Rows(reader *bufio.Reader, rows [][]uint64) error {
	var encoded [8]byte
	for _, row := range rows {
		for coefficient := range row {
			if _, err := io.ReadFull(reader, encoded[:]); err != nil {
				return err
			}
			row[coefficient] = binary.LittleEndian.Uint64(encoded[:])
		}
	}
	return nil
}

// readFPGARawSwitchingKeyExact decodes one complete payload into an already
// allocated key and rejects both truncation and trailing bytes. NTT/Montgomery
// representation is a manifest invariant and is not encoded in the payload.
func readFPGARawSwitchingKeyExact(src io.Reader, key *rlwe.SwitchingKey, shape fpgaRawSwitchingKeyShape) error {
	if err := validateFPGARawSwitchingKeyShape(key, shape); err != nil {
		return err
	}
	reader := bufio.NewReaderSize(src, 4<<20)
	for decomposition := range key.Value {
		for output := 0; output < shape.OutputPolynomials; output++ {
			poly := key.Value[decomposition][output]
			if err := readFPGARawUint64Rows(reader, poly.Q.Coeffs); err != nil {
				return fmt.Errorf("read Q at decomposition=%d output=%d: %w", decomposition, output, err)
			}
			if err := readFPGARawUint64Rows(reader, poly.P.Coeffs); err != nil {
				return fmt.Errorf("read P at decomposition=%d output=%d: %w", decomposition, output, err)
			}
			poly.Q.IsNTT = true
			poly.P.IsNTT = true
		}
	}
	if _, err := reader.ReadByte(); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing bytes after switching-key payload")
		}
		return fmt.Errorf("check switching-key payload boundary: %w", err)
	}
	return nil
}
