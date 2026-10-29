package mkfcpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

const kklssStagedCiphertextMagic = "KKSCT001"

func kklssStagingParameterFingerprint(params mkckks.Parameters) [sha256.Size]byte {
	var identity bytes.Buffer
	_ = binary.Write(&identity, binary.LittleEndian, uint32(params.LogN()))
	_ = binary.Write(&identity, binary.LittleEndian, uint32(params.LogSlots()))
	_ = binary.Write(&identity, binary.LittleEndian, math.Float64bits(params.Scale()))
	_ = binary.Write(&identity, binary.LittleEndian, uint32(len(params.Q())))
	for _, modulus := range params.Q() {
		_ = binary.Write(&identity, binary.LittleEndian, modulus)
	}
	_ = binary.Write(&identity, binary.LittleEndian, uint32(len(params.P())))
	for _, modulus := range params.P() {
		_ = binary.Write(&identity, binary.LittleEndian, modulus)
	}
	return sha256.Sum256(identity.Bytes())
}

func marshalKKLSSStagedCiphertext(params mkckks.Parameters, ciphertext *mkckks.Ciphertext) ([]byte, error) {
	if ciphertext == nil || ciphertext.Value["0"] == nil {
		return nil, fmt.Errorf("cannot stage a nil KKLSS ciphertext")
	}
	ids := make([]string, 0, len(ciphertext.IDSet().Value))
	for id := range ciphertext.IDSet().Value {
		if id == "" || id == "0" || len(id) > 128 {
			return nil, fmt.Errorf("invalid KKLSS ciphertext owner ID %q", id)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 || len(ids) > fullResNet20Parties {
		return nil, fmt.Errorf("staged KKLSS ciphertext has %d owner IDs", len(ids))
	}
	sort.Strings(ids)
	level := ciphertext.Level()
	if level < 0 || level > params.MaxLevel() {
		return nil, fmt.Errorf("staged KKLSS ciphertext level=%d is outside [0,%d]", level, params.MaxLevel())
	}
	var body bytes.Buffer
	body.WriteString(kklssStagedCiphertextMagic)
	parameterFingerprint := kklssStagingParameterFingerprint(params)
	body.Write(parameterFingerprint[:])
	for _, value := range []uint32{uint32(params.LogN()), uint32(level), uint32(len(ids))} {
		if err := binary.Write(&body, binary.LittleEndian, value); err != nil {
			return nil, err
		}
	}
	if err := binary.Write(&body, binary.LittleEndian, math.Float64bits(ciphertext.Scale)); err != nil {
		return nil, err
	}
	for _, id := range ids {
		if err := binary.Write(&body, binary.LittleEndian, uint16(len(id))); err != nil {
			return nil, err
		}
		body.WriteString(id)
	}
	componentNames := append([]string{"0"}, ids...)
	for _, name := range componentNames {
		polynomial := ciphertext.Value[name]
		if polynomial == nil || len(polynomial.Coeffs) != level+1 {
			return nil, fmt.Errorf("KKLSS component %q has invalid level allocation", name)
		}
		if polynomial.IsNTT {
			body.WriteByte(1)
		} else {
			body.WriteByte(0)
		}
		for modulusIndex := 0; modulusIndex <= level; modulusIndex++ {
			if len(polynomial.Coeffs[modulusIndex]) != params.N() {
				return nil, fmt.Errorf("KKLSS component %q modulus %d has %d coefficients, want %d", name, modulusIndex, len(polynomial.Coeffs[modulusIndex]), params.N())
			}
			for _, coefficient := range polynomial.Coeffs[modulusIndex] {
				if err := binary.Write(&body, binary.LittleEndian, coefficient); err != nil {
					return nil, err
				}
			}
		}
	}
	digest := sha256.Sum256(body.Bytes())
	return append(body.Bytes(), digest[:]...), nil
}

func unmarshalKKLSSStagedCiphertext(params mkckks.Parameters, data []byte) (*mkckks.Ciphertext, error) {
	if len(data) < len(kklssStagedCiphertextMagic)+sha256.Size+3*4+8+sha256.Size {
		return nil, fmt.Errorf("staged KKLSS ciphertext is truncated")
	}
	body, checksum := data[:len(data)-sha256.Size], data[len(data)-sha256.Size:]
	digest := sha256.Sum256(body)
	if !bytes.Equal(checksum, digest[:]) {
		return nil, fmt.Errorf("staged KKLSS ciphertext SHA-256 mismatch")
	}
	reader := bytes.NewReader(body)
	magic := make([]byte, len(kklssStagedCiphertextMagic))
	if _, err := reader.Read(magic); err != nil || string(magic) != kklssStagedCiphertextMagic {
		return nil, fmt.Errorf("invalid staged KKLSS ciphertext magic")
	}
	encodedFingerprint := make([]byte, sha256.Size)
	expectedFingerprint := kklssStagingParameterFingerprint(params)
	if _, err := reader.Read(encodedFingerprint); err != nil || !bytes.Equal(encodedFingerprint, expectedFingerprint[:]) {
		return nil, fmt.Errorf("staged KKLSS parameter fingerprint mismatch")
	}
	var logN, encodedLevel, idCount uint32
	for _, destination := range []*uint32{&logN, &encodedLevel, &idCount} {
		if err := binary.Read(reader, binary.LittleEndian, destination); err != nil {
			return nil, fmt.Errorf("read staged KKLSS header: %w", err)
		}
	}
	var scaleBits uint64
	if err := binary.Read(reader, binary.LittleEndian, &scaleBits); err != nil {
		return nil, fmt.Errorf("read staged KKLSS scale: %w", err)
	}
	if int(logN) != params.LogN() || int(encodedLevel) > params.MaxLevel() || idCount == 0 || int(idCount) > fullResNet20Parties {
		return nil, fmt.Errorf("staged KKLSS parameter identity mismatch: LogN=%d level=%d ids=%d", logN, encodedLevel, idCount)
	}
	ids := make([]string, int(idCount))
	idset := mkrlwe.NewIDSet()
	for index := range ids {
		var length uint16
		if err := binary.Read(reader, binary.LittleEndian, &length); err != nil || length == 0 || length > 128 {
			return nil, fmt.Errorf("invalid staged KKLSS owner ID length")
		}
		encoded := make([]byte, int(length))
		if _, err := reader.Read(encoded); err != nil {
			return nil, fmt.Errorf("read staged KKLSS owner ID: %w", err)
		}
		ids[index] = string(encoded)
		if ids[index] == "0" || idset.Has(ids[index]) {
			return nil, fmt.Errorf("invalid or duplicate staged KKLSS owner ID %q", ids[index])
		}
		idset.Add(ids[index])
	}
	level := int(encodedLevel)
	ciphertext := mkckks.NewCiphertext(params, idset, level, math.Float64frombits(scaleBits))
	componentNames := append([]string{"0"}, ids...)
	for _, name := range componentNames {
		isNTT, err := reader.ReadByte()
		if err != nil || isNTT > 1 {
			return nil, fmt.Errorf("invalid staged KKLSS representation flag for %q", name)
		}
		ciphertext.Value[name].IsNTT = isNTT == 1
		for modulusIndex := 0; modulusIndex <= level; modulusIndex++ {
			for coefficientIndex := 0; coefficientIndex < params.N(); coefficientIndex++ {
				if err := binary.Read(reader, binary.LittleEndian, &ciphertext.Value[name].Coeffs[modulusIndex][coefficientIndex]); err != nil {
					return nil, fmt.Errorf("read staged KKLSS component %q: %w", name, err)
				}
			}
		}
	}
	if reader.Len() != 0 {
		return nil, fmt.Errorf("staged KKLSS ciphertext has %d trailing bytes", reader.Len())
	}
	return ciphertext, nil
}

func writeKKLSSStagedCiphertext(path string, params mkckks.Parameters, ciphertext *mkckks.Ciphertext) error {
	encoded, err := marshalKKLSSStagedCiphertext(params, ciphertext)
	if err != nil {
		return err
	}
	return os.WriteFile(path, encoded, 0o600)
}

func readKKLSSStagedCiphertext(path string, params mkckks.Parameters) (*mkckks.Ciphertext, error) {
	encoded, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return unmarshalKKLSSStagedCiphertext(params, encoded)
}

func TestKKLSSStagedCiphertextRoundTripAndChecksum(t *testing.T) {
	params, err := kklssP2Parameters()
	if err != nil {
		t.Fatal(err)
	}
	idset := mkrlwe.NewIDSet()
	idset.Add("model-owner")
	input := mkckks.NewCiphertext(params, idset, 0, params.Scale())
	for componentIndex, name := range []string{"0", "model-owner"} {
		input.Value[name].IsNTT = componentIndex == 1
		for coefficientIndex := 0; coefficientIndex < params.N(); coefficientIndex++ {
			input.Value[name].Coeffs[0][coefficientIndex] = uint64((componentIndex+1)*(coefficientIndex+3)) % params.Q()[0]
		}
	}
	encoded, err := marshalKKLSSStagedCiphertext(params, input)
	if err != nil {
		t.Fatal(err)
	}
	output, err := unmarshalKKLSSStagedCiphertext(params, encoded)
	if err != nil {
		t.Fatal(err)
	}
	if output.Scale != input.Scale || output.Level() != input.Level() || !output.IDSet().Has("model-owner") || len(output.IDSet().Value) != 1 {
		t.Fatalf("staged KKLSS identity changed after round trip")
	}
	for _, name := range []string{"0", "model-owner"} {
		if output.Value[name].IsNTT != input.Value[name].IsNTT {
			t.Fatalf("component %s representation flag changed", name)
		}
		for coefficientIndex := range input.Value[name].Coeffs[0] {
			if output.Value[name].Coeffs[0][coefficientIndex] != input.Value[name].Coeffs[0][coefficientIndex] {
				t.Fatalf("component %s coefficient %d changed", name, coefficientIndex)
			}
		}
	}
	encoded[len(encoded)/2] ^= 1
	if _, err := unmarshalKKLSSStagedCiphertext(params, encoded); err == nil {
		t.Fatal("corrupted staged KKLSS ciphertext passed checksum validation")
	}
}
