package mkfcpu

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// The names and scales below are the exact Layer2/Block1 schedule in
// LowMemoryFHEResNet20 src/main.cpp lines 334--375. The upstream repository
// calls the first 16->32 residual block "layer4" in the frozen filenames.
const (
	realStage2InputChannels  = 16
	realStage2OutputChannels = 32
	realStage2InputWidth     = 32
	realStage2OutputWidth    = 16
	realStage2InputBlock     = realStage2InputWidth * realStage2InputWidth
	realStage2OutputBlock    = realStage2OutputWidth * realStage2OutputWidth
	realStage2LogicalSlots   = realStage2OutputChannels * realStage2OutputBlock
	realStage2MainScale      = 0.57
	realStage2SkipScale      = 0.40
	realStage2Conv2Scale     = 0.40
	stage2AuditTolerance     = 1e-15
)

type realStage2StrideWeights struct {
	// The two groups correspond to output channels 0--15 and 16--31.
	masks      [2][realStage2InputChannels][3][3][]complex128
	kernel     [realStage2OutputChannels][realStage2InputChannels][3][3]float64
	packedBias [2][]complex128
	bias       [realStage2OutputChannels]float64
}

type realStage2SkipWeights struct {
	masks      [2][realStage2InputChannels][]complex128
	kernel     [realStage2OutputChannels][realStage2InputChannels]float64
	packedBias [2][]complex128
	bias       [realStage2OutputChannels]float64
}

type realStage2ConvWeights struct {
	masks      [realStage2OutputChannels][3][3][]complex128
	kernel     [realStage2OutputChannels][realStage2OutputChannels][3][3]float64
	packedBias []complex128
	bias       [realStage2OutputChannels]float64
}

func stage2FrozenFile(weightsDirectory, name string) string {
	return filepath.Join(weightsDirectory, name)
}

func stage2ExpectLength(path string, values []float64, want int) error {
	if len(values) != want {
		return fmt.Errorf("%s: got %d values, want %d", path, len(values), want)
	}
	return nil
}

func stage2SparsePositionIsActive(y, x int) bool {
	return y%2 == 0 && x%2 == 0
}

// loadRealStage2StrideWeights audits the exact masks emitted by notebook
// "Algorithm 2 - Exporting Weights", cells 20--23. In particular it checks
// altalena's even-row/even-column stride mask, the 16 channel diagonals, both
// output-channel halves, and every 3x3 padding location before unrolling the
// frozen diagonal masks into the vectors consumed by the Go evaluator.
func loadRealStage2StrideWeights(weightsDirectory string, scale float64, slots int) (*realStage2StrideWeights, error) {
	if slots != realResNetStemActiveSlots {
		return nil, fmt.Errorf("stage2 P2 transition requires exactly %d slots, got %d", realResNetStemActiveSlots, slots)
	}
	loaded := new(realStage2StrideWeights)
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			fileChannel := group*realStage2InputChannels + diagonal
			for kernelY := 0; kernelY < 3; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					kernelIndex := kernelY*3 + kernelX + 1
					path := stage2FrozenFile(weightsDirectory, fmt.Sprintf("layer4-conv1bn1-ch%d-k%d.bin", fileChannel, kernelIndex))
					values, err := readFrozenFloatVector(path)
					if err != nil {
						return nil, err
					}
					if err := stage2ExpectLength(path, values, realResNetStemActiveSlots); err != nil {
						return nil, err
					}
					loaded.masks[group][diagonal][kernelY][kernelX] = make([]complex128, slots)
					deltaY, deltaX := kernelY-1, kernelX-1
					for localOutput := 0; localOutput < realStage2InputChannels; localOutput++ {
						sourceBlock := (localOutput + diagonal) % realStage2InputChannels
						sourceBase := sourceBlock * realStage2InputBlock
						coefficient := values[sourceBase+16*realStage2InputWidth+16]
						globalOutput := group*realStage2InputChannels + localOutput
						loaded.kernel[globalOutput][sourceBlock][kernelY][kernelX] = scale * coefficient
						destinationBase := localOutput * realStage2InputBlock
						for y := 0; y < realStage2InputWidth; y++ {
							for x := 0; x < realStage2InputWidth; x++ {
								position := y*realStage2InputWidth + x
								got := values[sourceBase+position]
								validPadding := y+deltaY >= 0 && y+deltaY < realStage2InputWidth && x+deltaX >= 0 && x+deltaX < realStage2InputWidth
								want := 0.0
								if stage2SparsePositionIsActive(y, x) && validPadding {
									want = coefficient
								}
								if math.Abs(got-want) > stage2AuditTolerance {
									return nil, fmt.Errorf("%s: invalid stride/padding mask at group=%d output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, group, globalOutput, sourceBlock, y, x, got, want)
								}
								loaded.masks[group][diagonal][kernelY][kernelX][destinationBase+position] = complex(scale*got, 0)
							}
						}
					}
				}
			}
		}

		biasPath := stage2FrozenFile(weightsDirectory, fmt.Sprintf("layer4-conv1bn1-bias%d.bin", group+1))
		biasValues, err := readFrozenFloatVector(biasPath)
		if err != nil {
			return nil, err
		}
		if err := stage2ExpectLength(biasPath, biasValues, realResNetStemActiveSlots); err != nil {
			return nil, err
		}
		loaded.packedBias[group] = make([]complex128, slots)
		for localOutput := 0; localOutput < realStage2InputChannels; localOutput++ {
			base := localOutput * realStage2InputBlock
			rawBias := biasValues[base]
			loaded.bias[group*realStage2InputChannels+localOutput] = scale * rawBias
			for y := 0; y < realStage2InputWidth; y++ {
				for x := 0; x < realStage2InputWidth; x++ {
					position := y*realStage2InputWidth + x
					want := 0.0
					if stage2SparsePositionIsActive(y, x) {
						want = rawBias
					}
					if math.Abs(biasValues[base+position]-want) > stage2AuditTolerance {
						return nil, fmt.Errorf("%s: invalid sparse bias at output=%d y=%d x=%d: got %.17g want %.17g", biasPath, group*realStage2InputChannels+localOutput, y, x, biasValues[base+position], want)
					}
					loaded.packedBias[group][base+position] = complex(scale*biasValues[base+position], 0)
				}
			}
		}
	}
	return loaded, nil
}

// loadRealStage2SkipWeights audits the 1x1 stride-2 projection exported by
// notebook cell 25 under the exact layer4dx-conv1bn1 prefix.
func loadRealStage2SkipWeights(weightsDirectory string, scale float64, slots int) (*realStage2SkipWeights, error) {
	if slots != realResNetStemActiveSlots {
		return nil, fmt.Errorf("stage2 P2 transition requires exactly %d slots, got %d", realResNetStemActiveSlots, slots)
	}
	loaded := new(realStage2SkipWeights)
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			fileChannel := group*realStage2InputChannels + diagonal
			path := stage2FrozenFile(weightsDirectory, fmt.Sprintf("layer4dx-conv1bn1-ch%d-k1.bin", fileChannel))
			values, err := readFrozenFloatVector(path)
			if err != nil {
				return nil, err
			}
			if err := stage2ExpectLength(path, values, realResNetStemActiveSlots); err != nil {
				return nil, err
			}
			loaded.masks[group][diagonal] = make([]complex128, slots)
			for localOutput := 0; localOutput < realStage2InputChannels; localOutput++ {
				sourceBlock := (localOutput + diagonal) % realStage2InputChannels
				sourceBase := sourceBlock * realStage2InputBlock
				coefficient := values[sourceBase+16*realStage2InputWidth+16]
				globalOutput := group*realStage2InputChannels + localOutput
				loaded.kernel[globalOutput][sourceBlock] = scale * coefficient
				destinationBase := localOutput * realStage2InputBlock
				for y := 0; y < realStage2InputWidth; y++ {
					for x := 0; x < realStage2InputWidth; x++ {
						position := y*realStage2InputWidth + x
						got := values[sourceBase+position]
						want := 0.0
						if stage2SparsePositionIsActive(y, x) {
							want = coefficient
						}
						if math.Abs(got-want) > stage2AuditTolerance {
							return nil, fmt.Errorf("%s: invalid 1x1 stride mask at group=%d output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, group, globalOutput, sourceBlock, y, x, got, want)
						}
						loaded.masks[group][diagonal][destinationBase+position] = complex(scale*got, 0)
					}
				}
			}
		}

		biasPath := stage2FrozenFile(weightsDirectory, fmt.Sprintf("layer4dx-conv1bn1-bias%d.bin", group+1))
		biasValues, err := readFrozenFloatVector(biasPath)
		if err != nil {
			return nil, err
		}
		if err := stage2ExpectLength(biasPath, biasValues, realResNetStemActiveSlots); err != nil {
			return nil, err
		}
		loaded.packedBias[group] = make([]complex128, slots)
		for localOutput := 0; localOutput < realStage2InputChannels; localOutput++ {
			base := localOutput * realStage2InputBlock
			rawBias := biasValues[base]
			loaded.bias[group*realStage2InputChannels+localOutput] = scale * rawBias
			for y := 0; y < realStage2InputWidth; y++ {
				for x := 0; x < realStage2InputWidth; x++ {
					position := y*realStage2InputWidth + x
					want := 0.0
					if stage2SparsePositionIsActive(y, x) {
						want = rawBias
					}
					if math.Abs(biasValues[base+position]-want) > stage2AuditTolerance {
						return nil, fmt.Errorf("%s: invalid projection bias at output=%d y=%d x=%d: got %.17g want %.17g", biasPath, group*realStage2InputChannels+localOutput, y, x, biasValues[base+position], want)
					}
					loaded.packedBias[group][base+position] = complex(scale*biasValues[base+position], 0)
				}
			}
		}
	}
	return loaded, nil
}

// loadRealStage2ConvWeights audits the post-transition 32-channel 3x3
// convolution files used by FHEController::convbn2.
func loadRealStage2ConvWeights(weightsDirectory, prefix string, scale float64, slots int) (*realStage2ConvWeights, error) {
	if slots%realStage2LogicalSlots != 0 {
		return nil, fmt.Errorf("stage2 logical layout needs a multiple of %d slots, got %d", realStage2LogicalSlots, slots)
	}
	loaded := new(realStage2ConvWeights)
	packingCopies := slots / realStage2LogicalSlots
	for diagonal := 0; diagonal < realStage2OutputChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				kernelIndex := kernelY*3 + kernelX + 1
				path := stage2FrozenFile(weightsDirectory, fmt.Sprintf("%s-ch%d-k%d.bin", prefix, diagonal, kernelIndex))
				values, err := readFrozenFloatVector(path)
				if err != nil {
					return nil, err
				}
				if err := stage2ExpectLength(path, values, realStage2LogicalSlots); err != nil {
					return nil, err
				}
				loaded.masks[diagonal][kernelY][kernelX] = make([]complex128, slots)
				deltaY, deltaX := kernelY-1, kernelX-1
				for outputChannel := 0; outputChannel < realStage2OutputChannels; outputChannel++ {
					sourceBlock := (outputChannel + diagonal) % realStage2OutputChannels
					sourceBase := sourceBlock * realStage2OutputBlock
					coefficient := values[sourceBase+8*realStage2OutputWidth+8]
					loaded.kernel[outputChannel][sourceBlock][kernelY][kernelX] = scale * coefficient
					destinationBase := outputChannel * realStage2OutputBlock
					for y := 0; y < realStage2OutputWidth; y++ {
						for x := 0; x < realStage2OutputWidth; x++ {
							position := y*realStage2OutputWidth + x
							got := values[sourceBase+position]
							valid := y+deltaY >= 0 && y+deltaY < realStage2OutputWidth && x+deltaX >= 0 && x+deltaX < realStage2OutputWidth
							want := 0.0
							if valid {
								want = coefficient
							}
							if math.Abs(got-want) > stage2AuditTolerance {
								return nil, fmt.Errorf("%s: invalid 16x16 padding mask at output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, outputChannel, sourceBlock, y, x, got, want)
							}
							for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
								loaded.masks[diagonal][kernelY][kernelX][copyIndex*realStage2LogicalSlots+destinationBase+position] = complex(scale*got, 0)
							}
						}
					}
				}
			}
		}
	}

	biasPath := stage2FrozenFile(weightsDirectory, prefix+"-bias.bin")
	biasValues, err := readFrozenFloatVector(biasPath)
	if err != nil {
		return nil, err
	}
	if err := stage2ExpectLength(biasPath, biasValues, realStage2LogicalSlots); err != nil {
		return nil, err
	}
	loaded.packedBias = make([]complex128, slots)
	for outputChannel := 0; outputChannel < realStage2OutputChannels; outputChannel++ {
		base := outputChannel * realStage2OutputBlock
		rawBias := biasValues[base]
		loaded.bias[outputChannel] = scale * rawBias
		for position := 0; position < realStage2OutputBlock; position++ {
			if math.Abs(biasValues[base+position]-rawBias) > stage2AuditTolerance {
				return nil, fmt.Errorf("%s: bias block %d is not constant at position %d", biasPath, outputChannel, position)
			}
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				loaded.packedBias[copyIndex*realStage2LogicalSlots+base+position] = complex(scale*rawBias, 0)
			}
		}
	}
	return loaded, nil
}

func plainRealStage2StrideConv(input []complex128, weights *realStage2StrideWeights) []complex128 {
	logical := make([]complex128, realStage2LogicalSlots)
	for outputChannel := 0; outputChannel < realStage2OutputChannels; outputChannel++ {
		base := outputChannel * realStage2OutputBlock
		for y := 0; y < realStage2OutputWidth; y++ {
			for x := 0; x < realStage2OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage2InputChannels; inputChannel++ {
					inputBase := inputChannel * realStage2InputBlock
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY := 2*y + kernelY - 1
							sourceX := 2*x + kernelX - 1
							if sourceY >= 0 && sourceY < realStage2InputWidth && sourceX >= 0 && sourceX < realStage2InputWidth {
								sum += input[inputBase+sourceY*realStage2InputWidth+sourceX] * complex(weights.kernel[outputChannel][inputChannel][kernelY][kernelX], 0)
							}
						}
					}
				}
				logical[base+y*realStage2OutputWidth+x] = sum
			}
		}
	}
	return repeatStage2Logical(logical, len(input))
}

func plainRealStage2Skip(input []complex128, weights *realStage2SkipWeights) []complex128 {
	logical := make([]complex128, realStage2LogicalSlots)
	for outputChannel := 0; outputChannel < realStage2OutputChannels; outputChannel++ {
		base := outputChannel * realStage2OutputBlock
		for y := 0; y < realStage2OutputWidth; y++ {
			for x := 0; x < realStage2OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage2InputChannels; inputChannel++ {
					inputBase := inputChannel * realStage2InputBlock
					sum += input[inputBase+(2*y)*realStage2InputWidth+2*x] * complex(weights.kernel[outputChannel][inputChannel], 0)
				}
				logical[base+y*realStage2OutputWidth+x] = sum
			}
		}
	}
	return repeatStage2Logical(logical, len(input))
}

func plainRealStage2Conv(input []complex128, weights *realStage2ConvWeights) []complex128 {
	logical := make([]complex128, realStage2LogicalSlots)
	for outputChannel := 0; outputChannel < realStage2OutputChannels; outputChannel++ {
		base := outputChannel * realStage2OutputBlock
		for y := 0; y < realStage2OutputWidth; y++ {
			for x := 0; x < realStage2OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage2OutputChannels; inputChannel++ {
					inputBase := inputChannel * realStage2OutputBlock
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY := y + kernelY - 1
							sourceX := x + kernelX - 1
							if sourceY >= 0 && sourceY < realStage2OutputWidth && sourceX >= 0 && sourceX < realStage2OutputWidth {
								sum += input[inputBase+sourceY*realStage2OutputWidth+sourceX] * complex(weights.kernel[outputChannel][inputChannel][kernelY][kernelX], 0)
							}
						}
					}
				}
				logical[base+y*realStage2OutputWidth+x] = sum
			}
		}
	}
	return repeatStage2Logical(logical, len(input))
}

func repeatStage2Logical(logical []complex128, slots int) []complex128 {
	if len(logical) != realStage2LogicalSlots || slots%realStage2LogicalSlots != 0 {
		panic("invalid stage2 logical repetition")
	}
	output := make([]complex128, slots)
	for copyIndex := 0; copyIndex < slots/realStage2LogicalSlots; copyIndex++ {
		copy(output[copyIndex*realStage2LogicalSlots:], logical)
	}
	return output
}

func plainSparseStage2Half(packed []complex128, group int) []complex128 {
	output := make([]complex128, realResNetStemActiveSlots)
	for localChannel := 0; localChannel < realStage2InputChannels; localChannel++ {
		logicalBase := (group*realStage2InputChannels + localChannel) * realStage2OutputBlock
		sparseBase := localChannel * realStage2InputBlock
		for y := 0; y < realStage2OutputWidth; y++ {
			for x := 0; x < realStage2OutputWidth; x++ {
				output[sparseBase+(2*y)*realStage2InputWidth+2*x] = packed[logicalBase+y*realStage2OutputWidth+x]
			}
		}
	}
	return output
}

func stage2RotatePlainLeft(input []complex128, rotation int) []complex128 {
	output := make([]complex128, len(input))
	rotation %= len(input)
	if rotation < 0 {
		rotation += len(input)
	}
	for destination := range output {
		output[destination] = input[(destination+rotation)%len(input)]
	}
	return output
}

func stage2AddPlain(left, right []complex128) []complex128 {
	output := make([]complex128, len(left))
	for i := range output {
		output[i] = left[i] + right[i]
	}
	return output
}

func stage2ApplyPlainMask(input, mask []complex128) []complex128 {
	output := make([]complex128, len(input))
	for i := range output {
		output[i] = input[i] * mask[i]
	}
	return output
}

func stage2AlternatingMask(slots, keep int) []complex128 {
	mask := make([]complex128, slots)
	for i := range mask {
		if i%(2*keep) < keep {
			mask[i] = 1
		}
	}
	return mask
}

func stage2RowMask(slots, row int) []complex128 {
	mask := make([]complex128, slots)
	for channel := 0; channel < realStage2InputChannels; channel++ {
		base := channel*realStage2InputBlock + row*realStage2OutputWidth
		for x := 0; x < realStage2OutputWidth; x++ {
			mask[base+x] = 1
		}
	}
	return mask
}

func stage2ChannelMask(slots, channel int) []complex128 {
	mask := make([]complex128, slots)
	base := channel * realStage2InputBlock
	for i := 0; i < realStage2OutputBlock; i++ {
		mask[base+i] = 1
	}
	return mask
}

// plainCompactStage2Half is a direct scalar simulation of
// FHEController::downsample1024to256 lines 1189--1220, specialized to one of
// the two 16-channel halves. It is used only to audit the encrypted packing
// circuit against the independent standard-convolution oracle.
func plainCompactStage2Half(input []complex128) []complex128 {
	current := append([]complex128(nil), input...)
	for _, step := range []struct {
		rotation int
		keep     int
	}{{1, 2}, {2, 4}, {4, 8}} {
		current = stage2ApplyPlainMask(stage2AddPlain(current, stage2RotatePlainLeft(current, step.rotation)), stage2AlternatingMask(len(current), step.keep))
	}
	current = stage2AddPlain(current, stage2RotatePlainLeft(current, 8))

	rows := make([]complex128, len(current))
	rowWork := current
	for row := 0; row < realStage2OutputWidth; row++ {
		rows = stage2AddPlain(rows, stage2ApplyPlainMask(rowWork, stage2RowMask(len(current), row)))
		if row+1 < realStage2OutputWidth {
			rowWork = stage2RotatePlainLeft(rowWork, realStage2InputWidth+realStage2OutputWidth)
		}
	}

	channels := make([]complex128, len(current))
	for channel := 0; channel < realStage2InputChannels; channel++ {
		channels = stage2AddPlain(channels, stage2ApplyPlainMask(rows, stage2ChannelMask(len(current), channel)))
		channels = stage2RotatePlainLeft(channels, -(realStage2InputBlock - realStage2OutputBlock))
	}
	return stage2RotatePlainLeft(channels, (realStage2InputBlock-realStage2OutputBlock)*realStage2InputChannels)
}

func plainPackStage2Halves(first, second []complex128) []complex128 {
	firstCompact := plainCompactStage2Half(first)
	secondCompact := stage2RotatePlainLeft(plainCompactStage2Half(second), -realStage2LogicalSlots/2)
	packed := stage2AddPlain(firstCompact, secondCompact)
	return stage2AddPlain(packed, stage2RotatePlainLeft(packed, -realStage2LogicalSlots))
}

func stage2EncodeUnitMask(params ckks.Parameters, encoder ckks.Encoder, level int, values []complex128) *ckks.Plaintext {
	// Five public masks lie on each surviving packing path. Choose their
	// common scale so that the product is exactly the product of the top three
	// active Q primes. One Rescale after packing therefore consumes those
	// three primes and returns to the ordinary ciphertext scale. This avoids
	// both the invalid scale-1 mask encoding and a low-precision scale reset.
	q := params.Q()
	if level < 3 {
		panic("stage2 packing requires at least three rescale primes")
	}
	packingDivisor := float64(q[level]) * float64(q[level-1]) * float64(q[level-2])
	maskScale := math.Pow(packingDivisor, 1.0/5.0)
	plaintext := ckks.NewPlaintext(params, level, maskScale)
	encoder.EncodeNTT(plaintext, values, params.LogSlots())
	return plaintext
}

func stage2MaskedCiphertext(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, input *ckks.Ciphertext, mask []complex128) *ckks.Ciphertext {
	return evaluator.MulNew(input, stage2EncodeUnitMask(params, encoder, input.Level(), mask))
}

func encryptedCompactStage2Half(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, input *ckks.Ciphertext) *ckks.Ciphertext {
	current := input.CopyNew()
	for _, step := range []struct {
		rotation int
		keep     int
	}{{1, 2}, {2, 4}, {4, 8}} {
		current = evaluator.AddNew(current, evaluator.RotateNew(current, step.rotation))
		current = stage2MaskedCiphertext(params, encoder, evaluator, current, stage2AlternatingMask(params.Slots(), step.keep))
	}
	current = evaluator.AddNew(current, evaluator.RotateNew(current, 8))

	// The upstream loop rotates a work ciphertext by 48 slots after each row.
	// Evaluate the same linear map in parallel so that every term traverses one
	// (rather than up to sixteen) public masks.
	var rows *ckks.Ciphertext
	for row := 0; row < realStage2OutputWidth; row++ {
		rowWork := current
		if row != 0 {
			rowWork = evaluator.RotateNew(current, row*(realStage2InputWidth+realStage2OutputWidth))
		}
		masked := stage2MaskedCiphertext(params, encoder, evaluator, rowWork, stage2RowMask(params.Slots(), row))
		if rows == nil {
			rows = masked
		} else {
			evaluator.Add(rows, masked, rows)
		}
	}

	// Likewise, masking each 1024-slot source channel and rotating it by
	// channel*768 implements the exact upstream channel compaction without a
	// serial chain of sixteen mask multiplications.
	var channels *ckks.Ciphertext
	for channel := 0; channel < realStage2InputChannels; channel++ {
		masked := stage2MaskedCiphertext(params, encoder, evaluator, rows, stage2ChannelMask(params.Slots(), channel))
		if channel != 0 {
			masked = evaluator.RotateNew(masked, channel*(realStage2InputBlock-realStage2OutputBlock))
		}
		if channels == nil {
			channels = masked
		} else {
			evaluator.Add(channels, masked, channels)
		}
	}
	return channels
}

func encryptedPackStage2Halves(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, first, second *ckks.Ciphertext) *ckks.Ciphertext {
	firstCompact := encryptedCompactStage2Half(params, encoder, evaluator, first)
	secondCompact := evaluator.RotateNew(encryptedCompactStage2Half(params, encoder, evaluator, second), -realStage2LogicalSlots/2)
	packed := evaluator.AddNew(firstCompact, secondCompact)
	return evaluator.AddNew(packed, evaluator.RotateNew(packed, -realStage2LogicalSlots))
}

func encryptedRealStage2StrideConv(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage2StrideWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage2InputBlock)
			}
			spatial := resNetBlockSpatialRotations(evaluator, channelShifted)
			for kernelY := 0; kernelY < 3; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[group][diagonal][kernelY][kernelX])
					if err != nil {
						return output, err
					}
					compact, err := interactiveOwnerFold(params, multi, shares, 1)
					if err != nil {
						return output, err
					}
					term := evaluator.MulRelinNew(spatial[kernelY][kernelX], compact)
					if sum == nil {
						sum = term
					} else {
						evaluator.Add(sum, term, sum)
					}
				}
			}
		}
		if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
			return output, err
		}
		biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		biasCompact, err := interactiveOwnerFold(params, biasMulti, shares, 1)
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(biasCompact, biasCompact.Level()-sum.Level())
		evaluator.Add(sum, biasCompact, sum)
		output[group] = sum
	}
	return output, nil
}

func encryptedRealStage2Skip(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage2SkipWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage2InputBlock)
			}
			multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[group][diagonal])
			if err != nil {
				return output, err
			}
			compact, err := interactiveOwnerFold(params, multi, shares, 1)
			if err != nil {
				return output, err
			}
			term := evaluator.MulRelinNew(channelShifted, compact)
			if sum == nil {
				sum = term
			} else {
				evaluator.Add(sum, term, sum)
			}
		}
		if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
			return output, err
		}
		biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		biasCompact, err := interactiveOwnerFold(params, biasMulti, shares, 1)
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(biasCompact, biasCompact.Level()-sum.Level())
		evaluator.Add(sum, biasCompact, sum)
		output[group] = sum
	}
	return output, nil
}

func stage2SpatialRotations(evaluator ckks.Evaluator, input *ckks.Ciphertext) [3][3]*ckks.Ciphertext {
	var shifted [3][3]*ckks.Ciphertext
	shifted[1][1] = input
	shifted[0][1] = evaluator.RotateNew(input, -realStage2OutputWidth)
	shifted[2][1] = evaluator.RotateNew(input, realStage2OutputWidth)
	shifted[1][0] = evaluator.RotateNew(input, -1)
	shifted[1][2] = evaluator.RotateNew(input, 1)
	shifted[0][0] = evaluator.RotateNew(shifted[0][1], -1)
	shifted[0][2] = evaluator.RotateNew(shifted[0][1], 1)
	shifted[2][0] = evaluator.RotateNew(shifted[2][1], -1)
	shifted[2][2] = evaluator.RotateNew(shifted[2][1], 1)
	return shifted
}

func encryptedRealStage2Conv(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage2ConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realStage2OutputChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*realStage2OutputBlock)
		}
		spatial := stage2SpatialRotations(evaluator, channelShifted)
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				multi, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.masks[diagonal][kernelY][kernelX])
				if err != nil {
					return nil, err
				}
				compact, err := interactiveOwnerFold(params, multi, shares, 1)
				if err != nil {
					return nil, err
				}
				term := evaluator.MulRelinNew(spatial[kernelY][kernelX], compact)
				if sum == nil {
					sum = term
				} else {
					evaluator.Add(sum, term, sum)
				}
			}
		}
	}
	if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
		return nil, err
	}
	biasMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 1, weights.packedBias)
	if err != nil {
		return nil, err
	}
	biasCompact, err := interactiveOwnerFold(params, biasMulti, shares, 1)
	if err != nil {
		return nil, err
	}
	evaluator.DropLevel(biasCompact, biasCompact.Level()-sum.Level())
	evaluator.Add(sum, biasCompact, sum)
	return sum, nil
}

func stage2RotationKeys() []int {
	rotations := []int{
		-realStage2InputWidth, -1, 1, realStage2InputWidth,
		2, 4, 8,
		-realStage2LogicalSlots / 2, -realStage2LogicalSlots,
		-realStage2OutputWidth, realStage2OutputWidth,
	}
	for row := 1; row < realStage2OutputWidth; row++ {
		rotations = append(rotations, row*(realStage2InputWidth+realStage2OutputWidth))
	}
	for channel := 1; channel < realStage2InputChannels; channel++ {
		rotations = append(rotations, channel*(realStage2InputBlock-realStage2OutputBlock))
	}
	for diagonal := 1; diagonal < realStage2InputChannels; diagonal++ {
		rotations = append(rotations, diagonal*realStage2InputBlock)
	}
	for diagonal := 1; diagonal < realStage2OutputChannels; diagonal++ {
		rotations = append(rotations, diagonal*realStage2OutputBlock)
	}
	seen := make(map[int]bool, len(rotations))
	deduplicated := make([]int, 0, len(rotations))
	for _, rotation := range rotations {
		if rotation != 0 && !seen[rotation] {
			seen[rotation] = true
			deduplicated = append(deduplicated, rotation)
		}
	}
	sort.Ints(deduplicated)
	return deduplicated
}

func stage2MaxDifference(left, right []complex128) (float64, float64, error) {
	if len(left) != len(right) {
		return 0, 0, fmt.Errorf("length mismatch: %d != %d", len(left), len(right))
	}
	var maximum, squared float64
	for i := range left {
		difference := cmplx.Abs(left[i] - right[i])
		maximum = math.Max(maximum, difference)
		squared += difference * difference
	}
	return maximum, math.Sqrt(squared / float64(len(left))), nil
}

func stage2AddPlainBranches(main, skip []complex128) []complex128 {
	output := make([]complex128, len(main))
	for i := range output {
		output[i] = main[i] + skip[i]
	}
	return output
}

func TestEncryptedRealResNet20Stage2Block1TransitionP2TwoParty(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	mainWeights, err := loadRealStage2StrideWeights(weightsDirectory, realStage2MainScale, params.Slots())
	if err != nil {
		t.Fatalf("layer4-conv1bn1 audit failed: %v", err)
	}
	skipWeights, err := loadRealStage2SkipWeights(weightsDirectory, realStage2SkipScale, params.Slots())
	if err != nil {
		t.Fatalf("layer4dx-conv1bn1 audit failed: %v", err)
	}
	conv2Weights, err := loadRealStage2ConvWeights(weightsDirectory, "layer4-conv2bn2", realStage2Conv2Scale, params.Slots())
	if err != nil {
		t.Fatalf("layer4-conv2bn2 audit failed: %v", err)
	}
	packedInput, plainInput, err := deterministicRealResNetBlockInput(params.Slots())
	if err != nil {
		t.Fatal(err)
	}

	plainMain := plainRealStage2StrideConv(plainInput, mainWeights)
	plainSkip := plainRealStage2Skip(plainInput, skipWeights)
	plainMainHalves := [2][]complex128{plainSparseStage2Half(plainMain, 0), plainSparseStage2Half(plainMain, 1)}
	plainSkipHalves := [2][]complex128{plainSparseStage2Half(plainSkip, 0), plainSparseStage2Half(plainSkip, 1)}
	packedMainAudit := plainPackStage2Halves(plainMainHalves[0], plainMainHalves[1])
	packedSkipAudit := plainPackStage2Halves(plainSkipHalves[0], plainSkipHalves[1])
	if maximum, _, err := stage2MaxDifference(packedMainAudit, plainMain); err != nil || maximum != 0 {
		t.Fatalf("scalar main packing audit failed: maxDifference=%g err=%v", maximum, err)
	}
	if maximum, _, err := stage2MaxDifference(packedSkipAudit, plainSkip); err != nil || maximum != 0 {
		t.Fatalf("scalar skip packing audit failed: maxDifference=%g err=%v", maximum, err)
	}

	kgen := ckks.NewKeyGenerator(params)
	shares := make([]*rlwe.SecretKey, 2)
	public := make([]*rlwe.PublicKey, 2)
	for party := range shares {
		shares[party], public[party] = kgen.GenKeyPair()
	}
	encoder := ckks.NewEncoder(params)
	inputMulti, err := makeOwnerCKKSMultiCiphertext(params, encoder, public, 0, packedInput)
	if err != nil {
		t.Fatal(err)
	}
	input, err := interactiveOwnerFold(params, inputMulti, shares, 0)
	if err != nil {
		t.Fatal(err)
	}

	rotations := stage2RotationKeys()
	evaluationCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	evaluationKey, err := generateCollectiveEvaluationKey(params, shares, rotations, evaluationCRS)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := ckks.NewEvaluator(params, evaluationKey)
	minLevel, logBound, ok := dckks.GetMinimumLevelForBootstrapping(128, params.Scale(), len(shares), params.Q())
	if !ok {
		t.Fatal("P2 lacks modulus levels for two-party collective refresh")
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	aggregate := aggregateSecret(params.Parameters, shares)
	started := time.Now()

	inputMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, input, plainInput)
	refreshedInput, inputRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, input, minLevel, logBound, refreshCRS, encoder, aggregate, plainInput)
	if err != nil {
		t.Fatalf("pre-transition refresh failed: %v", err)
	}

	mainHalves, err := encryptedRealStage2StrideConv(params, encoder, evaluator, public, shares, refreshedInput, mainWeights)
	if err != nil {
		t.Fatalf("layer4-conv1bn1 failed: %v", err)
	}
	skipHalves, err := encryptedRealStage2Skip(params, encoder, evaluator, public, shares, refreshedInput, skipWeights)
	if err != nil {
		t.Fatalf("layer4dx-conv1bn1 failed: %v", err)
	}
	mainHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[0], plainMainHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[1], plainMainHalves[1]),
	}
	skipHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[0], plainSkipHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[1], plainSkipHalves[1]),
	}

	packedMain := encryptedPackStage2Halves(params, encoder, evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage2Halves(params, encoder, evaluator, skipHalves[0], skipHalves[1])
	if err := evaluator.Rescale(packedMain, params.Scale(), packedMain); err != nil {
		t.Fatalf("packed-main scale normalization failed: %v", err)
	}
	if err := evaluator.Rescale(packedSkip, params.Scale(), packedSkip); err != nil {
		t.Fatalf("packed-skip scale normalization failed: %v", err)
	}
	packedMainMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedMain, plainMain)
	packedSkipMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, packedSkip, plainSkip)

	refreshedMain, mainRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, packedMain, minLevel, logBound, refreshCRS, encoder, aggregate, plainMain)
	if err != nil {
		t.Fatalf("post-pack main refresh failed: %v", err)
	}
	poly1 := referenceResNetBlockReLUPolynomial(realStage2MainScale)
	plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainMain, poly1, realStage2MainScale)
	if err != nil {
		t.Fatalf("first activation plaintext interval failed: %v", err)
	}
	activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedMain, poly1, params.Scale())
	if err != nil {
		t.Fatalf("first degree-%d activation failed: %v", resnetReferenceReLUDegree, err)
	}
	activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)

	plainConv2 := plainRealStage2Conv(plainActivation1, conv2Weights)
	conv2, err := encryptedRealStage2Conv(params, encoder, evaluator, public, shares, activation1, conv2Weights)
	if err != nil {
		t.Fatalf("layer4-conv2bn2 failed: %v", err)
	}
	conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
	if packedSkip.Level() < conv2.Level() {
		t.Fatalf("skip level %d below main conv2 level %d", packedSkip.Level(), conv2.Level())
	}
	evaluator.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
	evaluator.Add(conv2, packedSkip, conv2)
	plainResidual := stage2AddPlainBranches(plainConv2, plainSkip)
	residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)

	refreshedResidual, residualRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
	if err != nil {
		t.Fatalf("post-residual refresh failed: %v", err)
	}
	poly2 := referenceResNetBlockReLUPolynomial(realStage2SkipScale)
	plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, realStage2SkipScale)
	if err != nil {
		t.Fatalf("second activation plaintext interval failed: %v", err)
	}
	activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedResidual, poly2, params.Scale())
	if err != nil {
		t.Fatalf("second degree-%d activation failed: %v", resnetReferenceReLUDegree, err)
	}
	activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation2, plainActivation2)

	boundaries := []struct {
		name        string
		measurement blockBoundaryMeasurement
	}{
		{"input", inputMeasurement},
		{"input-refresh", blockBoundaryMeasurement{level: inputRefresh.outputLevel, maxError: inputRefresh.maxError, rmse: inputRefresh.rmse}},
		{"main-half0", mainHalfMeasurements[0]}, {"main-half1", mainHalfMeasurements[1]},
		{"skip-half0", skipHalfMeasurements[0]}, {"skip-half1", skipHalfMeasurements[1]},
		{"packed-main", packedMainMeasurement}, {"packed-skip", packedSkipMeasurement},
		{"main-refresh", blockBoundaryMeasurement{level: mainRefresh.outputLevel, maxError: mainRefresh.maxError, rmse: mainRefresh.rmse}},
		{"activation1", activation1Measurement}, {"conv2", conv2Measurement}, {"residual", residualMeasurement},
		{"residual-refresh", blockBoundaryMeasurement{level: residualRefresh.outputLevel, maxError: residualRefresh.maxError, rmse: residualRefresh.rmse}},
		{"activation2", activation2Measurement},
	}
	for _, boundary := range boundaries {
		if boundary.measurement.maxError >= 1e-3 {
			t.Fatalf("stage2 transition boundary %s failed: level=%d maxError=%g rmse=%g", boundary.name, boundary.measurement.level, boundary.measurement.maxError, boundary.measurement.rmse)
		}
	}

	elapsed := time.Since(started)
	t.Logf("real ResNet-20 Stage2/Layer2 Block1 transition P2: PASS; sourcePrefixMain=layer4-conv1bn1 sourcePrefixSkip=layer4dx-conv1bn1 sourcePrefixConv2=layer4-conv2bn2 mainScale=%.2f skipScale=%.2f conv2Scale=%.2f dataOwner=0 modelOwner=1 inputCiphertexts=1 encryptedWeightCiphertexts=608 encryptedBiasCiphertexts=5 ct-ct-products=608 collectiveRefreshes=3 degree59Activations=2 rotationKeys=%d rotationCalls=743 publicPackingMaskMultiplications=140 packingRescaleLevels=3 logicalPacking=32x16x16x2copies checkedSlotsPerPackedBoundary=%d", realStage2MainScale, realStage2SkipScale, realStage2Conv2Scale, len(rotations), params.Slots())
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	t.Logf("boundary pre-transition refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", inputRefresh.preDropLevel, inputRefresh.inputLevel, inputRefresh.outputLevel, inputRefresh.logBound, inputRefresh.maxError, inputRefresh.rmse)
	t.Logf("boundary stride-main halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", mainHalfMeasurements[0].level, mainHalfMeasurements[0].maxError, mainHalfMeasurements[0].rmse, mainHalfMeasurements[1].level, mainHalfMeasurements[1].maxError, mainHalfMeasurements[1].rmse)
	t.Logf("boundary stride-skip halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", skipHalfMeasurements[0].level, skipHalfMeasurements[0].maxError, skipHalfMeasurements[0].rmse, skipHalfMeasurements[1].level, skipHalfMeasurements[1].maxError, skipHalfMeasurements[1].rmse)
	t.Logf("boundary packed-main: level=%d maxError=%g rmse=%g; packed-skip: level=%d maxError=%g rmse=%g", packedMainMeasurement.level, packedMainMeasurement.maxError, packedMainMeasurement.rmse, packedSkipMeasurement.level, packedSkipMeasurement.maxError, packedSkipMeasurement.rmse)
	t.Logf("boundary post-pack main refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", mainRefresh.preDropLevel, mainRefresh.inputLevel, mainRefresh.outputLevel, mainRefresh.logBound, mainRefresh.maxError, mainRefresh.rmse)
	t.Logf("boundary activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage2MainScale, activation1Measurement.level, activation1Measurement.maxError, activation1Measurement.rmse, approximation1)
	t.Logf("boundary conv2(prefix=layer4-conv2bn2 scale=%.2f): level=%d maxError=%g rmse=%g", realStage2Conv2Scale, conv2Measurement.level, conv2Measurement.maxError, conv2Measurement.rmse)
	t.Logf("boundary residual(main+projectedSkip): level=%d maxError=%g rmse=%g", residualMeasurement.level, residualMeasurement.maxError, residualMeasurement.rmse)
	t.Logf("boundary post-residual refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", residualRefresh.preDropLevel, residualRefresh.inputLevel, residualRefresh.outputLevel, residualRefresh.logBound, residualRefresh.maxError, residualRefresh.rmse)
	t.Logf("boundary activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage2SkipScale, activation2Measurement.level, activation2Measurement.maxError, activation2Measurement.rmse, approximation2)
	t.Logf("functional-only elapsed=%s; this is a correctness-gate wall time, not a CPU or FPGA benchmark", elapsed)
}
