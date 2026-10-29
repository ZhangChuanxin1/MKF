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

// The names and scales below are the exact Layer3/Block1 schedule in the
// frozen LowMemoryFHEResNet20 src/main.cpp lines 254--288. The upstream
// repository calls this 32->64 transition "layer7" in its weight filenames.
const (
	realStage3InputChannels     = 32
	realStage3OutputChannels    = 64
	realStage3InputWidth        = 16
	realStage3OutputWidth       = 8
	realStage3InputBlock        = realStage3InputWidth * realStage3InputWidth
	realStage3OutputBlock       = realStage3OutputWidth * realStage3OutputWidth
	realStage3InputLogicalSlots = realStage3InputChannels * realStage3InputBlock
	realStage3LogicalSlots      = realStage3OutputChannels * realStage3OutputBlock
	realStage3MainScale         = 0.63
	realStage3SkipScale         = 0.40
	realStage3Conv2Scale        = 0.40
	stage3AuditTolerance        = 1e-15
)

type realStage3StrideWeights struct {
	// The two groups correspond to output channels 0--31 and 32--63.
	masks      [2][realStage3InputChannels][3][3][]complex128
	kernel     [realStage3OutputChannels][realStage3InputChannels][3][3]float64
	packedBias [2][]complex128
	bias       [realStage3OutputChannels]float64
}

type realStage3SkipWeights struct {
	masks      [2][realStage3InputChannels][]complex128
	kernel     [realStage3OutputChannels][realStage3InputChannels]float64
	packedBias [2][]complex128
	bias       [realStage3OutputChannels]float64
}

type realStage3ConvWeights struct {
	masks      [realStage3OutputChannels][3][3][]complex128
	kernel     [realStage3OutputChannels][realStage3OutputChannels][3][3]float64
	packedBias []complex128
	bias       [realStage3OutputChannels]float64
}

func stage3FrozenFile(weightsDirectory, name string) string {
	return filepath.Join(weightsDirectory, name)
}

func stage3ExpectLength(path string, values []float64, want int) error {
	if len(values) != want {
		return fmt.Errorf("%s: got %d values, want %d", path, len(values), want)
	}
	return nil
}

func stage3SparsePositionIsActive(y, x int) bool {
	return y%2 == 0 && x%2 == 0
}

// loadRealStage3StrideWeights audits the exact masks emitted by notebook
// "Algorithm 2 - Exporting Weights", Layer3 export cells. In particular it checks
// altalena2's even-row/even-column stride mask, the 32 channel diagonals, both
// output-channel halves, and every 3x3 padding location before unrolling the
// frozen diagonal masks into the vectors consumed by the Go evaluator.
func loadRealStage3StrideWeights(weightsDirectory string, scale float64, slots int) (*realStage3StrideWeights, error) {
	if slots != realResNetStemActiveSlots {
		return nil, fmt.Errorf("stage3 P2 transition requires exactly %d slots, got %d", realResNetStemActiveSlots, slots)
	}
	loaded := new(realStage3StrideWeights)
	inputCopies := slots / realStage3InputLogicalSlots
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			fileChannel := group*realStage3InputChannels + diagonal
			for kernelY := 0; kernelY < 3; kernelY++ {
				for kernelX := 0; kernelX < 3; kernelX++ {
					kernelIndex := kernelY*3 + kernelX + 1
					path := stage3FrozenFile(weightsDirectory, fmt.Sprintf("layer7-conv1bn1-ch%d-k%d.bin", fileChannel, kernelIndex))
					values, err := readFrozenFloatVector(path)
					if err != nil {
						return nil, err
					}
					if err := stage3ExpectLength(path, values, realStage3InputLogicalSlots); err != nil {
						return nil, err
					}
					loaded.masks[group][diagonal][kernelY][kernelX] = make([]complex128, slots)
					deltaY, deltaX := kernelY-1, kernelX-1
					for localOutput := 0; localOutput < realStage3InputChannels; localOutput++ {
						sourceBlock := (localOutput + diagonal) % realStage3InputChannels
						sourceBase := sourceBlock * realStage3InputBlock
						coefficient := values[sourceBase+8*realStage3InputWidth+8]
						globalOutput := group*realStage3InputChannels + localOutput
						loaded.kernel[globalOutput][sourceBlock][kernelY][kernelX] = scale * coefficient
						destinationBase := localOutput * realStage3InputBlock
						for y := 0; y < realStage3InputWidth; y++ {
							for x := 0; x < realStage3InputWidth; x++ {
								position := y*realStage3InputWidth + x
								got := values[sourceBase+position]
								validPadding := y+deltaY >= 0 && y+deltaY < realStage3InputWidth && x+deltaX >= 0 && x+deltaX < realStage3InputWidth
								want := 0.0
								if stage3SparsePositionIsActive(y, x) && validPadding {
									want = coefficient
								}
								if math.Abs(got-want) > stage3AuditTolerance {
									return nil, fmt.Errorf("%s: invalid stride/padding mask at group=%d output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, group, globalOutput, sourceBlock, y, x, got, want)
								}
								for copyIndex := 0; copyIndex < inputCopies; copyIndex++ {
									loaded.masks[group][diagonal][kernelY][kernelX][copyIndex*realStage3InputLogicalSlots+destinationBase+position] = complex(scale*got, 0)
								}
							}
						}
					}
				}
			}
		}

		biasPath := stage3FrozenFile(weightsDirectory, fmt.Sprintf("layer7-conv1bn1-bias%d.bin", group+1))
		biasValues, err := readFrozenFloatVector(biasPath)
		if err != nil {
			return nil, err
		}
		if err := stage3ExpectLength(biasPath, biasValues, realStage3InputLogicalSlots); err != nil {
			return nil, err
		}
		loaded.packedBias[group] = make([]complex128, slots)
		for localOutput := 0; localOutput < realStage3InputChannels; localOutput++ {
			base := localOutput * realStage3InputBlock
			rawBias := biasValues[base]
			loaded.bias[group*realStage3InputChannels+localOutput] = scale * rawBias
			for y := 0; y < realStage3InputWidth; y++ {
				for x := 0; x < realStage3InputWidth; x++ {
					position := y*realStage3InputWidth + x
					want := 0.0
					if stage3SparsePositionIsActive(y, x) {
						want = rawBias
					}
					if math.Abs(biasValues[base+position]-want) > stage3AuditTolerance {
						return nil, fmt.Errorf("%s: invalid sparse bias at output=%d y=%d x=%d: got %.17g want %.17g", biasPath, group*realStage3InputChannels+localOutput, y, x, biasValues[base+position], want)
					}
					for copyIndex := 0; copyIndex < inputCopies; copyIndex++ {
						loaded.packedBias[group][copyIndex*realStage3InputLogicalSlots+base+position] = complex(scale*biasValues[base+position], 0)
					}
				}
			}
		}
	}
	return loaded, nil
}

// loadRealStage3SkipWeights audits the 1x1 stride-2 projection exported by
// notebook Layer3 projection export under the exact layer7dx-conv1bn1 prefix.
func loadRealStage3SkipWeights(weightsDirectory string, scale float64, slots int) (*realStage3SkipWeights, error) {
	if slots != realResNetStemActiveSlots {
		return nil, fmt.Errorf("stage3 P2 transition requires exactly %d slots, got %d", realResNetStemActiveSlots, slots)
	}
	loaded := new(realStage3SkipWeights)
	inputCopies := slots / realStage3InputLogicalSlots
	for group := 0; group < 2; group++ {
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			fileChannel := group*realStage3InputChannels + diagonal
			path := stage3FrozenFile(weightsDirectory, fmt.Sprintf("layer7dx-conv1bn1-ch%d-k1.bin", fileChannel))
			values, err := readFrozenFloatVector(path)
			if err != nil {
				return nil, err
			}
			if err := stage3ExpectLength(path, values, realStage3InputLogicalSlots); err != nil {
				return nil, err
			}
			loaded.masks[group][diagonal] = make([]complex128, slots)
			for localOutput := 0; localOutput < realStage3InputChannels; localOutput++ {
				sourceBlock := (localOutput + diagonal) % realStage3InputChannels
				sourceBase := sourceBlock * realStage3InputBlock
				coefficient := values[sourceBase+8*realStage3InputWidth+8]
				globalOutput := group*realStage3InputChannels + localOutput
				loaded.kernel[globalOutput][sourceBlock] = scale * coefficient
				destinationBase := localOutput * realStage3InputBlock
				for y := 0; y < realStage3InputWidth; y++ {
					for x := 0; x < realStage3InputWidth; x++ {
						position := y*realStage3InputWidth + x
						got := values[sourceBase+position]
						want := 0.0
						if stage3SparsePositionIsActive(y, x) {
							want = coefficient
						}
						if math.Abs(got-want) > stage3AuditTolerance {
							return nil, fmt.Errorf("%s: invalid 1x1 stride mask at group=%d output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, group, globalOutput, sourceBlock, y, x, got, want)
						}
						for copyIndex := 0; copyIndex < inputCopies; copyIndex++ {
							loaded.masks[group][diagonal][copyIndex*realStage3InputLogicalSlots+destinationBase+position] = complex(scale*got, 0)
						}
					}
				}
			}
		}

		biasPath := stage3FrozenFile(weightsDirectory, fmt.Sprintf("layer7dx-conv1bn1-bias%d.bin", group+1))
		biasValues, err := readFrozenFloatVector(biasPath)
		if err != nil {
			return nil, err
		}
		if err := stage3ExpectLength(biasPath, biasValues, realStage3InputLogicalSlots); err != nil {
			return nil, err
		}
		loaded.packedBias[group] = make([]complex128, slots)
		for localOutput := 0; localOutput < realStage3InputChannels; localOutput++ {
			base := localOutput * realStage3InputBlock
			rawBias := biasValues[base]
			loaded.bias[group*realStage3InputChannels+localOutput] = scale * rawBias
			for y := 0; y < realStage3InputWidth; y++ {
				for x := 0; x < realStage3InputWidth; x++ {
					position := y*realStage3InputWidth + x
					want := 0.0
					if stage3SparsePositionIsActive(y, x) {
						want = rawBias
					}
					if math.Abs(biasValues[base+position]-want) > stage3AuditTolerance {
						return nil, fmt.Errorf("%s: invalid projection bias at output=%d y=%d x=%d: got %.17g want %.17g", biasPath, group*realStage3InputChannels+localOutput, y, x, biasValues[base+position], want)
					}
					for copyIndex := 0; copyIndex < inputCopies; copyIndex++ {
						loaded.packedBias[group][copyIndex*realStage3InputLogicalSlots+base+position] = complex(scale*biasValues[base+position], 0)
					}
				}
			}
		}
	}
	return loaded, nil
}

// loadRealStage3ConvWeights audits the post-transition 64-channel 3x3
// convolution files used by FHEController::convbn3.
func loadRealStage3ConvWeights(weightsDirectory string, scale float64, slots int) (*realStage3ConvWeights, error) {
	if slots%realStage3LogicalSlots != 0 {
		return nil, fmt.Errorf("stage3 logical layout needs a multiple of %d slots, got %d", realStage3LogicalSlots, slots)
	}
	loaded := new(realStage3ConvWeights)
	packingCopies := slots / realStage3LogicalSlots
	for diagonal := 0; diagonal < realStage3OutputChannels; diagonal++ {
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				kernelIndex := kernelY*3 + kernelX + 1
				path := stage3FrozenFile(weightsDirectory, fmt.Sprintf("layer7-conv2bn2-ch%d-k%d.bin", diagonal, kernelIndex))
				values, err := readFrozenFloatVector(path)
				if err != nil {
					return nil, err
				}
				if err := stage3ExpectLength(path, values, realStage3LogicalSlots); err != nil {
					return nil, err
				}
				loaded.masks[diagonal][kernelY][kernelX] = make([]complex128, slots)
				deltaY, deltaX := kernelY-1, kernelX-1
				for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
					sourceBlock := (outputChannel + diagonal) % realStage3OutputChannels
					sourceBase := sourceBlock * realStage3OutputBlock
					coefficient := values[sourceBase+4*realStage3OutputWidth+4]
					loaded.kernel[outputChannel][sourceBlock][kernelY][kernelX] = scale * coefficient
					destinationBase := outputChannel * realStage3OutputBlock
					for y := 0; y < realStage3OutputWidth; y++ {
						for x := 0; x < realStage3OutputWidth; x++ {
							position := y*realStage3OutputWidth + x
							got := values[sourceBase+position]
							valid := y+deltaY >= 0 && y+deltaY < realStage3OutputWidth && x+deltaX >= 0 && x+deltaX < realStage3OutputWidth
							want := 0.0
							if valid {
								want = coefficient
							}
							if math.Abs(got-want) > stage3AuditTolerance {
								return nil, fmt.Errorf("%s: invalid 8x8 padding mask at output=%d input=%d y=%d x=%d: got %.17g want %.17g", path, outputChannel, sourceBlock, y, x, got, want)
							}
							for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
								loaded.masks[diagonal][kernelY][kernelX][copyIndex*realStage3LogicalSlots+destinationBase+position] = complex(scale*got, 0)
							}
						}
					}
				}
			}
		}
	}

	biasPath := stage3FrozenFile(weightsDirectory, "layer7-conv2bn2-bias.bin")
	biasValues, err := readFrozenFloatVector(biasPath)
	if err != nil {
		return nil, err
	}
	if err := stage3ExpectLength(biasPath, biasValues, realStage3LogicalSlots); err != nil {
		return nil, err
	}
	loaded.packedBias = make([]complex128, slots)
	for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
		base := outputChannel * realStage3OutputBlock
		rawBias := biasValues[base]
		loaded.bias[outputChannel] = scale * rawBias
		for position := 0; position < realStage3OutputBlock; position++ {
			if math.Abs(biasValues[base+position]-rawBias) > stage3AuditTolerance {
				return nil, fmt.Errorf("%s: bias block %d is not constant at position %d", biasPath, outputChannel, position)
			}
			for copyIndex := 0; copyIndex < packingCopies; copyIndex++ {
				loaded.packedBias[copyIndex*realStage3LogicalSlots+base+position] = complex(scale*rawBias, 0)
			}
		}
	}
	return loaded, nil
}

// deterministicRealStage3Input is a bounded, nonnegative 32x16x16 feature
// map. It is repeated twice because the frozen Layer3 transition encodes
// 8,192 logical input slots and then widens the CKKS slot metadata to 16,384
// before downsample256to64.
func deterministicRealStage3Input(slots int) ([]complex128, []complex128, error) {
	if slots != realResNetStemActiveSlots || slots%realStage3InputLogicalSlots != 0 {
		return nil, nil, fmt.Errorf("stage3 P2 input requires exactly %d slots as repeated %d-slot layouts, got %d", realResNetStemActiveSlots, realStage3InputLogicalSlots, slots)
	}
	logical := make([]complex128, realStage3InputLogicalSlots)
	for channel := 0; channel < realStage3InputChannels; channel++ {
		base := channel * realStage3InputBlock
		for y := 0; y < realStage3InputWidth; y++ {
			for x := 0; x < realStage3InputWidth; x++ {
				// Values remain comfortably inside the frozen polynomial's
				// interval even after the BN-folded transition convolutions.
				logical[base+y*realStage3InputWidth+x] = complex(float64(((channel+3)*11+y*7+x*5)%29)/8192.0, 0)
			}
		}
	}
	packed := make([]complex128, slots)
	for copyIndex := 0; copyIndex < slots/realStage3InputLogicalSlots; copyIndex++ {
		copy(packed[copyIndex*realStage3InputLogicalSlots:], logical)
	}
	return packed, append([]complex128(nil), packed...), nil
}

func plainRealStage3StrideConv(input []complex128, weights *realStage3StrideWeights) []complex128 {
	logical := make([]complex128, realStage3LogicalSlots)
	for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
		base := outputChannel * realStage3OutputBlock
		for y := 0; y < realStage3OutputWidth; y++ {
			for x := 0; x < realStage3OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage3InputChannels; inputChannel++ {
					inputBase := inputChannel * realStage3InputBlock
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY := 2*y + kernelY - 1
							sourceX := 2*x + kernelX - 1
							if sourceY >= 0 && sourceY < realStage3InputWidth && sourceX >= 0 && sourceX < realStage3InputWidth {
								sum += input[inputBase+sourceY*realStage3InputWidth+sourceX] * complex(weights.kernel[outputChannel][inputChannel][kernelY][kernelX], 0)
							}
						}
					}
				}
				logical[base+y*realStage3OutputWidth+x] = sum
			}
		}
	}
	return repeatStage3Logical(logical, len(input))
}

func plainRealStage3Skip(input []complex128, weights *realStage3SkipWeights) []complex128 {
	logical := make([]complex128, realStage3LogicalSlots)
	for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
		base := outputChannel * realStage3OutputBlock
		for y := 0; y < realStage3OutputWidth; y++ {
			for x := 0; x < realStage3OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage3InputChannels; inputChannel++ {
					inputBase := inputChannel * realStage3InputBlock
					sum += input[inputBase+(2*y)*realStage3InputWidth+2*x] * complex(weights.kernel[outputChannel][inputChannel], 0)
				}
				logical[base+y*realStage3OutputWidth+x] = sum
			}
		}
	}
	return repeatStage3Logical(logical, len(input))
}

func plainRealStage3Conv(input []complex128, weights *realStage3ConvWeights) []complex128 {
	logical := make([]complex128, realStage3LogicalSlots)
	for outputChannel := 0; outputChannel < realStage3OutputChannels; outputChannel++ {
		base := outputChannel * realStage3OutputBlock
		for y := 0; y < realStage3OutputWidth; y++ {
			for x := 0; x < realStage3OutputWidth; x++ {
				sum := complex(weights.bias[outputChannel], 0)
				for inputChannel := 0; inputChannel < realStage3OutputChannels; inputChannel++ {
					inputBase := inputChannel * realStage3OutputBlock
					for kernelY := 0; kernelY < 3; kernelY++ {
						for kernelX := 0; kernelX < 3; kernelX++ {
							sourceY := y + kernelY - 1
							sourceX := x + kernelX - 1
							if sourceY >= 0 && sourceY < realStage3OutputWidth && sourceX >= 0 && sourceX < realStage3OutputWidth {
								sum += input[inputBase+sourceY*realStage3OutputWidth+sourceX] * complex(weights.kernel[outputChannel][inputChannel][kernelY][kernelX], 0)
							}
						}
					}
				}
				logical[base+y*realStage3OutputWidth+x] = sum
			}
		}
	}
	return repeatStage3Logical(logical, len(input))
}

func repeatStage3Logical(logical []complex128, slots int) []complex128 {
	if len(logical) != realStage3LogicalSlots || slots%realStage3LogicalSlots != 0 {
		panic("invalid stage3 logical repetition")
	}
	output := make([]complex128, slots)
	for copyIndex := 0; copyIndex < slots/realStage3LogicalSlots; copyIndex++ {
		copy(output[copyIndex*realStage3LogicalSlots:], logical)
	}
	return output
}

func plainSparseStage3Half(packed []complex128, group int) []complex128 {
	output := make([]complex128, realResNetStemActiveSlots)
	for copyIndex := 0; copyIndex < len(output)/realStage3InputLogicalSlots; copyIndex++ {
		copyBase := copyIndex * realStage3InputLogicalSlots
		for localChannel := 0; localChannel < realStage3InputChannels; localChannel++ {
			logicalBase := (group*realStage3InputChannels + localChannel) * realStage3OutputBlock
			sparseBase := copyBase + localChannel*realStage3InputBlock
			for y := 0; y < realStage3OutputWidth; y++ {
				for x := 0; x < realStage3OutputWidth; x++ {
					output[sparseBase+(2*y)*realStage3InputWidth+2*x] = packed[logicalBase+y*realStage3OutputWidth+x]
				}
			}
		}
	}
	return output
}

func stage3RotatePlainLeft(input []complex128, rotation int) []complex128 {
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

func stage3AddPlain(left, right []complex128) []complex128 {
	output := make([]complex128, len(left))
	for i := range output {
		output[i] = left[i] + right[i]
	}
	return output
}

func stage3ApplyPlainMask(input, mask []complex128) []complex128 {
	output := make([]complex128, len(input))
	for i := range output {
		output[i] = input[i] * mask[i]
	}
	return output
}

func stage3AlternatingMask(slots, keep int) []complex128 {
	mask := make([]complex128, slots)
	for i := range mask {
		if i%(2*keep) < keep {
			mask[i] = 1
		}
	}
	return mask
}

func stage3RowMask(slots, row int) []complex128 {
	mask := make([]complex128, slots)
	for channel := 0; channel < realStage3OutputChannels; channel++ {
		base := channel*realStage3InputBlock + row*realStage3OutputWidth
		for x := 0; x < realStage3OutputWidth; x++ {
			mask[base+x] = 1
		}
	}
	return mask
}

func stage3HalfMask(slots int, second bool) []complex128 {
	mask := make([]complex128, slots)
	start, end := 0, realStage3InputLogicalSlots
	if second {
		start, end = realStage3InputLogicalSlots, 2*realStage3InputLogicalSlots
	}
	for i := start; i < end; i++ {
		mask[i] = 1
	}
	return mask
}

func stage3ChannelMask(slots, channel int) []complex128 {
	mask := make([]complex128, slots)
	base := channel * realStage3InputBlock
	for i := 0; i < realStage3OutputBlock; i++ {
		mask[base+i] = 1
	}
	return mask
}

// plainPackStage3Halves is an independent scalar simulation of the exact
// downsample256to64 linear map in FHEController.cpp lines 1231--1274. The
// encrypted implementation below evaluates the same row/channel terms in
// parallel to cap multiplicative depth; the scalar equality check therefore
// audits all first/second-half, stride, row, channel, and four-copy semantics.
func plainPackStage3Halves(first, second []complex128) []complex128 {
	current := stage3AddPlain(
		stage3ApplyPlainMask(first, stage3HalfMask(len(first), false)),
		stage3ApplyPlainMask(second, stage3HalfMask(len(second), true)),
	)
	for _, step := range []struct {
		rotation int
		keep     int
	}{{1, 2}, {2, 4}} {
		current = stage3ApplyPlainMask(stage3AddPlain(current, stage3RotatePlainLeft(current, step.rotation)), stage3AlternatingMask(len(current), step.keep))
	}
	current = stage3AddPlain(current, stage3RotatePlainLeft(current, 4))

	rows := make([]complex128, len(current))
	for row := 0; row < 2*realStage3InputWidth; row++ {
		rowWork := current
		if row != 0 {
			rowWork = stage3RotatePlainLeft(current, row*(realStage3InputWidth+realStage3OutputWidth))
		}
		rows = stage3AddPlain(rows, stage3ApplyPlainMask(rowWork, stage3RowMask(len(current), row)))
	}

	channels := make([]complex128, len(current))
	for channel := 0; channel < realStage3OutputChannels; channel++ {
		masked := stage3ApplyPlainMask(rows, stage3ChannelMask(len(current), channel))
		if channel != 0 {
			masked = stage3RotatePlainLeft(masked, channel*(realStage3InputBlock-realStage3OutputBlock))
		}
		channels = stage3AddPlain(channels, masked)
	}
	channels = stage3AddPlain(channels, stage3RotatePlainLeft(channels, -realStage3LogicalSlots))
	return stage3AddPlain(channels, stage3RotatePlainLeft(channels, -2*realStage3LogicalSlots))
}

func stage3EncodeUnitMask(params ckks.Parameters, encoder ckks.Encoder, level int, values []complex128) *ckks.Plaintext {
	// Five public masks lie on each surviving packing path. Choose their
	// common scale so that the product is exactly the product of the top three
	// active Q primes. One Rescale after packing therefore consumes those
	// three primes and returns to the ordinary ciphertext scale. This avoids
	// both the invalid scale-1 mask encoding and a low-precision scale reset.
	q := params.Q()
	if level < 3 {
		panic("stage3 packing requires at least three rescale primes")
	}
	packingDivisor := float64(q[level]) * float64(q[level-1]) * float64(q[level-2])
	maskScale := math.Pow(packingDivisor, 1.0/5.0)
	plaintext := ckks.NewPlaintext(params, level, maskScale)
	encoder.EncodeNTT(plaintext, values, params.LogSlots())
	return plaintext
}

func stage3MaskedCiphertext(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, input *ckks.Ciphertext, mask []complex128) *ckks.Ciphertext {
	return evaluator.MulNew(input, stage3EncodeUnitMask(params, encoder, input.Level(), mask))
}

func encryptedPackStage3Halves(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, first, second *ckks.Ciphertext) *ckks.Ciphertext {
	firstSelected := stage3MaskedCiphertext(params, encoder, evaluator, first, stage3HalfMask(params.Slots(), false))
	secondSelected := stage3MaskedCiphertext(params, encoder, evaluator, second, stage3HalfMask(params.Slots(), true))
	current := evaluator.AddNew(firstSelected, secondSelected)
	for _, step := range []struct {
		rotation int
		keep     int
	}{{1, 2}, {2, 4}} {
		current = evaluator.AddNew(current, evaluator.RotateNew(current, step.rotation))
		current = stage3MaskedCiphertext(params, encoder, evaluator, current, stage3AlternatingMask(params.Slots(), step.keep))
	}
	current = evaluator.AddNew(current, evaluator.RotateNew(current, 4))

	// The upstream loop rotates a work ciphertext by 24 slots after each row.
	// Evaluate the same linear map in parallel so that every term traverses one
	// (rather than up to thirty-two) public row masks.
	var rows *ckks.Ciphertext
	for row := 0; row < 2*realStage3InputWidth; row++ {
		rowWork := current
		if row != 0 {
			rowWork = evaluator.RotateNew(current, row*(realStage3InputWidth+realStage3OutputWidth))
		}
		masked := stage3MaskedCiphertext(params, encoder, evaluator, rowWork, stage3RowMask(params.Slots(), row))
		if rows == nil {
			rows = masked
		} else {
			evaluator.Add(rows, masked, rows)
		}
	}

	// Likewise, masking each 256-slot source channel and rotating it by
	// channel*192 implements the exact upstream channel compaction without a
	// serial chain of sixty-four mask multiplications.
	var channels *ckks.Ciphertext
	for channel := 0; channel < realStage3OutputChannels; channel++ {
		masked := stage3MaskedCiphertext(params, encoder, evaluator, rows, stage3ChannelMask(params.Slots(), channel))
		if channel != 0 {
			masked = evaluator.RotateNew(masked, channel*(realStage3InputBlock-realStage3OutputBlock))
		}
		if channels == nil {
			channels = masked
		} else {
			evaluator.Add(channels, masked, channels)
		}
	}
	channels = evaluator.AddNew(channels, evaluator.RotateNew(channels, -realStage3LogicalSlots))
	return evaluator.AddNew(channels, evaluator.RotateNew(channels, -2*realStage3LogicalSlots))
}

func stage3InputSpatialRotations(evaluator ckks.Evaluator, input *ckks.Ciphertext) [3][3]*ckks.Ciphertext {
	var shifted [3][3]*ckks.Ciphertext
	shifted[1][1] = input
	shifted[0][1] = evaluator.RotateNew(input, -realStage3InputWidth)
	shifted[2][1] = evaluator.RotateNew(input, realStage3InputWidth)
	shifted[1][0] = evaluator.RotateNew(input, -1)
	shifted[1][2] = evaluator.RotateNew(input, 1)
	shifted[0][0] = evaluator.RotateNew(shifted[0][1], -1)
	shifted[0][2] = evaluator.RotateNew(shifted[0][1], 1)
	shifted[2][0] = evaluator.RotateNew(shifted[2][1], -1)
	shifted[2][2] = evaluator.RotateNew(shifted[2][1], 1)
	return shifted
}

func encryptedRealStage3StrideConv(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage3StrideWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage3InputBlock)
			}
			spatial := stage3InputSpatialRotations(evaluator, channelShifted)
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

func encryptedRealStage3Skip(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage3SkipWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage3InputBlock)
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

func stage3SpatialRotations(evaluator ckks.Evaluator, input *ckks.Ciphertext) [3][3]*ckks.Ciphertext {
	var shifted [3][3]*ckks.Ciphertext
	shifted[1][1] = input
	shifted[0][1] = evaluator.RotateNew(input, -realStage3OutputWidth)
	shifted[2][1] = evaluator.RotateNew(input, realStage3OutputWidth)
	shifted[1][0] = evaluator.RotateNew(input, -1)
	shifted[1][2] = evaluator.RotateNew(input, 1)
	shifted[0][0] = evaluator.RotateNew(shifted[0][1], -1)
	shifted[0][2] = evaluator.RotateNew(shifted[0][1], 1)
	shifted[2][0] = evaluator.RotateNew(shifted[2][1], -1)
	shifted[2][2] = evaluator.RotateNew(shifted[2][1], 1)
	return shifted
}

func encryptedRealStage3Conv(params ckks.Parameters, encoder ckks.Encoder, evaluator ckks.Evaluator, public []*rlwe.PublicKey, shares []*rlwe.SecretKey, input *ckks.Ciphertext, weights *realStage3ConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realStage3OutputChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*realStage3OutputBlock)
		}
		spatial := stage3SpatialRotations(evaluator, channelShifted)
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

func stage3RotationKeys() []int {
	rotations := []int{
		-realStage3InputWidth, -1, 1, realStage3InputWidth,
		2, 4, 8,
		-realStage3LogicalSlots, -2 * realStage3LogicalSlots,
		-realStage3OutputWidth, realStage3OutputWidth,
	}
	for row := 1; row < 2*realStage3InputWidth; row++ {
		rotations = append(rotations, row*(realStage3InputWidth+realStage3OutputWidth))
	}
	for channel := 1; channel < realStage3OutputChannels; channel++ {
		rotations = append(rotations, channel*(realStage3InputBlock-realStage3OutputBlock))
	}
	for diagonal := 1; diagonal < realStage3InputChannels; diagonal++ {
		rotations = append(rotations, diagonal*realStage3InputBlock)
	}
	for diagonal := 1; diagonal < realStage3OutputChannels; diagonal++ {
		rotations = append(rotations, diagonal*realStage3OutputBlock)
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

func stage3MaxDifference(left, right []complex128) (float64, float64, error) {
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

func stage3AddPlainBranches(main, skip []complex128) []complex128 {
	output := make([]complex128, len(main))
	for i := range output {
		output[i] = main[i] + skip[i]
	}
	return output
}

func TestEncryptedRealResNet20Stage3Block1TransitionP2TwoParty(t *testing.T) {
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}
	params, err := CKKSParameters("P2")
	if err != nil {
		t.Fatal(err)
	}
	mainWeights, err := loadRealStage3StrideWeights(weightsDirectory, realStage3MainScale, params.Slots())
	if err != nil {
		t.Fatalf("layer7-conv1bn1 audit failed: %v", err)
	}
	skipWeights, err := loadRealStage3SkipWeights(weightsDirectory, realStage3SkipScale, params.Slots())
	if err != nil {
		t.Fatalf("layer7dx-conv1bn1 audit failed: %v", err)
	}
	conv2Weights, err := loadRealStage3ConvWeights(weightsDirectory, realStage3Conv2Scale, params.Slots())
	if err != nil {
		t.Fatalf("layer7-conv2bn2 audit failed: %v", err)
	}
	packedInput, plainInput, err := deterministicRealStage3Input(params.Slots())
	if err != nil {
		t.Fatal(err)
	}

	plainMain := plainRealStage3StrideConv(plainInput, mainWeights)
	plainSkip := plainRealStage3Skip(plainInput, skipWeights)
	plainMainHalves := [2][]complex128{plainSparseStage3Half(plainMain, 0), plainSparseStage3Half(plainMain, 1)}
	plainSkipHalves := [2][]complex128{plainSparseStage3Half(plainSkip, 0), plainSparseStage3Half(plainSkip, 1)}
	packedMainAudit := plainPackStage3Halves(plainMainHalves[0], plainMainHalves[1])
	packedSkipAudit := plainPackStage3Halves(plainSkipHalves[0], plainSkipHalves[1])
	if maximum, _, err := stage3MaxDifference(packedMainAudit, plainMain); err != nil || maximum != 0 {
		t.Fatalf("scalar main packing audit failed: maxDifference=%g err=%v", maximum, err)
	}
	if maximum, _, err := stage3MaxDifference(packedSkipAudit, plainSkip); err != nil || maximum != 0 {
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

	rotations := stage3RotationKeys()
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

	mainHalves, err := encryptedRealStage3StrideConv(params, encoder, evaluator, public, shares, refreshedInput, mainWeights)
	if err != nil {
		t.Fatalf("layer7-conv1bn1 failed: %v", err)
	}
	skipHalves, err := encryptedRealStage3Skip(params, encoder, evaluator, public, shares, refreshedInput, skipWeights)
	if err != nil {
		t.Fatalf("layer7dx-conv1bn1 failed: %v", err)
	}
	mainHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[0], plainMainHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, mainHalves[1], plainMainHalves[1]),
	}
	skipHalfMeasurements := [2]blockBoundaryMeasurement{
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[0], plainSkipHalves[0]),
		measureRealResNetBlockBoundary(params, encoder, aggregate, skipHalves[1], plainSkipHalves[1]),
	}

	packedMain := encryptedPackStage3Halves(params, encoder, evaluator, mainHalves[0], mainHalves[1])
	packedSkip := encryptedPackStage3Halves(params, encoder, evaluator, skipHalves[0], skipHalves[1])
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
	poly1 := referenceResNetBlockReLUPolynomial(realStage3MainScale)
	plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainMain, poly1, realStage3MainScale)
	if err != nil {
		t.Fatalf("first activation plaintext interval failed: %v", err)
	}
	activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshedMain, poly1, params.Scale())
	if err != nil {
		t.Fatalf("first degree-%d activation failed: %v", resnetReferenceReLUDegree, err)
	}
	activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)

	plainConv2 := plainRealStage3Conv(plainActivation1, conv2Weights)
	conv2, err := encryptedRealStage3Conv(params, encoder, evaluator, public, shares, activation1, conv2Weights)
	if err != nil {
		t.Fatalf("layer7-conv2bn2 failed: %v", err)
	}
	conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
	if packedSkip.Level() < conv2.Level() {
		t.Fatalf("skip level %d below main conv2 level %d", packedSkip.Level(), conv2.Level())
	}
	evaluator.DropLevel(packedSkip, packedSkip.Level()-conv2.Level())
	evaluator.Add(conv2, packedSkip, conv2)
	plainResidual := stage3AddPlainBranches(plainConv2, plainSkip)
	residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)

	refreshedResidual, residualRefresh, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
	if err != nil {
		t.Fatalf("post-residual refresh failed: %v", err)
	}
	poly2 := referenceResNetBlockReLUPolynomial(realStage3SkipScale)
	plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, realStage3SkipScale)
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
			t.Fatalf("stage3 transition boundary %s failed: level=%d maxError=%g rmse=%g", boundary.name, boundary.measurement.level, boundary.measurement.maxError, boundary.measurement.rmse)
		}
	}

	elapsed := time.Since(started)
	t.Logf("real ResNet-20 Stage3/Layer3 Block1 transition P2: PASS; sourcePrefixMain=layer7-conv1bn1 sourcePrefixSkip=layer7dx-conv1bn1 sourcePrefixConv2=layer7-conv2bn2 mainScale=%.2f skipScale=%.2f conv2Scale=%.2f dataOwner=0 modelOwner=1 inputCiphertexts=1 encryptedWeightCiphertexts=1216 encryptedBiasCiphertexts=5 ct-ct-products=1216 collectiveRefreshes=3 degree59Activations=2 rotationKeys=%d rotationCalls=1409 publicPackingMaskMultiplications=200 packingRescaleLevels=3 logicalInputPacking=32x16x16x2copies logicalOutputPacking=64x8x8x4copies checkedSlotsPerPackedBoundary=%d", realStage3MainScale, realStage3SkipScale, realStage3Conv2Scale, len(rotations), params.Slots())
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	t.Logf("boundary pre-transition refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", inputRefresh.preDropLevel, inputRefresh.inputLevel, inputRefresh.outputLevel, inputRefresh.logBound, inputRefresh.maxError, inputRefresh.rmse)
	t.Logf("boundary stride-main halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", mainHalfMeasurements[0].level, mainHalfMeasurements[0].maxError, mainHalfMeasurements[0].rmse, mainHalfMeasurements[1].level, mainHalfMeasurements[1].maxError, mainHalfMeasurements[1].rmse)
	t.Logf("boundary stride-skip halves: group0 level=%d maxError=%g rmse=%g; group1 level=%d maxError=%g rmse=%g", skipHalfMeasurements[0].level, skipHalfMeasurements[0].maxError, skipHalfMeasurements[0].rmse, skipHalfMeasurements[1].level, skipHalfMeasurements[1].maxError, skipHalfMeasurements[1].rmse)
	t.Logf("boundary packed-main: level=%d maxError=%g rmse=%g; packed-skip: level=%d maxError=%g rmse=%g", packedMainMeasurement.level, packedMainMeasurement.maxError, packedMainMeasurement.rmse, packedSkipMeasurement.level, packedSkipMeasurement.maxError, packedSkipMeasurement.rmse)
	t.Logf("boundary post-pack main refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", mainRefresh.preDropLevel, mainRefresh.inputLevel, mainRefresh.outputLevel, mainRefresh.logBound, mainRefresh.maxError, mainRefresh.rmse)
	t.Logf("boundary activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage3MainScale, activation1Measurement.level, activation1Measurement.maxError, activation1Measurement.rmse, approximation1)
	t.Logf("boundary conv2(prefix=layer7-conv2bn2 scale=%.2f): level=%d maxError=%g rmse=%g", realStage3Conv2Scale, conv2Measurement.level, conv2Measurement.maxError, conv2Measurement.rmse)
	t.Logf("boundary residual(main+projectedSkip): level=%d maxError=%g rmse=%g", residualMeasurement.level, residualMeasurement.maxError, residualMeasurement.rmse)
	t.Logf("boundary post-residual refresh: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", residualRefresh.preDropLevel, residualRefresh.inputLevel, residualRefresh.outputLevel, residualRefresh.logBound, residualRefresh.maxError, residualRefresh.rmse)
	t.Logf("boundary activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", resnetReferenceReLUDegree, realStage3SkipScale, activation2Measurement.level, activation2Measurement.maxError, activation2Measurement.rmse, approximation2)
	t.Logf("functional-only elapsed=%s; this is a correctness-gate wall time, not a CPU or FPGA benchmark", elapsed)
}
