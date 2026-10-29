package mkfcpu

// This file mirrors only the encrypted-weight convolution helpers needed by
// the frozen ResNet-20 schedule.  Unlike the compatibility helpers, model
// ciphertexts remain sparse n+1 owner ciphertexts: there is no CKS ingress or
// aggregate-key single-secret representation before MulRelin.

import (
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// fullResNet20OwnerEvaluator is the common encrypted-weight interface for
// the proposed-MKF and KKLSS adapters. Embedding ckks.Evaluator preserves all
// public linear packing operations; ownerCarrier is the only construction-
// specific ingress operation.
type fullResNet20OwnerEvaluator interface {
	ckks.Evaluator
	ownerCarrier(ckks.Encoder, []*rlwe.PublicKey, int, []complex128) (*ckks.Ciphertext, error)
}

func paperMKFEncryptedRealResNetBlockConv(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realResNetBlockConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realResNetBlockChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*resnetChannelSlotSize)
		}
		spatial := resNetBlockSpatialRotations(evaluator, channelShifted)
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[diagonal][kernelY][kernelX])
				if err != nil {
					return nil, err
				}
				term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
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
	bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias)
	if err != nil {
		return nil, err
	}
	evaluator.DropLevel(bias, bias.Level()-sum.Level())
	evaluator.Add(sum, bias, sum)
	return sum, nil
}

func paperMKFEncryptedRealStage2StrideConv(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage2StrideWeights) ([2]*ckks.Ciphertext, error) {
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
					weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[group][diagonal][kernelY][kernelX])
					if err != nil {
						return output, err
					}
					term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
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
		bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(bias, bias.Level()-sum.Level())
		evaluator.Add(sum, bias, sum)
		output[group] = sum
	}
	return output, nil
}

func paperMKFEncryptedRealStage2Skip(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage2SkipWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage2InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage2InputBlock)
			}
			weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[group][diagonal])
			if err != nil {
				return output, err
			}
			term := evaluator.MulRelinNew(channelShifted, weight)
			if sum == nil {
				sum = term
			} else {
				evaluator.Add(sum, term, sum)
			}
		}
		if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
			return output, err
		}
		bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(bias, bias.Level()-sum.Level())
		evaluator.Add(sum, bias, sum)
		output[group] = sum
	}
	return output, nil
}

func paperMKFEncryptedRealStage2Conv(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage2ConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realStage2OutputChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*realStage2OutputBlock)
		}
		spatial := stage2SpatialRotations(evaluator, channelShifted)
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[diagonal][kernelY][kernelX])
				if err != nil {
					return nil, err
				}
				term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
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
	bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias)
	if err != nil {
		return nil, err
	}
	evaluator.DropLevel(bias, bias.Level()-sum.Level())
	evaluator.Add(sum, bias, sum)
	return sum, nil
}

func paperMKFEncryptedRealStage3StrideConv(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage3StrideWeights) ([2]*ckks.Ciphertext, error) {
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
					weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[group][diagonal][kernelY][kernelX])
					if err != nil {
						return output, err
					}
					term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
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
		bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(bias, bias.Level()-sum.Level())
		evaluator.Add(sum, bias, sum)
		output[group] = sum
	}
	return output, nil
}

func paperMKFEncryptedRealStage3Skip(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage3SkipWeights) ([2]*ckks.Ciphertext, error) {
	var output [2]*ckks.Ciphertext
	for group := 0; group < 2; group++ {
		var sum *ckks.Ciphertext
		for diagonal := 0; diagonal < realStage3InputChannels; diagonal++ {
			channelShifted := input
			if diagonal != 0 {
				channelShifted = evaluator.RotateNew(input, diagonal*realStage3InputBlock)
			}
			weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[group][diagonal])
			if err != nil {
				return output, err
			}
			term := evaluator.MulRelinNew(channelShifted, weight)
			if sum == nil {
				sum = term
			} else {
				evaluator.Add(sum, term, sum)
			}
		}
		if err := evaluator.Rescale(sum, params.Scale(), sum); err != nil {
			return output, err
		}
		bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias[group])
		if err != nil {
			return output, err
		}
		evaluator.DropLevel(bias, bias.Level()-sum.Level())
		evaluator.Add(sum, bias, sum)
		output[group] = sum
	}
	return output, nil
}

func paperMKFEncryptedRealStage3Conv(params ckks.Parameters, encoder ckks.Encoder, evaluator fullResNet20OwnerEvaluator, public []*rlwe.PublicKey, input *ckks.Ciphertext, weights *realStage3ConvWeights) (*ckks.Ciphertext, error) {
	var sum *ckks.Ciphertext
	for diagonal := 0; diagonal < realStage3OutputChannels; diagonal++ {
		channelShifted := input
		if diagonal != 0 {
			channelShifted = evaluator.RotateNew(input, diagonal*realStage3OutputBlock)
		}
		spatial := stage3SpatialRotations(evaluator, channelShifted)
		for kernelY := 0; kernelY < 3; kernelY++ {
			for kernelX := 0; kernelX < 3; kernelX++ {
				weight, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.masks[diagonal][kernelY][kernelX])
				if err != nil {
					return nil, err
				}
				term := evaluator.MulRelinNew(spatial[kernelY][kernelX], weight)
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
	bias, err := evaluator.ownerCarrier(encoder, public, realResNetTailModelOwner, weights.packedBias)
	if err != nil {
		return nil, err
	}
	evaluator.DropLevel(bias, bias.Level()-sum.Level())
	evaluator.Add(sum, bias, sum)
	return sum, nil
}
