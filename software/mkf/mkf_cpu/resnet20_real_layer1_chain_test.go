package mkfcpu

import (
	"os"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// realResNetLayer1BlockConfig follows LowMemoryFHEResNet20 src/main.cpp
// lines 416--475.  The upstream program numbers the three 16-channel
// residual blocks as weight prefixes layer1, layer2, and layer3.
type realResNetLayer1BlockConfig struct {
	name        string
	conv1Prefix string
	conv1Scale  float64
	conv2Prefix string
	conv2Scale  float64
}

var realResNetLayer1BlockConfigs = []realResNetLayer1BlockConfig{
	{name: "block1", conv1Prefix: "layer1-conv1bn1", conv1Scale: 1.00, conv2Prefix: "layer1-conv2bn2", conv2Scale: 0.52},
	{name: "block2", conv1Prefix: "layer2-conv1bn1", conv1Scale: 0.55, conv2Prefix: "layer2-conv2bn2", conv2Scale: 0.36},
	{name: "block3", conv1Prefix: "layer3-conv1bn1", conv1Scale: 0.63, conv2Prefix: "layer3-conv2bn2", conv2Scale: 0.42},
}

type realResNetLayer1BlockTrace struct {
	config         realResNetLayer1BlockConfig
	conv1          blockBoundaryMeasurement
	refresh1       blockRefreshMeasurement
	activation1    blockBoundaryMeasurement
	approximation1 float64
	conv2          blockBoundaryMeasurement
	residual       blockBoundaryMeasurement
	refresh2       blockRefreshMeasurement
	activation2    blockBoundaryMeasurement
	approximation2 float64
}

func runEncryptedRealResNet20Layer1Chain(t *testing.T, parameter string) {
	t.Helper()
	weightsDirectory := os.Getenv("MKF_RESNET20_WEIGHTS_DIR")
	if weightsDirectory == "" {
		t.Skip("set MKF_RESNET20_WEIGHTS_DIR to the frozen LowMemoryFHEResNet20 weights directory")
	}

	params, err := CKKSParameters(parameter)
	if err != nil {
		t.Fatal(err)
	}

	conv1Weights := make([]*realResNetBlockConvWeights, len(realResNetLayer1BlockConfigs))
	conv2Weights := make([]*realResNetBlockConvWeights, len(realResNetLayer1BlockConfigs))
	for blockIndex, config := range realResNetLayer1BlockConfigs {
		conv1Weights[blockIndex], err = loadRealResNetBlockConvWeights(weightsDirectory, config.conv1Prefix, config.conv1Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv1 weight audit failed: %v", config.name, err)
		}
		conv2Weights[blockIndex], err = loadRealResNetBlockConvWeights(weightsDirectory, config.conv2Prefix, config.conv2Scale, params.Slots())
		if err != nil {
			t.Fatalf("%s conv2 weight audit failed: %v", config.name, err)
		}
	}

	packedInput, plainInput, err := deterministicRealResNetBlockInput(params.Slots())
	if err != nil {
		t.Fatal(err)
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

	rotations := realResNetBlockRotationKeys()
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
		t.Fatalf("%s lacks modulus levels for two-party collective refresh", parameter)
	}
	refreshCRS, err := utils.NewPRNG()
	if err != nil {
		t.Fatal(err)
	}
	aggregate := aggregateSecret(params.Parameters, shares)

	started := time.Now()
	inputMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, input, plainInput)
	currentCiphertext := input
	currentPlaintext := plainInput
	traces := make([]realResNetLayer1BlockTrace, 0, len(realResNetLayer1BlockConfigs))

	for blockIndex, config := range realResNetLayer1BlockConfigs {
		skipCiphertext := currentCiphertext
		skipPlaintext := currentPlaintext

		plainConv1 := plainRealResNetBlockConv(currentPlaintext, conv1Weights[blockIndex])
		conv1, err := encryptedRealResNetBlockConv(params, encoder, evaluator, public, shares, currentCiphertext, conv1Weights[blockIndex])
		if err != nil {
			t.Fatalf("%s conv1 failed: %v", config.name, err)
		}
		conv1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv1, plainConv1)
		refreshed1, refresh1Measurement, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv1, minLevel, logBound, refreshCRS, encoder, aggregate, plainConv1)
		if err != nil {
			t.Fatalf("%s refresh1 failed: %v", config.name, err)
		}

		poly1 := referenceResNetBlockReLUPolynomial(config.conv1Scale)
		if poly1.Depth() != 6 {
			t.Fatalf("%s unexpected first degree-%d Chebyshev depth %d", config.name, resnetReferenceReLUDegree, poly1.Depth())
		}
		plainActivation1, approximation1, err := plainReferenceResNetBlockReLU(plainConv1, poly1, config.conv1Scale)
		if err != nil {
			t.Fatalf("%s activation1 plaintext gate failed: %v", config.name, err)
		}
		activation1, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed1, poly1, params.Scale())
		if err != nil {
			t.Fatalf("%s activation1 failed: %v", config.name, err)
		}
		activation1Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation1, plainActivation1)

		plainConv2 := plainRealResNetBlockConv(plainActivation1, conv2Weights[blockIndex])
		conv2, err := encryptedRealResNetBlockConv(params, encoder, evaluator, public, shares, activation1, conv2Weights[blockIndex])
		if err != nil {
			t.Fatalf("%s conv2 failed: %v", config.name, err)
		}
		conv2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainConv2)
		plainResidual := addPlainRealResNetBlockSkip(plainConv2, skipPlaintext, config.conv2Scale)
		if err := addScaledRealResNetBlockSkip(params, evaluator, conv2, skipCiphertext, config.conv2Scale); err != nil {
			t.Fatalf("%s residual addition failed: %v", config.name, err)
		}
		residualMeasurement := measureRealResNetBlockBoundary(params, encoder, aggregate, conv2, plainResidual)
		refreshed2, refresh2Measurement, err := refreshRealResNetBlockBoundary(params, evaluator, shares, conv2, minLevel, logBound, refreshCRS, encoder, aggregate, plainResidual)
		if err != nil {
			t.Fatalf("%s refresh2 failed: %v", config.name, err)
		}

		poly2 := referenceResNetBlockReLUPolynomial(config.conv2Scale)
		if poly2.Depth() != 6 {
			t.Fatalf("%s unexpected second degree-%d Chebyshev depth %d", config.name, resnetReferenceReLUDegree, poly2.Depth())
		}
		plainActivation2, approximation2, err := plainReferenceResNetBlockReLU(plainResidual, poly2, config.conv2Scale)
		if err != nil {
			t.Fatalf("%s activation2 plaintext gate failed: %v", config.name, err)
		}
		activation2, err := encryptedReferenceResNetBlockReLU(evaluator, refreshed2, poly2, params.Scale())
		if err != nil {
			t.Fatalf("%s activation2 failed: %v", config.name, err)
		}
		activation2Measurement := measureRealResNetBlockBoundary(params, encoder, aggregate, activation2, plainActivation2)

		trace := realResNetLayer1BlockTrace{
			config:         config,
			conv1:          conv1Measurement,
			refresh1:       refresh1Measurement,
			activation1:    activation1Measurement,
			approximation1: approximation1,
			conv2:          conv2Measurement,
			residual:       residualMeasurement,
			refresh2:       refresh2Measurement,
			activation2:    activation2Measurement,
			approximation2: approximation2,
		}
		for boundary, measurement := range map[string]blockBoundaryMeasurement{
			"conv1":       trace.conv1,
			"refresh1":    {level: trace.refresh1.outputLevel, maxError: trace.refresh1.maxError, rmse: trace.refresh1.rmse},
			"activation1": trace.activation1,
			"conv2":       trace.conv2,
			"residual":    trace.residual,
			"refresh2":    {level: trace.refresh2.outputLevel, maxError: trace.refresh2.maxError, rmse: trace.refresh2.rmse},
			"activation2": trace.activation2,
		} {
			if measurement.maxError >= 1e-3 {
				t.Fatalf("real Layer1 %s boundary %s failed: level=%d maxError=%g rmse=%g", config.name, boundary, measurement.level, measurement.maxError, measurement.rmse)
			}
		}
		traces = append(traces, trace)
		currentCiphertext = activation2
		currentPlaintext = plainActivation2
	}

	elapsed := time.Since(started)
	t.Logf("real folded ResNet-20 complete Layer1 chain %s: PASS; blocks=3 dataOwner=0 modelOwner=1 packedInputCiphertexts=1 encryptedWeightCiphertexts=864 encryptedBiasCiphertexts=6 channelDiagonalsPerConv=16 spatialOffsetsPerDiagonal=9 rotationKeys=%d rotationCallsPerConv=143 rotationCallsTotal=858 ct-ct-productsPerConv=144 ct-ct-productsTotal=864 refreshCount=6 activationCount=6 activationDegree=%d packingCopies=%d checkedSlotsPerBoundary=%d", parameter, len(rotations), resnetReferenceReLUDegree, params.Slots()/realResNetStemActiveSlots, len(plainInput))
	t.Logf("weight schedule audit: block1=(%s,%.2f;%s,%.2f) block2=(%s,%.2f;%s,%.2f) block3=(%s,%.2f;%s,%.2f)", realResNetLayer1BlockConfigs[0].conv1Prefix, realResNetLayer1BlockConfigs[0].conv1Scale, realResNetLayer1BlockConfigs[0].conv2Prefix, realResNetLayer1BlockConfigs[0].conv2Scale, realResNetLayer1BlockConfigs[1].conv1Prefix, realResNetLayer1BlockConfigs[1].conv1Scale, realResNetLayer1BlockConfigs[1].conv2Prefix, realResNetLayer1BlockConfigs[1].conv2Scale, realResNetLayer1BlockConfigs[2].conv1Prefix, realResNetLayer1BlockConfigs[2].conv1Scale, realResNetLayer1BlockConfigs[2].conv2Prefix, realResNetLayer1BlockConfigs[2].conv2Scale)
	t.Logf("boundary input: level=%d maxError=%g rmse=%g", inputMeasurement.level, inputMeasurement.maxError, inputMeasurement.rmse)
	for _, trace := range traces {
		t.Logf("boundary %s conv1(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv1Prefix, trace.config.conv1Scale, trace.conv1.level, trace.conv1.maxError, trace.conv1.rmse)
		t.Logf("boundary %s refresh1: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh1.preDropLevel, trace.refresh1.inputLevel, trace.refresh1.outputLevel, trace.refresh1.logBound, trace.refresh1.maxError, trace.refresh1.rmse)
		t.Logf("boundary %s activation1(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv1Scale, trace.activation1.level, trace.activation1.maxError, trace.activation1.rmse, trace.approximation1)
		t.Logf("boundary %s conv2(prefix=%s scale=%.2f): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv2Prefix, trace.config.conv2Scale, trace.conv2.level, trace.conv2.maxError, trace.conv2.rmse)
		t.Logf("boundary %s residual(+%.2f*skip): level=%d maxError=%g rmse=%g", trace.config.name, trace.config.conv2Scale, trace.residual.level, trace.residual.maxError, trace.residual.rmse)
		t.Logf("boundary %s refresh2: levels preDrop=%d input=%d output=%d logBound=%d maxError=%g rmse=%g", trace.config.name, trace.refresh2.preDropLevel, trace.refresh2.inputLevel, trace.refresh2.outputLevel, trace.refresh2.logBound, trace.refresh2.maxError, trace.refresh2.rmse)
		t.Logf("boundary %s activation2(degree=%d scale=%.2f): level=%d maxFHEError=%g rmse=%g maxApproxErrorVsReLU=%g", trace.config.name, resnetReferenceReLUDegree, trace.config.conv2Scale, trace.activation2.level, trace.activation2.maxError, trace.activation2.rmse, trace.approximation2)
	}
	t.Logf("functional-only elapsed=%s; this is a correctness-gate wall time, not a CPU or FPGA benchmark", elapsed)
}

func TestEncryptedRealResNet20CompleteLayer1ChainP2TwoParty(t *testing.T) {
	runEncryptedRealResNet20Layer1Chain(t, "P2")
}

func TestEncryptedRealResNet20CompleteLayer1ChainP3IRCandidateTwoParty(t *testing.T) {
	if os.Getenv("MKF_RUN_P3_REAL_LAYER1_CHAIN") != "1" {
		t.Skip("set MKF_RUN_P3_REAL_LAYER1_CHAIN=1 to run the real P3 ResNet-20 complete Layer1-chain functional gate")
	}
	runEncryptedRealResNet20Layer1Chain(t, "P3IRCandidate")
}
