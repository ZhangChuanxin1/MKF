package mkfcpu

// The production P2GN16 key-payload exporter is intentionally outside this
// P1 primitive/HLS artifact. Normal software execution remains unchanged, but
// an explicit request for the excluded exporter fails closed.

import (
	"fmt"
	"os"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/rlwe"
)

const p2gn16ProductionExportEnv = "MKF_P2GN16_PRODUCTION_KEY_EXPORT"

func p2gn16ProductionExportUnavailable() error {
	if os.Getenv(p2gn16ProductionExportEnv) != "" {
		return fmt.Errorf("%s is not supported by the U55C P1 HLS artifact", p2gn16ProductionExportEnv)
	}
	return nil
}

func stageP2GN16ProductionFoldKeysSameRun(_ ckks.Parameters, _ []*rlwe.SwitchingKey) error {
	return p2gn16ProductionExportUnavailable()
}

func maybeExportP2GN16ProductionKeysSameRun(_ ckks.Parameters, _ []int, _ rlwe.EvaluationKey) error {
	return p2gn16ProductionExportUnavailable()
}
