package mkfcpu

// The full-network hardware boundary exporter is outside this P1 HLS
// artifact. The default functional software path remains available; an
// explicit export request fails closed instead of producing an incompatible
// payload.

import (
	"fmt"
	"os"

	"github.com/ldsec/lattigo/v2/ckks"
)

const p2gn16BoundaryExportEnv = "MKF_P2GN16_FPGA_BOUNDARY_EXPORT"

type p2gn16BoundaryExporter struct{}

type p2gn16BoundaryExportConfiguration struct {
	tracePath  string
	sourceRoot string
	outputRoot string
}

func p2gn16BoundaryExportConfigurationFor(_ fullResNet20Request) (*p2gn16BoundaryExportConfiguration, error) {
	if os.Getenv(p2gn16BoundaryExportEnv) != "" {
		return nil, fmt.Errorf("%s is not supported by the U55C P1 HLS artifact", p2gn16BoundaryExportEnv)
	}
	return nil, nil
}

func newP2GN16BoundaryExporter(_, _, _ string) (*p2gn16BoundaryExporter, error) {
	return nil, fmt.Errorf("full-network hardware boundary export is not included in the U55C P1 HLS artifact")
}

func (*p2gn16BoundaryExporter) observe(_ string, _ []*ckks.Ciphertext, _ []float64) error {
	return fmt.Errorf("full-network hardware boundary export is not included in the U55C P1 HLS artifact")
}

func (*p2gn16BoundaryExporter) finalize() error {
	return fmt.Errorf("full-network hardware boundary export is not included in the U55C P1 HLS artifact")
}
