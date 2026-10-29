package mkfcpu

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// errResNet20MKFCPUBackendNotAudited remains the fail-closed sentinel for a
// missing or identity-mismatched audited backend.  The paper-faithful backend
// is installed from true_mkf_cpu_benchmark_backend_test.go only after the
// n+1 full-network functional gate has passed.
var errResNet20MKFCPUBackendNotAudited = errors.New("paper-faithful MKF ResNet-20 CPU backend is not audited")

// auditedResNet20MKFCPUBackendFactory is a narrow integration seam.  Leaving
// it nil, returning nil, or changing the frozen identity still stops the
// matched benchmark before any paper-facing output is written.
var auditedResNet20MKFCPUBackendFactory resNet20CPUBackendFactory

func newResNet20MKFCPUBackend() (resNet20CPUBackend, error) {
	if auditedResNet20MKFCPUBackendFactory == nil {
		return nil, errResNet20MKFCPUBackendNotAudited
	}
	backend, err := auditedResNet20MKFCPUBackendFactory()
	if err != nil {
		return nil, fmt.Errorf("construct audited paper-faithful MKF backend: %w", err)
	}
	if backend == nil {
		return nil, fmt.Errorf("%w: audited factory returned nil", errResNet20MKFCPUBackendNotAudited)
	}
	if backend.Implementation() != "MKF-CPU" || backend.ConstructionID() != paperMKFCKKSConstructionID || backend.ParameterID() != resNet20CPUParameter || backend.Parties() != resNet20CPUParties {
		return nil, fmt.Errorf("%w: identity mismatch implementation=%q construction=%q parameter=%q parties=%d", errResNet20MKFCPUBackendNotAudited, backend.Implementation(), backend.ConstructionID(), backend.ParameterID(), backend.Parties())
	}
	return backend, nil
}

func init() {
	registerResNet20CPUBackend("MKF-CPU", newResNet20MKFCPUBackend)
}

func TestResNet20MKFCPUBackendRegistrationUsesAuditedPaperPath(t *testing.T) {
	factory, ok := resNet20CPUBackendFactories["MKF-CPU"]
	if !ok {
		t.Fatal("MKF-CPU factory is not registered")
	}
	backend, err := factory()
	if err != nil || backend == nil {
		t.Fatalf("audited MKF backend is unavailable: backend=%T err=%v", backend, err)
	}
	if backend.ConstructionID() != paperMKFCKKSConstructionID || backend.ParameterID() != "P2" || backend.Parties() != 2 {
		t.Fatalf("unexpected audited MKF identity: construction=%q parameter=%q parties=%d", backend.ConstructionID(), backend.ParameterID(), backend.Parties())
	}
	for _, required := range []string{"online-only", "offline", "22 two-party collective refresh"} {
		if !strings.Contains(backend.MeasurementBoundary(), required) {
			t.Fatalf("audited MKF boundary omits %q: %s", required, backend.MeasurementBoundary())
		}
	}
}

func TestResNet20MKFCPUBackendRejectsIdentityMismatch(t *testing.T) {
	original := auditedResNet20MKFCPUBackendFactory
	auditedResNet20MKFCPUBackendFactory = func() (resNet20CPUBackend, error) {
		return &fakeResNet20CPUBackend{implementation: "not-MKF"}, nil
	}
	defer func() { auditedResNet20MKFCPUBackendFactory = original }()
	if _, err := newResNet20MKFCPUBackend(); !errors.Is(err, errResNet20MKFCPUBackendNotAudited) {
		t.Fatalf("identity-mismatched backend was accepted: %v", err)
	}
}
