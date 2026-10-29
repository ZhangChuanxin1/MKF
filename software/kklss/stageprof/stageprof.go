// Package stageprof records mutually exclusive online MulRelinNew stages.
// It is enabled only by the single-threaded profiling tests.
package stageprof

import "time"

type Stage int

const (
	Tensor Stage = iota
	Decomposition
	ExternalProduct
	ModulusManagement
	MemoryMovement
	StageCount
)

type Sample struct {
	Durations [StageCount]time.Duration
}

var active *Sample

func Activate(sample *Sample) { active = sample }
func Deactivate()             { active = nil }

func Start() time.Time {
	if active == nil {
		return time.Time{}
	}
	return time.Now()
}

func Add(stage Stage, start time.Time) {
	if active != nil && !start.IsZero() {
		active.Durations[stage] += time.Since(start)
	}
}
