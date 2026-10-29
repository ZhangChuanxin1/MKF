package mkbfv

import (
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"mk-lattigo/mkrlwe"
	"mk-lattigo/stageprof"
)

var profileStageSinkBFV *Ciphertext

func TestProfileStagesExactP1N16(t *testing.T) {
	const parties = 16
	params := NewParametersFromLiteral(PN14QP439)
	idset := mkrlwe.NewIDSet()
	users := make([]string, parties)
	for i := range users {
		users[i] = "user" + strconv.Itoa(i)
		idset.Add(users[i])
	}
	context, err := genTestParams(params, idset)
	if err != nil {
		t.Fatal(err)
	}
	cts := make([]*Ciphertext, parties)
	for i, user := range users {
		_, cts[i] = newTestVectors(context, user, 0, 2)
	}
	ct := cts[0]
	for i := range users {
		ct = context.evaluator.AddNew(ct, cts[i])
	}

	for i := 0; i < 5; i++ {
		profileStageSinkBFV = context.evaluator.MulRelinNew(ct, ct, context.rlkSet)
	}

	records := [][]string{{
		"scheme", "param", "n", "trial", "total_ns", "tensor_ns",
		"decomposition_ns", "external_product_ns", "modulus_management_ns",
		"memory_movement_ns", "other_ns",
	}}
	for trial := 0; trial < 10; trial++ {
		runtime.GC()
		sample := new(stageprof.Sample)
		stageprof.Activate(sample)
		start := time.Now()
		profileStageSinkBFV = context.evaluator.MulRelinNew(ct, ct, context.rlkSet)
		total := time.Since(start)
		stageprof.Deactivate()
		runtime.KeepAlive(profileStageSinkBFV)

		accounted := time.Duration(0)
		for _, duration := range sample.Durations {
			accounted += duration
		}
		if accounted > total {
			t.Fatalf("overlapping BFV timers: accounted=%s total=%s", accounted, total)
		}
		other := total - accounted
		record := []string{
			"KKLSS-BFV", "P1", "16", strconv.Itoa(trial),
			strconv.FormatInt(total.Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.Tensor].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.Decomposition].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.ExternalProduct].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.ModulusManagement].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.MemoryMovement].Nanoseconds(), 10),
			strconv.FormatInt(other.Nanoseconds(), 10),
		}
		records = append(records, record)
		fmt.Println("STAGEPROFILE," + joinBFVStageFields(record))
	}
	writeBFVStageRecords(t, records)
}

func joinBFVStageFields(fields []string) string {
	result := ""
	for i, field := range fields {
		if i != 0 {
			result += ","
		}
		result += field
	}
	return result
}

func writeBFVStageRecords(t *testing.T, records [][]string) {
	t.Helper()
	path := os.Getenv("KKLSS_STAGE_OUT")
	if path == "" {
		return
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := csv.NewWriter(file)
	if err := writer.WriteAll(records); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
