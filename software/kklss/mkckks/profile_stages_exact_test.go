package mkckks

import (
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
	"mk-lattigo/mkrlwe"
	"mk-lattigo/stageprof"
)

var profileStageSinkCKKS *Ciphertext

func TestProfileStagesExactP1N16(t *testing.T) {
	const parties = 16
	ckksParams, err := ckks.NewParametersFromLiteral(PN14QP439)
	if err != nil {
		t.Fatal(err)
	}
	params := NewParameters(ckksParams)
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
		_, cts[i] = newTestVectors(context, user, complex(-1, 1), complex(-1, 1))
	}
	ct0, ct1 := cts[0], cts[0]
	for i := range users {
		ct0 = context.evaluator.AddNew(ct0, cts[i])
		ct1 = context.evaluator.SubNew(ct1, cts[i])
	}

	for i := 0; i < 5; i++ {
		profileStageSinkCKKS = context.evaluator.MulRelinNew(ct0, ct1, context.rlkSet)
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
		profileStageSinkCKKS = context.evaluator.MulRelinNew(ct0, ct1, context.rlkSet)
		total := time.Since(start)
		stageprof.Deactivate()
		runtime.KeepAlive(profileStageSinkCKKS)

		accounted := time.Duration(0)
		for _, duration := range sample.Durations {
			accounted += duration
		}
		if accounted > total {
			t.Fatalf("overlapping CKKS timers: accounted=%s total=%s", accounted, total)
		}
		other := total - accounted
		record := []string{
			"KKLSS-CKKS", "P1", "16", strconv.Itoa(trial),
			strconv.FormatInt(total.Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.Tensor].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.Decomposition].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.ExternalProduct].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.ModulusManagement].Nanoseconds(), 10),
			strconv.FormatInt(sample.Durations[stageprof.MemoryMovement].Nanoseconds(), 10),
			strconv.FormatInt(other.Nanoseconds(), 10),
		}
		records = append(records, record)
		fmt.Println("STAGEPROFILE," + joinStageFields(record))
	}
	writeCKKSStageRecords(t, records)
}

func joinStageFields(fields []string) string {
	result := ""
	for i, field := range fields {
		if i != 0 {
			result += ","
		}
		result += field
	}
	return result
}

func writeCKKSStageRecords(t *testing.T, records [][]string) {
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
