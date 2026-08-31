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
)

var profileSinkBFV *Ciphertext

func TestProfileExactP1N16(t *testing.T) {
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
		profileSinkBFV = context.evaluator.MulRelinNew(ct, ct, context.rlkSet)
	}
	records := [][]string{{"scheme", "parameter", "parties", "trial", "elapsed_ns"}}
	for trial := 0; trial < 10; trial++ {
		runtime.GC()
		start := time.Now()
		profileSinkBFV = context.evaluator.MulRelinNew(ct, ct, context.rlkSet)
		elapsed := time.Since(start)
		runtime.KeepAlive(profileSinkBFV)
		fmt.Printf("PROFILE,KKLSS-BFV,P1,16,%d,%d\n",
			trial, elapsed.Nanoseconds())
		records = append(records, []string{
			"KKLSS-BFV", "P1", "16", strconv.Itoa(trial),
			strconv.FormatInt(elapsed.Nanoseconds(), 10),
		})
	}
	writeBFVTotalRecords(t, records)
}

func writeBFVTotalRecords(t *testing.T, records [][]string) {
	t.Helper()
	path := os.Getenv("KKLSS_TOTAL_OUT")
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
