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
)

var profileSinkCKKS *Ciphertext

func TestProfileExactP1N16(t *testing.T) {
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
		profileSinkCKKS = context.evaluator.MulRelinNew(ct0, ct1, context.rlkSet)
	}
	records := [][]string{{"scheme", "parameter", "parties", "trial", "elapsed_ns"}}
	for trial := 0; trial < 10; trial++ {
		runtime.GC()
		start := time.Now()
		profileSinkCKKS = context.evaluator.MulRelinNew(ct0, ct1, context.rlkSet)
		elapsed := time.Since(start)
		runtime.KeepAlive(profileSinkCKKS)
		fmt.Printf("PROFILE,KKLSS-CKKS,P1,16,%d,%d\n",
			trial, elapsed.Nanoseconds())
		records = append(records, []string{
			"KKLSS-CKKS", "P1", "16", strconv.Itoa(trial),
			strconv.FormatInt(elapsed.Nanoseconds(), 10),
		})
	}
	writeCKKSTotalRecords(t, records)
}

func writeCKKSTotalRecords(t *testing.T, records [][]string) {
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
