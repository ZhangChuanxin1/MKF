package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"mkf-cpu-experiments/internal/p2gn16staticvalidate"
)

func main() {
	var sourceRoot, exportRoot, receiptPath string
	flag.StringVar(&sourceRoot, "source-root", "", "absolute root of the frozen experiment source tree")
	flag.StringVar(&exportRoot, "export-root", "", "completed static export directory")
	flag.StringVar(&receiptPath, "receipt", "", "new validation receipt path; it must not already exist")
	flag.Parse()
	if flag.NArg() != 0 || sourceRoot == "" || exportRoot == "" || receiptPath == "" {
		fmt.Fprintln(os.Stderr, "usage: p2gn16_static_validate --source-root ABS --export-root ABS --receipt ABS")
		os.Exit(2)
	}
	var err error
	if sourceRoot, err = filepath.Abs(sourceRoot); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if exportRoot, err = filepath.Abs(exportRoot); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if receiptPath, err = filepath.Abs(receiptPath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	receipt, err := p2gn16staticvalidate.Validate(p2gn16staticvalidate.ProductionConfig(sourceRoot, exportRoot, receiptPath))
	if err != nil {
		fmt.Fprintf(os.Stderr, "P2GN16_STATIC_REENCODING_FAIL: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("P2GN16_STATIC_REENCODING_PASS receipt=%s references=%d unique=%d transcript_sha256=%s board_eligible=false performance_measured=false\n",
		receiptPath, receipt.Validation.ReferencesReencoded,
		receipt.Validation.UniqueObjectsReencoded, receipt.Transcript.SHA256)
}
