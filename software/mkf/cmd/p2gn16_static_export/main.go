package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"mkf-cpu-experiments/internal/p2gn16staticexport"
)

func main() {
	var sourceRoot, outputRoot string
	flag.StringVar(&sourceRoot, "source-root", "", "absolute root of the frozen experiment source tree")
	flag.StringVar(&outputRoot, "output-root", "", "new output directory; it must not already exist")
	flag.Parse()
	if flag.NArg() != 0 || sourceRoot == "" || outputRoot == "" {
		fmt.Fprintln(os.Stderr, "usage: p2gn16_static_export --source-root ABS --output-root ABS")
		os.Exit(2)
	}
	var err error
	sourceRoot, err = filepath.Abs(sourceRoot)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	outputRoot, err = filepath.Abs(outputRoot)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	manifest, err := p2gn16staticexport.Run(p2gn16staticexport.ProductionConfig(sourceRoot, outputRoot))
	if err != nil {
		fmt.Fprintf(os.Stderr, "P2GN16_STATIC_EXPORT_FAIL: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("P2GN16_STATIC_EXPORT_PASS output=%s references=%d dense=%d residual=%d transcript_sha256=%s board_eligible=false performance_measured=false\n",
		outputRoot, manifest.Coverage.References, manifest.Coverage.DenseMaskReferences,
		manifest.Coverage.FractionalMultReferences, manifest.EncodingTranscript.SHA256)
}
