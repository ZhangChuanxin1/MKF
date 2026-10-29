package mkfcpu

import "time"

// Profile uses mutually exclusive wall-clock regions. Total is the complete
// online Mult.+Relin. boundary. Key generation, encryption and verification
// are deliberately outside this boundary.
type Profile struct {
	Tensor        time.Duration
	Decomposition time.Duration
	External      time.Duration
	Modulus       time.Duration
	Memory        time.Duration
	Other         time.Duration
	Total         time.Duration

	Decompositions       int
	ExternalProductPolys int
}

func (p *Profile) finish(start time.Time) {
	p.Total = time.Since(start)
	accounted := p.Tensor + p.Decomposition + p.External + p.Modulus + p.Memory
	if accounted < p.Total {
		p.Other = p.Total - accounted
	}
}

func measure(dst *time.Duration, f func()) {
	start := time.Now()
	f()
	*dst += time.Since(start)
}
