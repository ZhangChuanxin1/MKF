#pragma once

#include <ap_int.h>
#include <hls_stream.h>

// Mutually exclusive top-level states are required for a valid cycle
// breakdown. Sub-block counters may overlap internally, but only one category
// below is charged for each wall-clock cycle.
enum class CycleClass : unsigned char {
    Idle = 0,
    Decomposition = 1,
    ExternalProduct = 2,
    NttIntt = 3,
    ModUpDown = 4,
    HbmStall = 5,
    Other = 6,
    Tensor = 7
};

enum class TimingClass : unsigned char {
    Idle = 0, Tensor = 1, Decomposition = 2,
    ExternalProduct = 3, ModUpDown = 4, Other = 5
};

struct CycleCounters {
    ap_uint<64> decomposition;
    ap_uint<64> external_product;
    ap_uint<64> ntt_intt;
    ap_uint<64> modup_down;
    ap_uint<64> hbm_stall;
    ap_uint<64> other;
    ap_uint<64> total;
};

static inline void emit_phase_event(hls::stream<ap_uint<6>> &events,
                                    CycleClass stage, TimingClass timing) {
#pragma HLS INLINE
    ap_uint<6> event = 0;
    event.range(5, 3) = static_cast<unsigned>(stage);
    event.range(2, 0) = static_cast<unsigned>(timing);
    events.write(event);
}

static inline void reset_counters(CycleCounters &c) {
#pragma HLS INLINE
    c.decomposition = 0;
    c.external_product = 0;
    c.ntt_intt = 0;
    c.modup_down = 0;
    c.hbm_stall = 0;
    c.other = 0;
    c.total = 0;
}

static inline void count_cycle(CycleCounters &c, CycleClass cls, bool active) {
#pragma HLS INLINE
    if (!active) return;
    ++c.total;
    switch (cls) {
    case CycleClass::Decomposition: ++c.decomposition; break;
    case CycleClass::ExternalProduct: ++c.external_product; break;
    case CycleClass::NttIntt: ++c.ntt_intt; break;
    case CycleClass::ModUpDown: ++c.modup_down; break;
    case CycleClass::HbmStall: ++c.hbm_stall; break;
    case CycleClass::Other: ++c.other; break;
    case CycleClass::Tensor: ++c.other; break;
    default: break;
    }
}

static inline void set_timing_class(CycleCounters &c, TimingClass cls) {
#pragma HLS INLINE
    // The physical stage/timing stream is emitted explicitly at algorithmic
    // phase boundaries.  Retain this call site for the HLS operation counter
    // without making a pointer-side-channel that HLS can optimize away.
    (void)c;
    (void)cls;
}

static inline void finish_counters(CycleCounters &c) {
#pragma HLS INLINE
    (void)c;
}

static inline bool counters_are_consistent(const CycleCounters &c) {
#pragma HLS INLINE
    return c.total == c.decomposition + c.external_product + c.ntt_intt +
                          c.modup_down + c.hbm_stall + c.other;
}
