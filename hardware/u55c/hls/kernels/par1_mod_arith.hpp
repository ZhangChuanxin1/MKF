#pragma once

#include <ap_int.h>

// Exact arithmetic for the 51--60 bit Par1 RNS primes.  The 128-bit
// intermediate prevents the overflow present in the old timing-only model.
static inline ap_uint<64> add_mod(ap_uint<64> a, ap_uint<64> b,
                                  ap_uint<64> q) {
#pragma HLS INLINE
    ap_uint<65> sum = (ap_uint<65>)a + b;
    return sum >= q ? (ap_uint<64>)(sum - q) : (ap_uint<64>)sum;
}

static inline ap_uint<64> sub_mod(ap_uint<64> a, ap_uint<64> b,
                                  ap_uint<64> q) {
#pragma HLS INLINE
    return a >= b ? (ap_uint<64>)(a - b) : (ap_uint<64>)(a + q - b);
}

// Shoup multiplication. b_shoup=floor(b*2^64/q) is prepared by the host.
// For a,b<q this needs no divider and the signed correction is bounded.
static inline ap_uint<64> mul_mod_shoup(ap_uint<64> a, ap_uint<64> b,
                                        ap_uint<64> b_shoup,
                                        ap_uint<64> q) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=12 max=20
    ap_uint<128> estimate_product = (ap_uint<128>)a * b_shoup;
#pragma HLS BIND_OP variable=estimate_product op=mul impl=dsp latency=5
    ap_uint<64> estimate = estimate_product >> 64;
    ap_uint<128> product = (ap_uint<128>)a * b;
    ap_uint<128> quotient_product = (ap_uint<128>)estimate * q;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=5
#pragma HLS BIND_OP variable=quotient_product op=mul impl=dsp latency=5
    ap_int<129> reduced = (ap_int<129>)product - (ap_int<129>)quotient_product;
    if (reduced < 0) reduced += q;
    if (reduced >= q) reduced -= q;
    if (reduced >= q) reduced -= q;
    return (ap_uint<64>)reduced;
}
