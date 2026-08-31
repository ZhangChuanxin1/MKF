#pragma once

#include <ap_int.h>

// Lattigo v2.3.0-compatible Montgomery reduction:
// x*y*2^-64 mod q, with q_inv = q^-1 mod 2^64.
static ap_uint<64> mred64(ap_uint<64> x, ap_uint<64> y,
                          ap_uint<64> q, ap_uint<64> q_inv) {
#pragma HLS INLINE
    ap_uint<128> xy = (ap_uint<128>)x * y;
#pragma HLS BIND_OP variable=xy op=mul impl=dsp latency=4
    ap_uint<64> xy_hi = xy.range(127, 64);
    ap_uint<64> xy_lo = xy.range(63, 0);

    ap_uint<128> inv_product = (ap_uint<128>)xy_lo * q_inv;
#pragma HLS BIND_OP variable=inv_product op=mul impl=dsp latency=4
    ap_uint<64> m = inv_product.range(63, 0);

    ap_uint<128> mq = (ap_uint<128>)m * q;
#pragma HLS BIND_OP variable=mq op=mul impl=dsp latency=4
    ap_uint<64> mq_hi = mq.range(127, 64);

    ap_uint<65> r = (ap_uint<65>)xy_hi + q - mq_hi;
    if (r >= q) r -= q;
    return (ap_uint<64>)r;
}

// Lattigo MRedConstant variant. Unlike mred64, this deliberately keeps the
// lazy result in [0,2q); InvNTT feeds operands above q into this primitive.
static ap_uint<64> mred64_lazy(ap_uint<64> x, ap_uint<64> y,
                              ap_uint<64> q, ap_uint<64> q_inv) {
#pragma HLS INLINE
    ap_uint<128> xy = (ap_uint<128>)x * y;
#pragma HLS BIND_OP variable=xy op=mul impl=dsp latency=4
    ap_uint<128> iq = (ap_uint<128>)xy.range(63, 0) * q_inv;
#pragma HLS BIND_OP variable=iq op=mul impl=dsp latency=4
    ap_uint<128> mq = (ap_uint<128>)iq.range(63, 0) * q;
#pragma HLS BIND_OP variable=mq op=mul impl=dsp latency=4
    return (ap_uint<64>)xy.range(127, 64) -
           (ap_uint<64>)mq.range(127, 64) + q;
}

template <int Lane>
static ap_uint<64> mred64_lane(ap_uint<64> x, ap_uint<64> y,
                               ap_uint<64> q, ap_uint<64> q_inv) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    ap_uint<128> xy = (ap_uint<128>)x * y;
#pragma HLS BIND_OP variable=xy op=mul impl=dsp latency=4
    ap_uint<64> xy_hi = xy.range(127, 64);
    ap_uint<64> xy_lo = xy.range(63, 0);
    ap_uint<128> inv_product = (ap_uint<128>)xy_lo * q_inv;
#pragma HLS BIND_OP variable=inv_product op=mul impl=dsp latency=4
    ap_uint<64> m = inv_product.range(63, 0);
    ap_uint<128> mq = (ap_uint<128>)m * q;
#pragma HLS BIND_OP variable=mq op=mul impl=dsp latency=4
    ap_uint<64> mq_hi = mq.range(127, 64);
    ap_uint<65> r = (ap_uint<65>)xy_hi + q - mq_hi;
    if (r >= q) r -= q;
    return (ap_uint<64>)r;
}

static void mred64_x8(const ap_uint<64> x[8], const ap_uint<64> y[8],
                      ap_uint<64> out[8], ap_uint<64> q,
                      ap_uint<64> q_inv) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=y complete
#pragma HLS ARRAY_PARTITION variable=out complete
    out[0] = mred64_lane<0>(x[0], y[0], q, q_inv);
    out[1] = mred64_lane<1>(x[1], y[1], q, q_inv);
    out[2] = mred64_lane<2>(x[2], y[2], q, q_inv);
    out[3] = mred64_lane<3>(x[3], y[3], q, q_inv);
    out[4] = mred64_lane<4>(x[4], y[4], q, q_inv);
    out[5] = mred64_lane<5>(x[5], y[5], q, q_inv);
    out[6] = mred64_lane<6>(x[6], y[6], q, q_inv);
    out[7] = mred64_lane<7>(x[7], y[7], q, q_inv);
}
