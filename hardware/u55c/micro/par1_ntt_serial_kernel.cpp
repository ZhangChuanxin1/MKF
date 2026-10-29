#include <ap_int.h>
#include "par1_montgomery.hpp"
#include "../hls/include/mkf_cycle_counters.hpp"
#include "../hls/include/par1_parameters.hpp"

// Conservative serial implementation: one butterfly per cycle with two
// ping-pong memories. It is intended to reach placement first; wider static
// stage variants can subsequently replace it without changing the interface.
extern "C" void par1_ntt_kernel(ap_uint<64> *coeff,
                                const ap_uint<64> *twiddle,
                                ap_uint<64> modulus,
                                ap_uint<64> modulus_inv,
                                ap_uint<64> ntt_n_inv,
                                bool inverse,
                                ap_uint<64> *reported) {
#pragma HLS INTERFACE m_axi port=coeff offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=twiddle offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=reported offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=coeff bundle=control
#pragma HLS INTERFACE s_axilite port=twiddle bundle=control
#pragma HLS INTERFACE s_axilite port=modulus bundle=control
#pragma HLS INTERFACE s_axilite port=modulus_inv bundle=control
#pragma HLS INTERFACE s_axilite port=ntt_n_inv bundle=control
#pragma HLS INTERFACE s_axilite port=inverse bundle=control
#pragma HLS INTERFACE s_axilite port=reported bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ap_uint<64> a[par1::polynomial_degree];
    static ap_uint<64> bmem[par1::polynomial_degree];
    static ap_uint<64> tw[par1::polynomial_degree];
#pragma HLS BIND_STORAGE variable=a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bmem type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=tw type=ram_2p impl=bram
    CycleCounters counters;
    reset_counters(counters);
    const ap_uint<64> two_q = modulus << 1;
    const ap_uint<64> four_q = modulus << 2;

    for (unsigned i = 0; i < par1::polynomial_degree; ++i) {
#pragma HLS PIPELINE II=1
        ap_uint<64> x = coeff[i];
        if (inverse) {
            for (unsigned k = 0; k < 7; ++k) {
#pragma HLS UNROLL
                if (x >= modulus) x -= modulus;
            }
        }
        a[i] = x;
    }
    for (unsigned i = 0; i < par1::polynomial_degree; ++i) {
#pragma HLS PIPELINE II=1
        tw[i] = twiddle[i];
    }

    bool src_a = true;
    if (!inverse) {
        unsigned t = par1::polynomial_degree >> 1;
        unsigned log_t = par1::log_n - 1;
        for (unsigned m = 1; m < par1::polynomial_degree; m <<= 1) {
            for (unsigned butterfly = 0; butterfly < par1::polynomial_degree / 2; ++butterfly) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=a inter false
#pragma HLS DEPENDENCE variable=bmem inter false
                unsigned group = butterfly >> log_t;
                unsigned j = butterfly & (t - 1);
                unsigned p = (group * t << 1) + j;
                unsigned p2 = p + t;
                ap_uint<64> u = src_a ? a[p] : bmem[p];
                if (u >= four_q) u -= four_q;
                ap_uint<64> rhs = src_a ? a[p2] : bmem[p2];
                ap_uint<64> v = mred64(rhs, tw[m + group], modulus, modulus_inv);
                if (src_a) {
                    bmem[p] = u + v;
                    bmem[p2] = u + two_q - v;
                } else {
                    a[p] = u + v;
                    a[p2] = u + two_q - v;
                }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a;
            t >>= 1;
            --log_t;
        }
    } else {
        unsigned t = 1;
        unsigned log_t = 0;
        for (unsigned m = par1::polynomial_degree >> 1; m != 0; m >>= 1) {
            for (unsigned butterfly = 0; butterfly < par1::polynomial_degree / 2; ++butterfly) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=a inter false
#pragma HLS DEPENDENCE variable=bmem inter false
                unsigned group = butterfly >> log_t;
                unsigned j = butterfly & (t - 1);
                unsigned p = (group * t << 1) + j;
                unsigned p2 = p + t;
                ap_uint<64> u = src_a ? a[p] : bmem[p];
                ap_uint<64> v = src_a ? a[p2] : bmem[p2];
                ap_uint<65> x = (ap_uint<65>)u + v;
                if (x >= two_q) x -= two_q;
                ap_uint<64> y = u + four_q - v;
                ap_uint<64> z = mred64_lazy(y, tw[m + group], modulus, modulus_inv);
                if (src_a) {
                    bmem[p] = (ap_uint<64>)x;
                    bmem[p2] = z;
                } else {
                    a[p] = (ap_uint<64>)x;
                    a[p2] = z;
                }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a;
            t <<= 1;
            ++log_t;
        }
        for (unsigned i = 0; i < par1::polynomial_degree; ++i) {
#pragma HLS PIPELINE II=1
            ap_uint<64> x = src_a ? a[i] : bmem[i];
            if (src_a) bmem[i] = mred64(x, ntt_n_inv, modulus, modulus_inv);
            else a[i] = mred64(x, ntt_n_inv, modulus, modulus_inv);
            count_cycle(counters, CycleClass::NttIntt, true);
        }
        src_a = !src_a;
    }

    for (unsigned i = 0; i < par1::polynomial_degree; ++i) {
#pragma HLS PIPELINE II=1
        coeff[i] = src_a ? a[i] : bmem[i];
    }
    reported[0] = counters.decomposition;
    reported[1] = counters.external_product;
    reported[2] = counters.ntt_intt;
    reported[3] = counters.modup_down;
    reported[4] = counters.hbm_stall;
    reported[5] = counters.other;
    reported[6] = counters.total;
}
