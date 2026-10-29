#include <ap_int.h>
#include "par1_montgomery.hpp"
#include "../hls/include/mkf_cycle_counters.hpp"
#include "../hls/include/par1_parameters.hpp"

// Bit-exact reference NTT/INTT hardware block for Par1. Twiddles are supplied
// in the exact order consumed by the stage loops and are generated/verified by
// the host against the Lattigo reference. This block is deliberately kept
// separate from the top-level Mul+Relin kernel so its cycle count can be
// reconciled with Table V without double-charging Table VII.
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

    CycleCounters counters;
    reset_counters(counters);
    static ap_uint<64> bank_a[par1::lanes][par1::polynomial_degree / par1::lanes];
    static ap_uint<64> bank_b[par1::lanes][par1::polynomial_degree / par1::lanes];
    static ap_uint<64> twiddle_bank[par1::butterflies_per_cycle]
                                    [par1::polynomial_degree / par1::butterflies_per_cycle];
#pragma HLS ARRAY_PARTITION variable=bank_a complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_b complete dim=1
#pragma HLS ARRAY_PARTITION variable=twiddle_bank complete dim=1
#pragma HLS BIND_STORAGE variable=bank_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=bank_b type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=twiddle_bank type=ram_2p impl=bram

    const ap_uint<64> two_q = modulus << 1;
    const ap_uint<64> four_q = modulus << 2;

    for (unsigned i = 0; i < par1::polynomial_degree; i += par1::lanes) {
#pragma HLS PIPELINE II=1
        for (unsigned lane = 0; lane < par1::lanes; ++lane) {
#pragma HLS UNROLL
            ap_uint<64> x = coeff[i + lane];
            // The forward transform deliberately returns a lazy residue.
            // Canonicalize at the inverse boundary so every inverse stage
            // begins within the range required by Lattigo's butterfly.
            if (inverse) {
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
                if (x >= modulus) x -= modulus;
            }
            bank_a[lane][i / par1::lanes] = x;
        }
    }

    // Cache the table once. During a stage, equal twiddles are broadcasts and
    // consecutive twiddles map to distinct cyclic banks, so the butterfly
    // datapath never requests eight independent words from one AXI port.
    for (unsigned i = 0; i < par1::polynomial_degree; ++i) {
#pragma HLS PIPELINE II=1
        twiddle_bank[i % par1::butterflies_per_cycle]
                    [i / par1::butterflies_per_cycle] = twiddle[i];
    }
    bool src_a = true;
    if (!inverse) {
        unsigned t = par1::polynomial_degree >> 1;
        unsigned log_t = par1::log_n - 1;
        for (unsigned m = 1; m < par1::polynomial_degree; m <<= 1) {
            for (unsigned b = 0; b < par1::polynomial_degree / 2;
                 b += par1::butterflies_per_cycle) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=bank_a inter false
#pragma HLS DEPENDENCE variable=bank_b inter false
                ap_uint<64> us[8], rhs[8], psis[8], vs[8];
                unsigned pos[8], pos2[8];
#pragma HLS ARRAY_PARTITION variable=us complete
#pragma HLS ARRAY_PARTITION variable=rhs complete
#pragma HLS ARRAY_PARTITION variable=psis complete
#pragma HLS ARRAY_PARTITION variable=vs complete
#pragma HLS ARRAY_PARTITION variable=pos complete
#pragma HLS ARRAY_PARTITION variable=pos2 complete
                for (unsigned lane = 0; lane < par1::butterflies_per_cycle; ++lane) {
#pragma HLS UNROLL
                    unsigned butterfly = b + lane;
                    unsigned group = butterfly >> log_t;
                    unsigned j = butterfly & (t - 1);
                    unsigned p = (group * t << 1) + j;
                    unsigned p2 = p + t;
                    pos[lane] = p;
                    pos2[lane] = p2;
                    unsigned tw_idx = m + group;
                    psis[lane] = twiddle_bank[tw_idx % par1::butterflies_per_cycle]
                                              [tw_idx / par1::butterflies_per_cycle];
                    us[lane] = src_a ? bank_a[p % par1::lanes][p / par1::lanes]
                                          : bank_b[p % par1::lanes][p / par1::lanes];
                    if (us[lane] >= four_q) us[lane] -= four_q;
                    rhs[lane] = src_a ? bank_a[p2 % par1::lanes][p2 / par1::lanes]
                                            : bank_b[p2 % par1::lanes][p2 / par1::lanes];
                }
                mred64_x8(rhs, psis, vs, modulus, modulus_inv);
                for (unsigned lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                    unsigned p = pos[lane], p2 = pos2[lane];
                    ap_uint<64> u = us[lane], v = vs[lane];
                    if (src_a) {
                        bank_b[p % par1::lanes][p / par1::lanes] = u + v;
                        bank_b[p2 % par1::lanes][p2 / par1::lanes] = u + two_q - v;
                    } else {
                        bank_a[p % par1::lanes][p / par1::lanes] = u + v;
                        bank_a[p2 % par1::lanes][p2 / par1::lanes] = u + two_q - v;
                    }
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
            for (unsigned b = 0; b < par1::polynomial_degree / 2;
                 b += par1::butterflies_per_cycle) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=bank_a inter false
#pragma HLS DEPENDENCE variable=bank_b inter false
                for (unsigned lane = 0; lane < par1::butterflies_per_cycle; ++lane) {
#pragma HLS UNROLL
                    unsigned butterfly = b + lane;
                    unsigned group = butterfly >> log_t;
                    unsigned j = butterfly & (t - 1);
                    unsigned p = (group * t << 1) + j;
                    unsigned p2 = p + t;
                    unsigned tw_idx = m + group;
                    ap_uint<64> psi = twiddle_bank[tw_idx % par1::butterflies_per_cycle]
                                                 [tw_idx / par1::butterflies_per_cycle];
                    ap_uint<64> u = src_a ? bank_a[p % par1::lanes][p / par1::lanes]
                                          : bank_b[p % par1::lanes][p / par1::lanes];
                    ap_uint<64> v = src_a ? bank_a[p2 % par1::lanes][p2 / par1::lanes]
                                          : bank_b[p2 % par1::lanes][p2 / par1::lanes];
                    ap_uint<65> x = (ap_uint<65>)u + v;
                    if (x >= two_q) x -= two_q;
                    // Lattigo v2.3 invbutterfly uses U + 4Q - V here.
                    // Using 2Q is only valid when both operands are already
                    // bounded below 2Q; lazy values at this boundary can be
                    // as large as 4Q and caused every even coefficient of the
                    // round trip to fail.
                    ap_uint<64> y = u + four_q - v;
                    if (src_a) {
                        bank_b[p % par1::lanes][p / par1::lanes] = (ap_uint<64>)x;
                        bank_b[p2 % par1::lanes][p2 / par1::lanes] = mred64_lazy(y, psi, modulus, modulus_inv);
                    } else {
                        bank_a[p % par1::lanes][p / par1::lanes] = (ap_uint<64>)x;
                        bank_a[p2 % par1::lanes][p2 / par1::lanes] = mred64_lazy(y, psi, modulus, modulus_inv);
                    }
                }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a;
            t <<= 1;
            ++log_t;
        }
        for (unsigned i = 0; i < par1::polynomial_degree; i += par1::lanes) {
#pragma HLS PIPELINE II=1
            for (unsigned lane = 0; lane < par1::lanes; ++lane) {
#pragma HLS UNROLL
                ap_uint<64> x = src_a ? bank_a[lane][i / par1::lanes]
                                      : bank_b[lane][i / par1::lanes];
                if (src_a)
                    bank_b[lane][i / par1::lanes] = mred64(x, ntt_n_inv, modulus, modulus_inv);
                else
                    bank_a[lane][i / par1::lanes] = mred64(x, ntt_n_inv, modulus, modulus_inv);
            }
            count_cycle(counters, CycleClass::NttIntt, true);
        }
        src_a = !src_a;
    }

    for (unsigned i = 0; i < par1::polynomial_degree; i += par1::lanes) {
#pragma HLS PIPELINE II=1
        for (unsigned lane = 0; lane < par1::lanes; ++lane) {
#pragma HLS UNROLL
            coeff[i + lane] = src_a ? bank_a[lane][i / par1::lanes]
                                    : bank_b[lane][i / par1::lanes];
        }
    }

    reported[0] = counters.decomposition;
    reported[1] = counters.external_product;
    reported[2] = counters.ntt_intt;
    reported[3] = counters.modup_down;
    reported[4] = counters.hbm_stall;
    reported[5] = counters.other;
    reported[6] = counters.total;
}
