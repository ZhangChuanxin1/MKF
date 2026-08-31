#include <ap_int.h>

#include "../include/mkf_cycle_counters.hpp"
#include "../include/par1_ckks_hls_constants.hpp"
#include "par1_montgomery.hpp"

namespace mkf_ckks_u55c {

using namespace par1_ckks_hls;
using word = ap_uint<64>;
using addr_t = unsigned long long;

static constexpr unsigned PARTIES = 16;
static constexpr unsigned COMPONENTS = PARTIES + 1;
static constexpr unsigned OUTPUT_COMPONENTS = PARTIES + 2;
static constexpr addr_t Q_POLY_WORDS = Q_COUNT * N;
static constexpr addr_t QP_POLY_WORDS = QP_COUNT * N;

enum ScratchSlot : unsigned {
    COEFF = 0,
    DECOMP0 = 1,
    DECOMP1 = 2,
    DECOMP2 = 3,
    ACC0 = 4,
    ACC1 = 5,
    MODUP_TEMP = 6,
    LEFT0 = 7,
    LEFT1 = 8,
    RIGHT0 = 9,
    RIGHT1 = 10,
    MU0 = 11,
    MU1 = 12,
    MU2 = 13,
    RESCALE_TEMP = 14,
};

static inline addr_t slot(addr_t scratch, unsigned index) {
    return scratch + static_cast<addr_t>(index) * QP_POLY_WORDS;
}

static inline addr_t limb(addr_t poly, unsigned modulus) {
    return poly + static_cast<addr_t>(modulus) * N;
}

static inline word canonical(word x, word q) {
#pragma HLS INLINE
    for (unsigned i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        if (x >= q) x -= q;
    }
    return x;
}

static inline word add_mod(word a, word b, word q) {
#pragma HLS INLINE
    ap_uint<65> sum = static_cast<ap_uint<65>>(canonical(a, q)) + canonical(b, q);
    if (sum >= q) sum -= q;
    return static_cast<word>(sum);
}

static inline word sub_mod(word a, word b, word q) {
#pragma HLS INLINE
    a = canonical(a, q);
    b = canonical(b, q);
    return a >= b ? word(a - b) : word(a + q - b);
}

static inline word mont_mul(word a, word b_mont, word q, word q_inv) {
#pragma HLS INLINE
    return canonical(mred64(a, b_mont, q, q_inv), q);
}

static void ntt_limb(word *hbm, addr_t poly, addr_t twiddle,
                     word q, word q_inv, word n_inv, bool inverse,
                     CycleCounters &counters,
                     hls::stream<ap_uint<6>> &phase_events,
                     CycleClass resume_stage, TimingClass resume_timing) {
    emit_phase_event(phase_events, CycleClass::NttIntt, resume_timing);
    static word ntt_a[N];
    static word ntt_b[N];
    static word ntt_tw[N];
#pragma HLS BIND_STORAGE variable=ntt_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ntt_b type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ntt_tw type=ram_2p impl=bram
    const word two_q = q << 1;
    const word four_q = q << 2;
    for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        ntt_a[i] = canonical(hbm[poly + i], q);
        ntt_tw[i] = hbm[twiddle + i];
    }
    bool src_a = true;
    if (!inverse) {
        unsigned t = N >> 1;
        unsigned log_t = LOG_N - 1;
        for (unsigned m = 1; m < N; m <<= 1) {
            for (unsigned butterfly = 0; butterfly < N / 2; ++butterfly) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ntt_a inter false
#pragma HLS DEPENDENCE variable=ntt_b inter false
                unsigned group = butterfly >> log_t;
                unsigned j = butterfly & (t - 1);
                unsigned p = (group * t << 1) + j;
                unsigned p2 = p + t;
                word u = src_a ? ntt_a[p] : ntt_b[p];
                if (u >= four_q) u -= four_q;
                word rhs = src_a ? ntt_a[p2] : ntt_b[p2];
                word v = mred64(rhs, ntt_tw[m + group], q, q_inv);
                if (src_a) {
                    ntt_b[p] = u + v;
                    ntt_b[p2] = u + two_q - v;
                } else {
                    ntt_a[p] = u + v;
                    ntt_a[p2] = u + two_q - v;
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
        for (unsigned m = N >> 1; m != 0; m >>= 1) {
            for (unsigned butterfly = 0; butterfly < N / 2; ++butterfly) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ntt_a inter false
#pragma HLS DEPENDENCE variable=ntt_b inter false
                unsigned group = butterfly >> log_t;
                unsigned j = butterfly & (t - 1);
                unsigned p = (group * t << 1) + j;
                unsigned p2 = p + t;
                word u = src_a ? ntt_a[p] : ntt_b[p];
                word v = src_a ? ntt_a[p2] : ntt_b[p2];
                ap_uint<65> x = static_cast<ap_uint<65>>(u) + v;
                if (x >= two_q) x -= two_q;
                word y = u + four_q - v;
                word z = mred64_lazy(y, ntt_tw[m + group], q, q_inv);
                if (src_a) {
                    ntt_b[p] = static_cast<word>(x);
                    ntt_b[p2] = z;
                } else {
                    ntt_a[p] = static_cast<word>(x);
                    ntt_a[p2] = z;
                }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a;
            t <<= 1;
            ++log_t;
        }
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word x = src_a ? ntt_a[i] : ntt_b[i];
            if (src_a)
                ntt_b[i] = mred64(x, n_inv, q, q_inv);
            else
                ntt_a[i] = mred64(x, n_inv, q, q_inv);
            count_cycle(counters, CycleClass::NttIntt, true);
        }
        src_a = !src_a;
    }
    for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        hbm[poly + i] = canonical(src_a ? ntt_a[i] : ntt_b[i], q);
    }
    emit_phase_event(phase_events, resume_stage, resume_timing);
}

static inline word modup_decomposition(unsigned beta, unsigned target,
                                       word x0, word x1) {
#pragma HLS INLINE
    const unsigned s0 = beta << 1;
    const unsigned s1 = s0 + 1;
    const word q0 = Q[s0];
    const word q1 = Q[s1];
    word y0 = mont_mul(x0, DECOMP_INV[beta][0], q0, MRED_Q[s0]);
    word y1 = mont_mul(x1, DECOMP_INV[beta][1], q1, MRED_Q[s1]);
    ap_uint<128> lhs = static_cast<ap_uint<128>>(y0) * q1 +
                       static_cast<ap_uint<128>>(y1) * q0;
    ap_uint<128> rhs = static_cast<ap_uint<128>>(q0) * q1;
    word target_q = QP[target];
    word result = mont_mul(y0, DECOMP_QSTAR[beta][target][0], target_q, MRED_QP[target]);
    result = add_mod(result,
                     mont_mul(y1, DECOMP_QSTAR[beta][target][1], target_q, MRED_QP[target]),
                     target_q);
    if (lhs >= rhs) result = add_mod(result, DECOMP_MINUS_Q[beta][target], target_q);
    return result;
}

static inline word modup_p_to_q(unsigned target, word x0, word x1) {
#pragma HLS INLINE
    word y0 = mont_mul(x0, P2Q_INV[0], P[0], MRED_P[0]);
    word y1 = mont_mul(x1, P2Q_INV[1], P[1], MRED_P[1]);
    ap_uint<128> lhs = static_cast<ap_uint<128>>(y0) * P[1] +
                       static_cast<ap_uint<128>>(y1) * P[0];
    ap_uint<128> rhs = static_cast<ap_uint<128>>(P[0]) * P[1];
    word result = mont_mul(y0, P2Q_QSTAR[target][0], Q[target], MRED_Q[target]);
    result = add_mod(result,
                     mont_mul(y1, P2Q_QSTAR[target][1], Q[target], MRED_Q[target]),
                     Q[target]);
    if (lhs >= rhs) result = add_mod(result, P2Q_MINUS_P[target], Q[target]);
    return result;
}

static void decompose_ntt(word *hbm, addr_t input_q, addr_t scratch,
                          addr_t fwd_q, addr_t inv_q,
                          addr_t fwd_p, addr_t inv_p,
                          CycleCounters &counters,
                          hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
    set_timing_class(counters, TimingClass::Decomposition);
    addr_t coeff = slot(scratch, COEFF);
    for (unsigned m = 0; m < Q_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(coeff, m) + i] = hbm[input_q + static_cast<addr_t>(m) * N + i];
        }
        ntt_limb(hbm, limb(coeff, m), inv_q + static_cast<addr_t>(m) * N,
                 Q[m], MRED_Q[m], N_INV_Q[m], true, counters, phase_events,
                 CycleClass::Decomposition, TimingClass::Decomposition);
    }
    for (unsigned beta = 0; beta < BETA; ++beta) {
        addr_t out = slot(scratch, DECOMP0 + beta);
        for (unsigned target = 0; target < QP_COUNT; ++target) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word x0 = hbm[limb(coeff, beta * 2) + i];
                word x1 = hbm[limb(coeff, beta * 2 + 1) + i];
                hbm[limb(out, target) + i] = modup_decomposition(beta, target, x0, x1);
                count_cycle(counters, CycleClass::Decomposition, true);
            }
            if (target == beta * 2) {
                for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                    hbm[limb(out, target) + i] =
                        hbm[input_q + static_cast<addr_t>(target) * N + i];
                }
            } else if (target < Q_COUNT) {
                ntt_limb(hbm, limb(out, target), fwd_q + static_cast<addr_t>(target) * N,
                         Q[target], MRED_Q[target], N_INV_Q[target], false, counters, phase_events,
                         CycleClass::Decomposition, TimingClass::Decomposition);
            } else {
                unsigned p = target - Q_COUNT;
                ntt_limb(hbm, limb(out, target), fwd_p + static_cast<addr_t>(p) * N,
                         P[p], MRED_P[p], N_INV_P[p], false, counters, phase_events,
                         CycleClass::Decomposition, TimingClass::Decomposition);
            }
        }
    }
    (void)inv_p;
}

static addr_t key_poly(addr_t key_base, unsigned key_index,
                       unsigned decomposition, unsigned output,
                       unsigned modulus) {
#pragma HLS INLINE
    addr_t poly_index = ((static_cast<addr_t>(key_index) * BETA + decomposition) * 2 + output);
    return key_base + (poly_index * QP_COUNT + modulus) * N;
}

static void external_product(word *hbm, addr_t key_base, unsigned key_index,
                             addr_t scratch, CycleCounters &counters,
                             hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(counters, TimingClass::ExternalProduct);
    for (unsigned output = 0; output < 2; ++output) {
        addr_t acc = slot(scratch, ACC0 + output);
        for (unsigned modulus = 0; modulus < QP_COUNT; ++modulus) {
            word q = QP[modulus];
            word q_inv = MRED_QP[modulus];
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word sum = 0;
                for (unsigned d = 0; d < BETA; ++d) {
#pragma HLS UNROLL
                    word a = hbm[limb(slot(scratch, DECOMP0 + d), modulus) + i];
                    word k = hbm[key_poly(key_base, key_index, d, output, modulus) + i];
                    sum = add_mod(sum, mont_mul(a, k, q, q_inv), q);
                }
                hbm[limb(acc, modulus) + i] = sum;
                count_cycle(counters, CycleClass::ExternalProduct, true);
            }
        }
    }
}

static void moddown_qp_to_q_ntt(word *hbm, addr_t scratch,
                                addr_t fwd_q, addr_t inv_p,
                                CycleCounters &counters,
                                hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    addr_t temp = slot(scratch, MODUP_TEMP);
    for (unsigned output = 0; output < 2; ++output) {
        addr_t acc = slot(scratch, ACC0 + output);
        for (unsigned p = 0; p < P_COUNT; ++p) {
            ntt_limb(hbm, limb(acc, Q_COUNT + p), inv_p + static_cast<addr_t>(p) * N,
                     P[p], MRED_P[p], N_INV_P[p], true, counters, phase_events,
                     CycleClass::ModUpDown, TimingClass::ModUpDown);
        }
        for (unsigned qidx = 0; qidx < Q_COUNT; ++qidx) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word p0 = hbm[limb(acc, Q_COUNT) + i];
                word p1 = hbm[limb(acc, Q_COUNT + 1) + i];
                hbm[limb(temp, qidx) + i] = modup_p_to_q(qidx, p0, p1);
                count_cycle(counters, CycleClass::ModUpDown, true);
            }
            ntt_limb(hbm, limb(temp, qidx), fwd_q + static_cast<addr_t>(qidx) * N,
                     Q[qidx], MRED_Q[qidx], N_INV_Q[qidx], false, counters, phase_events,
                     CycleClass::ModUpDown, TimingClass::ModUpDown);
            word neg_p_inv = Q[qidx] - P_INV_Q[qidx];
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word x = hbm[limb(acc, qidx) + i];
                word y = hbm[limb(temp, qidx) + i];
                word delta = add_mod(y, Q[qidx] - canonical(x, Q[qidx]), Q[qidx]);
                hbm[limb(acc, qidx) + i] = mont_mul(delta, neg_p_inv, Q[qidx], MRED_Q[qidx]);
                count_cycle(counters, CycleClass::ModUpDown, true);
            }
        }
    }
}

static void switch_one(word *hbm, addr_t input_q, addr_t key_base,
                       unsigned key_index, addr_t scratch,
                       addr_t fwd_q, addr_t inv_q,
                       addr_t fwd_p, addr_t inv_p,
                       CycleCounters &counters,
                       hls::stream<ap_uint<6>> &phase_events) {
    decompose_ntt(hbm, input_q, scratch, fwd_q, inv_q, fwd_p, inv_p, counters, phase_events);
    external_product(hbm, key_base, key_index, scratch, counters, phase_events);
    moddown_qp_to_q_ntt(hbm, scratch, fwd_q, inv_p, counters, phase_events);
}

static addr_t input_poly(addr_t input_base, unsigned side, unsigned component) {
#pragma HLS INLINE
    return input_base + (static_cast<addr_t>(side) * COMPONENTS + component) * Q_POLY_WORDS;
}

static void copy_q_poly(word *hbm, addr_t source, addr_t dest, unsigned level) {
    for (unsigned m = 0; m <= level; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(dest, m) + i] = canonical(hbm[source + static_cast<addr_t>(m) * N + i], Q[m]);
        }
    }
}

static void fold_side(word *hbm, addr_t input_base, unsigned side,
                      addr_t key_base, addr_t out0, addr_t out1,
                      addr_t scratch, addr_t fwd_q, addr_t inv_q,
                      addr_t fwd_p, addr_t inv_p,
                      CycleCounters &counters,
                      hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
    copy_q_poly(hbm, input_poly(input_base, side, 0), out0, Q_COUNT - 1);
    for (unsigned m = 0; m < Q_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(out1, m) + i] = 0;
        }
    }
    for (unsigned party = 0; party < PARTIES; ++party) {
        switch_one(hbm, input_poly(input_base, side, party + 1), key_base, party,
                   scratch, fwd_q, inv_q, fwd_p, inv_p, counters, phase_events);
        emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
        set_timing_class(counters, TimingClass::ExternalProduct);
        for (unsigned m = 0; m < Q_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[limb(out0, m) + i] = add_mod(hbm[limb(out0, m) + i],
                                                      hbm[limb(slot(scratch, ACC0), m) + i], Q[m]);
                hbm[limb(out1, m) + i] = add_mod(hbm[limb(out1, m) + i],
                                                      hbm[limb(slot(scratch, ACC1), m) + i], Q[m]);
                count_cycle(counters, CycleClass::ExternalProduct, true);
            }
        }
    }
}

static void tensor_product(word *hbm, addr_t scratch, CycleCounters &counters,
                           hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Tensor);
    set_timing_class(counters, TimingClass::Tensor);
    addr_t left0 = slot(scratch, LEFT0), left1 = slot(scratch, LEFT1);
    addr_t right0 = slot(scratch, RIGHT0), right1 = slot(scratch, RIGHT1);
    addr_t mu0 = slot(scratch, MU0), mu1 = slot(scratch, MU1), mu2 = slot(scratch, MU2);
    for (unsigned m = 0; m < Q_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word a0 = canonical(hbm[limb(left0, m) + i], Q[m]);
            word a1 = canonical(hbm[limb(left1, m) + i], Q[m]);
            word b0 = canonical(hbm[limb(right0, m) + i], Q[m]);
            word b1 = canonical(hbm[limb(right1, m) + i], Q[m]);
            word p0 = mont_mul(a0, R2_Q[m], Q[m], MRED_Q[m]);
            p0 = mont_mul(p0, b0, Q[m], MRED_Q[m]);
            word p2 = mont_mul(a1, R2_Q[m], Q[m], MRED_Q[m]);
            p2 = mont_mul(p2, b1, Q[m], MRED_Q[m]);
            word sa = add_mod(a0, a1, Q[m]);
            word sb = add_mod(b0, b1, Q[m]);
            word p1 = mont_mul(sa, R2_Q[m], Q[m], MRED_Q[m]);
            p1 = mont_mul(p1, sb, Q[m], MRED_Q[m]);
            p1 = sub_mod(sub_mod(p1, p0, Q[m]), p2, Q[m]);
            hbm[limb(mu0, m) + i] = p0;
            hbm[limb(mu1, m) + i] = p1;
            hbm[limb(mu2, m) + i] = p2;
            count_cycle(counters, CycleClass::Tensor, true);
        }
    }
}

static void rescale_one(word *hbm, addr_t poly, addr_t scratch,
                        addr_t fwd_q, addr_t inv_q,
                        CycleCounters &counters,
                        hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    addr_t temp = slot(scratch, RESCALE_TEMP);
    for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        hbm[limb(temp, Q_COUNT - 1) + i] = hbm[limb(poly, Q_COUNT - 1) + i];
    }
    ntt_limb(hbm, limb(temp, Q_COUNT - 1), inv_q + static_cast<addr_t>(Q_COUNT - 1) * N,
             Q[Q_COUNT - 1], MRED_Q[Q_COUNT - 1], N_INV_Q[Q_COUNT - 1], true, counters, phase_events,
             CycleClass::ModUpDown, TimingClass::ModUpDown);
    word p_half = (Q[Q_COUNT - 1] - 1) >> 1;
    for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        word x = canonical(hbm[limb(temp, Q_COUNT - 1) + i], Q[Q_COUNT - 1]);
        hbm[limb(temp, Q_COUNT - 1) + i] = add_mod(x, p_half, Q[Q_COUNT - 1]);
    }
    for (unsigned m = 0; m < Q_COUNT - 1; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word x = hbm[limb(temp, Q_COUNT - 1) + i];
            hbm[limb(temp, m) + i] = add_mod(x, Q[m] - p_half, Q[m]);
        }
        ntt_limb(hbm, limb(temp, m), fwd_q + static_cast<addr_t>(m) * N,
                 Q[m], MRED_Q[m], N_INV_Q[m], false, counters, phase_events,
                 CycleClass::ModUpDown, TimingClass::ModUpDown);
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word x = hbm[limb(temp, m) + i];
            word y = hbm[limb(poly, m) + i];
            word delta = sub_mod(x, y, Q[m]);
            hbm[limb(poly, m) + i] = mont_mul(delta, RESCALE_5_TO_4[m], Q[m], MRED_Q[m]);
            count_cycle(counters, CycleClass::ModUpDown, true);
        }
    }
}

static void debug_copy(word *hbm, addr_t &debug_cursor, addr_t source,
                       unsigned polys, unsigned level) {
    for (unsigned p = 0; p < polys; ++p) {
        for (unsigned m = 0; m <= level; ++m) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[debug_cursor++] = canonical(hbm[limb(source + static_cast<addr_t>(p) * QP_POLY_WORDS, m) + i], Q[m]);
            }
        }
    }
}

} // namespace mkf_ckks_u55c

// Native/C-simulation diagnostic entry point. It exercises the first fold
// key-switch and emits the three internal boundaries used to localize any
// representation mismatch. It is not selected as the synthesized top.
extern "C" void par1_mkf_ckks_switch_debug(
    ap_uint<64> *hbm, unsigned long long input_poly_base, unsigned long long key_base,
    unsigned long long scratch_base, unsigned long long ntt_forward_q_base,
    unsigned long long ntt_inverse_q_base, unsigned long long ntt_forward_p_base,
    unsigned long long ntt_inverse_p_base, unsigned long long debug_base,
    hls::stream<ap_uint<6>> &phase_events) {
    using namespace mkf_ckks_u55c;
    CycleCounters counters;
    reset_counters(counters);
    decompose_ntt(hbm, input_poly_base, scratch_base, ntt_forward_q_base,
                  ntt_inverse_q_base, ntt_forward_p_base,
                  ntt_inverse_p_base, counters, phase_events);
    addr_t cursor = debug_base;
    for (unsigned d = 0; d < BETA; ++d) {
        for (unsigned m = 0; m < QP_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i)
                hbm[cursor++] = canonical(hbm[limb(slot(scratch_base, DECOMP0 + d), m) + i], QP[m]);
        }
    }
    external_product(hbm, key_base, 0, scratch_base, counters, phase_events);
    for (unsigned output = 0; output < 2; ++output) {
        for (unsigned m = 0; m < QP_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i)
                hbm[cursor++] = canonical(hbm[limb(slot(scratch_base, ACC0 + output), m) + i], QP[m]);
        }
    }
    moddown_qp_to_q_ntt(hbm, scratch_base, ntt_forward_q_base,
                        ntt_inverse_p_base, counters, phase_events);
    for (unsigned output = 0; output < 2; ++output) {
        for (unsigned m = 0; m < Q_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i)
                hbm[cursor++] = canonical(hbm[limb(slot(scratch_base, ACC0 + output), m) + i], Q[m]);
        }
    }
}

extern "C" void par1_mkf_ckks_kernel(
    ap_uint<64> *hbm,
    unsigned long long input_base,
    unsigned long long key_base,
    unsigned long long output_base,
    unsigned long long scratch_base,
    unsigned long long ntt_forward_q_base,
    unsigned long long ntt_inverse_q_base,
    unsigned long long ntt_forward_p_base,
    unsigned long long ntt_inverse_p_base,
    unsigned long long debug_base,
    bool enable_debug,
    unsigned long long report_base, hls::stream<ap_uint<6>> &phase_events) {
#pragma HLS INTERFACE m_axi port=hbm offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=hbm bundle=control
#pragma HLS INTERFACE s_axilite port=input_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_base bundle=control
#pragma HLS INTERFACE s_axilite port=output_base bundle=control
#pragma HLS INTERFACE s_axilite port=scratch_base bundle=control
#pragma HLS INTERFACE s_axilite port=ntt_forward_q_base bundle=control
#pragma HLS INTERFACE s_axilite port=ntt_inverse_q_base bundle=control
#pragma HLS INTERFACE s_axilite port=ntt_forward_p_base bundle=control
#pragma HLS INTERFACE s_axilite port=ntt_inverse_p_base bundle=control
#pragma HLS INTERFACE s_axilite port=debug_base bundle=control
#pragma HLS INTERFACE s_axilite port=enable_debug bundle=control
#pragma HLS INTERFACE s_axilite port=report_base bundle=control
#pragma HLS INTERFACE axis port=phase_events
#pragma HLS INTERFACE s_axilite port=return bundle=control

    using namespace mkf_ckks_u55c;
    CycleCounters counters;
    reset_counters(counters);

    addr_t left0 = slot(scratch_base, LEFT0), left1 = slot(scratch_base, LEFT1);
    addr_t right0 = slot(scratch_base, RIGHT0), right1 = slot(scratch_base, RIGHT1);
    fold_side(hbm, input_base, 0, key_base, left0, left1, scratch_base,
              ntt_forward_q_base, ntt_inverse_q_base,
              ntt_forward_p_base, ntt_inverse_p_base, counters, phase_events);
    fold_side(hbm, input_base, 1, key_base, right0, right1, scratch_base,
              ntt_forward_q_base, ntt_inverse_q_base,
              ntt_forward_p_base, ntt_inverse_p_base, counters, phase_events);

    addr_t debug_cursor = debug_base;
    if (enable_debug) {
        debug_copy(hbm, debug_cursor, left0, 4, Q_COUNT - 1);
    }

    tensor_product(hbm, scratch_base, counters, phase_events);
    if (enable_debug) {
        debug_copy(hbm, debug_cursor, slot(scratch_base, MU0), 3, Q_COUNT - 1);
    }

    switch_one(hbm, slot(scratch_base, MU2), key_base, PARTIES, scratch_base,
               ntt_forward_q_base, ntt_inverse_q_base,
               ntt_forward_p_base, ntt_inverse_p_base, counters, phase_events);
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(counters, TimingClass::ExternalProduct);
    for (unsigned m = 0; m < Q_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(slot(scratch_base, MU0), m) + i] =
                add_mod(hbm[limb(slot(scratch_base, MU0), m) + i],
                        hbm[limb(slot(scratch_base, ACC0), m) + i], Q[m]);
            hbm[limb(slot(scratch_base, MU1), m) + i] =
                add_mod(hbm[limb(slot(scratch_base, MU1), m) + i],
                        hbm[limb(slot(scratch_base, ACC1), m) + i], Q[m]);
            count_cycle(counters, CycleClass::ExternalProduct, true);
        }
    }
    if (enable_debug) {
        debug_copy(hbm, debug_cursor, slot(scratch_base, MU0), 2, Q_COUNT - 1);
    }

    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    rescale_one(hbm, slot(scratch_base, MU0), scratch_base,
                ntt_forward_q_base, ntt_inverse_q_base, counters, phase_events);
    rescale_one(hbm, slot(scratch_base, MU1), scratch_base,
                ntt_forward_q_base, ntt_inverse_q_base, counters, phase_events);
    if (enable_debug) {
        debug_copy(hbm, debug_cursor, slot(scratch_base, MU0), 2, Q_COUNT - 2);
    }

    // Compact result followed by the n repeated Gamma components.
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
    set_timing_class(counters, TimingClass::Other);
    for (unsigned component = 0; component < OUTPUT_COMPONENTS; ++component) {
        addr_t source = component == 0 ? slot(scratch_base, MU0) : slot(scratch_base, MU1);
        for (unsigned m = 0; m < Q_COUNT - 1; ++m) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[output_base + (static_cast<addr_t>(component) * (Q_COUNT - 1) + m) * N + i] =
                    canonical(hbm[limb(source, m) + i], Q[m]);
                count_cycle(counters, CycleClass::Other, true);
            }
        }
    }

    hbm[report_base + 0] = counters.decomposition;
    hbm[report_base + 1] = counters.external_product;
    hbm[report_base + 2] = counters.ntt_intt;
    hbm[report_base + 3] = counters.modup_down;
    hbm[report_base + 4] = counters.hbm_stall;
    hbm[report_base + 5] = counters.other;
    hbm[report_base + 6] = counters.total;
    emit_phase_event(phase_events, CycleClass::Idle, TimingClass::Idle);
    finish_counters(counters);
}
