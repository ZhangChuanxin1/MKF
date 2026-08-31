#include <ap_int.h>

#include "../include/mkf_cycle_counters.hpp"
#include "../include/par1_bfv_hls_constants.hpp"
#include "par1_montgomery.hpp"

namespace mkf_bfv_u55c {
using namespace par1_bfv_hls;
using word = ap_uint<64>;
using addr_t = unsigned long long;

static constexpr unsigned PARTIES = 16;
static constexpr unsigned COMPONENTS = 17;
static constexpr unsigned OUTPUT_COMPONENTS = 18;
static constexpr addr_t Q_WORDS = Q_COUNT * N;
static constexpr addr_t SLOT_WORDS = R_COUNT * N;

enum Slot : unsigned {
    COEFF = 0, INPUT_NTT = 1, DECOMP0 = 2, DECOMP1 = 3, DECOMP2 = 4,
    ACC0 = 5, ACC1 = 6, TEMP = 7,
    LEFT0 = 8, LEFT1 = 9, RIGHT0 = 10, RIGHT1 = 11,
    LIFT_L0 = 12, LIFT_L1 = 13, LIFT_R0 = 14, LIFT_R1 = 15,
    PROD0 = 16, PROD1 = 17, PROD2 = 18,
    AUX0 = 19, AUX1 = 20
};

static inline addr_t slot(addr_t base, unsigned index) {
    return base + static_cast<addr_t>(index) * SLOT_WORDS;
}
static inline addr_t limb(addr_t poly, unsigned modulus) {
    return poly + static_cast<addr_t>(modulus) * N;
}
static inline word canonical(word x, word q) {
#pragma HLS INLINE
    // Every caller supplies an unsigned 64-bit residue.  The former eight
    // conditional subtractions only normalized values below 9q.  BFV's
    // lazy RNS paths can temporarily exceed that bound; a later subtraction
    // then wrapped at 64 bits and changed the residue modulo q.  Since the
    // smallest P1 modulus is above 2^53, binary conditional subtraction from
    // 2^10*q down to q reduces every possible 64-bit input exactly.
    ap_uint<75> value = x;
    for (int shift = 10; shift >= 0; --shift) {
#pragma HLS UNROLL
        ap_uint<75> multiple = static_cast<ap_uint<75>>(q) << shift;
        if (value >= multiple) value -= multiple;
    }
    return static_cast<word>(value);
}
static inline word add_mod(word a, word b, word q) {
#pragma HLS INLINE
    ap_uint<65> s = static_cast<ap_uint<65>>(canonical(a, q)) + canonical(b, q);
    if (s >= q) s -= q;
    return static_cast<word>(s);
}
static inline word sub_mod(word a, word b, word q) {
#pragma HLS INLINE
    a = canonical(a, q); b = canonical(b, q);
    return a >= b ? word(a - b) : word(a + q - b);
}
static inline word mont_mul(word a, word b, word q, word q_inv) {
#pragma HLS INLINE
    return canonical(mred64(a, b, q, q_inv), q);
}

static void ntt_limb(word *hbm, addr_t poly, addr_t twiddle,
                     word q, word q_inv, word n_inv, bool inverse,
                     CycleCounters &counters,
                     hls::stream<ap_uint<6>> &phase_events,
                     CycleClass resume_stage, TimingClass resume_timing) {
    emit_phase_event(phase_events, CycleClass::NttIntt, resume_timing);
    static word ntt_a[N], ntt_b[N], ntt_tw[N];
#pragma HLS BIND_STORAGE variable=ntt_a type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ntt_b type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=ntt_tw type=ram_2p impl=bram
    const word two_q = q << 1, four_q = q << 2;
    for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        ntt_a[i] = canonical(hbm[poly + i], q);
        ntt_tw[i] = hbm[twiddle + i];
    }
    bool src_a = true;
    if (!inverse) {
        unsigned t = N >> 1, log_t = LOG_N - 1;
        for (unsigned m = 1; m < N; m <<= 1) {
            for (unsigned bf = 0; bf < N / 2; ++bf) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ntt_a inter false
#pragma HLS DEPENDENCE variable=ntt_b inter false
                unsigned group = bf >> log_t, j = bf & (t - 1);
                unsigned p = (group * t << 1) + j, p2 = p + t;
                word u = src_a ? ntt_a[p] : ntt_b[p];
                if (u >= four_q) u -= four_q;
                word v = mred64(src_a ? ntt_a[p2] : ntt_b[p2], ntt_tw[m + group], q, q_inv);
                if (src_a) { ntt_b[p] = u + v; ntt_b[p2] = u + two_q - v; }
                else { ntt_a[p] = u + v; ntt_a[p2] = u + two_q - v; }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a; t >>= 1; --log_t;
        }
    } else {
        unsigned t = 1, log_t = 0;
        for (unsigned m = N >> 1; m; m >>= 1) {
            for (unsigned bf = 0; bf < N / 2; ++bf) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=ntt_a inter false
#pragma HLS DEPENDENCE variable=ntt_b inter false
                unsigned group = bf >> log_t, j = bf & (t - 1);
                unsigned p = (group * t << 1) + j, p2 = p + t;
                word u = src_a ? ntt_a[p] : ntt_b[p];
                word v = src_a ? ntt_a[p2] : ntt_b[p2];
                ap_uint<65> x = static_cast<ap_uint<65>>(u) + v;
                if (x >= two_q) x -= two_q;
                word y = u + four_q - v;
                word z = mred64_lazy(y, ntt_tw[m + group], q, q_inv);
                if (src_a) { ntt_b[p] = static_cast<word>(x); ntt_b[p2] = z; }
                else { ntt_a[p] = static_cast<word>(x); ntt_a[p2] = z; }
                count_cycle(counters, CycleClass::NttIntt, true);
            }
            src_a = !src_a; t <<= 1; ++log_t;
        }
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word x = src_a ? ntt_a[i] : ntt_b[i];
            if (src_a) ntt_b[i] = mred64(x, n_inv, q, q_inv);
            else ntt_a[i] = mred64(x, n_inv, q, q_inv);
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

static inline word decomp_modup(unsigned beta, unsigned target, word x0, word x1) {
#pragma HLS INLINE
    unsigned s0 = beta * 2, s1 = s0 + 1;
    word y0 = mont_mul(x0, DECOMP_INV[beta][0], Q[s0], MRED_Q[s0]);
    word y1 = mont_mul(x1, DECOMP_INV[beta][1], Q[s1], MRED_Q[s1]);
    ap_uint<128> lhs = static_cast<ap_uint<128>>(y0) * Q[s1] + static_cast<ap_uint<128>>(y1) * Q[s0];
    ap_uint<128> rhs = static_cast<ap_uint<128>>(Q[s0]) * Q[s1];
    word out = mont_mul(y0, DECOMP_QSTAR[beta][target][0], QP[target], MRED_QP[target]);
    out = add_mod(out, mont_mul(y1, DECOMP_QSTAR[beta][target][1], QP[target], MRED_QP[target]), QP[target]);
    if (lhs >= rhs) out = add_mod(out, DECOMP_MINUS_Q[beta][target], QP[target]);
    return out;
}

static inline word p_to_q(unsigned target, word x0, word x1) {
#pragma HLS INLINE
    word y0 = mont_mul(x0, P2Q_INV[0], P[0], MRED_P[0]);
    word y1 = mont_mul(x1, P2Q_INV[1], P[1], MRED_P[1]);
    ap_uint<128> lhs = static_cast<ap_uint<128>>(y0) * P[1] + static_cast<ap_uint<128>>(y1) * P[0];
    ap_uint<128> rhs = static_cast<ap_uint<128>>(P[0]) * P[1];
    word out = mont_mul(y0, P2Q_QSTAR[target][0], Q[target], MRED_Q[target]);
    out = add_mod(out, mont_mul(y1, P2Q_QSTAR[target][1], Q[target], MRED_Q[target]), Q[target]);
    if (lhs >= rhs) out = add_mod(out, P2Q_MINUS_P[target], Q[target]);
    return out;
}

static inline word q_to_qm(unsigned target, const word values[Q_COUNT]) {
#pragma HLS INLINE
    word y[Q_COUNT];
#pragma HLS ARRAY_PARTITION variable=y complete
    ap_uint<128> approx = 0;
    for (unsigned i = 0; i < Q_COUNT; ++i) {
#pragma HLS UNROLL
        y[i] = mont_mul(values[i], Q2M_INV[i], Q[i], MRED_Q[i]);
        approx += static_cast<ap_uint<128>>(y[i]) * RECIP96_Q[i];
    }
    unsigned v = static_cast<unsigned>(approx >> 96);
    word out = 0;
    for (unsigned i = 0; i < Q_COUNT; ++i) {
#pragma HLS UNROLL
        out = add_mod(out, mont_mul(y[i], Q2M_QSTAR[target][i], QM[target], MRED_QM[target]), QM[target]);
    }
    for (unsigned i = 0; i < Q_COUNT; ++i) {
#pragma HLS UNROLL
        if (i < v) out = add_mod(out, Q2M_MINUS_Q[target], QM[target]);
    }
    return out;
}

static inline word qm_to_q(unsigned target, const word values[QM_COUNT]) {
#pragma HLS INLINE
    word y[QM_COUNT];
#pragma HLS ARRAY_PARTITION variable=y complete
    ap_uint<128> approx = 0;
    for (unsigned i = 0; i < QM_COUNT; ++i) {
#pragma HLS UNROLL
        y[i] = mont_mul(values[i], M2Q_INV[i], QM[i], MRED_QM[i]);
        approx += static_cast<ap_uint<128>>(y[i]) * RECIP96_QM[i];
    }
    unsigned v = static_cast<unsigned>(approx >> 96);
    word out = 0;
    for (unsigned i = 0; i < QM_COUNT; ++i) {
#pragma HLS UNROLL
        out = add_mod(out, mont_mul(y[i], M2Q_QSTAR[target][i], Q[target], MRED_Q[target]), Q[target]);
    }
    for (unsigned i = 0; i < QM_COUNT; ++i) {
#pragma HLS UNROLL
        if (i < v) out = add_mod(out, M2Q_MINUS_M[target], Q[target]);
    }
    return out;
}

static void decompose(word *hbm, addr_t input, bool input_ntt, addr_t scratch,
                      addr_t fq, addr_t iq, addr_t fp, addr_t ip,
                      CycleCounters &counters,
                      hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
    set_timing_class(counters, TimingClass::Decomposition);
    addr_t coeff = slot(scratch, COEFF), inputNtt = slot(scratch, INPUT_NTT);
    for (unsigned m = 0; m < Q_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            word x = hbm[input + static_cast<addr_t>(m) * N + i];
            hbm[limb(coeff, m) + i] = x;
            hbm[limb(inputNtt, m) + i] = x;
        }
        if (input_ntt)
            ntt_limb(hbm, limb(coeff, m), iq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], true, counters, phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
        else
            ntt_limb(hbm, limb(inputNtt, m), fq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], false, counters, phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
    }
    for (unsigned d = 0; d < BETA; ++d) {
        addr_t out = slot(scratch, DECOMP0 + d);
        for (unsigned target = 0; target < QP_COUNT; ++target) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[limb(out, target) + i] = decomp_modup(d, target,
                    hbm[limb(coeff, d * 2) + i], hbm[limb(coeff, d * 2 + 1) + i]);
                count_cycle(counters, CycleClass::Decomposition, true);
            }
            if (target == d * 2) {
                for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                    hbm[limb(out, target) + i] = hbm[limb(inputNtt, target) + i];
                }
            } else if (target < Q_COUNT)
                ntt_limb(hbm, limb(out, target), fq + static_cast<addr_t>(target) * N, Q[target], MRED_Q[target], N_INV_Q[target], false, counters, phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
            else {
                unsigned p = target - Q_COUNT;
                ntt_limb(hbm, limb(out, target), fp + static_cast<addr_t>(p) * N, P[p], MRED_P[p], N_INV_P[p], false, counters, phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
            }
        }
    }
    (void)ip;
}

static inline addr_t key_poly(addr_t base, unsigned key, unsigned d, unsigned out, unsigned m) {
    return base + ((((static_cast<addr_t>(key) * BETA + d) * 2 + out) * QP_COUNT + m) * N);
}

static void external(word *hbm, addr_t keys, unsigned key, addr_t scratch,
                     CycleCounters &counters,
                     hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(counters, TimingClass::ExternalProduct);
    for (unsigned out = 0; out < 2; ++out) {
        addr_t acc = slot(scratch, ACC0 + out);
        for (unsigned m = 0; m < QP_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word sum = 0;
                for (unsigned d = 0; d < BETA; ++d) {
#pragma HLS UNROLL
                    sum = add_mod(sum, mont_mul(hbm[limb(slot(scratch, DECOMP0 + d), m) + i],
                        hbm[key_poly(keys, key, d, out, m) + i], QP[m], MRED_QP[m]), QP[m]);
                }
                hbm[limb(acc, m) + i] = sum;
                count_cycle(counters, CycleClass::ExternalProduct, true);
            }
        }
    }
}

static void moddown_key(word *hbm, bool output_ntt, addr_t scratch,
                        addr_t fq, addr_t iq, addr_t ip,
                        CycleCounters &counters,
                        hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    addr_t temp = slot(scratch, TEMP);
    for (unsigned out = 0; out < 2; ++out) {
        addr_t acc = slot(scratch, ACC0 + out);
        for (unsigned p = 0; p < P_COUNT; ++p)
            ntt_limb(hbm, limb(acc, Q_COUNT + p), ip + static_cast<addr_t>(p) * N, P[p], MRED_P[p], N_INV_P[p], true, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
        if (!output_ntt) {
            for (unsigned m = 0; m < Q_COUNT; ++m)
                ntt_limb(hbm, limb(acc, m), iq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], true, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
        }
        for (unsigned m = 0; m < Q_COUNT; ++m) {
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[limb(temp, m) + i] = p_to_q(m, hbm[limb(acc, Q_COUNT) + i], hbm[limb(acc, Q_COUNT + 1) + i]);
                count_cycle(counters, CycleClass::ModUpDown, true);
            }
            if (output_ntt)
                ntt_limb(hbm, limb(temp, m), fq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], false, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
            word neg = Q[m] - P_INV_Q[m];
            for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
                word delta = sub_mod(hbm[limb(temp, m) + i], hbm[limb(acc, m) + i], Q[m]);
                hbm[limb(acc, m) + i] = mont_mul(delta, neg, Q[m], MRED_Q[m]);
                count_cycle(counters, CycleClass::ModUpDown, true);
            }
        }
    }
}

static void switch_one(word *hbm, addr_t input, addr_t keys, unsigned key,
                       bool input_ntt, bool output_ntt, addr_t scratch,
                       addr_t fq, addr_t iq, addr_t fp, addr_t ip,
                       CycleCounters &counters,
                       hls::stream<ap_uint<6>> &phase_events) {
    decompose(hbm, input, input_ntt, scratch, fq, iq, fp, ip, counters, phase_events);
    external(hbm, keys, key, scratch, counters, phase_events);
    moddown_key(hbm, output_ntt, scratch, fq, iq, ip, counters, phase_events);
}

static inline addr_t input_poly(addr_t base, unsigned side, unsigned component) {
    return base + (static_cast<addr_t>(side) * COMPONENTS + component) * Q_WORDS;
}

static void fold(word *hbm, addr_t inputs, unsigned side, addr_t keys,
                 addr_t out0, addr_t out1, addr_t scratch,
                 addr_t fq, addr_t iq, addr_t fp, addr_t ip,
                 CycleCounters &counters,
                 hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
    for (unsigned m = 0; m < Q_COUNT; ++m) for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        hbm[limb(out0, m) + i] = canonical(hbm[input_poly(inputs, side, 0) + static_cast<addr_t>(m) * N + i], Q[m]);
        hbm[limb(out1, m) + i] = 0;
    }
    for (unsigned party = 0; party < PARTIES; ++party) {
        switch_one(hbm, input_poly(inputs, side, party + 1), keys, party, false, false,
                   scratch, fq, iq, fp, ip, counters, phase_events);
        emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
        set_timing_class(counters, TimingClass::ExternalProduct);
        for (unsigned m = 0; m < Q_COUNT; ++m) for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(out0, m) + i] = add_mod(hbm[limb(out0, m) + i], hbm[limb(slot(scratch, ACC0), m) + i], Q[m]);
            hbm[limb(out1, m) + i] = add_mod(hbm[limb(out1, m) + i], hbm[limb(slot(scratch, ACC1), m) + i], Q[m]);
            count_cycle(counters, CycleClass::ExternalProduct, true);
        }
    }
}

static void lift_left(word *hbm, addr_t input, addr_t output,
                      addr_t fq, addr_t fqm, CycleCounters &counters,
                      hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    for (unsigned i = 0; i < N; ++i) {
        word values[Q_COUNT];
#pragma HLS ARRAY_PARTITION variable=values complete
        for (unsigned m = 0; m < Q_COUNT; ++m) {
#pragma HLS UNROLL
            values[m] = hbm[limb(input, m) + i];
            hbm[limb(output, m) + i] = values[m];
        }
        for (unsigned m = 0; m < QM_COUNT; ++m) hbm[limb(output, Q_COUNT + m) + i] = q_to_qm(m, values);
        count_cycle(counters, CycleClass::ModUpDown, true);
    }
    for (unsigned m = 0; m < R_COUNT; ++m) {
        if (m < Q_COUNT) ntt_limb(hbm, limb(output, m), fq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], false, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
        else { unsigned k = m - Q_COUNT; ntt_limb(hbm, limb(output, m), fqm + static_cast<addr_t>(k) * N, QM[k], MRED_QM[k], N_INV_QM[k], false, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown); }
    }
}

static void lift_right(word *hbm, addr_t input, addr_t output,
                       addr_t fq, addr_t fqm, CycleCounters &counters,
                       hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    for (unsigned i = 0; i < N; ++i) {
        word scaledQ[Q_COUNT];
#pragma HLS ARRAY_PARTITION variable=scaledQ complete
        for (unsigned m = 0; m < Q_COUNT; ++m) {
#pragma HLS UNROLL
            scaledQ[m] = mont_mul(hbm[limb(input, m) + i], QM_MFORM_Q[m], Q[m], MRED_Q[m]);
        }
        word qmValues[QM_COUNT];
#pragma HLS ARRAY_PARTITION variable=qmValues complete
        for (unsigned m = 0; m < QM_COUNT; ++m) {
#pragma HLS UNROLL
            word lifted = q_to_qm(m, scaledQ);
            qmValues[m] = mont_mul(lifted, QM[m] - Q_INV_QM[m], QM[m], MRED_QM[m]);
            hbm[limb(output, Q_COUNT + m) + i] = qmValues[m];
        }
        for (unsigned m = 0; m < Q_COUNT; ++m) hbm[limb(output, m) + i] = qm_to_q(m, qmValues);
        count_cycle(counters, CycleClass::ModUpDown, true);
    }
    for (unsigned m = 0; m < R_COUNT; ++m) {
        if (m < Q_COUNT) ntt_limb(hbm, limb(output, m), fq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], false, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
        else { unsigned k = m - Q_COUNT; ntt_limb(hbm, limb(output, m), fqm + static_cast<addr_t>(k) * N, QM[k], MRED_QM[k], N_INV_QM[k], false, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown); }
    }
}

static void tensor(word *hbm, addr_t scratch, CycleCounters &counters,
                   hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Tensor);
    set_timing_class(counters, TimingClass::Tensor);
    addr_t a0 = slot(scratch, LIFT_L0), a1 = slot(scratch, LIFT_L1);
    addr_t b0 = slot(scratch, LIFT_R0), b1 = slot(scratch, LIFT_R1);
    for (unsigned m = 0; m < R_COUNT; ++m) for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        word x0 = canonical(hbm[limb(a0, m) + i], R[m]), x1 = canonical(hbm[limb(a1, m) + i], R[m]);
        word y0 = canonical(hbm[limb(b0, m) + i], R[m]), y1 = canonical(hbm[limb(b1, m) + i], R[m]);
        word p0 = mont_mul(mont_mul(x0, R2_R[m], R[m], MRED_R[m]), y0, R[m], MRED_R[m]);
        word p2 = mont_mul(mont_mul(x1, R2_R[m], R[m], MRED_R[m]), y1, R[m], MRED_R[m]);
        word p1 = mont_mul(mont_mul(add_mod(x0, x1, R[m]), R2_R[m], R[m], MRED_R[m]), add_mod(y0, y1, R[m]), R[m], MRED_R[m]);
        p1 = sub_mod(sub_mod(p1, p0, R[m]), p2, R[m]);
        hbm[limb(slot(scratch, PROD0), m) + i] = p0;
        hbm[limb(slot(scratch, PROD1), m) + i] = p1;
        hbm[limb(slot(scratch, PROD2), m) + i] = p2;
        count_cycle(counters, CycleClass::Tensor, true);
    }
}

static void quantize_one(word *hbm, addr_t product, addr_t output,
                         addr_t iq, addr_t iqm, CycleCounters &counters,
                         hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(counters, TimingClass::ModUpDown);
    for (unsigned m = 0; m < R_COUNT; ++m) {
        for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(product, m) + i] = mont_mul(hbm[limb(product, m) + i], T_MFORM_R[m], R[m], MRED_R[m]);
        }
        if (m < Q_COUNT) ntt_limb(hbm, limb(product, m), iq + static_cast<addr_t>(m) * N, Q[m], MRED_Q[m], N_INV_Q[m], true, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
        else { unsigned k = m - Q_COUNT; ntt_limb(hbm, limb(product, m), iqm + static_cast<addr_t>(k) * N, QM[k], MRED_QM[k], N_INV_QM[k], true, counters, phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown); }
    }
    for (unsigned i = 0; i < N; ++i) {
        word qmValues[QM_COUNT];
#pragma HLS ARRAY_PARTITION variable=qmValues complete
        for (unsigned k = 0; k < QM_COUNT; ++k) qmValues[k] = hbm[limb(product, Q_COUNT + k) + i];
        for (unsigned m = 0; m < Q_COUNT; ++m) {
            word lifted = qm_to_q(m, qmValues);
            word delta = sub_mod(lifted, hbm[limb(product, m) + i], Q[m]);
            hbm[limb(output, m) + i] = mont_mul(delta, Q[m] - QM_INV_Q[m], Q[m], MRED_Q[m]);
        }
        count_cycle(counters, CycleClass::ModUpDown, true);
    }
}

static void debug_compact(word *hbm, addr_t &cursor, addr_t start,
                          unsigned polys, unsigned moduli, const word *mods) {
    for (unsigned p = 0; p < polys; ++p) for (unsigned m = 0; m < moduli; ++m) for (unsigned i = 0; i < N; ++i)
        hbm[cursor++] = canonical(hbm[limb(start + static_cast<addr_t>(p) * SLOT_WORDS, m) + i], mods[m]);
}
} // namespace mkf_bfv_u55c

extern "C" void par1_mkf_bfv_kernel(
    ap_uint<64> *hbm, unsigned long long input_base, unsigned long long key_base,
    unsigned long long output_base, unsigned long long scratch_base,
    unsigned long long fwd_q, unsigned long long inv_q,
    unsigned long long fwd_p, unsigned long long inv_p,
    unsigned long long fwd_qm, unsigned long long inv_qm,
    unsigned long long debug_base, unsigned int enable_debug,
    unsigned long long report_base, hls::stream<ap_uint<6>> &phase_events) {
#pragma HLS INTERFACE m_axi port=hbm offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=hbm bundle=control
#pragma HLS INTERFACE s_axilite port=input_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_base bundle=control
#pragma HLS INTERFACE s_axilite port=output_base bundle=control
#pragma HLS INTERFACE s_axilite port=scratch_base bundle=control
#pragma HLS INTERFACE s_axilite port=fwd_q bundle=control
#pragma HLS INTERFACE s_axilite port=inv_q bundle=control
#pragma HLS INTERFACE s_axilite port=fwd_p bundle=control
#pragma HLS INTERFACE s_axilite port=inv_p bundle=control
#pragma HLS INTERFACE s_axilite port=fwd_qm bundle=control
#pragma HLS INTERFACE s_axilite port=inv_qm bundle=control
#pragma HLS INTERFACE s_axilite port=debug_base bundle=control
#pragma HLS INTERFACE s_axilite port=enable_debug bundle=control
#pragma HLS INTERFACE s_axilite port=report_base bundle=control
#pragma HLS INTERFACE axis port=phase_events
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace mkf_bfv_u55c;
    CycleCounters counters; reset_counters(counters);
    fold(hbm, input_base, 0, key_base, slot(scratch_base, LEFT0), slot(scratch_base, LEFT1), scratch_base, fwd_q, inv_q, fwd_p, inv_p, counters, phase_events);
    fold(hbm, input_base, 1, key_base, slot(scratch_base, RIGHT0), slot(scratch_base, RIGHT1), scratch_base, fwd_q, inv_q, fwd_p, inv_p, counters, phase_events);
    addr_t dc = debug_base;
    if (enable_debug) debug_compact(hbm, dc, slot(scratch_base, LEFT0), 4, Q_COUNT, Q);
    lift_left(hbm, slot(scratch_base, LEFT0), slot(scratch_base, LIFT_L0), fwd_q, fwd_qm, counters, phase_events);
    lift_left(hbm, slot(scratch_base, LEFT1), slot(scratch_base, LIFT_L1), fwd_q, fwd_qm, counters, phase_events);
    lift_right(hbm, slot(scratch_base, RIGHT0), slot(scratch_base, LIFT_R0), fwd_q, fwd_qm, counters, phase_events);
    lift_right(hbm, slot(scratch_base, RIGHT1), slot(scratch_base, LIFT_R1), fwd_q, fwd_qm, counters, phase_events);
    if (enable_debug) debug_compact(hbm, dc, slot(scratch_base, LIFT_L0), 4, R_COUNT, R);
    tensor(hbm, scratch_base, counters, phase_events);
    if (enable_debug) debug_compact(hbm, dc, slot(scratch_base, PROD0), 3, R_COUNT, R);
    quantize_one(hbm, slot(scratch_base, PROD0), slot(scratch_base, LEFT0), inv_q, inv_qm, counters, phase_events);
    quantize_one(hbm, slot(scratch_base, PROD1), slot(scratch_base, LEFT1), inv_q, inv_qm, counters, phase_events);
    quantize_one(hbm, slot(scratch_base, PROD2), slot(scratch_base, RIGHT0), inv_q, inv_qm, counters, phase_events);
    if (enable_debug) debug_compact(hbm, dc, slot(scratch_base, LEFT0), 3, Q_COUNT, Q);
    switch_one(hbm, slot(scratch_base, RIGHT0), key_base, PARTIES, false, false, scratch_base, fwd_q, inv_q, fwd_p, inv_p, counters, phase_events);
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(counters, TimingClass::ExternalProduct);
    for (unsigned m = 0; m < Q_COUNT; ++m) for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        hbm[limb(slot(scratch_base, LEFT0), m) + i] = add_mod(hbm[limb(slot(scratch_base, LEFT0), m) + i], hbm[limb(slot(scratch_base, ACC0), m) + i], Q[m]);
        hbm[limb(slot(scratch_base, LEFT1), m) + i] = add_mod(hbm[limb(slot(scratch_base, LEFT1), m) + i], hbm[limb(slot(scratch_base, ACC1), m) + i], Q[m]);
        count_cycle(counters, CycleClass::ExternalProduct, true);
    }
    if (enable_debug) debug_compact(hbm, dc, slot(scratch_base, LEFT0), 2, Q_COUNT, Q);
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
    set_timing_class(counters, TimingClass::Other);
    for (unsigned component = 0; component < OUTPUT_COMPONENTS; ++component) {
        addr_t source = component == 0 ? slot(scratch_base, LEFT0) : slot(scratch_base, LEFT1);
        for (unsigned m = 0; m < Q_COUNT; ++m) for (unsigned i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[output_base + (static_cast<addr_t>(component) * Q_COUNT + m) * N + i] = canonical(hbm[limb(source, m) + i], Q[m]);
            count_cycle(counters, CycleClass::Other, true);
        }
    }
    hbm[report_base] = counters.decomposition; hbm[report_base + 1] = counters.external_product;
    hbm[report_base + 2] = counters.ntt_intt; hbm[report_base + 3] = counters.modup_down;
    hbm[report_base + 4] = counters.hbm_stall; hbm[report_base + 5] = counters.other;
    hbm[report_base + 6] = counters.total;
    emit_phase_event(phase_events, CycleClass::Idle, TimingClass::Idle);
    finish_counters(counters);
}
