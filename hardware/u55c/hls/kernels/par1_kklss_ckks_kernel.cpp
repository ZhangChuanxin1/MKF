#include "par1_mkf_ckks_kernel.cpp"

// Exact KKLSS P1/n=16 multiplication-with-relinearization pipeline.  The
// included translation unit supplies the already validated P1 Montgomery,
// NTT/INTT and P->Q conversion primitives; this top implements the different
// KKLSS algebra and the alpha=1, beta=6 switching-key layout.
namespace kklss_ckks_u55c {
using namespace mkf_ckks_u55c;

static constexpr unsigned K_BETA = 6;
static constexpr unsigned K_COMPONENTS = 17;
static constexpr unsigned K_QP_WORDS = QP_COUNT * N;
static constexpr unsigned K_Q_WORDS = Q_COUNT * N;

enum KSlot : unsigned {
    K_COEFF=0, K_D0=1, K_D1=2, K_D2=3, K_D3=4, K_D4=5, K_D5=6,
    K_ACC=7, K_MODUP=8,
    K_X0=9, K_X1=10, K_X2=11, K_X3=12, K_X4=13, K_X5=14,
    K_Y0=15, K_Y1=16, K_Y2=17, K_Y3=18, K_Y4=19, K_Y5=20,
    K_OUT0=21, K_OUT_LAST=37, K_BASE_L=38, K_BASE_R=39, K_TEMP=40
};

static inline addr_t kslt(addr_t base, unsigned s) {
    return base + static_cast<addr_t>(s) * K_QP_WORDS;
}
static inline word r2_qp(unsigned m) {
    if (m < Q_COUNT) return R2_Q[m];
    return m == Q_COUNT ? word(17592051826944ULL) : word(16906086627934464ULL);
}
static inline addr_t kkey(addr_t base, unsigned party, unsigned component,
                          unsigned d, unsigned m) {
    addr_t p = ((static_cast<addr_t>(party) * 3 + component) * K_BETA + d);
    return base + (p * QP_COUNT + m) * N;
}
static inline addr_t ku(addr_t base, unsigned d, unsigned m) {
    return base + (static_cast<addr_t>(d) * QP_COUNT + m) * N;
}
static inline addr_t kin(addr_t base, unsigned side, unsigned component) {
    return base + (static_cast<addr_t>(side) * K_COMPONENTS + component) * K_Q_WORDS;
}

static void kcopy_q(word *hbm, addr_t src, addr_t dst, unsigned level) {
    for (unsigned m=0; m<=level; ++m)
        for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
            hbm[limb(dst,m)+i] = canonical(hbm[src+static_cast<addr_t>(m)*N+i], Q[m]);
        }
}

static void kdecompose(word *hbm, addr_t input, addr_t scratch,
                       addr_t fwd_q, addr_t fwd_p, CycleCounters &c,
                       hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Decomposition, TimingClass::Decomposition);
    set_timing_class(c, TimingClass::Decomposition);
    for (unsigned d=0; d<K_BETA; ++d) {
        addr_t out=kslt(scratch,K_D0+d);
        for (unsigned m=0; m<QP_COUNT; ++m) {
            word q=QP[m];
            for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
                word x=hbm[input+static_cast<addr_t>(d)*N+i];
                hbm[limb(out,m)+i]=canonical(x,q);
                count_cycle(c,CycleClass::Decomposition,true);
            }
            if (m<Q_COUNT)
                ntt_limb(hbm,limb(out,m),fwd_q+static_cast<addr_t>(m)*N,
                         Q[m],MRED_Q[m],N_INV_Q[m],false,c,phase_events,
                         CycleClass::Decomposition,TimingClass::Decomposition);
            else {
                unsigned p=m-Q_COUNT;
                ntt_limb(hbm,limb(out,m),fwd_p+static_cast<addr_t>(p)*N,
                         P[p],MRED_P[p],N_INV_P[p],false,c,phase_events,
                         CycleClass::Decomposition,TimingClass::Decomposition);
            }
        }
    }
}

static void kzero_vectors(word *hbm, addr_t scratch) {
    for (unsigned v=K_X0; v<=K_Y5; ++v)
        for (unsigned m=0; m<QP_COUNT; ++m)
            for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
                hbm[limb(kslt(scratch,v),m)+i]=0;
            }
}

static void kacc_vector(word *hbm, addr_t key_base, unsigned party,
                        unsigned component, unsigned vector0, addr_t scratch,
                        CycleCounters &c,hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(c, TimingClass::ExternalProduct);
    for (unsigned d=0; d<K_BETA; ++d)
        for (unsigned m=0; m<QP_COUNT; ++m) {
            word q=QP[m], qi=MRED_QP[m];
            for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
                addr_t dst=limb(kslt(scratch,vector0+d),m)+i;
                word a=hbm[limb(kslt(scratch,K_D0+d),m)+i];
                word k=hbm[kkey(key_base,party,component,d,m)+i];
                hbm[dst]=add_mod(hbm[dst],mont_mul(a,k,q,qi),q);
                count_cycle(c,CycleClass::ExternalProduct,true);
            }
        }
}

static void kmform_vectors(word *hbm, addr_t scratch, CycleCounters &c,
                           hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(c, TimingClass::ExternalProduct);
    for (unsigned v=K_X0; v<=K_Y5; ++v)
        for (unsigned m=0; m<QP_COUNT; ++m) {
            word q=QP[m], qi=MRED_QP[m], r2=r2_qp(m);
            for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
                addr_t a=limb(kslt(scratch,v),m)+i;
                hbm[a]=mont_mul(hbm[a],r2,q,qi);
                count_cycle(c,CycleClass::ExternalProduct,true);
            }
        }
}

static void kexternal(word *hbm, addr_t source0, bool source_is_crs,
                      addr_t source_base, unsigned party, unsigned component,
                      addr_t scratch, addr_t inv_q, addr_t inv_p,
                      CycleCounters &c,hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(c, TimingClass::ExternalProduct);
    addr_t acc=kslt(scratch,K_ACC);
    for (unsigned m=0; m<QP_COUNT; ++m) {
        emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
        set_timing_class(c, TimingClass::ExternalProduct);
        word q=QP[m], qi=MRED_QP[m];
        for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
            word sum=0;
            for (unsigned d=0; d<K_BETA; ++d) {
#pragma HLS UNROLL
                word a=hbm[limb(kslt(scratch,K_D0+d),m)+i];
                addr_t ka = source_is_crs ? ku(source_base,d,m)
                    : (source0 ? limb(kslt(scratch,source0+d),m)
                               : kkey(source_base,party,component,d,m));
                sum=add_mod(sum,mont_mul(a,hbm[ka+i],q,qi),q);
            }
            hbm[limb(acc,m)+i]=sum;
            count_cycle(c,CycleClass::ExternalProduct,true);
        }
        set_timing_class(c, TimingClass::ModUpDown);
        if (m<Q_COUNT)
            ntt_limb(hbm,limb(acc,m),inv_q+static_cast<addr_t>(m)*N,
                     Q[m],MRED_Q[m],N_INV_Q[m],true,c,phase_events,
                     CycleClass::ModUpDown,TimingClass::ModUpDown);
        else {
            unsigned p=m-Q_COUNT;
            ntt_limb(hbm,limb(acc,m),inv_p+static_cast<addr_t>(p)*N,
                     P[p],MRED_P[p],N_INV_P[p],true,c,phase_events,
                     CycleClass::ModUpDown,TimingClass::ModUpDown);
        }
    }
    addr_t tmp=kslt(scratch,K_MODUP);
    for (unsigned m=0; m<Q_COUNT; ++m) {
        for (unsigned i=0; i<N; ++i) {
#pragma HLS PIPELINE II=1
            word p0=hbm[limb(acc,Q_COUNT)+i], p1=hbm[limb(acc,Q_COUNT+1)+i];
            word y=modup_p_to_q(m,p0,p1);
            word x=hbm[limb(acc,m)+i];
            hbm[limb(tmp,m)+i]=mont_mul(sub_mod(x,y,Q[m]),P_INV_Q[m],Q[m],MRED_Q[m]);
            count_cycle(c,CycleClass::ModUpDown,true);
        }
    }
}

static void kbase_tensor(word *hbm, addr_t input, addr_t scratch,
                         addr_t fwd_q, addr_t inv_q, CycleCounters &c,
                         hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Tensor);
    set_timing_class(c, TimingClass::Tensor);
    addr_t l0=kslt(scratch,K_BASE_L), r0=kslt(scratch,K_BASE_R);
    kcopy_q(hbm,kin(input,0,0),l0,Q_COUNT-1);
    kcopy_q(hbm,kin(input,1,0),r0,Q_COUNT-1);
    for (unsigned m=0;m<Q_COUNT;++m) {
        ntt_limb(hbm,limb(l0,m),fwd_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,phase_events,CycleClass::Other,TimingClass::Tensor);
        ntt_limb(hbm,limb(r0,m),fwd_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,phase_events,CycleClass::Other,TimingClass::Tensor);
        addr_t out=kslt(scratch,K_OUT0);
        for (unsigned i=0;i<N;++i) {
#pragma HLS PIPELINE II=1
            word lm=mont_mul(hbm[limb(l0,m)+i],R2_Q[m],Q[m],MRED_Q[m]);
            hbm[limb(out,m)+i]=mont_mul(lm,hbm[limb(r0,m)+i],Q[m],MRED_Q[m]);
            count_cycle(c,CycleClass::Tensor,true);
        }
        ntt_limb(hbm,limb(out,m),inv_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],true,c,phase_events,CycleClass::Other,TimingClass::Tensor);
    }
    for (unsigned party=0;party<PARTIES;++party) {
        addr_t a=kslt(scratch,K_COEFF), b=kslt(scratch,K_TEMP), out=kslt(scratch,K_OUT0+party+1);
        kcopy_q(hbm,kin(input,0,party+1),a,Q_COUNT-1);
        kcopy_q(hbm,kin(input,1,party+1),b,Q_COUNT-1);
        for (unsigned m=0;m<Q_COUNT;++m) {
            ntt_limb(hbm,limb(a,m),fwd_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,phase_events,CycleClass::Other,TimingClass::Tensor);
            ntt_limb(hbm,limb(b,m),fwd_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,phase_events,CycleClass::Other,TimingClass::Tensor);
            for (unsigned i=0;i<N;++i) {
#pragma HLS PIPELINE II=1
                word lm=mont_mul(hbm[limb(l0,m)+i],R2_Q[m],Q[m],MRED_Q[m]);
                word rm=mont_mul(hbm[limb(r0,m)+i],R2_Q[m],Q[m],MRED_Q[m]);
                word z=mont_mul(lm,hbm[limb(b,m)+i],Q[m],MRED_Q[m]);
                z=add_mod(z,mont_mul(rm,hbm[limb(a,m)+i],Q[m],MRED_Q[m]),Q[m]);
                hbm[limb(out,m)+i]=z;
                count_cycle(c,CycleClass::Tensor,true);
            }
            ntt_limb(hbm,limb(out,m),inv_q+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],true,c,phase_events,CycleClass::Other,TimingClass::Tensor);
        }
    }
}

static void kadd_acc(word *hbm, addr_t dst, addr_t scratch, CycleCounters &c,
                     hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ExternalProduct, TimingClass::ExternalProduct);
    set_timing_class(c, TimingClass::ExternalProduct);
    addr_t src=kslt(scratch,K_MODUP);
    for(unsigned m=0;m<Q_COUNT;++m) for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
        addr_t d=limb(dst,m)+i;
        hbm[d]=add_mod(hbm[d],hbm[limb(src,m)+i],Q[m]);
        count_cycle(c,CycleClass::ExternalProduct,true);
    }
}

static void krescale_coeff(word *hbm, addr_t poly, CycleCounters &c,
                           hls::stream<ap_uint<6>> &phase_events) {
    emit_phase_event(phase_events, CycleClass::ModUpDown, TimingClass::ModUpDown);
    set_timing_class(c, TimingClass::ModUpDown);
    word p=Q[Q_COUNT-1], half=(p-1)>>1;
    for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
        word last=add_mod(hbm[limb(poly,Q_COUNT-1)+i],half,p);
        for(unsigned m=0;m<Q_COUNT-1;++m){
#pragma HLS UNROLL
            word qi=Q[m], hm=canonical(half,qi);
            word neg=canonical(hbm[limb(poly,m)+i],qi);
            word s=canonical(last,qi);
            s=add_mod(s,qi-hm,qi);
            s=sub_mod(s,neg,qi);
            hbm[limb(poly,m)+i]=mont_mul(s,RESCALE_5_TO_4[m],qi,MRED_Q[m]);
            count_cycle(c,CycleClass::ModUpDown,true);
        }
    }
}
}

extern "C" void par1_kklss_ckks_kernel(
    ap_uint<64>* hbm, unsigned long long input_base,
    unsigned long long key_base, unsigned long long crs_base,
    unsigned long long output_base, unsigned long long scratch_base,
    unsigned long long fwd_q, unsigned long long inv_q,
    unsigned long long fwd_p, unsigned long long inv_p,
    unsigned long long report_base, hls::stream<ap_uint<6>> &phase_events) {
#pragma HLS INTERFACE m_axi port=hbm offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=hbm bundle=control
#pragma HLS INTERFACE s_axilite port=input_base bundle=control
#pragma HLS INTERFACE s_axilite port=key_base bundle=control
#pragma HLS INTERFACE s_axilite port=crs_base bundle=control
#pragma HLS INTERFACE s_axilite port=output_base bundle=control
#pragma HLS INTERFACE s_axilite port=scratch_base bundle=control
#pragma HLS INTERFACE s_axilite port=fwd_q bundle=control
#pragma HLS INTERFACE s_axilite port=inv_q bundle=control
#pragma HLS INTERFACE s_axilite port=fwd_p bundle=control
#pragma HLS INTERFACE s_axilite port=inv_p bundle=control
#pragma HLS INTERFACE s_axilite port=report_base bundle=control
#pragma HLS INTERFACE axis port=phase_events
#pragma HLS INTERFACE s_axilite port=return bundle=control
    using namespace kklss_ckks_u55c;
    CycleCounters c; reset_counters(c); emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other); kzero_vectors(hbm,scratch_base);
    for(unsigned party=0;party<PARTIES;++party){
        kdecompose(hbm,kin(input_base,0,party+1),scratch_base,fwd_q,fwd_p,c,phase_events);
        kacc_vector(hbm,key_base,party,1,K_X0,scratch_base,c,phase_events);
        kdecompose(hbm,kin(input_base,1,party+1),scratch_base,fwd_q,fwd_p,c,phase_events);
        kacc_vector(hbm,key_base,party,0,K_Y0,scratch_base,c,phase_events);
    }
    kmform_vectors(hbm,scratch_base,c,phase_events);
    kbase_tensor(hbm,input_base,scratch_base,fwd_q,inv_q,c,phase_events);
    for(unsigned party=0;party<PARTIES;++party){
        kdecompose(hbm,kin(input_base,1,party+1),scratch_base,fwd_q,fwd_p,c,phase_events);
        kexternal(hbm,K_X0,false,0,0,0,scratch_base,inv_q,inv_p,c,phase_events);
        kadd_acc(hbm,kslt(scratch_base,K_OUT0+party+1),scratch_base,c,phase_events);
    }
    for(unsigned party=0;party<PARTIES;++party){
        kdecompose(hbm,kin(input_base,0,party+1),scratch_base,fwd_q,fwd_p,c,phase_events);
        kexternal(hbm,K_Y0,false,0,0,0,scratch_base,inv_q,inv_p,c,phase_events);
        kdecompose(hbm,kslt(scratch_base,K_MODUP),scratch_base,fwd_q,fwd_p,c,phase_events);
        kexternal(hbm,0,false,key_base,party,2,scratch_base,inv_q,inv_p,c,phase_events);
        kadd_acc(hbm,kslt(scratch_base,K_OUT0),scratch_base,c,phase_events);
        kexternal(hbm,0,true,crs_base,0,0,scratch_base,inv_q,inv_p,c,phase_events);
        kadd_acc(hbm,kslt(scratch_base,K_OUT0+party+1),scratch_base,c,phase_events);
    }
    emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
    set_timing_class(c, TimingClass::Other);
    for(unsigned component=0;component<K_COMPONENTS;++component){
        addr_t src=kslt(scratch_base,K_OUT0+component); krescale_coeff(hbm,src,c,phase_events);
        emit_phase_event(phase_events, CycleClass::Other, TimingClass::Other);
        for(unsigned m=0;m<Q_COUNT-1;++m) for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
            hbm[output_base+(static_cast<addr_t>(component)*(Q_COUNT-1)+m)*N+i]=canonical(hbm[limb(src,m)+i],Q[m]);
            count_cycle(c,CycleClass::Other,true);
        }
    }
    hbm[report_base+0]=c.decomposition; hbm[report_base+1]=c.external_product;
    hbm[report_base+2]=c.ntt_intt; hbm[report_base+3]=c.modup_down;
    hbm[report_base+4]=c.hbm_stall; hbm[report_base+5]=c.other; hbm[report_base+6]=c.total;
    emit_phase_event(phase_events, CycleClass::Idle, TimingClass::Idle);
    finish_counters(c);
}
