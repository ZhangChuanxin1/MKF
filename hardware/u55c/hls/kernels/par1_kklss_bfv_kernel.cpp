#include "par1_mkf_bfv_kernel.cpp"

namespace kklss_bfv_u55c {
using namespace mkf_bfv_u55c;
static constexpr unsigned K_BETA=6, K_COMPONENTS=17;
static constexpr addr_t K_SLOT_WORDS=R_COUNT*N;
enum KS {D10=0,D15=5,D20=6,D25=11,ACC=12,MODUP=13,
 X10=14,X15=19,X20=20,X25=25,Y10=26,Y15=31,Y20=32,Y25=37,
 L0=38,R0=39,A=40,B=41,PROD=42,OUT0=43,OUT16=59};
static inline addr_t s(addr_t b,unsigned x){return b+static_cast<addr_t>(x)*K_SLOT_WORDS;}
static inline addr_t in(addr_t b,unsigned comp){return b+static_cast<addr_t>(comp)*Q_COUNT*N;}
static inline addr_t key(addr_t b,unsigned party,unsigned branch,unsigned comp,unsigned d,unsigned m){
 addr_t x=(((static_cast<addr_t>(party)*2+branch)*3+comp)*K_BETA+d);
 return b+(x*QP_COUNT+m)*N;
}
static inline addr_t crs(addr_t b,unsigned d,unsigned m){return b+(static_cast<addr_t>(d)*QP_COUNT+m)*N;}
static inline word r2qp(unsigned m){if(m<Q_COUNT)return R2_R[m];return m==Q_COUNT?word(17592051826944ULL):word(16906086627934464ULL);}
static inline word k_q_to_qm(unsigned target,const word values[Q_COUNT]){
 return q_to_qm(target,values);
}
static inline word k_qm_to_q(unsigned target,const word values[QM_COUNT]){
 return qm_to_q(target,values);
}
static inline word k_q_to_qm_lazy(unsigned target,const word values[Q_COUNT]){
 word y[Q_COUNT];ap_uint<128> approx=0,acc=0;for(unsigned i=0;i<Q_COUNT;++i){y[i]=mont_mul(values[i],Q2M_INV[i],Q[i],MRED_Q[i]);approx+=static_cast<ap_uint<128>>(y[i])*RECIP96_Q[i];acc+=static_cast<ap_uint<128>>(y[i])*Q2M_QSTAR[target][i];}
 unsigned v=(unsigned)(approx>>96);word corr=0;for(unsigned i=0;i<Q_COUNT;++i)if(i<v)corr=add_mod(corr,Q2M_MINUS_Q[target],QM[target]);word lo=static_cast<word>(acc),hi=static_cast<word>(acc>>64);ap_uint<128> mp=static_cast<ap_uint<128>>(word(lo*MRED_QM[target]))*QM[target];word hhi=static_cast<word>(mp>>64);ap_uint<65> z=static_cast<ap_uint<65>>(hi)+QM[target]+corr-hhi;return static_cast<word>(z);
}
static inline word k_qm_to_q_lazy(unsigned target,const word values[QM_COUNT]){
 word y[QM_COUNT];ap_uint<128> approx=0,acc=0;for(unsigned i=0;i<QM_COUNT;++i){y[i]=mont_mul(values[i],M2Q_INV[i],QM[i],MRED_QM[i]);approx+=static_cast<ap_uint<128>>(y[i])*RECIP96_QM[i];acc+=static_cast<ap_uint<128>>(y[i])*M2Q_QSTAR[target][i];}
 unsigned v=(unsigned)(approx>>96);word corr=0;for(unsigned i=0;i<QM_COUNT;++i)if(i<v)corr=add_mod(corr,M2Q_MINUS_M[target],Q[target]);word lo=static_cast<word>(acc),hi=static_cast<word>(acc>>64);ap_uint<128> mp=static_cast<ap_uint<128>>(word(lo*MRED_Q[target]))*Q[target];word hhi=static_cast<word>(mp>>64);ap_uint<65> z=static_cast<ap_uint<65>>(hi)+Q[target]+corr-hhi;return static_cast<word>(z);
}

static void lift_l_coeff(word*h,addr_t src,addr_t dst,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 emit_phase_event(e,CycleClass::ModUpDown,TimingClass::ModUpDown);
 set_timing_class(c,TimingClass::ModUpDown);
 for(unsigned i=0;i<N;++i){
  word v[Q_COUNT];
#pragma HLS ARRAY_PARTITION variable=v complete
  for(unsigned m=0;m<Q_COUNT;++m){
#pragma HLS UNROLL
   v[m]=canonical(h[src+static_cast<addr_t>(m)*N+i],Q[m]); h[limb(dst,m)+i]=v[m];
  }
  for(unsigned m=0;m<QM_COUNT;++m)h[limb(dst,Q_COUNT+m)+i]=k_q_to_qm_lazy(m,v);
  count_cycle(c,CycleClass::ModUpDown,true);
 }
}
static void lift_r_coeff(word*h,addr_t src,addr_t dst,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 emit_phase_event(e,CycleClass::ModUpDown,TimingClass::ModUpDown);
 set_timing_class(c,TimingClass::ModUpDown);
 for(unsigned i=0;i<N;++i){
  word qv[Q_COUNT];
#pragma HLS ARRAY_PARTITION variable=qv complete
  for(unsigned m=0;m<Q_COUNT;++m){
#pragma HLS UNROLL
   qv[m]=mont_mul(h[src+static_cast<addr_t>(m)*N+i],QM_MFORM_Q[m],Q[m],MRED_Q[m]);
  }
  word mv[QM_COUNT];
#pragma HLS ARRAY_PARTITION variable=mv complete
  for(unsigned m=0;m<QM_COUNT;++m){
#pragma HLS UNROLL
   word x=k_q_to_qm(m,qv); mv[m]=mont_mul(x,QM[m]-Q_INV_QM[m],QM[m],MRED_QM[m]);
   h[limb(dst,Q_COUNT+m)+i]=mv[m];
  }
  for(unsigned m=0;m<Q_COUNT;++m)h[limb(dst,m)+i]=k_qm_to_q_lazy(m,mv);
  count_cycle(c,CycleClass::ModUpDown,true);
 }
}
static void ntt_r(word*h,addr_t p,bool inverse,addr_t tq,addr_t tm,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 for(unsigned m=0;m<R_COUNT;++m){
  if(m<Q_COUNT)ntt_limb(h,limb(p,m),tq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],inverse,c,e,CycleClass::Other,TimingClass::Tensor);
  else{unsigned k=m-Q_COUNT;ntt_limb(h,limb(p,m),tm+static_cast<addr_t>(k)*N,QM[k],MRED_QM[k],N_INV_QM[k],inverse,c,e,CycleClass::Other,TimingClass::Tensor);}
 }
}
static void decomp_r(word*h,addr_t src,addr_t scratch,addr_t fq,addr_t fp,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 emit_phase_event(e,CycleClass::Decomposition,TimingClass::Decomposition);
 set_timing_class(c,TimingClass::Decomposition);
 for(unsigned branch=0;branch<2;++branch)for(unsigned d=0;d<K_BETA;++d){
  addr_t out=s(scratch,(branch?D20:D10)+d); unsigned source=branch*K_BETA+d;
  for(unsigned m=0;m<QP_COUNT;++m){
   for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
    h[limb(out,m)+i]=canonical(h[limb(src,source)+i],QP[m]);count_cycle(c,CycleClass::Decomposition,true);
   }
   if(m<Q_COUNT)ntt_limb(h,limb(out,m),fq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,e,CycleClass::Decomposition,TimingClass::Decomposition);
   else{unsigned p=m-Q_COUNT;ntt_limb(h,limb(out,m),fp+static_cast<addr_t>(p)*N,P[p],MRED_P[p],N_INV_P[p],false,c,e,CycleClass::Decomposition,TimingClass::Decomposition);}
  }
 }
}
static void decomp_q(word*h,addr_t src,addr_t scratch,addr_t fq,addr_t fp,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 emit_phase_event(e,CycleClass::Decomposition,TimingClass::Decomposition);
 set_timing_class(c,TimingClass::Decomposition);
 for(unsigned d=0;d<K_BETA;++d){addr_t out=s(scratch,D10+d);for(unsigned m=0;m<QP_COUNT;++m){
  for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
   h[limb(out,m)+i]=canonical(h[limb(src,d)+i],QP[m]);count_cycle(c,CycleClass::Decomposition,true);
  }
  if(m<Q_COUNT)ntt_limb(h,limb(out,m),fq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],false,c,e,CycleClass::Decomposition,TimingClass::Decomposition);
  else{unsigned p=m-Q_COUNT;ntt_limb(h,limb(out,m),fp+static_cast<addr_t>(p)*N,P[p],MRED_P[p],N_INV_P[p],false,c,e,CycleClass::Decomposition,TimingClass::Decomposition);}
 }}
}
static void zero_vec(word*h,addr_t b){for(unsigned v=X10;v<=Y25;++v)for(unsigned m=0;m<QP_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
 h[limb(s(b,v),m)+i]=0;}}
static void acc_vec(word*h,addr_t kb,unsigned party,unsigned comp,unsigned va,unsigned vb,addr_t scratch,CycleCounters&c,hls::stream<ap_uint<6>>&e){
 emit_phase_event(e,CycleClass::ExternalProduct,TimingClass::ExternalProduct);
 set_timing_class(c,TimingClass::ExternalProduct);
 for(unsigned branch=0;branch<2;++branch)for(unsigned d=0;d<K_BETA;++d)for(unsigned m=0;m<QP_COUNT;++m){word q=QP[m],qi=MRED_QP[m];for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  addr_t dst=limb(s(scratch,(branch?vb:va)+d),m)+i;word a=h[limb(s(scratch,(branch?D20:D10)+d),m)+i];
  h[dst]=add_mod(h[dst],mont_mul(a,h[key(kb,party,branch,comp,d,m)+i],q,qi),q);count_cycle(c,CycleClass::ExternalProduct,true);
 }}}
static void mform_vec(word*h,addr_t b,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::ExternalProduct,TimingClass::ExternalProduct);set_timing_class(c,TimingClass::ExternalProduct);for(unsigned v=X10;v<=Y25;++v)for(unsigned m=0;m<QP_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
 addr_t a=limb(s(b,v),m)+i;h[a]=mont_mul(h[a],r2qp(m),QP[m],MRED_QP[m]);count_cycle(c,CycleClass::ExternalProduct,true);
}}
static void finish_external(word*h,addr_t scratch,addr_t iq,addr_t ip,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::ModUpDown,TimingClass::ModUpDown);set_timing_class(c,TimingClass::ModUpDown);addr_t a=s(scratch,ACC),t=s(scratch,MODUP);
 for(unsigned m=0;m<QP_COUNT;++m){if(m<Q_COUNT)ntt_limb(h,limb(a,m),iq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);else{unsigned p=m-Q_COUNT;ntt_limb(h,limb(a,m),ip+static_cast<addr_t>(p)*N,P[p],MRED_P[p],N_INV_P[p],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);}}
 for(unsigned m=0;m<Q_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  word y=p_to_q(m,h[limb(a,Q_COUNT)+i],h[limb(a,Q_COUNT+1)+i]);h[limb(t,m)+i]=mont_mul(sub_mod(h[limb(a,m)+i],y,Q[m]),P_INV_Q[m],Q[m],MRED_Q[m]);count_cycle(c,CycleClass::ModUpDown,true);
 }
}
static void ext_vec(word*h,unsigned va,unsigned vb,addr_t scratch,addr_t iq,addr_t ip,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::ExternalProduct,TimingClass::ExternalProduct);set_timing_class(c,TimingClass::ExternalProduct);addr_t a=s(scratch,ACC);
 for(unsigned m=0;m<QP_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  word sum=0;for(unsigned d=0;d<K_BETA;++d){
#pragma HLS UNROLL
   sum=add_mod(sum,mont_mul(h[limb(s(scratch,D10+d),m)+i],h[limb(s(scratch,va+d),m)+i],QP[m],MRED_QP[m]),QP[m]);
   sum=add_mod(sum,mont_mul(h[limb(s(scratch,D20+d),m)+i],h[limb(s(scratch,vb+d),m)+i],QP[m],MRED_QP[m]),QP[m]);
  }h[limb(a,m)+i]=sum;count_cycle(c,CycleClass::ExternalProduct,true);
 }finish_external(h,scratch,iq,ip,c,e);
}
static void ext_single(word*h,addr_t base,unsigned party,unsigned comp,bool is_crs,addr_t scratch,addr_t iq,addr_t ip,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::ExternalProduct,TimingClass::ExternalProduct);set_timing_class(c,TimingClass::ExternalProduct);addr_t a=s(scratch,ACC);
 for(unsigned m=0;m<QP_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  word sum=0;for(unsigned d=0;d<K_BETA;++d){
#pragma HLS UNROLL
   addr_t k=is_crs?crs(base,d,m):key(base,party,0,comp,d,m);sum=add_mod(sum,mont_mul(h[limb(s(scratch,D10+d),m)+i],h[k+i],QP[m],MRED_QP[m]),QP[m]);
  }h[limb(a,m)+i]=sum;count_cycle(c,CycleClass::ExternalProduct,true);
 }finish_external(h,scratch,iq,ip,c,e);
}
static void addacc(word*h,addr_t dst,addr_t b,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::ExternalProduct,TimingClass::ExternalProduct);set_timing_class(c,TimingClass::ExternalProduct);for(unsigned m=0;m<Q_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
 addr_t d=limb(dst,m)+i;h[d]=add_mod(h[d],h[limb(s(b,MODUP),m)+i],Q[m]);count_cycle(c,CycleClass::ExternalProduct,true);
}}
// Keep the diagnostic copy address as an explicit value expression.  The
// reference-increment form did not produce observable stage data in the
// diagnostic path even though the same OUT slots later reached `output`.
// This path is diagnostic-only and is intentionally independent of the
// performance counters and arithmetic schedule.
static void stage_copy(word*h,addr_t base,addr_t scratch){
 for(unsigned comp=0;comp<K_COMPONENTS;++comp)for(unsigned m=0;m<Q_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  addr_t off=(static_cast<addr_t>(comp)*Q_COUNT+m)*N+i;
  h[base+off]=canonical(h[limb(s(scratch,OUT0+comp),m)+i],Q[m]);
 }
}
static void product_quantize(word*h,addr_t x,addr_t y,addr_t out,addr_t scratch,addr_t iq,addr_t im,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::Other,TimingClass::Tensor);set_timing_class(c,TimingClass::Tensor);addr_t p=s(scratch,PROD);
 for(unsigned m=0;m<R_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  word xm=mont_mul(h[limb(x,m)+i],R2_R[m],R[m],MRED_R[m]);h[limb(p,m)+i]=mont_mul(xm,h[limb(y,m)+i],R[m],MRED_R[m]);count_cycle(c,CycleClass::Tensor,true);
 }
 emit_phase_event(e,CycleClass::ModUpDown,TimingClass::ModUpDown);set_timing_class(c,TimingClass::ModUpDown);for(unsigned m=0;m<R_COUNT;++m){for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  h[limb(p,m)+i]=mont_mul(h[limb(p,m)+i],T_MFORM_R[m],R[m],MRED_R[m]);
 }if(m<Q_COUNT)ntt_limb(h,limb(p,m),iq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);else{unsigned k=m-Q_COUNT;ntt_limb(h,limb(p,m),im+static_cast<addr_t>(k)*N,QM[k],MRED_QM[k],N_INV_QM[k],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);}}
 for(unsigned i=0;i<N;++i){word mv[QM_COUNT];for(unsigned k=0;k<QM_COUNT;++k)mv[k]=h[limb(p,Q_COUNT+k)+i];for(unsigned m=0;m<Q_COUNT;++m){word lifted=k_qm_to_q(m,mv);h[limb(out,m)+i]=mont_mul(sub_mod(lifted,h[limb(p,m)+i],Q[m]),Q[m]-QM_INV_Q[m],Q[m],MRED_Q[m]);}count_cycle(c,CycleClass::ModUpDown,true);}
}
static void cross_quantize(word*h,addr_t l0,addr_t r0,addr_t a,addr_t b,addr_t out,addr_t scratch,addr_t iq,addr_t im,CycleCounters&c,hls::stream<ap_uint<6>>&e){emit_phase_event(e,CycleClass::Other,TimingClass::Tensor);set_timing_class(c,TimingClass::Tensor);addr_t p=s(scratch,PROD);
 for(unsigned m=0;m<R_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  word lm=mont_mul(h[limb(l0,m)+i],R2_R[m],R[m],MRED_R[m]);word rm=mont_mul(h[limb(r0,m)+i],R2_R[m],R[m],MRED_R[m]);
 word z=mont_mul(lm,h[limb(b,m)+i],R[m],MRED_R[m]);z=add_mod(z,mont_mul(rm,h[limb(a,m)+i],R[m],MRED_R[m]),R[m]);h[limb(p,m)+i]=z;count_cycle(c,CycleClass::Tensor,true);
 }
 emit_phase_event(e,CycleClass::ModUpDown,TimingClass::ModUpDown);set_timing_class(c,TimingClass::ModUpDown);for(unsigned m=0;m<R_COUNT;++m){for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  h[limb(p,m)+i]=mont_mul(h[limb(p,m)+i],T_MFORM_R[m],R[m],MRED_R[m]);
 }if(m<Q_COUNT)ntt_limb(h,limb(p,m),iq+static_cast<addr_t>(m)*N,Q[m],MRED_Q[m],N_INV_Q[m],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);else{unsigned k=m-Q_COUNT;ntt_limb(h,limb(p,m),im+static_cast<addr_t>(k)*N,QM[k],MRED_QM[k],N_INV_QM[k],true,c,e,CycleClass::ModUpDown,TimingClass::ModUpDown);}}
 for(unsigned i=0;i<N;++i){word mv[QM_COUNT];for(unsigned k=0;k<QM_COUNT;++k)mv[k]=h[limb(p,Q_COUNT+k)+i];for(unsigned m=0;m<Q_COUNT;++m){word lifted=k_qm_to_q(m,mv);h[limb(out,m)+i]=mont_mul(sub_mod(lifted,h[limb(p,m)+i],Q[m]),Q[m]-QM_INV_Q[m],Q[m],MRED_Q[m]);}count_cycle(c,CycleClass::ModUpDown,true);}
}
}

extern "C" void par1_kklss_bfv_lift_debug(ap_uint<64>*h,unsigned long long input,unsigned long long left,unsigned long long right,hls::stream<ap_uint<6>>&e){
 using namespace kklss_bfv_u55c;CycleCounters c;reset_counters(c);lift_l_coeff(h,in(input,0),left,c,e);lift_r_coeff(h,in(input,0),right,c,e);
}

extern "C" void par1_kklss_bfv_external_debug(ap_uint<64>*h,unsigned long long input,unsigned long long keys,unsigned long long scratch,unsigned long long fq,unsigned long long iq,unsigned long long fp,unsigned long long ip,unsigned long long debug,hls::stream<ap_uint<6>>&e){
 using namespace kklss_bfv_u55c;CycleCounters c;reset_counters(c);zero_vec(h,scratch);
 for(unsigned p=0;p<PARTIES;++p){lift_l_coeff(h,in(input,p+1),s(scratch,A),c,e);decomp_r(h,s(scratch,A),scratch,fq,fp,c,e);acc_vec(h,keys,p,1,X10,X20,scratch,c,e);}mform_vec(h,scratch,c,e);
 lift_r_coeff(h,in(input,1),s(scratch,B),c,e);decomp_r(h,s(scratch,B),scratch,fq,fp,c,e);
 addr_t cur=debug;for(unsigned set=0;set<4;++set){unsigned base=set==0?X10:set==1?X20:set==2?D10:D20;for(unsigned d=0;d<K_BETA;++d)for(unsigned m=0;m<QP_COUNT;++m)for(unsigned i=0;i<N;++i)h[cur++]=canonical(h[limb(s(scratch,base+d),m)+i],QP[m]);}
 ext_vec(h,X10,X20,scratch,iq,ip,c,e);for(unsigned m=0;m<Q_COUNT;++m)for(unsigned i=0;i<N;++i)h[cur++]=canonical(h[limb(s(scratch,MODUP),m)+i],Q[m]);
}

extern "C" void par1_kklss_bfv_kernel(ap_uint<64>*h,unsigned long long input,unsigned long long keys,unsigned long long u,unsigned long long output,unsigned long long scratch,unsigned long long fq,unsigned long long iq,unsigned long long fp,unsigned long long ip,unsigned long long fm,unsigned long long im,unsigned long long debug_base,unsigned int enable_debug,unsigned long long report,hls::stream<ap_uint<6>>&phase_events){
#pragma HLS INTERFACE m_axi port=h offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=h bundle=control
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=keys bundle=control
#pragma HLS INTERFACE s_axilite port=u bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=scratch bundle=control
#pragma HLS INTERFACE s_axilite port=fq bundle=control
#pragma HLS INTERFACE s_axilite port=iq bundle=control
#pragma HLS INTERFACE s_axilite port=fp bundle=control
#pragma HLS INTERFACE s_axilite port=ip bundle=control
#pragma HLS INTERFACE s_axilite port=fm bundle=control
#pragma HLS INTERFACE s_axilite port=im bundle=control
#pragma HLS INTERFACE s_axilite port=debug_base bundle=control
#pragma HLS INTERFACE s_axilite port=enable_debug bundle=control
#pragma HLS INTERFACE s_axilite port=report bundle=control
#pragma HLS INTERFACE axis port=phase_events
#pragma HLS INTERFACE s_axilite port=return bundle=control
 using namespace kklss_bfv_u55c;CycleCounters c;reset_counters(c);emit_phase_event(phase_events,CycleClass::Other,TimingClass::Other);zero_vec(h,scratch);
 for(unsigned p=0;p<PARTIES;++p){lift_l_coeff(h,in(input,p+1),s(scratch,A),c,phase_events);decomp_r(h,s(scratch,A),scratch,fq,fp,c,phase_events);acc_vec(h,keys,p,1,X10,X20,scratch,c,phase_events);lift_r_coeff(h,in(input,p+1),s(scratch,B),c,phase_events);decomp_r(h,s(scratch,B),scratch,fq,fp,c,phase_events);acc_vec(h,keys,p,0,Y10,Y20,scratch,c,phase_events);}mform_vec(h,scratch,c,phase_events);
 lift_l_coeff(h,in(input,0),s(scratch,L0),c,phase_events);lift_r_coeff(h,in(input,0),s(scratch,R0),c,phase_events);emit_phase_event(phase_events,CycleClass::Other,TimingClass::Tensor);set_timing_class(c,TimingClass::Tensor);ntt_r(h,s(scratch,L0),false,fq,fm,c,phase_events);ntt_r(h,s(scratch,R0),false,fq,fm,c,phase_events);product_quantize(h,s(scratch,L0),s(scratch,R0),s(scratch,OUT0),scratch,iq,im,c,phase_events);
 for(unsigned p=0;p<PARTIES;++p){lift_l_coeff(h,in(input,p+1),s(scratch,A),c,phase_events);lift_r_coeff(h,in(input,p+1),s(scratch,B),c,phase_events);emit_phase_event(phase_events,CycleClass::Other,TimingClass::Tensor);set_timing_class(c,TimingClass::Tensor);ntt_r(h,s(scratch,A),false,fq,fm,c,phase_events);ntt_r(h,s(scratch,B),false,fq,fm,c,phase_events);cross_quantize(h,s(scratch,L0),s(scratch,R0),s(scratch,A),s(scratch,B),s(scratch,OUT0+p+1),scratch,iq,im,c,phase_events);}
 const addr_t debug_stage_words=static_cast<addr_t>(K_COMPONENTS)*Q_COUNT*N;
 if(enable_debug)stage_copy(h,debug_base,scratch);
 for(unsigned p=0;p<PARTIES;++p){lift_r_coeff(h,in(input,p+1),s(scratch,B),c,phase_events);decomp_r(h,s(scratch,B),scratch,fq,fp,c,phase_events);ext_vec(h,X10,X20,scratch,iq,ip,c,phase_events);addacc(h,s(scratch,OUT0+p+1),scratch,c,phase_events);}
 if(enable_debug)stage_copy(h,debug_base+debug_stage_words,scratch);
 for(unsigned p=0;p<PARTIES;++p){lift_l_coeff(h,in(input,p+1),s(scratch,A),c,phase_events);decomp_r(h,s(scratch,A),scratch,fq,fp,c,phase_events);ext_vec(h,Y10,Y20,scratch,iq,ip,c,phase_events);decomp_q(h,s(scratch,MODUP),scratch,fq,fp,c,phase_events);ext_single(h,keys,p,2,false,scratch,iq,ip,c,phase_events);addacc(h,s(scratch,OUT0),scratch,c,phase_events);ext_single(h,u,0,0,true,scratch,iq,ip,c,phase_events);addacc(h,s(scratch,OUT0+p+1),scratch,c,phase_events);}
 if(enable_debug)stage_copy(h,debug_base+2*debug_stage_words,scratch);
 emit_phase_event(phase_events,CycleClass::Other,TimingClass::Other);set_timing_class(c,TimingClass::Other);for(unsigned comp=0;comp<K_COMPONENTS;++comp)for(unsigned m=0;m<Q_COUNT;++m)for(unsigned i=0;i<N;++i){
#pragma HLS PIPELINE II=1
  h[output+(static_cast<addr_t>(comp)*Q_COUNT+m)*N+i]=canonical(h[limb(s(scratch,OUT0+comp),m)+i],Q[m]);count_cycle(c,CycleClass::Other,true);
 }
 h[report]=c.decomposition;h[report+1]=c.external_product;h[report+2]=c.ntt_intt;h[report+3]=c.modup_down;h[report+4]=c.hbm_stall;h[report+5]=c.other;h[report+6]=c.total;
 emit_phase_event(phase_events,CycleClass::Idle,TimingClass::Idle);finish_counters(c);
}
