#include <ap_int.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../include/par1_ckks_hls_constants.hpp"

extern "C" void par1_kklss_ckks_kernel(ap_uint<64>*,uint64_t,uint64_t,uint64_t,
    uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,ap_uint<3>&,ap_uint<3>&);
namespace { using word=ap_uint<64>; using addr_t=uint64_t; using namespace par1_ckks_hls;
addr_t alignw(addr_t x){return (x+4095)&~addr_t(4095);}
std::vector<uint64_t> load(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("open "+p);size_t n=f.tellg();f.seekg(0);std::vector<uint64_t>v(n/8);f.read((char*)v.data(),n);return v;}
void put(std::vector<word>&h,addr_t a,const std::vector<uint64_t>&v){for(size_t i=0;i<v.size();++i)h[a+i]=v[i];}
bool cmp(const std::vector<word>&h,addr_t a,const std::vector<uint64_t>&e){addr_t pw=(Q_COUNT-1)*N;for(size_t i=0;i<e.size();++i){unsigned m=(i%pw)/N;uint64_t q=Q[m].to_uint64();if(h[a+i].to_uint64()%q!=e[i]%q){std::cerr<<"mismatch word="<<i<<" m="<<m<<" actual="<<h[a+i].to_uint64()%q<<" expected="<<e[i]%q<<'\n';return false;}}return true;}
}
int main(int argc,char**argv){if(argc!=3){std::cerr<<"usage: test FIXTURE MKF_CONSTANT_FIXTURE\n";return 2;}std::string r=argv[1],c=argv[2];
 auto in=load(r+"/input_ciphertexts_q.bin"),key=load(r+"/relinearization_keys_qp.bin"),u=load(r+"/crs_u_qp.bin"),exp=load(r+"/expected_ciphertext_q.bin");
 auto fq=load(c+"/constants/ntt_forward_q.bin"),iq=load(c+"/constants/ntt_inverse_q.bin"),fp=load(c+"/constants/ntt_forward_p.bin"),ip=load(c+"/constants/ntt_inverse_p.bin");
 addr_t z=0,ib=z;z=alignw(z+in.size());addr_t kb=z;z=alignw(z+key.size());addr_t ub=z;z=alignw(z+u.size());addr_t ob=z;z=alignw(z+exp.size());addr_t fqb=z;z=alignw(z+fq.size());addr_t iqb=z;z=alignw(z+iq.size());addr_t fpb=z;z=alignw(z+fp.size());addr_t ipb=z;z=alignw(z+ip.size());addr_t sb=z;z=alignw(z+41ULL*QP_COUNT*N);addr_t rb=z;z+=7;
 std::vector<word>h(z);put(h,ib,in);put(h,kb,key);put(h,ub,u);put(h,fqb,fq);put(h,iqb,iq);put(h,fpb,fp);put(h,ipb,ip);std::cout<<"HBM bytes="<<h.size()*8ULL<<'\n';
 ap_uint<3> stage_class=0,timing_class=0;par1_kklss_ckks_kernel(h.data(),ib,kb,ub,ob,sb,fqb,iqb,fpb,ipb,rb,stage_class,timing_class);bool ok=cmp(h,ob,exp);std::cout<<(ok?"KKLSS-CKKS final: PASS":"KKLSS-CKKS final: FAIL")<<"\ncounters=";for(int i=0;i<7;++i)std::cout<<(i?",":"")<<h[rb+i].to_uint64();std::cout<<'\n';return ok?0:1;}
