#include <ap_int.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../include/par1_bfv_hls_constants.hpp"

extern "C" void par1_mkf_bfv_kernel(ap_uint<64> *, uint64_t, uint64_t, uint64_t,
                                     uint64_t, uint64_t, uint64_t, uint64_t,
                                     uint64_t, uint64_t, uint64_t, uint64_t,
                                     bool, uint64_t, ap_uint<3> &, ap_uint<3> &);

namespace {
using word = ap_uint<64>; using addr_t = uint64_t;
using namespace par1_bfv_hls;
addr_t align_words(addr_t x) { return (x + 4095) & ~addr_t(4095); }
std::vector<uint64_t> load(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open " + path);
    size_t bytes = static_cast<size_t>(f.tellg()); f.seekg(0);
    std::vector<uint64_t> v(bytes / 8); f.read(reinterpret_cast<char *>(v.data()), bytes);
    if (!f) throw std::runtime_error("read " + path); return v;
}
void put(std::vector<word> &h, addr_t at, const std::vector<uint64_t> &v) {
    for (size_t i = 0; i < v.size(); ++i) h[at + i] = v[i];
}
bool compare(const std::vector<word> &h, addr_t at, const std::vector<uint64_t> &v,
             unsigned moduli, bool r_basis, const std::string &name) {
    addr_t poly = static_cast<addr_t>(moduli) * N;
    for (size_t i = 0; i < v.size(); ++i) {
        unsigned m = static_cast<unsigned>((i % poly) / N);
        uint64_t q = (r_basis ? R[m] : Q[m]).to_uint64();
        if (h[at + i].to_uint64() % q != v[i] % q) {
            std::cerr << name << " mismatch word=" << i << " modulus=" << m
                      << " actual=" << h[at + i].to_uint64() % q
                      << " expected=" << v[i] % q << '\n'; return false;
        }
    }
    std::cout << name << ": PASS (" << v.size() << " uint64 values)\n"; return true;
}
}

int main(int argc, char **argv) {
    if (argc != 2) return 2; std::string root = argv[1];
    auto input=load(root+"/input_ciphertexts_q.bin"), keys=load(root+"/fold_and_relin_keys_qp.bin");
    auto output=load(root+"/expected_output_q.bin"), fq=load(root+"/constants/ntt_forward_q.bin");
    auto iq=load(root+"/constants/ntt_inverse_q.bin"), fp=load(root+"/constants/ntt_forward_p.bin");
    auto ip=load(root+"/constants/ntt_inverse_p.bin"), fm=load(root+"/constants/ntt_forward_q_mul.bin");
    auto im=load(root+"/constants/ntt_inverse_q_mul.bin"), fold=load(root+"/stage_fold_q.bin");
    auto lifted=load(root+"/stage_lifted_r.bin"), tensor=load(root+"/stage_tensor_r.bin");
    auto quant=load(root+"/stage_quantized_q.bin"), relin=load(root+"/stage_relinearized_q.bin");
    addr_t c=0, inputBase=c; c=align_words(c+input.size());
    addr_t keyBase=c; c=align_words(c+keys.size()); addr_t outputBase=c; c=align_words(c+output.size());
    addr_t fqBase=c; c=align_words(c+fq.size()); addr_t iqBase=c; c=align_words(c+iq.size());
    addr_t fpBase=c; c=align_words(c+fp.size()); addr_t ipBase=c; c=align_words(c+ip.size());
    addr_t fmBase=c; c=align_words(c+fm.size()); addr_t imBase=c; c=align_words(c+im.size());
    addr_t scratch=c; c=align_words(c+21ULL*R_COUNT*N); addr_t debug=c;
    addr_t foldAt=c; c+=fold.size(); addr_t liftedAt=c; c+=lifted.size(); addr_t tensorAt=c; c+=tensor.size();
    addr_t quantAt=c; c+=quant.size(); addr_t relinAt=c; c+=relin.size(); addr_t report=align_words(c); c=report+7;
    std::vector<word> hbm(c); put(hbm,inputBase,input); put(hbm,keyBase,keys); put(hbm,fqBase,fq);
    put(hbm,iqBase,iq); put(hbm,fpBase,fp); put(hbm,ipBase,ip); put(hbm,fmBase,fm); put(hbm,imBase,im);
    std::cout << "HBM bytes=" << hbm.size()*8ULL << '\n';
    ap_uint<3> stage_class=0,timing_class=0; par1_mkf_bfv_kernel(hbm.data(),inputBase,keyBase,outputBase,scratch,fqBase,iqBase,fpBase,ipBase,fmBase,imBase,debug,true,report,stage_class,timing_class);
    bool pass=true; pass&=compare(hbm,foldAt,fold,Q_COUNT,false,"fold");
    pass&=compare(hbm,liftedAt,lifted,R_COUNT,true,"lifted"); pass&=compare(hbm,tensorAt,tensor,R_COUNT,true,"tensor");
    pass&=compare(hbm,quantAt,quant,Q_COUNT,false,"quantized"); pass&=compare(hbm,relinAt,relin,Q_COUNT,false,"relinearized");
    pass&=compare(hbm,outputBase,output,Q_COUNT,false,"final output");
    std::cout<<"counters="; for(unsigned i=0;i<7;++i){if(i)std::cout<<',';std::cout<<hbm[report+i].to_uint64();}std::cout<<'\n';
    return pass?0:1;
}
