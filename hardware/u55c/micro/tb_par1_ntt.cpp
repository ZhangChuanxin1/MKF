#include <ap_int.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "par1_parameters.hpp"

extern "C" void par1_ntt_kernel(ap_uint<64>*, const ap_uint<64>*, ap_uint<64>,
                                 ap_uint<64>, ap_uint<64>, bool, ap_uint<64>*);

static std::vector<uint64_t> load(const char *path, size_t count) {
    std::vector<uint64_t> v(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(uint64_t));
    if (!f || static_cast<size_t>(f.gcount()) != count * sizeof(uint64_t)) {
        std::cerr << "cannot read " << path << "\n"; std::exit(2);
    }
    return v;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::string d=argv[1];
    auto in=load((d+"/ntt_input.bin").c_str(),par1::polynomial_degree);
    auto exp=load((d+"/ntt_expected.bin").c_str(),par1::polynomial_degree);
    auto invexp=load((d+"/intt_expected.bin").c_str(),par1::polynomial_degree);
    auto tw=load((d+"/ntt_twiddle.bin").c_str(),par1::polynomial_degree);
    auto itw=load((d+"/intt_twiddle.bin").c_str(),par1::polynomial_degree);
    auto c=load((d+"/ntt_constants.bin").c_str(),3);
    std::vector<ap_uint<64>> data(in.begin(),in.end()), psi(tw.begin(),tw.end());
    ap_uint<64> reported[7];
    par1_ntt_kernel(data.data(),psi.data(),c[0],c[1],c[2],false,reported);
    size_t errors=0;
    for(size_t i=0;i<data.size();++i) if(uint64_t(data[i])%c[0]!=exp[i]%c[0] && errors++<8)
        std::cerr<<"NTT mismatch "<<i<<" got="<<uint64_t(data[i])%c[0]<<" expected="<<exp[i]%c[0]<<"\n";
    if(errors){std::cerr<<"NTT FAIL errors="<<errors<<"\n";return 1;}
    for(size_t i=0;i<psi.size();++i)psi[i]=itw[i];
    par1_ntt_kernel(data.data(),psi.data(),c[0],c[1],c[2],true,reported);
    for(size_t i=0;i<data.size();++i) if(uint64_t(data[i])%c[0]!=invexp[i]%c[0] && errors++<8)
        std::cerr<<"INTT mismatch "<<i<<" got="<<uint64_t(data[i])%c[0]<<" expected="<<invexp[i]%c[0]<<"\n";
    if(errors){std::cerr<<"INTT FAIL errors="<<errors<<"\n";return 1;}
    std::cout<<"NTT/INTT PASS N="<<data.size()<<"\n";
    return 0;
}
