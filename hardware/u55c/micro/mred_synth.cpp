#include <ap_int.h>
#include "par1_montgomery.hpp"

extern "C" void mred_synth(const ap_uint<64> *x, const ap_uint<64> *y,
                            ap_uint<64> *z, ap_uint<64> q,
                            ap_uint<64> q_inv, unsigned count) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=z offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=z bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=q_inv bundle=control
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (unsigned i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        z[i] = mred64(x[i], y[i], q, q_inv);
    }
}
