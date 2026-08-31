#include <ap_int.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/par1_ckks_hls_constants.hpp"

extern "C" void par1_mkf_ckks_kernel(ap_uint<64> *, uint64_t, uint64_t,
                                      uint64_t, uint64_t, uint64_t, uint64_t,
                                      uint64_t, uint64_t, uint64_t, bool,
                                      uint64_t, ap_uint<3> &, ap_uint<3> &);
extern "C" void par1_mkf_ckks_switch_debug(ap_uint<64> *, uint64_t, uint64_t,
                                            uint64_t, uint64_t, uint64_t,
                                            uint64_t, uint64_t, uint64_t);

namespace {

using word = ap_uint<64>;
using addr_t = uint64_t;
using namespace par1_ckks_hls;

addr_t align_words(addr_t value, addr_t alignment = 4096) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<uint64_t> load_u64(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    auto bytes = static_cast<size_t>(input.tellg());
    if (bytes % sizeof(uint64_t)) throw std::runtime_error("unaligned file " + path);
    input.seekg(0);
    std::vector<uint64_t> result(bytes / sizeof(uint64_t));
    input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error("short read " + path);
    return result;
}

void put(std::vector<word> &hbm, addr_t offset, const std::vector<uint64_t> &data) {
    for (size_t i = 0; i < data.size(); ++i) hbm[offset + i] = data[i];
}

bool compare_mod(const std::vector<word> &hbm, addr_t offset,
                 const std::vector<uint64_t> &expected,
                 unsigned modulus_count, const std::string &label) {
    const addr_t poly_words = static_cast<addr_t>(modulus_count) * N;
    for (size_t i = 0; i < expected.size(); ++i) {
        unsigned modulus = static_cast<unsigned>((i % poly_words) / N);
        uint64_t q = modulus_count == QP_COUNT ? QP[modulus].to_uint64() : Q[modulus].to_uint64();
        uint64_t actual_value = hbm[offset + i].to_uint64() % q;
        uint64_t expected_value = expected[i] % q;
        if (actual_value != expected_value) {
            std::cerr << label << " mismatch word=" << i << " modulus=" << modulus
                      << " actual=" << actual_value << " expected=" << expected_value << '\n';
            return false;
        }
    }
    std::cout << label << ": PASS (" << expected.size() << " uint64 values)\n";
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " FIXTURE_DIRECTORY\n";
        return 2;
    }
    const std::string root = argv[1];
    auto input = load_u64(root + "/input_ciphertexts_q.bin");
    auto keys = load_u64(root + "/fold_and_relin_keys_qp.bin");
    auto expected = load_u64(root + "/expected_output_q.bin");
    auto fwd_q = load_u64(root + "/constants/ntt_forward_q.bin");
    auto inv_q = load_u64(root + "/constants/ntt_inverse_q.bin");
    auto fwd_p = load_u64(root + "/constants/ntt_forward_p.bin");
    auto inv_p = load_u64(root + "/constants/ntt_inverse_p.bin");
    auto fold = load_u64(root + "/stage_fold_q.bin");
    auto tensor = load_u64(root + "/stage_tensor_q.bin");
    auto relin = load_u64(root + "/stage_relinearized_q.bin");
    auto rescaled = load_u64(root + "/stage_rescaled_q.bin");
    auto switch_decomp = load_u64(root + "/debug_switch0_decomposition_qp.bin");
    auto switch_external = load_u64(root + "/debug_switch0_external_qp.bin");
    auto switch_moddown = load_u64(root + "/debug_switch0_moddown_q.bin");

    addr_t cursor = 0;
    const addr_t input_base = cursor; cursor = align_words(cursor + input.size());
    const addr_t key_base = cursor; cursor = align_words(cursor + keys.size());
    const addr_t output_base = cursor; cursor = align_words(cursor + expected.size());
    const addr_t fwd_q_base = cursor; cursor = align_words(cursor + fwd_q.size());
    const addr_t inv_q_base = cursor; cursor = align_words(cursor + inv_q.size());
    const addr_t fwd_p_base = cursor; cursor = align_words(cursor + fwd_p.size());
    const addr_t inv_p_base = cursor; cursor = align_words(cursor + inv_p.size());
    const addr_t scratch_base = cursor; cursor = align_words(cursor + 15ULL * (Q_COUNT + P_COUNT) * N);
    const addr_t switch_debug_base = cursor;
    const addr_t switch_decomp_debug = cursor; cursor += switch_decomp.size();
    const addr_t switch_external_debug = cursor; cursor += switch_external.size();
    const addr_t switch_moddown_debug = cursor; cursor += switch_moddown.size();
    cursor = align_words(cursor);
    const addr_t debug_base = cursor;
    const addr_t fold_debug = cursor; cursor += fold.size();
    const addr_t tensor_debug = cursor; cursor += tensor.size();
    const addr_t relin_debug = cursor; cursor += relin.size();
    const addr_t rescaled_debug = cursor; cursor += rescaled.size();
    const addr_t report_base = align_words(cursor); cursor = report_base + 7;

    std::vector<word> hbm(cursor);
    put(hbm, input_base, input);
    put(hbm, key_base, keys);
    put(hbm, fwd_q_base, fwd_q);
    put(hbm, inv_q_base, inv_q);
    put(hbm, fwd_p_base, fwd_p);
    put(hbm, inv_p_base, inv_p);

    std::cout << "HBM image words=" << hbm.size() << " bytes=" << hbm.size() * 8ULL << '\n';
    const addr_t first_left_component = input_base + static_cast<addr_t>(Q_COUNT) * N;
    par1_mkf_ckks_switch_debug(hbm.data(), first_left_component, key_base,
                               scratch_base, fwd_q_base, inv_q_base, fwd_p_base,
                               inv_p_base, switch_debug_base);
    bool pass = true;
    pass &= compare_mod(hbm, switch_decomp_debug, switch_decomp, QP_COUNT, "switch0 decomposition");
    pass &= compare_mod(hbm, switch_external_debug, switch_external, QP_COUNT, "switch0 external");
    pass &= compare_mod(hbm, switch_moddown_debug, switch_moddown, Q_COUNT, "switch0 moddown");

    ap_uint<3> stage_class = 0, timing_class = 0;
    par1_mkf_ckks_kernel(hbm.data(), input_base, key_base, output_base,
                         scratch_base, fwd_q_base, inv_q_base, fwd_p_base,
                         inv_p_base, debug_base, true, report_base, stage_class, timing_class);

    pass &= compare_mod(hbm, fold_debug, fold, Q_COUNT, "fold");
    pass &= compare_mod(hbm, tensor_debug, tensor, Q_COUNT, "tensor");
    pass &= compare_mod(hbm, relin_debug, relin, Q_COUNT, "relinearized");
    pass &= compare_mod(hbm, rescaled_debug, rescaled, Q_COUNT - 1, "rescaled");
    pass &= compare_mod(hbm, output_base, expected, Q_COUNT - 1, "final output");
    std::cout << "counters=";
    for (unsigned i = 0; i < 7; ++i) {
        if (i) std::cout << ',';
        std::cout << hbm[report_base + i].to_uint64();
    }
    std::cout << '\n';
    return pass ? 0 : 1;
}
