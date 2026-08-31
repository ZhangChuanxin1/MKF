#pragma once

namespace par1 {

constexpr unsigned log_n = 14;
constexpr unsigned polynomial_degree = 1u << log_n;
constexpr unsigned parties = 16;
constexpr unsigned gadget_dimension = 3;
constexpr unsigned lanes = 16;
constexpr unsigned butterflies_per_cycle = lanes / 2;

namespace ckks {
constexpr unsigned log_slots = 13;
constexpr unsigned log_scale = 52;
constexpr unsigned long long q[] = {
    0x7ffffffffe70001ULL,
    0xffffffff00001ULL,
    0xfffffffe40001ULL,
    0xfffffffe20001ULL,
    0xfffffffbe0001ULL,
    0xfffffffa60001ULL,
};
constexpr unsigned long long p[] = {
    0xffffffffffc0001ULL,
    0xfffffffff840001ULL,
};
} // namespace ckks

namespace bfv {
constexpr unsigned long long plaintext_modulus = 65537;
constexpr unsigned long long q[] = {
    0x1fffffffe30001ULL,
    0x1fffffffd10001ULL,
    0x1fffffffbf0001ULL,
    0x1fffffffb60001ULL,
    0x1fffffff920001ULL,
    0x3fffffffd60001ULL,
};
constexpr unsigned long long q_mul[] = {
    0x1fffffffd80001ULL,
    0x1fffffffc50001ULL,
    0x1fffffffb90001ULL,
    0x1fffffffa50001ULL,
    0x1fffffff900001ULL,
    0x3fffffffca0001ULL,
};
constexpr unsigned long long p[] = {
    0xffffffffffc0001ULL,
    0xfffffffff840001ULL,
};
} // namespace bfv

} // namespace par1
