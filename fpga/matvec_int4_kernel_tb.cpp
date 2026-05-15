// matvec_int4_kernel_tb.cpp — host-compilable unit test for the HLS kernel.
//
// Uses fpga/hls_stubs.hpp to emulate Xilinx types on a standard C++ compiler.
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas \
//       -I. fpga/matvec_int4_kernel_tb.cpp -o /tmp/matvec_int4_kernel_tb
// Run:
//   /tmp/matvec_int4_kernel_tb
//
// The test exercises both the dense and sparse paths of matvec_int4_tile
// against a plain-C++ bit-exact reference.  Any mismatch fails the process
// with a non-zero exit code.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/matvec_int4.hpp"

// ---------------------------------------------------------------------------
// Helpers: sign-extend nibble, pack/unpack
// ---------------------------------------------------------------------------
static inline int8_t sext4(uint8_t n) {
    int8_t v = static_cast<int8_t>(n & 0x0F);
    if (v & 0x08) v -= 16;
    return v;
}

static inline uint8_t pack_nibbles(int8_t lo, int8_t hi) {
    uint8_t l = static_cast<uint8_t>(lo & 0x0F);
    uint8_t h = static_cast<uint8_t>(hi & 0x0F);
    return l | (h << 4);
}

// ---------------------------------------------------------------------------
// Build a packed weight matrix (row-major, 2 nibbles/byte).
// ---------------------------------------------------------------------------
std::vector<uint8_t> pack_weights(const std::vector<int8_t>& w_q,
                                   int out_f, int in_f) {
    std::vector<uint8_t> packed;
    packed.reserve(out_f * ((in_f + 1) / 2));
    for (int r = 0; r < out_f; ++r) {
        for (int c = 0; c < in_f; c += 2) {
            int8_t lo = w_q[r * in_f + c];
            int8_t hi = (c + 1 < in_f) ? w_q[r * in_f + c + 1] : 0;
            packed.push_back(pack_nibbles(lo, hi));
        }
    }
    return packed;
}

// ---------------------------------------------------------------------------
// Reference matvec (dequantised int4 * int8 -> int32, no scale).
// ---------------------------------------------------------------------------
std::vector<int32_t> reference_matvec(const std::vector<int8_t>& w_q,
                                      const std::vector<int8_t>& x,
                                      int out_f, int in_f) {
    std::vector<int32_t> y(out_f, 0);
    for (int r = 0; r < out_f; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < in_f; ++c) {
            acc += static_cast<int32_t>(w_q[r * in_f + c]) *
                   static_cast<int32_t>(x[c]);
        }
        y[r] = acc;
    }
    return y;
}

// ---------------------------------------------------------------------------
// Test 1: dense path, small shape
// ---------------------------------------------------------------------------
template<int IN_F, int OUT_F>
bool test_dense_small() {
    static_assert(IN_F % LANES == 0, "IN_F must be multiple of LANES");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> wdist(-7, 7);   // INT4 signed range
    std::uniform_int_distribution<int> adist(-127, 127); // INT8 signed range

    std::vector<int8_t> w_q(OUT_F * IN_F);
    std::vector<int8_t> x(IN_F);
    for (auto& v : w_q) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : x)   v = static_cast<int8_t>(adist(rng));

    auto packed = pack_weights(w_q, OUT_F, IN_F);
    auto y_ref = reference_matvec(w_q, x, OUT_F, IN_F);

    // Drive streams
    hls::stream<packed_w_beat_t>   s_w;
    hls::stream<packed_a_beat_t>   s_a;
    hls::stream<ap_int<32>>        s_y;

    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int W_BEAT_BYTES = LANES / 2;
    const int row_bytes = (IN_F + 1) / 2;

    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            packed_a_beat_t ab;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));

                // Two activation bytes per weight byte (one per nibble)
                int col_base = b * LANES + lane * 2;
                uint8_t a_lo = static_cast<uint8_t>(x[col_base + 0]);
                uint8_t a_hi = static_cast<uint8_t>(x[col_base + 1]);
                ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a_lo));
                ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a_hi));
            }
            s_w.write(wb);
            s_a.write(ab);
        }
    }

    float dummy_scales[TILE_ROWS] = {0};
    matvec_int4_tile_dense<IN_F, OUT_F>(s_w, s_a, dummy_scales, s_y);

    std::vector<int32_t> y_hw;
    for (int r = 0; r < OUT_F; ++r) {
        ap_int<32> v = s_y.read();
        y_hw.push_back(static_cast<int32_t>(v));
    }

    bool ok = true;
    int64_t max_err = 0;
    for (int i = 0; i < OUT_F; ++i) {
        int64_t err = std::llabs(static_cast<int64_t>(y_hw[i]) - static_cast<int64_t>(y_ref[i]));
        if (err > max_err) max_err = err;
        if (y_hw[i] != y_ref[i]) {
            std::printf("MISMATCH at row %d: hw=%d ref=%d\n", i, y_hw[i], y_ref[i]);
            ok = false;
        }
    }
    std::printf("dense_small<%d,%d> max_err=%lld %s\n",
                IN_F, OUT_F, static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 2: dense path, large shape (OUT_F > TILE_ROWS, triggers tile loop)
// ---------------------------------------------------------------------------
template<int IN_F, int OUT_F>
bool test_dense_large() {
    static_assert(IN_F % LANES == 0, "IN_F must be multiple of LANES");

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> wdist(-7, 7);
    std::uniform_int_distribution<int> adist(-127, 127);

    std::vector<int8_t> w_q(OUT_F * IN_F);
    std::vector<int8_t> x(IN_F);
    for (auto& v : w_q) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : x)   v = static_cast<int8_t>(adist(rng));

    auto packed = pack_weights(w_q, OUT_F, IN_F);
    auto y_ref = reference_matvec(w_q, x, OUT_F, IN_F);

    hls::stream<packed_w_beat_t> s_w;
    hls::stream<packed_a_beat_t> s_a;
    hls::stream<ap_int<32>>      s_y;

    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int W_BEAT_BYTES = LANES / 2;
    const int row_bytes = (IN_F + 1) / 2;

    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            packed_a_beat_t ab;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));

                int col_base = b * LANES + lane * 2;
                uint8_t a_lo = static_cast<uint8_t>(x[col_base + 0]);
                uint8_t a_hi = static_cast<uint8_t>(x[col_base + 1]);
                ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a_lo));
                ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a_hi));
            }
            s_w.write(wb);
            s_a.write(ab);
        }
    }

    float dummy_scales[TILE_ROWS] = {0};
    matvec_int4_tile_dense<IN_F, OUT_F>(s_w, s_a, dummy_scales, s_y);

    std::vector<int32_t> y_hw;
    for (int r = 0; r < OUT_F; ++r) {
        y_hw.push_back(static_cast<int32_t>(s_y.read()));
    }

    bool ok = true;
    int64_t max_err = 0;
    for (int i = 0; i < OUT_F; ++i) {
        int64_t err = std::llabs(static_cast<int64_t>(y_hw[i]) - static_cast<int64_t>(y_ref[i]));
        if (err > max_err) max_err = err;
        if (y_hw[i] != y_ref[i]) {
            std::printf("MISMATCH at row %d: hw=%d ref=%d\n", i, y_hw[i], y_ref[i]);
            ok = false;
        }
    }
    std::printf("dense_large<%d,%d> max_err=%lld %s\n",
                IN_F, OUT_F, static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test 3: sparse path with 2:4 mask — half the MACs are zero.
// ---------------------------------------------------------------------------
template<int IN_F, int OUT_F>
bool test_sparse_2of4() {
    static_assert(IN_F % LANES == 0, "IN_F must be multiple of LANES");

    std::mt19937 rng(2025);
    std::uniform_int_distribution<int> wdist(-7, 7);
    std::uniform_int_distribution<int> adist(-127, 127);

    std::vector<int8_t> w_q(OUT_F * IN_F, 0);
    std::vector<int8_t> x(IN_F);

    for (int r = 0; r < OUT_F; ++r) {
        for (int c = 0; c < IN_F; c += 4) {
            int alive0 = (c + 0 < IN_F) ? wdist(rng) : 0;
            int alive1 = (c + 1 < IN_F) ? wdist(rng) : 0;
            w_q[r * IN_F + c + 0] = alive0;
            w_q[r * IN_F + c + 1] = 0;   // pruned
            w_q[r * IN_F + c + 2] = alive1;
            w_q[r * IN_F + c + 3] = 0;   // pruned
        }
    }
    for (auto& v : x) v = static_cast<int8_t>(adist(rng));

    auto packed = pack_weights(w_q, OUT_F, IN_F);
    auto y_ref = reference_matvec(w_q, x, OUT_F, IN_F);

    hls::stream<packed_w_beat_t>   s_w;
    hls::stream<sparse_mask_beat_t> s_mask;
    hls::stream<packed_a_beat_t>   s_a;
    hls::stream<ap_int<32>>        s_y;

    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int W_BEAT_BYTES = LANES / 2;
    const int row_bytes = (IN_F + 1) / 2;

    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            packed_a_beat_t ab;
            sparse_mask_beat_t mb;
            for (int i = 0; i < mb.LIMBS; ++i) mb.limb[i] = 0;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));

                int col_base = b * LANES + lane * 2;
                uint8_t a_lo = static_cast<uint8_t>(x[col_base + 0]);
                uint8_t a_hi = static_cast<uint8_t>(x[col_base + 1]);
                ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a_lo));
                ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a_hi));

                int keep0 = (w_q[r * IN_F + col_base + 0] != 0) ? 1 : 0;
                int keep1 = (w_q[r * IN_F + col_base + 1] != 0) ? 1 : 0;
                mb.limb[0] |= (keep0 << (2 * lane + 0));
                mb.limb[0] |= (keep1 << (2 * lane + 1));
            }
            s_w.write(wb);
            s_mask.write(mb);
            s_a.write(ab);
        }
    }

    float dummy_scales[TILE_ROWS] = {0};
    matvec_int4_tile<IN_F, OUT_F, 1>(s_w, s_mask, s_a, dummy_scales, s_y);

    std::vector<int32_t> y_hw;
    for (int r = 0; r < OUT_F; ++r) {
        y_hw.push_back(static_cast<int32_t>(s_y.read()));
    }

    bool ok = true;
    int64_t max_err = 0;
    for (int i = 0; i < OUT_F; ++i) {
        int64_t err = std::llabs(static_cast<int64_t>(y_hw[i]) - static_cast<int64_t>(y_ref[i]));
        if (err > max_err) max_err = err;
        if (y_hw[i] != y_ref[i]) {
            std::printf("MISMATCH at row %d: hw=%d ref=%d\n", i, y_hw[i], y_ref[i]);
            ok = false;
        }
    }
    std::printf("sparse_2of4<%d,%d> max_err=%lld %s\n",
                IN_F, OUT_F, static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    bool all_ok = true;

    // Small shapes (fit inside one TILE_ROWS chunk)
    all_ok &= test_dense_small<64, 4>();
    all_ok &= test_dense_small<128, 8>();
    all_ok &= test_dense_small<256, 16>();

    // Large shape (OUT_F > TILE_ROWS, triggers outer tile loop)
    all_ok &= test_dense_large<128, 64>();

    // Sparse 2:4
    all_ok &= test_sparse_2of4<128, 32>();

    std::printf("\n%s\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
