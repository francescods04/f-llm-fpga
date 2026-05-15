// matvec_int2_tb.cpp — host-compilable unit test for INT2 matvec.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas \
//       -I. fpga/matvec_int2_tb.cpp -o /tmp/matvec_int2_tb
// Run:
//   /tmp/matvec_int2_tb

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/matvec_int2.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline int8_t sext2(uint8_t n) {
    // 00 -> -1, 01 -> 0, 10 -> +1, 11 -> 0
    switch (n & 0x03) {
        case 0: return -1;
        case 1: return 0;
        case 2: return 1;
        default: return 0;
    }
}

static inline uint8_t pack_2bits(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    auto enc = [](int8_t v) -> uint8_t {
        if (v == -1) return 0;
        if (v == 0)  return 1;
        if (v == 1)  return 2;
        return 3;
    };
    return enc(v0) | (enc(v1) << 2) | (enc(v2) << 4) | (enc(v3) << 6);
}

// ---------------------------------------------------------------------------
// Build packed weight matrix (row-major, 4 ternary values per byte).
// ---------------------------------------------------------------------------
std::vector<uint8_t> pack_weights(const std::vector<int8_t>& w_q, int out_f, int in_f) {
    std::vector<uint8_t> packed;
    packed.reserve(out_f * ((in_f + 3) / 4));
    for (int r = 0; r < out_f; ++r) {
        for (int c = 0; c < in_f; c += 4) {
            int8_t v0 = w_q[r * in_f + c + 0];
            int8_t v1 = (c + 1 < in_f) ? w_q[r * in_f + c + 1] : 0;
            int8_t v2 = (c + 2 < in_f) ? w_q[r * in_f + c + 2] : 0;
            int8_t v3 = (c + 3 < in_f) ? w_q[r * in_f + c + 3] : 0;
            packed.push_back(pack_2bits(v0, v1, v2, v3));
        }
    }
    return packed;
}

// ---------------------------------------------------------------------------
// Reference matvec (ternary * int8 -> int32).
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
// Test: small shape
// ---------------------------------------------------------------------------
template<int IN_F, int OUT_F>
bool test_small() {
    static_assert(IN_F % LANES == 0, "IN_F must be multiple of LANES");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> wdist(-1, 1);   // ternary
    std::uniform_int_distribution<int> adist(-127, 127); // INT8

    std::vector<int8_t> w_q(OUT_F * IN_F);
    std::vector<int8_t> x(IN_F);
    for (auto& v : w_q) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : x)   v = static_cast<int8_t>(adist(rng));

    auto packed = pack_weights(w_q, OUT_F, IN_F);
    auto y_ref = reference_matvec(w_q, x, OUT_F, IN_F);

    // Drive streams
    hls::stream<packed_w_beat_t> s_w;
    hls::stream<packed_a_beat_t> s_a;
    hls::stream<ap_int<32>>      s_y;

    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int W_BEAT_BYTES = LANES / 4;
    const int row_bytes = (IN_F + 3) / 4;

    // Activation vector (one copy, buffered inside kernel)
    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
            int col_base = b * LANES + lane * 4;
            uint8_t a0 = static_cast<uint8_t>(x[col_base + 0]);
            uint8_t a1 = static_cast<uint8_t>(x[col_base + 1]);
            uint8_t a2 = static_cast<uint8_t>(x[col_base + 2]);
            uint8_t a3 = static_cast<uint8_t>(x[col_base + 3]);
            ab.set_range(32 * lane + 7,  32 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a0));
            ab.set_range(32 * lane + 15, 32 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a1));
            ab.set_range(32 * lane + 23, 32 * lane + 16, ap_uint<A_BEAT_BYTES * 8>(a2));
            ab.set_range(32 * lane + 31, 32 * lane + 24, ap_uint<A_BEAT_BYTES * 8>(a3));
        }
        s_a.write(ab);
    }

    // Weight matrix
    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            for (int j = 0; j < wb.LIMBS; ++j) wb.limb[j] = 0;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));
            }
            s_w.write(wb);
        }
    }

    float dummy_scales[TILE_ROWS] = {0};
    matvec_int2_tile<IN_F, OUT_F>(s_w, s_a, dummy_scales, s_y);

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
    std::printf("int2_small<%d,%d> max_err=%lld %s\n",
                IN_F, OUT_F, static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Test: large shape (OUT_F > TILE_ROWS)
// ---------------------------------------------------------------------------
template<int IN_F, int OUT_F>
bool test_large() {
    static_assert(IN_F % LANES == 0, "IN_F must be multiple of LANES");

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> wdist(-1, 1);
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
    constexpr int W_BEAT_BYTES = LANES / 4;
    const int row_bytes = (IN_F + 3) / 4;

    // Activation
    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
            int col_base = b * LANES + lane * 4;
            uint8_t a0 = static_cast<uint8_t>(x[col_base + 0]);
            uint8_t a1 = static_cast<uint8_t>(x[col_base + 1]);
            uint8_t a2 = static_cast<uint8_t>(x[col_base + 2]);
            uint8_t a3 = static_cast<uint8_t>(x[col_base + 3]);
            ab.set_range(32 * lane + 7,  32 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a0));
            ab.set_range(32 * lane + 15, 32 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a1));
            ab.set_range(32 * lane + 23, 32 * lane + 16, ap_uint<A_BEAT_BYTES * 8>(a2));
            ab.set_range(32 * lane + 31, 32 * lane + 24, ap_uint<A_BEAT_BYTES * 8>(a3));
        }
        s_a.write(ab);
    }

    // Weights
    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            for (int j = 0; j < wb.LIMBS; ++j) wb.limb[j] = 0;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));
            }
            s_w.write(wb);
        }
    }

    float dummy_scales[TILE_ROWS] = {0};
    matvec_int2_tile<IN_F, OUT_F>(s_w, s_a, dummy_scales, s_y);

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
    std::printf("int2_large<%d,%d> max_err=%lld %s\n",
                IN_F, OUT_F, static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    bool ok = true;
    ok &= test_small<64, 16>();
    ok &= test_small<128, 16>();
    ok &= test_small<256, 32>();
    ok &= test_large<128, 64>();

    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
