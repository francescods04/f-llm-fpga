// fllm_loader_tb.cpp — end-to-end test: load FLLM binary → FPGA kernel → verify.
//
// Build:
//   g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
//       -o /tmp/fllm_loader_tb fpga/fllm_loader_tb.cpp && /tmp/fllm_loader_tb

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "fpga/fllm_loader.hpp"
#include "fpga/hls_stubs.hpp"
#include "fpga/matvec_int4.hpp"

static inline int8_t sext4(uint8_t n) {
    int8_t v = static_cast<int8_t>(n & 0x0F);
    if (v & 0x08) v -= 16;
    return v;
}

// Decompress packed FLLM weights to int8 for reference matvec
std::vector<int8_t> decompress_fllm(const fllm_weight_t& w) {
    uint32_t row_bytes = (w.in_features + 1) / 2;
    std::vector<int8_t> out(w.out_features * w.in_features);
    for (uint32_t r = 0; r < w.out_features; ++r) {
        for (uint32_t c = 0; c < w.in_features; c += 2) {
            uint8_t byte = w.packed[r * row_bytes + c / 2];
            int8_t lo = sext4(byte);
            int8_t hi = sext4(byte >> 4);
            out[r * w.in_features + c] = lo;
            if (c + 1 < w.in_features) {
                out[r * w.in_features + c + 1] = hi;
            }
        }
    }
    return out;
}

// Reference matvec: int8 weights * int8 activations -> int32
std::vector<int32_t> reference_matvec(
    const std::vector<int8_t>& w,
    const std::vector<int8_t>& x,
    uint32_t out_f, uint32_t in_f) {
    std::vector<int32_t> y(out_f, 0);
    for (uint32_t r = 0; r < out_f; ++r) {
        int32_t acc = 0;
        for (uint32_t c = 0; c < in_f; ++c) {
            acc += static_cast<int32_t>(w[r * in_f + c]) * static_cast<int32_t>(x[c]);
        }
        y[r] = acc;
    }
    return y;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "checkpoints/dummy-fpga/dummy_up.fllm";
    std::printf("=== FLLM loader + matvec testbench ===\n");
    std::printf("Loading: %s\n", path);

    fllm_weight_t w;
    try {
        w = load_fllm(path);
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }

    std::printf("Shape: out=%u in=%u bits=4 group_size=%u\n",
                w.out_features, w.in_features, w.group_size);
    std::printf("Packed bytes: %zu  Scales: %zu\n",
                w.packed.size(), w.scales.size());

    if (w.in_features % LANES != 0) {
        std::printf("SKIP: in_features=%u not multiple of LANES=%d\n", w.in_features, LANES);
        return 0;
    }

    // Decompress for reference
    std::vector<int8_t> w_deq = decompress_fllm(w);

    // Generate random INT8 activation
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> adist(-127, 127);
    std::vector<int8_t> x(w.in_features);
    for (auto& v : x) v = static_cast<int8_t>(adist(rng));

    // Build reference
    auto y_ref = reference_matvec(w_deq, x, w.out_features, w.in_features);

    // Build FPGA streams (depth must cover the full weight matrix)
    hls::stream<packed_w_beat_t> s_w(2000000);
    hls::stream<packed_a_beat_t> s_a(10000);
    hls::stream<ap_int<32>> s_y(5000);

    constexpr int W_BEAT_BYTES = LANES / 2;
    uint32_t row_bytes = (w.in_features + 1) / 2;
    uint32_t BEATS_PER_ROW = w.in_features / LANES;

    // Activation (one copy, buffered in kernel)
    for (uint32_t b = 0; b < BEATS_PER_ROW; ++b) {
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
            uint32_t col_base = b * LANES + lane * 2;
            uint8_t a_lo = static_cast<uint8_t>(x[col_base + 0]);
            uint8_t a_hi = static_cast<uint8_t>(x[col_base + 1]);
            ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<A_BEAT_BYTES * 8>(a_lo));
            ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<A_BEAT_BYTES * 8>(a_hi));
        }
        s_a.write(ab);
    }

    // Weights
    for (uint32_t r = 0; r < w.out_features; ++r) {
        for (uint32_t b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w_beat_t wb;
            for (int j = 0; j < wb.LIMBS; ++j) wb.limb[j] = 0;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                uint32_t byte_idx = r * row_bytes + b * W_BEAT_BYTES + lane;
                uint8_t w_byte = w.packed[byte_idx];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<W_BEAT_BYTES * 8>(w_byte));
            }
            s_w.write(wb);
        }
    }

    // Run FPGA kernel (dummy_up.fllm is 4096 x 14336 => IN=14336, OUT=4096)
    // We dispatch to the matching template instantiation.
    float dummy_scales[TILE_ROWS] = {0};
    matvec_int4_tile_dense<14336, 4096>(s_w, s_a, dummy_scales, s_y);

    // Compare
    bool ok = true;
    int64_t max_err = 0;
    for (uint32_t r = 0; r < w.out_features; ++r) {
        ap_int<32> v = s_y.read();
        int32_t hw = static_cast<int32_t>(v);
        int64_t err = std::llabs(static_cast<int64_t>(hw) - static_cast<int64_t>(y_ref[r]));
        if (err > max_err) max_err = err;
        if (hw != y_ref[r]) {
            std::printf("MISMATCH at row %u: hw=%d ref=%d\n", r, hw, y_ref[r]);
            ok = false;
        }
    }

    std::printf("Max error: %lld %s\n", static_cast<long long>(max_err),
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
