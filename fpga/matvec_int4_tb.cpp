// matvec_int4_tb.cpp - cycle-faithful behavioral testbench for matvec_int4.
//
// Compiles with a regular C++ host compiler (no HLS tools required):
//   g++ -O2 -std=c++17 fpga/matvec_int4_tb.cpp -o /tmp/matvec_int4_tb
//
// This emulates the same INT4xINT8 -> INT32 datapath the Vitis HLS kernel
// will synthesize. It loads a packed-INT4 weight binary written by
// `src/fllm/export.py` (FLLM v1 format) and a separate INT8 activation
// binary, runs the bit-exact reference matvec, and prints:
//   - first 8 output values
//   - max abs error vs a Python golden file (optional, --golden path)
//   - cycle count estimate under (LANES, TILE_ROWS, NUM_TILES) parallelism
//
// The point is to share *one* arithmetic definition between the Python
// reference and the future HLS implementation. When the real HLS code is
// written it must match this testbench's outputs byte for byte.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int LANES = 32;
constexpr int TILE_ROWS = 16;
constexpr int NUM_TILES = 4;

struct Header {
    char magic[4];          // "FLLM"
    uint16_t version;       // 1
    uint8_t bits;           // 4
    uint8_t flags;          // bit0 = per-channel scales
    uint32_t out_features;
    uint32_t in_features;
    uint32_t group_size;    // 0 = per-row
    uint8_t reserved[12];
};
static_assert(sizeof(Header) == 32, "Header layout mismatch");

// Sign-extend a 4-bit nibble to int8.
inline int8_t sext4(uint8_t n) {
    int8_t v = static_cast<int8_t>(n & 0x0F);
    if (v & 0x08) v -= 16;
    return v;
}

struct Weight {
    Header header;
    std::vector<uint8_t> packed;   // out_features * ceil(in_features/2)
    std::vector<float> scales;     // out_features * (in/group or 1)
};

bool load_weight(const std::string& path, Weight& w) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&w.header), sizeof(Header));
    if (std::strncmp(w.header.magic, "FLLM", 4) != 0) return false;
    if (w.header.version != 1 || w.header.bits != 4) return false;
    const uint32_t row_bytes = (w.header.in_features + 1) / 2;
    const size_t body_bytes = static_cast<size_t>(w.header.out_features) * row_bytes;
    w.packed.resize(body_bytes);
    f.read(reinterpret_cast<char*>(w.packed.data()), body_bytes);
    const uint32_t groups_per_row = w.header.group_size > 0
        ? w.header.in_features / w.header.group_size : 1;
    const size_t scales_count = static_cast<size_t>(w.header.out_features) * groups_per_row;
    w.scales.resize(scales_count);
    f.read(reinterpret_cast<char*>(w.scales.data()), scales_count * sizeof(float));
    return static_cast<bool>(f);
}

// Dequantize and matvec: y[r] = sum_c sext4(W[r,c]) * scale[r,c/group] * x[c]
void matvec_reference(const Weight& w, const std::vector<float>& x, std::vector<float>& y) {
    const uint32_t out_f = w.header.out_features;
    const uint32_t in_f = w.header.in_features;
    const uint32_t gsz = w.header.group_size;
    const uint32_t row_bytes = (in_f + 1) / 2;
    y.assign(out_f, 0.0f);
    for (uint32_t r = 0; r < out_f; ++r) {
        const uint8_t* row = w.packed.data() + static_cast<size_t>(r) * row_bytes;
        float acc = 0.0f;
        for (uint32_t c = 0; c < in_f; ++c) {
            uint8_t byte = row[c / 2];
            int8_t q = (c & 1) ? sext4(byte >> 4) : sext4(byte);
            const uint32_t group_idx = gsz > 0 ? c / gsz : 0;
            const uint32_t groups_per_row = gsz > 0 ? in_f / gsz : 1;
            float scale = w.scales[static_cast<size_t>(r) * groups_per_row + group_idx];
            acc += static_cast<float>(q) * scale * x[c];
        }
        y[r] = acc;
    }
}

// Cycle estimate matching cycle_sim.py's matvec_cycles for one tile bank.
uint64_t estimate_cycles(uint32_t out_f, uint32_t in_f) {
    const uint32_t parallel_rows = TILE_ROWS * NUM_TILES;
    const uint32_t rows = (out_f + parallel_rows - 1) / parallel_rows;
    const uint32_t beats_per_row = (in_f + LANES - 1) / LANES;
    const uint32_t fill = beats_per_row + TILE_ROWS;
    return static_cast<uint64_t>(rows) * beats_per_row + fill;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <weight.bin> <activation.bin> [--golden golden.bin]\n", argv[0]);
        return 1;
    }
    Weight w;
    if (!load_weight(argv[1], w)) {
        std::fprintf(stderr, "failed to load weight: %s\n", argv[1]);
        return 2;
    }
    std::ifstream af(argv[2], std::ios::binary);
    if (!af) {
        std::fprintf(stderr, "failed to open activation: %s\n", argv[2]);
        return 3;
    }
    std::vector<float> x(w.header.in_features);
    af.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));

    std::vector<float> y;
    matvec_reference(w, x, y);

    std::printf("weight: out=%u in=%u bits=%u group_size=%u\n",
        w.header.out_features, w.header.in_features, w.header.bits, w.header.group_size);
    std::printf("first 8 outputs:");
    for (int i = 0; i < 8 && i < static_cast<int>(y.size()); ++i) {
        std::printf(" %+.4f", y[i]);
    }
    std::printf("\n");

    if (argc >= 5 && std::strcmp(argv[3], "--golden") == 0) {
        std::ifstream gf(argv[4], std::ios::binary);
        if (!gf) {
            std::fprintf(stderr, "failed to open golden: %s\n", argv[4]);
            return 4;
        }
        std::vector<float> g(y.size());
        gf.read(reinterpret_cast<char*>(g.data()), g.size() * sizeof(float));
        float max_abs = 0.0f, mean_abs = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) {
            float e = std::abs(y[i] - g[i]);
            if (e > max_abs) max_abs = e;
            mean_abs += e;
        }
        mean_abs /= static_cast<float>(y.size());
        std::printf("vs golden: max_abs_err=%.6f mean_abs_err=%.6f\n", max_abs, mean_abs);
    }

    const uint64_t cycles = estimate_cycles(w.header.out_features, w.header.in_features);
    std::printf("estimated cycles (LANES=%d TILE_ROWS=%d NUM_TILES=%d): %llu\n",
        LANES, TILE_ROWS, NUM_TILES, static_cast<unsigned long long>(cycles));
    return 0;
}
