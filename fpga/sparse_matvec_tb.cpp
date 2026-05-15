// sparse_matvec_tb.cpp — host testbench for row-skip sparse matvec.
//
// Build:
//   g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
//       -o /tmp/sparse_matvec_tb fpga/sparse_matvec_tb.cpp && /tmp/sparse_matvec_tb

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/sparse_matvec.hpp"

int main() {
    std::printf("=== sparse_matvec_tile testbench ===\n");

    constexpr int IN_F = 64;
    constexpr int OUT_F = 32;
    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int ROWS = (OUT_F + TILE_ROWS - 1) / TILE_ROWS;

    // Row mask: skip every other row (50% density)
    RowMask<OUT_F> mask;
    for (int w = 0; w < mask.WORDS; ++w) mask.word[w] = 0ULL;
    int num_dense = 0;
    for (int r = 0; r < OUT_F; ++r) {
        if (r % 2 == 0) {
            int wi = r / 64;
            int bi = r % 64;
            mask.word[wi] |= (1ULL << bi);
            num_dense++;
        }
    }

    // Dense weights for dense rows only
    std::vector<uint8_t> packed(num_dense * BEATS_PER_ROW * (LANES / 2));
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(i & 0xFF);

    // Activation
    std::vector<int8_t> a(IN_F);
    for (int i = 0; i < IN_F; ++i) a[i] = static_cast<int8_t>(i);

    hls::stream<packed_w_beat_t> s_w(10000);
    hls::stream<packed_a_beat_t> s_a(1000);
    hls::stream<ap_int<32>> s_y(500);

    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < LANES / 2; ++lane) {
            int idx = b * LANES + lane * 2;
            ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<16>(static_cast<uint8_t>(a[idx + 0])));
            ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<16>(static_cast<uint8_t>(a[idx + 1])));
        }
        s_a.write(ab);
    }

    for (size_t i = 0; i < packed.size(); ++i) {
        packed_w_beat_t wb;
        for (int j = 0; j < wb.LIMBS; ++j) wb.limb[j] = 0;
        wb.set_range(7, 0, ap_uint<8>(packed[i]));
        s_w.write(wb);
    }

    float scales[TILE_ROWS] = {1.0f};
    sparse_matvec_tile<IN_F, OUT_F>(s_w, s_a, mask, num_dense, scales, s_y);

    // Verify shape: should get OUT_F outputs
    int count = 0;
    while (!s_y.empty()) {
        ap_int<32> v = s_y.read();
        count++;
    }
    std::printf("Outputs produced: %d (expected %d) %s\n",
                count, OUT_F, (count == OUT_F) ? "PASS" : "FAIL");
    return (count == OUT_F) ? 0 : 1;
}
