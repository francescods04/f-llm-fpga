// fllm_loader.hpp — minimal C++ loader for FLLM v1 packed INT4 binaries.
//
// Usage:
//   fllm_weight_t w = load_fllm("checkpoints/dummy-fpga/dummy_qkv.fllm");
//   // w.packed is a std::vector<uint8_t> of ceil(in/2)*out bytes
//   // w.scales is a std::vector<float> of out_features entries

#ifndef FLLM_LOADER_HPP
#define FLLM_LOADER_HPP

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

struct fllm_weight_t {
    uint32_t out_features;
    uint32_t in_features;
    uint32_t group_size;  // 0 = per-row scales
    std::vector<uint8_t> packed;   // ceil(in/2) * out bytes
    std::vector<float>   scales;   // out_features floats (or out*groups)
};

inline fllm_weight_t load_fllm(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }

    // Header: 32 bytes
    // magic[4], version[2], bits[1], flags[1], out_f[4], in_f[4], group_size[4], reserved[12]
    uint8_t header[32];
    if (std::fread(header, 1, 32, f) != 32) {
        std::fclose(f);
        throw std::runtime_error("Failed to read header from " + path);
    }

    if (header[0] != 'F' || header[1] != 'L' || header[2] != 'L' || header[3] != 'M') {
        std::fclose(f);
        throw std::runtime_error("Invalid FLLM magic in " + path);
    }

    uint16_t version = header[4] | (header[5] << 8);
    if (version != 1) {
        std::fclose(f);
        throw std::runtime_error("Unsupported FLLM version in " + path);
    }

    uint8_t bits = header[6];
    // uint8_t flags = header[7];

    uint32_t out_features = *reinterpret_cast<uint32_t*>(&header[8]);
    uint32_t in_features  = *reinterpret_cast<uint32_t*>(&header[12]);
    uint32_t group_size   = *reinterpret_cast<uint32_t*>(&header[16]);

    // Packed weights: ceil(in/2) * out bytes
    uint32_t row_bytes = (in_features + 1) / 2;
    size_t packed_bytes = static_cast<size_t>(row_bytes) * out_features;
    std::vector<uint8_t> packed(packed_bytes);
    if (std::fread(packed.data(), 1, packed_bytes, f) != packed_bytes) {
        std::fclose(f);
        throw std::runtime_error("Failed to read packed weights from " + path);
    }

    // Scales: float32 per output channel (or per group)
    uint32_t num_scales = (group_size == 0) ? out_features : (out_features * (in_features / group_size));
    std::vector<float> scales(num_scales);
    if (std::fread(scales.data(), sizeof(float), num_scales, f) != num_scales) {
        std::fclose(f);
        throw std::runtime_error("Failed to read scales from " + path);
    }

    std::fclose(f);

    return fllm_weight_t{
        out_features,
        in_features,
        group_size,
        std::move(packed),
        std::move(scales),
    };
}

#endif  // FLLM_LOADER_HPP
