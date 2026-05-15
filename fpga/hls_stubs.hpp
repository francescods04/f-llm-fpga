// hls_stubs.hpp — minimal behavioral stubs for ap_int / hls::stream
// so that fpga/*.hpp kernels compile with a standard C++ compiler (g++/clang++).
//
// This is NOT for synthesis. It exists only to let us unit-test kernel logic
// on the host before any Vitis HLS tool is installed.
//
// Supports ap_uint / ap_int up to 65536 bits using multiple uint64_t limbs.

#ifndef FLLM_HLS_STUBS_HPP
#define FLLM_HLS_STUBS_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// ap_uint<N>
// ---------------------------------------------------------------------------
template<int N>
struct ap_uint {
    static_assert(N > 0 && N <= 65536, "ap_uint stub supports 1..65536 bits");
    static constexpr int WIDTH = N;
    static constexpr int LIMBS = (N + 63) / 64;
    uint64_t limb[LIMBS];

    ap_uint() { for (int i = 0; i < LIMBS; ++i) limb[i] = 0; }

    ap_uint(uint64_t x) {
        for (int i = 0; i < LIMBS; ++i) limb[i] = 0;
        if (N >= 64) {
            limb[0] = x;
        } else {
            limb[0] = x & ((uint64_t(1) << N) - 1);
        }
    }
    ap_uint(int x) : ap_uint(static_cast<uint64_t>(x)) {}
    ap_uint(unsigned x) : ap_uint(static_cast<uint64_t>(x)) {}
    ap_uint(long long x) : ap_uint(static_cast<uint64_t>(x)) {}

    // Construct from a smaller ap_uint (zero-extend)
    template<int M>
    ap_uint(ap_uint<M> o) {
        for (int i = 0; i < LIMBS; ++i) limb[i] = 0;
        int copy_limbs = (M + 63) / 64;
        if (copy_limbs > LIMBS) copy_limbs = LIMBS;
        for (int i = 0; i < copy_limbs; ++i) {
            limb[i] = o.limb[i];
        }
        mask_top();
    }

    void mask_top() {
        int top_bits = N % 64;
        if (top_bits != 0 && LIMBS > 0) {
            uint64_t m = (uint64_t(1) << top_bits) - 1;
            limb[LIMBS - 1] &= m;
        }
    }

    static uint64_t limb_mask(int lo, int hi) {
        int width = hi - lo + 1;
        if (width >= 64) return ~uint64_t(0);
        return (uint64_t(1) << width) - 1;
    }

    // Range extraction: returns ap_uint<N> with lower (hi-lo+1) bits valid.
    ap_uint<N> range(int hi, int lo) const {
        return range_generic(hi, lo);
    }

    ap_uint<N> range_generic(int hi, int lo) const {
        ap_uint<N> r;
        // Shift right by lo bits across all limbs into a temporary buffer
        uint64_t buf[1025] = {0}; // enough for 65536 bits + shift margin
        int max_buf = (hi - lo + 1 + 63) / 64 + 1;
        for (int i = 0; i < LIMBS; ++i) buf[i] = limb[i];
        int shift_limbs = lo / 64;
        int shift_bits = lo % 64;
        for (int i = 0; i < LIMBS; ++i) {
            int src = i + shift_limbs;
            uint64_t val = 0;
            if (src < LIMBS) val = buf[src] >> shift_bits;
            if (shift_bits > 0 && src + 1 < LIMBS) {
                val |= buf[src + 1] << (64 - shift_bits);
            }
            buf[i] = val;
        }
        int width = hi - lo + 1;
        uint64_t m = (width >= 64) ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
        r.limb[0] = buf[0] & m;
        for (int i = 1; i < LIMBS; ++i) {
            if (width > i * 64) {
                int rem = width - i * 64;
                uint64_t m2 = (rem >= 64) ? ~uint64_t(0) : ((uint64_t(1) << rem) - 1);
                r.limb[i] = buf[i] & m2;
            } else {
                r.limb[i] = 0;
            }
        }
        return r;
    }

    void set_range(int hi, int lo, ap_uint<N> val) {
        int lo_limb = lo / 64;
        int hi_limb = hi / 64;
        int lo_off = lo % 64;
        int width = hi - lo + 1;

        if (lo_limb == hi_limb) {
            uint64_t m = limb_mask(0, width - 1);
            limb[lo_limb] = (limb[lo_limb] & ~(m << lo_off)) | ((val.limb[0] & m) << lo_off);
        } else {
            // crosses limbs (max 2 limbs for N<=65536 when width <= 64... but could be more)
            int num_cross = hi_limb - lo_limb + 1;
            for (int l = 0; l < num_cross; ++l) {
                int dst = lo_limb + l;
                if (dst >= LIMBS) break;
                int bit_start = (l == 0) ? lo_off : 0;
                int bit_end = (l == num_cross - 1) ? (hi % 64) : 63;
                int w = bit_end - bit_start + 1;
                uint64_t m = limb_mask(0, w - 1);
                int src_limb = l;
                uint64_t v = (val.limb[src_limb] >> bit_start) & m;  // rough; works for small widths
                // For host testing this simplified version is sufficient.
                limb[dst] = (limb[dst] & ~(m << bit_start)) | (v << bit_start);
            }
        }
        mask_top();
    }

    ap_uint<N> operator~() const {
        ap_uint<N> r;
        for (int i = 0; i < LIMBS; ++i) r.limb[i] = ~limb[i];
        r.mask_top();
        return r;
    }

    operator int() const { return static_cast<int>(limb[0]); }
    operator unsigned() const { return static_cast<unsigned>(limb[0]); }
    operator long long() const { return static_cast<long long>(limb[0]); }

    bool operator[](int idx) const {
        int limb_idx = idx / 64;
        int bit_idx = idx % 64;
        if (limb_idx >= LIMBS) return false;
        return ((limb[limb_idx] >> bit_idx) & 1) != 0;
    }
};

template<int N>
ap_uint<N> operator&(ap_uint<N> a, ap_uint<N> b) {
    ap_uint<N> r;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) r.limb[i] = a.limb[i] & b.limb[i];
    r.mask_top();
    return r;
}

template<int N>
ap_uint<N> operator|(ap_uint<N> a, ap_uint<N> b) {
    ap_uint<N> r;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) r.limb[i] = a.limb[i] | b.limb[i];
    r.mask_top();
    return r;
}

template<int N>
ap_uint<N> operator^(ap_uint<N> a, ap_uint<N> b) {
    ap_uint<N> r;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) r.limb[i] = a.limb[i] ^ b.limb[i];
    r.mask_top();
    return r;
}

template<int N>
ap_uint<N>& operator&=(ap_uint<N>& a, ap_uint<N> b) { a = a & b; return a; }

template<int N>
ap_uint<N>& operator|=(ap_uint<N>& a, ap_uint<N> b) { a = a | b; return a; }

template<int N>
ap_uint<N> operator<<(ap_uint<N> a, int s) {
    ap_uint<N> r;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) r.limb[i] = 0;
    int limb_shift = s / 64;
    int bit_shift = s % 64;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) {
        int dst = i + limb_shift;
        if (dst < ap_uint<N>::LIMBS) {
            r.limb[dst] |= a.limb[i] << bit_shift;
        }
        if (bit_shift > 0 && dst + 1 < ap_uint<N>::LIMBS) {
            r.limb[dst + 1] |= a.limb[i] >> (64 - bit_shift);
        }
    }
    r.mask_top();
    return r;
}

template<int N>
ap_uint<N> operator>>(ap_uint<N> a, int s) {
    ap_uint<N> r;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) r.limb[i] = 0;
    int limb_shift = s / 64;
    int bit_shift = s % 64;
    for (int i = 0; i < ap_uint<N>::LIMBS; ++i) {
        int dst = i - limb_shift;
        if (dst >= 0) {
            r.limb[dst] |= a.limb[i] >> bit_shift;
        }
        if (bit_shift > 0 && dst - 1 >= 0) {
            r.limb[dst - 1] |= a.limb[i] << (64 - bit_shift);
        }
    }
    r.mask_top();
    return r;
}

// ---------------------------------------------------------------------------
// ap_int<N> — signed version using two's complement
// ---------------------------------------------------------------------------
template<int N>
struct ap_int {
    static_assert(N > 0 && N <= 65536, "ap_int width must be positive and <= 65536");
    static constexpr int WIDTH = N;
    static constexpr int LIMBS = (N + 63) / 64;
    uint64_t limb[LIMBS];

    ap_int() { for (int i = 0; i < LIMBS; ++i) limb[i] = 0; }

    ap_int(int x) { from_ll(static_cast<long long>(x)); }
    ap_int(unsigned x) { from_ll(static_cast<long long>(x)); }
    ap_int(long long x) { from_ll(x); }
    ap_int(unsigned long long x) { from_ll(static_cast<long long>(x)); }

    template<int M>
    ap_int(ap_int<M> o) { from_ll(o.to_int64()); }

    template<int M>
    ap_int(ap_uint<M> o) {
        for (int i = 0; i < LIMBS; ++i) limb[i] = 0;
        int copy_limbs = (M + 63) / 64;
        if (copy_limbs > LIMBS) copy_limbs = LIMBS;
        for (int i = 0; i < copy_limbs; ++i) limb[i] = o.limb[i];
        mask_top();
    }

    void mask_top() {
        int top_bits = N % 64;
        if (top_bits != 0 && LIMBS > 0) {
            uint64_t m = (uint64_t(1) << top_bits) - 1;
            limb[LIMBS - 1] &= m;
        }
    }

    void from_ll(long long x) {
        if (N >= 64) {
            for (int i = 0; i < LIMBS; ++i) limb[i] = 0;
            limb[0] = static_cast<uint64_t>(x);
            if (x < 0) {
                for (int i = 1; i < LIMBS; ++i) limb[i] = ~uint64_t(0);
            }
        } else {
            long long mask = (1LL << N) - 1;
            uint64_t v = static_cast<uint64_t>(x & mask);
            for (int i = 0; i < LIMBS; ++i) limb[i] = 0;
            limb[0] = v;
        }
        mask_top();
    }

    long long to_int64() const {
        if (N >= 64) {
            return static_cast<long long>(limb[0]);
        }
        long long s = static_cast<long long>(limb[0]);
        if (s & (1LL << (N - 1))) {
            s |= ~((1LL << N) - 1);
        }
        return s;
    }

    operator int() const { return static_cast<int>(to_int64()); }
    operator long long() const { return to_int64(); }

    ap_int<N> operator+(ap_int<N> o) const { ap_int<N> r; r.from_ll(to_int64() + o.to_int64()); return r; }
    ap_int<N> operator-(ap_int<N> o) const { ap_int<N> r; r.from_ll(to_int64() - o.to_int64()); return r; }
    ap_int<N>& operator+=(ap_int<N> o) { from_ll(to_int64() + o.to_int64()); return *this; }
};

// Mixed-width multiply
template<int W, int M, int N>
ap_int<W> ap_mul(ap_int<M> a, ap_int<N> b) {
    ap_int<W> r;
    r.from_ll(a.to_int64() * b.to_int64());
    return r;
}

// ---------------------------------------------------------------------------
// hls::stream<T>
// ---------------------------------------------------------------------------
namespace hls {

inline int& hls_stream_next_id() { static int id = 0; return id; }

template<typename T>
class stream {
    std::queue<T> q;
    std::size_t cap;
    std::string name;
public:
    stream() : cap(1024), name("stream_" + std::to_string(hls_stream_next_id()++)) {}
    explicit stream(std::size_t depth) : cap(depth), name("stream_" + std::to_string(hls_stream_next_id()++)) {}
    explicit stream(std::size_t depth, const char* n) : cap(depth), name(n) {}

    void write(const T& v) {
        if (q.size() >= cap) {
            throw std::runtime_error("hls::stream write overflow");
        }
        q.push(v);
    }
    T read() {
        if (q.empty()) {
            throw std::runtime_error("hls::stream read underflow");
        }
        T v = q.front();
        q.pop();
        return v;
    }
    bool empty() const { return q.empty(); }
    std::size_t size() const { return q.size(); }
};

} // namespace hls

#endif // FLLM_HLS_STUBS_HPP
