// hls_stubs.hpp — minimal behavioral stubs for ap_int / hls::stream
// so that fpga/*.hpp kernels compile with a standard C++ compiler (g++/clang++).
//
// This is NOT for synthesis. It exists only to let us unit-test kernel logic
// on the host before any Vitis HLS tool is installed.

#ifndef FLLM_HLS_STUBS_HPP
#define FLLM_HLS_STUBS_HPP

#include <cstdint>
#include <cstdlib>
#include <queue>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Minimal ap_uint<N> and ap_int<N> (powers of two only, up to 1024)
// ---------------------------------------------------------------------------
template<int N>
struct ap_uint {
    static_assert(N > 0 && N <= 1024, "ap_uint stub supports 1..1024 bits");
    using backing_t = 
        typename std::conditional<(N <= 8), uint8_t,
        typename std::conditional<(N <= 16), uint16_t,
        typename std::conditional<(N <= 32), uint32_t,
        typename std::conditional<(N <= 64), uint64_t,
        uint64_t>::type>::type>::type>::type;

    backing_t v;
    static constexpr int WIDTH = N;

    ap_uint() : v(0) {}
    ap_uint(backing_t x) : v(x & mask()) {}
    ap_uint(int x) : v(static_cast<backing_t>(x) & mask()) {}

    static constexpr backing_t mask() {
        if (N >= 64) return ~backing_t(0);
        return (backing_t(1) << N) - 1;
    }

    // Range extraction: range(hi, lo) inclusive
    ap_uint<N> range(int hi, int lo) const {
        ap_uint<N> r;
        r.v = (v >> lo) & ((backing_t(1) << (hi - lo + 1)) - 1);
        return r;
    }

    // Assignment to a sub-range (not implemented in stub; not needed for testbench)
    void set_range(int hi, int lo, ap_uint<N> val) {
        backing_t m = ((backing_t(1) << (hi - lo + 1)) - 1);
        v = (v & ~(m << lo)) | ((val.v & m) << lo);
    }

    // Bitwise NOT
    ap_uint<N> operator~() const { ap_uint<N> r; r.v = (~v) & mask(); return r; }

    // Conversion to int for indexing
    operator int() const { return static_cast<int>(v); }
    operator unsigned() const { return static_cast<unsigned>(v); }

    backing_t to_uint() const { return v & mask(); }
};

template<int N>
ap_uint<N> operator&(ap_uint<N> a, ap_uint<N> b) { ap_uint<N> r; r.v = a.v & b.v; return r; }

template<int N>
ap_uint<N> operator|(ap_uint<N> a, ap_uint<N> b) { ap_uint<N> r; r.v = a.v | b.v; return r; }

template<int N>
ap_uint<N> operator^(ap_uint<N> a, ap_uint<N> b) { ap_uint<N> r; r.v = a.v ^ b.v; return r; }

template<int N>
ap_uint<N>& operator&=(ap_uint<N>& a, ap_uint<N> b) { a.v &= b.v; return a; }

template<int N>
ap_uint<N>& operator|=(ap_uint<N>& a, ap_uint<N> b) { a.v |= b.v; return a; }

template<int N>
ap_uint<N> operator<<(ap_uint<N> a, int s) { ap_uint<N> r; r.v = (a.v << s) & a.mask(); return r; }

template<int N>
ap_uint<N> operator>>(ap_uint<N> a, int s) { ap_uint<N> r; r.v = a.v >> s; return r; }

// ap_int<N> — signed version using two's complement
template<int N>
struct ap_int {
    static_assert(N > 0 && N <= 1024, "ap_int stub supports 1..1024 bits");
    using backing_u = 
        typename std::conditional<(N <= 8), uint8_t,
        typename std::conditional<(N <= 16), uint16_t,
        typename std::conditional<(N <= 32), uint32_t,
        typename std::conditional<(N <= 64), uint64_t,
        uint64_t>::type>::type>::type>::type;

    backing_u v;
    static constexpr int WIDTH = N;

    ap_int() : v(0) {}
    ap_int(int x) {
        if (N < 32) {
            int mask = (1 << N) - 1;
            v = static_cast<backing_u>(x & mask);
        } else {
            v = static_cast<backing_u>(x);
        }
    }
    ap_int(long long x) {
        v = static_cast<backing_u>(x);
    }

    // Sign-extend to signed long long for arithmetic
    long long to_int64() const {
        if (N >= 64) return static_cast<long long>(v);
        long long s = static_cast<long long>(v);
        if (s & (1LL << (N - 1))) {
            s |= ~((1LL << N) - 1);
        }
        return s;
    }

    operator int() const { return static_cast<int>(to_int64()); }
    operator long long() const { return to_int64(); }

    ap_int<N> operator+(ap_int<N> o) const { ap_int<N> r; r.v = (v + o.v); return r; }
    ap_int<N> operator-(ap_int<N> o) const { ap_int<N> r; r.v = (v - o.v); return r; }
    ap_int<N> operator*(ap_int<N> o) const { ap_int<N> r; r.v = (v * o.v); return r; }
    ap_int<N> operator*(int o) const { ap_int<N> r; r.v = (v * static_cast<backing_u>(o)); return r; }
    ap_int<N>& operator+=(ap_int<N> o) { v += o.v; return *this; }

    bool operator!=(ap_int<N> o) const { return v != o.v; }
};

// Mixed-width multiply: ap_int<M> * ap_int<N> -> ap_int<M>
template<int M, int N>
ap_int<M> operator*(ap_int<M> a, ap_int<N> b) {
    ap_int<M> r;
    r.v = a.v * static_cast<typename ap_int<M>::backing_u>(b.to_int64());
    return r;
}

// ap_uint range specialization for the syntax .range(hi,lo)
template<int N>
struct ap_range_proxy {
    ap_uint<N>* parent;
    int hi, lo;
    ap_range_proxy(ap_uint<N>* p, int h, int l) : parent(p), hi(h), lo(l) {}
    operator ap_uint<N>() const { return parent->range(hi, lo); }
    ap_range_proxy& operator=(ap_uint<N> val) {
        parent->set_range(hi, lo, val);
        return *this;
    }
};

// Enable .range() syntax via macro (ugly but matches HLS header style)
#define RANGE_PROXY_STUB(T, name) \
    ap_range_proxy<T::WIDTH> name(int hi, int lo) { return ap_range_proxy<T::WIDTH>(this, hi, lo); } \
    ap_uint<T::WIDTH> range(int hi, int lo) const { return this->range(hi, lo); }

// Since we can't add methods to the struct after definition, we use a wrapper
// macro for the testbench. In real HLS the headers provide this natively.

// ---------------------------------------------------------------------------
// Minimal hls::stream<T>
// ---------------------------------------------------------------------------
namespace hls {

template<typename T>
class stream {
    std::queue<T> q;
    std::size_t cap;
public:
    stream() : cap(16) {}
    explicit stream(std::size_t depth) : cap(depth) {}

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

// ---------------------------------------------------------------------------
// Pragma stubs — no-ops so #pragma HLS ... compiles away.
// ---------------------------------------------------------------------------
// The existing kernel headers use these pragmas.  We can't intercept them
// without a preprocessor, but g++ ignores unknown pragmas with -Wno-unknown-pragmas.
// So we rely on the build flag: -Wno-unknown-pragmas

// ---------------------------------------------------------------------------
// Missing HLS macros that appear in kernel code.
// ---------------------------------------------------------------------------
// #pragma HLS ARRAY_PARTITION variable=accum complete dim=1  -> no-op
// #pragma HLS PIPELINE II=1 style=flp                        -> no-op
// #pragma HLS UNROLL                                          -> no-op
// #pragma HLS INTERFACE ...                                   -> no-op
// #pragma HLS INLINE off                                      -> no-op
// #pragma HLS STREAM variable=dummy_mask depth=1              -> no-op

#endif // FLLM_HLS_STUBS_HPP
