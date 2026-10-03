#include "mnn/backend.hpp"
#include <algorithm>
#include <cmath>

namespace mnn {
namespace {
constexpr float kEps = 1e-7f;

class CpuBackend final : public Backend {
public:
    const char* name() const override { return "cpu"; }
    float* alloc(size_t n) override { return new float[n](); }
    void free(float* p) override { delete[] p; }
    void fill(float* p, size_t n, float v) override { std::fill(p, p + n, v); }

    void matmul(const float* A, const float* B, float* C, int M, int K, int N,
                bool tA, bool tB, bool acc) override {
        if (!acc) std::fill(C, C + (size_t)M * N, 0.f);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                float a = tA ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
                float* c = C + (size_t)i * N;
                if (!tB) {
                    const float* b = B + (size_t)k * N;
                    for (int j = 0; j < N; ++j) c[j] += a * b[j];
                } else {
                    for (int j = 0; j < N; ++j) c[j] += a * B[(size_t)j * K + k];
                }
            }
    }

    void add_bias(const float* x, const float* b, float* y, int rows, int cols) override {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) y[(size_t)i * cols + j] = x[(size_t)i * cols + j] + b[j];
    }
    void col_sum(const float* x, float* out, int rows, int cols) override {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) out[j] += x[(size_t)i * cols + j];
    }
    void axpy(float* y, const float* x, size_t n, float alpha) override {
        for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
    }

    void relu(const float* x, float* y, size_t n) override {
        for (size_t i = 0; i < n; ++i) y[i] = x[i] > 0.f ? x[i] : 0.f;
    }
    void relu_bwd(const float* x, const float* gy, float* gx, size_t n) override {
        for (size_t i = 0; i < n; ++i) if (x[i] > 0.f) gx[i] += gy[i];
    }

    void softmax(const float* x, float* y, int rows, int cols) override {
        for (int i = 0; i < rows; ++i) {
            const float* xr = x + (size_t)i * cols; float* yr = y + (size_t)i * cols;
            float mx = *std::max_element(xr, xr + cols), s = 0.f;
            for (int j = 0; j < cols; ++j) { yr[j] = std::exp(xr[j] - mx); s += yr[j]; }
            for (int j = 0; j < cols; ++j) yr[j] /= s;
        }
    }
    void softmax_bwd(const float* y, const float* gy, float* gx, int rows, int cols) override {
        for (int i = 0; i < rows; ++i) {
            const float* yr = y + (size_t)i * cols; const float* gr = gy + (size_t)i * cols;
            float* xr = gx + (size_t)i * cols; float dot = 0.f;
            for (int j = 0; j < cols; ++j) dot += gr[j] * yr[j];
            for (int j = 0; j < cols; ++j) xr[j] += yr[j] * (gr[j] - dot);
        }
    }

    double ce_loss(const float* p, const float* t, size_t n) override {
        double s = 0; for (size_t i = 0; i < n; ++i) s -= (double)t[i] * std::log(p[i] + kEps); return s;
    }
    void ce_bwd(const float* p, const float* t, float* gp, size_t n, float scale) override {
        for (size_t i = 0; i < n; ++i) gp[i] += -scale * t[i] / (p[i] + kEps);
    }
    double mse_loss(const float* p, const float* t, size_t n) override {
        double s = 0; for (size_t i = 0; i < n; ++i) { double d = (double)p[i] - t[i]; s += d * d; } return s;
    }
    void mse_bwd(const float* p, const float* t, float* gp, size_t n, float scale) override {
        for (size_t i = 0; i < n; ++i) gp[i] += scale * 2.f * (p[i] - t[i]);
    }

    void sgd_step(float* w, const float* g, float* vel, size_t n, float lr, float mom) override {
        for (size_t i = 0; i < n; ++i) { vel[i] = mom * vel[i] + g[i]; w[i] -= lr * vel[i]; }
    }
    void adam_step(float* w, const float* g, float* m, float* v, size_t n,
                   float lr, float b1, float b2, float eps, float c1, float c2) override {
        for (size_t i = 0; i < n; ++i) {
            m[i] = b1 * m[i] + (1 - b1) * g[i];
            v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
            w[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
        }
    }
};
}  // namespace

Backend& cpu_backend() { static CpuBackend b; return b; }
}  // namespace mnn
