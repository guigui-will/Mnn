// Backend CUDA. Usa memória unificada (cudaMallocManaged): os mesmos ponteiros
// valem no host e no device, então Tensor/autodiff não precisam mudar nada.
// Os kernels rodam no stream padrão (ordenados); sync() só é necessário antes
// do host ler/escrever os dados.
#include "mnn/backend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_)); } while (0)

namespace mnn {
namespace {
constexpr float kEps = 1e-7f;
constexpr int TILE = 16, TPB = 256;
inline int blocks(size_t n) { return (int)((n + TPB - 1) / TPB); }

__global__ void k_fill(float* p, size_t n, float v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] = v;
}

// matmul com tiles em memória compartilhada; transposição resolvida na indexação
__global__ void k_matmul(const float* A, const float* B, float* C, int M, int K, int N,
                         bool tA, bool tB, bool acc) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + tx, bk = t * TILE + ty;
        As[ty][tx] = (row < M && ak < K) ? (tA ? A[(size_t)ak * M + row] : A[(size_t)row * K + ak]) : 0.f;
        Bs[ty][tx] = (bk < K && col < N) ? (tB ? B[(size_t)col * K + bk] : B[(size_t)bk * N + col]) : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) { size_t idx = (size_t)row * N + col; C[idx] = acc ? C[idx] + sum : sum; }
}

__global__ void k_add_bias(const float* x, const float* b, float* y, int rows, int cols) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)rows * cols) y[i] = x[i] + b[i % cols];
}
__global__ void k_col_sum(const float* x, float* out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= cols) return;
    float s = 0.f; for (int i = 0; i < rows; ++i) s += x[(size_t)i * cols + j];
    out[j] += s;
}
__global__ void k_axpy(float* y, const float* x, size_t n, float a) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i];
}
__global__ void k_relu(const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}
__global__ void k_relu_bwd(const float* x, const float* gy, float* gx, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n && x[i] > 0.f) gx[i] += gy[i];
}
__global__ void k_softmax(const float* x, float* y, int rows, int cols) {  // 1 thread por linha
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;
    const float* xr = x + (size_t)r * cols; float* yr = y + (size_t)r * cols;
    float mx = xr[0]; for (int j = 1; j < cols; ++j) mx = fmaxf(mx, xr[j]);
    float s = 0.f; for (int j = 0; j < cols; ++j) { yr[j] = expf(xr[j] - mx); s += yr[j]; }
    for (int j = 0; j < cols; ++j) yr[j] /= s;
}
__global__ void k_softmax_bwd(const float* y, const float* gy, float* gx, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;
    const float* yr = y + (size_t)r * cols; const float* gr = gy + (size_t)r * cols; float* xr = gx + (size_t)r * cols;
    float dot = 0.f; for (int j = 0; j < cols; ++j) dot += gr[j] * yr[j];
    for (int j = 0; j < cols; ++j) xr[j] += yr[j] * (gr[j] - dot);
}
__global__ void k_ce_terms(const float* p, const float* t, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) out[i] = -t[i] * logf(p[i] + kEps);
}
__global__ void k_ce_bwd(const float* p, const float* t, float* gp, size_t n, float scale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) gp[i] += -scale * t[i] / (p[i] + kEps);
}
__global__ void k_mse_terms(const float* p, const float* t, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) { float d = p[i] - t[i]; out[i] = d * d; }
}
__global__ void k_mse_bwd(const float* p, const float* t, float* gp, size_t n, float scale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) gp[i] += scale * 2.f * (p[i] - t[i]);
}
__global__ void k_sgd(float* w, const float* g, float* vel, size_t n, float lr, float mom) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { vel[i] = mom * vel[i] + g[i]; w[i] -= lr * vel[i]; }
}
__global__ void k_adam(float* w, const float* g, float* m, float* v, size_t n,
                       float lr, float b1, float b2, float eps, float c1, float c2) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    m[i] = b1 * m[i] + (1.f - b1) * g[i];
    v[i] = b2 * v[i] + (1.f - b2) * g[i] * g[i];
    w[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
}

#define LAUNCH(kernel, n, ...) do { kernel<<<blocks(n), TPB>>>(__VA_ARGS__); CK(cudaGetLastError()); } while (0)

class CudaBackend final : public Backend {
    double sum_terms(void (*launch)(const float*, const float*, float*, size_t), const float* p, const float* t, size_t n) {
        float* tmp = nullptr; CK(cudaMallocManaged(&tmp, n * sizeof(float)));
        launch(p, t, tmp, n); CK(cudaDeviceSynchronize());
        double s = 0; for (size_t i = 0; i < n; ++i) s += tmp[i];
        CK(cudaFree(tmp)); return s;
    }
public:
    const char* name() const override { return "cuda"; }
    float* alloc(size_t n) override {
        float* p = nullptr; CK(cudaMallocManaged(&p, n * sizeof(float)));
        CK(cudaDeviceSynchronize()); CK(cudaMemset(p, 0, n * sizeof(float))); return p;
    }
    void free(float* p) override { cudaFree(p); }
    void sync() override { CK(cudaDeviceSynchronize()); }
    void fill(float* p, size_t n, float v) override { LAUNCH(k_fill, n, p, n, v); }

    void matmul(const float* A, const float* B, float* C, int M, int K, int N, bool tA, bool tB, bool acc) override {
        dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        k_matmul<<<grd, blk>>>(A, B, C, M, K, N, tA, tB, acc); CK(cudaGetLastError());
    }
    void add_bias(const float* x, const float* b, float* y, int rows, int cols) override { LAUNCH(k_add_bias, (size_t)rows * cols, x, b, y, rows, cols); }
    void col_sum(const float* x, float* out, int rows, int cols) override { LAUNCH(k_col_sum, (size_t)cols, x, out, rows, cols); }
    void axpy(float* y, const float* x, size_t n, float a) override { LAUNCH(k_axpy, n, y, x, n, a); }
    void relu(const float* x, float* y, size_t n) override { LAUNCH(k_relu, n, x, y, n); }
    void relu_bwd(const float* x, const float* gy, float* gx, size_t n) override { LAUNCH(k_relu_bwd, n, x, gy, gx, n); }
    void softmax(const float* x, float* y, int rows, int cols) override { LAUNCH(k_softmax, (size_t)rows, x, y, rows, cols); }
    void softmax_bwd(const float* y, const float* gy, float* gx, int rows, int cols) override { LAUNCH(k_softmax_bwd, (size_t)rows, y, gy, gx, rows, cols); }

    double ce_loss(const float* p, const float* t, size_t n) override {
        return sum_terms([](const float* a, const float* b, float* o, size_t m) { LAUNCH(k_ce_terms, m, a, b, o, m); }, p, t, n);
    }
    void ce_bwd(const float* p, const float* t, float* gp, size_t n, float s) override { LAUNCH(k_ce_bwd, n, p, t, gp, n, s); }
    double mse_loss(const float* p, const float* t, size_t n) override {
        return sum_terms([](const float* a, const float* b, float* o, size_t m) { LAUNCH(k_mse_terms, m, a, b, o, m); }, p, t, n);
    }
    void mse_bwd(const float* p, const float* t, float* gp, size_t n, float s) override { LAUNCH(k_mse_bwd, n, p, t, gp, n, s); }

    void sgd_step(float* w, const float* g, float* vel, size_t n, float lr, float mom) override { LAUNCH(k_sgd, n, w, g, vel, n, lr, mom); }
    void adam_step(float* w, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2, float eps, float c1, float c2) override {
        LAUNCH(k_adam, n, w, g, m, v, n, lr, b1, b2, eps, c1, c2);
    }
};
}  // namespace

Backend& cuda_backend() { static CudaBackend b; return b; }
}  // namespace mnn
