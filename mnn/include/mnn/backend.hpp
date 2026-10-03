// Interface de backend: todo o "trabalho numérico" passa por aqui.
// Tensor/autodiff/camadas só conhecem esta interface; CPU e CUDA a implementam.
#pragma once
#include <cstddef>
#include <string>

namespace mnn {

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;

    // memória (sempre inicializada com zeros)
    virtual float* alloc(size_t n) = 0;
    virtual void free(float* p) = 0;
    virtual void fill(float* p, size_t n, float v) = 0;
    virtual void sync() {}  // espera o dispositivo terminar (no-op na CPU)

    // C(MxN) (+)= op(A)(MxK) * op(B)(KxN). Matrizes row-major.
    // tA: A está guardada como KxM; tB: B está guardada como NxK.
    virtual void matmul(const float* A, const float* B, float* C, int M, int K, int N,
                        bool tA, bool tB, bool accumulate) = 0;

    virtual void add_bias(const float* x, const float* b, float* y, int rows, int cols) = 0;
    virtual void col_sum(const float* x, float* out, int rows, int cols) = 0;  // out += soma por coluna
    virtual void axpy(float* y, const float* x, size_t n, float alpha) = 0;    // y += alpha * x

    virtual void relu(const float* x, float* y, size_t n) = 0;
    virtual void relu_bwd(const float* x, const float* gy, float* gx, size_t n) = 0;  // gx += gy*(x>0)

    virtual void softmax(const float* x, float* y, int rows, int cols) = 0;
    virtual void softmax_bwd(const float* y, const float* gy, float* gx, int rows, int cols) = 0;

    // Losses: retornam a SOMA (o chamador divide). Backward acumula em gp.
    virtual double ce_loss(const float* p, const float* t, size_t n) = 0;
    virtual void ce_bwd(const float* p, const float* t, float* gp, size_t n, float scale) = 0;
    virtual double mse_loss(const float* p, const float* t, size_t n) = 0;
    virtual void mse_bwd(const float* p, const float* t, float* gp, size_t n, float scale) = 0;

    // Passos dos otimizadores
    virtual void sgd_step(float* w, const float* g, float* vel, size_t n, float lr, float momentum) = 0;
    virtual void adam_step(float* w, const float* g, float* m, float* v, size_t n,
                           float lr, float b1, float b2, float eps, float c1, float c2) = 0;
};

Backend& cpu_backend();
#ifdef MNN_WITH_CUDA
Backend& cuda_backend();
#endif

Backend& backend();               // backend atual (padrão: CPU)
void set_backend(Backend& b);     // troque ANTES de criar tensores
bool use_backend(const std::string& name);  // "cpu" ou "cuda"; false se indisponível

}  // namespace mnn
