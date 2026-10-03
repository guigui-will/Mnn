// Tensor 2D (rows x cols) + autodiff por modo reverso (grafo dinâmico).
#pragma once
#include "backend.hpp"
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mnn {

// Memória do dispositivo com RAII; lembra qual backend a criou.
struct Buffer {
    Backend* be; float* ptr; size_t n;
    explicit Buffer(size_t n_, Backend* b = &backend()) : be(b), ptr(b->alloc(n_)), n(n_) {}
    ~Buffer() { be->free(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Nó do grafo: dados, gradiente (lazy), pais e a função de backward.
struct TensorImpl {
    int rows = 0, cols = 0;
    std::shared_ptr<Buffer> data, grad;
    bool requires_grad = false;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void(TensorImpl&)> backward_fn;  // acumula nos grads dos pais
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    Tensor() = default;
    Tensor(int r, int c, bool requires_grad = false) : impl(std::make_shared<TensorImpl>()) {
        impl->rows = r; impl->cols = c; impl->requires_grad = requires_grad;
        impl->data = std::make_shared<Buffer>((size_t)r * c);
    }

    static Tensor zeros(int r, int c, bool rg = false) { return Tensor(r, c, rg); }
    static Tensor from_vector(int r, int c, const std::vector<float>& v, bool rg = false) {
        if (v.size() != (size_t)r * c) throw std::runtime_error("from_vector: tamanho incompatível");
        Tensor t(r, c, rg); backend().sync();
        std::copy(v.begin(), v.end(), t.p());
        return t;
    }
    static Tensor randn(int r, int c, float stddev, std::mt19937& rng, bool rg = false) {
        Tensor t(r, c, rg); backend().sync();
        std::normal_distribution<float> d(0.f, stddev);
        for (size_t i = 0; i < t.numel(); ++i) t.p()[i] = d(rng);
        return t;
    }

    bool defined() const { return (bool)impl; }
    int rows() const { return impl->rows; }
    int cols() const { return impl->cols; }
    size_t numel() const { return (size_t)impl->rows * impl->cols; }
    bool requires_grad() const { return impl->requires_grad; }
    bool has_grad() const { return (bool)impl->grad; }

    float* p() const { return impl->data->ptr; }            // ponteiro bruto (sem sync)
    float* gp() const { ensure_grad(); return impl->grad->ptr; }

    // acesso seguro pelo host (sincroniza com o dispositivo)
    float* data() const { impl->data->be->sync(); return p(); }
    std::vector<float> to_vector() const { float* d = data(); return {d, d + numel()}; }
    std::vector<float> grad_vector() const { impl->data->be->sync(); float* g = gp(); return {g, g + numel()}; }
    float item() const { if (numel() != 1) throw std::runtime_error("item(): tensor não é escalar"); return data()[0]; }

    void zero_grad() const { if (has_grad()) impl->data->be->fill(impl->grad->ptr, numel(), 0.f); }

    // Torna este tensor o resultado de uma op diferenciável.
    void attach(std::vector<Tensor> parents, std::function<void(TensorImpl&)> fn) {
        impl->requires_grad = true;
        for (auto& q : parents) impl->parents.push_back(q.impl);
        impl->backward_fn = std::move(fn);
    }

    // Backpropagation a partir de um escalar.
    void backward() const {
        if (numel() != 1) throw std::runtime_error("backward(): só para escalares");
        std::vector<TensorImpl*> order; std::unordered_set<TensorImpl*> seen;
        std::function<void(TensorImpl*)> dfs = [&](TensorImpl* n) {
            if (!seen.insert(n).second) return;
            for (auto& q : n->parents) dfs(q.get());
            order.push_back(n);  // ordem topológica: pais antes dos filhos
        };
        dfs(impl.get());
        impl->data->be->fill(gp(), 1, 1.f);  // dL/dL = 1
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            TensorImpl* n = *it;
            if (!n->backward_fn) continue;
            for (auto& q : n->parents)
                if (q->requires_grad && !q->grad) q->grad = std::make_shared<Buffer>((size_t)q->rows * q->cols, q->data->be);
            n->backward_fn(*n);
        }
    }

private:
    void ensure_grad() const {
        if (!impl->grad) impl->grad = std::make_shared<Buffer>(numel(), impl->data->be);
    }
};

// ---------------------------------------------------------------- operações

inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.cols() != b.rows()) throw std::runtime_error("matmul: dimensões incompatíveis");
    int M = a.rows(), K = a.cols(), N = b.cols();
    Tensor out(M, N);
    backend().matmul(a.p(), b.p(), out.p(), M, K, N, false, false, false);
    if (a.requires_grad() || b.requires_grad())
        out.attach({a, b}, [a, b, M, K, N](TensorImpl& o) {
            Backend& be = *o.data->be;
            if (a.requires_grad()) be.matmul(o.grad->ptr, b.p(), a.gp(), M, N, K, false, true, true);  // dA += dC * B^T
            if (b.requires_grad()) be.matmul(a.p(), o.grad->ptr, b.gp(), K, M, N, true, false, true);  // dB += A^T * dC
        });
    return out;
}

// y = x + b  (b: 1 x cols, broadcast nas linhas)
inline Tensor add_bias(const Tensor& x, const Tensor& b) {
    if (b.rows() != 1 || b.cols() != x.cols()) throw std::runtime_error("add_bias: bias deve ser 1 x cols");
    Tensor out(x.rows(), x.cols());
    backend().add_bias(x.p(), b.p(), out.p(), x.rows(), x.cols());
    if (x.requires_grad() || b.requires_grad())
        out.attach({x, b}, [x, b](TensorImpl& o) {
            Backend& be = *o.data->be;
            if (x.requires_grad()) be.axpy(x.gp(), o.grad->ptr, x.numel(), 1.f);
            if (b.requires_grad()) be.col_sum(o.grad->ptr, b.gp(), x.rows(), x.cols());
        });
    return out;
}

inline Tensor relu(const Tensor& x) {
    Tensor out(x.rows(), x.cols());
    backend().relu(x.p(), out.p(), x.numel());
    if (x.requires_grad())
        out.attach({x}, [x](TensorImpl& o) { o.data->be->relu_bwd(x.p(), o.grad->ptr, x.gp(), x.numel()); });
    return out;
}

// softmax por linha (estável numericamente)
inline Tensor softmax(const Tensor& x) {
    Tensor out(x.rows(), x.cols());
    backend().softmax(x.p(), out.p(), x.rows(), x.cols());
    if (x.requires_grad())
        out.attach({x}, [x](TensorImpl& o) {
            o.data->be->softmax_bwd(o.data->ptr, o.grad->ptr, x.gp(), x.rows(), x.cols());
        });
    return out;
}

// Cross-entropy média: pred = probabilidades (N x C), target = one-hot (N x C)
inline Tensor cross_entropy(const Tensor& pred, const Tensor& target) {
    if (pred.rows() != target.rows() || pred.cols() != target.cols()) throw std::runtime_error("cross_entropy: shapes diferentes");
    Tensor out(1, 1); int n = pred.rows();
    float v = (float)(backend().ce_loss(pred.p(), target.p(), pred.numel()) / n);
    backend().sync(); out.p()[0] = v;
    if (pred.requires_grad())
        out.attach({pred}, [pred, target, n](TensorImpl& o) {
            Backend& be = *o.data->be; be.sync();
            be.ce_bwd(pred.p(), target.p(), pred.gp(), pred.numel(), o.grad->ptr[0] / n);
        });
    return out;
}

// Erro quadrático médio sobre todos os elementos
inline Tensor mse(const Tensor& pred, const Tensor& target) {
    if (pred.rows() != target.rows() || pred.cols() != target.cols()) throw std::runtime_error("mse: shapes diferentes");
    Tensor out(1, 1); size_t n = pred.numel();
    float v = (float)(backend().mse_loss(pred.p(), target.p(), n) / n);
    backend().sync(); out.p()[0] = v;
    if (pred.requires_grad())
        out.attach({pred}, [pred, target, n](TensorImpl& o) {
            Backend& be = *o.data->be; be.sync();
            be.mse_bwd(pred.p(), target.p(), pred.gp(), n, o.grad->ptr[0] / n);
        });
    return out;
}

}  // namespace mnn
