// Camadas, losses, otimizadores e Model (Sequential).
#pragma once
#include "tensor.hpp"
#include <cmath>
#include <iostream>
#include <string>

namespace mnn {

// ------------------------------------------------------------------ camadas
struct Module {
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor> parameters() { return {}; }
    virtual std::string name() const = 0;
};

class Linear : public Module {
public:
    Tensor W, b;  // W: in x out, b: 1 x out
    int in, out;
    Linear(int in_, int out_, std::mt19937& rng) : in(in_), out(out_) {
        W = Tensor::randn(in, out, std::sqrt(2.f / in), rng, true);  // init He
        b = Tensor::zeros(1, out, true);
    }
    Tensor forward(const Tensor& x) override { return add_bias(matmul(x, W), b); }
    std::vector<Tensor> parameters() override { return {W, b}; }
    std::string name() const override { return "Linear(" + std::to_string(in) + "->" + std::to_string(out) + ")"; }
};

struct ReLU : Module {
    Tensor forward(const Tensor& x) override { return relu(x); }
    std::string name() const override { return "ReLU"; }
};

struct Softmax : Module {
    Tensor forward(const Tensor& x) override { return softmax(x); }
    std::string name() const override { return "Softmax"; }
};

// ------------------------------------------------------------------- losses
struct CrossEntropyLoss { Tensor operator()(const Tensor& p, const Tensor& t) const { return cross_entropy(p, t); } };
struct MSELoss          { Tensor operator()(const Tensor& p, const Tensor& t) const { return mse(p, t); } };

// -------------------------------------------------------------- otimizadores
class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;
    void zero_grad() { for (auto& p : params_) p.zero_grad(); }
    virtual void step() = 0;
protected:
    std::vector<Tensor> params_;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, float lr, float momentum = 0.f) : Optimizer(std::move(params)), lr_(lr), mom_(momentum) {
        for (auto& p : params_) vel_.push_back(Tensor::zeros(p.rows(), p.cols()));
    }
    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            if (!params_[i].has_grad()) continue;
            backend().sgd_step(params_[i].p(), params_[i].gp(), vel_[i].p(), params_[i].numel(), lr_, mom_);
        }
    }
private:
    float lr_, mom_; std::vector<Tensor> vel_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor> params, float lr = 1e-3f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(std::move(params)), lr_(lr), b1_(b1), b2_(b2), eps_(eps) {
        for (auto& p : params_) { m_.push_back(Tensor::zeros(p.rows(), p.cols())); v_.push_back(Tensor::zeros(p.rows(), p.cols())); }
    }
    void step() override {
        ++t_;
        float c1 = 1.f - std::pow(b1_, (float)t_), c2 = 1.f - std::pow(b2_, (float)t_);
        for (size_t i = 0; i < params_.size(); ++i) {
            if (!params_[i].has_grad()) continue;
            backend().adam_step(params_[i].p(), params_[i].gp(), m_[i].p(), v_[i].p(), params_[i].numel(),
                                lr_, b1_, b2_, eps_, c1, c2);
        }
    }
private:
    float lr_, b1_, b2_, eps_; int t_ = 0; std::vector<Tensor> m_, v_;
};

// -------------------------------------------------------------------- model
class Model {
public:
    template <class L, class... Args>
    Model& add(Args&&... args) { layers_.push_back(std::make_shared<L>(std::forward<Args>(args)...)); return *this; }

    Tensor forward(const Tensor& x) {
        Tensor h = x;
        for (auto& l : layers_) h = l->forward(h);
        return h;
    }
    std::vector<Tensor> parameters() {
        std::vector<Tensor> ps;
        for (auto& l : layers_) for (auto& p : l->parameters()) ps.push_back(p);
        return ps;
    }
    void summary(std::ostream& os = std::cout) {
        size_t total = 0;
        for (auto& l : layers_) {
            size_t n = 0; for (auto& p : l->parameters()) n += p.numel();
            os << "  " << l->name() << (n ? "  [" + std::to_string(n) + " params]" : "") << "\n"; total += n;
        }
        os << "  total: " << total << " parâmetros\n";
    }
private:
    std::vector<std::shared_ptr<Module>> layers_;
};

}  // namespace mnn
