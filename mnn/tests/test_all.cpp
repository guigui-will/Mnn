// Testes do mini framework. Uso: ./test_all [cpu|cuda]
#include "mnn/nn.hpp"
#include "../examples/datasets.hpp"
#include <cstdio>
#include <string>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) ++g_pass; else { ++g_fail; std::printf("  FALHOU (%s:%d): ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

using namespace mnn;

// ---- 1. matmul (4 combinações de transposição) contra referência ingênua em double
static void test_matmul() {
    std::printf("[matmul] ");
    std::mt19937 rng(1); int M = 7, K = 5, N = 9;
    for (int mask = 0; mask < 4; ++mask) {
        bool tA = mask & 1, tB = mask & 2;
        Tensor A = Tensor::randn(tA ? K : M, tA ? M : K, 1, rng), B = Tensor::randn(tB ? N : K, tB ? K : N, 1, rng);
        Tensor C = Tensor::randn(M, N, 1, rng);            // testa também acumulação
        auto c0 = C.to_vector(); auto a = A.to_vector(), b = B.to_vector();
        backend().matmul(A.p(), B.p(), C.p(), M, K, N, tA, tB, true); backend().sync();
        double maxerr = 0;
        for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
            double s = c0[i * N + j];
            for (int k = 0; k < K; ++k) s += (double)(tA ? a[k * M + i] : a[i * K + k]) * (tB ? b[j * K + k] : b[k * N + j]);
            maxerr = std::max(maxerr, std::abs(s - C.data()[i * N + j]));
        }
        CHECK(maxerr < 1e-4, "matmul tA=%d tB=%d erro=%g", tA, tB, maxerr);
    }
    std::printf("ok\n");
}

// ---- 2. softmax: linhas somam 1, estável com valores grandes
static void test_softmax() {
    std::printf("[softmax] ");
    Tensor x = Tensor::from_vector(2, 3, {1, 2, 3, 1000, 1001, 1002});
    auto y = softmax(x).to_vector();
    for (int r = 0; r < 2; ++r) {
        float s = y[r * 3] + y[r * 3 + 1] + y[r * 3 + 2];
        CHECK(std::abs(s - 1.f) < 1e-5, "linha %d soma %g", r, s);
        CHECK(std::isfinite(y[r * 3]), "NaN/Inf na linha %d", r);
    }
    CHECK(std::abs(y[3] - y[0]) < 1e-5, "softmax deve ser invariante a deslocamento");
    std::printf("ok\n");
}

// ---- 3. gradient check: backprop vs diferenças finitas centrais
template <class Loss>
static void grad_check(const char* label, Model& m, const Tensor& x, const Tensor& t, Loss loss_fn) {
    std::printf("[gradcheck %s] ", label);
    auto params = m.parameters();
    for (auto& p : params) p.zero_grad();
    loss_fn(m.forward(x), t).backward();
    backend().sync();
    // eps pequeno o bastante para não atravessar o "kink" do ReLU (|pré-ativação| mínima ~8e-3 nestes dados)
    const float eps = 3e-3f; int bad = 0, total = 0; double worst = 0;
    for (auto& p : params) {
        auto g = p.grad_vector(); float* w = p.data();
        for (size_t i = 0; i < p.numel(); ++i) {
            float orig = w[i];
            w[i] = orig + eps; double lp = loss_fn(m.forward(x), t).item();
            w[i] = orig - eps; double lm = loss_fn(m.forward(x), t).item();
            w[i] = orig;
            double num = (lp - lm) / (2 * eps), err = std::abs(num - g[i]);
            double tol = 1e-3 + 2e-2 * std::max(std::abs(num), (double)std::abs(g[i]));
            worst = std::max(worst, err); ++total; bad += err > tol;
        }
    }
    CHECK(bad == 0, "%d/%d gradientes divergem (pior erro abs %g)", bad, total, worst);
    std::printf("%d parâmetros, pior erro %.2e\n", total, worst);
}

static void test_gradcheck() {
    std::mt19937 rng(7);
    Tensor x = Tensor::randn(6, 3, 1.f, rng);
    std::vector<float> oh(6 * 4, 0.f); for (int i = 0; i < 6; ++i) oh[i * 4 + i % 4] = 1.f;
    Tensor t = Tensor::from_vector(6, 4, oh);
    { Model m; m.add<Linear>(3, 5, rng).add<ReLU>().add<Linear>(5, 4, rng).add<Softmax>();
      grad_check("MLP+Softmax+CE", m, x, t, CrossEntropyLoss()); }
    { Model m; m.add<Linear>(3, 5, rng).add<ReLU>().add<Linear>(5, 4, rng);
      Tensor tr = Tensor::randn(6, 4, 1.f, rng);
      grad_check("MLP+MSE", m, x, tr, MSELoss()); }
}

// ---- 4. treino: regressão linear com SGD+momentum (MSE)
static void test_linear_regression() {
    std::printf("[treino: y=2x+1, SGD] ");
    std::mt19937 rng(3);
    std::vector<float> xs, ys; for (int i = 0; i < 32; ++i) { float v = -1 + 2.f * i / 31; xs.push_back(v); ys.push_back(2 * v + 1); }
    Tensor x = Tensor::from_vector(32, 1, xs), y = Tensor::from_vector(32, 1, ys);
    auto lin = std::make_shared<Linear>(1, 1, rng); SGD opt(lin->parameters(), 0.1f, 0.9f);
    for (int e = 0; e < 200; ++e) { opt.zero_grad(); mse(lin->forward(x), y).backward(); opt.step(); }
    float w = lin->W.data()[0], b = lin->b.data()[0];
    CHECK(std::abs(w - 2) < 1e-2 && std::abs(b - 1) < 1e-2, "w=%g b=%g (esperado 2 e 1)", w, b);
    std::printf("w=%.4f b=%.4f\n", w, b);
}

// ---- 5. treino: XOR (não-linear -> precisa de camada oculta + ReLU)
static void test_xor() {
    std::printf("[treino: XOR, Adam] ");
    std::mt19937 rng(5);
    Tensor x = Tensor::from_vector(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y = Tensor::from_vector(4, 2, {1, 0, 0, 1, 0, 1, 1, 0});
    Model m; m.add<Linear>(2, 8, rng).add<ReLU>().add<Linear>(8, 2, rng).add<Softmax>();
    Adam opt(m.parameters(), 0.05f); float last = 0;
    for (int e = 0; e < 300; ++e) { opt.zero_grad(); Tensor l = CrossEntropyLoss()(m.forward(x), y); l.backward(); opt.step(); last = l.item(); }
    float acc = accuracy(m.forward(x), {0, 1, 1, 0});
    CHECK(acc == 1.f && last < 0.05f, "acc=%g loss=%g", acc, last);
    std::printf("loss=%.4f acc=%.0f%%\n", last, 100 * acc);
}

// ---- 6. treino: espiral 3 classes, validada num conjunto de teste separado
static void test_spiral() {
    std::printf("[treino: espiral, Adam] ");
    std::mt19937 rng(42);
    Dataset tr = make_spiral(100, 3, rng), te = make_spiral(100, 3, rng);
    Model m; m.add<Linear>(2, 64, rng).add<ReLU>().add<Linear>(64, 64, rng).add<ReLU>().add<Linear>(64, 3, rng).add<Softmax>();
    Adam opt(m.parameters(), 0.01f); float first = 0, last = 0;
    for (int e = 0; e < 300; ++e) {
        opt.zero_grad(); Tensor l = CrossEntropyLoss()(m.forward(tr.x), tr.y); l.backward(); opt.step();
        last = l.item(); if (e == 0) first = last;
    }
    float acc_tr = accuracy(m.forward(tr.x), tr.labels), acc_te = accuracy(m.forward(te.x), te.labels);
    CHECK(last < first * 0.3f, "loss não caiu: %g -> %g", first, last);
    CHECK(acc_tr > 0.95f && acc_te > 0.90f, "acc treino=%g teste=%g", acc_tr, acc_te);
    std::printf("loss %.3f->%.3f, acc treino %.1f%% teste %.1f%%\n", first, last, 100 * acc_tr, 100 * acc_te);
}

int main(int argc, char** argv) {
    std::string be = argc > 1 ? argv[1] : "cpu";
    if (!use_backend(be)) { std::printf("backend '%s' indisponível neste build\n", be.c_str()); return 2; }
    std::printf("=== mnn tests (backend: %s) ===\n", backend().name());
    test_matmul(); test_softmax(); test_gradcheck();
    test_linear_regression(); test_xor(); test_spiral();
    std::printf("=== %d checks ok, %d falhas ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
