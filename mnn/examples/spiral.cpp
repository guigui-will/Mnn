// Treina um MLP 2-64-64-3 na espiral. Uso: ./spiral [cpu|cuda]
#include "mnn/nn.hpp"
#include "datasets.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string be = argc > 1 ? argv[1] : "cpu";
    if (!mnn::use_backend(be)) { std::fprintf(stderr, "backend '%s' indisponível neste build\n", be.c_str()); return 1; }
    std::printf("backend: %s\n", mnn::backend().name());

    std::mt19937 rng(42);
    Dataset d = make_spiral(100, 3, rng);

    mnn::Model model;
    model.add<mnn::Linear>(2, 64, rng).add<mnn::ReLU>()
         .add<mnn::Linear>(64, 64, rng).add<mnn::ReLU>()
         .add<mnn::Linear>(64, 3, rng).add<mnn::Softmax>();
    model.summary();

    mnn::Adam opt(model.parameters(), 0.01f);
    mnn::CrossEntropyLoss loss_fn;
    for (int epoch = 0; epoch <= 300; ++epoch) {
        opt.zero_grad();
        mnn::Tensor probs = model.forward(d.x);   // forward
        mnn::Tensor loss = loss_fn(probs, d.y);   // loss
        loss.backward();                          // backward (autodiff)
        opt.step();                               // atualiza pesos
        if (epoch % 50 == 0)
            std::printf("epoch %3d  loss %.4f  acc %.1f%%\n", epoch, loss.item(), 100 * accuracy(probs, d.labels));
    }
}
