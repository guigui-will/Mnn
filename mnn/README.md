# mnn — mini framework de redes neurais em C++17

Tensor 2D + autodiff reverso + camadas + otimizadores, com backend CPU e CUDA.

    Tensor → matmul → forward → loss → backward (autodiff) → gradientes → optimizer

| Peça | Onde |
|---|---|
| Tensor, grafo dinâmico, `backward()`, ops (matmul, add_bias, relu, softmax, cross_entropy, mse) | `include/mnn/tensor.hpp` |
| Linear, ReLU, Softmax, losses, SGD(+momentum), Adam, Model | `include/mnn/nn.hpp` |
| Interface de backend | `include/mnn/backend.hpp` |
| CPU / CUDA | `src/cpu_backend.cpp` / `src/cuda_backend.cu` |

## Uso
    make && ./build/test_all cpu && ./build/spiral cpu
    make CUDA=1 && ./build/test_all cuda     # precisa de nvcc + GPU
    # ou CMake: cmake -B b -DMNN_CUDA=ON && cmake --build b

    mnn::Model m;
    m.add<mnn::Linear>(2,64,rng).add<mnn::ReLU>().add<mnn::Linear>(64,3,rng).add<mnn::Softmax>();
    mnn::Adam opt(m.parameters(), 0.01f);
    opt.zero_grad();
    auto loss = mnn::CrossEntropyLoss()(m.forward(x), y);
    loss.backward(); opt.step();

## Como o autodiff funciona
Cada op cria um tensor-resultado que guarda seus pais e uma closure de backward.
`loss.backward()` ordena o grafo topologicamente e percorre ao contrário, acumulando
dL/dpai += (regra da cadeia local). Ex.: C=A·B ⇒ dA += dC·Bᵀ, dB += Aᵀ·dC.

## Limitações conhecidas
Tensores só 2D; sem batching de dataset (full-batch); escolha o backend antes de criar tensores;
CUDA usa memória unificada e kernels simples (matmul com tiles 16x16, sem cuBLAS).
