#pragma once
#include "mnn/tensor.hpp"
#include <cmath>
#include <random>

// Espiral clássica (cs231n): K classes, N pontos por classe, 2 features.
struct Dataset { mnn::Tensor x, y; std::vector<int> labels; };

inline Dataset make_spiral(int N, int K, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.f, 0.2f);
    std::vector<float> X, Y((size_t)N * K * K, 0.f); std::vector<int> lab;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < N; ++i) {
            float r = (float)i / N, t = k * 4.f + 4.f * i / N + noise(rng);
            X.push_back(r * std::sin(t)); X.push_back(r * std::cos(t));
            Y[((size_t)k * N + i) * K + k] = 1.f; lab.push_back(k);
        }
    return {mnn::Tensor::from_vector(N * K, 2, X), mnn::Tensor::from_vector(N * K, K, Y), lab};
}

inline float accuracy(const mnn::Tensor& probs, const std::vector<int>& labels) {
    auto p = probs.to_vector(); int C = probs.cols(), ok = 0;
    for (int i = 0; i < probs.rows(); ++i) {
        int best = 0; for (int j = 1; j < C; ++j) if (p[(size_t)i * C + j] > p[(size_t)i * C + best]) best = j;
        ok += best == labels[i];
    }
    return (float)ok / probs.rows();
}
