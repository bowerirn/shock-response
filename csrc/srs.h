#pragma once

#include <ATen/core/TensorBody.h>
#include <vector>

// Forward
std::vector<at::Tensor> iir_srs_forward(
    at::Tensor X,
    at::Tensor As,
    at::Tensor Bs
);

// Backward
at::Tensor iir_srs_backward(
    at::Tensor grad_S,
    at::Tensor argmax,
    at::Tensor sign,
    at::Tensor As,
    at::Tensor Bs,
    int T
);


