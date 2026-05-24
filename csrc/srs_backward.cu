#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include "srs.h"


__global__ void srs_grad_kernel(
    const double* grad_S, // (BF)
    const int* argmax, // (BF)
    const int* sign, // (BF)
    const double* As, // (2) or (2F)
    const double* Bs, // (3) or (3F)
    double* grad_X, // (BT) zeros
    int B,
    int F,
    int T,
    int a_stride,
    int b_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * F) return;

    int batch = idx / F;
    int filt  = idx % F;

    const double* a = As + filt * a_stride;
    const double* b = Bs + filt * b_stride;

    const double a1 = a[0], a2 = a[1];
    const double b0 = b[0], b1 = b[1], b2 = b[2];

    const int k = argmax[idx];
    double dLdy_i = grad_S[idx] * static_cast<double>(sign[idx]);

    int out_idx = batch * T + k;
    atomicAdd(&grad_X[out_idx], b0 * dLdy_i);
    if (k - 1 >= 0) atomicAdd(&grad_X[out_idx - 1], b1 * dLdy_i);
    if (k - 2 >= 0) atomicAdd(&grad_X[out_idx - 2], b2 * dLdy_i);

    double dLdy_ip1 = dLdy_i;
    double dLdy_ip2 = 0.0;

    // SRS grad doesn't contribute until the argmax, and only at the argmax
    // We can thus manually start at the argmax, then simplify the loop after
    for (int i = k - 1; i >= 0; i--) {
        dLdy_i = -a1 * dLdy_ip1 - a2 * dLdy_ip2;

        out_idx = batch * T + i;
        atomicAdd(&grad_X[out_idx], b0 * dLdy_i);
        if (i - 1 >= 0) atomicAdd(&grad_X[out_idx -1], b1 * dLdy_i);
        if (i - 2 >= 0) atomicAdd(&grad_X[out_idx - 2], b2 * dLdy_i);

        dLdy_ip2 = dLdy_ip1;
        dLdy_ip1 = dLdy_i;
    }
}




at::Tensor iir_srs_backward(
    at::Tensor grad_S, // Shape: (B, F)
    at::Tensor argmax, // Shape: (B, F)
    at::Tensor sign, // Shape: (B, F)
    at::Tensor As, // Shape: (2,) or (1, 2) or (F, 2)
    at::Tensor Bs, // Shape: (3,) or (1, 3) or (F, 3)
    int T
) {
    TORCH_CHECK(grad_S.is_cuda(), "grad_S must be CUDA");
    TORCH_CHECK(argmax.is_cuda(), "argmax must be CUDA");
    TORCH_CHECK(sign.is_cuda(), "sign must be CUDA");
    TORCH_CHECK(As.is_cuda(), "As must be CUDA");
    TORCH_CHECK(Bs.is_cuda(), "Bs must be CUDA");

    TORCH_CHECK(grad_S.dtype() == at::kDouble, "grad_S must be float64");
    TORCH_CHECK(argmax.dtype() == at::kInt, "argmax must be int32");
    TORCH_CHECK(sign.dtype() == at::kInt, "sign must be int32");
    TORCH_CHECK(As.dtype() == at::kDouble, "As must be float64");
    TORCH_CHECK(Bs.dtype() == at::kDouble, "Bs must be float64");

    TORCH_CHECK(grad_S.is_contiguous(), "grad_S must be contiguous");
    TORCH_CHECK(argmax.is_contiguous(), "argmax must be contiguous");
    TORCH_CHECK(sign.is_contiguous(), "sign must be contiguous");
    TORCH_CHECK(As.is_contiguous(), "As must be contiguous");
    TORCH_CHECK(Bs.is_contiguous(), "Bs must be contiguous");

    TORCH_CHECK(grad_S.dim() == 2, "grad_S must have shape (B, F)");
    TORCH_CHECK(argmax.dim() == 2, "argmax must have shape (B, F)");
    TORCH_CHECK(sign.dim() == 2, "sign must have shape (B, F)");

    TORCH_CHECK(argmax.sizes() == grad_S.sizes(), "argmax must have same shape as grad_S");
    TORCH_CHECK(sign.sizes() == grad_S.sizes(), "sign must have same shape as grad_S");

    const int B = static_cast<int>(grad_S.size(0));
    int F_from_grad = static_cast<int>(grad_S.size(1));

    TORCH_CHECK(T > 0, "T must be positive");

    int F, a_stride, b_stride;

    if (As.dim() == 1) {
        TORCH_CHECK(As.size(0) == 2, "As with dim 1 must have shape (2,)");
        TORCH_CHECK(Bs.dim() == 1 && Bs.size(0) == 3, "Bs must have shape (3,) when As has shape (2,)");
        F = 1;
        a_stride = 0;
        b_stride = 0;
    } else if (As.dim() == 2 && As.size(0) == 1 && As.size(1) == 2) {
        TORCH_CHECK(Bs.dim() == 2 && Bs.size(0) == 1 && Bs.size(1) == 3,
                    "Bs must have shape (1, 3) when As has shape (1, 2)");
        F = 1;
        a_stride = 0;
        b_stride = 0;
    } else if (As.dim() == 2 && As.size(1) == 2) {
        TORCH_CHECK(Bs.dim() == 2 && Bs.size(1) == 3, "Bs must have shape (F, 3)");
        TORCH_CHECK(As.size(0) == Bs.size(0), "As and Bs must have same number of filters");
        F = static_cast<int>(As.size(0));
        a_stride = 2;
        b_stride = 3;
    } else {
        TORCH_CHECK(false, "As must have shape (2,), (1, 2), or (F, 2)");
    }

    TORCH_CHECK(F_from_grad == F,
                "grad_S shape (B, F) does not match filter bank size inferred from As/Bs");

    // Compiler doesn't like these checks ¯\_(ツ)_/¯
    // TORCH_CHECK(at::all(argmax >= 0).item<bool>(), "argmax must be nonnegative");
    // TORCH_CHECK(at::all(argmax < T).item<bool>(), "argmax entries must be < T");


    auto grad_X = at::zeros(
        {B, T},
        grad_S.options()
    );

    int threads = 256;
    int blocks = (B * F + threads - 1) / threads;

    srs_grad_kernel<<<blocks, threads>>>(
        grad_S.data_ptr<double>(),
        argmax.data_ptr<int>(),
        sign.data_ptr<int>(),
        As.data_ptr<double>(),
        Bs.data_ptr<double>(),
        grad_X.data_ptr<double>(),
        B, F, T, a_stride, b_stride
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "iso_srs_backward launch failed: ", cudaGetErrorString(err));

    return grad_X;
}

