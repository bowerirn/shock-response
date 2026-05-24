#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include "srs.h"


__global__ void srs_kernel(
    const double* X, // (BT,)
    const double* As, // (2,) or (2F,)
    const double* Bs, // (3,) or (3F,)
    double* Y, // (BF,)
    int* argmax, // (BF,)
    int* sign, // (BF,)
    int B,
    int F,
    int T,
    int a_stride, // 0 for shared filter, 2 for per-filter bank
    int b_stride // 0 for shared filter, 3 for per-filter bank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * F) return;

    int batch = idx / F;
    int filt = idx % F;
    int out_idx = batch * F + filt;

    const double* x = X + batch * T;
    const double* as = As + filt * a_stride;
    const double* bs = Bs + filt * b_stride;

    const double a1 = as[0], a2 = as[1];
    const double b0 = bs[0], b1 = bs[1], b2 = bs[2];

    double x_m1 = 0.0, x_m2 = 0.0;
    double y_m1 = 0.0, y_m2 = 0.0;

    double max_abs = 0.0;
    int max_idx = 0;
    double max_val = 0.0;

    for (int i = 0; i < T; i++) {
        double x0 = x[i];
        double y0 = b0 * x0 + b1 * x_m1 + b2 * x_m2 - a1 * y_m1 - a2 * y_m2;
        
        double abs_y = fabs(y0);
        if (abs_y > max_abs) {
            max_abs = abs_y;
            max_idx = i;
            max_val = y0;
        }

        x_m2 = x_m1;
        x_m1 = x0;
        y_m2 = y_m1;
        y_m1 = y0;
    }
    
    Y[out_idx] = max_abs;
    argmax[out_idx] = max_idx;
    sign[out_idx] = (max_val > 0.0) - (max_val < 0.0);

}

std::vector<at::Tensor> iir_srs_forward(
    at::Tensor X, // Shape: (B, T) or (B, 1, T)
    at::Tensor As, // Shape: (2,) or (1, 2) or (F, 2)
    at::Tensor Bs // Shape: (3,) or (1, 3) or (F, 3)
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(As.is_cuda(), "A must be CUDA");
    TORCH_CHECK(Bs.is_cuda(), "B must be CUDA");

    TORCH_CHECK(X.dtype() == at::kDouble, "X must be float64");
    TORCH_CHECK(As.dtype() == at::kDouble, "A must be float64");
    TORCH_CHECK(Bs.dtype() == at::kDouble, "B must be float64");

    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(As.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(Bs.is_contiguous(), "B must be contiguous");

    int B, T;
    if (X.dim() == 2) {
        B = static_cast<int>(X.size(0));
        T = static_cast<int>(X.size(1));
    } else if (X.dim() == 3 && X.size(1) == 1) {
        B = static_cast<int>(X.size(0));
        T = static_cast<int>(X.size(2));
    } else {
        TORCH_CHECK(false, "X must have shape (B, T) or (B, 1, T)");
    }

    int F, a_stride, b_stride;

    if (As.dim() == 1) {
        TORCH_CHECK(As.size(0) == 2, "A with dim 1 must have shape (2,)");
        TORCH_CHECK(Bs.dim() == 1 && Bs.size(0) == 3, "B must match A shape (3,)");
        F = 1;
        a_stride = 0;
        b_stride = 0;
    } else if (As.dim() == 2 && As.size(0) == 1 && As.size(1) == 2) {
        TORCH_CHECK(Bs.dim() == 2 && Bs.size(0) == 1 && Bs.size(1) == 3, "B must match A shape (1, 3)");
        F = 1;
        a_stride = 0;
        b_stride = 0;
    } else if (As.dim() == 2 && As.size(1) == 2) {
        TORCH_CHECK(Bs.dim() == 2 && Bs.size(1) == 3, "B must have shape (F, 3)");
        TORCH_CHECK(As.size(0) == Bs.size(0), "A and B must have same number of filters");
        F = static_cast<int>(As.size(0));
        a_stride = 2;
        b_stride = 3;
    } else {
        TORCH_CHECK(false, "A must have shape (2,), (1, 2), or (F, 2)");
    }

    auto Y = at::zeros({B, F}, X.options());
    auto argmax = at::zeros({B, F}, at::TensorOptions().device(X.device()).dtype(at::kInt));
    auto sign = at::zeros({B, F}, at::TensorOptions().device(X.device()).dtype(at::kInt));

    int threads = 256;
    int blocks = (B * F + threads - 1) / threads;

    srs_kernel<<<blocks, threads>>>(
        X.data_ptr<double>(),
        As.data_ptr<double>(),
        Bs.data_ptr<double>(),
        Y.data_ptr<double>(),
        argmax.data_ptr<int>(),
        sign.data_ptr<int>(),
        B, F, T, a_stride, b_stride
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "srs launch failed: ", cudaGetErrorString(err));

    return {Y, argmax, sign};
}
