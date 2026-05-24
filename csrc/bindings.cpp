#include <torch/extension.h>
#include "srs.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &iir_srs_forward, "IIR SRS forward");
    m.def("backward", &iir_srs_backward, "IIR SRS backward");
}