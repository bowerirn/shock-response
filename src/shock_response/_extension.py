import os
from pathlib import Path
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

ROOT = Path(__file__).resolve().parents[2]

srs_ext = load(
    name="srs_ext",
    sources=[
        str(ROOT / "csrc" / "bindings.cpp"),
        str(ROOT / "csrc" / "srs_forward.cu"),
        str(ROOT / "csrc" / "srs_backward.cu"),
    ],
    extra_cflags=["/O2", "/Zc:preprocessor"],
    extra_cuda_cflags=["-O3", "-Xcompiler", "/Zc:preprocessor"],
    verbose=True,
)