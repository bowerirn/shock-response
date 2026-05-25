import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if sys.platform == "win32":
    cxx_flags = ["/O2", "/Zc:preprocessor"]
    nvcc_flags = ["-O3", "-Xcompiler", "/Zc:preprocessor"]
else:
    cxx_flags = ["-O2"]
    nvcc_flags = ["-O3"]

setup(
    name="srs-cuda",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="srs_ext",
            sources=[
                "csrc/bindings.cpp",
                "csrc/srs_forward.cu",
                "csrc/srs_backward.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)