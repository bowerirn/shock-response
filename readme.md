# shock-response

CUDA-accelerated Shock Response Spectrum (SRS) computation for PyTorch.

This package provides a custom CUDA extension for computing SRS values and gradients with respect to the input signal. It is designed for use in PyTorch workflows, including loss functions and differentiable signal processing pipelines.

---

## Installation (as a Git submodule)

From your main project:

```bash
git submodule add git@github.com:bowerirn/shock-response.git external/shock-response
git submodule update --init --recursive
cd external/shock-response
```

Then install into your Python environment:

### Windows Installation

1. Open an x64 Native Tools Command Prompt for Visual Studio terminal
2. Activate conda for VS Native Tools
```bat
<path>\<to>\anaconda3\Scripts\activate.bat
```
3. Activate your conda environment
```bat
conda activate <env-name>
```
4. Set the following variables:
```bat
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<your-version>"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
```
5. Install the module
```bat
python -m pip install -e . --no-build-isolation
```

### Linux Installation

1. Activate your conda environment
```bash
conda activate <env-name>
```
2. Set the following variables:
```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
```
3. Install the module
```bash
python -m pip install -e . --no-build-isolation
```


## Requirements

You need:

* Python (3.9–3.12 recommended)  
* CUDA-enabled PyTorch  
* CUDA Toolkit (nvcc)
* A compatible C++ compiler:
    * Windows: Microsoft Visual Studio Build Tools (MSVC)
    * Linux: gcc / g++

> PyTorch, CUDA Toolkit, GPU driver, and Visual Studio (for Windows) must be compatible.

I used the following for Windows:
- Python 3.10
- PyTorch 2.11.0+cu128
- CUDA 12.8
- Visual Studio Build Tools 2022


## Usage

```python
import torch
from shock_response.srs import CudaSRS

device = "cuda"
dtype = torch.float64

fs = 32000
damping = 0.03
freqs = torch.tensor([50.0, 100.0, 200.0, 500.0], device=device, dtype=dtype)

x = torch.randn(8, 8192, device=device, dtype=dtype)

srs_fn = CudaSRS(freqs, damping=damping, fs=fs).to(device)

y = srs_fn(x)

print(y.shape)  # (B, F)
```

## Autograd

Gradients are implemented with respect to the input signal:

```python
x = torch.randn(8, 8192, device="cuda", dtype=torch.float64, requires_grad=True)

srs_fn = CudaSRS(freqs).cuda()
y = srs_fn(x)

loss = y.sum()
loss.backward()

print(x.grad.shape)
```

Gradients with respect to filter coefficients are not implemented.

## Notes
* The CUDA extension is compiled during installation, not at import time
* After installation, users do not need a compiler environment to run the code
* A compiler is only required when:
    * installing
    * rebuilding
    * modifying CUDA/C++ source

For Windows, this means you only need the VS Native Tools terminal when installing or rebuilding the module.

## Troubleshooting
### `CUDA_HOME environment variable is not set`

Set your CUDA path:

**Windows**

```bat
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<your-version>"
```

**Linux**

```bash
export CUDA_HOME=/usr/local/cuda
```

___

### Windows: `DISTUTILS_USE_SDK is not set`

If using a Visual Studio dev prompt:

```bat
set DISTUTILS_USE_SDK=1
```

___

### Windows: `cl.exe not found`

You must install from a:

```bat
x64 Native Tools Command Prompt for Visual Studio
```
