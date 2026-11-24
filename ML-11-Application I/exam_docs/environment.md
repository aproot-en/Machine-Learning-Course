# Environment Notes (PHASE 0)

## Current environment after recovery (2026-11-10)

The project uses an isolated `.venv` with Python 3.10, PyTorch 2.11.0+cu128,
NumPy 1.26.4, OpenCV 4.9.0.80, and TensorBoard 2.20.0. TensorFlow is not required
or installed in the recovered environment. Install from the project root with
`python -m pip install -r requirements.txt`; ML requirements are included from
`exam_cnn/requirements.txt`. The complete tested environment is recorded in
`requirements-full-freeze.txt`.

The sections below are historical development notes and describe the previous
Conda environment, including conflicts that do not apply to the recovered venv.

## Python / Conda
- Conda env: **`aegs`** (Python 3.10.21) — dedicated to this project only.
- The previous `tf_gpu` env (shared with other CV projects: ultralytics, labelImg,
  PyQt5, pygame) was **deleted** on 2026-09-13 at user request and replaced by `aegs`.
- Python pinned to 3.10 (not 3.11 as originally suggested in the plan) because
  TensorFlow 2.10.1 does not officially support Python 3.11.

## GPU / CUDA
- GPU: NVIDIA GeForce RTX 5060 Ti (Blackwell, compute capability 12.0 / sm_120)
- Driver: 596.49, CUDA (driver-reported): 13.2

### TensorFlow — works, but capped at 2.10.1 and JIT-slow on first GPU op
TensorFlow 2.10 is the **last** release with native-Windows GPU support; TF 2.11+
requires WSL2 for GPU on Windows (verified 2026-09-13 by inspecting the 2.21.0 Windows
wheel metadata — GPU deps only ship as `[and-cuda]` extras built for Linux manylinux,
no Windows wheels exist for them). Upgrading TensorFlow on this native-Windows env is
**not possible** without also moving to WSL2, which is out of scope (env must stay
native Windows per project decision).

TF 2.10.1 has no prebuilt kernels for compute capability 12.0, so it JIT-compiles
kernels from PTX on first use (~1 minute in testing here, not the ~30 min the warning
threatens, but real). Once compiled, subsequent calls with the same op/shape are fast.

### PyTorch — fully GPU-accelerated, no JIT
Installed: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`
via `pip install --index-url https://download.pytorch.org/whl/cu128 ...`.
Verified: `torch.cuda.get_device_capability(0)` = `(12, 0)`, `sm_120` present natively
(no JIT), Conv2D/matmul run at full speed immediately.

### ⚠️ Known issue: import order matters when using both in one process
Importing `tensorflow` **before** `torch` in the same Python process crashes torch:

    OSError: [WinError 127] The specified procedure could not be found. Error loading
    "...\\torch\\lib\\shm.dll" or one of its dependencies.

This is a native DLL conflict between the two frameworks' bundled runtime libraries.
Workaround verified: **always `import torch` before `import tensorflow`** in any script
or service that needs both. Importing only one of the two has no issue either way.
If a single process never needs both, this doesn't matter.

**Recommendation for PHASE 3+:** implement the CNN 15-layer model in PyTorch, since
that's the framework with real GPU acceleration on this machine. Keep TensorFlow
available for any existing/legacy code, but don't rely on it for training speed.

### ⚠️ 2026-09-13 update: TensorFlow is now BROKEN in this env (PHASE 6 tradeoff)
Installing `onnx`/`onnxruntime` for PHASE 6 model export pulled in `protobuf==7.36.1`
(and briefly `numpy==2.2.6`, immediately re-pinned back to 1.26.4 — see below). TF 2.10.1
requires `protobuf<3.20`, and protobuf 7's generated-code format is incompatible with
what TF 2.10.1's bundled `_pb2.py` files expect:

    TypeError: Descriptors cannot be created directly. ... your generated code is out
    of date and must be regenerated with protoc >= 3.19.0.

`import tensorflow` now fails outright. This was a deliberate tradeoff, not an
accident left unfixed: TensorFlow has no active role in this codebase (PyTorch does
all training/inference; nothing imports `tensorflow`), while `onnx`/`onnxruntime` are
required for PHASE 6's real export step. Downgrading protobuf back to <3.20 to restore
TF would very likely break onnx/onnxruntime instead (they need a modern protobuf) —
not attempted, since nothing needs both at once.
If TensorFlow is ever needed again standalone: `pip uninstall onnx onnxruntime
protobuf && pip install "protobuf<3.20"` should restore it (untested).

### ⚠️ numpy is load-bearing pinned to 1.26.4 — do not let anything upgrade it
`opencv-python==4.9.0.80` (used throughout `exam_cnn/data/preprocessing.py` and
`exam_cnn/data/cleaning.py`) is a pre-numpy-2 build and **hard-crashes on import**
under numpy 2.x (`AttributeError: _ARRAY_API not found`). `pandas`, `scikit-learn`,
`onnx`, `onnxruntime`, and `onnxscript` all resolve to numpy>=2 by default when pip
is left to its own devices — installing any of them fresh (or upgrading) can
silently drag numpy to 2.x and break the entire vision pipeline. This happened
twice in practice while building PHASE 6 (`pip install onnx onnxruntime`, then
again with `pip install onnxscript` for `torch.onnx.export`'s dynamo path) —
both times fixed by immediately re-running `pip install "numpy==1.26.4"`. After
installing anything new, always finish with that command and smoke-test
`import cv2` before trusting the environment.

### Note: `torch.onnx.export` needs `dynamo=False` on this machine
PyTorch 2.11's default ("dynamo") ONNX exporter needs the `onnxscript` package
and its verbose success message prints a unicode checkmark that crashes with
`UnicodeEncodeError` on Windows' default `cp1252` console. `exam_cnn/models/export.py`
passes `dynamo=False` to use the older, stable TorchScript-tracing exporter
instead, which has neither problem.

## Reinstalling this env from scratch
```
conda create -n aegs python=3.10 -y
conda activate aegs
pip install tensorflow==2.10.1
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128
pip install -r requirements.txt   # installs the rest; numpy is pinned to 1.26.4 last
pip install "numpy==1.26.4"       # re-pin: opencv-python/pandas/scikit-learn latest all want numpy>=2, which breaks TF 2.10.1
```
`numpy==1.26.4` and `opencv-python==4.9.0.80` are pinned specifically because their
newer releases require numpy>=2, which TensorFlow 2.10.1 is not compatible with.
