#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:?usage: install_validation_deps.sh /path/to/python}"
uv_bin="${UV_BIN:-/home/ubuntu/.local/bin/uv}"

# PyIQA 0.1.16 now declares transformers>=5.0 in its package metadata, but
# Difix is pinned to transformers==4.38.0. None of the six restoration metrics
# uses Transformers, so install PyIQA without its broad tooling dependencies
# and then install only the runtime packages used by those metrics.
"$uv_bin" pip install \
  --python "$python_bin" \
  --no-deps \
  pyiqa==0.1.16

"$uv_bin" pip install \
  --python "$python_bin" \
  clearml==2.1.10 \
  addict==2.4.0 \
  facexlib==0.3.0 \
  opencv-python-headless==5.0.0.93 \
  timm==1.0.28

"$python_bin" - <<'PY'
import clearml
import cv2
import diffusers
import pyiqa
import timm
import torch
import torchvision
import transformers

print("clearml", clearml.__version__)
print("pyiqa", pyiqa.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("timm", timm.__version__)
print("cv2", cv2.__version__)
PY
