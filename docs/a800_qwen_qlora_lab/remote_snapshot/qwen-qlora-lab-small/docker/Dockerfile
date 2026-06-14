FROM hiyouga/llamafactory:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    HF_HOME=/root/.cache/huggingface \
    MODELSCOPE_CACHE=/root/.cache/modelscope \
    PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# Qwen3.5 requires Transformers 5.x. The base image has torch 2.6.0+cu124,
# which lacks torch.float8_e8m0fnu imported by Transformers 5.x. sitecustomize.py
# below aliases the symbol before Transformers imports it. This is a local
# compatibility shim for learning/smoke training, not a production runtime patch.
RUN python -m pip install -U -i https://mirrors.aliyun.com/pypi/simple/ \
      ms-swift==4.3.0 transformers==5.10.2 accelerate==1.14.0 peft==0.19.1 \
      bitsandbytes qwen-vl-utils modelscope huggingface_hub==1.19.0 datasets \
      pillow matplotlib sentencepiece tiktoken protobuf einops trl==0.29.1 decord && \
    python - <<'PY'
from pathlib import Path
site = Path('/opt/conda/lib/python3.11/site-packages/sitecustomize.py')
site.write_text('''\ntry:\n    import torch\n    if not hasattr(torch, "float8_e8m0fnu"):\n        if hasattr(torch, "float8_e4m3fn"):\n            torch.float8_e8m0fnu = torch.float8_e4m3fn\n        elif hasattr(torch, "float8_e5m2"):\n            torch.float8_e8m0fnu = torch.float8_e5m2\nexcept Exception:\n    pass\n''', encoding='utf-8')
print(site)
PY

WORKDIR /workspace/qwen-qlora-lab
CMD ["/bin/bash"]