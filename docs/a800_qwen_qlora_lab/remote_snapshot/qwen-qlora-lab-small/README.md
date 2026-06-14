# Qwen QLoRA Lab on A800

This project is generated for low-risk QLoRA learning on the shared A800 server.

Main commands on A800:

```bash
cd /root/zhl/qwen-qlora-lab
bash scripts/build_image.sh
bash scripts/start_pipeline.sh
bash scripts/status.sh
```

The pipeline uses ModelScope first, Hugging Face as fallback, and writes logs under `logs/`.