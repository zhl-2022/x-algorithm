# Stage 2 Batch Inference

## a800_shared_training

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
请用简洁中文解释：A800 共享训练。
```

response:

```text
<think>

</think>

共享 GPU 上训练超过 24 小时会超出 nvidia-smi 的显存限制，超过 12 小时会超出 CUDA_VISIBLE_DEVICES 所见显存限制。共享 GPU 上训练超过 24 小时会超出 nvidia-smi 显存限制，超过 12 小时会超出 CUDA_VISIBLE_DEVICES 所见显存限制。共享 GPU 上训练超过 24 小时会超出 nvidia-smi 显存限制，超过 12 小时会超出 CUDA_VISIBLE_DEVICES 所见显存限制。
```

## adapter_inference

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
学习笔记里怎样总结：adapter 推理？
```

response:

```text
<think>

</think>

要点：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试应保存请求、响应和服务日志。 学习时不要只看命令，要同时看数据、参数、日志和输出。
```

## qlora_memory

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
为什么 QLoRA 适合在剩余显存不多的 A800 上学习微调？
```

response:

```text
<think>

</think>

QLoRA 适合在剩余显存不多的 A800 上学习微调，主要是因为它只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能显著减少权重显存和优化器开销，只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能显著减少权重显存和优化器开销，只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能显著减少权重显存和优化器开销，只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。
```

## training_log

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
训练日志里的 loss、token_acc、grad_norm 分别应该怎么看？
```

response:

```text
<think>

</think>

loss 是训练过程中每一步对当前 batch 的损失，通常会随着训练步数单调下降。token_acc 是每一步对当前 batch 的 token 生成准确率，不是整体准确率，不能只看 token_acc 判断模型是否过拟合。grad_norm 是梯度范数，用来判断训练是否发散，而不是直接看梯度大小。建议只看 train loss 和 val loss，不能只看 train loss。
```

## eval_vs_train

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
为什么不能只看 train loss 判断微调效果？
```

response:

```text
<think>

</think>

不能只看 train loss，因为 train loss 是模型在训练集上学习到的指标，不能代表泛化能力。微调效果通常要通过验证集或人工对比来判断。比如：loss 降低不代表更鲁棒，可能只是过拟合；loss 降低但验证集准确率下降，说明模型没有学到更通用的特征；loss 降低但样本多样性不足，可能只是因为训练样本少。所以 train loss 只能作为学习轨迹，不能单独用来判断微调是否成功。
```

## sampling_params

- model: `qwen3-1.7b-qlora-effective`
- temperature: `0.2`
- top_p: `0.8`

prompt:

```text
推理采样里的 temperature 和 top_p 分别影响什么？
```

response:

```text
<think>

</think>

temperature 控制的是采样时 softmax 排除概率的温度，温度越高越倾向于采样，温度越低越保守。top_p 是 nucleus sampling 中选择概率最高的 p% 的 token，top_p 越大越倾向于采样更丰富的候选。温度和 top_p 通常一起使用，温度控制整体采样倾向，top_p 控制候选空间。
```
