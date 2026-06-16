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

简洁回答：共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程和空闲空间；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
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

实践中要点是：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试应保存请求、响应和服务日志。 学习时不要只看命令，要同时看数据、参数、日志和输出。
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

简洁回答：QLoRA 会用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter；这样能显著减少权重显存和优化器开销；适合共享 GPU 或单卡学习环境。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
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

建议：训练日志至少要记录训练命令、loss 曲线、学习率、checkpoint 路径和显存变化；没有验证集时不能只靠 train loss 判断效果；应固定测试 prompt 做对比。 这条规则用于保证实验可复现、低风险。
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

简洁回答：不能只靠 train loss 判断微调效果，原因有三点：一是 train loss 会随训练步数单调上升；二是 loss 与模型性能没有直接因果关系；三是 高质量数据、充足训练步数和良好参数才能保证实验可复现。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
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

可以这样理解：temperature 控制采样多样性，值越小越专注，越大越开放；top_p 是基于概率排名的约束采样，能减少小数据过拟合；n_tokens 是单条样本最多同时采样的 token 数。 面试时要强调资源约束、参数选择和验证闭环。
```
