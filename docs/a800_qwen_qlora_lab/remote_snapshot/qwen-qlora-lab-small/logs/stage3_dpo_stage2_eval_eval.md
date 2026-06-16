# Stage 2 Keyword Evaluation

- cases: `6`
- passed: `5`
- average_score: `0.8333`

| id | score | passed | missed | forbidden |
|---|---:|---:|---|---|
| `a800_shared_training` | `1.0` | `True` |  |  |
| `adapter_inference` | `1.0` | `True` |  |  |
| `qlora_memory` | `1.0` | `True` |  |  |
| `training_log` | `0.75` | `True` | 过拟合 / 验证集 / 泛化 |  |
| `eval_vs_train` | `0.5` | `False` | 验证集 / eval<br>固定评测 / 人工评估 / benchmark |  |
| `sampling_params` | `0.75` | `True` | 稳定 / 随机 |  |

## a800_shared_training

note: 检查模型是否学到共享 GPU 训练的安全流程。

prompt:

```text
请用简洁中文解释：A800 共享训练。
```

response:

```text
<think>

</think>

建议：共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程、空闲空间和显存利用率；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。 这条规则用于保证实验可复现、低风险。
```

## adapter_inference

note: 检查模型是否能讲清 adapter 推理不是重新训练。

prompt:

```text
学习笔记里怎样总结：adapter 推理？
```

response:

```text
<think>

</think>

可以这样理解：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试应保存请求、响应和服务日志。 面试时要强调资源约束、参数选择和验证闭环。
```

## qlora_memory

note: 检查 QLoRA 的核心动机。

prompt:

```text
为什么 QLoRA 适合在剩余显存不多的 A800 上学习微调？
```

response:

```text
<think>

</think>

QLoRA 会用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter；这样能显著减少权重显存和优化器开销。因此，QLoRA 更适合共享 GPU 或单卡学习环境。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
```

## training_log

note: 检查训练日志指标理解。

prompt:

```text
训练日志里的 loss、token_acc、grad_norm 分别应该怎么看？
```

response:

```text
<think>

</think>

loss：训练过程中每条样本都只计算 loss，没有做归一化、截断、padding 等操作；loss 会随着训练步数增加而减小；loss 通常只在同一 batch 内做对比，而不是每条样本单独做实验。  
token_acc：每条样本只计算 token level 的准确率；如果只在同一 batch 内做对比，不能只看一条样本；每条样本至少要同时有 token、attention mask 和 label。  
grad_norm：grad norm 代表训练步数、batch size 和权重衰减共同作用下的梯度大小；grad norm 越大，说明权重更稀疏、学习率更高、batch size 更大或更少。
```

## eval_vs_train

note: 检查评测意识。

prompt:

```text
为什么不能只看 train loss 判断微调效果？
```

response:

```text
<think>

</think>

不能只看 train loss 判断微调效果，原因有以下几点：

1. **train loss 会先下降后上升**：微调时 train loss 通常先下降（训练更充分），但会先出现下降，再出现明显上升，这可能是因为过拟合、学习率太低或训练步数不够。这种现象不能只靠 train loss 判断效果。

2. **train loss 不能单独反映最终效果**：train loss 会先下降后上升，但最终测试 loss 会更直观地反映模型性能。如果 train loss 下降但 test loss 上升，说明微调效果不明显；如果 test loss 也下降，才说明微调有效。

3. **train loss 不能单独反映微调步数**：train loss 会先下降后上升，但不能单独用来判断微调步数。微调步数通常由学习率、batch size 和训练资源决定，而不是只靠 train loss 判断。

4. **train loss 不能单独反映微调参数选择**：train loss 只能反映训练步数、学习率、batch size 和参数选择，不能单独用来判断参数选择是否合理。微调效果
```

## sampling_params

note: 检查采样参数理解。

prompt:

```text
推理采样里的 temperature 和 top_p 分别影响什么？
```

response:

```text
<think>

</think>

温度（temperature）控制的是采样时输出的多样性。温度越低，输出越保守；温度越高，输出越开放。温度参数通常在 0.1 到 0.8 之间，用来平衡模型表达能力和推理速度。

top_p 代表 softmax 前 1/p 个概率最高的 token 会被采样。top_p 会先对 token 概率排序，再取前 p 个 token 采样。top_p 通常用于更保守地生成高质量样本，适合实验阶段。
```
