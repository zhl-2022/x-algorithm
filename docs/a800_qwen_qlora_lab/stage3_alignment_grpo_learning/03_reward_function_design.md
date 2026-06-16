# 03 GRPO 奖励函数设计

GRPO 的核心不是“多训几步”，而是 reward function。奖励函数写错，模型会认真学错。

本阶段使用一个故意简单的规则奖励函数：

```text
scripts/stage3_reward_plugin.py
```

它不是生产级奖励模型，作用是让你能逐项解释：

1. 哪些回答被奖励。
2. 哪些回答被惩罚。
3. reward 设计会带来什么副作用。

## 1. GRPO 数据和 reward 的关系

GRPO 数据文件：

```text
data/stage3_grpo_prompts.jsonl
```

一条样本类似：

```json
{
  "id": "grpo_a800_safety",
  "messages": [
    {"role": "system", "content": "你是 A800 Qwen 微调学习教练..."},
    {"role": "user", "content": "给出共享 A800 上启动训练前的安全检查清单。"}
  ],
  "required": ["nvidia-smi", "显存", "CUDA_VISIBLE_DEVICES", "日志"],
  "forbidden": ["直接停止", "随便杀", "不用检查"],
  "rubric": "奖励安全检查、GPU 限定和日志意识。"
}
```

字段解释：

| 字段 | 作用 |
|---|---|
| `messages` | 给模型生成回答的 prompt |
| `required` | 希望回答里覆盖的关键概念 |
| `forbidden` | 出现后要扣分的危险表达 |
| `rubric` | 给人看的评分标准 |

`ms-swift` 的 GRPO 会把这些额外字段传给自定义 ORM，所以 reward function 可以读取 `required` 和 `forbidden`。

## 2. 奖励函数入口

核心类：

```python
class Stage3QualityORM(ORM):
    def __call__(self, completions, required=None, forbidden=None, rubric=None, **kwargs):
        ...
```

你要理解：

| 参数 | 含义 |
|---|---|
| `completions` | 模型对同一批 prompt 生成的回答 |
| `required` | 数据里传入的必备关键词 |
| `forbidden` | 数据里传入的禁止表达 |
| `rubric` | 当前脚本不直接打分，但保留给人工复盘 |
| `**kwargs` | 其它额外字段，方便以后扩展 |

注册 reward：

```python
orms["stage3_quality"] = Stage3QualityORM
```

训练时通过下面参数使用：

```bash
--external_plugins scripts/stage3_reward_plugin.py
--reward_funcs stage3_quality
```

## 3. 当前 reward 怎么打分

当前得分由 4 类组成：

| 维度 | 最高分 | 解释 |
|---|---:|---|
| required 覆盖 | `0.45` | 必备关键词出现越多，分越高 |
| 结构化表达 | `0.20` | 分点、步骤、清单更高 |
| 长度合适 | `0.15` | 太短或太长都不满分 |
| 安全/验证意识 | `0.20` | 出现风险、检查、验证、记录等词加分 |
| forbidden 惩罚 | `-0.35` | 出现危险表达扣分 |

最终分数会限制在 `[0, 1]`。

## 4. 为什么不能只奖励关键词

只奖励关键词会导致模型学会这种坏行为：

```text
nvidia-smi 显存 CUDA_VISIBLE_DEVICES 日志。
```

这句话关键词全有，但没有解释、没有步骤、没有风险边界。

所以当前 reward 还加入了：

1. 结构化表达。
2. 长度区间。
3. 安全/验证意识。
4. forbidden 惩罚。

但你也要清楚：这仍然很粗糙。

## 5. 当前 reward 的已知缺陷

| 缺陷 | 后果 |
|---|---|
| 关键词匹配很浅 | 模型可能堆关键词 |
| 不理解语义等价 | “查看 GPU 占用”不一定匹配 `nvidia-smi` |
| 不验证事实 | 错误但带关键词的回答可能拿高分 |
| 不看上下文完整性 | 回答漏掉关键步骤时未必严惩 |
| 不区分严重错误 | 小格式问题和危险操作可能都只扣固定分 |

这正是你要学习的地方：GRPO 的难点不是调用框架，而是 reward 设计和失败归因。

## 6. Reward 自检

本地同步到远端后运行：

```powershell
.\scripts\a800\run_stage3_reward_selftest.ps1
```

它会在远端执行：

```bash
python3 scripts/stage3_reward_plugin.py
```

预期输出类似：

```json
{"scores": [1.0, 0.0]}
```

你要确认：

1. 好答案得分高于坏答案。
2. forbidden 命中会扣分。
3. 没有 `ms-swift` 环境时，本地也能做基础自检。

## 7. 你必须能回答的问题

学完这份文档后，你要能回答：

1. 为什么 `required` 不能太多？
2. 为什么 `forbidden` 不能写得太泛？
3. 为什么结构化奖励可能让模型变模板化？
4. 为什么长度奖励可能鼓励废话？
5. 如果模型开始堆关键词，下一轮 reward 怎么改？
6. 如果模型回答正确但没出现指定词，应该改数据还是改 reward？

## 8. 下一轮 reward 改进方向

第一轮只做规则 reward。后续可以逐步升级：

| 版本 | 改法 |
|---|---|
| v1 | 当前关键词 + 格式 + 安全规则 |
| v2 | 增加语义同义词，例如 `GPU 占用` 等价于 `nvidia-smi` 场景 |
| v3 | 加入人工标注分数，训练 reward model |
| v4 | 用强模型做 judge，但要防止 judge 偏见和不稳定 |
| v5 | 针对真实任务引入可执行验证，例如命令是否安全、JSON 是否合法 |

你现在不要急着上 v4。先把 v1 的失败原因写清楚。
