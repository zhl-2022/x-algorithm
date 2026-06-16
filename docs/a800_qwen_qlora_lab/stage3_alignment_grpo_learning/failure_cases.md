# Stage 3 Failure Cases

训练前保持本文件为空模板。每次 DPO 或 GRPO 跑完后，至少补 3 个失败案例。

## Case 1

- prompt:
- model:
- expected:
- actual:
- score:
- failure_type:
- root_cause:
- next_action:

## Case 2

- prompt:
- model:
- expected:
- actual:
- score:
- failure_type:
- root_cause:
- next_action:

## Case 3

- prompt:
- model:
- expected:
- actual:
- score:
- failure_type:
- root_cause:
- next_action:

## Failure Type Reference

| 类型 | 解释 |
|---|---|
| `missing_required` | 漏掉必须概念 |
| `forbidden_hit` | 出现危险表达 |
| `keyword_stuffing` | 堆关键词但没有解释 |
| `over_formatting` | 格式漂亮但内容空 |
| `hallucination` | 编造不存在命令、路径或结论 |
| `too_verbose` | 废话过多 |
| `too_short` | 回答过短，不可执行 |
| `reward_mismatch` | 人工判断和 reward 分数冲突 |
