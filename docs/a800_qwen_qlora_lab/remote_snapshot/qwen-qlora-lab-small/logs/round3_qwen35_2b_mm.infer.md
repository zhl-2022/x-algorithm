# round3_qwen35_2b_mm

## text3

checkpoint: outputs/qwen35_2b_mm/v0-20260613-213024/checkpoint-20
dataset: data/infer_text_prompts.jsonl
result_path: logs/round3_qwen35_2b_mm.text3.result.jsonl
command_log: logs/round3_qwen35_2b_mm.text3.command.log

exit_code: 0
result_rows: 3

result_preview:
- row 1: <think> Thinking Process:  1.  **Analyze the Request:**     *   Topic: QLoRA (Quantized LoRA).     *   Context: Shared A800 environment (shared A800 GPUs).     *   Constraint: Exactly three sentences (三句话).     *   Langu
- row 2: <think> Here's a thinking process that leads to the suggested answer:  1.  **Analyze the Request:**     *   **Topic:** Offline experiments for recommendation systems (推荐系统离线实验).     *   **Task:** List five categories of 

command_tail:
          (norm): Qwen3_5RMSNorm((2048,), eps=1e-06)
          (rotary_emb): Qwen3_5TextRotaryEmbedding()
        )
      )
      (lm_head): Linear(in_features=2048, out_features=248320, bias=False)
    )
  )
)
[INFO:swift] Start time of running main: 2026-06-13 21:53:38.810604
[INFO:swift] swift.__version__: 4.3.0
[INFO:swift] request_config: RequestConfig(max_tokens=256, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stop=[], seed=None, stream=False, logprobs=False, top_logprobs=None, prompt_logprobs=None, n=1, best_of=None, presence_penalty=0.0, frequency_penalty=0.0, length_penalty=1.0, return_details=False, structured_outputs_regex=None)
Map (num_proc=1): 100%|██████████| 3/3 [00:00<?, ? examples/s]Map (num_proc=1): 6 examples [00:00, 14.78 examples/s]        
[INFO:swift] val_dataset: Dataset({
    features: ['messages'],
    num_rows: 3
})
  0%|          | 0/3 [00:00<?, ?it/s] 33%|███▎      | 1/3 [00:25<00:50, 25.46s/it] 67%|██████▋   | 2/3 [00:50<00:25, 25.17s/it]100%|██████████| 3/3 [01:15<00:00, 25.05s/it]100%|██████████| 3/3 [01:15<00:00, 25.11s/it]
{'num_prompt_tokens': 74, 'num_generated_tokens': 768, 'num_samples': 3, 'runtime': 75.33507579192519, 'samples/s': 0.039822087765411865, 'tokens/s': 10.194454467945437}
[INFO:swift] The inference results have been saved to result_path: `/workspace/qwen-qlora-lab/logs/round3_qwen35_2b_mm.text3.result.jsonl`.
[INFO:swift] End time of running main: 2026-06-13 21:54:55.101260


## mm2

checkpoint: outputs/qwen35_2b_mm/v0-20260613-213024/checkpoint-20
dataset: data/infer_mm_prompts.jsonl
result_path: logs/round3_qwen35_2b_mm.mm2.result.jsonl
command_log: logs/round3_qwen35_2b_mm.mm2.command.log

exit_code: 0
result_rows: 2

result_preview:
- row 1: <think> 用户希望我描述这张图并指出训练闭环。  1.  **图片内容分析**：     *   标题：Offline RecSys Pipeline（离线推荐系统流水线）。     *   流程图：三个矩形框，从左到右依次是：         *   `raw logs -> samples`（原始日志 -> 样本）         *   `recall -> ranker`（召回 -> 排序模型）         *   `
- row 2: <think> 用户希望我解释这张图，并说明离线评测应该关注什么。  1.  **分析图片内容：**     *   标题：Ranking Metrics（排名指标）。     *   三个框：         *   Recall@20（召回率@20）：蓝色框。         *   NDCG@20（NDCG@20）：绿色框。         *   AUC / LogLoss（AUC / LogLoss）：棕色框。     *  

command_tail:
          (rotary_emb): Qwen3_5TextRotaryEmbedding()
        )
      )
      (lm_head): Linear(in_features=2048, out_features=248320, bias=False)
    )
  )
)
[INFO:swift] Start time of running main: 2026-06-13 21:55:21.909647
[INFO:swift] swift.__version__: 4.3.0
[INFO:swift] request_config: RequestConfig(max_tokens=256, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stop=[], seed=None, stream=False, logprobs=False, top_logprobs=None, prompt_logprobs=None, n=1, best_of=None, presence_penalty=0.0, frequency_penalty=0.0, length_penalty=1.0, return_details=False, structured_outputs_regex=None)
Generating train split: 0 examples [00:00, ? examples/s]Generating train split: 2 examples [00:00, 932.79 examples/s]
Map (num_proc=1):   0%|          | 0/2 [00:00<?, ? examples/s]Map (num_proc=1): 100%|██████████| 2/2 [00:00<00:00,  9.20 examples/s]
[INFO:swift] val_dataset: Dataset({
    features: ['messages', 'images'],
    num_rows: 2
})
  0%|          | 0/2 [00:00<?, ?it/s] 50%|█████     | 1/2 [00:27<00:27, 27.12s/it]100%|██████████| 2/2 [00:52<00:00, 26.07s/it]100%|██████████| 2/2 [00:52<00:00, 26.23s/it]
{'num_prompt_tokens': 950, 'num_generated_tokens': 512, 'num_samples': 2, 'runtime': 52.4657923290506, 'samples/s': 0.03812007617185243, 'tokens/s': 9.758739499994222}
[INFO:swift] The inference results have been saved to result_path: `/workspace/qwen-qlora-lab/logs/round3_qwen35_2b_mm.mm2.result.jsonl`.
[INFO:swift] End time of running main: 2026-06-13 21:56:15.308574

