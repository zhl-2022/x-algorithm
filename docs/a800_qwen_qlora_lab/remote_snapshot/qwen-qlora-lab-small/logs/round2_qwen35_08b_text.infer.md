# round2_qwen35_08b_text

## text3

checkpoint: outputs/qwen35_08b_text/v1-20260613-212200/checkpoint-20
dataset: data/infer_text_prompts.jsonl
result_path: logs/round2_qwen35_08b_text.text3.result.jsonl
command_log: logs/round2_qwen35_08b_text.text3.command.log

exit_code: 0
result_rows: 3

result_preview:
- row 1: <think> Thinking Process:  1.  **Analyze the Request:**     *   Topic: QLoRA (Low-Rank Adaptation) model architecture.     *   Constraint: Explain why it's suitable for shared A800 training environments.     *   Format: 
- row 2: <think> 这是一个关于推荐系统离线实验记录的问题。用户询问的是需要记录的五类信息。这是一个常见的机器学习/深度学习实验记录问题，通常出现在第 10 个学习样本中。  让我思考一下推荐系统离线实验的常见记录类别。通常包括： 1. 实验配置 2. 数据特征 3. 模型参数 4. 训练/测试指标 5. 可复现步骤  但这是第 10 个学习样本，可能更关注可复现和低风险操作。让我提供几个常见的第 10 个学习样本记录类别。  实际上，第 

command_tail:
          (norm): Qwen3_5RMSNorm((1024,), eps=1e-06)
          (rotary_emb): Qwen3_5TextRotaryEmbedding()
        )
      )
      (lm_head): Linear(in_features=1024, out_features=248320, bias=False)
    )
  )
)
[INFO:swift] Start time of running main: 2026-06-13 21:51:55.648333
[INFO:swift] swift.__version__: 4.3.0
[INFO:swift] request_config: RequestConfig(max_tokens=256, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stop=[], seed=None, stream=False, logprobs=False, top_logprobs=None, prompt_logprobs=None, n=1, best_of=None, presence_penalty=0.0, frequency_penalty=0.0, length_penalty=1.0, return_details=False, structured_outputs_regex=None)
Map (num_proc=1): 100%|██████████| 3/3 [00:00<?, ? examples/s]Map (num_proc=1): 6 examples [00:00, 15.41 examples/s]        
[INFO:swift] val_dataset: Dataset({
    features: ['messages'],
    num_rows: 3
})
  0%|          | 0/3 [00:00<?, ?it/s] 33%|███▎      | 1/3 [00:24<00:49, 24.93s/it] 67%|██████▋   | 2/3 [00:50<00:25, 25.08s/it]100%|██████████| 3/3 [01:14<00:00, 24.97s/it]100%|██████████| 3/3 [01:14<00:00, 24.99s/it]
{'num_prompt_tokens': 74, 'num_generated_tokens': 768, 'num_samples': 3, 'runtime': 74.96890152897686, 'samples/s': 0.04001659273132667, 'tokens/s': 10.244247739219627}
[INFO:swift] The inference results have been saved to result_path: `/workspace/qwen-qlora-lab/logs/round2_qwen35_08b_text.text3.result.jsonl`.
[INFO:swift] End time of running main: 2026-06-13 21:53:11.509887

