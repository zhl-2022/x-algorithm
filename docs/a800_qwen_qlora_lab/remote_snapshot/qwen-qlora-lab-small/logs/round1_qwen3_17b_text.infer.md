# round1_qwen3_17b_text

## text3

checkpoint: outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-20
dataset: data/infer_text_prompts.jsonl
result_path: logs/round1_qwen3_17b_text.text3.result.jsonl
command_log: logs/round1_qwen3_17b_text.text3.command.log

exit_code: 0
result_rows: 3

result_preview:
- row 1: <think>  </think>  QLoRA 适合共享 A800 学习环境的原因包括：1) 高效的内存管理，2) 低资源占用，3) 适合大规模并行训练。
- row 2: <think> 嗯，用户让我列出推荐系统离线实验需要记录的五类信息。首先，我需要理解推荐系统离线实验的基本内容。离线实验通常是在没有实时数据的情况下进行的，比如在训练模型时，使用历史数据来评估效果。用户可能是在做推荐系统的开发，需要记录实验结果以便后续分析和优化。  首先，我应该考虑实验的主要目标。推荐系统有很多类型，比如协同过滤、基于内容的推荐，或者混合模型。离线实验需要评估不同模型的效果，比如准确率、召回率、F1值等。所以可能需要记

command_tail:
        (norm): Qwen3RMSNorm((2048,), eps=1e-06)
        (rotary_emb): Qwen3RotaryEmbedding()
      )
      (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
    )
  )
)
[INFO:swift] Start time of running main: 2026-06-13 21:50:37.417667
[INFO:swift] swift.__version__: 4.3.0
[INFO:swift] request_config: RequestConfig(max_tokens=256, temperature=None, top_k=None, top_p=None, repetition_penalty=None, num_beams=1, stop=[], seed=None, stream=False, logprobs=False, top_logprobs=None, prompt_logprobs=None, n=1, best_of=None, presence_penalty=0.0, frequency_penalty=0.0, length_penalty=1.0, return_details=False, structured_outputs_regex=None)
Generating train split: 0 examples [00:00, ? examples/s]Generating train split: 3 examples [00:00, 1826.52 examples/s]
Map (num_proc=1):   0%|          | 0/3 [00:00<?, ? examples/s]Map (num_proc=1): 100%|██████████| 3/3 [00:00<00:00, 16.82 examples/s]
[INFO:swift] val_dataset: Dataset({
    features: ['messages'],
    num_rows: 3
})
  0%|          | 0/3 [00:00<?, ?it/s] 33%|███▎      | 1/3 [00:04<00:09,  4.87s/it] 67%|██████▋   | 2/3 [00:28<00:15, 15.67s/it]100%|██████████| 3/3 [00:51<00:00, 19.00s/it]100%|██████████| 3/3 [00:51<00:00, 17.02s/it]
{'num_prompt_tokens': 73, 'num_generated_tokens': 559, 'num_samples': 3, 'runtime': 51.06827013194561, 'samples/s': 0.05874489173510028, 'tokens/s': 10.946131493307018}
[INFO:swift] The inference results have been saved to result_path: `/workspace/qwen-qlora-lab/logs/round1_qwen3_17b_text.text3.result.jsonl`.
[INFO:swift] End time of running main: 2026-06-13 21:51:29.509763

