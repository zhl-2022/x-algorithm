#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/workspace/qwen-qlora-lab') if Path('/workspace/qwen-qlora-lab').exists() else Path('/root/zhl/qwen-qlora-lab')
DATA = ROOT / 'data'
IMAGES = ROOT / 'images'
DATA.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

text_topics = [
    ('推荐系统召回', '召回阶段负责从大规模候选池中快速找出可能相关的物品，常见方法包括 ItemCF、双塔向量召回和图模型召回。'),
    ('推荐系统排序', '排序阶段使用更丰富的用户、物品和上下文特征，对召回候选进行精排，目标通常是点击率、完播率或综合效用。'),
    ('QLoRA 显存优化', 'QLoRA 通过 4bit 量化冻结基座模型并训练低秩适配器，在较低显存下完成指令微调学习。'),
    ('LoRA 参数含义', 'LoRA rank 控制低秩矩阵容量，alpha 控制缩放强度，target modules 决定哪些线性层插入适配器。'),
    ('A800 共享训练', '共享 GPU 上训练应先检查 nvidia-smi，限制 batch size 和序列长度，并保留训练日志避免影响已有服务。'),
    ('ModelScope 下载', '当服务器访问 Hugging Face 不稳定时，可以优先通过 ModelScope 下载 Qwen 模型权重到本地目录。'),
    ('多模态数据格式', '多模态微调样本通常包含 messages 和 images 字段，文本中用 image 标签标记图像插入位置。'),
    ('训练日志记录', '一次可复现实验至少记录模型、数据、命令、显存、训练步数、checkpoint 路径和推理样例。'),
    ('推荐模型评估', '离线评估常用 Recall@K、NDCG@K、AUC 和 LogLoss，TopK 指标更贴近候选召回和重排效果。'),
    ('显存保护策略', '如果显存峰值过高，应优先降低 max_length、batch size 和候选规模，再考虑更小模型。'),
]

rows = []
for i in range(180):
    topic, answer = text_topics[i % len(text_topics)]
    rows.append({
        'messages': [
            {'role': 'system', 'content': '你是一个帮助学习推荐系统、服务器运维和大模型微调的中文技术助手。'},
            {'role': 'user', 'content': f'请用简洁中文解释：{topic}。'},
            {'role': 'assistant', 'content': answer + f' 这是第 {i + 1} 个学习样本，重点是可复现和低风险操作。'}
        ]
    })

with (DATA / 'text_sft.jsonl').open('w', encoding='utf-8') as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:
    raise SystemExit(f'Pillow is required to generate multimodal data: {exc}')

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
except Exception:
    font = None
    small_font = None

mm_specs = [
    ('pipeline', 'Offline RecSys Pipeline', ['raw logs -> samples', 'recall -> ranker', 'metrics -> report'], '这张图展示了离线推荐系统从日志到评测报告的三段流程。'),
    ('gpu', 'A800 Shared GPU Policy', ['check nvidia-smi', 'limit memory usage', 'write logs'], '这张图强调共享 A800 上训练前检查显存、限制显存并保存日志。'),
    ('qlora', 'QLoRA Training Loop', ['4-bit base model', 'LoRA adapters', 'small supervised data'], '这张图说明 QLoRA 使用 4bit 基座模型和 LoRA adapter 进行小样本监督微调。'),
    ('metrics', 'Ranking Metrics', ['Recall@20', 'NDCG@20', 'AUC / LogLoss'], '这张图列出推荐实验常见指标：Recall、NDCG、AUC 和 LogLoss。'),
    ('download', 'Model Download Fallback', ['ModelScope first', 'Hugging Face fallback', 'verify safetensors'], '这张图说明模型下载优先 ModelScope，失败后回退 Hugging Face，并检查权重文件。'),
    ('safety', 'Low Risk Training', ['batch size = 1', 'max length <= 1024', 'stop on OOM'], '这张图说明低风险训练应控制 batch size、序列长度并在 OOM 时停止。'),
]

mm_rows = []
for idx, (name, title, bullets, answer) in enumerate(mm_specs):
    path = IMAGES / f'{idx + 1:02d}_{name}.png'
    img = Image.new('RGB', (900, 520), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 900, 72), fill=(28, 89, 128))
    draw.text((32, 20), title, fill=(255, 255, 255), font=font)
    colors = [(45, 125, 154), (88, 145, 103), (151, 99, 67)]
    for j, bullet in enumerate(bullets):
        x0 = 70 + j * 270
        y0 = 170
        draw.rounded_rectangle((x0, y0, x0 + 210, y0 + 120), radius=12, fill=colors[j], outline=(30, 30, 30), width=2)
        draw.text((x0 + 18, y0 + 42), bullet, fill=(255, 255, 255), font=small_font)
        if j < len(bullets) - 1:
            draw.line((x0 + 220, y0 + 60, x0 + 255, y0 + 60), fill=(60, 60, 60), width=4)
            draw.polygon([(x0 + 255, y0 + 60), (x0 + 242, y0 + 51), (x0 + 242, y0 + 69)], fill=(60, 60, 60))
    draw.text((70, 360), 'Task: describe the technical meaning of this diagram in Chinese.', fill=(20, 20, 20), font=small_font)
    img.save(path)
    mm_rows.append({
        'messages': [
            {'role': 'system', 'content': '你是一个能读懂技术流程图的中文助手。'},
            {'role': 'user', 'content': '<image> 请描述这张技术图的核心含义。'},
            {'role': 'assistant', 'content': answer}
        ],
        'images': [str(path)]
    })

with (DATA / 'mm_sft.jsonl').open('w', encoding='utf-8') as f:
    for row in mm_rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f'wrote {DATA / "text_sft.jsonl"} rows={len(rows)}')
print(f'wrote {DATA / "mm_sft.jsonl"} rows={len(mm_rows)}')