# srv4 MLU 训练容器方案

## 结论

当前项目是推荐系统学习与训练，默认使用新版寒武纪 PyTorch 镜像：

```text
cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310
```

不默认使用 `klb/llamafactory-mlu:v1`，因为它更适合 LLaMAFactory 大模型微调，
而当前项目下一步主要是 MovieLens、MF、Two-Tower 等推荐模型实验。

## 当前服务器状态

| 项目 | 信息 |
|---|---|
| 目标 | `srv4` |
| 主机名 | `node2` |
| 设备 | 8 张 `MLU590-H8`，单卡 80GB |
| 建议训练卡 | Card `2`、Card `3` |
| 可见设备设置 | `MLU_VISIBLE_DEVICES=2,3` |
| 设备监控 | `cnmon` |
| 项目远端路径 | `/root/zhl/x-algorithm` |

训练前必须重新执行：

```bash
cnmon
```

确认 Card `2`、Card `3` 仍为空闲状态。

## 推荐启动方式

先通过本地 PowerShell 连接服务器：

```powershell
srv
```

在服务器上运行：

```bash
cd /root/zhl/x-algorithm
bash scripts/mlu/start_xalgorithm_mlu.sh
```

这个命令会启动一个交互式临时容器，退出后自动删除容器：

```bash
docker run -it --rm \
  --name xalgorithm-mlu-dev \
  --ipc host \
  --privileged \
  -v /etc/cambricon:/etc/cambricon:ro \
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro \
  -v /root/zhl:/root/zhl \
  -v /data1/model:/data1/model:ro \
  -e MLU_VISIBLE_DEVICES=2,3 \
  -e LANG=C.UTF-8 \
  -e TZ=Asia/Shanghai \
  -e PYTHONUNBUFFERED=1 \
  -w /root/zhl/x-algorithm \
  cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310 \
  bash -lc "source /torch/venv3/pytorch/bin/activate 2>/dev/null || true; exec bash"
```

进入容器后检查：

```bash
python -c 'import torch, torch_mlu; print(torch.mlu.is_available()); print(torch.mlu.device_count())'
```

期望输出：

```text
True
2
```

## 后台开发容器

如果需要保留容器，使用：

```bash
cd /root/zhl/x-algorithm
bash scripts/mlu/start_xalgorithm_mlu.sh --detached
```

进入后台容器：

```bash
docker exec -it xalgorithm-mlu bash
```

进入后台容器后建议先执行：

```bash
source /torch/venv3/pytorch/bin/activate
```

停止后台容器：

```bash
docker rm -f xalgorithm-mlu
```

## 与旧 LLaMAFactory 命令的差异

| 旧设置 | 当前设置 | 原因 |
|---|---|---|
| `klb/llamafactory-mlu:v1` | `cambricon-base/pytorch:v25.12.0...` | 当前任务是推荐模型训练，不是 LLaMAFactory 微调 |
| `MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` | `MLU_VISIBLE_DEVICES=2,3` | 避免看到并误占业务卡 |
| `--restart unless-stopped` | 默认不使用 | 学习训练容器不应重启后自动占卡 |
| `--pid host` | 默认不使用 | 当前 PyTorch 推荐训练不需要共享宿主 PID |
| Web UI 端口映射 | 默认不映射 | 当前训练脚本不需要 Web UI |

## 本地检查命令

从本地 PowerShell 执行：

```powershell
powershell -NoProfile -File scripts\mlu\check_srv4_mlu.ps1
```

该脚本会检查：

1. `srv4` 是否连通。
2. `cnmon` 当前设备状态。
3. `xalgorithm-mlu` 容器是否已经存在。
4. 推荐镜像能否通过 `torch_mlu` 看到 2 张 MLU。
