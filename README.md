# LingBot-VLA 2.0 × RoboTwin：clean 数据后训练

本目录是天池比赛 532514 的本地开发工程。起点为 `robbyant/lingbot-vla-v2-6b`，训练只使用官方 RoboTwin 2.0 Aloha-AgileX 的 50 任务 × 50 条 clean 示范。比赛规则明确要求行为克隆（BC），在线、离线强化学习均不允许；不能使用 randomized 数据训练。完整规则及来源见 [赛事核查](docs/competition-research.md)，用户提供的正式模板保存在 [official-templates](docs/official-templates)。

环境安装、两种真实单步训练及导出模型离线推理已完成；已下载全部 clean 原始包，当前转换数据为一个任务的 50 条示范。本机 LoRA 和末两层 action expert 单步训练的峰值显存分别约为 12.60 GiB 和 13.46 GiB。当前没有完整 50 任务训练或闭环成功率成绩。

## 环境

Miniconda 全局安装在 Linux home 的 `~/miniconda3`，环境名 `lingbot-vla`（Python 3.12、PyTorch 2.8/CUDA 12.8）和 `robotwin-sim`（Python 3.10），均位于 `~/miniconda3/envs/`。Conda 已初始化到 `~/.bashrc`，新打开的 WSL Bash 终端可直接使用：

```bash
conda activate lingbot-vla
```

当前终端如需立即加载初始化，执行 `source ~/.bashrc`。进入项目时，使用项目激活脚本同时设置源码、模型和缓存路径：

```bash
cd ~/eclipseaws/lingbot-vla-robotwin
source scripts/activate.sh
python scripts/doctor.py --output artifacts/doctor.json
```

新机器配置：

```bash
bash scripts/bootstrap.sh
bash scripts/setup_sim.sh --with-curobo
```

`bootstrap.sh` 固定官方 v2 源码版本；`setup_sim.sh` 使用官方要求的 RoboTwin 提交，并针对 RTX 50 系列使用 CUDA 12.8 的 PyTorch 2.8。仿真上游的 PyTorch 2.4.1/CUDA 12.1 不支持本机 Blackwell，故这里更换了 Torch 版本，评测记录会保留实际软件栈。

项目保留用户的 Windows PATH 和其他环境配置，不主动过滤或隔离。安装和运行入口显式选择并检查 WSL 内的 Linux Python、Conda、Git、Curl 和 CUDA 编译器；工具解析记录见 `artifacts/wsl-tool-resolution.json`，Bash 启动检查见 `artifacts/bash-startup-validation.json`。WSL 使用 Windows NVIDIA 驱动提供的 Linux CUDA 接口，这是正常的 GPU 透传，不安装 Linux NVIDIA 驱动。`CONDA_ROOT` 可指定另一个 Linux Conda 前缀。

WSL 的 CUDA 计算与 NVIDIA Vulkan 渲染是两项独立能力。RoboTwin 官方不支持 WSL 渲染；本机 SAPIEN 相机创建报错 `failed to find a rendering device`，闭环评测需迁移到支持 NVIDIA Vulkan 的原生 Linux。`doctor.py` 对渲染进行隔离测试，闭环评测在加载大模型前也会检查渲染。不能用训练损失或离线动作误差代替任务成功率。

## 下载与数据准备

下载脚本按固定 Hugging Face commit 筛选文件，支持断点续传和 SHA256 校验，记录来源到 `artifacts/download_*.json`。基模约 25.5 GB；Qwen3 配置/分词器约 12 MB，基模已经包含 VLM 权重；50 个 clean 原始压缩包约 23.8 GB。仿真对象、机器人和背景资产另外约 14.9 GB，解压和数据转换还需要空间。

`configs/raw_clean_hf_metadata.json` 保存固定官方版本的 50 个 clean 压缩包 SHA256 清单，随代码提交；数据解压与审计使用这份清单核验来源。

```bash
python scripts/download_assets.py --asset base
python scripts/download_assets.py --asset qwen
python scripts/download_assets.py --asset clean
python scripts/download_assets.py --asset sim
python scripts/prepare_sim_assets.py
source scripts/activate.sh robotwin-sim
cd vendor/RoboTwin
python script/update_embodiment_config_path.py
cd ../..
source scripts/activate.sh
```

先验证一个任务的数据转换：

```bash
python scripts/prepare_data.py --tasks adjust_bottle --allow-subset
```

准备比赛训练数据：

```bash
python scripts/prepare_data.py --tasks all
```

转换遵循固定版本 pi0 处理器的帧对齐：图像与状态 `[t]` 预测关节命令 `[t+1]`，两臂各 6 关节 + 1 夹爪共 14 维；只选官方 seen 指令。不能直接使用另一个公开的 EEF16 LeRobot 数据集，它与此模型的 RoboTwin 映射不兼容。

转换输出 LeRobot 数据、`data/clean_manifest.json` 和 `data/clean_norm_stats.json`。每任务必须恰好 50 条官方示范，默认第 45–49 条做离线验证，归一化只从第 0–44 条计算。数据审计检查来源、示范数、关节和相机映射；不使用上游混合 clean/randomized 的 manifest 或 norm stats。

`--allow-subset` 只用于跑通流程，产物会标记为任务子集。正式联合训练需要全部 50 个任务，共享一份权重。

## 训练

无需下载 6B 权重的合成训练、保存、恢复与 LoRA 合并测试：

```bash
bash train.sh --smoke --device cuda --output-dir artifacts/training-smoke-cuda
python -m pytest -q
```

真实 LoRA 单步验证（先转换 `adjust_bottle`）：

```bash
bash train.sh --config configs/train_lora.yaml --allow-subset \
  --max-steps 1 --gradient-accumulation 1 --output-dir runs/lora-one-step
```

联合训练入口：

```bash
bash train.sh --config configs/train_lora.yaml
bash train.sh --config configs/train_expert.yaml
```

LoRA 配置冻结 VLM 和 action MoE 主权重，训练 action attention 的低秩增量与 state/action 投影。action expert 本地配置只训练最后 2/36 层、归一化和投影，是部分 expert 微调。要在服务器训练完整 action expert：

```bash
bash train.sh --config configs/train_expert.yaml --expert-last-n-layers -1 \
  --output-dir runs/expert-full-clean
```

本地实现使用 BF16 常驻权重、FP32 可训练参数/AdamW、冻结前缀 KV、action 层梯度检查点和梯度累积。加载原始 FP32 分片时逐张量转换并直接送入 GPU，避免在 15 GB WSL 内存中复制整个模型。SDPA 和 PyTorch MoE 参考实现保留官方结构和 checkpoint 键，可以先完成本地验证；训练速度与官方 fused/多卡版本不同。显存报告的下界不包含激活和 CUDA 工作区，能否在 16/24 GB 训练须以实际峰值为准。

保存位置：

```text
runs/<experiment>/
├── training_config.yaml / lingbotvla_cli.yaml
├── run_metadata.json / episode_splits.json
├── assets/clean_manifest.json / norm_stats.json
├── configs/robot_configs/robotwin.yaml
├── metrics.jsonl / latest_checkpoint.txt
└── checkpoints/global_step_N/
    ├── training.pt
    └── hf_ckpt/                 # 完整权重，LoRA 已合并
```

恢复训练时传入 `--resume runs/<experiment>/checkpoints/global_step_N/training.pt`。`training.pt` 包含可训练权重、优化器、调度器和进度；`hf_ckpt` 用于推理，不能当作完整优化器恢复状态。

训练默认只计算 action 的 flow-matching BC 损失，保留基模的深度/视频查询及输出头，但不加载教师模型或计算相关辅助损失。默认 BF16 导出节省显存；把这些权重转换为 FP32 不能恢复加载时舍弃的精度。

`action_chunk_tail: mask`（默认）保留轨迹末端不足未来 `chunk_size` 步的观测，时间 padding 不参与损失。`action_chunk_tail: drop` 只保留训练集的完整动作窗口，验证集仍保留全部观测；它会减少末段状态作为训练起点的覆盖，不会改变输出动作块长度。`drop_last` 只处理不足 batch 的批次，不能替代此设置。当前 `adjust_bottle` 的 6,478 个训练起点中，2,205 个窗口不足 50 步；丢弃后保留 4,273 个，详见 `artifacts/action-chunk-tail-audit.json`。

梯度累积对整个更新窗口的有效动作坐标求平均，梯度统一归一化后再裁剪和更新，验证损失也按有效坐标加权。采样方式、batch 大小、episode 划分和损失归一化写入 checkpoint，恢复时核对；修复前的 checkpoint 缺少这些信息，不能按新版本进行精确续训。重新实验应使用新的输出目录。

调参使用每任务 45 条训练、5 条验证示范。确定超参数后，可从官方基模重新训练全部 2,500 条 clean 示范；先重算无留出的归一化，再使用相同训练划分：

```bash
python scripts/prepare_data.py --tasks all --audit-only --val-episodes 0
bash train.sh --config configs/train_lora.yaml --val-episodes 0 \
  --output-dir runs/lora-all-clean
```

安装遵循官方 v2 的 Torch/Transformers 固定版本，并以 `--no-deps` 接入 LeRobot 和可选深度源码。`pip check` 会报告其声明版本范围和未安装的可选机器人/教师依赖，记录在 `artifacts/pip-check-training.txt`；实际数据转换和训练使用的接口单独验证。深度教师训练和 LeRobot 机器人遥操作不属于当前入口。

## 测试与评测

WSL 上可以做 clean 留出示范的动作预测诊断：

```bash
python scripts/open_loop.py \
  --checkpoint runs/lora-one-step/checkpoints/global_step_1/hf_ckpt
```

在支持 SAPIEN NVIDIA Vulkan 的原生 Linux 上，先做一个任务、每种环境 2 次 BF16 闭环验证：

```bash
bash eval.sh --checkpoint runs/lora-one-step/checkpoints/global_step_1/hf_ckpt --smoke
```

完整比赛评测：

```bash
bash eval.sh --checkpoint runs/lora_clean/checkpoints/global_step_1000/hf_ckpt \
  --tasks all --setting both --trials 100 --precision fp32 --team-id YOUR_TEAM_ID
```

50 个任务 × 两种 setting × 100 次，共 10,000 次。只有足够显存的服务器能运行完整 FP32；本机 24 GB 使用 BF16 诊断，16 GB 还需更小模型输入或服务器。默认本地流式服务器的 SDPA/MoE 数值与官方 fused 内核等价性尚未完成验证；使用原生官方服务器加 `--native-server`，但其初始化需要大量主机内存。官方 clean+randomized 参考成绩为 clean 93.52%、randomized 92.80%，不能把它当作 clean-only 已复现成绩。

评测修改保存在 `src/lrvla/evaluation.py`，运行时生成 client 副本，不修改官方仓库：clean 用 seen 指令、randomized 用 unseen 指令；每场景至多启动 5 次专家尝试，耗尽后仍运行模型；只加载一份固定权重，重置不切换模型；成功采用 `check_success()`。程序保存计数与进程状态，失败任务不能默认为完成。

结果目录包含详细 `results.json`、独立的精度/命令记录和按附件正式 schema 输出的 `official_results.json`。缺测任务为 `0/0`，不能当作零成功率或完整比赛结果。正式提交使用自己的团队 ID。

## 材料打包与迁移

完成真实训练和评测后，按附件 [复现报告模板](docs/official-templates/复现与调优说明模板.md)填写报告，并打包到本地：

```bash
python scripts/package_submission.py \
  --results runs/evaluation/<run>/official_results.json \
  --run runs/lora_clean \
  --report reproduction_report.md \
  --output artifacts/submission.zip
```

脚本校验正式 JSON 字段及次数，拒绝把空模板打包成测得结果。`02_代码材料` 包含 README、train.sh、eval.sh、configs、src、scripts；`03_模型材料` 保留完整单份导出权重。用户账户凭据、Conda 环境、缓存和数据不进入包。这里只生成本地文件，上传比赛由用户完成。

结果须携带同目录的 `evaluation_plan.json` 和详细 `results.json`；打包会核对实测计数及实际评测权重的 SHA256，避免把一个模型的结果配给另一个模型。默认随包保存官方源码和 SHA256 快照，解压后无需 `.git` 即可校验版本并运行安装脚本；`--no-vendor` 可改为安装时按固定提交重新下载。

迁移时保留源代码、官方版本记录、模型、clean 数据和训练输出，重新运行两个环境安装脚本。不要复制 WSL 的 Conda 前缀到不同路径。需要恢复优化器时另外保留 `training.pt`；仅部署则使用完整 `hf_ckpt` 与其 run 配置/归一化。初始流程使用单 GPU；大规模多卡训练可基于已固定的官方训练入口继续开发，当前本地 LoRA/部分 expert 入口不声称支持 DDP/FSDP。

## 目录

`vendor/` 是固定版本官方源码；`src/lrvla/` 是数据审计、训练适配、流式加载和评测逻辑；`configs/` 为本地训练配置；`scripts/` 为安装/下载/转换/诊断入口；`tests/` 验证数据隔离、动作掩码、梯度、恢复、合并导出和真实 JSON schema；`artifacts/` 保存实际安装和验证记录。

`docs/` 中仅官方附件模板和附官方来源的赛事规则核查纳入 Git 与提交包；提交规范、开发说明和本机验证记录保留本地并忽略。
