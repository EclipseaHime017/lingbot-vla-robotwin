# LingBot-VLA 2.0 × RoboTwin

基于 `robbyant/lingbot-vla-v2-6b` 的单 GPU 后训练工程，支持 RoboTwin 2.0 Aloha-AgileX clean 示范数据的下载、转换、训练和动作预测评估。模型以三路 RGB、关节状态和任务指令为输入，通过 flow-matching 行为克隆（BC）学习动作块。

当前提供两种训练方式：action attention 的 **LoRA** 微调，以及 **Action Expert** 的部分层或全部层微调。两种方式都冻结视觉语言模型（VLM），支持梯度累积、梯度检查点、训练状态恢复和完整推理权重导出。导出时自动合并 LoRA。仓库另提供离线动作预测和 RoboTwin 闭环仿真入口。

## 1. 仓库框架

```text
lingbot-vla-robotwin/
├── configs/                 # 训练 YAML、任务列表、原始数据哈希清单
│   ├── train_lora.yaml
│   └── train_expert.yaml
├── src/lrvla/               # 数据加载、模型适配、训练、checkpoint 和推理
├── scripts/                 # 环境安装、资源下载、数据转换及诊断入口
├── tests/                   # 数据、梯度、恢复、导出等回归测试
├── vendor/                  # 固定版本的 LingBot-VLA 与 RoboTwin 源码
├── models/                  # 基模权重和 Qwen 处理器资源
├── data/
│   ├── raw_archives/        # 原始数据压缩包
│   ├── raw/                 # 解压后的 HDF5 与指令
│   ├── lerobot/             # 转换后的 LeRobot 数据
│   ├── clean_manifest.json  # 任务、轨迹及数据来源索引
│   └── clean_norm_stats.json
├── runs/                    # 训练日志、恢复状态和导出权重
├── artifacts/               # 下载、环境诊断及验证记录
├── bootstrap/               # 安装器与依赖 wheel 暂存
├── .cache/                  # pip、Hugging Face 等缓存
├── requirements-extra.txt
├── pyproject.toml
├── train.sh                 # 训练入口
└── eval.sh                  # 闭环仿真入口
```

核心模块按职责拆分：`training_data.py` 负责轨迹划分和动作窗口；`training_models.py` 负责可训练参数选择及 BC 损失；`training.py` 执行训练和验证；`training_checkpoint.py` 保存状态、核对恢复条件及合并导出；`training_inference.py` 和 `evaluation.py` 提供推理与评估适配。

`vendor/`、`models/`、`data/`、`runs/`、`artifacts/` 和缓存目录由脚本生成，不纳入 Git。安装环境位于 Linux home 下，项目目录只保存源码、资源和运行输出。

## 2. 环境配置与资源下载

### 2.1 安装训练环境

面向 Linux x86_64，可在 WSL Ubuntu 中进行 CUDA 训练。先进入仓库，安装基础工具，再运行环境脚本：

```bash
cd ~/eclipseaws/lingbot-vla-robotwin
sudo apt-get update
sudo apt-get install -y python3 git curl ca-certificates build-essential

bash scripts/bootstrap.sh
source scripts/activate.sh
```

`bootstrap.sh` 将 Miniconda 安装到 **`~/miniconda3`**，创建 `lingbot-vla` 环境，获取固定版本的 LingBot-VLA 和 RoboTwin 源码并安装依赖。已有 Miniconda 时复用该安装；设置 `CONDA_ROOT` 可指定其他 Linux Conda 前缀。

| 环境 | Python | 主要依赖 | 用途 |
|---|---|---|---|
| `lingbot-vla` | 3.12 | PyTorch 2.8.0 / CUDA 12.8、Transformers 4.57.3、LeRobot 0.4.2 | 数据准备、训练、推理 |
| `robotwin-sim`（可选） | 3.10 | PyTorch 2.8.0 / CUDA 12.8、SAPIEN 3.0.0b1、MPLib 0.2.1、CuRobo 0.7.8 | 闭环仿真 |

默认安装 FlashAttention。本地训练使用 SDPA 和 PyTorch MoE 参考实现，可用以下命令跳过 FlashAttention 安装：

```bash
INSTALL_FLASH_ATTN=0 bash scripts/bootstrap.sh
```

新终端中使用 `source scripts/activate.sh` 激活训练环境，同时设置源码、模型及项目缓存路径。脚本在 WSL 中显式选择并检查 Linux Python、Conda、Git、Curl 等工具，保留现有 PATH。WSL 的 CUDA 接口由 Windows NVIDIA 驱动提供。

检查训练环境：

```bash
python scripts/doctor.py --scope train --output artifacts/doctor-train.json
```

### 2.2 下载模型与数据

下载脚本固定资源版本，支持断点续传，并记录文件大小及可用的 SHA256 校验信息。

```bash
python scripts/download_assets.py --asset base
python scripts/download_assets.py --asset qwen
python scripts/download_assets.py --asset clean
```

| 资源 | 本地路径 |
|---|---|
| LingBot-VLA 6B 基模 | `models/lingbot-vla-v2-6b/` |
| Qwen3-VL 配置、分词器与处理器 | `models/Qwen3-VL-4B-Instruct/` |
| RoboTwin clean 原始压缩包 | `data/raw_archives/dataset/<task>/aloha-agilex_clean_50.zip` |

基模已经包含 VLM 权重，`qwen` 下载项只获取处理器资源。可先下载单任务，或用 `--plan` 查看资源大小：

```bash
python scripts/download_assets.py --asset clean --tasks adjust_bottle
python scripts/download_assets.py --asset base --plan
```

转换单任务或全部 50 个任务：

```bash
# 单任务流程验证
python scripts/prepare_data.py --tasks adjust_bottle --allow-subset

# 全部任务
python scripts/prepare_data.py --tasks all
```

当前转换器要求每任务 50 条 Aloha-AgileX clean 示范，输出 LeRobot 数据、manifest 和归一化统计。帧对齐为图像与状态 `[t]` 对应动作 `[t+1]`，动作使用两臂关节与夹爪的 14 维绝对目标，并从原始 seen 指令池选择任务描述。

默认每任务第 0–44 条轨迹用于训练，第 45–49 条用于验证；归一化统计仅从训练轨迹计算。任务按 manifest 合并，样本在每轮打乱，共享一套模型参数。使用任务子集时，数据准备和训练都需要加 `--allow-subset`。

### 2.3 可选：安装仿真环境

闭环仿真需要独立的 `robotwin-sim` 环境及场景资产：

```bash
bash scripts/setup_sim.sh --with-curobo

source scripts/activate.sh
python scripts/download_assets.py --asset sim
python scripts/prepare_sim_assets.py

source scripts/activate.sh robotwin-sim
cd vendor/RoboTwin
python script/update_embodiment_config_path.py
cd ../..
python scripts/doctor.py --scope sim --output artifacts/doctor-sim.json

source scripts/activate.sh
```

`setup_sim.sh --with-curobo` 按需安装 Linux CUDA 12.8 编译器和 C++ 编译工具，编译 CuRobo。训练及离线动作预测只需要 `lingbot-vla` 环境。

闭环运行需要 SAPIEN 相机的 GPU Vulkan 渲染，即使不显示窗口，也需要生成 RGB 观测。本机 WSL 的 CUDA 训练可用，相机渲染尚未通过；闭环运行应使用具备 NVIDIA Vulkan 渲染能力的 Linux 环境，并先通过环境诊断。

## 3. 训练配置与指令

### 3.1 公共配置

训练入口读取 YAML，命令行中提供的参数覆盖 YAML 对应项。两份预设都使用 BF16 常驻权重、FP32 可训练参数及 AdamW，默认只计算动作头的 flow-matching BC 损失。

| YAML 配置 | 默认值 | 含义 |
|---|---|---|
| `base_checkpoint` | `models/lingbot-vla-v2-6b` | 基模路径 |
| `qwen_path` | `models/Qwen3-VL-4B-Instruct` | 处理器资源路径 |
| `manifest` | `data/clean_manifest.json` | 训练任务及数据索引 |
| `norm_stats` | `data/clean_norm_stats.json` | 训练数据归一化统计 |
| `batch_size` | `1` | 每次前向、反向的样本数 |
| `gradient_accumulation` | `16` | 每次更新累积的小 batch 数 |
| `max_steps` | `1000` | 优化器更新次数 |
| `warmup_steps` | `100` | 学习率预热的更新次数 |
| `chunk_size` | `50` | 每个观测对应的动作块长度 |
| `action_chunk_tail` | `mask` | 不完整动作窗口的处理方式 |
| `img_size` | `256` | 图像预处理尺寸 |
| `image_augment` | `false` | 是否启用训练图像增强 |
| `num_workers` | `0` | 数据加载 worker 数量 |
| `val_episodes` | `5` | 每任务留出的验证轨迹数 |
| `gradient_checkpointing` | `true` | 减少反向传播的激活显存 |
| `eval_every` / `save_every` | `100` | 验证、保存的更新间隔 |
| `eval_batches` | `8` | 每次验证最多处理的小 batch 数 |
| `export` / `export_dtype` | `true` / `bfloat16` | 结束后导出完整推理权重及其精度 |

有效 batch 约为 `batch_size × gradient_accumulation`。例如 `batch_size=8`、累积 4 次、`max_steps=2500`，共执行 **10,000 个小 batch、2,500 次参数更新**。这里的 step、预热、验证和保存间隔都按优化器更新计数。

`action_chunk_tail: mask` 保留轨迹末端观测，padding 不参与损失；累积梯度按整个更新窗口的有效动作坐标数统一归一化。`drop` 只保留训练集的完整动作窗口，验证集仍保留全部观测。截断会减少末段观测作为训练起点的覆盖。

可复制预设 YAML 调整实验。LoRA rank、alpha、学习率、动作块长度和尾部策略等需要修改 YAML；常用命令行覆盖项可通过帮助查看：

```bash
bash train.sh --help
```

### 3.2 LoRA

配置文件：[configs/train_lora.yaml](configs/train_lora.yaml)。

```yaml
mode: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_targets: [q_proj, k_proj, v_proj, o_proj]
train_projections: true
lr: 0.0001
output_dir: runs/lora_clean
```

LoRA 加在 Action Expert 36 层 attention 的 q/k/v/o 投影，共 144 个线性层；同时训练 state/action/time 投影。VLM 与动作专家其他主权重冻结。

```bash
# 全部任务训练
bash train.sh --config configs/train_lora.yaml

# 已准备单任务数据时，先验证一次真实更新
bash train.sh --config configs/train_lora.yaml --allow-subset \
  --max-steps 1 --gradient-accumulation 1 \
  --output-dir runs/lora-one-step

# batch 8，累积 4 次，执行 10,000 个小 batch
bash train.sh --config configs/train_lora.yaml \
  --batch-size 8 --gradient-accumulation 4 --max-steps 2500 \
  --output-dir runs/lora-b8-ga4
```

最后一条命令需要足够显存，使用任务子集时同样加 `--allow-subset`。CPU 计算线程数可通过 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 调整；显存与吞吐应以实际配置的短测为准。

### 3.3 Action Expert

配置文件：[configs/train_expert.yaml](configs/train_expert.yaml)。

```yaml
mode: expert
expert_last_n_layers: 2
train_projections: true
lr: 0.00005
output_dir: runs/expert_partial_clean
```

默认训练最后 2/36 层的全部参数，包括 attention、MoE 和层归一化，并训练动作专家末层 norm 及 state/action/time 投影。`expert_last_n_layers` 可设为 `1`–`36`，或用 `-1` 选择全部动作专家；VLM 始终冻结。

```bash
# 默认：末两层 Action Expert
bash train.sh --config configs/train_expert.yaml

# 单任务部分 Expert 训练
bash train.sh --config configs/train_expert.yaml --allow-subset \
  --output-dir runs/expert-subset

# 更大显存服务器：全部 Action Expert
bash train.sh --config configs/train_expert.yaml \
  --expert-last-n-layers -1 --output-dir runs/expert-full-clean

# 不加载 6B 权重，检查参数量及显存下界
bash train.sh --config configs/train_expert.yaml --device cpu --dry-run
```

`--dry-run` 需要已安装的源码与 Qwen 配置资源，其显存下界不包含激活、KV cache 和临时工作区。全部 Action Expert 的 FP32 梯度及 AdamW 状态需求较大，当前预设面向本地部分层微调；训练入口会在权重和优化器状态下界超过可用显存时拒绝启动。当前入口为单 GPU，未实现 DDP/FSDP 训练。

### 3.4 保存、恢复与导出

```text
runs/<experiment>/
├── training_config.yaml / lingbotvla_cli.yaml
├── run_metadata.json / episode_splits.json
├── assets/                  # 数据索引、归一化和处理器资源
├── configs/robot_configs/robotwin.yaml
├── metrics.jsonl / latest_checkpoint.txt
└── checkpoints/
    ├── global_step_100/training.pt
    └── global_step_1000/
        ├── training.pt
        └── hf_ckpt/         # 结束时导出的完整权重，LoRA 已合并
```

按 `save_every` 保存 `training.pt`；正常完成训练且 `export: true` 时，在最终 step 目录导出 `hf_ckpt/`。恢复使用 `training.pt`，推理使用 `hf_ckpt/`。

```bash
bash train.sh --config configs/train_lora.yaml \
  --resume runs/lora_clean/checkpoints/global_step_100/training.pt
```

恢复时保持基模、数据、归一化、可训练参数、输入几何、batch、累积次数、随机种子及尾部策略一致；可以调整总更新次数和保存间隔。修复前缺少新损失及采样元数据的 checkpoint 不支持新规则下的精确续训。不同配置的实验使用不同 `output_dir`。

要取消验证留出，先重新计算对应的归一化统计，再训练：

```bash
python scripts/prepare_data.py --tasks all --audit-only --val-episodes 0
bash train.sh --config configs/train_lora.yaml --val-episodes 0 \
  --output-dir runs/lora-all-clean
```

### 3.5 流程检查与推理

无需 6B 权重的合成训练、保存、恢复及 LoRA 合并检查：

```bash
bash train.sh --smoke --device cpu --output-dir artifacts/training-smoke
python -m pytest -q
```

对导出模型进行离线动作预测，输入包含示范中的 RGB、状态和指令，输出留出轨迹的动作误差：

```bash
python scripts/open_loop.py \
  --checkpoint runs/lora-one-step/checkpoints/global_step_1/hf_ckpt
```

仿真环境就绪后进行闭环检查：

```bash
bash eval.sh --checkpoint runs/lora-one-step/checkpoints/global_step_1/hf_ckpt \
  --tasks adjust_bottle --setting clean --trials 2 --precision bf16 \
  --use-length 10
```

默认每次生成 50 条动作，`--use-length 10` 返回并执行前 10 条，然后获取新观测重新规划。离线动作误差用于预测诊断，闭环结果用于衡量实际任务成功率。评估入口使用命令行参数；训练使用 YAML。
