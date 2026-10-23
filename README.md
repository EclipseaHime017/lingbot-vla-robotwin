# LingBot-VLA 2.0 × RoboTwin

基于 `robbyant/lingbot-vla-v2-6b` 的后训练工程，支持 RoboTwin 2.0 Aloha-AgileX clean 示范数据的下载、转换、训练和动作预测评估。模型以三路 RGB、关节状态和任务指令为输入，通过 flow-matching 行为克隆（BC）学习动作块。

当前提供两种训练方式：action attention 的 **LoRA** 微调，以及 **Action Expert** 的部分层或全部层微调。两种方式都冻结视觉语言模型（VLM），支持梯度累积、梯度检查点、训练状态恢复和完整推理权重导出；完整 Expert 支持多卡 FSDP 训练。导出时自动合并 LoRA。仓库另提供离线动作预测和 RoboTwin 闭环仿真入口。

## 1. 仓库框架

```text
lingbot-vla-robotwin/
├── configs/                 # 训练 YAML、任务列表、原始数据哈希清单
│   ├── train_lora.yaml
│   ├── train_expert.yaml
│   ├── train_expert_full.yaml # 多卡完整 Action Expert，VLM 冻结
│   └── eval.yaml             # 后端、精度、动作长度、去噪步数
├── src/lrvla/               # 数据加载、模型适配、训练、checkpoint 和推理
├── scripts/                 # 环境安装、资源下载、数据转换及诊断入口
│   └── run_with_batch_fallback.py # CUDA OOM 后降低 batch 重试的多卡入口
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
├── requirements-visualization.txt  # 可选的数据集查看器
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
cd /path/to/lingbot-vla-robotwin
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

可选的 Rerun 数据集查看器依赖单列在 `requirements-visualization.txt`，本地默认安装，可设置 `INSTALL_RERUN=0` 跳过。训练、数据转换和动作预测不使用该查看器。

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

闭环运行需要 SAPIEN 相机的 GPU Vulkan 渲染，即使不显示窗口，也需要生成 RGB 观测。WSL 的 CUDA 训练与 Vulkan 相机渲染支持需分别检查；闭环运行应使用具备 NVIDIA Vulkan 渲染能力的 Linux 环境，并先通过环境诊断。

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
| `validation_frames_per_episode` | `1` | 每条留出轨迹固定抽取的验证帧数 |
| `resample_seen_instructions` | `false` | 是否从同轨迹 seen 指令池确定性重采样 |
| `sequence_wise_loss_coeff` / `router_z_loss_coeff` | `0.001` / `0.0001` | MoE 负载均衡与路由 z-loss 系数 |
| `export` / `export_dtype` | `true` / `float32` | 结束后导出完整推理权重及其精度 |

有效 batch 约为 `batch_size × gradient_accumulation`。例如 `batch_size=8`、累积 4 次、`max_steps=2500`，共执行 **10,000 个小 batch、2,500 次参数更新**。这里的 step、预热、验证和保存间隔都按优化器更新计数。

`action_chunk_tail: mask` 保留轨迹末端观测，padding 不参与损失；累积梯度按整个更新窗口的有效动作坐标数统一归一化。`drop` 只保留训练集的完整动作窗口，验证集仍保留全部观测。截断会减少末段观测作为训练起点的覆盖。

训练目标为有效动作坐标上的 flow-matching BC 均值，加样本均值归一化的 MoE sequence balance 和 router z-loss；两者分别在整个更新窗口及全部 GPU 上归一化。日志分别记录 `bc_loss`、`moe_balance_loss`、`router_z_loss` 和每层全局专家负载。LoRA 的路由器虽然冻结，辅助损失仍能通过 hidden states 更新 LoRA；将两个系数都设为 `0` 可运行纯动作 BC。

验证按每个任务的每条留出轨迹分层抽帧，50 个任务、每任务 5 条留出轨迹时默认评估 250 帧；旧 `eval_batches` 不再截断验证集。样本列表保存在 `validation_samples.json`，日志记录覆盖范围及各任务 BC 损失。验证的 flow 噪声和时间按样本身份固定，与 batch 和 GPU 数无关。验证始终关闭图像增强和指令重采样。

默认仍使用转换时固定的 seen 指令。要启用同轨迹指令重采样，先生成旁路文件，无需重新编码视频：

```bash
python scripts/prepare_seen_instructions.py --manifest data/clean_manifest.json
```

随后在训练 YAML 设置 `resample_seen_instructions: true`、`seen_instructions: data/clean_seen_instructions.json`。仅从当前轨迹的 seen 列表取指令，按 seed、epoch、任务、轨迹和帧确定性选择，不增加样本数。旁路文件内容哈希及增强参数纳入续训校验。

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

`--dry-run` 需要已安装的源码与 Qwen 配置资源，其显存下界不包含激活、KV cache 和临时工作区。全部 Action Expert 的 FP32 梯度及 AdamW 状态需求较大，当前预设面向本地部分层微调；训练入口会在权重和优化器状态下界超过可用显存时拒绝启动。单 GPU 可训练部分 Expert；全部 Expert 的多卡 FSDP 配置与启动方式见 3.6。

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

默认导出 FP32：冻结部分读取 SHA-256 匹配的原始基模，训练部分使用 FP32 主权重，LoRA 用原始 FP32 base 加 FP32 adapter 增量合并。训练仍使用冻结 BF16 权重与 BF16 autocast，可训练参数从原始 FP32 基模初始化。`export_dtype: bfloat16` 可生成较小的导出，但事后上转 FP32 无法恢复舍入精度。

已有训练状态可在 CPU 上另行导出，不加载优化器、不要求原训练 GPU 数，也不覆盖已有导出：

```bash
python scripts/export_checkpoint.py \
  --checkpoint runs/expert_full_clean/checkpoints/global_step_10000/distributed \
  --base-checkpoint models/lingbot-vla-v2-6b
```

默认生成相邻的 `hf_ckpt_float32/`；单卡将 `--checkpoint` 指向 `training.pt`。保留同一 run 的处理器、配置和归一化资源。只有 BF16 推理文件而没有 FP32 训练状态时，无法恢复训练部分的原始精度。

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
  --tasks adjust_bottle --setting demo_clean --trials 2 --precision bf16 \
  --backend streamed --use-length 10
```

默认每次生成 50 条动作，`--use-length 10` 返回并执行前 10 条，然后获取新观测重新规划。离线动作误差用于预测诊断，闭环结果用于衡量实际任务成功率。评测默认读取 [configs/eval.yaml](configs/eval.yaml)：原生后端、FP32、执行 50 条、去噪 10 步、关闭 compile、无动作平滑。这是可复现的基线配置，尚未调优；CLI 可覆盖 YAML，低显存检查可显式选择 `--backend streamed`。

对同一导出模型运行全部任务、两种环境各 100 个场景：

```bash
bash eval.sh --checkpoint runs/expert_full_clean/checkpoints/global_step_10000/hf_ckpt \
  --config configs/eval.yaml --tasks all --setting both --trials 100 \
  --output runs/evaluation/expert-full
```

这会运行 50 × 2 × 100 个场景，使用原生 FP32 推理，需要相应显存和仿真环境。结果写入输出目录的 `results.json`；添加 `--skip-completed` 可保留已完成任务，重新运行失败任务，并校验模型权重与评测协议一致。

多 GPU 入口按任务长度分组，用多个独立仿真进程并行运行不同的“任务 × 环境设定”。FP32 模型服务约占 25 GB 显存，单个仿真进程约 9 GB，因此 40 GB 的 A100 上，模型服务旁边只能再放一路仿真，两路会超出显存。2026-10-22 至 10-23，集群四卡节点上的并行评测共 4 次使计算节点宕机；最后一次使用下述分卡方式，每卡显存仅 24.9 GB，说明宕机并非显存超限，原因尚未查明。查明之前不要在集群上运行多卡并行评测。两份并行配置都使用 `sim_layout` 按角色分卡：每个四卡节点用 2 张卡各运行一份模型服务和一路仿真，另 2 张卡只运行仿真（每卡 3 路），每份服务对应 4 路仿真，单节点共 8 路并发。仿真进程通过本机 WebSocket 访问模型服务，渲染与运动规划绑定在各自的卡上。[configs/eval_parallel.yaml](configs/eval_parallel.yaml) 覆盖全部 10,000 场景；[configs/eval_proxy.yaml](configs/eval_proxy.yaml) 覆盖 12 个任务、两种设定、各 20 场景，共 480 场景。代理结果用于模型筛选，完整结果仍需运行全部任务。

```bash
# 仅生成分组与参数清单，不加载模型或启动仿真
python scripts/evaluate_parallel.py --config configs/eval_proxy.yaml \
  --checkpoint runs/expert_full_clean/checkpoints/global_step_10000/hf_ckpt \
  --gpus 0,1,2,3 \
  --output runs/evaluation/proxy-layout --prepare-only

# 在已分配 GPU 的计算环境中运行，并复用完成的任务及环境组合
python scripts/evaluate_parallel.py \
  --workflow runs/evaluation/proxy-two-sims/workflow.json --gpus 0,1,2,3
```

省略 `--gpus` 时读取 `CUDA_VISIBLE_DEVICES`；已有设备分配时只能从中选择。入口将 CUDA 与 Vulkan 绑定到同一物理 GPU，每卡使用独立推理端口和输出目录。单卡也可用 `bash eval.sh ... --sim-workers 2 --policy-rng scene` 共享服务。每个任务仍按原顺序推进 seed，不将同一任务简单切成重复的 seed 段。仿真进程的 BLAS/OMP 线程数限制为 1；同卡的推理请求串行处理，CPU 仿真并行。

共享模式固定 `robotwin` 配置、整段动作返回和无动作平滑，禁止客户端切换模型。`scene` 随机协议按任务、设定和最终场景 seed 生成模型噪声种子，各连接独立保存 Torch CPU/CUDA 随机流，避免请求交错改变某场景的随机序列。该协议及缓存模式进入恢复校验，使用新的输出目录，不与旧的全局随机流结果混合。推理去噪步数和动作执行长度保持 YAML 中的配置。

每组目录保存 `results.json`，根目录汇总为全部分组的 `results.json`；只有日志场景数完整、客户端成功退出、模型身份和协议匹配才标记完成。`workers/parallel_status.json` 记录进程退出状态。每回合输出 `LRVLA_EPISODE_METRICS`，区分专家预跑、策略初始化、语言生成、取观测、推理 RPC 和动作执行耗时；多仿真时 `resource_usage.jsonl` 每五秒记录 GPU 利用率和显存，采样值可能漏过短时峰值。

每卡两路、四路的显存及吞吐需要在目标硬件上实测；单服务加一套仿真的占用不能直接推算额外三套仿真一定能容纳。先用 `--sim-workers 2`，测量后在新输出目录尝试 `4`。运行时间也取决于任务长度、共享服务等待和任务尾部，代理规模不保证在固定时限内完成。

可选的专家缓存目前提供实验性 `record/replay`，默认关闭。准备 workflow 时添加：

```bash
# 首次正常执行专家预检并记录；其余参数同上，使用新的输出目录
--expert-cache runs/evaluation/scene-bank --expert-cache-mode record

# 后续 checkpoint 在独立输出目录中重放同一组场景
--expert-cache runs/evaluation/scene-bank --expert-cache-mode replay
```

缓存保留每场景最多五次尝试、所有失败预算耗尽的场景、最终 seed 和完整专家物体／手臂描述。重放保留策略初始化、稳定性检查和完整 RGB 闭环，恢复 Python 指令生成状态，再执行原指令生成与 NumPy 抽样，并核对指令和首次官方 RGB／关节观测；不缓存模型动作或策略成功结果。身份包含任务、设定、场景数、起始 seed、协议、源码／配置摘要、仿真依赖版本及资产清单。小型物理和机器人配置按内容哈希，网格／纹理由大小和修改时间识别，后者不是完整内容校验。20 场景与 100 场景使用不同指令生成数量，缓存分别保存。

缓存首次构建仍需专家预跑；部分记录不支持续写，未完成时使用新缓存目录。重放尚需在目标仿真环境中完成真实渲染对照验证，因此其结果不会标记为可直接与完整基线比较。模型选择可以复用验证过的缓存，正式全量评测默认保持缓存关闭。

闭环客户端适配位于 [src/lrvla/evaluation.py](src/lrvla/evaluation.py)，运行时生成客户端副本，不修改上游文件：场景数参数化；移除未使用的 `get_model` 查找；每场景最多五次专家启动，失败后推进 seed，策略使用最后一次 seed；策略初始化的 `UnStableError` 计为一次失败并继续，其他错误仍终止。初始化失败也可能导致零次模型调用，环境预检还需检查真实 RGB 推理和动作日志。

专家元数据缺失时由 [src/lrvla/scene_metadata.py](src/lrvla/scene_metadata.py) 恢复场景物体描述，再使用原始指令模板。客户端以 `check_success()` 判断成功；上游 `take_action` 也在插值子步调用该函数，成功时设置 `eval_success` 并返回。客户端移除按任务切换权重的路径，全程使用同一 checkpoint。原生服务保留上游模型和内核，仅将配置的去噪步数设置到实际采样模型；后端、精度、去噪步数、执行长度、compile 和平滑均进入结果及恢复校验。新旧客户端协议使用不同输出目录。

### 3.6 多卡完整 Action Expert

`configs/train_expert_full.yaml` 使用 `mode: expert`、`expert_last_n_layers: -1`，训练全部 36 层 Expert 及状态／动作投影，冻结视觉语言 backbone。可训练参数约 17.89 亿。`torchrun` 启动时，训练入口使用 FSDP 分片 Expert 参数、梯度和 AdamW 状态；`fsdp_wrap: decoder_layer` 按解码层分别包装，将临时汇集的参数组由约 6.66 GiB 降到每层约 0.185 GiB。冻结参数保持 BF16，每张卡保留一份，Expert 的参数及优化器状态使用 FP32，计算使用 BF16 autocast。

当前预设面向四卡：每卡 batch 为 16，`global_batch_size: 64`，`gradient_accumulation: 1`，不做梯度累积；`require_no_gradient_accumulation: true` 防止修改每卡 batch 后自动引入累积。每项作业执行 1000 次优化器更新，合计约 64000 个训练样本。50 个 clean 任务按帧合并并随机打乱；每任务前 45 个 episode 训练，后 5 个验证。终止帧保留，填充动作不计入 BC 损失，MoE 辅助损失按全局样本数归一化。学习率为 `5e-5`，前 100 次更新预热，每 100 次更新验证和保存。

准备好全部 clean 数据及其 norm provenance 后，可直接用 `torchrun` 固定 batch 训练，或使用独立启动器在 CUDA 显存不足时从基模重新开始一次较小 batch 的训练。每卡 batch 与全局 batch 必须同时匹配 GPU 数。

手动固定四卡 batch 64：

```bash
source scripts/activate.sh
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train.py --config configs/train_expert_full.yaml \
  --batch-size 16 --global-batch-size 64 --gradient-accumulation 1 \
  --output-dir runs/expert4-fixed64
```

手动固定 batch 32 时，将上面的每卡 batch 改为 `--batch-size 8`，全局 batch 改为 `--global-batch-size 32`，并使用新的输出目录；`--gradient-accumulation 1` 保持不变。也可直接在训练 YAML 中同步设置 `batch_size: 8`、`global_batch_size: 32`、`gradient_accumulation: 1`，省略这三个命令行覆盖参数。命令行参数优先于训练 YAML。

自动在 CUDA 显存不足时由 batch 64 降为 32：

```bash
source scripts/activate.sh
python scripts/run_with_batch_fallback.py \
  --gpus 4 --config "$PWD/configs/train_expert_full.yaml" \
  --output-dir "$PWD/runs/expert4-auto64-32" \
  --global-batch-size 64 --fallback-global-batch-size 32
```

自动方案根据 GPU 数计算两次尝试的每卡 batch，并固定不累积，不能额外指定 `--batch-size`、`--gradient-accumulation` 或 `--resume`。附加的训练参数通过末尾的 `--` 传递，例如 `-- --max-steps 10000`。八卡固定 batch 64 对应每卡 batch 8、累积 1；直接训练时将 `--nproc-per-node` 改为 8，并同步设置每卡 batch。各次运行使用独立目录。

自动方案先使用真实全局 batch 64（每卡 16）。只有发生明确的 CUDA 显存不足时，启动器才终止本次训练的全部 rank，并从基模重新开始全局 batch 32（每卡 8），仍不累积。两个尝试分别写入 `batch64/` 和 `batch32/`，尝试记录位于 `attempts.json`；较小 batch 完成 1000 次更新时消费约 32000 个样本。非显存错误不会自动降低 batch，两种 batch 都显存不足时作业失败并保留日志。增加 batch 后的实际显存与吞吐以正式训练日志为准。

完整 Expert 的每次分片 checkpoint 总计约 20 GiB，应将 `runs/` 放在足够大的存储中。多卡 checkpoint 是包含 `shards/`、每 rank 随机状态及 `complete.json` 的目录；`latest_checkpoint.txt` 指向此目录。使用 `--resume <目录>` 恢复，要求相同 GPU 数和数据采样配置。训练结束后只由 rank 0 导出完整、可独立推理的 `hf_ckpt`，其中包含冻结 backbone 和训练后的 Expert。日志记录每张卡的实际分配／保留显存峰值、更新耗时及 checkpoint 耗时。

采样器在每个 epoch 中先全局随机排列所有训练帧，再分配给各 GPU，消费完当前排列后才进入下一轮。以 494328 个训练帧、全局 batch 64 为例，一轮需要 7724 次更新，末尾仅补齐 8 个重复帧；从基模训练 10000 次更新会完成全部训练样本的首轮遍历。日志的 `epoch` 和 `batch_offset` 可用于核对遍历进度。训练规模直接修改现有 `configs/train_expert_full.yaml` 的 `max_steps`，或用 `--max-steps` 覆盖。

10000 × 64 约等于 1.29 遍该训练集。覆盖全部样本不代表收敛；总步数应结合覆盖全任务的验证曲线和独立评估决定，仓库不自动延长训练或解冻 VLM。
