# 代码材料目录要求

请将与初赛评测结果对应的实现代码放入 `02_代码材料/` 目录，确保主办方可以根据提交材料复现训练与评测流程。

建议目录结构：

```text
02_代码材料/
├── README.md
├── train.sh
├── eval.sh
├── configs/
└── src/
```

## 提交要求

1. `README.md` 需说明代码运行方式、依赖安装方式和主要目录结构。
2. `train.sh` 需支持使用 RoboTwin 2.0 Aloha-AgileX 50 个任务 clean 数据启动训练。
3. `eval.sh` 需支持加载提交的 checkpoint，并完成 clean / randomized 两个 setting 下的评测。
4. `configs/` 需包含训练和评测使用的配置文件，包括模型配置、训练配置和 robot config。
5. 如修改 RoboTwin 官方评测代码，需在 README 中说明修改内容和对应文件位置。

