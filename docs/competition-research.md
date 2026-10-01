# Competition requirements verified on 2026-10-01

## Authoritative competition scope

Tianchi competition 532514 uses **LingBot-VLA 2.0**, starting from
`robbyant/lingbot-vla-v2-6b`. The v1 4B model and the officially post-trained
`lingbot-vla-v2-6b-robotwin` checkpoint are not eligible training starts.

Training uses RoboTwin 2.0 **Aloha-AgileX**, all **50 tasks**, **50 clean
demonstrations per task**: 2,500 demonstrations total. Randomized data is test-only.
Evaluate the same 50 tasks under clean and randomized conditions, targeting
100 trials per task per setting: 10,000 trials total.

The submission is a ZIP containing evaluation JSON, source/configuration,
`train.sh`, `eval.sh`, the corresponding checkpoint and a reproduction report.
The deadline is **2026-10-26 24:00, UTC+08:00**. The user subsequently supplied
the official result/report template archive; its extracted files and archive
checksum are preserved in `docs/official-templates/`.

Sources: [competition information](https://tianchi.aliyun.com/competition/entrance/532514/information),
[designated base weights](https://huggingface.co/robbyant/lingbot-vla-v2-6b),
[official dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0).

## Training and inference boundaries

The official rule explainer explicitly requires **behavior cloning** from clean
demonstrations. **Both online and offline RL are prohibited**, including learned
value/advantage weighting, reward construction and bootstrapping. Additional
simulator rollouts, rendered observations, external data, generated language,
unseen instructions, HER and official RoboTwin checkpoint distillation are also
prohibited.

Allowed approaches include LoRA/expert/full supervised adaptation, visual/state
noise, demonstration-statistic filtering or weighting, replacement within the
released seen-instruction pool, training EMA, and distillation from the base or
the participant's own eligible model. LoRA must be merged for the submitted
full checkpoint, with implementation documented.

Final inference uses **one jointly trained weight set**. Task/instruction-based
deterministic routing, multiple checkpoint ensembles, multi-seed voting and
failure-triggered episode resets are prohibited. Within one model, documented
chunk-length changes, replanning, smoothing, precision changes, temporal output
averaging and confidence gating using its own outputs are permitted.

RoboTwin `check_success()` defines success. Official reevaluation uses the
submitted `eval.sh` inference settings.

Source: [official rule explainer](https://tianchi.aliyun.com/forum/post/1081761).

## Evaluation details from the official Q&A

- `clean2clean` uses **seen** instructions; `clean2random` uses **unseen** instructions.
- Submit one checkpoint cotrained on all 50 tasks. Partial trial counts may be
  submitted when necessary; official reevaluation determines the final score.
- Expert collection is limited to **five initiated attempts**, regardless of
  whether seeds change. After reaching that cap, continue policy evaluation and
  record the result; do not silently discard that scene.
- Preliminary submissions are limited to **three in total**, not three per day.
- CUDA/CuRobo and ROCm/MPLib results are acceptable with reproducible environment
  documentation. Preliminary inference time and VRAM have no stated cap.

Source: [official Q&A](https://tianchi.aliyun.com/competition/entrance/532514/customize898).

## Official software and comparison baseline

The maintained codebase is [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2).
Its setup specifies Python 3.12 and PyTorch 2.8.0. The backbone is Qwen3-VL-4B;
the full auxiliary-training recipe also needs MoGe-2, LingBot-Depth and DINO-Video
teacher assets. These are separate from the designated VLA checkpoint.

The released **clean + randomized** post-training recipe reports **93.52% clean /
92.80% randomized** with Muon. This is a comparison target, not an eligible
initial checkpoint or a clean-only competition baseline. A verified official
clean-only baseline was not located.

Source: [official model README](https://github.com/Robbyant/lingbot-vla-v2#readme).

## Dataset, simulator and hardware implications

The official setup guide pins RoboTwin to
`13c3c47ff4312dd62484bcd51be034af55c062d1`. It converts the released raw clean
episodes using RoboTwin's pi0 processor and then generates LeRobot v2.1 data.
The sim and inference environments should remain separate; sim dependencies
expect NumPy 1.26.x.

The guide reports roughly 32 GB for an FP32 inference server plus simulator;
24 GB generally needs BF16. BF16 can materially change success rates. A local
BF16 smoke result should record precision and should not be presented as an
FP32 reproduction. Actual viability on 16/24 GB remains subject to measurement.

Source: [official RoboTwin preparation/evaluation guide](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/README.md).

The upstream [training manifest](https://github.com/Robbyant/lingbot-vla-v2/blob/main/assets/training_data/robotwin.txt)
mixes clean and randomized and even contains a `piper_clean_50` entry. Do not
reuse it for competition training. Generate and audit an Aloha-AgileX clean-only
manifest and recompute normalization from those same demonstrations.
The 50 canonical task names are embedded in the
[official evaluation launcher](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/start_robotwin_infer_and_eval.sh).

## Verification method and unresolved details

The main competition page is client-rendered. Requirements were cross-checked
through the unauthenticated, public endpoints used by its frontend:

- [competition detail API](https://tianchi.aliyun.com/v3/proxy/competition/api/race/getDetail?raceId=532514)
  exposes `race.information`, `race.introduction` and Q&A tab 898.
- [rule explainer API](https://tianchi.aliyun.com/forum/api/forum/post/details?postId=1081761&seo=false)
  exposes the full explainer although the generic web reader displayed a 404.
- [information-file listing](https://tianchi.aliyun.com/v3/proxy/competition/api/race/queryInformation?raceId=532514)
  showed a masked 6 KB ZIP and no download URL without registration.

No authenticated accounts were accessed. The user subsequently supplied the
official ZIP templates, removing the submission-schema access blocker.
Competition-specific scene seeds, exact settings weighting and any newer
organizer notices remain unverified. Use the preserved official JSON template
when preparing final submission materials.
