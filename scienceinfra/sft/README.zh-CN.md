# 多轮 SFT 轨迹采集与导出（实验性）

[English](README.md)

本模块复用 ScienceInfra 科学任务、Harbor **Terminus-2** 和任务 verifier，补充
采集调度、对话与 ATIF 对齐、原生 reasoning 保存、源码重放、监督掩码、Qwen 标签
和统计导出。这是轨迹导出与 SFT 数据准备的初步实现，不执行 SFT/RL 训练，也没有
接入 PSRL 的在线策略更新。

模型推理在远程 API 服务端；科学代码和 verifier 在执行主机的 CPU Docker 中运行。
不需要本地模型、GPU、PSRL 或 vLLM。

## 支持范围

| 组件 | 约定 |
|---|---|
| 任务 | 内置 `demo` 的 LAPS CPU `repair/easy` |
| 任务参考提交 | `49bd221bb18077a349e7f860569c3c5f5a4de64f` |
| Harbor | 0.23.0，提交 `3c82380859d187957cfd5cd64802b076d9779550` |
| 示例教师 | 通过 OpenAI 兼容 HTTPS API 调用 GLM-5.3，原生 `reasoning_content` |
| 执行环境 | Linux amd64、Python 3.12+、Docker Engine、Compose、Buildx |
| 源码重放 | 固定 LAPS `solve.sh` 配方，源文件集合保持不变 |
| Qwen 编码 | 不可变 tokenizer 提交，模板必须保留历史 thinking |

参考提交描述兼容基线，不代表修改后的仓库仍等于该提交。计划记录所有 SFT Python
模块和任务文件的哈希；应额外归档 Git 提交、diff 与依赖清单。API 别名不能证明
服务端权重版本，因此导出记录 `teacher_revision=unattested`。

新增/删除源文件、修改构建系统、其他科学环境需要新的重放 adapter。这里没有迁入
此前 Apple Silicon 镜像适配。压缩、分叉、续接和无法严格对齐的对话会被拒绝，
不补写缺失回合，不从最终答案生成 CoT。

## 1. 安装

先检出包含本模块的仓库版本。以下 Bash 命令从仓库根目录执行：

```bash
export SCIENCEINFRA_ROOT="$PWD"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'harbor @ git+https://github.com/harbor-framework/harbor.git@3c82380859d187957cfd5cd64802b076d9779550'
python -m pip install -e '.[sft]'
python -m pip check
python -m scienceinfra.sft.rollout doctor
```

按 [Docker 安装说明](https://docs.docker.com/engine/install/)配置 Docker 并授予执行
用户使用权限。`doctor` 检查 Python 导入与 Docker，不调用教师。镜像构建阶段需要
下载固定科学源码和构建依赖；本工具复制任务后将 agent 运行阶段设为 `no-network`，
宿主进程仍可连接模型 API。

## 2. 从源码生成任务和 Parquet

```bash
bash scripts/prepare/prepare_all.sh \
  --repo "$SCIENCEINFRA_ROOT/demo" --envs laps \
  --difficulty easy --categories repair \
  --data-root "$SCIENCEINFRA_ROOT/data" --stages compile,lines,dataset
export TASK_PARQUET="$SCIENCEINFRA_ROOT/data/laps/repair_easy/train/L2.parquet"
```

依次组装 Harbor task、解析源码行号、构建 Parquet。参考版本含 32 个 easy repair
候选，train 为 24、eval 为 8。采集使用 train，保留 eval 做评测；本模块不替任意
输入重新设计防泄漏划分。Parquet 含绝对任务路径，需在执行主机重新生成。

L2 提供缺陷文件和描述；L1 还使用解析出的行号；L3 不追加定位提示。实际模型输入
来自任务指令与 `extra_info.hint`，不是 Parquet 的 `prompt` 列。导出保留提示级别，
不能把 L2 结果称为无提示修复。详见[任务准备指南](../datasets/README.md)。

## 3. 单个任务运行

`python -m scienceinfra.sft.rollout xxx` 中的 `xxx` 是子命令，决定执行哪个阶段。
各子命令的含义如下；`doctor` 不需要 `--run-dir`，其余阶段使用同一个运行目录。

| `xxx` | 作用与主要产物 | 调用教师 API |
|---|---|---|
| `doctor` | 检查 Python、Harbor 和 Docker 等依赖；只检测服务，不启动任务或生成轨迹。 | 否 |
| `init` | 读取任务 Parquet，选择并复制任务包，保存模型、尝试次数、并发等配置及文件哈希到 `plan.json`；运行目录必须是新目录，不启动 Docker。 | 否 |
| `anchor` | 通过 Harbor/Docker 运行 oracle 官方修复和 nop 缺陷基线，要求修复奖励分别为 1 和 0、实测 floor 一致；保存验证记录和缺陷源码，不占教师尝试次数。 | 否 |
| `preflight` | 向计划中的模型发送一个小请求，检查正文和原生 `reasoning_content` 非空，保存回包和通过记录；这只是接口探测，不是任务轨迹，也不计入 SFT 样本。 | 是，有费用 |
| `rollout` | 按计划的任务、尝试次数和并发启动 Harbor Terminus-2，让教师多轮查看、修改、编译和测试代码；保存原始对话、ATIF、日志、源码产物和 verifier 结果，并支持已有运行对账。 | 是，有费用 |
| `review` | 核验科学得分、运行异常和对话/ATIF 对齐，确定助手回合的监督掩码；对候选源码执行独立 CPU Docker 重放，记录入库结论或排除原因。 | 否 |
| `export` | 重新核验已审查证据，将合格完整对话写入新 `--output` 目录的 `sft/train.jsonl`，同时导出尝试记录、统计与哈希清单；不启动 Docker，不生成新回答。 | 否 |

按下方示例依次执行 `init → anchor → preflight → rollout → review → export`。
`init` 中的 `--attempts 1` 只安排后续一次教师尝试，不会立即生成数据。
`export` 输出消息格式的 SFT 数据；Qwen token 标签由第 7 节的独立编码命令生成。

将 `TEACHER_API_BASE` 设为所用服务的 API 地址，`TEACHER_MODEL` 设为该服务接受的
模型 ID。`https://api.example.com/v1` 是占位地址，`GLM-5.3` 是通用示例，实际
模型 ID 以部署配置为准。`--api-base` 必须显式传入；模型 ID 无需额外添加
`openai/` 前缀，采集器会为 Harbor 自动添加该适配器前缀。

公开提交中应使用通用表述，不包含私有服务名称、地址、路由标识或密钥。运行计划和
收据会记录实际 API 配置，导出数据也保留教师模型 ID；发布生成产物前需检查这些
字段，原始证据应私下保留。

```bash
export TEACHER_API_BASE="https://api.example.com/v1"
export TEACHER_MODEL="GLM-5.3"
export RUN_DIR="$SCIENCEINFRA_ROOT/outputs/sft/smoke"
export RELEASE_DIR="$SCIENCEINFRA_ROOT/outputs/sft/smoke-release"
python -m scienceinfra.sft.rollout init \
  --dataset "$TASK_PARQUET" --run-dir "$RUN_DIR" \
  --limit 1 --attempts 1 --concurrency 1 \
  --api-base "$TEACHER_API_BASE" \
  --model "$TEACHER_MODEL"
python -m scienceinfra.sft.rollout anchor --run-dir "$RUN_DIR"
```

`anchor` 要求官方修复 oracle 的修复奖励为 1、缺陷基线 nop 的修复奖励为 0，且
实测 floor 一致，同时保存缺陷源码。此阶段无需 API key，不调用教师。

```bash
read -r -s -p 'Model API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
python -m scienceinfra.sft.rollout preflight --run-dir "$RUN_DIR"
python -m scienceinfra.sft.rollout rollout --run-dir "$RUN_DIR"
unset OPENAI_API_KEY
python -m scienceinfra.sft.rollout review --run-dir "$RUN_DIR"
python -m scienceinfra.sft.rollout export \
  --run-dir "$RUN_DIR" --output "$RELEASE_DIR"
```

`unset OPENAI_API_KEY` 是 shell 命令，不是 `rollout` 的子命令。它清除当前 shell
中的密钥变量，后续启动的进程不再继承该变量；不会撤销服务端密钥或删除运行记录。
采集完成后的 `review` 和 `export` 不需要教师 API key。

**preflight 和 rollout 会产生 API 费用。** preflight 要求实际回包包含非空原生
reasoning 和正文，成功只证明接口可用。review 使用 CPU Docker 重放源码，不调用
教师；export 离线执行。密钥通过环境变量传入，不写入参数和计划。

Terminus-2 使用 JSON 文本动作协议：终端反馈保留为 `user`，模型动作保留为
`assistant.content`，接口推理保存到 `reasoning_content`，不伪装为原生 function call。

## 4. 入库标准

1. Harbor trial 完成且无运行异常，结果与原始奖励一致，floor 与归一化有效。
2. 科学奖励、修复奖励、等价性及任务要求的检查全部通过，grading report 也通过。
3. 原始对话与 ATIF 的初始指令、助手正文、终端反馈逐项相等，无压缩/分叉/续接，
   结尾具有 Terminus-2 所需的两次完成确认。
4. 至少两轮可监督助手输出，其中至少一轮包含操作。缺失原生 reasoning、协议警告、
   显式终端错误等回合保留原文作为上下文，但标记 `loss=false`。
5. 模型源码相对缺陷基线改变。在全新缺陷环境中导入这些源码，移除官方修复，清理
   编译产物，使用固定构建与检查配方再次取得科学满分。

这是自动证据审查，不是科学专家逐条评审。命令匹配和错误掩码是保守启发式规则。
重放证明候选源码在固定配方下通过检查，不证明每步操作都必要，也不支持任意模型
改写的构建脚本。哈希用于检测相对已有收据的变化，不是防止整条证据链被重写的签名。

## 5. 多次尝试、并行与恢复

小规模试跑通过后，新建批次：

```bash
export RUN_DIR="$SCIENCEINFRA_ROOT/outputs/sft/batch-4x"
export RELEASE_DIR="$SCIENCEINFRA_ROOT/outputs/sft/batch-4x-release"
python -m scienceinfra.sft.rollout init \
  --dataset "$TASK_PARQUET" --run-dir "$RUN_DIR" \
  --limit 4 --attempts 4 --concurrency 2 \
  --api-base "$TEACHER_API_BASE" --model "$TEACHER_MODEL" \
  --max-turns 64 --max-output-tokens 8192 --context-tokens 65536
```

按第 3 节对新目录依次执行 anchor、输入 key、preflight、rollout、review、export。
这里安排 16 次独立尝试，rollout 最多并行两条；anchor 与 review 顺序执行。
科学编译自身也消耗 CPU/内存，应按主机资源配置。`context-tokens` 是模型元数据，
不是严格保证的输入 token 上限。

- 原 run 再执行 rollout 可对账已有结果；原进程尚在运行时拒绝重复启动。中断尝试
  保留并消耗槽位，不自动再付费重试。
- Harbor trial 和模型传输失败的自动重试关闭；harness 协议恢复仍可能产生额外调用。
- 认证、模型、配额等错误生成 `api_blocker.json` 并停止尚未启动的工作，已运行的
  尝试可能继续结束。排查后先将 blocker 文件归档到历史目录，再恢复运行。
- 显式重试失败的 CPU 重放：
  `python -m scienceinfra.sft.rollout review --run-dir "$RUN_DIR" --retry-replay`。
  旧重放和审查记录会归档，不再请求教师。
- 修改配置、任务或 SFT 模块代码时新建 run；每次 export 使用新目录。没有合格数据
  时保留排除报告并返回退出码 2，不生成虚构样本。

## 6. 交付文件与统计

run 保存计划、任务副本、oracle/nop、接口探测、各次尝试的 Harbor 原始对话、ATIF、
终端记录、源码及独立重放。release 包含：

```text
sft/train.jsonl       # 一行一条完整多轮轨迹
attempts.jsonl        # 所有计划槽位，包括失败与未开始的尝试
tasks.jsonl
evidence.json
statistics.json
manifest.json
```

统计分别报告科学成功率、SFT 入库率、独立任务数、排除原因、交互轮数分布。成功率
分母为已结束尝试。user 回合包含初始任务及自动终端反馈，不是人工发言次数。
API token 是包含重复上下文的请求累计量，不是最终 SFT 序列长度，不包含 preflight。
交付时保留原 run 供审计；生成物可能含源码、机器路径与模型回包，本 PR 不含历史数据。

## 7. Qwen 标签

为目标 tokenizer 选择已核验的不可变 40 位 Hugging Face 提交：

```bash
python -m pip install -e '.[sft-qwen]'
export QWEN_MODEL='Qwen/Qwen3.8-27B'
export QWEN_REV='<填写已核验的40位tokenizer提交>'
python -m scienceinfra.sft.tokenize_qwen \
  --release-dir "$RELEASE_DIR" \
  --output-dir "$SCIENCEINFRA_ROOT/outputs/sft/batch-4x-qwen" \
  --model "$QWEN_MODEL" --revision "$QWEN_REV" --max-tokens 65536
```

仅下载 tokenizer，不下载权重。已缓存同一提交的文件时可加 `--local-files-only`。
编码验证 release 哈希、模板与 token 前缀边界，保留所有助手回合的原生 reasoning，
仅监督已审查助手区间，其他 token 的 `labels=-100`。超长完整轨迹被排除而不截断；
不兼容模板会失败，不静默丢弃历史 thinking。

输出包括 token 版 `train.jsonl`、tokenizer 文件和 `report.json`，记录序列/监督 token
统计、排除原因、版本与哈希。下游训练器需保留 labels、使用兼容 collator 并单独验证。
本 PR 完成数据准备，不包含训练器集成；教师示范也不是 PSRL 在线 GRPO 所需的
on-policy token/logprob 数据。

## 开发与验证

```bash
python -m pip install ruff==0.15.6
python -m unittest discover -s tests/sft -p 'test_*.py' -v
ruff check scienceinfra/sft tests/sft
ruff format --check scienceinfra/sft tests/sft
```

测试使用合成对话和证据，mock 重放执行，不启动 Docker、不调用教师。
可通过 `SFT_QWEN_TOKENIZER_DIR` 启用缓存真实 tokenizer 的离线检查。
实际完成与尚未完成的验证见 [VALIDATION.md](VALIDATION.md)。
