# Courtroom Simulator — Copyright Infringement Detection

This repository contains a LVLM-based courtroom simulator that evaluates whether a generated (accused) image infringes a copyrighted (original) image. The system coordinates three types of agents — Expert, two Lawyers (Plaintiff and Defendant), and a Judge — to simulate courtroom deliberations and produce a final decision.

## 1. 功能概述
Courtroom Simulator 使用视觉-语言大模型（LVLM）完成以下任务：

- Expert: 执行抽象 (abstraction) → 过滤 (filtration) → 比较 (comparison) 链条，输出 `Judgment(score, confidence, rationale)`。
- Lawyers: 原告/被告律师根据专家分析生成开庭陈述、反驳及对法官提问的简短回答。
- Judge: 根据专家与律师陈述进行迭代评估，可直接判决（Verdict）或提出澄清问题（Question），最终输出 `FinalDecision`；若循环结束仍无定论，回退到专家判断。
- 支持批量评估（`batch_run_trials`），并保存每案详细结果和汇总指标。

## 2. 主要组件与架构

- `LLMAdapterAgent`: LVLM 适配器，封装不同后端（`gpt-4o`、`qwen2.5-vl`、`qwen3-vl-plus`）的接口，提供 `_chat`、`_abstraction_chat`、`_filtering_chat`、`_debate_chat` 等。
- `ExpertAgent`: 运行 AF-C（abstraction → filtration → comparison），并维护专家知识库 `K_e = {'E': [...], 'C': [...]}`（存储可信 AF-C 结果）。提供 `reflect_and_summary()` 将反思写入知识库。
- `LawyerAgent`: 原告/被告律师，提供 `opening_statement()`、`rebuttal()`、`answer_question()`，并维护律师知识库 `K_l = {'E_l': [...], 'C': [...]}`，以及 `reflect_and_summary()`。
- `JudgeAgent`: 法官，提供 `evaluate()`（返回 `verdict` 或 `question`）与 `run_trial()`（控制庭审循环、最终决策与回退规则），维护 `K_j = {'E_j': [...], 'C': [...]}` 并实现 `reflect_and_summary()`。
- `DebateCoordinator` / `MetaJudge` / `CopyJudgeAFC`: 用于多代理辩论、元判决与不同消融实验流程。

流程简述：
1. Expert 分析两张图片，生成抽象、过滤结果与比较判断。若置信足够高则写入 Expert 知识库。
2. 律师根据 Expert 的结果生成开庭陈述。
3. Judge 对陈述与 Expert 的过滤结果进行评估：若直接判决且置信高则结束；否则可能提出问题或让双方反驳，循环至 `max_rounds`。在最后一轮后会进行一次强制重评。
4. 若最终仍无判决，回退到 Expert 的 judgment 作为决策依据。
5. 各智能体在案件结束前后会调用 `reflect_and_summary()` 把经验写入各自知识库。

## 3. PromptTemplates（关键提示词）

定义在 `PromptTemplates` 中的模板包括：

- `ABSTRACTION`: 对两张图像进行抽象分解（构图、主题、色彩等）。
- `FILTRATION`: 标识不可受著作权保护的元素并提取原创元素。
- `COMPARISON`: 基于过滤后的唯一元素给出 `Score: [0-1], Confidence: [0-1], Reason: [text]`。
- `FEEDBACK`: 辩论轮次的同行反馈格式。
- `META`: 元法官用于汇总多个判决与按 LVLM 生成最终判决的模板。

Judge 的 `evaluate()` 使用严格的输出格式（两种 action：`Verdict` 或 `Question`），代码中实现了解析器以从 judge 输出中提取结构化字段。

## 4. 配置说明（`config.json` 关键信息）

- `agent_type`: 模型后端（如 `qwen2.5-vl`，`gpt-4o` 等）。
- `qwen_local_path`: 本地 Qwen 模型目录（当使用本地模型时）。
- `api_key_env`: 远端模型的 API Key 环境变量名（如 `OPENAI_API_KEY`）。
- `device`: 运行设备（如 `cuda:3` 或 `cpu`）。
- `temperature`, `max_tokens`: 模型参数。
- `meta_mode`: 元判决模式（`weighted`、`llm`、`lvlm`）。
- `debate_agent_num`: 辩论代理数量。
- `ablation`: 实验消融配置（例如 `LVLM+AFC+MAD+DEM`）。
- `gamma`: 判定阈值。
- `max_rounds`: 庭审轮数上限（默认 3）。
- `judge_confidence_threshold`: 法官做最终判定所需置信度（默认 0.75）。
- 数据路径：`dataset_dir`, `test_set_path`, `test_label_path` 等。

示例在仓库 `config.json` 中可见。

## 5. 运行方式

前提：
- 安装依赖（torch、transformers、Pillow 等）；
- 若使用本地 Qwen 模型，确保 `qwen_local_path` 指向正确模型；若使用远端模型，设置 API Key 环境变量。
- 准备数据集结构：`{dataset_dir}/Test/gen/gen_<image_name>.jpg` 和 `.../Test/real/real_<image_name>.jpg`。

命令示例：

单例演示（使用 `config.json` 中的 demo 路径）：
```bash
python3 courtroom_simulator.py --cfg_path /path/to/config.json
```

批量运行（当 `test_set_path` 与 `dataset_dir` 在配置中被设置）：
```bash
python3 courtroom_simulator.py --cfg_path /data1/humw/Codes/Image_Copy_Detection/Copyright_Infringement_Detection/config.json
```

运行结果（batch 模式）会输出到 `./outputs/<timestamp>_...` 目录下，包含：

- `detailed_results.json`: 每案详细记录（专家输出、最终判决等）；
- `final_results.json`: 简洁映射 `image_name` → 0/1（是否侵权）；
- `metrics.txt`: 若提供 `test_label_path`，则计算并写入 TP/TN/FP/FN、Accuracy、Precision、Recall、F1 等指标；否则写明缺少标签文件。

## 6. 扩展与注意事项

1. 知识库持久化：当前 `K_e` / `K_l` / `K_j` 仅保存在内存。建议将其序列化到磁盘（JSON/SQLite）并实现检索接口以支持长期学习。
2. 输出解析：Judge 要求严格格式，但仍有回退解析。生产化时建议强化 prompt 与解析的鲁棒性（或使用结构化输出 API）。
3. 性能：本地 Qwen 模型需要足够显存，`GLOBAL_QWEN_MODEL` 在初始化时会被加载一次并复用。
4. 测试：建议加入 MockAgent（覆盖 `_chat` 等方法）以便单元测试控制流与逻辑，而不依赖真实 LVLM。

## 7. 后续建议（我可以帮你做的事）

- 将这份文档保存为独立文件 `docs/COURTROOM_SIMULATOR.md`；
- 为 `LLMAdapterAgent` 添加 Mock 子类并编写一个快速测试脚本以在不调用真实模型的情况下验证 `run_trial`、知识库写入与 `batch_run_trials` 输出格式；
- 实现知识库的磁盘持久化接口（可选 JSON 或 SQLite）。

如果你希望，我可以直接在仓库中创建 `docs/COURTROOM_SIMULATOR.md` 或生成 Mock 测试脚本并运行一次验证。告诉我接下来想要哪一步。

