# 纯文本语言模型全量训练测试方案

> **目标**：在现有 500 步吞吐基准之外，对 4 个纯文模型跑**全量 finetune 训练**，
> 同时输出 CINN vs No-CINN 的 step time 与 dev 收敛指标（acc / ppl）。
> 严格按论文/官方 finetune 配方设置 batch / seq / lr / epochs。
>
> **本次跑两档**：
>
> - **核心档**：每模型 1 个论文主任务，CINN + No-CINN 各一次，**~7 h**
> - **完整档**：每模型多任务覆盖，CINN + No-CINN 各一次，**~20 h**

---

## 1. 模型与范围

| # | 模型 | 任务头 | 评测指标 | 范围 |
| --- | --- | --- | --- | --- |
| 1 | **Bert-base-uncased** | `BertForSequenceClassification` | dev_acc | ✅ CINN vs No-CINN |
| 2 | **Ernie-3.0-nano-zh** | `ErnieForSequenceClassification` | dev_acc | ✅ CINN vs No-CINN |
| 3 | **GPT-2-medium** | `GPTForSequenceClassification` | dev_acc | ✅ CINN vs No-CINN |
| 4 | **Small Llama 168M** | `LlamaForCausalLM` | dev_ppl = exp(loss) | ✅ CINN vs No-CINN |

> ⚠️ Llama2-7B 不在范围内：CINN OOM。Small Llama 168M（8L/1024H/8H）作为同结构替代。

---

## 2. 测试矩阵（核心档 + 完整档）

### 2.1 核心档（必跑）

每模型选 1 个论文主任务，CINN + No-CINN 各跑一次。

| 模型 | 数据集 | batch | seq | epochs | 总 step | lr | warmup | CINN+NoCINN 合计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bert-base-uncased | SST-2 | 32 | 128 | 3 | 6,315 | 2e-5 | 10% | ~28 min |
| Ernie-3.0-nano-zh | ChnSentiCorp | 32 | 128 | 3 | 900 | 5e-5 | 10% | ~2 min |
| GPT-2-medium | SST-2 | 16 | 128 | 3 | 12,630 | 2e-5 | 10% | ~160 min |
| Small Llama 168M | WikiText-103 (×1ep CLM) | 32 | 128 | 1 | ~44K | 3e-4 | 2000 step | ~220 min |
| **核心档总计** | | | | | | | | **~7 小时** |

### 2.2 完整档（在核心档基础上追加可选任务）

| 模型 | 任务集合 | 追加 step | 追加耗时 |
| --- | --- | --- | --- |
| Bert-base | SST-2 + MRPC + MNLI | +688 + 36,816 | +~3 h |
| Ernie-nano-zh | ChnSentiCorp + TNEWS + LCQMC | +5,003 + 22,367 | +~1.5 h |
| GPT-2-medium | SST-2 + MRPC | +688 | ~3 h |
| Small Llama 168M | WikiText-103 ×3ep + Dolly-15K SFT | +131K + 600 | +~12 h |
| **完整档总计**（含核心档） | | | **~20 小时** |

> 折算依据：本仓库 500 步实测（bs=1, seq=128）的 CINN/No-CINN 单步时间 × 6（batch 1→16~32 放大系数）。
> 单步详细估时见附录 A.7。

---

## 3. 数据集准备

| 数据集 | PaddleNLP 加载名 | 任务类型 | 字段 | 大小 (train/dev/test) |
| --- | --- | --- | --- | --- |
| SST-2 | `load_dataset('glue', name='sst2')` | 英文 2 分类 | `sentence`, `label` | 67,349 / 872 / 1,821 |
| MRPC | `load_dataset('glue', name='mrpc')` | 英文句对 2 分类 | `sentence1`, `sentence2`, `label` | 3,668 / 408 / 1,725 |
| MNLI | `load_dataset('glue', name='mnli')` | 英文句对 3 分类 | `premise`, `hypothesis`, `label` | 392,702 / 9,815 / 9,796 |
| ChnSentiCorp | `load_dataset('chnsenticorp')` | 中文 2 分类 | `text`, `label` | 9,600 / 1,200 / 1,200 |
| TNEWS | `load_dataset('clue', name='tnews')` | 中文 17 分类 | `text`, `label` | 53,360 / 10,000 / 10,000 |
| LCQMC | `load_dataset('lcqmc')` | 中文句对 2 分类 | `query`, `title`, `label` | 238,574 / 8,802 / 12,500 |
| WikiText-103 | `load_dataset('wikitext', name='wikitext-103-v1')` | 英文 CLM | `text`（长文档） | ~180M / ~4M / ~4M tokens |
| Dolly-15K | `load_dataset('databricks-dolly-15k')` | 英文 SFT | `instruction`, `context`, `response` | 15,000 / — / — |

**统一处理流程**：

```python
# Step 1: 加载
train_ds, dev_ds, *_ = paddlenlp.datasets.load_dataset('<name>', name='<subset>',
                                                        splits=['train', 'dev'])
# Step 2: tokenizer
tokenizer = AutoTokenizer.from_pretrained('<model_name>')
# Step 3a (clf): tokenize 单/双句, trunc/pad 到 seq_len
# Step 3b (CLM): 全量 token 拼接后用 group_texts 切成定长 [N, seq_len]
# Step 4: paddle.io.DataLoader(batch_size, shuffle, drop_last)
```

> 网络受限时设 `data_home=...` 走本地缓存。第一次加载下载到 `~/.paddlenlp/datasets/`，
> 单数据集 1 MB ~ 500 MB，建议提前 prefetch 全部 8 个数据集。

---

## 4. 脚本结构与运行选项

新增 1 个统一入口脚本 + 2 个任务实现脚本，**不动现有 4 个 500 步脚本**。

### 4.1 文件清单

```
paddle_tests/
├── benchmark_train_full.py            # 统一 CLI 入口（本次新增）
├── full_train_clf.py                  # clf 全量训练实现，覆盖 Bert/Ernie/GPT-2
├── full_train_clm.py                  # CLM/SFT 全量训练实现，覆盖 Small Llama
├── full_train_data.py                 # 数据集加载 + tokenize + DataLoader
├── full_train_utils.py                # to_cinn_net / 评测 / CSV / 画图
└── results_full/                      # 输出目录（CSV + PNG + log）
```

### 4.2 CLI 设计（核心需求）

统一入口 `benchmark_train_full.py` 支持**单模型跑** / **批量跑** / **核心档** / **完整档**：

```bash
# ===== 模式 1：跑单个模型 =====
python benchmark_train_full.py --model bert     # 仅 Bert，按 --suite 决定核心 or 完整档
python benchmark_train_full.py --model ernie
python benchmark_train_full.py --model gpt2
python benchmark_train_full.py --model llama    # = Small Llama 168M

# ===== 模式 2：批量跑（默认）=====
python benchmark_train_full.py --model all      # 4 个模型按矩阵全跑

# ===== 模式 3：套件选择 =====
python benchmark_train_full.py --model all --suite core      # 核心档（~7 h）
python benchmark_train_full.py --model all --suite full      # 完整档（~20 h）
python benchmark_train_full.py --model all --suite both      # 先核心档再追加完整档增量

# ===== 模式 4：CINN 开关 =====
python benchmark_train_full.py --model bert --cinn_mode both        # 默认，跑 CINN+NoCINN
python benchmark_train_full.py --model bert --cinn_mode cinn_only
python benchmark_train_full.py --model bert --cinn_mode nocinn_only

# ===== 模式 5：跑指定数据集（覆盖默认 suite）=====
python benchmark_train_full.py --model bert --datasets sst2,mrpc

# ===== 模式 6：调试 / sanity 短跑 =====
python benchmark_train_full.py --model ernie --max_train_steps 100 --eval_steps 50

# ===== 模式 7：续跑（断点）=====
python benchmark_train_full.py --model llama --resume results_full/llama_cinn_step_20000.pdparams
```

### 4.3 完整 CLI 参数表

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--model` | str | `all` | `bert` / `ernie` / `gpt2` / `llama` / `all` |
| `--suite` | str | `core` | `core` / `full` / `both` |
| `--cinn_mode` | str | `both` | `both` / `cinn_only` / `nocinn_only` |
| `--datasets` | str | (suite 自动) | 逗号分隔覆盖 suite 默认（如 `sst2,mrpc,mnli`） |
| `--batch_size` | int | (任务自动) | 覆盖论文默认 batch |
| `--seq_len` | int | (任务自动) | 覆盖论文默认 seq |
| `--epochs` | int | (任务自动) | 覆盖论文默认 epoch |
| `--lr` | float | (任务自动) | 覆盖论文默认 lr |
| `--warmup_ratio` | float | 0.1 | warmup step 比例 |
| `--max_train_steps` | int | -1 | 强制截断 step（调试用） |
| `--eval_steps` | int | (epoch 末) | 每 N step eval 一次；不传则只在每个 epoch 末 |
| `--log_interval` | int | 50 | 打印间隔 |
| `--seed` | int | 42 | 随机种子（保证 CINN/No-CINN 同初始化） |
| `--output_dir` | str | `./results_full` | CSV / PNG / log 输出目录 |
| `--resume` | str | None | 从 checkpoint 续跑 |
| `--device` | str | `gpu:0` | paddle 设备 |

### 4.4 默认 suite 解析（脚本里查表）

```python
SUITE = {
    'core': {
        'bert':  [('sst2', dict(bs=32, seq=128, epochs=3, lr=2e-5))],
        'ernie': [('chnsenticorp', dict(bs=32, seq=128, epochs=3, lr=5e-5))],
        'gpt2':  [('sst2', dict(bs=16, seq=128, epochs=3, lr=2e-5))],
        'llama': [('wikitext103', dict(bs=32, seq=128, epochs=1, lr=3e-4))],
    },
    'full': {
        'bert':  [('sst2',...), ('mrpc',...), ('mnli',...)],
        'ernie': [('chnsenticorp',...), ('tnews',...), ('lcqmc',...)],
        'gpt2':  [('sst2',...), ('mrpc',...)],
        'llama': [('wikitext103', dict(epochs=3,...)), ('dolly15k',...)],
    },
}
```

> `both` = 先 `core`，再增量补 `full \ core`，避免重跑核心档任务。

---

## 5. 单次实验的输出产物

每个 `(model, dataset, cinn_mode)` 组合产出：

```
results_full/
├── bert_sst2_cinn_steps.csv               # step, train_loss, train_acc, dev_loss, dev_acc, step_time_ms
├── bert_sst2_nocinn_steps.csv
├── bert_sst2_compare.png                  # 双子图：左 loss/acc 曲线，右 step_time
├── bert_sst2_summary.json                 # 最终 dev_acc, 平均 step_time, 加速比
└── bert_sst2_run.log                      # 完整 stdout
```

汇总：

```
results_full/
└── summary_all.csv     # 每行 (model, dataset, cinn, dev_metric, step_time, total_step, total_time)
```

---

## 6. 落地步骤

1. 实现 `full_train_data.py`（8 个数据集 + tokenizer + group_texts）。
2. 实现 `full_train_clf.py`（forward/eval/CSV，covers Bert/Ernie/GPT-2）。
3. 实现 `full_train_clm.py`（CLM forward + ppl，covers Small Llama）。
4. 实现 `benchmark_train_full.py` 统一 CLI（解析 `--model/--suite`，调度上面两个）。
5. **Sanity check**：`python benchmark_train_full.py --model ernie --max_train_steps 100`
   验证 Ernie ChnSentiCorp 100 步内 loss 下降。
6. **核心档**：`python benchmark_train_full.py --model all --suite core` （~7 h）。
7. 检查核心档 dev 指标进入论文区间（SST-2 Bert ≥90%、ChnSentiCorp Ernie ≥92%、WikiText ppl ≤50）。
8. **完整档增量**：`python benchmark_train_full.py --model all --suite both` （仅跑 full \ core，~13 h）。
9. 填写第 7 节"执行结果"表。

---

## 7. 执行结果（待填）

### 7.1 clf 任务（dev_acc）

| 档位 | 模型 | 任务 | CINN step (ms) | No-CINN step (ms) | CINN dev_acc | No-CINN dev_acc | 论文区间 | 总 step | 总耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 核心 | Bert-base | SST-2 | | | | | ≥90% | 6,315 | |
| 核心 | Ernie-nano-zh | ChnSentiCorp | | | | | ≥92% | 900 | |
| 核心 | GPT-2-medium | SST-2 | | | | | ≥88% | 12,630 | |
| 完整 | Bert-base | MRPC | | | | | ≥84% | 688 | |
| 完整 | Bert-base | MNLI | | | | | ≥83% | 36,816 | |
| 完整 | Ernie-nano-zh | TNEWS | | | | | ≥56% | 5,003 | |
| 完整 | Ernie-nano-zh | LCQMC | | | | | ≥86% | 22,367 | |
| 完整 | GPT-2-medium | MRPC | | | | | ≥80% | 688 | |

### 7.2 CLM/SFT 任务（dev_ppl 或 train_loss）

| 档位 | 模型 | 任务 | CINN step (ms) | No-CINN step (ms) | CINN dev_ppl | No-CINN dev_ppl | 目标区间 | 总 step | 总耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 核心 | Small Llama 168M | WikiText-103 ×1ep | | | | | ≤50 | ~44K | |
| 完整 | Small Llama 168M | WikiText-103 ×3ep | | | | | ≤30 | ~131K | |
| 完整 | Small Llama 168M | Dolly-15K SFT | | | | | loss ≤1.5 | ~600 | |

### 7.3 不进对比（仅记录）

| 模型 | 状态 | 备注 |
| --- | --- | --- |
| Llama2-7B | ❌ CINN OOM | 论文 step ~500K，2T tokens，单卡跑不动；本计划不做 7B 全量 |

---

## 附录 A：参考资料与原始论文配置

### A.1 BERT（Devlin et al., 2018, NAACL）

- 预训练（参考）：bs=256×512=131K tokens，1M step，Adam lr=1e-4，4 TPUv2 × 4 天，Wikipedia+BookCorpus。
- Fine-tune：SST-2 32×128×3ep×2e-5；MRPC 16×128×3~5ep×2e-5；MNLI 32×128×3ep×3e-5。

### A.2 GPT-2（Radford et al., 2019）

- 预训练（参考）：bs=512×1024，~800K step，WebText 40GB（未公开）。
- Fine-tune：社区默认对齐 HuggingFace `run_glue`，SST-2 16~32×128×3ep×2e-5。

### A.3 ERNIE-3.0-nano-zh（Sun et al., 2021, arXiv:2107.02137）

- 预训练（参考）：4TB 中文语料，nano-zh 4L/312H/17.9M params。
- Fine-tune：ChnSentiCorp 32×128×3ep×5e-5；TNEWS 32×128×3ep×3e-5；LCQMC 32×128×3ep×3e-5。

### A.4 Llama 2（Touvron et al., 2023, arXiv:2307.09288）

- 预训练（参考）：bs=4M tokens×4096 seq，~500K step，2T tokens，Meta RSC ~184K GPU·小时。
- 7B/13B lr=3e-4，cosine + 2000 步 warmup，AdamW β=(0.9, 0.95)，wd=0.1，clip=1.0。
- **本次用 Small Llama 168M**（8L/1024H/8H/vocab=32000，结构同 Llama2，参数随机初始化）。

### A.5 数据集 / 评测基准

- GLUE：Wang et al., 2018, EMNLP。
- CLUE：Xu et al., 2020, COLING 2020。
- WikiText-103：Merity et al., 2017, ICLR。
- TinyLlama：Zhang et al., 2024, arXiv:2401.02385（缩比训练参考）。

### A.6 评测口径说明

- clf 头：loss = CrossEntropy(N 类)，初值 ~ln(N)，看 **dev_acc**（直接对照论文）。
- CLM 头：loss = next-token CE(vocab≈32K)，初值 ~10，看 **dev_ppl = exp(mean_loss)**。
- 两类 loss 量级差 10×，**结果图分两组绘制**（clf 一组、CLM 一组）。

### A.7 单步耗时折算明细（核心档/完整档总时长依据）

> 折算依据：本仓库 500 步实测（bs=1, seq=128）的 CINN/No-CINN 单步时间 × 6（batch 1→16~32 放大系数）。±30%。

| 模型 | 任务 | 总 step | CINN ms/step | NoCINN ms/step | CINN | NoCINN |
| --- | --- | --- | --- | --- | --- | --- |
| Bert | SST-2 | 6,315 | ~95 | ~170 | ~10 min | ~18 min |
| Bert | MRPC | 688 | ~50 | ~90 | ~35 s | ~62 s |
| Bert | MNLI | 36,816 | ~95 | ~170 | ~58 min | ~104 min |
| GPT-2 | SST-2 | 12,630 | ~250 | ~510 | ~53 min | ~107 min |
| GPT-2 | MRPC | 688 | ~250 | ~510 | ~3 min | ~6 min |
| Ernie | ChnSentiCorp | 900 | ~50 | ~100 | ~45 s | ~90 s |
| Ernie | TNEWS | 5,003 | ~50 | ~100 | ~4 min | ~8 min |
| Ernie | LCQMC | 22,367 | ~50 | ~100 | ~19 min | ~37 min |
| Small Llama | WikiText-103 ×1ep | ~44K | ~110 | ~190 | ~80 min | ~140 min |
| Small Llama | WikiText-103 ×3ep | ~131K | ~110 | ~190 | ~4 h | ~7 h |
| Small Llama | Dolly-15K SFT | ~600 | ~440 | ~760 | ~5 min | ~8 min |

### A.8 与现有 500 步脚本的关系

| 现有脚本 | 关系 | 处理方式 |
| --- | --- | --- |
| `benchmark_train_nlp_models.py` | 500 步吞吐基准 | **保留不动**，作为稳态吞吐对照 |
| `profile_nlp_models.py` | 单步 profile | **保留不动** |
| `benchmark_train_llama2.py` | Llama2-7B 单步 | **保留不动**（仅 No-CINN） |
| `benchmark_train_llama2_compare.py` | Small Llama 500 步 | **保留不动**，作为稳态吞吐对照 |
| 本次新增 `benchmark_train_full.py` | 全量 finetune 训练 | 共享 `to_cinn_net` 等工具函数，独立数据/eval 管线 |

> 对拍验证：全量脚本前 500 步 loss/step_time 应与 500 步脚本前 500 步重合（CINN/No-CINN 各两条曲线）。
