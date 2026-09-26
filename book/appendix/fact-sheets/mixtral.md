# Mixtral fact sheet

Every fact the book relies on about Mixtral 8x7B lives here, once, with a pinpoint citation. Chapters, exercises, solutions and labs link to a row instead of restating it. See [how fact sheets work](../c-fact-sheets.md).

**Sources.**
- `[Mixtral]` is Jiang et al., *Mixtral of Experts*, arXiv:2401.04088**v1**. See [reference 37](../b-references.md#37-mixtral-of-experts).
- `config.json` is the published file in `mistralai/Mixtral-8x7B-v0.1` @fc7ac94 (commit `fc7ac94680e38d7348cfa806e51218e6273104b0`).

Checked against the sources: 2026-09-26.

## Architecture

| Fact | Value | Source |
|---|---|---|
| Layers | 32 | [Mixtral] Table 1; `config.json` `num_hidden_layers` @fc7ac94 |
| Model dimension | 4,096 | [Mixtral] Table 1 |
| Attention | GQA: 32 query heads (`n_heads`), 8 KV heads (`n_kv_heads`), head dimension 128. Four query heads share each KV head; this is not MHA | [Mixtral] Table 1; `config.json` `num_key_value_heads` @fc7ac94 |
| Experts | 8 per layer, top-2 routing | [Mixtral] Table 1, §2.1 |
| Expert type and width | A SwiGLU block with hidden dimension 14,336 | [Mixtral] Table 1, §2.1 |
| Vocabulary | 32,000 | [Mixtral] Table 1 |
| RoPE $\theta$ | 1,000,000 | `config.json` `rope_theta` @fc7ac94 |
| Load balancing in training | Not described in the paper. Its only mention of load balancing is about spreading work across GPUs under expert parallelism | [Mixtral] §2.1 |
| `router_aux_loss_coef` in the released config | 0.02 | `config.json` @fc7ac94 |

## Context and parameters

| Fact | Value | Source |
|---|---|---|
| Context length | 32,768 tokens. The model "was trained with a context size of 32k tokens", fully dense; the paper describes no shorter first stage | [Mixtral] Abstract, §2, Table 1 |
| Parameters, as reported | 47B total; 13B active per token | [Mixtral] Abstract, §1 |
| Parameters, recomputed | 46.7B total, 12.9B active: 32 layers × (experts $3 \times 4096 \times 14336$ each, plus 41.9M attention) plus untied embeddings $2 \times 32{,}000 \times 4096$. This is our arithmetic | `config.json` @fc7ac94 |
