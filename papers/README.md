# Reference Papers

The PDFs are **not committed**. Fetch them with:

```bash
python scripts/download_papers.py
```

## Reading order (read each one alongside its phase)

| # | Paper | Read during | Focus on |
|---|-------|-------------|----------|
| 1 | Bengio et al., 2003: *A Neural Probabilistic Language Model* ([JMLR](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)) | Phase 1–3 | Why learned embeddings plus next-token prediction give a language model |
| 2 | Vaswani et al., 2017: *Attention Is All You Need* ([1706.03762](https://arxiv.org/abs/1706.03762)) | Phase 4–5 | §3.2 attention and √d_k, §3.3 FFN, §3.5 sinusoidal PE |
| 3 | Ba, Kiros & Hinton, 2016: *Layer Normalization* ([1607.06450](https://arxiv.org/abs/1607.06450)) | Phase 2 | LN vs BN, invariance properties |
| 4 | Radford et al., 2018: *GPT-1* ([PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)) | Phase 5 | Decoder-only LM objective |
| 5 | Radford et al., 2019: *GPT-2* ([PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) | Phase 5–7 | Pre-LN, residual init scaling, BPE on bytes |
| 6 | Kingma & Ba, 2014: *Adam* ([1412.6980](https://arxiv.org/abs/1412.6980)) | Phase 6 | Algorithm 1 and bias correction |
| 7 | Devlin et al., 2018: *BERT* ([1810.04805](https://arxiv.org/abs/1810.04805)) | Background | Encoder vs decoder, masked vs causal LM |
| 8 | Hoffmann et al., 2022: *Chinchilla* ([2203.15556](https://arxiv.org/abs/2203.15556)) | Phase 8 | Params vs tokens trade-off (Exp. E) |
| 9 | Llama Team, 2024: *The Llama 3 Herd of Models* ([2407.21783](https://arxiv.org/abs/2407.21783)) | Phase 8 | Modern choices (RoPE, RMSNorm, SwiGLU) for the discussion section |

### Supplementary (these directly justify choices in this repo)

| Paper | Why |
|-------|-----|
| Hendrycks & Gimpel: *GELU* ([1606.08415](https://arxiv.org/abs/1606.08415)) | Activation and its tanh approximation |
| Xiong et al., 2020: *On Layer Normalization in the Transformer Architecture* ([2002.04745](https://arxiv.org/abs/2002.04745)) | Why we use Pre-LN |
| Sennrich et al., 2016: *BPE for NMT* ([1508.07909](https://arxiv.org/abs/1508.07909)) | BPE tokenizer (Phase 7) |
| Loshchilov & Hutter, 2019: *AdamW* ([1711.05101](https://arxiv.org/abs/1711.05101)) | Decoupled weight decay |
