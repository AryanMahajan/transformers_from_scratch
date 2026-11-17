# Phase 3: Tokenization and Dataset (study notes)

Revision notes for Phase 3. This phase is algorithms and data rather than
calculus, but its choices (V, T, the metric) shape every later experiment.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

> **Scope note:** BPE is deferred until after the model trains end to end
> (see ROADMAP). This phase covers the char tokenizer plus the dataset.

---

## 1. Character-level tokenizer: `src/tokenizer/char_tokenizer.py`

### What it's for
`text → ids` (training data) and `ids → text` (generation). It fixes **V**,
which sets:
- the Embedding size `V×D`
- the LM head `D×V`
- the softmax cost per position
- the step-0 loss, `log V`

### Vocabulary size vs sequence length
| | char | BPE | word |
|---|---|---|---|
| V | ~65 | 1k–50k | 100k+ |
| tokens per text | most | ~3–4× fewer | fewest |
| OOV / `<unk>` | rare | none (byte-level) | common |
| model must learn | spelling up | mostly semantics | sparse stats |

A context of T tokens covers about T chars at char level, or about 4T chars
with BPE. Attention is O(T²), so this trade-off matters. Char-level is the
safe start for a NumPy model: tiny V and no training.

### Comparing losses: bits per character
Cross-entropy is in **nats per token**, and tokens differ in size across
tokenizers. The fair metric is

$$\text{bpc} = \frac{\text{total nats}}{\ln 2\cdot\#\text{chars}}\quad(\text{char-level: } \text{loss}/\ln 2)$$

The uniform baseline is $\log_2 V$ (6.02 bpc for V = 65). Use bpc (or per-char
perplexity) when comparing models in Phase 8.

### Design
- **Layout:** special tokens first (fixed ids), then `sorted(set(text))`.
  It's deterministic, so the same corpus gives the same ids and checkpoints
  stay valid.
- **`<unk>`** (id 0) takes unseen chars. The round trip holds **only for
  in-vocab text**. With `specials=()`, unknown chars raise instead.
- **encode** → `np.int32` via `np.fromiter` (no intermediate list).
- **decode** validates the id range (−1 would wrap silently, the same trap as
  Embedding) and accepts any shape, flattening it.
- **save/load:** JSON `{"type", "specials", "tokens"}` with tokens in id
  order, UTF-8, `ensure_ascii=False`.

### Gotchas
- **Code points, not graphemes:** `"é"` (precomposed) is 1 token, while
  `"e"` + combining accent is 2. Fine for English. Byte-level BPE removes the
  issue.
- A char tokenizer doesn't parse the string `"<unk>"` inside input text.
  That's 5 ordinary chars. Decoding the unk *id* outputs the literal
  `"<unk>"`.

### Tests (`tests/test_char_tokenizer.py`)
- Round trip (including empty and `\n`).
- Specials first, then sorted chars.
- Deterministic regardless of char order.
- dtype, shape and contiguous ids.
- Unknown chars → `<unk>`, and raises when there's no unk.
- decode range checks and 2-D input.
- Unicode (emoji = 1 code point).
- save/load round trip.
- Constructor validation.

---

## Self-check questions
1. Which parts of the model does V affect, and how?
2. Why does BPE let the same context length T "see" more text? Why does that matter for attention?
3. Why are nats per token not comparable across tokenizers? Define bpc.
4. What is the bpc of a uniform model with V = 65?
5. Why sort the vocabulary?
6. When does `decode(encode(s)) == s` fail for our char tokenizer?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Tokenization in depth (char → BPE, why it matters) | Karpathy, *Let's build the GPT Tokenizer*: https://www.youtube.com/watch?v=zduSFxRajkE |
| BPE for NMT (for later) | Sennrich et al. 2016: `papers/s3_sennrich2016_bpe.pdf` |
