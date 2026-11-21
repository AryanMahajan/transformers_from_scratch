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

## 2. Dataset: Tiny Shakespeare (`scripts/prepare_data.py`, `src/training/dataset.py`)

### Card
| Field | Value |
|---|---|
| Source | Karpathy's char-rnn `input.txt`: about 40k lines from Shakespeare's plays, concatenated |
| Licence | Public domain (Shakespeare) |
| SHA-256 | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Size | 1,115,394 bytes = 1,115,394 chars (pure ASCII) · 40,000 lines · 202,651 words |
| Split | contiguous 90/10: **train 1,003,854** chars · **val 111,540** chars |
| Tokenizer | char-level, vocab built on **train only**: 65 chars + `<unk>` = **V = 66** |
| Val chars unseen in train | none (`<unk>` count in val = 0) |
| Most frequent | `' '` 153,275 · `e` 85,496 · `t` 60,384 · `o` 59,521 · `a` 49,718 |

Re-create everything with `python scripts/prepare_data.py`. Stats are written
to `data/processed/meta.json`.

### Baselines to beat (bits per character)
| Model | bpc | loss (nats) |
|---|---|---|
| Uniform over V = 66 | **6.04** | 4.19 (= log 66, the expected step-0 loss) |
| Unigram (char frequencies, no context) | **4.77** | 3.31 |
| Small char Transformers on this data (literature, roughly) | ~1.5 | ~1.0 |

Anything **below 4.77 bpc** means the model is using **context**, not just
letter frequencies. The gap between the unigram and my model is what the
architecture contributes.

### Design decisions
- **Contiguous split.** Training windows are T consecutive chars. A random
  char or line split would put val text inside train windows (leakage, so
  the val loss would be optimistic). One cut means at most T chars of
  context at the seam.
- **Vocab from train only.** It treats val as truly unseen, like fitting a
  normaliser on train only. (nanoGPT builds it on the full text, which is
  slightly leaky.) The script reports unseen val chars: there are 0 here.
- **SHA-256 recorded** so others can check they have the identical file.
- **Committed:** the tokenizer JSON (small, and it pins the ids checkpoints
  depend on). **Not committed:** the raw text and `.npy` files (re-creatable).

### Robust download
- `raw.githubusercontent.com` **is blocked on some networks**, including
  mine: TLS handshake failure. The script falls back to the jsDelivr CDN
  mirror of the same GitHub file. Tested from a clean state, the SHA matches.
- It writes to `.part`, checks the expected size, then **renames
  atomically**. A failed download can never leave a truncated file that
  later runs would skip as "already exists".

---

## Self-check questions
1. Which parts of the model does V affect, and how?
2. Why does BPE let the same context length T "see" more text? Why does that matter for attention?
3. Why are nats per token not comparable across tokenizers? Define bpc.
4. What is the bpc of a uniform model with V = 65?
5. Why sort the vocabulary?
6. When does `decode(encode(s)) == s` fail for my char tokenizer?
7. Why split contiguously rather than randomly? What leakage remains?
8. Why build the vocab from the training split only?
9. What do the 6.04 and 4.77 bpc baselines mean? What does beating 4.77 prove?
10. Why write the download to `.part` and rename it?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| Tokenization in depth (char → BPE, why it matters) | Karpathy, *Let's build the GPT Tokenizer*: https://www.youtube.com/watch?v=zduSFxRajkE |
| BPE for NMT (for later) | Sennrich et al. 2016: `papers/s3_sennrich2016_bpe.pdf` |
