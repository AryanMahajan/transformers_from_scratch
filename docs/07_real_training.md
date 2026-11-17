# Phase 7: Real training (study notes)

Revision notes for Phase 7. This phase moves from the memorisation demo to
training on the full corpus: checkpointing, a real run, generation and
evaluation.

> 📚 **Further reading** links are optional. These notes are meant to be
> enough on their own.

---

## 1. Checkpointing and exact resume: `src/training/checkpoint.py`

### Goal
**train → save → exit → reload → resume**, and the resumed run must be
**bit-identical** to one that never stopped. "Approximately the same" hides
bugs. Bit-identical is testable with `==`.

### What the future of a run depends on
| State | Why it's needed | If you forget it |
|---|---|---|
| model config + weights | rebuild the network | — |
| AdamW m, v | per-parameter step sizes | first steps after resume use v = 0, so every step ≈ lr·sign(g): a jolt |
| AdamW t | bias correction $1-\beta^t$ | correction restarts as if t = 1 |
| step | LR schedule position | warmup runs again |
| **batch RNG state** | which windows come next | **silent drift**: different batches (tested) |
| eval RNG state | which eval batches | eval numbers not comparable |
| train config | loop settings | — |
| tokenizer | decode generated text | ids are meaningless without it |
| history | continuous plots | curves start over |

The RNG is the one people forget. Re-seeding with the original seed replays
the batches from step 0, not from step s. The test
`test_forgetting_the_rng_state_breaks_resume` shows the losses diverge.

### File format: one `.npz`, no pickle
```
param/<name>    adam_m/<name>    adam_v/<name>     arrays (float32)
meta            JSON string: configs, tokenizer, step, adam_t, RNG states, history
```
- **No pickle** (`allow_pickle=False`): loading a pickle can execute arbitrary
  code. Arrays + JSON can't.
- NumPy's `bit_generator.state` is a dict of Python ints, so it's JSON-safe.
- `format_version` lets old checkpoints be detected if the format changes.
- **Size ≈ 3× the weights** (weights + m + v). Measured: 801k params →
  9.67 MB file vs 3.21 MB of weights. For inference only the weights are
  needed (`load_model`).

### Atomic save
Write `latest.npz.tmp`, then `os.replace(tmp, path)`. A crash mid-save leaves
the previous checkpoint intact; the rename is atomic on one filesystem.
(Writing through a file handle stops `np.savez` from appending `.npz` to the
tmp name.)

### The Trainer refactor
`train()` kept its state in local variables, so nothing could be saved.
`Trainer` now owns **model, opt, rng, eval_rng, step, history**, which is
exactly what's checkpointed. `train()` is a thin wrapper, so old code and
tests are unchanged.

**Loop order** (what makes resume exact):
```
[eval if step % eval_interval == 0] → [stop?] → train step → step += 1 → [save if step % ckpt_interval == 0]
```
A checkpoint saved after step s contains evals < s. The resumed run evaluates
at s with the restored eval RNG, exactly what the uninterrupted run did next.
Saving *before* the eval would double-record it; saving after the eval but
restoring at s would skip it.

`run(stop_at=k)` simulates being killed after k steps (used by the tests).

### Also added
- `tokens_per_sec` logged every `log_interval` steps (Phase 7 metric).
- `CharTokenizer.to_dict()/from_dict()` so the tokenizer embeds in the
  checkpoint (`save`/`load` reuse them).

### Tests (`tests/test_checkpoint.py`)
- **40 straight == 20 → save → rebuild from file → 20 more:** weights, m,
  v, and every history entry `==`, for tied + sinusoidal and untied + learned.
- Forgetting the RNG state → losses diverge.
- Round-trip contents: configs, step, t, RNG state, 3 arrays per parameter,
  tokenizer, same outputs, float32 preserved.
- Atomic: no `.tmp` left, overwrites `latest.npz`.
- Final checkpoint at the end; unknown format rejected.

### Housekeeping: line endings
Python's `write_text` on Windows writes `\r\n`. Earlier scripted edits had
committed 10 files with CRLF, so their diffs showed every line changed.
Fix: `.gitattributes` with `* text=auto eol=lf` + `git add --renormalize .`
(verified: no content changes, `git diff --ignore-cr-at-eol` empty).

---

## Self-check questions
1. List everything a checkpoint must contain for an exact resume, and what goes wrong if each is missing.
2. Why does forgetting the RNG state cause drift, even with the same seed?
3. What happens to the first steps after resume if Adam's m and v are not restored? If t is not restored?
4. Why avoid pickle for checkpoints?
5. Why write to a temp file and rename?
6. Why is a training checkpoint ~3× the size of the weights? What does inference need?
7. In the loop order eval → stop → step → save, why does resume reproduce the eval at step s exactly once?

---

## 📚 Further reading (optional)
| Topic | Source |
|---|---|
| NumPy `.npz` format and `allow_pickle` | NumPy docs, `numpy.load`: https://numpy.org/doc/stable/reference/generated/numpy.load.html |
| Saving/restoring RNG state | NumPy docs, `BitGenerator.state`: https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.state.html |
