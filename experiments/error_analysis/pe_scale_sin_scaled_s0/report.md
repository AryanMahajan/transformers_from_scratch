# Error analysis: `pe_scale/sin_scaled_s0`

421 validation windows × 127 predictions = 53,467 tokens. Mean 2.245 bits/char (3.217 nats/token, 2.07 chars/token). Top-1 accuracy 31.9%, top-5 53.4%.

## By kind of target token

| category | tokens | share of tokens | share of loss | bits/token | top-1 acc |
|---|---|---|---|---|---|
| word start | 16,701 | 31.2% | 44.4% | 6.60 | 7.6% |
| word continuation | 22,449 | 42.0% | 36.9% | 4.08 | 40.2% |
| line start | 3,459 | 6.5% | 7.2% | 5.17 | 10.6% |
| punctuation | 5,536 | 10.4% | 7.9% | 3.55 | 40.7% |
| newline | 4,436 | 8.3% | 1.5% | 0.85 | 90.3% |
| other | 886 | 1.7% | 2.0% | 5.52 | 13.3% |

## By training frequency of the target (equal-occurrence quarters)

| bin | distinct tokens | share of loss | bits/token | top-1 acc |
|---|---|---|---|---|
| most frequent | 14 | 13.5% | 2.51 | 60.5% |
| frequent | 46 | 26.7% | 4.96 | 20.6% |
| less frequent | 91 | 27.2% | 5.02 | 25.8% |
| rarest | 329 | 32.6% | 6.06 | 20.6% |

## Speaker names (316 of 871 speaker turns are by characters never seen in training)

| tokens | count | share of tokens | share of loss | bits/token | top-1 acc |
|---|---|---|---|---|---|
| name, seen in train | 3,184 | 6.0% | 2.3% | 1.76 | 68.0% |
| name, unseen in train | 1,965 | 3.7% | 6.7% | 8.41 | 13.8% |
| not a name | 48,318 | 90.4% | 91.1% | 4.68 | 30.3% |

## Calibration (ECE = 0.040)

| confidence | tokens | mean confidence | accuracy |
|---|---|---|---|
| 0.0–0.1 | 9,844 | 0.073 | 0.070 |
| 0.1–0.2 | 13,444 | 0.145 | 0.128 |
| 0.2–0.3 | 7,618 | 0.245 | 0.195 |
| 0.3–0.4 | 4,705 | 0.347 | 0.283 |
| 0.4–0.5 | 3,307 | 0.448 | 0.362 |
| 0.5–0.6 | 2,493 | 0.548 | 0.464 |
| 0.6–0.7 | 2,248 | 0.649 | 0.593 |
| 0.7–0.8 | 2,214 | 0.751 | 0.705 |
| 0.8–0.9 | 2,629 | 0.851 | 0.773 |
| 0.9–1.0 | 4,965 | 0.966 | 0.915 |

## Tokens costing the most loss in total

| token | count | share of loss | bits/token |
|---|---|---|---|
| `','` | 2124 | 2.3% | 2.74 |
| `'\n'` | 4436 | 1.5% | 0.85 |
| `'.'` | 914 | 1.5% | 3.94 |
| `':'` | 1255 | 1.3% | 2.66 |
| `'O'` | 284 | 1.3% | 11.38 |
| `'A'` | 767 | 1.2% | 3.86 |
| `' '` | 414 | 1.0% | 6.08 |
| `'s'` | 805 | 1.0% | 2.95 |
| `'S'` | 319 | 0.9% | 7.34 |
| `' a'` | 417 | 0.9% | 5.29 |

## Most confident mistakes

| context (last 40 chars) | predicted (p) | actual (p) |
|---|---|---|
| ` I hear them,--Ding-dong, bell.↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `w I hear them,--Ding-dong, bell.↵↵FERDIN` | `IUS` (1.00) | `A` (0.000) |
| `'d.↵↵MIRANDA:↵Alack, for mercy!↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `ain↵And have no vigour in them.↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `o win it↵From me, the lord on't.↵↵FERDIN` | `IUS` (1.00) | `A` (0.000) |
| `y father↵To be inclined my way!↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `, the burthen bear.↵Hark, hark!↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `ies a brave form. But 'tis a spirit.↵↵PR` | `INC` (1.00) | `O` (0.000) |
| `Your tale, sir, would cure deafness.↵↵PR` | `INC` (1.00) | `O` (0.000) |
| `der, sir;↵But certainly a maid.↵↵FERDINA` | `:` (1.00) | `N` (0.000) |
| `nothing natural↵I ever saw so noble.↵↵PR` | `INC` (1.00) | `O` (0.000) |
| `r or five women once that tended me?↵↵PR` | `INC` (1.00) | `O` (0.000) |
