# Phase 8 results

All numbers are validation **bits per character** (BPE-512 tokens, 2.07 chars/token). Final = 100-batch eval after 3000 steps; best periodic = minimum of the 20-batch evals every 250 steps (noisier, shows overfitting). `a ± b` = mean ± sample std over seeds.

**Seed noise:** final val bpc over 3 seeds of the centre config = 2.225 ± 0.018. Single-seed differences below ~2σ = 0.035 are not meaningful; a 3-seed mean has std ≈ σ/√3 = 0.010.

### Ablation: token-embedding scale x positional encoding

| | no scale | tokens x sqrt(D) |
|---|---|---|
| sinusoidal | 2.583 ± 0.077 | 2.225 ± 0.018 |
| learned | 2.215 ± 0.010 | 2.216 ± 0.008 |

### Seed noise (centre)

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| seed 0 | 232,576 | 0.003 | 1 | **2.232** | 2.231 | 1.888 | 0.344 |
| seed 1 | 232,576 | 0.003 | 1 | **2.205** | 2.217 | 1.889 | 0.315 |
| seed 2 | 232,576 | 0.003 | 1 | **2.237** | 2.246 | 1.946 | 0.291 |

### A: depth

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| N = 2 | 132,736 | 0.003 | 3 | **2.273 ± 0.015** | 2.279 ± 0.017 | 2.033 ± 0.027 | 0.240 ± 0.014 |
| N = 4 | 232,576 | 0.003 | 3 | **2.225 ± 0.018** | 2.231 ± 0.015 | 1.908 ± 0.033 | 0.317 ± 0.027 |
| N = 6 | 332,416 | 0.003 | 3 | **2.210 ± 0.010** | 2.211 ± 0.021 | 1.834 ± 0.013 | 0.377 ± 0.003 |

### B: heads

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| H = 2 | 232,576 | 0.003 | 3 | **2.208 ± 0.011** | 2.212 ± 0.016 | 1.908 ± 0.022 | 0.300 ± 0.012 |
| H = 4 | 232,576 | 0.003 | 3 | **2.225 ± 0.018** | 2.231 ± 0.015 | 1.908 ± 0.033 | 0.317 ± 0.027 |
| H = 8 | 232,576 | 0.003 | 3 | **2.225 ± 0.006** | 2.231 ± 0.005 | 1.908 ± 0.020 | 0.318 ± 0.019 |

### C: context

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| T = 64 | 232,576 | 0.003 | 3 | **2.219 ± 0.012** | 2.209 ± 0.002 | 1.880 ± 0.015 | 0.339 ± 0.022 |
| T = 128 | 232,576 | 0.003 | 3 | **2.225 ± 0.018** | 2.231 ± 0.015 | 1.908 ± 0.033 | 0.317 ± 0.027 |
| T = 256 | 232,576 | 0.003 | 3 | **2.252 ± 0.019** | 2.249 ± 0.026 | 1.964 ± 0.030 | 0.288 ± 0.019 |
| T = 512 | 232,576 | 0.003 | 3 | **2.312 ± 0.034** | 2.298 ± 0.040 | 2.075 ± 0.042 | 0.236 ± 0.008 |

### D: positions

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| sinusoidal | 232,576 | 0.003 | 3 | **2.225 ± 0.018** | 2.231 ± 0.015 | 1.908 ± 0.033 | 0.317 ± 0.027 |
| learned | 240,768 | 0.003 | 3 | **2.216 ± 0.008** | 2.221 ± 0.007 | 1.874 ± 0.004 | 0.342 ± 0.007 |

### E: model size

| config | params | lr | seeds | **final val bpc** | best periodic val bpc | train bpc | gap bpc |
|---|---|---|---|---|---|---|---|
| D = 64 | 232,576 | 0.003 | 3 | **2.225 ± 0.018** | 2.231 ± 0.015 | 1.908 ± 0.033 | 0.317 ± 0.027 |
| D = 128 | 858,368 | 0.003 | 3 | **2.299 ± 0.011** | 2.209 ± 0.022 | 1.465 ± 0.016 | 0.835 ± 0.027 |
| D = 256 | 3,289,600 | 0.001 | 1 | **2.599** | 2.195 | 0.892 | 1.707 |
| D = 384, N = 6 | 10,841,856 | 0.0006 | 1 | **3.075** | 2.217 | 0.305 | 2.769 |
