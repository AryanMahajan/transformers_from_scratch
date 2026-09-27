"""Download the reference papers into papers/.

PDFs are not committed (see .gitignore) because redistribution rights vary
by publisher; run this script to fetch them locally:

    python scripts/download_papers.py
"""

import sys
import urllib.request
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"

# (filename, url) — ordered by suggested reading order.
PAPERS = [
    ("01_bengio2003_neural_probabilistic_lm.pdf",
     "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"),
    ("02_vaswani2017_attention_is_all_you_need.pdf",
     "https://arxiv.org/pdf/1706.03762"),
    ("03_ba2016_layer_normalization.pdf",
     "https://arxiv.org/pdf/1607.06450"),
    ("04_radford2018_gpt1.pdf",
     "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"),
    ("05_radford2019_gpt2.pdf",
     "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
    ("06_kingma2014_adam.pdf",
     "https://arxiv.org/pdf/1412.6980"),
    ("07_devlin2018_bert.pdf",
     "https://arxiv.org/pdf/1810.04805"),
    ("08_hoffmann2022_chinchilla.pdf",
     "https://arxiv.org/pdf/2203.15556"),
    ("09_llama3_2024_herd_of_models.pdf",
     "https://arxiv.org/pdf/2407.21783"),
    # Supplementary: directly relevant to design choices in this repo.
    ("s1_hendrycks2016_gelu.pdf",
     "https://arxiv.org/pdf/1606.08415"),
    ("s2_xiong2020_pre_ln_transformer.pdf",
     "https://arxiv.org/pdf/2002.04745"),
    ("s3_sennrich2016_bpe.pdf",
     "https://arxiv.org/pdf/1508.07909"),
    ("s4_loshchilov2019_adamw.pdf",
     "https://arxiv.org/pdf/1711.05101"),
]


def main() -> int:
    PAPERS_DIR.mkdir(exist_ok=True)
    failures = 0
    for name, url in PAPERS:
        dest = PAPERS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name}")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            print(f"[ok]   {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # network errors should not abort the rest
            failures += 1
            print(f"[fail] {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
