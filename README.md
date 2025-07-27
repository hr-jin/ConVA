# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation

This repository contains the official implementation and data for our **ACL 2025 Main Conference** paper **"Internal Value Alignment in Large Language Models through Controlled Value Vector Activation"**.

![Teaser Figure](figs/framework.pdf)

## Overview

This work proposes a effective method to internally align large language models with specific human values through controlled activation of learned value vectors.


## Installation

```bash
git clone https://github.com/hr-jin/ConVA.git
cd ConVA
pip install -r requirements.txt
```

## Usage

### 1. Training Value Vectors 

Configure your training parameters in `scripts/train_cav.sh`, e.g.:
- Target value concept
- Dataset processing settings
- Backbone model selection

Then run:
```bash
bash scripts/train_cav.sh
```

Trained Value Vectors will be saved in `saved/` by default.

### 2. Controlling Model Values with Value Vectors

Configure inference parameters in `scripts/run_cav.sh`, e.g.:
- Target value to control
- Number of layers to perturb
- Token positions for intervention
- Classification probability `p` (controls perturbation strength)

Then run:
```bash
bash scripts/run_cav.sh
```

## Citation

If you use this work, please cite our **ACL 2025** paper:

```bibtex
@inproceedings{jin-etal-2025-internal,
    title = "Internal Value Alignment in Large Language Models through Controlled Value Vector Activation",
    author = "Jin, Haoran  and
      Li, Meng  and
      Wang, Xiting  and
      Xu, Zhihao  and
      Huang, Minlie  and
      Jia, Yantao  and
      Lian, Defu",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1326/",
    pages = "27347--27371",
    ISBN = "979-8-89176-251-0",
    abstract = "Aligning Large Language Models (LLMs) with human values has attracted increasing attention since it provides clarity, transparency, and the ability to adapt to evolving scenarios. In this paper, we introduce a Controlled Value Vector Activation (ConVA) method that directly aligns the internal values of LLMs by interpreting how a value is encoded in their latent representations and modifies relevant activations to ensure consistent values in LLMs. To ensure an accurate and unbiased interpretation, we propose a context-controlled value vector identification method. To consistently control values without sacrificing model performance, we introduce a gated value vector activation method for effective and minimum degree of value control. Experiments show that our method achieves the highest control success rate across 10 basic values without hurting LLM performance and fluency, and ensures target values even with opposite and potentially malicious input prompts. Source code and data are available at https://github.com/hr-jin/ConVA."
}
```
