# SPIRE-NeurIPS2025 (Anonymized Submission)

This repository contains an anonymized implementation of **SPIRE** (Shared-Private Inter-Regional Encoder), a dual-encoder autoencoder framework for modeling shared and region-specific latent dynamics in neural recordings. This work was submitted to NeurIPS 2025.

---

## Overview

SPIRE disentangles shared vs. private latent representations across brain regions (e.g., GPi and STN) from local field potential (LFP) recordings. The model uses dual GRU encoders and decoders, trained with reconstruction, alignment, and orthogonality losses.

Although the full dataset is restricted due to privacy regulations, a demo using synthetic data is provided to allow reproducibility and inspection of core model behavior.

---


## Installation

1. Clone the repository:
```bash
git clone https://github.com/spire-anon/spire-neurips2025.git
cd spire-neurips2025
```
2. Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Demo
A synthetic demo is provided to showcase the training and evaluation pipeline:
# In examples/demo_run_2region.ipynb:
- Loads random synthetic LFP data
- Trains the SPIRE model on it
- Evaluates reconstruction performance

To run:

Open examples/demo_run_2region.ipynb in Jupyter or VS Code. Run all cells. No real data is needed to test the functionality.

## Notes on Data
Due to the use of clinical recordings from pediatric patients, the original dataset cannot be released. This repository provides only the anonymized code.


For questions or clarifications, please refer to the paper associated with this anonymized submission.