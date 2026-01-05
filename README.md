
## Overview
This repository contains federated fine-tuning of large language models using [Flower](https://flower.ai/). It adapts the [FlowerTune LLM Leaderboard](https://flower.ai/benchmarks/llm-leaderboard/) for medical tasks.

## Directory Structure
```bash
llm-federated-finetune/
├── flowertune-eval-medical/
│   ├── benchmarks.py
│   ├── eval.py
│   ├── README.md
│   ├── requirements.txt
│   └── utils.py
├── flowertune-llm/
│   ├── flowertune/
│   │   ├── client_app.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── server_app.py
│   │   └── strategy.py
│   └── pyproject.toml
├── flowertune-unsloth/
│   ├── med_llm/
│   │   ├── client_app.py
│   │   ├── dataset.py
│   │   ├── models.py
│   │   ├── server_app.py
│   │   └── strategy.py
│   └── pyproject.toml
└── README.md
```

### flowertune-llm
- Follows the baseline FlowerTune LLM leaderboard.
- Uses tuned hyperparameters and models to improve performance within the leaderboard rules.

### flowertune-unsloth
- Implements LLM fine-tuning using [Unsloth](https://unsloth.ai/) for faster, lower-memory training.
- The scores below come from these training runs.

### flowertune-eval-medical
- Contains the official evaluation scripts from **FlowerTune LLM Leaderboard** for the medical task category.

## Scores
At the time of writing (over a year ago), the following results were achieved using [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). They were not officially submitted to the leaderboard.

- PubMedQA: 68.00
- MedMCQA: 48.24
- MedQA: 57.34
- CareQA: 57.62
- Average: 57.8
- Communication Cost: 6.0 GB

Baseline reference scores from Flower team, using [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3):

- PubMedQA: 59.00
- MedMCQA: 23.69
- MedQA: 27.10
- CareQA: 21.47
- Average: 32.82
- Communication Cost: 40.7 GB

## Status and Compatibility
- The code was originally implemented using Flower’s `Strategy` and `NumpyClient` APIs. Flower has since moved to the [Messaging API](https://flower.ai/docs/framework/how-to-upgrade-to-message-api.html), so updates may be required.
- Unsloth may also have changed since these experiments, so the `flowertune-unsloth` setup could be out of date.

## Run
- **flowertune-llm**:
    ```bash
    cd flowertune-llm 
    python3 -m venv llm-env && source llm-env/bin/activate  
    pip install -e .
    flwr run .
    ```

- **flowertune-unsloth**:
    ```bash
    cd flowertune-unsloth
    python3 -m venv unsloth-env && source unsloth-env/bin/activate 
    pip install -e .     
    flwr run .
    ```
- **flowertune-eval-medical**: Refer to the `README.md` file under the directory.
