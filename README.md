# Omni-DNA
---


## 📂 Repository Structure

### 🔧 Key Components

- **`seq_pack/`** – Implements SEQPACK: long-context compression module. You can also regard it as a trainable RAG (retrieval-augmented generation) module.
- **`FT_CLS_Head/`** – Finetuning base models with classification heads.
- **`multitask_sft/`** – Multitask and cross-modal supervised finetuning (e.g., DNA→Function).
- **`dna_2_text/`** – DNA-to-text generation pipeline.
- **`datasets/`** – Utilities for Genomic Benchmarks & NT datasets.
- **`utils.py`** – Vocabulary extension and general utilities.
- **`genetic_diseases_variants_prediction/`** – Implements genetic disease variant prediction tasks.

### 🧪 Example Scripts

- `scripts/cls_head_ft.py` – MLP-based finetuning on classification tasks.
- `scripts/sft_multitask.py` – Multitask inference.
- `scripts/dna_2_text.py` – DNA-to-natural language.


---

## Installation

1. Create a virtual environment:
   ```bash
   conda create -n omni_dna python=3.10 -y
   conda activate omni_dna
   ```
2. Install dependencies:
   ```bash
   pip install trl==0.13 transformers datasets datasets ai2-olmo
   # for replicating the dna2image, the following packages are also needed
   # pip install torchvision matplotlib pytorch_lightning
   ```

3. Clone the repository:
   ```bash
   git clone <REPO_URL_REDACTED>
   cd Omni-DNA
   ```

4. for the clinvar disease classification taks, we use the code from GV-Rep: A Large-Scale Dataset for Genetic Variant Representation Learning you could add this repo as a submodule and export the path to the `CLINVAR_DATA_PATH` environment variable.
   ```bash
   git submodule add <REPO_URL_REDACTED> src
   export CLINVAR_DATA_PATH=/absolute/path/to/your/external_path/src
   ```
---

## Model Details

### Base Models

| Size          | Training Tokens | Layers | Hidden Size | Attention Heads | Context Length | Hugging Face Identifier |
|--------------|----------------|--------|-------------|-----------------|----------------|--------------------------|
| Omni-DNA 20M  | 300B           | 8      | 256         | 8               | 250            | `<REDACTED>` |
| Omni-DNA 60M  | 300B           | 8      | 512         | 8               | 250            | `<REDACTED>` |
| Omni-DNA 116M | 300B           | 12     | 768         | 16              | 250            | `<REDACTED>` |
| Omni-DNA 300M | 300B           | 16     | 1024        | 16              | 250            | `<REDACTED>` |
| Omni-DNA 700M | 300B           | 16     | 1536        | 16              | 250            | `<REDACTED>` |
| Omni-DNA 1B   | 300B           | 16     | 2048        | 16              | 250            | `<REDACTED>` |

### SFT Models

| Model Name               | Base Model | Hugging Face Identifier |
|--------------------------|------------|--------------------------|
| Omni-DNA-Multitask       | Omni-DNA 1B | `<REDACTED>` |
| Omni-DNA-DNA2Function    | Omni-DNA 1B | `<REDACTED>` |

---

## Capabilities

Omni-DNA is trained to perform **multiple genomic tasks** including:

- **Finetuning Base Models with MLP attached**
- **SFT for Customized Generation Task**

---

## Examples

### Finetuning Base Models with MLP attached

```python
# Sample code to fine-tune a model on genomic datasets
# Sensitive paths and identifiers removed for privacy

valid_omni_dna_path = {
    "MODEL_1", "MODEL_2", ...
}
```

### Supervised Finetuning (SFT) Example

```json
[
  {
    "instruction": "ATGCGTAC",
    "task": "TASK1:complementary DNA strand",
    "output": "TACGCATG"
  },
  {
    "instruction": "CGCATAT",
    "task": "TASK1:complementary DNA strand",
    "output": "GCGTATA"
  },
  {
    "instruction": "GCGAGATATAAAAA",
    "task": "TASK2:Classify the given DNA sequence based on its function.",
    "output": "Class: Promoter region"
  }
]
```

```python
# Example supervised finetuning code
# Redacted sensitive tokens and URLs

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from src.utils import compute_added_vocabs, extend_model_tokenizer

added_vocabs = ...
path_for_extended_model = ...
extend_model_tokenizer(added_vocabs, path_for_extended_model)
model = AutoModelForCausalLM.from_pretrained(path_for_extended_model)
tokenizer = AutoTokenizer.from_pretrained(path_for_extended_model)

dataset = load_dataset("json", data_files={"train": "path/to/train.json"})["train"]

def formatting_prompts_func(example):
    return [f"{example['instruction']} {example['task']} [SEP] {example['output']}"]

response_template = "[SEP]"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(
    per_device_train_batch_size=6,
    per_device_eval_batch_size=8,
    max_seq_length=512,
    output_dir="./finetuned_omni_dna",
    num_train_epochs=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
```

---

## Replicating Experiments in the Paper

```bash
export PYTHONPATH=$(pwd)

# Finetune base model
python scripts/cls_head_ft.py --dataset nt_downstream --task promoter_tata --model <REDACTED_MODEL> --seed 123 --learning_rate 0.000005 --batch_size 8 --num_of_epoch 10

# Other experiments
python scripts/cls_head_ft_sweep.py
python scripts/sft_multitask.py --model_tokenizer_path <REDACTED_MODEL>
python scripts/dna_2_text.py --output_path current_working_dir

```

---

## Note on Model & Data Availability

Due to the anonymity policy and institutional data-sharing restrictions, we are **unable to release certain pretrained model checkpoints and datasets** at this time. Additionally, all source code has undergone a **sanitization process** to remove sensitive identifiers, including usernames, internal paths, and organizational references.

As a result, **some scripts may not run directly out-of-the-box** without user-provided replacements or minor modifications. This includes restoring model paths, dataset locations, or environment-specific configurations.

However, this does **not affect the reproducibility of the core methodology**, including:

- Model architecture and training pipeline design
- Data formatting logic and preprocessing routines
- Evaluation scripts and fine-tuning procedures
