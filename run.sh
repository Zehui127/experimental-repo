#!/bin/bash
export PYTHONPATH=$(pwd)

# finetuning with MLP on Genomic Benchmark and Nucleotide Transformer Downstream tasks.
python scripts/cls_head_ft.py --dataset nt_downstream --task promoter_tata --model /Omni-DNA-116M --seed 123 --learning_rate 0.000005 --batch_size 8 --num_of_epoch 10
# or finetuning with MLP with hyperparamter sweeping
python scripts/cls_head_ft_sweep.py

# inference with multi-tasking model
python scripts/sft_multitask.py --model_tokenizer_path /Omni-DNA-Multitask

# inference with dna 2 text model, output_path to save output results
python scripts/dna_2_text.py --output_path current_working_dir
