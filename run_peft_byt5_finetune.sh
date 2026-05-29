#!/bin/bash
python peft_byt5_finetune.py \
--model_path "base-model/byt5s-orig-s300k/" \
--train_path "data/ch_train_prod.csv" \
--val_path "data/ch_val_prod.csv" \
--max_source_len 172 \
--max_target_len 172 \
--seed 42 \
--tokenization "none" \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--logging_steps 50 \
--eval_steps 100 \
--save_steps 100

