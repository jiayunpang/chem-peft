#!/bin/bash
python byt5_full_finetune.py \
--model_path "base-model/JiayunPang/byt5-small-finetuned-uspto-1k-tpl-chemical-reactions/" \
--train_path "data/ch_train_prod.csv" \
--val_path "data/ch_val_prod.csv" \
--max_source_len 172 \
--max_target_len 172 \
--seed 42 \
--freeze none \
--tokenization none \
--logging_steps 50 \ 
--eval_steps 100 \
--save_steps 100 \

