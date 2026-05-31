#!/bin/bash
python byt5_full_finetune_infer.py \
--model_path "output/byt5-small-full-finetuned-ch-functionalisation/" \
--test_path "data/ch_test_prod.csv" \
--max_length 344 \
--infer_mode search_beam \
--num_beams 5 \
--num_return_sequences 5 \
--infer_batch 32 \
--seed 42 \
--tokenization none \
--print_predictions
