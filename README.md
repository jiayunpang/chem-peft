# CHEM-PEFT: LoRA Fine-Tuning T5-Style Language Models for Organic Reaction Prediction
This codebase covers functionality related to Low-Rank Adaptation (LoRA) fine-tuning and inference of T5-style models. LoRA is a parameter-efficient fine-tuning (PEFT) strategy, alternative to full fine-tuning.

## Installation
The requirement.txt file list all depending Python libraries and is provided to create a conda environment.

## Usage

### Training

The running bash script is run_peft_byt5_finetune.sh which calls the actual Python code/script peft_byt5_finetune.py. You can run the script by simply executing bash run_peft_byt5_finetune.sh. The LoRA fine-tuning generates an adapter which is used with the base model for inference.

Some hyperparameters are hard-coded in the python file. Most of these parameters are directly associated with the standard TrainingArguments from HuggingFace, see the following links for further guidance:
https://huggingface.co/docs/transformers/en/main_classes/trainer
https://huggingface.co/docs/transformers/v4.41.1/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
https://huggingface.co/docs/diffusers/training/lora

### Inference
The running bash script is run_peft_byt5_infer.sh which calls the actual Python code/script peft_byt5_infer.py. You can run the script by simply executing bash run_peft_t5_infer.sh.

The data is provided under data, including the C-H functionalisation dataset (ch_train_prod.csv, ch_val_prod.csv and ch_test_prod.csv) and the test set of the ten ramdonly selected reaction classes from the USPTO_1k_TPL. 

Three models are provided on huggingfacec https://huggingface.co/JiayunPang

• byt5-small-finetuned-uspto-1k-tpl-chemical-reactions/ - This is the general full fine-tuned model from Byt5 small using USPTO_1K_TPL. This model contains the general knowledge of 1000 classes of organic chemistry reactions and is the base model for the subsequent C-H functionalisation task-specific fine-tuning.

• byt5-small-full-finetuned-ch-functionalisation/ - This is the task-specific full-finetuning model using the C-H dataset (from the base modeL byt5-small-finetuned-uspto-1k-tpl-chemical-reactions/).

• byt5-small-lora-finetuned-ch-functionalisation-adapter/ - This is the task-specific LoRA fine-tuning adapter using the C-H dataset. To run inference of the C-H functionalisation, you will need to load the base model + the LoRA adapter, for example:

python peft_byt5_infer.py --adapter_path "output/byt5-small-lora-finetuned-ch-functionalisation-adapter/" --model_path "base-model/byt5-small-finetuned-uspto-1k-tpl-chemical-reactions/" --print_predictions --test_path "data/ch_test_prod.csv" ...
You can replace ch_test_prod.csv with uspto_tpl_prod_test_prefix_c1.csv to test how much knowledge the model retains in this specific class from USPTP.  

## Data Format

Some example dataset is provided under data/. The format is quite simple:

•	It is a csv file with two columns, where “,” is the delimiter

•	First column is the “Input” column and the second column is the "Output" column. 

The prefix "Product:" indicates the prediction is for the product (in the Output).

•	The T5-style models are multi-task in nature and can be fine-tuned either in a single task or multi-task fashion. When only the input sequence is provided without a specific prefix, then the model is task-specific fine-tuned for a single task. If two or three prefix are provided, the model is task-specific fine-tuned in a multi-task fashion. Each task is marked by a specific prefix (Product:, Reactants:, Reagents:), followed by SMILES of the different components of reaction arranged in the following format: [Product: reactants.reagents,product], [Reagents: reactants.product,reagents] and [Reactants: product,reactants].

## Reference
Jiayun Pang, Ahmed M. Zaitoun, Xacobe Couso Cambeiro and Ivan Vulić. Modular Multi-Task Learning for Chemical Reaction Prediction. (2026) arXiv preprint https://arxiv.org/abs/2602.10404
