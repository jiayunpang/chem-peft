# To silence TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model
import argparse

from tokenization.tokenization_utils import smi_tokenizer_spaces

device = "cuda" if torch.cuda.is_available() else "cpu"
template = "{sentence}"


# ── Arguments ────────────────────────────────────────────────────────────────

def load_arguments():
    parser = argparse.ArgumentParser()

    # Model & data
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--train_path",     type=str, required=True)
    parser.add_argument("--val_path",       type=str, required=True)
    parser.add_argument("--experiment_name",type=str, default="byt5s-lora-finetune",
                        help="Used to name the output folder.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Sequence lengths
    parser.add_argument("--max_source_len", type=int, default=96)
    parser.add_argument("--max_target_len", type=int, default=192)

    # Tokenization
    parser.add_argument("--tokenization", type=str, default="none",
                        choices=["none", "map", "shrink", "map_shrink", "spaces", "shrink_spaces"])

    # LoRA
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Training schedule
    parser.add_argument("--num_train_epochs",          type=int,   default=100)
    parser.add_argument("--per_device_train_batch_size",type=int,  default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int,  default=32)
    parser.add_argument("--gradient_accumulation_steps",type=int,  default=2)
    parser.add_argument("--learning_rate",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",              type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type",         type=str,   default="linear")
    parser.add_argument("--warmup_steps",              type=int,   default=0)

    # Logging / evaluation / saving  ← key for reliable loss tracking
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log loss every N steps. Keep well below max_steps.")
    parser.add_argument("--eval_steps",    type=int, default=1000,
                        help="Run evaluation every N steps.")
    parser.add_argument("--save_steps",    type=int, default=1000,
                        help="Save a checkpoint every N steps. "
                             "Must be a multiple of eval_steps when load_best_model_at_end=True.")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Keep only the N most recent checkpoints.")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop if eval exact_match does not improve for N evaluations. "
                             "Set to 0 to disable.")

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = load_arguments()	
    args.model_path = Path(args.model_path).as_posix() # convert once, used everywhere
    print(args)

    # ── Sanity-check step sizes ───────────────────────────────────────────────
    # These mismatches are the most common reason loss never appears in
    # trainer_state.json.  We warn loudly rather than silently misconfigure.
    if args.save_steps % args.eval_steps != 0:
        print(
            f"[WARNING] save_steps ({args.save_steps}) is not a multiple of "
            f"eval_steps ({args.eval_steps}). This can cause issues with "
            "'load_best_model_at_end=True'. Consider aligning them."
        )
    if args.logging_steps > args.eval_steps:
        print(
            f"[WARNING] logging_steps ({args.logging_steps}) > eval_steps "
            f"({args.eval_steps}). You will see fewer loss entries than "
            "evaluation entries in trainer_state.json."
        )

    # ── Seeds ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    original_dir = os.getcwd()
    os.chdir(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(".", legacy=False, local_files_only=True)

    # use_cache must be False when gradient_checkpointing=True; harmless otherwise.
    model = T5ForConditionalGeneration.from_pretrained(".", use_cache=False, local_files_only=True)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(
        "csv",
        data_files={"train": [args.train_path], "val": [args.val_path]},
    )

    def preprocess_function(sample, padding="max_length"):
        inputs, outputs = [], []
        for in_mol, out_mol in zip(sample["Input"], sample["Output"]):
            if args.tokenization == "spaces":
                in_mol  = smi_tokenizer_spaces(in_mol)
                out_mol = smi_tokenizer_spaces(out_mol)
            inputs.append(template.replace("{sentence}", in_mol))
            outputs.append(out_mol)

        model_inputs = tokenizer(
            inputs, max_length=args.max_source_len, padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=outputs,
            max_length=args.max_target_len,
            padding=padding,
            truncation=True,
        )
        # Replace pad token id with -100 so it is ignored in the loss
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["Input", "Output"]
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    em = evaluate.load("exact_match")

    def compute_eval_metrics_training(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds  > 0, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)

        result_em = em.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "exact_match":  round(100 * float(result_em["exact_match"]), 2),
            "avg_gold_len": sum(len(s) for s in decoded_labels) / len(decoded_labels),
            "avg_pred_len": sum(len(s) for s in decoded_preds)  / len(decoded_preds),
        }

    # ── Output folder with timestamp ──────────────────────────────────────────
    # Fixed: timestamp is now actually used in the folder name
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"./output/{args.experiment_name}_{timestamp}"
    print(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Save all args to disk so experiments are reproducible
    with open(os.path.join(output_folder, "run_args.json"), "w") as f:
        args_dict = {k: str(v) for k, v in vars(args).items()}
        json.dump(args_dict, f, indent=2)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_folder,

        # Batch & gradient
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Generation
        generation_max_length=args.max_target_len,
        predict_with_generate=True,
        fp16=False,

        # Optimiser
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,

        # ── Loss is written to trainer_state.json every logging_steps ──
        logging_dir=f"{output_folder}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,       # e.g. every 100 steps
        logging_first_step=True,                # log step-0 loss immediately

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,             # e.g. every 1000 steps

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,             # must align with eval_steps
        save_total_limit=args.save_total_limit,

        # Best-model tracking
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
    )

    # ── Data collator (no model= arg needed with PEFT) ────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics_training,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_folder)
    print(f"Model and trainer state saved to {output_folder}")


if __name__ == "__main__":
    main()
