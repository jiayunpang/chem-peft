import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import datetime
import json
import pickle
import random
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from datasets import load_dataset
from rdkit import Chem
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          GenerationConfig, set_seed)
from peft import PeftModel

from utils_other import generate_batch_custom, print_predictions
from tokenization.tokenization_utils import smi_tokenizer_spaces, simple_spaces
from generation.generation_utils import set_generation_config
from evaluation.evaluation_utils import evaluate_batch

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Arguments ────────────────────────────────────────────────────────────────

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_output",
                        help="Directory to save predictions and results.")
    parser.add_argument("--output_prefix", type=str, default="predictions",
                        help="Prefix for output filenames.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smiles_check", type=str, default="no", choices=["yes", "no"])

    # Generation parameters
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=210)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--penalty_alpha", type=float, default=0.0)
    parser.add_argument("--diversity_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--num_beam_groups", type=int, default=1)

    parser.add_argument("--infer_batch", type=int, default=1)
    parser.add_argument("--final_n", type=int, default=1)
    parser.add_argument("--load_in_8bit", type=bool, default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--infer_mode", type=str, required=True, default="greedy",
                        choices=["greedy", "search_beam", "diverse_beam",
                                 "contrastive", "sampling_beam", "nucleus"])

    parser.add_argument("--tokenization", type=str, default="none",
                        choices=["none", "map", "shrink", "map_shrink",
                                 "spaces", "shrink_spaces", "simple_spaces"])

    parser.add_argument("--print_predictions", type=bool, default=False,
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = load_arguments()
    print(args)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Save inference args for reproducibility
    with open(os.path.join(args.output_dir, "infer_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Generation config ─────────────────────────────────────────────────────
    generation_config = set_generation_config(args)

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset("csv", data_files={"test": [args.test_path]})["test"]
    print(f"Loaded {len(dataset)} test samples")

    # ── Load model & adapter ──────────────────────────────────────────────────
    # Use os.chdir workaround for newer huggingface_hub versions
    original_dir = os.getcwd()

    os.chdir(args.model_path)
    if args.load_in_8bit:
        bmodel = AutoModelForSeq2SeqLM.from_pretrained(".", load_in_8bit=True)
    else:
        bmodel = AutoModelForSeq2SeqLM.from_pretrained(".").to(device)
    tokenizer = AutoTokenizer.from_pretrained(".")
    os.chdir(original_dir)

    os.chdir(args.adapter_path)
    model_ = PeftModel.from_pretrained(bmodel, ".")
    os.chdir(original_dir)

    model = model_.merge_and_unload()
    model.eval()
    print("Model and adapter loaded successfully")

    # ── Prepare data ──────────────────────────────────────────────────────────
    spaces = False
    in_gold_mol_pairs = []

    for i in range(len(dataset)):
        orig_input, orig_output = dataset[i]["Input"], dataset[i]["Output"]

        if args.tokenization == "none":
            tokenized_input = orig_input
        elif args.tokenization == "spaces":
            tokenized_input = smi_tokenizer_spaces(orig_input)
            spaces = True
        elif args.tokenization == "simple_spaces":
            tokenized_input = simple_spaces(orig_input)
            spaces = True
        else:
            print(f"ERROR: Unsupported tokenization: {args.tokenization}")
            sys.exit(-1)

        in_gold_mol_pairs.append((tokenized_input, orig_output))

    in_gold_mol_batches = list(generate_batch_custom(in_gold_mol_pairs, args.infer_batch))

    # ── Inference ─────────────────────────────────────────────────────────────
    all_predictions = []
    all_golds = []
    all_inputs = []

    start_time = timer()
    total_batches = len(in_gold_mol_batches)

    for batch_idx, in_gold_mol_batch in enumerate(in_gold_mol_batches):
        batch_inputs = [x[0] for x in in_gold_mol_batch]
        batch_golds = [x[1] for x in in_gold_mol_batch]

        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            max_length=args.max_length,
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
            # Handle both output formats
            output_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
            batch_pred_mols = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        ex_batch_pred_mols = [
            batch_pred_mols[j:j + args.num_return_sequences]
            for j in range(0, len(batch_pred_mols), args.num_return_sequences)
        ]

        all_predictions.append(ex_batch_pred_mols)
        all_golds.append(batch_golds)
        all_inputs.append(batch_inputs)

        # Progress update every 10% or every 100 batches
        if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
            elapsed = timer() - start_time
            print(f"  Batch {batch_idx + 1}/{total_batches} "
                  f"({100 * (batch_idx + 1) / total_batches:.0f}%) "
                  f"- elapsed: {datetime.timedelta(seconds=int(elapsed))}")

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_path = os.path.join(args.output_dir, f"{args.output_prefix}_pred.pkl")
    gold_path = os.path.join(args.output_dir, f"{args.output_prefix}_gold.pkl")

    with open(pred_path, "wb") as f:
        pickle.dump(all_predictions, f)

    with open(gold_path, "wb") as f:
        pickle.dump(all_golds, f)

    print(f"Predictions saved to {pred_path}")
    print(f"Gold labels saved to {gold_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    results = {}
    for K in [1, 2, 3, 5]:
        total = 0
        correct = 0
        for batch_golds, ex_batch_pred_mols in zip(all_golds, all_predictions):
            correct_batch, total_batch = evaluate_batch(batch_golds, ex_batch_pred_mols, K, spaces)
            correct += correct_batch
            total += total_batch

        acc = float(correct) / total
        results[f"top_{K}"] = round(acc * 100, 2)
        print(f"K={K} ||| Correct / Total: {correct} / {total} [{round(acc * 100, 2)}%]")

    # Save results to JSON
    results_path = os.path.join(args.output_dir, f"{args.output_prefix}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    if args.print_predictions:
        print_predictions(all_inputs, all_golds, all_predictions, spaces)

    end_time = timer()
    print(f"Total inference time: {datetime.timedelta(seconds=int(end_time - start_time))}")


if __name__ == "__main__":
    main()
