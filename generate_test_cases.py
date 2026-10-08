import transformers
from baselines import get_method_class, init_method
import yaml
import argparse
import csv
import torch
import os
import re
from os.path import join
from eval_utils import get_experiment_config, replace_values_recursive
import pandas as pd
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

import argparse

BEHAVIORS_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
LANG_SUFFIX_TO_CODE = {
    "en": "EN",
    "eo": "EO",
    "es": "EO",
    "tk": "TK",
}


def _extract_behaviors_suffix(behaviors_path):
    stem = os.path.splitext(os.path.basename(behaviors_path))[0]
    match = re.search(r"(_[a-z]{2}(?:-[A-Z]{2})?)$", stem)
    return match.group(1) if match else ""


def _normalize_behavior_lang(behavior_lang):
    if not behavior_lang:
        return None
    key = str(behavior_lang).strip().lower()
    return LANG_SUFFIX_TO_CODE.get(key)


def _infer_language_code(behaviors_path, row_keys, behavior_lang=None):
    forced = _normalize_behavior_lang(behavior_lang)
    if forced and f"Behavior_{forced}" in row_keys:
        return forced

    suffix = _extract_behaviors_suffix(behaviors_path).lower().lstrip("_")
    if suffix in LANG_SUFFIX_TO_CODE:
        code = LANG_SUFFIX_TO_CODE[suffix]
        if f"Behavior_{code}" in row_keys:
            return code

    for code in ("EN", "EO", "TK"):
        if f"Behavior_{code}" in row_keys:
            return code
    return None


def _canonicalize_behavior_row(row, lang_code):
    out = dict(row)
    if lang_code:
        behavior_key = f"Behavior_{lang_code}"
        context_key = f"ContextString_{lang_code}"
        if (not out.get("Behavior")) and behavior_key in out:
            out["Behavior"] = out.get(behavior_key, "")
        if (not out.get("ContextString")) and context_key in out:
            out["ContextString"] = out.get(context_key, "")
    out.setdefault("Behavior", "")
    out.setdefault("ContextString", "")
    return out


def _load_behaviors_csv(behaviors_path, behavior_lang=None):
    last_error = None
    for encoding in BEHAVIORS_CSV_ENCODINGS:
        try:
            with open(behaviors_path, 'r', encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)
                behaviors = []
                for row in reader:
                    normalized_row = {
                        (key.lstrip('\ufeff').strip() if key is not None else key): value
                        for key, value in row.items()
                    }
                    behaviors.append(normalized_row)
            if behaviors:
                lang_code = _infer_language_code(behaviors_path, set(behaviors[0].keys()), behavior_lang=behavior_lang)
                behaviors = [_canonicalize_behavior_row(b, lang_code) for b in behaviors]
                if lang_code:
                    print(f"Info: mapped multilingual behavior columns using language={lang_code}.")
            if encoding != "utf-8":
                print(f"Warning: behaviors file {behaviors_path} is not UTF-8; loaded with {encoding}.")
            return behaviors
        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(
        f"Unable to decode {behaviors_path}. Tried encodings: {', '.join(BEHAVIORS_CSV_ENCODINGS)}. "
        f"Last error: {last_error}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method (options are in baselines/__init__.py)")
    parser.add_argument("--experiment_name", type=str,
                        help="The name of the experiment (options are in configs/methods_config/{{method_name}}_config.yaml)")
    parser.add_argument("--method_config_file", type=str, default=None,
                        help="The path to the config file with method hyperparameters. This will default to configs/method_configs/{{method_name}}_config.yaml")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/final_runnable_with_abc_judge.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--behavior_lang", type=str, default=None, choices=["en", "eo", "es", "tk"],
                        help="Optional language selector for multilingual behavior CSVs without suffix in filename.")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    parser.add_argument("--behavior_start_idx", type=int, default=None,
                        help="Start index for behaviors_path data (inclusive)")
    parser.add_argument("--behavior_end_idx", type=int, default=None,
                        help="End index for behaviors_path data (exclusive)")
    parser.add_argument("--behavior_ids_subset", type=str, default='',
                        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, this will override behavior_start_idx and behavior_end_idx for selecting a subset of behaviors.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="An optional run id for output files, will be appended to output filenames. Must be an integer.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results")
    
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    method_name = args.method_name
    experiment_name = args.experiment_name
    save_dir = os.path.abspath(args.save_dir)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    # Load method config file
    config_file = f"configs/method_configs/{method_name}_config.yaml" if not args.method_config_file else args.method_config_file
    with open(config_file) as file:
        method_configs = yaml.full_load(file)
    
    # Load default method parameters
    if 'default_method_hyperparameters' not in method_configs:
        raise ValueError(f"Default method hyperparameters not found in {config_file}")
    method_config = method_configs.get('default_method_hyperparameters') or {}

    # Update method parameters with experiment parameters
    experiment_config = get_experiment_config(experiment_name, model_configs, method_configs)
    method_config.update(experiment_config)
    model_names = list(model_configs.keys())
    if model_names:
        matches = re.findall(rf"({'|'.join(model_names)})", experiment_name)
        exp_model_configs = {str(i + 1): model_configs[match] for i, match in enumerate(matches)}
    else:
        exp_model_configs = {}
    replace_values_recursive(method_config, exp_model_configs, model_configs=model_configs)

    if args.run_id:
        assert args.run_id.isdigit(), f"epxected run_id to be an integer; got {args.run_id}"
    
    # ========== generate test cases ========== #
    # load behaviors csv
    behaviors = _load_behaviors_csv(args.behaviors_path, behavior_lang=args.behavior_lang)
    if args.behavior_ids_subset:
        print(f"Selecting subset from provided behavior IDs: {args.behavior_ids_subset}")
        if os.path.exists(args.behavior_ids_subset):
            with open(args.behavior_ids_subset, 'r') as f:
                behavior_ids_subset = f.read().splitlines()
        else:
            behavior_ids_subset = args.behavior_ids_subset.split(',')
        behaviors = [b for b in behaviors if b['BehaviorID'] in behavior_ids_subset]
    else:
        behavior_start_idx, behavior_end_idx = args.behavior_start_idx, args.behavior_end_idx
        if args.behavior_start_idx is not None and args.behavior_end_idx is not None:
            print(f"Selecting subset from [{behavior_start_idx}, {behavior_end_idx}]")
            behaviors = behaviors[args.behavior_start_idx:args.behavior_end_idx]

    world_size = torch.cuda.device_count()

    print(f"============= {method_name}, {experiment_name} Config =============")
    print('\n'.join([f"{k}: {v}" for k, v in method_config.items()]))
    print("============= GPU Allocation Configs =============")
    print(f"Total GPUs: {world_size}")
    print("=============")

    # ==== Filtering existed runs =====
    method_class = get_method_class(method_name)

    if not args.overwrite:
        filtered_behaviors = []
        for behavior in behaviors:
            # flatten csv row
            behavior_id = behavior['BehaviorID']
            test_case_file = method_class.get_output_file_path(save_dir, behavior_id, file_type='test_cases', run_id=args.run_id)
            if not os.path.exists(test_case_file):
                filtered_behaviors.append(behavior)
            else:
                print(f"===>Skipping {behavior_id}")
        behaviors = filtered_behaviors
        
        if len(behaviors) == 0:
            print("Found existing test cases for all current behaviors. Exiting.")
            return

    # ==== Init Method ====
    method = init_method(method_class, method_config)

    # ====== Run attack ======
    print("#Behaviors to run:" , len(behaviors))
    test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose)
    
    # ====== Save test cases and logs ======
    print("Saving outputs to: ", save_dir)
    method.save_test_cases(save_dir, test_cases, logs, method_config=method_config, run_id=args.run_id)
    print("Finished")

if __name__ == "__main__":
    main()
