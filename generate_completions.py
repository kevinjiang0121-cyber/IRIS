import transformers
import json
import argparse
import os
import csv
import hashlib
import re
from datetime import datetime, timezone
from tqdm import tqdm 
import torch
from baselines import get_template, load_model_and_tokenizer, load_vllm_model
from api_models import api_models_map
from functools import partial
from PIL import Image
from torchvision import transforms
from accelerate.utils import find_executable_batch_size

import yaml

try:
    from vllm import SamplingParams
except Exception:
    SamplingParams = None

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

BEHAVIORS_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
LANG_SUFFIX_TO_CODE = {
    "en": "en",
    "eo": "eo",
    "es": "es",
    "tk": "tk",
}


def _sanitize_path_component(text):
    if not isinstance(text, str):
        text = str(text)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._") or "model"


def _extract_behaviors_suffix(behaviors_path):
    stem = os.path.splitext(os.path.basename(behaviors_path))[0]
    # Only treat language-style suffixes as variants, e.g., "_en" or "_en-US".
    match = re.search(r"(_[a-z]{2}(?:-[A-Z]{2})?)$", stem)
    return match.group(1) if match else ""


def _extract_effective_behaviors_suffix(behaviors_path, behavior_lang=None):
    suffix = _extract_behaviors_suffix(behaviors_path)
    if suffix:
        return suffix
    if behavior_lang:
        normalized = LANG_SUFFIX_TO_CODE.get(str(behavior_lang).strip().lower())
        if normalized:
            return f"_{normalized}"
    return ""


def _resolve_timestamped_completion_save_path(base_save_path, model_name, behaviors_path, behavior_lang=None):
    """
    Save under: <completions_root>/<model_name>/<timestamp>_<model_name>.json
    If base_save_path is a .json file path, use its parent as completions_root.
    If base_save_path is a directory path, use it directly as completions_root.
    """
    if not base_save_path:
        raise ValueError("--save_path is required.")

    normalized = os.path.normpath(base_save_path)
    if normalized.lower().endswith(".json"):
        completions_root = os.path.dirname(normalized) or "."
    else:
        completions_root = normalized

    model_safe = _sanitize_path_component(model_name)
    behaviors_suffix = _extract_effective_behaviors_suffix(behaviors_path, behavior_lang=behavior_lang)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_safe}{behaviors_suffix}.json"
    return os.path.join(completions_root, model_safe, filename)


def _load_behaviors_csv_rows(behaviors_path):
    last_error = None
    for encoding in BEHAVIORS_CSV_ENCODINGS:
        try:
            with open(behaviors_path, 'r', encoding=encoding, newline='') as f:
                rows = list(csv.reader(f))
            if encoding != "utf-8":
                print(f"Warning: behaviors file {behaviors_path} is not UTF-8; loaded with {encoding}.")
            return rows
        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(
        f"Unable to decode {behaviors_path}. Tried encodings: {', '.join(BEHAVIORS_CSV_ENCODINGS)}. "
        f"Last error: {last_error}"
    )


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _to_iso_utc_from_epoch(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _redact_secret_fields(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            lk = k.lower().replace("_", "")
            if "token" in lk or "apikey" in lk:
                out[k] = "***"
            else:
                out[k] = _redact_secret_fields(v)
        return out
    if isinstance(d, list):
        return [_redact_secret_fields(x) for x in d]
    return d


def _extract_api_model_from_generation_function(generation_function):
    if isinstance(generation_function, partial) and generation_function.func is _api_model_generate:
        return generation_function.keywords.get("model")
    return None


def _save_api_metadata_sidecar(args, model_config, test_cases, generation_function):
    api_model = _extract_api_model_from_generation_function(generation_function)
    if api_model is None or not hasattr(api_model, "get_last_generation_metadata"):
        return

    records = api_model.get_last_generation_metadata()
    if not records:
        return

    if len(records) != len(test_cases):
        print(f"Warning: API metadata length mismatch (records={len(records)}, test_cases={len(test_cases)}).")

    enriched = []
    for i, record in enumerate(records):
        item = dict(record)
        if i < len(test_cases):
            tc = test_cases[i]
            item["behavior_id"] = tc["behavior_id"]
            item["test_case_sha256"] = _hash_text(str(tc["test_case"]))
        # Keep provider and normalized response timestamp explicit for downstream analysis.
        item["provider"] = item.get("provider")
        item["response_created_utc"] = _to_iso_utc_from_epoch(item.get("response_created"))
        enriched.append(item)

    behaviors_suffix = _extract_effective_behaviors_suffix(args.behaviors_path, behavior_lang=args.behavior_lang)
    configured_model_name_with_suffix = f"{args.model_name}{behaviors_suffix}"

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_schema_version": 2,
        "api_provider": "openrouter_or_openai_compatible",
        "configured_model_name": args.model_name,
        "configured_model_name_with_suffix": configured_model_name_with_suffix,
        "behaviors_suffix": behaviors_suffix,
        "model_config": _redact_secret_fields(model_config),
        "behaviors_path": args.behaviors_path,
        "test_cases_path": args.test_cases_path,
        "save_path": args.save_path,
        "max_new_tokens": args.max_new_tokens,
        "thinking_mode": args.thinking_mode,
        "num_samples": len(test_cases),
        "records": enriched,
    }

    meta_path = f"{args.save_path}.metadata.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True) if os.path.dirname(meta_path) else None
    with open(meta_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved API metadata to {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model in the models config file")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/final_runnable_with_abc_judge.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--behavior_lang", type=str, default=None, choices=["en", "eo", "es", "tk"],
                        help="Optional language selector used for output naming when behaviors filename has no suffix.")
    parser.add_argument("--test_cases_path", type=str,
                        help="The path to the test cases file to generate completions for")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving completions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for completions")
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="Whether to generate completions with vLLM (if applicable)")
    parser.add_argument("--incremental_update", action="store_true",
                        help="Whether to incrementally update completions or generate a new completions file from scratch")
    parser.add_argument("--thinking_mode", type=str, default="auto", choices=["auto", "on", "off"],
                        help="Control model reasoning mode when tokenizer chat template supports it: auto/on/off")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    original_save_path = args.save_path
    args.save_path = _resolve_timestamped_completion_save_path(
        args.save_path, args.model_name, args.behaviors_path, behavior_lang=args.behavior_lang
    )
    print(f"Resolved save path: {args.save_path} (from --save_path={original_save_path})")
    if args.incremental_update:
        print("Warning: incremental_update is disabled with timestamped outputs (always writes a new file).")
        args.incremental_update = False

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)
    
    num_gpus = model_configs[args.model_name].get('num_gpus', 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", _redact_secret_fields(model_config))
    
    # ========== generate test cases ========== #
    # load behaviors csv
    behaviors = _load_behaviors_csv_rows(args.behaviors_path)[1:]  # ignore the header
   
    with open(args.test_cases_path, 'r') as f:
        test_cases_data = json.load(f)

    generation_function = load_generation_function(model_config,
                                                args.max_new_tokens,
                                                test_cases_path=args.test_cases_path,
                                                generate_with_vllm=args.generate_with_vllm,
                                                thinking_mode=args.thinking_mode)

    returned_data = {}

    if args.incremental_update and os.path.exists(args.save_path):
        with open(args.save_path, 'r') as file:
            returned_data = json.load(file)
        
        # Remove test cases that have already been generated
        new_test_cases_data = {}
        new_returned_data = {}
        for behavior_id, _test_cases in test_cases_data.items():
            # Check if behavior_id is in returned_data and completions have been generated for all test cases
            if behavior_id in returned_data and len(returned_data[behavior_id]) == len(_test_cases):
                continue
            print(behavior_id, len(_test_cases))
            new_test_cases_data[behavior_id] = _test_cases
            new_returned_data[behavior_id] = []  # If not all completions are present, we will regenerate all completions
        test_cases_data = new_test_cases_data

    # ==== Flatten dict to array ======
    test_cases = []
    for behavior_id, _test_cases in test_cases_data.items():
        test_cases.extend([dict(behavior_id=behavior_id, test_case=t) for t in _test_cases])

    if len(test_cases) == 0:
        print('No test cases to generate completions for')
        return
    else:
        print(f'Generating completions for {len(test_cases)} test cases')

    # ==== Generate ====
    print('Generating completions...')
    generations = generation_function(test_cases=[t['test_case'] for t in test_cases])
    print('Done')

    # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
    for t, g in zip(test_cases, generations):
        returned_data.setdefault(t['behavior_id'], [])
        returned_data[t['behavior_id']].append(dict(test_case=t['test_case'], generation=g))

    # Create directories for save_path
    print(f'Saving completions to {args.save_path}...')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(returned_data, file, indent=4)
    print('Saved')

    # Save sidecar metadata for API-based generations (e.g., OpenRouter routing/version traces).
    _save_api_metadata_sidecar(args, model_config, test_cases, generation_function)


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations

def _hf_generate_with_batching(model, tokenizer, test_cases, template, **generation_kwargs):
    @find_executable_batch_size(starting_batch_size=len(test_cases))
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i:i+batch_size]
            inputs = [template['prompt'].format(instruction=s) for s in batched_test_cases]
            inputs = tokenizer(inputs, return_tensors='pt', padding=True)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], **generation_kwargs).cpu()
            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            generations.extend(batch_generations)
        return generations
    return inner_generation_loop()

def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    first = test_cases[0]
    if (
        isinstance(first, (tuple, list))
        and len(first) == 2
        and not (len(first) > 0 and isinstance(first[0], dict))
    ):
        images, prompts = zip(*test_cases)
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)   
    return model.generate(prompts=test_cases, **generation_kwargs)

def _resolve_enable_thinking(thinking_mode):
    if thinking_mode == "auto":
        return None
    return thinking_mode == "on"


def load_generation_function(model_config, max_new_tokens, test_cases_path, generate_with_vllm=False, thinking_mode="auto"):
    model_name_or_path = model_config['model_name_or_path']
    enable_thinking = _resolve_enable_thinking(thinking_mode)

    if (model := api_models_map(**model_config)):
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        if thinking_mode in ("on", "off"):
            generation_kwargs["reasoning_enabled"] = (thinking_mode == "on")
        return partial(_api_model_generate, model=model, **generation_kwargs)
    elif generate_with_vllm:
        if SamplingParams is None:
            raise RuntimeError("vLLM generation requested but vllm is not installed.")
        print('Using vLLM generation')
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        model = load_vllm_model(**model_config)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None),
                                enable_thinking=enable_thinking)
        return partial(_vllm_generate, model=model, template=TEMPLATE, **generation_kwargs)
    else:
        print('Using HF generation')
        model, tokenizer = load_model_and_tokenizer(**model_config)
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None),
                                enable_thinking=enable_thinking)
        return partial(_hf_generate_with_batching, model=model, tokenizer=tokenizer, template=TEMPLATE, **generation_kwargs)

if __name__ == "__main__":
    main()
