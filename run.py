"""Headless reproduction of frozen evaluation inputs; no GPU or annotation UI."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = ['gpt-4o-2024-11-20', 'claude-sonnet-4.5', 'claude-opus-4.5',
           'gemini_3_flash_preview', 'deepseek-v3.2', 'qwen3-max']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def validate():
    import csv
    manifest = read(ROOT/'configs/experiments.json')
    ids = {r['BehaviorID'] for r in csv.DictReader((ROOT/'data/behavior_datasets/final_runnable_with_abc_judge.csv').open(encoding='utf-8'))}
    assert len(ids) == 105
    for name, spec in manifest.items():
        prompts = read(ROOT/spec['prompts'])
        assert set(prompts) == ids, name
        assert all(len(v) == (5 if name == 'HumanJailbreak' else 1) for v in prompts.values()), name
        assert all(isinstance(s, str) and s for v in prompts.values() for s in v), name
    print(f'Validated {len(manifest)} conditions, 105 matched behaviors each; no API calls.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['validate', 'generate', 'score', 'backtranslate'])
    p.add_argument('--condition', default='EN')
    p.add_argument('--model', default=TARGETS[0], help='Key in model configuration')
    p.add_argument('--models', type=Path, default=ROOT/'configs/models.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--judge', default='mistral-large-2512')
    p.add_argument('--judge-prompts', type=Path)
    p.add_argument('--input', type=Path, help='Optional alternate completion file for scoring')
    p.add_argument('--score-name', default='judge', help='Use a different name for sensitivity runs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.stage == 'validate':
        validate(); return
    specs = read(ROOT/'configs/experiments.json')
    if a.condition == 'all' or a.model == 'all':
        conditions = list(specs) if a.condition == 'all' else [a.condition]
        models = TARGETS if a.model == 'all' else [a.model]
        if a.input:
            p.error('--input requires a single condition and model.')
        for condition in conditions:
            if a.stage == 'backtranslate' and condition not in ['Zulu', 'Yoruba']:
                continue
            for model in models:
                cmd = [sys.executable, str(ROOT/'run.py'), a.stage, '--condition', condition,
                       '--model', model, '--models', str(a.models.resolve()), '--output', str(a.output.resolve()),
                       '--max-new-tokens', str(a.max_new_tokens), '--judge', a.judge, '--score-name', a.score_name]
                if a.judge_prompts: cmd += ['--judge-prompts', str(a.judge_prompts.resolve())]
                if a.dry_run: cmd += ['--dry-run']
                subprocess.run(cmd, cwd=ROOT, check=True)
        return
    spec = specs[a.condition]
    folder = a.output.resolve()/a.condition/a.model
    completion = folder/'completions.json'
    if a.stage == 'generate':
        prompts = read(ROOT/spec['prompts'])
        if a.dry_run:
            print(f'{a.condition}: {sum(map(len, prompts.values()))} target calls; model={a.model}'); return
        from api_models import api_models_map
        config = read(a.models)[a.model]['model']
        model = api_models_map(**config)
        if model is None:
            raise RuntimeError('Set OPENROUTER_API_KEY or OPENAI_API_KEY and configure the endpoint.')
        fingerprint = {'prompt_sha256': hashlib.sha256((ROOT/spec['prompts']).read_bytes()).hexdigest(),
                       'model': a.model, 'max_new_tokens': a.max_new_tokens, 'temperature': 0,
                       'thinking_mode': 'off', 'model_config_sha256': hashlib.sha256(a.models.read_bytes()).hexdigest()}
        metadata_path = folder/'run.json'
        if completion.exists():
            if not metadata_path.exists() or read(metadata_path) != fingerprint:
                raise ValueError('Resume configuration differs: choose a new output directory.')
            result = read(completion)
        else:
            result = {}; write(metadata_path, fingerprint)
        for bid, items in prompts.items():
            saved = result.setdefault(bid, [])
            for item in items[len(saved):]:
                output = model.generate([item], max_new_tokens=a.max_new_tokens, temperature=0,
                                        top_p=1.0, thinking_mode='off')[0]
                if output == '$ERROR$':
                    raise RuntimeError(f'Provider request failed for {bid}; checkpoint retained, retry to resume.')
                saved.append({'test_case': item, 'generation': output})
                write(completion, result)
        print(completion)
        return
    if a.stage == 'backtranslate':
        if a.condition not in ['Zulu', 'Yoruba']:
            p.error('Backtranslation requires Zulu or Yoruba.')
        cmd = [sys.executable, str(ROOT/'scripts/backtranslate_lrl_completions_to_english.py'), '--source_lang', a.condition.lower(),
               '--completions_path', str(completion), '--save_path', str(folder/'backtranslated.json'), '--skip_existing']
    else:
        language, template = spec['language'], spec['template']
        prompt_file = a.judge_prompts or ROOT/spec['judge_prompts']
        if a.input and a.input.name == 'backtranslated.json':
            language, template, prompt_file = 'en', 1, ROOT/'configs/judge_prompts.yaml'
        cmd = [sys.executable, str(ROOT/'evaluate_completions.py'), '--cls_model_name', a.judge,
               '--models_config_file', str(a.models.resolve()), '--behaviors_path', str(ROOT/'data/behavior_datasets/final_runnable_with_abc_judge.csv'),
               '--behavior_lang', language, '--judge_template_id', str(template),
               '--judge_prompts_path', str(prompt_file.resolve()), '--completions_path', str((a.input or completion).resolve()),
               '--save_path', str(folder/(a.score_name+'.json')), '--num_tokens', '512',
               '--cls_max_new_tokens', '260', '--cls_thinking_mode', 'off']
    if a.dry_run:
        print(json.dumps(cmd)); return
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
