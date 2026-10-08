# IRIS: Intent Recovery in LLM Safety Evaluation

Code and data for **Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation**.

**Haitong Jiang, Chunlin Liu, Sihan Tang, Chan Wu, Xiaoqing Su, Yuhong Feng**

Shenzhen University; Sihan Tang: School of Intelligence Science and Engineering, Harbin Institute of Technology (Shenzhen), Shenzhen, China.
Corresponding author: Yuhong Feng (yuhongf@szu.edu.cn).

IRIS reports operative task recovery (UR) alongside attack success rate (ASR) to study how similar outcomes can reflect different levels of task engagement. The evaluation uses 105 matched behaviors and six target models across language, encoding, and reconstruction conditions.

- [Paper PDF](paper/IRIS.pdf)
- [Paper source](paper/IRIS.tex)
- [Data package](artifacts/data.zip): redacted inputs and aggregate results
- [Experiment conditions](configs/experiments.json)

## Setup

Python 3.10+ (3.11 recommended). For the original construction pipeline:

```bash
pip install -r requirements_windows.txt
# Linux, including optional local inference: pip install -r requirements.txt
```

For API-only replay, `pip install -r requirements_api.txt` is sufficient.
Set `OPENROUTER_API_KEY` in the environment. Original entry points read
`configs/model_configs/models.yaml`; `run.py` reads equivalent `configs/models.json`.
All credentials are placeholders. Historical endpoints may require updated IDs.

## Source inputs and methods

| Material | Location |
|---|---|
| Complete English and 14-entry TK jailbreak pools | `baselines/human_jailbreaks*/jailbreaks.py` |
| DirectRequest, HumanJailbreak, Caesar, ArtPrompt, FormalLogic, LRL, T1/T2/T3 and clarification implementations | `baselines/` |
| Matched 105 behaviors, EN/TK/EO, T1/T2/T3, clarified TK and semantic cores | `data/behavior_datasets/final_runnable_with_abc_judge.csv` |
| Zulu/Yoruba translations and final FormalLogic cache | `data/full105/` |
| Translation/reformulation scripts | `scripts/generate_low_resource_translations.py`, `scripts/generate_formal_logic_translations.py` |
| Exact executed prompt sets | `data/prompts/` |

The matched 105-item table contains the English, Toki Pona and Esperanto
source texts and is the canonical construction input. Review notes, risk ratings, translator identities and
annotation timestamps are excluded. Three EN/Caesar constructions differ from
executed inputs only in whitespace; frozen files preserve exact executed text.

## Rebuild inputs with original entry points

```bash
python generate_test_cases.py --method_name DirectRequest --experiment_name default --behavior_lang tk --save_dir outputs/constructed/TK
python generate_test_cases.py --method_name Stage2Triplet --experiment_name default --behavior_lang en --save_dir outputs/constructed/T1-T3
python generate_test_cases.py --method_name Stage2Disambig --experiment_name default --behavior_lang tk --save_dir outputs/constructed/TP-disambig
python generate_test_cases.py --method_name HumanJailbreaksToki --experiment_name random_subset_5 --behavior_lang tk --save_dir outputs/constructed/HumanJailbreak
python generate_test_cases.py --method_name CaesarCipher --experiment_name default --behavior_lang en --save_dir outputs/constructed/Caesar
python generate_test_cases.py --method_name FormalLogic --experiment_name default --behavior_lang en --save_dir outputs/constructed/FormalLogic
python generate_test_cases.py --method_name LowResourceLang --experiment_name zulu --behavior_lang en --save_dir outputs/constructed/Zulu
python generate_test_cases.py --method_name LowResourceLang --experiment_name yoruba --behavior_lang en --save_dir outputs/constructed/Yoruba
python generate_test_cases.py --method_name ArtPrompt --experiment_name default --behavior_lang en --save_dir outputs/constructed/ArtPrompt
```

ArtPrompt masking calls an API. FormalLogic/LRL use included caches; regenerating
caches needs API/translation services. Use each generator's `--help`.
ArtPrompt individual files can be merged with `merge_test_cases.py --method_name
ArtPrompt --save_dir outputs/constructed/ArtPrompt`.

`generate_completions.py` and `evaluate_completions.py` retain the original
inference and judging implementation. The `run.py` entry point orchestrates API inference and scoring. Local-GPU
execution is not validated.
For convenient API-only replay use the following commands.

## Run one condition

```bash
python run.py generate --condition TK --model gpt-4o-2024-11-20 --dry-run
python run.py generate --condition TK --model gpt-4o-2024-11-20
python run.py score --condition TK --model gpt-4o-2024-11-20
python analyze.py summary outputs/TK/gpt-4o-2024-11-20/judge.json
```

Generation checkpoints after every response. Rerunning resumes the saved prefix
only when input/configuration hashes match. Use a new `--output` directory for a
new run. Defaults are temperature 0, target output budget 512 tokens, reasoning
off, and judge output budget 260 tokens. Scoring uses the original frozen template
renderer and parser, including its 512-word response clipping and prompt-snippet
limits. API failures stop generation and are not silently converted to safe outcomes.

The scorer's legacy `UR` summary denotes recognition alone. Use `analyze.py` for
headline **operative UR = P(IR=2 and IS=NA)** and **ASR = P(SO>=2)**. Both use all
attempts, including blocked outputs. Missing judgments are counted and flagged;
reported positive rates are lower bounds when judgments are incomplete.

## Experiment coverage

| Experiment | Condition names | Analysis |
|---|---|---|
| Core interface comparison | EN, TK, Zulu, Yoruba, Caesar, FormalLogic, ArtPrompt | Operative UR, ASR, recognition, Shift, B |
| Matched English reconstruction | T1, T2, T3 | Rates, paired bootstrap, exact McNemar + Holm |
| Source-side clarification | TP-disambig | Automated rates |
| Constructed-language control | Esperanto | Automated rates |
| Limited template search | HumanJailbreak | Five frozen attempts; behavior-level best-of-five coverage |
| Translation control | Zulu, Yoruba + backtranslate | Same responses translated into English, then rescored |
| Judge sensitivity | TK rescored with alternate judge/template | Canonical IR=2/IS=M cohort retention and unanimity |

Run generation and scoring for each condition and each of the six target keys in
`run.TARGETS`. Every condition has the same 105 behavior IDs; HumanJailbreak has
525 prompts. The template subset is the frozen seed-1 best-of-five selection.
T1/T2/T3 preserve the order of the executed reconstruction inputs.

```bash
python run.py generate --condition all --model all --dry-run
python run.py generate --condition all --model all
python run.py score --condition all --model all
```

The full matrix makes 10,710 target calls plus judge calls and incurs provider
charges. `--dry-run` and `validate` do not call providers. Marginal operative-UR
intervals use 10,000 binary bootstrap draws with seed 20260930 per cell; Monte
Carlo endpoints may differ slightly from a shared random stream across cells.

```bash
python analyze.py matched outputs/T1/gpt-4o-2024-11-20/judge.json outputs/T3/gpt-4o-2024-11-20/judge.json
```

Supply all six `T1-file T3-file` pairs in **one invocation** to apply Holm correction
across the six-model family. A single pair only corrects a family of one.

### Backtranslation

```bash
python run.py backtranslate --condition Zulu --model gpt-4o-2024-11-20
python run.py score --condition Zulu --model gpt-4o-2024-11-20 --input outputs/Zulu/gpt-4o-2024-11-20/backtranslated.json --score-name backtranslated-judge
```

Uses the original Google Translate wrapper to translate prompts and completions;
the target responses themselves are unchanged. Internet access is required.

### Judge sensitivity

```bash
python run.py score --condition TK --model gpt-4o-2024-11-20 --judge grok-4.20 --score-name grok
python run.py score --condition TK --model gpt-4o-2024-11-20 --judge kimi-k2-thinking --score-name kimi
python run.py score --condition TK --model gpt-4o-2024-11-20 --judge-prompts configs/judge_prompts_v25.yaml --score-name v25
python analyze.py sensitivity outputs/TK/gpt-4o-2024-11-20/grok.json outputs/TK/gpt-4o-2024-11-20/kimi.json --reference outputs/TK/gpt-4o-2024-11-20/judge.json
```

Canonical Mistral recognition is guaranteed in the selected cohort, so unanimity
of Grok and Kimi there equals three-judge recognition unanimity. Repeat over all
six targets and pool retained/selected **counts**, not unweighted percentages.
Alternate endpoint versions and reasoning budgets must match the intended audit;
this runner does not claim fresh calls reproduce the historical selected cohort.

## Reproduction boundary

Frozen inputs are included because human-authored translations and reconstructions
cannot be regenerated exactly by an automatic translation command. Only experiment
input text and behavioral metadata are retained; construction review notes,
annotator identities, human labels, and correction histories are excluded.

Fresh API runs reproduce the protocol, not necessarily the historical numbers:
provider drift and stochastic behavior remain. The paper's human agreement,
paired-human-reading endpoints, and human representative-review columns require
excluded human labels. The runner reports automated judgments without manual overrides.

The convenience runner starts at frozen inputs. The original construction methods
are also included; human translation judgments and manual review cannot be automated.
It includes no old generated answers or machine-judgment archives.

## Attribution and package integrity

The execution/scoring pipeline derives from HarmBench (Center for AI Safety).
Behavior sources include HarmBench, StrongREJECT and JailbreakBench. Interface
implementations include ArtPrompt and Logic Jailbreak. Preserve upstream notices;
the software license does not override data/resource terms. Software copyright notices are provided in LICENSE. `MANIFEST.json` lists file sizes and hashes.

Translation generators default to all 105 matched behavior IDs and write new caches to `outputs/assets/`; included frozen caches are left intact.
