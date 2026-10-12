---
language:
- en
- tok
- eo
- zu
- yo
pretty_name: IRIS public redacted inputs and aggregate results
size_categories:
- n<1K
tags:
- arxiv:2610.11766
- llm-safety
- intent-recovery
configs:
- config_name: behaviors
  default: true
  data_files:
  - split: evaluation
    path: data/behaviors.jsonl
- config_name: supplementary_inputs
  data_files:
  - split: evaluation
    path: data/supplementary_inputs.jsonl
- config_name: headline_metrics
  data_files:
  - split: aggregate
    path: data/headline_metrics.jsonl
- config_name: operative_UR_intervals
  data_files:
  - split: aggregate
    path: data/operative_UR_intervals.jsonl
- config_name: t1_t3
  data_files:
  - split: aggregate
    path: data/t1_t3.jsonl
- config_name: iaa
  data_files:
  - split: aggregate
    path: data/iaa.jsonl
- config_name: paired_reading
  data_files:
  - split: aggregate
    path: data/paired_reading.jsonl
- config_name: backtranslation_control
  data_files:
  - split: aggregate
    path: data/backtranslation_control.jsonl
---

# IRIS — review draft, not an authorized release

Data accompanying [Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation](https://arxiv.org/abs/2610.11766), by Haitong Jiang, Chunlin Liu, Sihan Tang, Chan Wu, Xiaoqing Su and Yuhong Feng.
[Code and original public archive](https://github.com/kevinjiang0121-cyber/IRIS).

**Publication hold:** no dataset-wide license has been established. The missing
license metadata is deliberate, not a grant of rights. See SOURCE_LICENSES.md.
Do not upload this draft until the component rights and text review are resolved.
The MIT license for the software does not cover these data automatically.

## Scope and construction

Derived only from `artifacts/data.zip` at IRIS commit
`625fbd36b6e53317b9a4e774caae139041c0214c`.
Archive SHA-256: `4ccf4a4b833ad456d8ed80157458f688bd54be98ba810e5bcb2cc59d357fed19`.
No API generation, translation, inference or new annotations occur during conversion.
JSONL preserves strings and redaction placeholders; result numbers are parsed as
integers/floats without rescaling. No root-repository frozen prompts are imported.

The archive preserves older May inputs alongside current-submission aggregates.
These are different evidence layers: inputs here are redacted inspection assets,
not a guarantee of byte-exact historical executed prompts. T1/T2/T3 input texts
are not in this bundle. The supplementary FormalLogic cache has 96 generated
reformulations and nine manually backfilled items according to the public notes.

## Configurations and fields

| Config | Rows | Unit |
|---|---:|---|
| behaviors | 105 | Matched behavior |
| supplementary_inputs | 315 | Behavior × FormalLogic/Yoruba/Zulu |
| headline_metrics | 60 | Six models × ten conditions |
| operative_UR_intervals | 60 | Same model-condition cells |
| t1_t3 | 6 | One paired comparison per model |
| iaa | 12 | Model × metric agreement aggregate |
| paired_reading | 12 | Model × metric endpoint aggregate |
| backtranslation_control | 12 | Language × model control aggregate |

`evaluation` is the complete evaluation panel, not a train/test partition.
Pilot files, duplicate split lists, historical summaries, operational file paths
and backfill notes are not converted into additional configurations.

- **behaviors**: `BehaviorID` is the unchanged join key; `Source` is the original
  source label (33 HarmBench, 52 JailbreakBench, 20 StrongReject/rewrite).
  `FunctionalCategory`, `SemanticCategory`, `Tags` retain source metadata.
  `Behavior_EN/EO/TK` and `ContextString_EN/EO/TK` are behavior/context strings
  in English, Esperanto and Toki Pona. Empty strings remain empty, not missing labels.
- **supplementary_inputs**: `BehaviorID`, `variant`, `text`, `source_file` and
  `license_status` (pending component review). These are bare supplementary
  assets, not full executed scaffolds or model responses.
- **headline_metrics**: `model`, `condition` (EN, TOK, Zulu, Yoruba, Caesar,
  FormalLogic, ArtPrompt, T1, T2, T3), `n=105`; `operative_count` and
  `harm_outside_operative` are integer counts. `recognition_rate`, `operative_UR`,
  `ASR`, `Shift`, `B` are percentages. Operative UR is P(IR=2 and IS=NA),
  recognition is P(IR=2), ASR is P(SO>=2); Shift records frame change and B
  blocked/empty outputs. Counts/rates do not expose per-item outcomes.
- **operative_UR_intervals**: `model`, `condition`, `operative_UR`, `ci_low`,
  `ci_high` in percent, marginal 95% bootstrap intervals, 10,000 draws,
  seed 20260930. Cells join to headline metrics on model and condition.
- **t1_t3**: `model`, `n`, `n00/n01/n10/n11` are paired T1/T3 binary
  operative-UR counts; `a_rate/b_rate` are T1/T3 percentages; `delta_pp`,
  `ci_low/ci_high` are percentage points; `p_exact/p_holm` are probabilities
  for exact McNemar and Holm across six models. Paired bootstrap: 10,000 draws,
  seed 20261005.
- **iaa**: `model`, `metric`, `n`, `n00/n01/n10/n11` count reader A1/A2
  outcomes; `agreement` is a fraction, `kappa` unitless; `a_rate/b_rate`,
  `strict/permissive` are percentages (individual, intersection/union).
- **paired_reading**: `model`, `metric`, `judge`, `A1`, `A2`,
  `strict/permissive` endpoints are percentages, before adjudication.
- **backtranslation_control**: `language`, `model`, `lrl_ur`,
  `lrl_combined_asr`, `bt_v22en_ur`, `bt_v22en_combined_asr`, `delta_ur`,
  `delta_combined_asr`, `lrl_empty`, `bt_v22en_empty`: all numeric values
  are fractions, including deltas. Here `ur` means recognition only, not
  operative UR. This is a retained May control, not a newly recomputed result.

`TK`/`EO` input suffixes correspond to current paper labels `TOK`/`EPO`.
Display model names in current tables differ from endpoint IDs in the older
backtranslation table. No speculative endpoint mapping is added.

## Loading

Local inspection (install `datasets` separately):

```python
from datasets import load_dataset
panel = load_dataset('json', data_files={'evaluation': 'data/behaviors.jsonl'})
```

After an authorized Hub release, the declared configs support:

```python
panel = load_dataset('ArthurTThomas/IRIS', 'behaviors', split='evaluation')
summary = load_dataset('ArthurTThomas/IRIS', 'headline_metrics', split='aggregate')
```

JSONL and explicit YAML file mappings target Dataset Viewer support without
remote Python execution. Live Viewer operation is not verified before upload.

## Sensitive content and limitations

Contains harmful behavior descriptions for safety evaluation, including hate,
violence, sexual abuse and illegal activity. These are evaluation inputs, not
verified facts about people. The archive replaces many identity-like strings
with bracketed placeholders. Some name-like strings remain in BehaviorID;
multilingual and contextual identifiers need manual review. Automated email,
phone and token scans are triage only and do not certify anonymization.

No raw generated answers, request logs, raw judge records, item-level human
labels, annotator identities or correction histories are included. Aggregate
reader statistics are already public. Historical human agreement and individual
judgments cannot be reconstructed from these tables. Fresh API runs reproduce
protocols subject to provider drift, not guaranteed historical scores. This is
not a complete training dataset or a complete reproduction evidence archive.

## Attribution

Cite the IRIS paper and relevant upstream benchmarks: HarmBench
(arXiv:2402.04249), StrongREJECT (NeurIPS 2024), JailbreakBench
(arXiv:2404.01318). Derived translations/redactions do not erase upstream rights.
See SOURCE_LICENSES.md for terms, provenance gaps and release gates.
