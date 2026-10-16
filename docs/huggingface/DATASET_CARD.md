---
language: [en, tok, eo, zu, yo]
license: other
license_name: iris-component-licenses
license_link: https://huggingface.co/datasets/ArthurTThomas/IRIS/blob/main/LICENSE_DATA.md
pretty_name: IRIS
tags: [arxiv:2610.11766, llm-safety, intent-recovery]
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

# IRIS

Public redacted inputs and aggregate results for
[Same Outcome, Different Evidence: Intent Recovery in LLM Safety Evaluation](https://arxiv.org/abs/2610.11766).
[Code](https://github.com/kevinjiang0121-cyber/IRIS). Planned dataset: `ArthurTThomas/IRIS`.

IRIS-authored data contributions use CC BY 4.0; upstream materials retain their
terms, including CC BY-SA for Wikipedia-derived contexts and their translations.
See [LICENSE_DATA.md](LICENSE_DATA.md). Prepared for upload to the Hub; no dataset upload is performed by these scripts.

## Data

| Config | Rows | Description |
|---|---:|---|
| behaviors | 105 | Redacted matched inputs: English, Esperanto, Toki Pona |
| supplementary_inputs | 315 | FormalLogic, Yoruba and Zulu input assets |
| headline_metrics | 60 | Six models × ten conditions |
| operative_UR_intervals | 60 | Marginal 95% operative-UR intervals |
| t1_t3 | 6 | Paired T1/T3 counts and tests |
| iaa | 12 | Aggregate reader agreement |
| paired_reading | 12 | Reader/judge endpoints |
| backtranslation_control | 12 | Legacy LRL/backtranslation comparison |

Behavior fields retain source names: `BehaviorID` is the join key;
`Source`, `FunctionalCategory`, `SemanticCategory`, `Tags` are metadata;
`Behavior_EN/EO/TK` and `ContextString_EN/EO/TK` are strings. Empty strings
are retained. Supplementary fields are `BehaviorID`, `variant`, `text`,
`source_file` and `license_status` (points to the component terms, not a new license grant).

Current result rates and endpoints are **percentages**, with `n=105` in
count tables. Operative UR = P(IR=2 and IS=NA), recognition = P(IR=2), and
ASR = P(SO>=2). `Shift` is frame change; `B` is blocked/empty output.
`operative_count`, `harm_outside_operative` and contingency fields
`n00/n01/n10/n11` are counts. Interval bounds use the estimate's units;
T1/T3 `delta_pp` and its bounds use percentage points.
`p_exact/p_holm` and `agreement` are fractions; `kappa` is unitless.
Reader `strict/permissive` means intersection/union; `a_rate/b_rate` refers
to T1/T3 or A1/A2 according to the table.
In **backtranslation_control**, all rates/deltas are fractions and `ur`
means recognition only. Input TK/EO corresponds to paper TOK/EPO.

## Loading

After publication:

```python
from datasets import load_dataset
data = load_dataset("ArthurTThomas/IRIS", "behaviors", split="evaluation")
```

Local conversion and checking from the code repository:

```bash
python scripts/prepare_hf_dataset.py --output /tmp/iris-hf
python scripts/check_hf_dataset.py /tmp/iris-hf
```

Conversion uses the existing public ZIP only. Loading checks require
`datasets` and `PyYAML`; conversion itself uses the standard library.
No GPU or paid API calls are made. Output hashes and schemas are recorded in
`conversion_report.json`.

## Scope and limitations

The inputs contain harmful evaluation requests. Conversion pseudonymizes names,
locations and pet names in two public HarmBench profiles; stable IDs, ages and
task descriptors remain. The source ZIP is unchanged, and changed fields are
listed in `conversion_report.json`.

This bundle contains public inputs and aggregate results. It excludes T1/T2/T3
texts, raw answers, judge records and item-level human labels, so historical
human statistics cannot be recomputed from it. Fresh API runs may differ.

Sources comprise 33 HarmBench, 52 JailbreakBench and 20 StrongReject/rewrite
behaviors. [SOURCE_PROVENANCE.csv](SOURCE_PROVENANCE.csv) records the available
matches; some are reconstructed, and eight JailbreakBench-labeled rewrites
have no confirmed item mapping. See [LICENSE_DATA.md](LICENSE_DATA.md) for
component licenses and attribution. Cite IRIS and the relevant benchmarks.

## Upload and verify

The output directory is the complete upload unit (12 files). Do not upload the
source ZIP, code repository or earlier review bundles. From your own machine:

```bash
pip install datasets PyYAML huggingface_hub
python scripts/prepare_hf_dataset.py --output /tmp/iris-hf-upload
python scripts/check_hf_dataset.py /tmp/iris-hf-upload
hf auth login
hf upload ArthurTThomas/IRIS /tmp/iris-hf-upload . --repo-type dataset --commit-message "Release IRIS public inputs and aggregate results"
```

The last command publishes the prepared files. Authentication stays on your
machine; do not paste tokens into chat. After upload, rerun the example above
and inspect all eight configs in the Viewer. Link the dataset to the paper via
the arXiv tag already present in this card. No live Viewer check is claimed
before upload.
