# StrongREJECT item provenance reconstruction — 2026-10-12

## Conclusion

Do not remove all 20 items based on the coarse IRIS source label. Public-source
reconstruction identifies 19 strong correspondences: 11 custom, five DAN,
two AdvBench and one MasterKey. One additional custom-source candidate remains
uncertain (academic papers versus movies). Four matches are exact; 15 are
strong textual reconstructions. None of the 19 strong matches is labeled
MaliciousInstruct, HarmfulQ or OpenAI System Card in the inspected CSV.

This is reconstructed provenance, not recovered construction history or final
legal clearance. Preserve all IRIS data unchanged while investigating.

## Evidence and reproducibility

The maintained loader in
[dsbowen/strong_reject](https://github.com/dsbowen/strong_reject/blob/main/strong_reject/load_datasets.py)
still loads the original repository's CSV.
Inspected original commit: `f7cad6c17e624e21d8df2278e918ae1dddb4cb56`.
Dataset path: `strongreject_dataset/strongreject_dataset.csv`;
blob SHA: `7c4a1927abe5016b19ff57d00eabcf34f4bbeeab`.
Also inspected the archived 2024_02 CSV; its row order differs.
STRONGREJECT_PROVENANCE.csv uses one-based parsed data-record numbers, not
physical line numbers. Original prompt hashes identify records without copying
their text into the audit. The mapping was manually assessed after lexical
candidate retrieval; similarity rankings alone are unreliable for short
rewrites, truncations and reordered clauses.

IRIS input: the existing redacted ZIP, SHA-256
`4ccf4a4b833ad456d8ed80157458f688bd54be98ba810e5bcb2cc59d357fed19`.
Mappings cover all 20 StrongReject/rewrite records exactly once. No input text,
derived text or aggregate value is modified by this audit.

## Source-specific decisions

| Strong correspondences | Count | Evidence and remaining boundary |
|---|---:|---|
| custom | 11 | Original README explicitly releases custom generated data under MIT; preserve Center for Human-Compatible AI copyright/license. IRIS additions/translations need their own rights decision |
| DAN | 5 | Original README identifies DAN as MIT; verazuo/jailbreak_llms LICENSE is MIT, copyright 2023 Vera. Preserve notices and provenance |
| AdvBench | 2 | Original README identifies these questions as Zou et al. 2023 MIT; llm-attacks/llm-attacks LICENSE is MIT, copyright 2023 Andy Zou. The README's arobey1/advbench link is a different adversarial robustness project, so its license is not used as item-specific proof |
| MasterKey | 1 | Text reconstructs record 167. Official site links LLMSecurity/MasterKey; repository Apache-2.0 does not itself establish licensing of the externally hosted evaluation question. Dataset terms remain unresolved |
| Candidate custom mapping | 1 | Record 193 concerns pirated academic papers; IRIS concerns paywalled movies. Shared topic is insufficient to confirm historical derivation; retain as candidate only |

Relevant primary sources:
- [Original StrongREJECT README](https://github.com/alexandrasouly/strongreject#license)
- [Original license](https://github.com/alexandrasouly/strongreject/blob/main/LICENSE)
- [DAN license](https://github.com/verazuo/jailbreak_llms/blob/main/LICENSE)
- [Zou et al. repository license](https://github.com/llm-attacks/llm-attacks/blob/main/LICENSE)
- [MasterKey evaluation questions](https://sites.google.com/view/ndss-masterkey/jailbreak-questions)
- [Official source-code link](https://sites.google.com/view/ndss-masterkey/source-code)
- [MasterKey repository](https://github.com/LLMSecurity/MasterKey)

## What remains

For employee_computer_monitoring, establish dataset-specific permission from
MasterKey's published terms or authors; do not substitute a software license.
For unauthorized_paid_content_access, seek an exact publicly archived counterpart
or author recollection; the public original/2024_02 CSVs inspected do not provide
an exact match. Neither issue requires deleting all 20 inputs now.

Also confirm IRIS-created rewrites, translations and added contextual profiles.
In particular Evan_rumor_fabrication adds a redacted profile absent from the
matched StrongREJECT record: provenance of the base prompt does not license or
validate that addition. Preserve the absence of a blanket MIT metadata grant
and the publication hold until component-specific release decisions are made.
