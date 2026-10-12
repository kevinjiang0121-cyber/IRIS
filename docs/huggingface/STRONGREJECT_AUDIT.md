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

## Follow-up correction and remaining issue

A broader semantic check found a better source for unauthorized_paid_content_access:
current CSV parsed record 194 is custom and concerns free watching/downloading
of normally paid movies; the 2024_02 CSV contains the same question at record
195. The previous record-193 academic-papers candidate was wrong and is
superseded. The updated mapping records the same-topic movie counterpart and
its prompt hash. This supports StrongREJECT custom/MIT provenance, while the
historical rewrite chain remains unverified because construction records were
not retained. The existence of this counterpart is stronger evidence than the
previous lexical match, not proof of the exact historical edit.

Updated count: 12 custom counterparts, five DAN, two AdvBench, one MasterKey.
19 base counterparts therefore have MIT evidence; only MasterKey's dataset
permission remains unresolved. Four matches are exact, 15 strong textual
reconstructions and the movies match is semantic/history-unverified.

MasterKey follow-up: its actual default branch is master; the main branch only
contains LICENSE. The master tree contains software and no question dataset.
Its README refers users to arXiv:2305.13860 for the evaluation dataset; the
linked paper is CC BY 4.0 but that cannot automatically license separately
hosted questions. The official Google Sites question page inspected contains
no dataset license statement. No dataset-permission clarification was found
in the inspected public issue titles/bodies. Accordingly neither Apache-2.0
software terms nor the paper's CC BY license is assigned to this IRIS item.
Next decisive evidence is an explicit author/dataset-owner statement covering
this question, its translations and redistribution on Hugging Face. No request
has been sent to authors by this audit. All inputs remain unchanged.

## Decisive MasterKey evidence — 2026-10-13

The previous permission-unresolved conclusion is superseded for this specific
published base question. It is printed verbatim (apart from typographic apostrophe)
in [arXiv:2307.08715v2, Appendix A, Privacy row](https://arxiv.org/html/2307.08715v2#A1).
Appendix A and section III state that the authors manually created the evaluation
questions. The article is explicitly [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
No third-party credit or item-specific exclusion is attached to this row in the
inspected article. Thus there is a direct CC BY 4.0 reuse basis for the text
published in that article, rather than an inference from the software license.
This does not establish the license of the complete external question dataset.

CC BY 4.0 permits sharing and adaptations subject to its conditions. Attribute
Gelei Deng and coauthors and the MasterKey paper, link the version and license,
retain supplied notices, and indicate rephrasing/translation. Do not describe
this component as MIT or add incompatible restrictions. IRIS-created additions
still require their own rights/terms decision.

Suggested component attribution: Base question adapted from Gelei Deng et al.,
MasterKey: Automated Jailbreaking of Large Language Model Chatbots,
arXiv:2307.08715v2, Appendix A, Privacy example, CC BY 4.0. Obtained through
StrongREJECT, record 167; IRIS changes question framing and supplies translated
and reformulated variants. This is reconstructed source attribution, not a
claim that IRIS originally copied directly from the paper.

Updated base-input conclusion for these 20 rows: 19 counterparts with MIT
evidence and this one published question with CC BY 4.0 evidence. The movie
historical edit chain remains unverified. No all-dataset MIT grant is inferred.
All IRIS data and aggregates remain unchanged and no dataset has been uploaded.
