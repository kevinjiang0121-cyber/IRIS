# Data sources, attribution and license scope

IRIS-authored contributions use CC BY 4.0. Imported material retains its
upstream terms. This document does not replace upstream license grants.

| Component | License evidence | Attribution and scope |
|---|---|---|
| IRIS-authored rewrites, contexts, human translations, compilation and aggregate tables | CC BY 4.0, authorized by the maintainer in this release preparation | Cite Haitong Jiang and the IRIS coauthors; link arXiv:2610.11766 and indicate subsequent changes. Applies only to rights held by the IRIS contributors |
| JailbreakBench inputs | Dataset card declares MIT | [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/README.md); retain upstream notices; selected item/context provenance still requires final review |
| HarmBench inputs | Repository MIT evidence | [HarmBench LICENSE](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE), copyright 2024 centerforaisafety; retain notices; selected third-party contexts still require final review |
| StrongREJECT custom counterparts | Explicit MIT grant for custom generated data | [Original README](https://github.com/alexandrasouly/strongreject#license) and [LICENSE](https://github.com/alexandrasouly/strongreject/blob/main/LICENSE), copyright 2024 Center for Human-Compatible AI |
| DAN counterparts | MIT evidence | [DAN LICENSE](https://github.com/verazuo/jailbreak_llms/blob/main/LICENSE), copyright 2023 Vera |
| AdvBench counterparts | MIT evidence for Zou et al. 2023 | [llm-attacks LICENSE](https://github.com/llm-attacks/llm-attacks/blob/main/LICENSE), copyright 2023 Andy Zou. The unrelated arobey1/advbench project is not the source of these questions |
| MasterKey selected published question | CC BY 4.0 article, Appendix A, Privacy example | Gelei Deng et al., [MasterKey, arXiv:2307.08715v2](https://arxiv.org/html/2307.08715v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Via StrongREJECT record 167; IRIS rephrases as tips and provides translations/reformulations |
| Generated supplementary caches | Underlying texts retain upstream terms | The maintainer's grant applies to IRIS-held contributions only; generated-asset service terms must be checked for the actual providers before release |
| Software | MIT | Root LICENSE applies to software, including conversion tooling |

## StrongREJECT reconstruction

[STRONGREJECT_PROVENANCE.csv](STRONGREJECT_PROVENANCE.csv) and
[STRONGREJECT_AUDIT.md](STRONGREJECT_AUDIT.md) document the 20 source
correspondences. Twelve are custom, five DAN, two AdvBench and one MasterKey.
Nineteen base counterparts have MIT evidence; the selected MasterKey question
has direct CC BY 4.0 evidence in the author-written article appendix. This is
not a grant for the entire external MasterKey dataset.

Four matches are exact, 15 are strong textual reconstructions, and the
paid-movie counterpart is semantic with historical edits unverified.
Construction history was not retained. Public-source reconstruction must not
be described as original construction records. No source texts are removed.

## Changes and required notices

IRIS performs selection, redaction, rewriting, translation and reformulation.
Keep supplied copyright, license and disclaimer notices for imported material.
For the MasterKey-derived component retain authors, article/version and license
links, and indicate these changes. Do not impose restrictions incompatible with
the upstream grant or imply endorsement.

Before upload, include the applicable full MIT license notices in the final
bundle, complete remaining selected-item/context checks, check generated-asset
terms and review multilingual strings/identifiers. A software license, a
citation or a successful loading test does not complete those checks.
No manuscript/figure assets, raw answers, raw judge records, item-level human
labels or other unpublished materials are part of the converted bundle.
