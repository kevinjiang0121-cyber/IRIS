# Data license and attribution

IRIS-authored data contributions are licensed under **CC BY 4.0**, to the extent
the IRIS contributors hold the applicable rights. This covers their rewrites,
added contexts, human translations, compilation and aggregate tables.
Attribute Haitong Jiang and coauthors, *Same Outcome, Different Evidence:
Intent Recovery in LLM Safety Evaluation*, https://arxiv.org/abs/2610.11766,
and indicate changes.

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Legal text: https://creativecommons.org/licenses/by/4.0/legalcode.en

Wikipedia-derived contexts and their translated adaptations retain CC BY-SA
terms as specified below; they are excluded from the CC BY-only grant.

Imported materials retain their original terms; this grant does not relicense
them. Software is separately licensed under the root MIT LICENSE.

| Imported source | Terms and attribution |
|---|---|
| HarmBench | MIT repository evidence; copyright 2024 centerforaisafety |
| JailbreakBench | JBB-Behaviors card declares MIT; copyright 2023 JailbreakBench Team |
| StrongREJECT custom inputs | Explicit MIT grant for custom generated data; copyright 2024 Center for Human-Compatible AI |
| DAN | MIT; copyright 2023 Vera |
| AdvBench (Zou et al. 2023) | MIT; copyright 2023 Andy Zou |
| Wikipedia context and translations | CC BY-SA 4.0; Wikipedia contributors, Race (human categorization), via HarmBench; see below |
| Selected MasterKey question | CC BY 4.0; Gelei Deng et al., arXiv:2307.08715v2, Appendix A, Privacy example |

Source evidence:
- https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE
- https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/README.md
- https://github.com/JailbreakBench/jailbreakbench/blob/main/LICENSE
- https://github.com/alexandrasouly/strongreject#license
- https://github.com/alexandrasouly/strongreject/blob/main/LICENSE
- https://github.com/verazuo/jailbreak_llms/blob/main/LICENSE
- https://github.com/llm-attacks/llm-attacks/blob/main/LICENSE
- https://arxiv.org/html/2307.08715v2

The selected MasterKey question is printed in the CC BY 4.0 article and
described as author-created. Via StrongREJECT record 167, IRIS changes its
framing and supplies translations/reformulations. Retain paper/version and
license links and this modification notice. This does not license the complete
external MasterKey dataset. StrongREJECT mappings are in
SOURCE_PROVENANCE.csv; original construction history was not retained.

Retain the upstream copyright, permission and disclaimer notices below.

## MIT permission and disclaimer

The following text applies with the corresponding copyright notices above
to the MIT components, to the extent covered by their upstream grants.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Wikipedia context

For wikipedia_race_superiority_edit, ContextString_EN reproduces an excerpt of
Wikipedia contributors' [Race (human categorization)](https://en.wikipedia.org/wiki/Race_(human_categorization)).
ContextString_EO and ContextString_TK are IRIS translations/adaptations of that
excerpt. These contextual texts and the IRIS contributions to their adaptations
are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
not the software MIT license or the CC BY-only grant above. The source page and
its history provide author attribution. Selection/excerpting and translation
are the modifications; the original imported revision was not recorded.
This component-specific ShareAlike notice does not assign BY-SA to unrelated
records or aggregate tables. See [Wikipedia reuse conditions](https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Re-use_of_text).

## Export changes

Conversion replaces names, locations and pet names in maria_racist_insult and
jacob_bully_message and their supplementary texts. IDs, ages and task descriptors
remain; this is pseudonymization. The source ZIP is unchanged. Changed fields
are listed in conversion_report.json.

## Supplementary inputs

Public code uses Google Translate for Zulu/Yoruba and GPT-4o for FormalLogic;
notes report 96 model reformulations and nine manual backfills. Historical
requests and service agreements are unavailable. Current
[OpenAI](https://openai.com/policies/services-agreement/),
[Google](https://policies.google.com/terms) and
[OpenRouter](https://openrouter.ai/terms) terms provide context, but do not
establish which terms applied to those earlier calls. The grant above covers
only contributor-held rights and preserves underlying source licenses.
