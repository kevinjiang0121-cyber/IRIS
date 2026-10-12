# Hugging Face migration audit and release plan

Audit date: 2026-10-11. IRIS main inspected at
`625fbd36b6e53317b9a4e774caae139041c0214c`. Issue #1 was open with zero comments
at inspection. This patch does not reply, accept an invitation or publish data.

## Decision

Worth pursuing for discoverability and simple loading, conditionally. The public
archive is technically sufficient for a useful redacted-input/aggregate dataset,
but not yet sufficient for an unconditional full-dataset redistribution license.
Papers association and dataset hosting are separate decisions.

## Evidence and boundaries

The 26-file ZIP contains 105 behaviors (52 JailbreakBench, 33 HarmBench,
20 StrongReject/rewrite), full and pilot caches, legacy summaries and six
current-submission CSV tables. Source labels lack exact upstream IDs/revisions.
The ZIP's SHA-256 and its internal member hashes are verified by the converter.

The root canonical CSV has additional reconstruction fields and the root
repository contains exact executed prompts. Neither is imported. Using them
would change the migration scope and may defeat ZIP redaction. The archive's
current tables are manuscript aggregates, not item-level gold labels. Old
result paths point to excluded files; the converter omits historical path
metadata and duplicate tables. No private material is requested or inferred.

Eight JSONL configurations contain 582 rows in total, across different units;
this is not 582 independent behaviors. Default is the 105-row behavior panel.
The 315-row supplementary input table is joined by BehaviorID and variant.
Results remain separate configurations with their own schemas and units.
All field descriptions, exclusions and metric aliases are in DATASET_CARD.md.

Automated archive scans found no email/token candidates; converter scans also
check a conservative international-phone pattern. Name-like strings remain in
BehaviorID (including maria and Evan). Those may refer to benchmark examples,
but their real-person status is not established. A multilingual human text
review is still required. The script does not silently alter IDs or content.

## Local conversion and checks

```bash
python scripts/prepare_hf_dataset.py --output /tmp/iris-hf-review
python scripts/check_hf_dataset.py /tmp/iris-hf-review
```

Conversion uses only Python's standard library, never GPU/API calls. Loading
checks additionally require `datasets` and `PyYAML`; do not install the original
inference requirements. The converter refuses an unreviewed ZIP, internal
manifest mismatch, unknown schemas/counts, missing/duplicate IDs, ragged CSV,
nonfinite values, inconsistent point estimates/counts or suspicious contact
patterns. Output must be a new directory. Its report includes schemas, source
hashes and output hashes; draft status is not removed by successful checks.

## Actual release sequence

1. Confirm interest in the invitation and claim/link the paper in HF Papers
   under an author-controlled account. No dataset upload is needed for this.
2. Resolve SOURCE_LICENSES.md: map original sources and rights for selected
   rows, retain required license notices, obtain author/contributor decisions
   for translations/caches/aggregates. Do not use blanket MIT metadata.
3. Manually review all exported strings and residual identifiers. If any edits
   are necessary, create a separately reviewed redacted source revision and
   provenance log; do not reuse the pinned converter as if bytes were unchanged.
4. Choose full approved panel or a clearly labeled cleared subset. Adapt card,
   schema/count checks and configuration coverage to that approved release.
5. Review generated README, SOURCE_LICENSES, JSONL and conversion_report;
   remove draft labels only after decisions are recorded. Keep scripts in GitHub.
6. With explicit publication authorization, upload only that approved bundle to
   an author/org-controlled dataset repository. Test each remote `load_dataset`
   config and the live Viewer, pin a release revision, then link it to the paper.

Keep: typed loading, exact approved source values, stable joins, verified hashes,
accurate units, explicit reproduction limits, resolved component terms.
Kill/hold: blanket data MIT, unresolved inherited rights, new private answers or
labels, claims of full historical reproduction, silent replacement of redacted
ZIP assets with repository prompts, or an unreviewed sensitive field.

## Validation performed

All eight configs loaded using datasets 5.1.0;
source text and numeric round trips passed. Two independent conversions were
byte-identical; tampered ZIP and existing output rejection passed. Live Hub
Viewer and remote loading were not tested because no dataset was uploaded.

## Subsequent item-level audit (2026-10-12)

See STRONGREJECT_AUDIT.md and STRONGREJECT_PROVENANCE.csv: 19 strong
correspondences, plus one uncertain candidate. MIT evidence was found for 18
strongly corresponding base prompts; two items need further provenance/terms
work. Preserve all inputs pending investigation. No exclusion was committed.

Follow-up: the movie input corresponds semantically to custom record 194, not
academic-papers record 193. Updated mapping: 19 base counterparts with MIT
evidence, one MasterKey dataset permission unresolved. Historical movie edits
remain unverified. No input deletion or license assignment occurs.

MasterKey follow-up (2026-10-13): the selected question itself appears in the
CC BY 4.0 arXiv v2 Appendix A. Direct published-text reuse evidence resolves
the previous base-question permission gap; retain component CC BY attribution
and modification notices. No blanket license or publication is authorized.
