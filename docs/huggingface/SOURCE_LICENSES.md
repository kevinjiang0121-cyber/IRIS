# Source terms and unresolved rights — audited 2026-10-11

This is a provenance audit, not a new license. Public accessibility does not
itself grant redistribution rights. Do not tag the complete dataset `mit`.

| Component | Public evidence | Status/action |
|---|---|---|
| IRIS conversion software | Root LICENSE/NOTICE: MIT software only | Preserve root software notice; does not license outputs |
| JailbreakBench, 52 labeled rows | [JBB-Behaviors card](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/README.md) explicitly labels dataset MIT | Strongest dataset-level evidence; retain notices, trace original items and inherited sources |
| HarmBench, 33 labeled rows | [LICENSE](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE) MIT, copyright 2024 centerforaisafety; [README](https://github.com/centerforaisafety/HarmBench/blob/main/README.md) | Repository MIT is evidence, but selected contextual content and third-party origins need item-level review |
| StrongReject/rewrite, 20 labeled rows | [LICENSE](https://github.com/dsbowen/strong_reject/blob/main/LICENSE) MIT, copyright 2024 Dillon Bowen; [README License section](https://github.com/dsbowen/strong_reject#license) lists inherited datasets | Item reconstruction finds MIT evidence for 19 base counterparts; the movie counterpart is semantically matched with historical edits unverified, and one MasterKey item remains permission-unresolved. See STRONGREJECT_AUDIT.md. IRIS-added materials require separate decisions |
| IRIS human translations, redactions, generated caches and manual FOL backfills | ZIP source labels and public construction notes | Author/contributor rights and generated-asset service terms not specified; require maintainer confirmation |
| Current manuscript aggregate tables | Already public in ZIP; no explicit data grant in NOTICE | Authors should establish applicable terms before a Hub release |
| ArtPrompt / Logic Jailbreak implementations | Attribution in NOTICE and bibliography | Not exported as datasets by this converter; their code licenses do not authorize benchmark data |
| Paper figure assets | Separate figure attribution in NOTICE | Excluded entirely from this conversion |

The IRIS source field supplies benchmark names, not original item IDs or
revisions. A subsequent public-source reconstruction now identifies 20 counterparts (12 custom, five DAN, two AdvBench, one MasterKey);
the movie counterpart is semantic with historical edits unverified. See [item audit](STRONGREJECT_AUDIT.md) and
[20-row mapping](STRONGREJECT_PROVENANCE.csv). The original StrongREJECT README
explicitly grants MIT for custom generated data; DAN and AdvBench also have MIT
evidence. MasterKey's externally hosted dataset terms remain unresolved; the movie
base counterpart is custom/MIT, though historical edits are unverified. Reconstruction does not certify
historical lineage or rights for IRIS-added contexts/translations. No rows are
removed, and no blanket MIT metadata is assigned.

Before public upload, record upstream repository/dataset revisions, original
item identifiers, license URLs/full required notices, modifications (translation,
redaction, rewriting), and a component-specific decision. Confirm rights for
IRIS-authored additions and aggregates. Preserve license texts/copyright notices
in the final release for every approved component. No upstream license texts
are silently replaced by IRIS's MIT software license.

A private review bundle is not permission to publish. If full clearance cannot
be completed, a narrower release may contain only cleared aggregates or cleared
input subsets, with accurate row counts and coverage limits; do not advertise
that subset as the full 105-behavior panel. Gating access is not a substitute for
redistribution permission. Linking HF Papers to the existing GitHub archive can
proceed separately from dataset migration.

## MasterKey resolution for the selected item (2026-10-13)

employee_computer_monitoring's base question is included in the author-written
Privacy example in arXiv:2307.08715v2 Appendix A. The article is CC BY 4.0;
no item-specific third-party exclusion is indicated there. This supplies a
direct reuse basis for that published question, not a blanket license for the
external MasterKey dataset. Preserve Deng et al. attribution, article/version
and CC BY 4.0 links, and indicate IRIS rephrasing and translation. See the latest
section of STRONGREJECT_AUDIT.md; earlier unresolved statements are superseded
for this specific base question. This component must not be tagged as MIT.
