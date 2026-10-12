# Historical validation evidence (immutable)

These are the original raw episodes, manifest and selection artifact supplied in
`selection-20261012.zip`. No numbers or promotion outcomes were changed.

- 768 rows, 2 validation cities, 16 configurations, 4 matched episodes, 6 methods.
- Selected controller: **EARS-Safe-v7**. All four v9 candidates failed promotion.
- EARS-Info-v9 improved mean J but breached the worst-setting regression gate.
- Code hashes in the original manifest used CRLF file bytes. They correspond to
  the reviewed v9 implementation before the publication workflow was added.
- Legacy map metadata does not establish explicit mission-target provenance.
- The checkpoint hash is recorded; the checkpoint itself was not in the upload.

The publication study treats Info-v9 as an explicitly post-validation research
candidate. It does not relabel the original selection as a successful promotion.

Regenerate the audit:

```bash
python experiments/report_publication_v9.py --validation-only --output-root results/publication-v9-validation
```
