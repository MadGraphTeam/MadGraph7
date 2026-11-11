# Eleven-case current-source report release

The canonical 35-page report selects `progress-3/selection.json` and the
preceding-source `closed11-entry-cache-supplement-1.json` dossier. It retains
both sources explicitly. All 66 current-source ordinary lanes pass `3e-6`;
54 pass `1e-8`. Ordinary-target misses, baseline receipt limitations, the
interrupted EW attempt and all historical failures remain visible. Precise
performance acceptance and broad stress/CMS qualification remain open.

The sealed source-only proposal is retained under
`experiments/current-source-final-campaign-supplement-proposal-1` with index
SHA-256 `bc293fb7d696d9861fcf5ef3cf009365a62b4bf6f35ce4149467bcc125a22fe2`.
The root's separate layout patch is recorded here; it avoids a nearly blank
overflow page and removes the zero-pending sentence. The numerical context is
byte-identical to the reviewed preview. Visual review covered pages 32–35.
Pages 2–31 retain their text after page-counter normalization; the cover updates
the ordinary coverage summary. The old 34-page report remains available at
Git commit `75590e978f3a1c45231d6ed1e74a616960ede6be`.

The canonical builder rendered the document and then verified its input/PDF
hashes in a separate serial `--verify-only` invocation. It launched no physics
measurements. See the main benchmark README for exact reproduction commands.
The existing campaign's incomplete performance gate remains incomplete.
