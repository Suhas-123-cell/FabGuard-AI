# Reference corpus

`corpus.json` holds 58 passages, each small enough to be a single chunk.

- Four entries (`uored-paper-2023`, `timken-damage-guide-2023`, `skf-bearing-damage-guide`,
  `skf-vibration-diagnostic-guide`) are short project-written summaries with a source link and
  section. FabGuard does not redistribute the linked publisher documents. The UORED paper is
  CC BY 4.0.
- 54 entries with ids starting `note-` are original technical notes written for this project
  from general engineering knowledge. Each carries a `provenance` field saying so. They are not
  copied from, and do not cite, a specific standard or vendor document. Standards such as
  ISO 10816/20816 are paraphrased for orientation only; check the current edition before relying
  on a number.

Applicability is explicit: the UORED entry is tagged `uored`, everything else `general`. The
passages give general inspection guidance. They do not create machine-specific alarm limits or
authorize maintenance work. `tests/test_corpus.py` checks the size, unique ids, provenance field
and that no passage trips the instruction-shaped-text detector.

`evals/retrieval/queries.json` labels each evaluation query with the passages that answer it, so
changing a note's id means updating that file.
