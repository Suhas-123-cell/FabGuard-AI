# Retrieval evaluation

43 labelled queries over 58 passages; Recall@5, MRR and hit@1 (first result relevant). MRR looks at the top 20. Generated 2026-11-20T06:34:20+00:00 on PostgreSQL 16.15 (Debian 16.15-1.pgdg12+2); embedding model BAAI/bge-small-en-v1.5.

| Configuration | Recall@5 | MRR | Hit@1 | Recall@5 terminology | Recall@5 descriptive |
|---|---:|---:|---:|---:|---:|
| fts_all_terms | 0.279 | 0.291 | 0.279 | 0.538 | 0.167 |
| fts_any_term | 0.907 | 0.843 | 0.767 | 1.000 | 0.867 |
| hash_vector | 0.651 | 0.627 | 0.535 | 0.769 | 0.600 |
| hash_score_sum | 0.651 | 0.643 | 0.558 | 0.769 | 0.600 |
| hash_rrf | 0.849 | 0.782 | 0.674 | 0.962 | 0.800 |
| bge_vector | 0.930 | 0.857 | 0.767 | 1.000 | 0.900 |
| bge_score_sum | 0.930 | 0.845 | 0.744 | 1.000 | 0.900 |
| bge_rrf | 0.942 | 0.920 | 0.884 | 1.000 | 0.917 |
