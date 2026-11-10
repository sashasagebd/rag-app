# Experiments

Corpus `dl-slides` (23 lecture decks), **dev split** (35 questions: 12 single_fact, 6 exact, 7 multi_hop, 5 adversarial, 5 unanswerable). Haiku 5.5 judges every run.

**Method.** Each experiment changes one thing (`configs/exp-*.toml`, each `extends` the config it builds on), runs on dev, and is compared with `python -m eval.compare <reference> <experiment>`. With 35 questions one question is ~3 points, so verdicts rest on the **flip list** (which questions changed and why), not on small metric deltas. Winners are combined, and the final config runs **once** on the test split.

**Metrics.** hit@k / recall@k / MRR: did retrieval find the cited sources (recall matters for multi_hop). correct: judge says the answer matches the reference. f.abst: said "not found" although the answer exists (lower is better). abstain: said "not found" on unanswerable questions (higher is better). cite.ok: a citation covers a source.

## Results (dev)

| # | change (vs. its reference) | hit@k | recall | MRR | correct | f.abst | abstain | cite.ok | cost/run* | flips (better / worse) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | baseline: 500-token chunks, bge-small, vector top 5, Haiku, prompt v1 | 0.90 | 0.81 | 0.64 | 0.80 | 0.13 | 1.00 | 0.96 | $0.017 | | reference |
| E1 | k = 10 | 0.90 | 0.84 | 0.64 | 0.83 | 0.10 | 1.00 | 0.93 | $0.025 | +005 +030 / −018 | keep |
| E2 | 200-token chunks | 0.93 | 0.88 | 0.76 | 0.87 | 0.03 | 1.00 | 0.97 | $0.012 | +001 +005 +026 / −032 | **keep** |
| E3 | LaTeX → plain symbols before embedding | 0.87 | 0.78 | 0.64 | 0.80 | 0.17 | 1.00 | 0.96 | $0.016 | +035 / −007 | drop |
| E4 | hybrid: vector + keyword search (RRF) | 0.87 | 0.79 | 0.62 | 0.83 | 0.10 | 1.00 | 0.93 | $0.018 | +005 / — | drop |
| E5 | rerank top 20 with bge-reranker-base → 5 | 0.97 | 0.92 | 0.83 | 0.93 | 0.03 | 1.00 | 0.97 | $0.018 | +005 +017 +026 +030 / — | **keep** (slow, see notes) |
| E6 | bge-base embeddings (768 dims) | 0.93 | 0.86 | 0.69 | 0.90 | 0.07 | 1.00 | 0.96 | $0.017 | +017 +026 +035 / — | drop (for now) |
| C1 | E2 + E5 (rerank 20 → 5 short chunks) | 0.90 | 0.83 | 0.72 | 0.83 | 0.10 | 1.00 | 0.93 | $0.012 | +001 +005 / −007 −035 | drop |
| C2 | E2 + E5 + E1 (rerank 30 → 10 short chunks) | **1.00** | **0.97** | 0.78 | 0.90 | 0.07 | 1.00 | 0.89 | $0.016 | +001 +005 +017 +026 +030 / −007 −048 | keep (retrieval) |
| E7 | C2 + Sonnet 5.5 answers | 1.00 | 0.97 | 0.78 | 0.93 | 0.07 | 1.00 | 0.93 | $0.244 | vs C2: +007 +035 +048 / −026 −030 | drop (15× cost for +1) |
| E8 | C2 + answer prompt v2 | **1.00** | **0.97** | 0.78 | **1.00** | **0.00** | **1.00** | 0.97 | $0.017 | vs C2: +007 +035 +048 / — | **winner** |

\* Full price for the whole dev run (35 answers + judge calls), as if nothing were cached.

Numbers are from the re-run after the label fixes below (2026-11-10). Every config abstains on the new unanswerable dl-051.

**Winner: `configs/exp-e8-prompt-v2.toml`.** 200-token chunks, vector search for 30 candidates, bge-reranker-base keeps the best 10, Haiku answers with prompt v2.

## Final result (test, run once)

The winner was run **once** on the 15 test questions, which no experiment looked at. Config unchanged afterwards.

| split | n | hit@10 | recall | MRR | correct | partial | f.abst | abstain | cite.ok | cost/run |
|---|---|---|---|---|---|---|---|---|---|---|
| dev (winner) | 35 | 1.00 | 0.97 | 0.78 | 1.00 | 0.00 | 0.00 | 1.00 | 0.97 | $0.017 |
| **test (winner)** | **15** | **1.00** | **0.96** | **0.76** | **0.92** | 0.08 | **0.00** | **1.00** | **1.00** | $0.007 |

By category on test: single_fact 5/5, exact 2/2, adversarial 2/2, unanswerable 3/3 abstained, multi_hop 2/3 correct + 1 partial.

**Test holds up.** Retrieval is as good as on dev (every source found; recall 0.96), no false "not found", every unanswerable question abstained, every citation valid. The one miss, **dl-037** (multi_hop, *partial*), is an answer-completeness miss, not retrieval: both source slides were retrieved, but the answer gave only one of the three things the proposal adds (uncertainty estimate) and left out the fair baseline comparison and the fixed metric. The drop from dev's 1.00 to 0.92 is that one question. Dev's 1.00 was expected to be optimistic, since prompt v2 was written after seeing dev's failures.

Results file: `results/20261110-223529_dl-slides_test_final.json`. It records `dirty: true` because it ran before the Phase 3 code was committed. The code it ran is the Phase 3 commit that follows.

## Notes

**E1, k = 10.** Fixed what it targeted: multi_hop correct 0.57 → 0.86 (dl-005 now retrieves S08's slide; dl-030 lists all four quiz sessions). dl-018's regression is noise: one correct detail swapped for another. Doesn't help `exact`: those chunks aren't just below rank 5, they're far down.

**E2, 200-token chunks.** The biggest single gain for `exact` (correct 0.67 → 1.00): long slides split into focused pieces, so "8 heads" and "1×1 convolution" stop being diluted. 1,639 chunks instead of 611, and cheaper per run (shorter context). dl-028 lost its cited source but was answered correctly from another s19 slide stating the same figure (a missing second source).

**E3, LaTeX cleaning.** Hurt slightly. Lost the Grading slide for dl-007. The symbols the embedding model sees (`\times`) apparently carry signal it uses; plain `×` doesn't help. Dropped.

**E4, hybrid search.** Roughly neutral. Keyword matching fixed dl-005 (S08's "HW1 is due today" shares words with the question) but pushed dl-004's source out. With natural-language questions, OR-ed keywords match too broadly. Might be worth revisiting with weighting, not as configured.

**E5, reranker.** Best single change: 4 fixes, 0 regressions, every retrieval change positive. **Cost is latency**: ~8 s per question on the dev laptop's CPU for 20 × 500-token chunks; on the homelab's i5-6400T expect 2–3× that. Reranking cost scales with chunk length, which is why it pairs with E2: 20 short chunks take ~3 s, 30 take ~4.8 s (laptop).

**E6, bge-base.** Strong on `exact` but worse on multi_hop retrieval, and ~3× slower to embed. It exposed a labeling error: "unanswerable" dl-039 (weight tying) is discussed in the slides (s04 "Where node reuse returns this semester", s17's untied head), so it was replaced (see Label fixes). Worth retrying combined with the winner later.

**C1 vs C2.** Rerank to 5 short chunks was too few (C1 lost context for dl-007 / dl-035). Rerank 30 → 10 short chunks (C2) gives near-perfect retrieval: hit 1.00, recall 0.97, every retrieval change positive. Its two losses were both **answer** failures: with 10 short chunks Haiku answered "not found" to false-premise questions (dl-007 "how much is the final exam worth?", dl-048 "how does clipping let an RNN learn long dependencies?") instead of correcting the premise.

**E7, Sonnet.** Fixed both premise failures but abstained on two questions Haiku got right. Net +1 question for 15× the cost. Not worth it.

**E8, prompt v2.** Targets the observed failure: v2 tells the model that a premise the documents contradict is an answer, and that "not found" is only for questions the documents say nothing about. Fixed dl-007 and dl-048 with Haiku at no extra cost. Every answerable dev question is now correct, and it still abstains on every unanswerable one, including dl-051, whose retrieved chunks (computation-graph slides about "nodes") are a deliberate near-miss for v2's "only when the documents say nothing relevant" rule. Caveat: v2 was written after seeing dev failures, so dev overstates it; the test split is the real check.

## Label fixes found by experiments

- **Scoring change**: a `source_refs` entry may now be a list of alternatives (the same fact on several slides; any one is enough). Before, a second source added as a plain ref counted as *required*, halving recall when retrieval found either one.
- **dl-028, dl-029**: one needed source, stated on two slides → one alternatives group each.
- **dl-030**: Quiz 1's session is on s09 *or* s02 → alternatives group; Quiz 2 (s13) and Quizzes 3–4 (s23) stay required.
- **dl-039 removed**, replaced by **dl-051** (dev, unanswerable): "How does a graph neural network aggregate information from a node's neighbours?" Checked by grep ("graph neural", "GNN", "message passing": none) and by reading what retrieval returns: computation-graph slides, all with rerank scores ≤ 0.013.

## Open items

- **Test run**: done (see Final result). Don't re-tune against it; a new idea gets evaluated on dev first.
- **Multi-hop completeness**: the only test miss (dl-037) listed one of three items. Candidates for a later dev experiment: a prompt line about answering every part of the question, or a less strict length limit than "one to three sentences".
- **Latency**: reranking 30 short chunks is ~5 s per question on the laptop CPU, more on the homelab. Options before Phase 5: fewer candidates, a smaller cross-encoder, or accept it.
