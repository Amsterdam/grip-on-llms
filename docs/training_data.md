# Training Data

---

## Why do we evaluate training data?

A model’s behaviour is bounded by the corpus it was trained on.  An ideal dataset consists of the following aspects:

* personal data are handled in line with the GDPR and the AI Act;
* copyrighted works are respected (or licensed);
* Finetuning data is annotated in an ethical manner.

Without visibility into the underlying data it is impossible to interpret a model’s strengths, weaknesses, or compliance posture.

---


## Transparency of training data

We classify every model against three transparency levels:

| Level         | Definition                                                                                                                                              | Leaderboard flag |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **Open**      | The full corpus is published under an open licence.                                                                        | *open* ✓         |
| **Described** | The corpus is not released in full, but is documented in a paper, card, or Hugging Face model card with enough detail to understand scope and provenance. | *described* △    |
| **Closed**    | No meaningful disclosure is provided.                                                                                                                   | *opaque* ✕       |

These labels don't affect any technical score.

---

## Key risk areas

| Risk area                | What we look for                                                                 | Good Practise                                                        |
| ------------------------ | -------------------------------------------------------------------------------- |----------------------------------------------------------------------|
| **Personal data**        | Presence of identifiable natural persons (§4 GDPR).                              | Removal, strong aggregation, or documented legal basis.              |
| **Copyrighted material** | Text still under copyright that were ingested without an explicit licence. | Usage under a valid licence or documented exception. |

We prefer providers that describe clear mitigations strategies are described for a LLM in for example the research paper.

---

## Ethical red flags (disqualifying)

When trustworthy sources shows that a training corpus contains serious ethical violations, for example large‑scale scraping of paid medical forums, or leaked personal messages, we mark the model as **non‑admissible** on the leaderboard and remove it from ranking.

---

## Dutch language coverage

Because we serve Dutch public‑sector use‑cases, we value corpora that include a substantial share of Dutch‑language material (original or translated).  Models that document Dutch coverage receive a **“NL‑supported”** icon.

---

## Practical note on data‑subject requests

Greater transparency comes with a practical obligation: if a citizen asks whether their personal data appear in a training corpus, we are *in principle* required to investigate.  In practice, present‑day foundation models are trained on hundreds of billions of tokens; reconstructing the presence (or absence) of a specific individual is not technically feasible with reasonable effort.  For that reason, our policy is to **acknowledge the request and explain the practical impossibility** of an exact answer, while pointing the requester to the model developer’s privacy contact.

---

