Simplification
============================

Low literacy skills hinder societal participation in tasks such as voting, paying taxes, reissuing documents, or applying for social benefits.
Simpler content also benefits advanced readers, reducing cognitive load and increasing efficiency.
These days, large language models are seen as a promising alternative to traditional simplification methods and are being increasingly used by our colleagues to simplify texts and improve content.


Current Benchmarks
-----------------------

- **[Amsterdam Simplification](https://amsterdamintelligence.com/posts/automatic-text-simplification):** This benchmark uses a dataset containing 1,311 automatically aligned complex-simple sentence pairs from documents provided by the Communications Department of the City of Amsterdam [1]. It evaluates the model's ability to simplify text while maintaining meaning.

- **[INT Duidelijke Taal](https://ivdnt.org/onderzoek-projecten/afgeronde-projecten/duidelijke-taal/):** This benchmark uses sentences from the SoNaR corpus, automatically simplified using GPT-4o and manually evaluated for simplicity, accuracy, and fluency [2]. We filter higher quality data by selecting samples with "Accuratesse Gem." > 70, preserving all samples regardless of fluency, and ensuring simplifications are simpler than the original.

#TODO: Add disclaimer about implementation and simple vs detailed prompt

Evaluation Metrics
-----------------------

#TODO: (long story short: nothing works, but we use SARI for now)

### Mapping to Categories

Finally, we describe our methodology for mapping the raw scores from the benchmarks to the categories visualized in our [leaderboard](https://amsterdam.github.io/grip-on-llms).
As all currently supported benchmarks consist of comparable tasks (sentence simplification)
and use the same scores (SARI),
we simply average the scores from the different benchmarks.

Afterwards, we use the following performance categories, based on recent reports of
SARI scores in literature:


|           | Average SARI      | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | 0-26   | Very Low   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | 26–32   | Low        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | 32–38   | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | 38–44   | High       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | 44–   | Very High  |

References
----------

- [1] Vlantis, Daniel, Iva Gornishka, and Shuai Wang. "Benchmarking the simplification of Dutch municipal text." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.
- [2] Menselijke evaluatie van geautomatiseerde tekstvereenvoudiging: resultaten van crowdsourcing (Version 1.0) (2024) [Data set]. Available at the Dutch Language Institute: <https://hdl.handle.net/10032/tm-a2-y8>.