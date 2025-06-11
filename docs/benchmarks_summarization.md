Summarization
============================

#TODO: Why?


Current Benchmarks
-----------------------

The summarization benchmarks are initially available in English and are being translated to Dutch by use of LLMs. A dedicated translation module has been added, with support for basic llm-based and HuggingFace translations. Currently, the implementation focuses on the XSum and CNN/Daily Mail datasets.

- **[CNN/Daily Mail Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail):** This benchmark uses a dataset of articles and their summaries, instructing models to summarize content into 3-4 sentences [1][2]. The dataset is non-anonymized and sourced from the HuggingFace Hub.

- **[XSum Summarization](https://huggingface.co/datasets/EdinburghNLP/xsum):** Benchmarking extreme summarization using the XSum dataset, which aims to create a short, one-sentence summary of online articles from the BBC [3]. We instruct the model to summarize to "1 sentence" according to the task.

#TODO: Add disclaimer about implementation and simple vs detailed prompt

Evaluation Metrics
-----------------------

#TODO: (long story short: we use BERTScore for now)

### Mapping to Categories

Finally, we describe our methodology for mapping the raw scores from the benchmarks to the categories visualized in our [leaderboard](https://amsterdam.github.io/grip-on-llms).
As all currently supported benchmarks consist of comparable tasks (article summarization)
and use the same scores (BERTScore),
we simply average the scores from the different benchmarks.

Afterwards, we use the following performance categories, based on recent reports of
SARI scores in literature:


|           | Average BERTScore | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | 0-0.5   | Very Low   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | 0.5–0.55   | Low        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | 0.55–0.6   | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | 0.6–0.65   | High       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | 0.65-1   | Very High  |

References
----------

- [1] Hermann, Karl Moritz, et al. "Teaching machines to read and comprehend." Advances in neural information processing systems 28 (2015).
- [2] See, Abigail, Peter J. Liu, and Christopher D. Manning. "Get to the point: Summarization with pointer-generator networks." arXiv preprint arXiv:1704.04368 (2017).
- [3] Narayan, Shashi, Shay B. Cohen, and Mirella Lapata. "Don't give me the details, just the summary! Topic-aware convolutional neural networks for extreme summarization." arXiv preprint arXiv:1808.08745 (2018).