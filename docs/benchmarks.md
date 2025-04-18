DRAFT: Benchmarking Overview
============================

To Do
-----
*   Explain how everything was translated
*   Review and change some of the links

Introduction
------------

This documentation provides an overview of the benchmarks used to evaluate various language models within our organization. Benchmarks are essential tools for assessing model performance across different tasks, including translation, text simplification, summarization, and reasoning. They help us understand the strengths and weaknesses of models and guide improvements.

Current Benchmarks
------------------

### Knowledge - Reasoning (Dutch Translations)

- **[MMLU](http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/):** Implementation of the MMLU benchmark, which contains 57 tasks aimed at measuring world knowledge and problem-solving ability. We adopt a pragmatic, user-centered approach to compare diverse models, including closed-source ones, in a municipal context. This involves generating answers for direct comparison, performing a single pass, and using a zero-shot setup.

- **[ARC](http://nlp.uoregon.edu/download/okapi-eval/datasets/m_arc/):** Implementation of AI2’s Reasoning Challenge (ARC) benchmark, a common sense reasoning, multiple-choice question-answering dataset. We use a Dutch translation of the original dataset, focusing on generating answers for direct comparison, performing a single pass, and employing a zero-shot setup.

### Text Simplification

- **[Amsterdam Simplification](https://amsterdamintelligence.com/posts/automatic-text-simplification):** This benchmark uses a dataset containing 1,311 automatically aligned complex-simple sentence pairs from documents provided by the Communications Department of the City of Amsterdam [1][2]. It evaluates the model's ability to simplify text while maintaining meaning.

- **[INT Duidelijke Taal](https://ivdnt.org/onderzoek-projecten/afgeronde-projecten/duidelijke-taal/):** This benchmark uses sentences from the SoNaR corpus, automatically simplified using GPT-4o and manually evaluated for simplicity, accuracy, and fluency [3]. We filter higher quality data by selecting samples with "Accuratesse Gem." > 70, preserving all samples regardless of fluency, and ensuring simplifications are simpler than the original.

### Summarization

- **[CNN/Daily Mail Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail):** This benchmark uses a dataset of articles and their summaries, instructing models to summarize content into 3-4 sentences [4][5]. The dataset is non-anonymized and sourced from the HuggingFace Hub.

- **[XSum Summarization](https://huggingface.co/datasets/EdinburghNLP/xsum):** Benchmarking extreme summarization using the XSum dataset, which aims to create a short, one-sentence summary of online articles from the BBC [7]. We instruct the model to summarize to "1 sentence" according to the task.

### Tiny Benchmarks

- **[Tiny MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU):** This benchmark implements a smaller version of the MMLU benchmark, containing 57 tasks to measure world knowledge and problem-solving ability [5]. It uses formatted inputs from TinyBenchmarks available on the HuggingFace Hub [6].

### Why Use Tiny Benchmarks?

Tiny benchmarks are valuable because they allow for efficient evaluation with fewer examples, making them ideal for quick assessments and iterative testing. They are sampled in a way that provides similar scores and evaluations as complete benchmarks, ensuring reliable results. Additionally, they are more sustainable, requiring less computational power and resources.

References
----------

- [1] Vlantis, Daniel, Iva Gornishka, and Shuai Wang. "Benchmarking the simplification of Dutch municipal text." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.
- [2] Oostdijk, Nelleke, et al. "SoNaR user documentation." Online: <https://ticclops.uvt.nl/SoNaR_end-user_documentation_v1.4> (2013).
- [3] Menselijke evaluatie van geautomatiseerde tekstvereenvoudiging: resultaten van crowdsourcing (Version 1.0) (2024) [Data set]. Available at the Dutch Language Institute: <https://hdl.handle.net/10032/tm-a2-y8>.
- [4] Hermann, Karl Moritz, et al. "Teaching machines to read and comprehend." Advances in neural information processing systems 28 (2015).
- [5] See, Abigail, Peter J. Liu, and Christopher D. Manning. "Get to the point: Summarization with pointer-generator networks." arXiv preprint arXiv:1704.04368 (2017).
- [6] Hendrycks, Dan, et al. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).
- [7] Narayan, Shashi, Shay B. Cohen, and Mirella Lapata. "Don't give me the details, just the summary! Topic-aware convolutional neural networks for extreme summarization." arXiv preprint arXiv:1808.08745 (2018).