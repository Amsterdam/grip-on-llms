Factuality
============================

Factuality is one of the aspects of LLMs that our colleagues consider most important,
both in their roles as experts and as LLM users.
They define factuality as an LLM's ability to accurately answer objective knowledge-based questions about certain facts, such as those related to science or geography.
Thus, factuality is not about context-dependent knowledge, such as values, traditions, or societal norms.

Naturally, most people associate factuality with a percentage of correctly answered questions - that is, accuracy.

Current Benchmarks
-----------------------

We've implemented the evaluation of LLMs using automatic translations of
the following commonly used benchmarks:

- **[MMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU):**
Implementation of the tinyBenchmarks version of the MMLU benchmark,
which contains 57 tasks aimed at measuring world knowledge and problem-solving ability [1].

- **[ARC-Challenge](https://huggingface.co/datasets/tinyBenchmarks/tinyAI2_arc):**
Implementation of the tinyBenchmarks version of the  AI2’s Reasoning Challenge (ARC) benchmark,
a common sense reasoning, multiple-choice question-answering dataset. [2]

- **[TruthfulQA](https://huggingface.co/datasets/tinyBenchmarks/tinyTruthfulQA):** Implementation of the tinyBenchmarks version of the TruthfulQA benchmark
which measures whether an LLM is truthful in generating answers to questions [3]

### tinyBenchmarks Implementation

While there are Dutch translations of the original datasets, we opt for using
our own (automatic) translations of the tinyBenchmarks versions of the datasets [4].
This implementation allows us to obtain good performance estimates only from 100 carefully curated samples, making it ideal for quick assessments and iterative testing.
Even more importantly, tinyBenchmarks are more sustainable, requiring less computational resources and energy.

### User-centered Evaluation

Following the assumption that civil servants will directly interact with a model,
we focus on generating answers for direct comparison (rather than looking at underlying probabilities).
We use simple regex-matching to extract the answers for automatic evaluation.

For the listed benchmarks, we employ a zero-shot setup and perform a single pass with greedy decoding.

Evaluation Metrics
-----------------------

Multiple-choice question-answering tasks are commonly evaluated using accuracy.
However, because we only use the 100 samples from the tinyBenchmarks,
we also use a special scoring methodology and tinyBenchmarks' gp-IRT estimator.
gp-IRT estimates how good a model would perform on the full original benchmark
by taking into account which samples the model answered correctly
and how difficult these samples were.
In this way, it is shown to be more reliable than directly calculating accuracy.

### Mapping to Categories

Finally, we describe our methodology for mapping the raw scores from the benchmarks to the categories visualized in our [leaderboard](https://amsterdam.github.io/grip-on-llms).
As all currently supported benchmarks consist of comparable tasks (multiple-choice questions)
and use the same scores (gp-IRT estimation of accuracy),
we simply average the scores from the different benchmarks.

Afterwards, we use the following performance categories:

<!-- <br><span class="circle" style="color: #EC0000;">●</span> 0.0-0.5 - Very Low
<br><span class="circle" style="color: #FF9100;">●</span> 0.5-0.6 - Low
<br><span class="circle" style="color: #FFE600;">●</span> 0.6-0.7 - Medium
<br><span class="circle" style="color: #BED200;">●</span> 0.7-0.8 - High
<br><span class="circle" style="color: #00A03C;">●</span> 0.8-1.0 - Very High -->


<!-- <br>![Red Circle](https://placehold.co/20x20/EC0000/EC0000.png) 0.0-0.5 - Very Low
<br>![Orange Circle](https://placehold.co/20x20/FF9100/FF9100.png) 0.0-0.5 - Very Low
<br>![Yellow Circle](https://placehold.co/20x20/FFE600/FFE600.png) 0.0-0.5 - Very Low
<br>![Lime Circle](https://placehold.co/20x20/BED200/BED200.png) 0.0-0.5 - Very Low
<br>![Green Circle](https://placehold.co/20x20/00A03C/00A03C.png) 0.0-0.5 - Very Low -->


|           | Average gp-IRT    | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | 0.0–0.5   | Very Low   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | 0.5–0.6   | Low        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | 0.6–0.7   | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | 0.7–0.8   | High       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | 0.8–1.0   | Very High  |

References
----------

- [1] Hendrycks, Dan, et al. ["Measuring massive multitask language understanding."](https://arxiv.org/pdf/2009.03300) arXiv preprint arXiv:2009.03300 (2020).
- [2] Clark, Peter, et al. ["Think you have solved question answering? try arc, the ai2 reasoning challenge."](https://arxiv.org/pdf/1803.05457) arXiv preprint arXiv:1803.05457 (2018).
- [3] Lin, Stephanie, Jacob Hilton, and Owain Evans. ["Truthfulqa: Measuring how models mimic human falsehoods."](https://arxiv.org/pdf/2109.07958) arXiv preprint arXiv:2109.07958 (2021).
- [4] Polo, Felipe Maia, et al. ["tinyBenchmarks: evaluating LLMs with fewer examples."](https://arxiv.org/pdf/2402.14992) arXiv preprint arXiv:2402.14992 (2024).