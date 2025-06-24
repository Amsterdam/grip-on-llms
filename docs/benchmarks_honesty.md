Honesty
============================

One of the most important aspects of Large Language Models is
their ability to produce only factual information.
On one hand, this means that a model should be able to accurately answer objective knowledge-based questions (that is ["factuality"](./benchmarks_factuality.md)).
On the other hand, models must also be able to admit when they do not know the answer to such questions and avoid confident fabrication or hallucination (that is "honesty").

Recent work related to honesty often looks at different types of situations and scenarios
in which a model must admit their own limitations.
For example, language models do not have access to latest information and they should acknowledge that when prompted to give recent information about e.g. whether or election results.
They should also be able to handle prompts with misleading or insufficient information.

Below, we describe our own honesty benchmark which adapts, translates and extends
existing benchmark in order to evaluate honesty in the Dutch municipal context.

HonestCityBench
-----------------------

The HonestCityBench contains [x] pro devided in multiple categories...
#TODO:

This benchmark is inspired by:
- **[HonestLLM](https://github.com/Flossiee/HonestyLLM/tree/main):** [1]
- **[BeHonest](https://github.com/GAIR-NLP/BeHonest):** [2]
- **[UnknownBench](https://github.com/genglinliu/UnknownBench/):** [3]


Evaluation (LLM-as-a-judge)
-----------------------
#TODO:

### Mapping to Categories

Finally, we describe our methodology for mapping the raw scores from the benchmarks to the categories visualized in our [leaderboard](https://amsterdam.github.io/grip-on-llms).
#TODO

Afterwards, we use the following performance categories:

|           | Score             | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | TBA   | Very Low   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | TBA   | Low        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | TBA   | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | TBA   | High       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | TBA   | Very High  |

References
----------

- [1] Chujie, Gao, et al. 
["Honestllm: Toward an honest and helpful large language model."](https://proceedings.neurips.cc/paper_files/paper/2024/file/0d99a8c048befb6dd6e17d7684adacac-Paper-Conference.pdf)
Advances in Neural Information Processing Systems 37 (2024): 7213-7255.
- [2] Chern, Steffi, et al.
["BeHonest: Benchmarking Honesty in Large Language Models."](https://arxiv.org/pdf/2406.13261)
arXiv preprint arXiv:2406.13261 (2024).
- [3] Liu, Genglin, et al.
["Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric Knowledge."](https://arxiv.org/pdf/2311.09731)
arXiv preprint arXiv:2311.09731 (2023).