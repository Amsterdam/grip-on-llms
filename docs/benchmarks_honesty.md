Honesty
============================

One of the most important aspects of LLMs is their ability to produce only factual information.
On one hand, this means that a model should be able to accurately answer
objective knowledge-based questions (that is [**"factuality"**](./benchmarks_factuality.md)).
On the other hand, models must also be able to admit when they do not know the answer to such
questions and avoid confident fabrication or hallucination (that is **"honesty"**).

Recent work related to honesty often looks at different types of situations and scenarios
in which a model must admit their own limitations.
For example, language models do not have access to latest information and they should acknowledge
hat when prompted to give recent information about e.g. whether or election results.
They should also be able to handle prompts with misleading or insufficient information.

Below, we describe our own honesty benchmark which adapts, translates and extends
existing benchmark in order to evaluate honesty in the Dutch municipal context.

HonestCityBench
-----------------------

The HonestCityBench contains 530 prompts devided in 5 categories:
- **no latest information**
    * Related to information which changes over time.
    * Response requires latest information from a trusted source or external service.
- **user info wrong**
    * User prompt contains wrong information.
    * Response requires first correcting the wrong premise. Should be related to verifiable facts.
- **user info incomplete**
    * Contain insufficient information making them ambiguous and impossible to answer correctly.
    * Response must acknowledge the missing information.
- **no expert**
    * Highly specific questions requiring a narrow expertise (e.g. public health, legal, finance).
    * Responding without disclaiming lack of expertise could have severe implications.
- **no multimodal**
    * Related to other modalities (e.g. images, music, tables).
    * The model cannot possibly fulfill the request because it is an LLM.
    * Proceeding with a response wuthout communicating the limitations could lead to distrust,
    frustration and repeated requests leading to further costs and environmental impact.


Evaluation (LLM-as-a-judge)
-----------------------
Evaluating honesty is a challenging task as there are multiple different ways in which a model
can ackwoledge its limitations and phrase its inability to respond to a prompt.
At the same time, manual evaluation is time-consuming and unscalable.
Therefore, we chose to perform LLM-as-a-judge evaluation - that is, let (another)
Large Language Model to automatically judge whether a model properly handled a request and
honestly acknowledged its limitations.

To ensure that the quality of the automatic judgements, we use the following protocol:
- **test prompt selection:** we select 50 test prompts which will be used for initial evaluation
- **response generation:** we generate responses to these prompts using ~10 diverse LLMs
    (small versions of Mistral, Phi, Llama, Qwen, etc).
- **Human annotations:** 2 human annotators evaluate each of the generated responses.
    This phase was also used to clarify the final definitions of categories and to
    improve the evaluation guidelines for the automated judges.
- **LLM judge annotations:** we let a wide range of models perform the evaluation using the
    refined category definitions and guidelines.

The final selection of judges (currently gpt-4o-mini, gemma-12b-instruct & qwen-8b)
have been selected based on our pragmatic
[evaluation of a number of potential judges](../notebooks/honesty/HonestyJudgeEval.ipynb).
We selected these judges based on 3 criteria:
- high interannotator aggrement with our human annotators
- high correlation of the final rankings
- low environmental impact (i.e., preference for smaller models)

In case multiple judges have been passed, the final judgement is simply the majority vote.

Finally, we calcualte percentage of properly refused answers per model.


### Mapping to Categories

Finally, we map the raw scores from the benchmarks to the categories visualized in our [leaderboard](https://amsterdam.github.io/grip-on-llms).

We use the following performance categories:

|           | Score             | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | 0.0–0.2 | Very Low   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | 0.2-0.4  | Low        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | 0.4-0.6 | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | 0.6-0.8 | High       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | 0.8-1.0 | Very High  |


Acknowledgements
-----------------------

This benchmark was inspired by:
- **[HonestLLM](https://github.com/Flossiee/HonestyLLM/tree/main):** [1]
- **[BeHonest](https://github.com/GAIR-NLP/BeHonest):** [2]
- **[UnknownBench](https://github.com/genglinliu/UnknownBench/):** [3]

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
