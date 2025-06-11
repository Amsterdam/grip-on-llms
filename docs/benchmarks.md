Benchmarking Overview
============================

Introduction
------------

This documentation provides an overview of the benchmarks used to evaluate various [aspects of language models](./aspects.md) within our organization.
Benchmarks are essential tools for assessing model performance across different tasks, including translation, text simplification, summarization, and reasoning.
They help us understand the strengths and weaknesses of models and guide improvements.


Choice of Benchmarks and Implementation
---------------------------------------

### Reuse, Reduce, Recycle

Whenever possible, we have a preference for using existing benchmarks and reusing the work and findings of experts in the field.
Ideally, we would like to use benchmarks directly curated in Dutch, such as e.g. BZK's [Social Bias Benchmark](https://github.com/MinBZK/llm-benchmark/blob/main/benchmarks/social-bias/README.md).
However, many existing benchmarks are only available in English.
In these cases, we automatically translate the known benchmarks.
Finally, if no suitable benchmark exists or meets our needs or quality standards, we (semi-)manually curate a benchmark from scratch.


Furthermore, we aim for an efficient and environmentally friendly implementation of benchmark.
We have a preference for using smaller benchmarks, supporting initiatives such as [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) or simply running the evaluation scripts on a random sample of evaluation prompts in order to reduce the environmental impact of our benchmarking process.

### LLM-as-a-judge -> Human-in-the-loop

While we aim to use benchmarks which allow for objective quantifiable evaluation, the rapid advancements in the LLM world do not always allow for such objective evaluation. Unfortunately, manual evaluation is also not always possible or scalable.
As a last option, we sometimes accept using LLM as a judge, acknowledging the multiple challenges and biases related to it.

We make the following agreements about using LLM as a judge within our benchmarks:
- we **do not use it for sensitive tasks**, where alignment with human values is important
- always design the task well, make it **as concrete as possible**, using clear criteria and definitions. Never let an LLM decide on a definition (e.g. what is inclusive or simple language).
- essentially ensure that humans can also perform the task with a **high inter-annotator agreement**
- always try to **"evaluate the evaluator"** (annotate a number of examples ourselves and check how the LLM-to-be-used-as-a-judge performs the task). This would also help with better understanding of the evaluation task and refinement of the evaluation prompt and criteria
- check [recent literature](https://arxiv.org/pdf/2411.15594?) for practical tips on improving performance and avoiding biases, such as e.g. aggregating results from multiple rounds or multiple models
- consider how to perform the evaluations for Dutch, depending on the task, goal and models, it could be better to instruct the model in English or directly in Dutch


How to add benchmarks
---------------------
1. Check the quality of the benchmark, how it was collected and whether it poses any technical or ethical concerns.
    * Be critical about whether the benchmark contributes to assessing the aspect of interest.
    * Think whether the benchmark isn't outdated or saturated.
1. Check if a Dutch version of the benchmark exists.
    * If yes, check the quality of translations in case it was automatically translated.
    * If not, ensure we support translation functionality (e.g. as we did for [XSum](/llm_eval/benchmarks/summarization/xsum.py) where we pass a [translator object](/llm_eval/translators/translator_router.py)). Consider publishing the translated version for others to reuse.
1. Add a new module, extending the [BaseBenchmark](/llm_eval/benchmarks/base.py) class. In this way we ensure consistently running, scoring and documenting benchmarks for the leaderboard.
1. Add to the list of benchmarks in the corresponding aspect page (see all [aspects](./aspects.md))
1. After running, ensure the results are reflected in the scores on the **[leaderboard](/llm-eval-website/_data/models.json)** (this process will be automated soon). Also, adjust the aspect description on the leaderboard to mention the benchmark or include a relevant example.
