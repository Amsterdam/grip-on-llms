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

Whenever possible, we have a preference for using existing benchmarks and
reusing the work and findings of experts in the field.
Ideally, we would like to use benchmarks directly curated in Dutch, such as e.g. BZK's
[Social Bias Benchmark](https://github.com/MinBZK/llm-benchmark/blob/main/benchmarks/social-bias/README.md).
However, many existing benchmarks are only available in English.
In these cases, we automatically translate the known benchmarks.
Unfortunately, translating is not always an ideal option
as important aspects of the questions, such as their structure or content,
might get lost in the translation process.
A notable example are safety and inclusion benchmarks, where hateful speech or stereotypes
might not translate well between languages and cultures.
Thus, if no suitable benchmark exists or meets our needs or quality standards, we (semi-)manually curate a benchmark from scratch.


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
1. Add to the list of benchmarks in the [generate_leaderboard_results.py](/scripts/generate_leaderboard_results.py) script and run the benchmark(s) and models of interest using the script.
1. If applicable run any scripts for post-processing of the results, for example:
    * [scripts/add_honesty_judgements.py](/scripts/add_honesty_judgements.py) to retroactively add the honesty judgements
    * [scripts/add_summarization_scores.py](/scripts/add_summarization_scores.py) to add summarization scores in case anything went wrong with loading the required embedding models at run time
    * [scripts/add_costs.py](/scripts/add_costs.py) - to calculate the costs incured by running open generation tasks.
1. Next, adjust and follow the [process_leaderboard_data.ipynb](/notebooks/process_leaderboard_data.ipynb) notebook to process the generated results and to ensure that no runs failed. Furthermore, map scores to categories. The final purpose of the notebook is to generate the [models.json](/llm-eval-website/_data/models.json) file which is used for visualizing the LLM Overview.
1. Finally, make the corresponding changes in the [LLM Overview](/llm-eval-website/_includes/llm_overview.html). Do not forget all corresponding descriptions in the [English](/llm-eval-website/_data/translations_en.yml) and [Dutch](/llm-eval-website/_data/translations_nl.yml) translations.
