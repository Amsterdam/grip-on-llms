# Model Costs

---

## Why publish cost‑per‑prompt?

Listing a single *cost‑per‑prompt* number lets you compare:

* closed‑source API models that charge per token, and
* self‑hosted models that run on our Azure infrastructure

Moreover, it is more intuitive and easier to understand than price per million tokens.

---
## Data for Comparison:
We utilize the summarization and text simplification datasets to measure the price as it is open-ended questions. This way we also take into account whether models tend to write longer answers than others.
We explicitly exclude multiple-choice tasks.


---

## How we calculate the number you see

| Model type                           | Formula we apply                                               | What it means for you                                                                                                                                   |
|--------------------------------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **API model** Azure OpenAI Endpoint) | input tokens ×  vendor price + output tokens * × *vendor price* | We estimate the number of tokens using [Tiktoken](https://github.com/openai/tiktoken?tab=readme-ov-file#-tiktoken). The input and output tokens are the average amount of input and outputs tokens utilized by an LLM.     |
| **Self-hosted model (In Azure AML)**       | GPU hourly rate  ×  average prompt duration                    | We only measure the cost during inference, the VM warmup time and loading model time are excluded. We calcultate the average duration time of a prompt. We run all experiments on an H100 compute, making the run times comparable, although we realize that different compute resources might be more suitable for bigger or smaller models. |

The GPU hourly rates provided by Azure are in US dollars. We apply the conversion rate mentioned on [their website](https://azure.microsoft.com/nl-nl/pricing/details/cognitive-services/openai-service/) to convert these rates into euros. The Azure OpenAI vendor prices are already listed in euros on the same website. We use these values to publish the final outcomes in the LLM overview.

---

## Caveats & future work

* **Prices change**: API vendors and Azure update rates regularly. 
* **Region**: All costs assume *West Europe* data centre.  Other regions differ.
* **Spot instances**: For open‑source models we show on‑demand rates only; spot pricing can be cheaper but is not guaranteed.
